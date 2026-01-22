import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

from marinegym.utils.torch import quaternion_to_euler, quat_rotate_inverse

from .modules.distributions import IndependentNormal
from .ppo.common import GAE
from .utils.valuenorm import ValueNorm1

try:
    import pypose as pp  # noqa: F401
    _HAS_PYPOSE = True
except Exception:
    _HAS_PYPOSE = False


# train.py task=Hover algo=ac_mpc_pypose wandb.mode=offline headless=true

# 느리면 먼저: train.py task=Hover algo=ac_mpc_pypose env.num_envs=16 algo.mpc_N=5 algo.mpc_ilqr_iters=1 wandb.mode=offline headless=true



def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


def _extract_action_bounds(action_spec, device, action_dim: int):
    try:
        leaf = action_spec[("agents", "action")]
    except Exception:
        leaf = action_spec

    low = None
    high = None

    for cand in ("low", "minimum"):
        if hasattr(leaf, cand):
            low = getattr(leaf, cand)
            break
    for cand in ("high", "maximum"):
        if hasattr(leaf, cand):
            high = getattr(leaf, cand)
            break

    if (low is None or high is None) and hasattr(leaf, "space"):
        if hasattr(leaf.space, "low") and hasattr(leaf.space, "high"):
            low, high = leaf.space.low, leaf.space.high

    if low is None or high is None:
        low = -torch.ones(action_dim, device=device)
        high = torch.ones(action_dim, device=device)

    low = torch.as_tensor(low, device=device).reshape(-1)[:action_dim]
    high = torch.as_tensor(high, device=device).reshape(-1)[:action_dim]
    return low, high


@dataclass
class ACMPCPyPoseConfig:
    name: str = "ac_mpc_pypose"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    priv_actor: bool = False
    priv_critic: bool = False
    checkpoint_path: Optional[str] = None

    use_mpc_head: bool = True
    entropy_coef: float = 0.01
    actor_log_std_init: float = 0.0
    actor_log_std_min: float = -2.0
    actor_log_std_max: float = 1.0

    # Used by `scripts/train.py` to keep parity with `ac_mpc` config updates.
    obs_has_target_quat: bool = False
    obs_time_encoding_dim: int = 0

    mpc_N: int = 10
    mpc_dt: float = 0.02
    mpc_nx: int = 12
    mpc_nu: int = 6

    mpc_param_yaml: Optional[str] = None
    mpc_alloc_npz: Optional[str] = None

    mpc_uabs_eps: float = 1e-3
    mpc_ilqr_iters: int = 3
    mpc_lqr_reg: float = 1e-4

    cost_q_lb: float = 0.1
    cost_q_ub: float = 100000.0
    cost_p_lb: float = 0.1
    cost_p_ub: float = 100000.0
    cost_hidden: int = 512

    water_density: float = 997.0
    gravity: float = 9.81

    mpc_state_mode: str = "hover_obs"

    mpc_mass: Optional[float] = None
    mpc_inertia: Optional[List[float]] = None

    # (No-op for PyPose MPC) kept for compatibility with `scripts/train.py`.
    acados_codegen_dir: str = "./acados_codegen"
    acados_json: Optional[str] = None


cs = ConfigStore.instance()
cs.store("ac_mpc_pypose", node=ACMPCPyPoseConfig, group="algo")
cs.store("ac_mpc_pypose_priv", node=ACMPCPyPoseConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ac_mpc_pypose_priv_critic", node=ACMPCPyPoseConfig(priv_critic=True), group="algo")


class NeuralCostMapHorizon(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_cost: int,
        horizon: int,
        q_lb: float,
        q_ub: float,
        p_lb: float,
        p_ub: float,
        hidden: int = 512,
    ):
        super().__init__()
        self.n_cost = int(n_cost)
        self.horizon = int(horizon)
        self.q_lb = float(q_lb)
        self.q_ub = float(q_ub)
        self.p_lb = float(p_lb)
        self.p_ub = float(p_ub)

        out_dim = 2 * self.horizon * self.n_cost
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(feat)
        raw = raw.view(raw.shape[0], self.horizon, 2 * self.n_cost)
        q_raw = raw[..., : self.n_cost]
        p_raw = raw[..., self.n_cost :]

        q = self.q_lb + (self.q_ub - self.q_lb) * torch.sigmoid(q_raw)
        p = self.p_lb + (self.p_ub - self.p_lb) * torch.sigmoid(p_raw)
        return q, p


def _safe_cos(theta: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    c = torch.cos(theta)
    return torch.where(c.abs() < eps, c.sign() * eps, c)


def _rot_zyx(rpy: torch.Tensor) -> torch.Tensor:
    if _HAS_PYPOSE:
        return pp.euler2SO3(rpy).matrix()
    phi, theta, psi = rpy.unbind(-1)
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)

    R = torch.empty(rpy.shape[:-1] + (3, 3), dtype=rpy.dtype, device=rpy.device)
    R[..., 0, 0] = cpsi * cth
    R[..., 0, 1] = cpsi * sth * sphi - spsi * cphi
    R[..., 0, 2] = cpsi * sth * cphi + spsi * sphi
    R[..., 1, 0] = spsi * cth
    R[..., 1, 1] = spsi * sth * sphi + cpsi * cphi
    R[..., 1, 2] = spsi * sth * cphi - cpsi * sphi
    R[..., 2, 0] = -sth
    R[..., 2, 1] = cth * sphi
    R[..., 2, 2] = cth * cphi
    return R


def _dR_zyx(rpy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phi, theta, psi = rpy.unbind(-1)
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)

    dphi = torch.empty(rpy.shape[:-1] + (3, 3), dtype=rpy.dtype, device=rpy.device)
    dphi[..., 0, 0] = 0.0
    dphi[..., 0, 1] = cpsi * sth * cphi + spsi * sphi
    dphi[..., 0, 2] = -cpsi * sth * sphi + spsi * cphi
    dphi[..., 1, 0] = 0.0
    dphi[..., 1, 1] = spsi * sth * cphi - cpsi * sphi
    dphi[..., 1, 2] = -spsi * sth * sphi - cpsi * cphi
    dphi[..., 2, 0] = 0.0
    dphi[..., 2, 1] = cth * cphi
    dphi[..., 2, 2] = -cth * sphi

    dth = torch.empty_like(dphi)
    dth[..., 0, 0] = -cpsi * sth
    dth[..., 0, 1] = cpsi * cth * sphi
    dth[..., 0, 2] = cpsi * cth * cphi
    dth[..., 1, 0] = -spsi * sth
    dth[..., 1, 1] = spsi * cth * sphi
    dth[..., 1, 2] = spsi * cth * cphi
    dth[..., 2, 0] = -cth
    dth[..., 2, 1] = -sth * sphi
    dth[..., 2, 2] = -sth * cphi

    dpsi = torch.empty_like(dphi)
    dpsi[..., 0, 0] = -spsi * cth
    dpsi[..., 0, 1] = -spsi * sth * sphi - cpsi * cphi
    dpsi[..., 0, 2] = -spsi * sth * cphi + cpsi * sphi
    dpsi[..., 1, 0] = cpsi * cth
    dpsi[..., 1, 1] = cpsi * sth * sphi - spsi * cphi
    dpsi[..., 1, 2] = cpsi * sth * cphi + spsi * sphi
    dpsi[..., 2, 0] = 0.0
    dpsi[..., 2, 1] = 0.0
    dpsi[..., 2, 2] = 0.0

    return dphi, dth, dpsi


def _euler_T(rpy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phi, theta, _psi = rpy.unbind(-1)
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    sth = torch.sin(theta)
    cth = _safe_cos(theta)

    tan = sth / cth
    inv_cth = 1.0 / cth

    T = torch.empty(rpy.shape[:-1] + (3, 3), dtype=rpy.dtype, device=rpy.device)
    T[..., 0, 0] = 1.0
    T[..., 0, 1] = sphi * tan
    T[..., 0, 2] = cphi * tan
    T[..., 1, 0] = 0.0
    T[..., 1, 1] = cphi
    T[..., 1, 2] = -sphi
    T[..., 2, 0] = 0.0
    T[..., 2, 1] = sphi * inv_cth
    T[..., 2, 2] = cphi * inv_cth

    dphi = torch.empty_like(T)
    dphi[..., 0, 0] = 0.0
    dphi[..., 0, 1] = cphi * tan
    dphi[..., 0, 2] = -sphi * tan
    dphi[..., 1, 0] = 0.0
    dphi[..., 1, 1] = -sphi
    dphi[..., 1, 2] = -cphi
    dphi[..., 2, 0] = 0.0
    dphi[..., 2, 1] = cphi * inv_cth
    dphi[..., 2, 2] = -sphi * inv_cth

    sec2 = 1.0 / (cth * cth)
    dinv_cth = sth * sec2

    dth = torch.zeros_like(T)
    dth[..., 0, 1] = sphi * sec2
    dth[..., 0, 2] = cphi * sec2
    dth[..., 2, 1] = sphi * dinv_cth
    dth[..., 2, 2] = cphi * dinv_cth

    return T, dphi, dth


class ILQRMPCLayer(nn.Module):
    def __init__(self, cfg: ACMPCPyPoseConfig, u_min: np.ndarray, u_max: np.ndarray):
        super().__init__()
        if not _HAS_PYPOSE:
            raise RuntimeError("PyPose is required for ac_mpc_pypose (pip install pypose).")

        self.cfg = cfg
        self.nx = int(cfg.mpc_nx)
        self.nu = int(cfg.mpc_nu)
        self.N = int(cfg.mpc_N)
        self.dt = float(cfg.mpc_dt)

        self.u_min = torch.as_tensor(u_min, dtype=torch.float32).reshape(-1)
        self.u_max = torch.as_tensor(u_max, dtype=torch.float32).reshape(-1)
        assert self.u_min.numel() == self.nu and self.u_max.numel() == self.nu

        if cfg.mpc_alloc_npz is None:
            raise RuntimeError("cfg.mpc_alloc_npz is required (generated in train.py).")

        with np.load(cfg.mpc_alloc_npz, allow_pickle=True) as alloc:
            B = alloc["B"].astype(np.float64)
        if B.shape != (6, self.nu):
            raise ValueError(f"Allocation matrix B must be shape (6,{self.nu}), got {B.shape}")
        self.register_buffer("B_alloc", torch.from_numpy(B).to(dtype=torch.float32), persistent=False)

        prm = self._load_params(cfg)
        m = float(prm["mass"])
        inertia = prm["inertia"].astype(np.float64).reshape(3)
        added_mass = prm["added_mass"].astype(np.float64).reshape(6)
        Dlin = prm["linear_damping"].astype(np.float64).reshape(6)
        Dquad = prm["quadratic_damping"].astype(np.float64).reshape(6)
        vol = float(prm["volume"])
        cobm = float(prm["coBM"])

        rho = float(cfg.water_density)
        g = float(cfg.gravity)

        M_diag = np.array([m, m, m, inertia[0], inertia[1], inertia[2]], dtype=np.float64) + added_mass
        if np.any(M_diag <= 0):
            raise ValueError(f"Invalid mass/inertia/added_mass leading to non-positive M diag: {M_diag}")

        self.register_buffer("M_inv", torch.from_numpy((1.0 / M_diag).astype(np.float32)), persistent=False)
        self.register_buffer("Dlin", torch.from_numpy(Dlin.astype(np.float32)), persistent=False)
        self.register_buffer("Dquad", torch.from_numpy(Dquad.astype(np.float32)), persistent=False)

        W = m * g
        Bbuoy = rho * g * vol
        self._k_wb = float(W - Bbuoy)
        self._d_cobm = float(cobm)
        self._Bbuoy = float(Bbuoy)

    def _load_params(self, cfg: ACMPCPyPoseConfig) -> Dict[str, Any]:
        params = {
            "mass": 10.0,
            "inertia": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "added_mass": np.zeros(6, dtype=np.float64),
            "linear_damping": np.zeros(6, dtype=np.float64),
            "quadratic_damping": np.zeros(6, dtype=np.float64),
            "volume": 0.01,
            "coBM": 0.0,
        }

        try:
            if getattr(cfg, "mpc_mass", None) is not None:
                params["mass"] = float(cfg.mpc_mass)
        except Exception:
            pass
        try:
            if getattr(cfg, "mpc_inertia", None) is not None:
                params["inertia"] = np.array(cfg.mpc_inertia, dtype=np.float64).reshape(3)
        except Exception:
            pass

        if cfg.mpc_param_yaml is None:
            return params

        path = os.path.abspath(cfg.mpc_param_yaml)
        if not os.path.exists(path):
            raise FileNotFoundError(f"mpc_param_yaml not found: {path}")

        y = yaml.safe_load(open(path, "r"))
        params["volume"] = float(y.get("volume", params["volume"]))
        params["coBM"] = float(y.get("coBM", params["coBM"]))
        hydro = y.get("hydro_coef", {}) or {}
        params["added_mass"] = np.array(hydro.get("added_mass", params["added_mass"]), dtype=np.float64).reshape(6)
        params["linear_damping"] = np.array(hydro.get("linear_damping", params["linear_damping"]), dtype=np.float64).reshape(6)
        params["quadratic_damping"] = np.array(hydro.get("quadratic_damping", params["quadratic_damping"]), dtype=np.float64).reshape(6)

        if "mass" in y:
            params["mass"] = float(y["mass"])
        if "inertia" in y:
            params["inertia"] = np.array(y["inertia"], dtype=np.float64).reshape(3)

        return params

    def _squash(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_min = self.u_min.to(device=v.device, dtype=v.dtype)
        u_max = self.u_max.to(device=v.device, dtype=v.dtype)
        u_mid = 0.5 * (u_min + u_max)
        u_scale = 0.5 * (u_max - u_min)
        t = torch.tanh(v)
        u = u_mid + u_scale * t
        du_dv = u_scale * (1.0 - t * t)
        d2u_dv2 = -2.0 * u_scale * t * (1.0 - t * t)
        return u, du_dv, d2u_dv2

    def _g_rest_and_jac(self, rpy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = rpy[..., 0]
        theta = rpy[..., 1]
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        cth, sth = torch.cos(theta), torch.sin(theta)

        k = float(self._k_wb)
        d = float(self._d_cobm)
        Bbuoy = float(self._Bbuoy)

        g = torch.zeros(rpy.shape[:-1] + (6,), dtype=rpy.dtype, device=rpy.device)
        g[..., 0] = k * sth
        g[..., 1] = -k * sphi * cth
        g[..., 2] = -k * cphi * cth
        g[..., 3] = -d * Bbuoy * cth * sphi
        g[..., 4] = -d * Bbuoy * sth
        g[..., 5] = 0.0

        dg = torch.zeros(rpy.shape[:-1] + (6, 3), dtype=rpy.dtype, device=rpy.device)
        dg[..., 0, 1] = k * cth
        dg[..., 1, 0] = -k * cphi * cth
        dg[..., 1, 1] = k * sphi * sth
        dg[..., 2, 0] = k * sphi * cth
        dg[..., 2, 1] = k * cphi * sth
        dg[..., 3, 0] = -d * Bbuoy * cth * cphi
        dg[..., 3, 1] = d * Bbuoy * sth * sphi
        dg[..., 4, 1] = -d * Bbuoy * cth
        return g, dg

    def _dyn_step(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rpos = x[..., 0:3]
        rpy = x[..., 3:6]
        nu_b = x[..., 6:12]
        v_b = nu_b[..., 0:3]
        w_b = nu_b[..., 3:6]

        u, du_dv, _d2u = self._squash(v)

        eps = float(self.cfg.mpc_uabs_eps)
        u_abs = torch.sqrt(u * u + eps * eps)
        u_sq = u * u_abs

        tau = torch.einsum("ij,...j->...i", self.B_alloc.to(device=x.device, dtype=x.dtype), u_sq)

        eps_nu = float(self.cfg.mpc_uabs_eps)
        nu_abs = torch.sqrt(nu_b * nu_b + eps_nu * eps_nu)
        damp = self.Dlin.to(device=x.device, dtype=x.dtype) + self.Dquad.to(device=x.device, dtype=x.dtype) * nu_abs

        g_rest, _dg = self._g_rest_and_jac(rpy)

        invM = self.M_inv.to(device=x.device, dtype=x.dtype)
        nu_dot = invM * (tau - damp * nu_b - g_rest)

        R = _rot_zyx(rpy)
        rpos_dot = -torch.einsum("...ij,...j->...i", R, v_b)

        T, _dT_dphi, _dT_dth = _euler_T(rpy)
        rpy_dot = torch.einsum("...ij,...j->...i", T, w_b)

        x_next = x + self.dt * torch.cat([rpos_dot, rpy_dot, nu_dot], dim=-1)
        return x_next, u, du_dv

    def _dyn_jac(
        self, x: torch.Tensor, v: torch.Tensor, u: torch.Tensor, du_dv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rpy = x[..., 3:6]
        nu_b = x[..., 6:12]
        v_b = nu_b[..., 0:3]
        w_b = nu_b[..., 3:6]

        device = x.device
        dtype = x.dtype

        B = x.shape[0]
        A = torch.zeros((B, self.nx, self.nx), device=device, dtype=dtype)
        A[:, torch.arange(self.nx), torch.arange(self.nx)] = 1.0

        R = _rot_zyx(rpy)
        dR_dphi, dR_dth, dR_dpsi = _dR_zyx(rpy)
        drpos_dphi = -torch.einsum("...ij,...j->...i", dR_dphi, v_b)
        drpos_dth = -torch.einsum("...ij,...j->...i", dR_dth, v_b)
        drpos_dpsi = -torch.einsum("...ij,...j->...i", dR_dpsi, v_b)
        A[:, 0:3, 3] += self.dt * drpos_dphi
        A[:, 0:3, 4] += self.dt * drpos_dth
        A[:, 0:3, 5] += self.dt * drpos_dpsi
        A[:, 0:3, 6:9] += self.dt * (-R)

        T, dT_dphi, dT_dth = _euler_T(rpy)
        drpy_dphi = torch.einsum("...ij,...j->...i", dT_dphi, w_b)
        drpy_dth = torch.einsum("...ij,...j->...i", dT_dth, w_b)
        A[:, 3:6, 3] += self.dt * drpy_dphi
        A[:, 3:6, 4] += self.dt * drpy_dth
        A[:, 3:6, 9:12] += self.dt * T

        g_rest, dg = self._g_rest_and_jac(rpy)
        _ = g_rest
        invM = self.M_inv.to(device=device, dtype=dtype)
        A[:, 6:12, 3:6] += self.dt * (-invM[:, None] * dg)

        eps_nu = float(self.cfg.mpc_uabs_eps)
        nu_abs = torch.sqrt(nu_b * nu_b + eps_nu * eps_nu)
        damp_lin = self.Dlin.to(device=device, dtype=dtype)
        damp_quad = self.Dquad.to(device=device, dtype=dtype)
        dd = damp_lin + damp_quad * (2.0 * nu_b * nu_b + eps_nu * eps_nu) / nu_abs
        A[:, 6:12, 6:12] += self.dt * torch.diag_embed(-invM * dd)

        eps_u = float(self.cfg.mpc_uabs_eps)
        u_abs = torch.sqrt(u * u + eps_u * eps_u)
        du_sq_du = (2.0 * u * u + eps_u * eps_u) / u_abs
        dtau_du = self.B_alloc.to(device=device, dtype=dtype).unsqueeze(0) * du_sq_du.unsqueeze(1)
        dtau_dv = dtau_du * du_dv.unsqueeze(1)
        Bmat = torch.zeros((B, self.nx, self.nu), device=device, dtype=dtype)
        Bmat[:, 6:12, :] = self.dt * (invM.unsqueeze(-1) * dtau_dv)

        return A, Bmat

    def _rollout(self, x0: torch.Tensor, v_seq: torch.Tensor):
        xs = [x0]
        us = []
        du_dvs = []
        x = x0
        for t in range(self.N):
            x, u, du_dv = self._dyn_step(x, v_seq[:, t])
            xs.append(x)
            us.append(u)
            du_dvs.append(du_dv)
        x_seq = torch.stack(xs, dim=1)
        u_seq = torch.stack(us, dim=1)
        du_dv_seq = torch.stack(du_dvs, dim=1)
        return x_seq, u_seq, du_dv_seq

    def forward(self, x0: torch.Tensor, q_seq: torch.Tensor, p_seq: torch.Tensor) -> torch.Tensor:
        assert x0.dim() == 2 and q_seq.dim() == 3 and p_seq.dim() == 3
        B = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        v_seq = torch.zeros((B, self.N, self.nu), device=device, dtype=dtype)

        reg = float(getattr(self.cfg, "mpc_lqr_reg", 1e-4))
        iters = int(getattr(self.cfg, "mpc_ilqr_iters", 3))

        n_cost = self.nx + self.nu
        if q_seq.shape[1] != self.N or p_seq.shape[1] != self.N or q_seq.shape[2] != n_cost or p_seq.shape[2] != n_cost:
            raise ValueError(
                f"Expected q_seq/p_seq shape (B,{self.N},{n_cost}), got q_seq={tuple(q_seq.shape)}, p_seq={tuple(p_seq.shape)}"
            )

        for _ in range(iters):
            x_seq, u_seq, du_dv_seq = self._rollout(x0, v_seq)

            A_seq = []
            B_seq = []
            for t in range(self.N):
                A_t, B_t = self._dyn_jac(x_seq[:, t], v_seq[:, t], u_seq[:, t], du_dv_seq[:, t])
                A_seq.append(A_t)
                B_seq.append(B_t)
            A_seq = torch.stack(A_seq, dim=1)
            B_seq = torch.stack(B_seq, dim=1)

            qx = q_seq[..., : self.nx]
            qu = q_seq[..., self.nx :]
            px = p_seq[..., : self.nx]
            pu = p_seq[..., self.nx :]

            l_x = 2.0 * qx * x_seq[:, : self.N, :] + px
            l_xx = torch.diag_embed(2.0 * qx)

            u = u_seq
            v = v_seq
            u_mid = 0.5 * (self.u_min.to(device=device, dtype=dtype) + self.u_max.to(device=device, dtype=dtype))
            u_scale = 0.5 * (self.u_max.to(device=device, dtype=dtype) - self.u_min.to(device=device, dtype=dtype))
            tnh = torch.tanh(v)
            du_dv = u_scale * (1.0 - tnh * tnh)
            d2u_dv2 = -2.0 * u_scale * tnh * (1.0 - tnh * tnh)

            l_u = 2.0 * qu * u + pu
            l_v = l_u * du_dv
            l_vv = torch.diag_embed(2.0 * qu * (du_dv * du_dv) + l_u * d2u_dv2)

            qxT = qx[:, -1, :]
            pxT = px[:, -1, :]
            xT = x_seq[:, -1, :]
            Vx = 2.0 * qxT * xT + pxT
            Vxx = torch.diag_embed(2.0 * qxT)

            K_seq = torch.zeros((B, self.N, self.nu, self.nx), device=device, dtype=dtype)
            k_seq = torch.zeros((B, self.N, self.nu), device=device, dtype=dtype)

            Iu = torch.eye(self.nu, device=device, dtype=dtype).unsqueeze(0)
            for t in reversed(range(self.N)):
                A_t = A_seq[:, t]
                B_t = B_seq[:, t]

                Qx = l_x[:, t] + torch.einsum("...ij,...j->...i", A_t.transpose(-1, -2), Vx)
                Qu = l_v[:, t] + torch.einsum("...ij,...j->...i", B_t.transpose(-1, -2), Vx)

                Qxx = l_xx[:, t] + torch.einsum("...ji,...jk,...kl->...il", A_t, Vxx, A_t)
                Quu = l_vv[:, t] + torch.einsum("...ji,...jk,...kl->...il", B_t, Vxx, B_t)
                Qux = torch.einsum("...ji,...jk,...kl->...il", B_t, Vxx, A_t)

                Quu_reg = Quu + reg * Iu
                K = -torch.linalg.solve(Quu_reg, Qux)
                k = -torch.linalg.solve(Quu_reg, Qu.unsqueeze(-1)).squeeze(-1)

                K_seq[:, t] = K
                k_seq[:, t] = k

                Vx = (
                    Qx
                    + torch.einsum("...ji,...jk,...k->...i", K, Quu_reg, k)
                    + torch.einsum("...ji,...j->...i", K, Qu)
                    # Qux is (nu, nx). We need Qux^T @ k -> (nx,)
                    + torch.einsum("...ij,...j->...i", Qux.transpose(-1, -2), k)
                )
                Vxx = (
                    Qxx
                    + torch.einsum("...ji,...jk,...kl->...il", K, Quu_reg, K)
                    + torch.einsum("...ji,...jk->...ik", K, Qux)
                    # Qux^T @ K -> (nx, nx)
                    + torch.einsum("...ij,...jk->...ik", Qux.transpose(-1, -2), K)
                )

            x = x0
            v_new = []
            for t in range(self.N):
                dx = x - x_seq[:, t]
                dv = k_seq[:, t] + torch.einsum("...ij,...j->...i", K_seq[:, t], dx)
                v_t = v_seq[:, t] + dv
                v_new.append(v_t)
                x, _u, _du_dv = self._dyn_step(x, v_t)
            v_seq = torch.stack(v_new, dim=1)

        x_seq, u_seq, _ = self._rollout(x0, v_seq)
        return u_seq[:, 0, :]


class MPCActorPyPose(nn.Module):
    def __init__(self, cfg: ACMPCPyPoseConfig, obs_dim: int, action_low: torch.Tensor, action_high: torch.Tensor, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.action_dim = int(action_low.numel())
        self.action_low = action_low.detach().to(device).reshape(-1)
        self.action_high = action_high.detach().to(device).reshape(-1)

        if int(cfg.mpc_nu) != self.action_dim:
            cfg.mpc_nu = self.action_dim

        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, cfg.cost_hidden),
            nn.ReLU(),
            nn.Linear(cfg.cost_hidden, cfg.cost_hidden),
            nn.ReLU(),
        )

        n_cost = int(cfg.mpc_nx + cfg.mpc_nu)
        self.cost_map = NeuralCostMapHorizon(
            in_dim=cfg.cost_hidden,
            n_cost=n_cost,
            horizon=int(cfg.mpc_N),
            q_lb=cfg.cost_q_lb,
            q_ub=cfg.cost_q_ub,
            p_lb=cfg.cost_p_lb,
            p_ub=cfg.cost_p_ub,
            hidden=cfg.cost_hidden,
        )

        u_min = self.action_low.cpu().numpy().astype(np.float64)
        u_max = self.action_high.cpu().numpy().astype(np.float64)
        self.mpc = ILQRMPCLayer(cfg, u_min=u_min, u_max=u_max)

        self.actor_std = nn.Parameter(torch.full((self.action_dim,), float(getattr(cfg, "actor_log_std_init", 0.0))))

    def _obs_to_x0(self, obs2: torch.Tensor) -> torch.Tensor:
        if self.cfg.mpc_state_mode != "hover_obs":
            return obs2[:, : self.cfg.mpc_nx]

        rpos = obs2[:, 0:3]
        quat_wxyz = obs2[:, 3:7]
        vel_w = obs2[:, 7:13]

        qn = torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True)
        quat_wxyz = quat_wxyz / (qn + 1e-8)

        rpy = quaternion_to_euler(quat_wxyz)
        pitch = rpy[:, 1].clamp(min=-math.pi / 2 + 1e-3, max=math.pi / 2 - 1e-3)
        rpy = torch.stack([rpy[:, 0], pitch, rpy[:, 2]], dim=-1)

        # If the env appends target quaternion, use orientation *error* so x0==0 means "at target pose".
        if bool(getattr(self.cfg, "obs_has_target_quat", False)):
            try:
                tdim = int(getattr(self.cfg, "obs_time_encoding_dim", 0) or 0)
                if tdim > 0:
                    target_quat = obs2[:, -tdim - 4 : -tdim]
                else:
                    target_quat = obs2[:, -4:]
                tnorm = torch.linalg.norm(target_quat, dim=-1, keepdim=True)
                target_quat = target_quat / (tnorm + 1e-8)
                target_rpy = quaternion_to_euler(target_quat)
                target_pitch = target_rpy[:, 1].clamp(min=-math.pi / 2 + 1e-3, max=math.pi / 2 - 1e-3)
                target_rpy = torch.stack([target_rpy[:, 0], target_pitch, target_rpy[:, 2]], dim=-1)

                # Wrap angle difference to [-pi, pi] for stability.
                rpy = torch.atan2(torch.sin(rpy - target_rpy), torch.cos(rpy - target_rpy))
            except Exception:
                pass

        v_w = vel_w[:, 0:3]
        w_w = vel_w[:, 3:6]
        v_b = quat_rotate_inverse(quat_wxyz, v_w)
        w_b = quat_rotate_inverse(quat_wxyz, w_w)

        x0 = torch.cat([rpos, rpy, v_b, w_b], dim=-1)
        return x0

    def forward(self, obs: torch.Tensor):
        orig_shape = obs.shape[:-1]
        obs2 = obs.reshape(-1, obs.shape[-1])

        feat = self.feature_net(obs2)
        q_seq, p_seq = self.cost_map(feat)
        x0 = self._obs_to_x0(obs2)

        u0 = self.mpc(x0, q_seq, p_seq)
        loc = torch.max(torch.min(u0, self.action_high), self.action_low)
        loc = loc.reshape(orig_shape + (self.action_dim,))

        log_std = self.actor_std
        log_std = log_std.clamp(
            min=float(getattr(self.cfg, "actor_log_std_min", -2.0)),
            max=float(getattr(self.cfg, "actor_log_std_max", 1.0)),
        )
        scale = torch.exp(log_std).view(*(1,) * (loc.dim() - 1), -1).expand_as(loc)
        return loc, scale


class ACMPCPyPosePolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: ACMPCPyPoseConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = float(getattr(cfg, "entropy_coef", 0.01))
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        if not self.cfg.use_mpc_head:
            raise RuntimeError("ac_mpc_pypose is expected to run with use_mpc_head=True.")

        obs_key = ("agents", "observation")
        obs_dim = int(observation_spec[obs_key].shape[-1])
        low, high = _extract_action_bounds(action_spec, self.device, self.action_dim)

        actor_core = MPCActorPyPose(
            self.cfg,
            obs_dim=obs_dim,
            action_low=low,
            action_high=high,
            device=self.device,
        )
        actor_module = TensorDictModule(
            actor_core,
            [("agents", "observation")],
            ["loc", "scale"]
        )

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        if self.cfg.priv_critic:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            self.critic = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                )
            ).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
                [("agents", "observation")], ["state_value"]
            ).to(self.device)

        # Actor has no lazy modules; avoid running iLQR on a dummy observation at init time.
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.)
            self.actor.apply(init_)
            self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]

        rewards = tensordict[("next", "agents", "reward")]
        dones = einops.repeat(
            tensordict[("next", "terminated")],
            "t e 1 -> t e a 1",
            a=self.n_agents
        )

        values = self.value_norm.denormalize(tensordict["state_value"])
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = (adv - adv.mean()) / adv.std().clamp(1e-7)

        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for _ in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)

        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1. - self.clip_param, 1. + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = -self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]

        values_clipped = b_values + (values - b_values).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.max(
            self.critic_loss_fn(b_returns, values),
            self.critic_loss_fn(b_returns, values_clipped),
        )

        loss = policy_loss + entropy_loss + value_loss

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5.0)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.actor_opt.step()
        self.critic_opt.step()

        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var().clamp(1e-8)

        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    usable = (tensordict.shape[0] // num_minibatches) * num_minibatches
    perm = torch.randperm(usable, device=tensordict.device).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
