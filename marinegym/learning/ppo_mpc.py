# ppo_mpc.py
# MIT License (c) 2023 Botian Xu, Tsinghua University
# + Modifications: Differentiable MPC head via acados solution sensitivities (p_global)

import os
import math
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Union, Optional, Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore

from .utils.valuenorm import ValueNorm1
from .modules.distributions import IndependentNormal
from .ppo.common import GAE

import threading


# -------------------------
# (Optional) acados imports
# -------------------------
try:
    import casadi as ca
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    _HAS_ACADOS = True
except Exception:
    _HAS_ACADOS = False
    ca = None
    AcadosOcp = None
    AcadosOcpSolver = None
    AcadosModel = None


# -------------------------
# Quaternion helpers (torch)
# -------------------------
def _quat_xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    # q: (..., 4) [x,y,z,w] -> [w,x,y,z]
    return torch.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], dim=-1)

def _quat_wxyz_conj(q: torch.Tensor) -> torch.Tensor:
    # q: (...,4) [w,x,y,z]
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def _quat_wxyz_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Hamilton product, a,b: (...,4) wxyz
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw*bw - ax*bx - ay*by - az*bz
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    return torch.stack([w, x, y, z], dim=-1)

def _quat_wxyz_rotate(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # rotate v by q (world = q ⊗ v ⊗ q*)
    zeros = torch.zeros_like(v[..., :1])
    vq = torch.cat([zeros, v], dim=-1)
    return _quat_wxyz_mul(_quat_wxyz_mul(q_wxyz, vq), _quat_wxyz_conj(q_wxyz))[..., 1:4]

def _quat_wxyz_rotate_inverse(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # rotate v by q^{-1} = q*
    return _quat_wxyz_rotate(_quat_wxyz_conj(q_wxyz), v)

def _quat_to_euler_zyx_from_wxyz(q: torch.Tensor) -> torch.Tensor:
    # returns roll, pitch, yaw (phi, theta, psi), ZYX convention
    w, x, y, z = q.unbind(-1)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    return torch.stack([roll, pitch, yaw], dim=-1)

def _guess_quat_order_from_obs(q4: torch.Tensor) -> str:
    """
    Hover 초기 자세가 identity 근처라는 가정에서 auto 추정.
    wxyz면 q[...,0]이 1에 가깝고, xyzw면 q[...,3]이 1에 가까움.
    """
    a0 = torch.abs(q4[..., 0]).mean()
    a3 = torch.abs(q4[..., 3]).mean()
    return "wxyz" if a0 > a3 else "xyzw"


# -------------------------
# Config
# -------------------------
@dataclass
class ACMPCConfig:
    name: str = "ac_mpc"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    # ✅ ConfigStore에서 쓰는 키들 (원본 PPO가 기대)
    priv_actor: bool = False
    priv_critic: bool = False
    checkpoint_path: Optional[str] = None

    # ---- Differentiable MPC head ----
    use_mpc_head: bool = True
    entropy_coef: float = 0.01
    actor_log_std_init: float = 0.0
    actor_log_std_min: float = -2.0
    actor_log_std_max: float = 1.0
    # If observation includes target quaternion, set MPC cost minimum to target rpy.
    mpc_track_target_rpy: bool = True
    obs_has_target_quat: bool = False
    obs_time_encoding_dim: int = 0

    mpc_N: int = 12
    mpc_dt: float = 0.02  # algo.mpc_dt=${sim.dt}

    mpc_nx: int = 12
    mpc_nu: int = 6  # env action_dim으로 동기화됨

    mpc_param_yaml: Optional[str] = None
    mpc_alloc_npz: Optional[str] = None

    mpc_uabs_eps: float = 1e-3

    cost_q_lb: float = 1e-3
    cost_q_ub: float = 1e2
    cost_p_ub: float = 1e1
    cost_hidden: int = 256
    # Cost-map initialization. "hover" gives a reasonable starting MPC behavior.
    cost_init: str = "hover"

    mpc_state_mode: str = "hover_obs"
    # "xyzw" / "wxyz" / "auto"
    mpc_quat_order: str = "auto"

    water_density: float = 997.0
    gravity: float = 9.81

    mpc_fail_action: str = "previous"
    mpc_grad_on_fail: bool = False

    # ✅ (필수) acados codegen/cache 경로
    acados_codegen_dir: str = "./acados_codegen"
    # ✅ (선택) json 경로를 따로 주고 싶으면 사용
    acados_json: Optional[str] = None

    # (Optional) override dynamics parameters if YAML doesn't include them.
    mpc_mass: Optional[float] = None
    mpc_inertia: Optional[List[float]] = None


cs = ConfigStore.instance()
cs.store("ac_mpc", node=ACMPCConfig, group="algo")
cs.store("ac_mpc_priv", node=ACMPCConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ac_mpc_priv_critic", node=ACMPCConfig(priv_critic=True), group="algo")


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


# -------------------------
# Neural Cost Map (Qdiag positive + p signed)
# -------------------------
class NeuralCostMap(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_cost: int,
        q_lb: float,
        q_ub: float,
        p_ub: float,
        hidden: int = 256,
    ):
        super().__init__()
        self.n_cost = n_cost
        self.q_lb = q_lb
        self.q_ub = q_ub
        self.p_ub = p_ub

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * n_cost),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        raw = self.net(feat)
        q_raw = raw[..., : self.n_cost]
        p_raw = raw[..., self.n_cost :]

        q = self.q_lb + (self.q_ub - self.q_lb) * torch.sigmoid(q_raw)  # positive
        p = self.p_ub * torch.tanh(p_raw)  # signed
        return torch.cat([q, p], dim=-1)  # p_global


# -------------------------
# Differentiable MPC via acados sensitivities (autograd.Function)
# -------------------------
class AcadosMPCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0: torch.Tensor, p_global: torch.Tensor, mpc_layer: "AcadosMPCLayer"):
        assert _HAS_ACADOS, "acados/casadi not available."
        assert x0.dim() == 2 and p_global.dim() == 2

        bs = x0.shape[0]
        nu = mpc_layer.nu
        npg = p_global.shape[1]

        x0_np = x0.detach().cpu().double().numpy()
        p_np = p_global.detach().cpu().double().numpy()

        need_grad_p = (ctx.needs_input_grad[1] and torch.is_grad_enabled())

        u0_out = np.zeros((bs, nu), dtype=np.float64)
        sens_u_out = np.zeros((bs, nu, npg), dtype=np.float64) if need_grad_p else None

        for i in range(bs):
            u0_i, sens_u_i = mpc_layer.solve_and_sens(x0_np[i], p_np[i], need_grad_p=need_grad_p)
            u0_out[i] = u0_i
            if need_grad_p:
                sens_u_out[i] = sens_u_i

        u0 = torch.from_numpy(u0_out).to(device=x0.device, dtype=x0.dtype)

        # ❗ save_for_backward에는 Tensor만 가능(None 금지)
        if need_grad_p:
            sens_u = torch.from_numpy(sens_u_out).to(device=x0.device, dtype=x0.dtype)
        else:
            sens_u = torch.empty(0, device=x0.device, dtype=x0.dtype)
        ctx.save_for_backward(sens_u)

        return u0

    @staticmethod
    def backward(ctx, grad_u0: torch.Tensor):
        (sens_u,) = ctx.saved_tensors
        grad_x0 = None
        grad_p = None
        grad_mpc_layer = None

        if sens_u.numel() != 0:
            grad_p = torch.bmm(
                sens_u.transpose(1, 2),       # (B, np, nu)
                grad_u0.unsqueeze(-1)         # (B, nu, 1)
            ).squeeze(-1)                     # (B, np)

        return grad_x0, grad_p, grad_mpc_layer


# -------------------------
# Acados MPC layer
# -------------------------
import os
import yaml
import numpy as np
import threading
from typing import Any, Dict

import torch
import torch.nn as nn

# (acados imports assumed)
# import casadi as ca
# from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


class AcadosMPCLayer(nn.Module):
    def __init__(self, cfg: ACMPCConfig, u_min: np.ndarray, u_max: np.ndarray):
        super().__init__()
        if not _HAS_ACADOS:
            raise RuntimeError("acados/casadi not installed.")

        self.cfg = cfg
        self.nx = int(cfg.mpc_nx)
        self.nu = int(cfg.mpc_nu)
        self.N = int(cfg.mpc_N)
        self.dt = float(cfg.mpc_dt)

        self.u_min = np.asarray(u_min, dtype=np.float64).reshape(-1)
        self.u_max = np.asarray(u_max, dtype=np.float64).reshape(-1)
        assert self.u_min.shape[0] == self.nu and self.u_max.shape[0] == self.nu, \
            f"u_min/u_max dim({self.u_min.shape[0]}) != mpc_nu({self.nu})"

        if cfg.mpc_alloc_npz is None:
            raise RuntimeError("cfg.mpc_alloc_npz is required (generated in train.py).")

        # Load thruster allocation (B) and (optionally) quaternion order used by the env/alloc generator.
        self.quat_order = None
        with np.load(cfg.mpc_alloc_npz, allow_pickle=True) as alloc:
            self.B = alloc["B"].astype(np.float64)  # (6,nu)
            # If available, read quaternion order used by the env/alloc generator.
            if "quat_order" in getattr(alloc, "files", []):
                qo = alloc["quat_order"]
                # npz may store str / bytes / 0-d array
                if isinstance(qo, np.ndarray) and getattr(qo, "shape", None) == ():
                    qo = qo.item()
                if isinstance(qo, bytes):
                    qo = qo.decode("utf-8", errors="ignore")
                qo = str(qo).strip().lower()
                if qo in ("xyzw", "wxyz"):
                    self.quat_order = qo
                    # Best-effort: also override cfg if it's set to auto (may be OmegaConf).
                    if getattr(cfg, "mpc_quat_order", "auto") == "auto":
                        try:
                            cfg.mpc_quat_order = qo
                        except Exception:
                            pass

        assert self.B.shape == (6, self.nu), f"B shape mismatch: {self.B.shape}, expected (6,{self.nu})"

        # ✅ 핵심: solver는 여기서 만들지 않는다 (deepcopy에서 터짐)
        self.solver = None
        self._solver_lock = threading.Lock()

        self._last_u0 = np.zeros((self.nu,), dtype=np.float64)
        self._last_u = None
        self._last_x = None

        # Diagnostics
        self._n_solves = 0
        self._n_fail = 0
        self._n_bad_inputs = 0
        self._last_status = 0

    def get_and_reset_diagnostics(self) -> Dict[str, float]:
        total = float(self._n_solves)
        fail = float(self._n_fail)
        bad = float(self._n_bad_inputs)
        last = float(self._last_status)
        self._n_solves = 0
        self._n_fail = 0
        self._n_bad_inputs = 0
        self._last_status = 0
        return {
            "mpc/solves": total,
            "mpc/fails": fail,
            "mpc/bad_inputs": bad,
            "mpc/fail_rate": (fail / total) if total > 0 else 0.0,
            "mpc/last_status": last,
        }

    # ✅ deepcopy/pickle 시 solver(ctypes pointer)를 항상 제거
    def __getstate__(self):
        state = self.__dict__.copy()
        state["solver"] = None
        state["_solver_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.solver = None
        self._solver_lock = threading.Lock()

    def _ensure_solver(self):
        if self.solver is None:
            with self._solver_lock:
                if self.solver is None:
                    self.solver = self._build_solver(self.cfg)

    def forward(self, x0: torch.Tensor, p_global: torch.Tensor) -> torch.Tensor:
        return AcadosMPCFunction.apply(x0, p_global, self)

    def _load_params(self, cfg: ACMPCConfig) -> Dict[str, Any]:
        params = {
            "mass": 10.0,
            "inertia": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "added_mass": np.zeros(6, dtype=np.float64),
            "linear_damping": np.zeros(6, dtype=np.float64),
            "quadratic_damping": np.zeros(6, dtype=np.float64),
            "volume": 0.01,
            "coBM": 0.0,
        }

        # Allow explicit overrides (e.g., from sim mass/inertia) if provided.
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

    def _build_solver(self, cfg: ACMPCConfig):
        assert self.nx == 12, "This MPC assumes nx=12 = [rpos(3), rpy(3), nu_body(6)]"

        ocp = AcadosOcp()
        model = AcadosModel()
        model.name = "bluerov_mpc_alloc"

        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)

        n_cost = self.nx + self.nu
        np_global = 2 * n_cost
        p_global = ca.SX.sym("p_global", np_global)

        rpos = x[0:3]
        rpy  = x[3:6]
        nu_b = x[6:12]

        phi, theta, psi = rpy[0], rpy[1], rpy[2]
        cphi, sphi = ca.cos(phi), ca.sin(phi)
        cth,  sth  = ca.cos(theta), ca.sin(theta)
        cpsi, spsi = ca.cos(psi), ca.sin(psi)

        R = ca.SX(3, 3)
        R[0, 0] = cpsi*cth
        R[0, 1] = cpsi*sth*sphi - spsi*cphi
        R[0, 2] = cpsi*sth*cphi + spsi*sphi
        R[1, 0] = spsi*cth
        R[1, 1] = spsi*sth*sphi + cpsi*cphi
        R[1, 2] = spsi*sth*cphi - cpsi*sphi
        R[2, 0] = -sth
        R[2, 1] = cth*sphi
        R[2, 2] = cth*cphi

        v_b = nu_b[0:3]
        w_b = nu_b[3:6]

        # Euler ZYX rate mapping (safe near cos(theta)=0).
        cth_safe = ca.if_else(cth >= 0, ca.fmax(cth, 1e-3), ca.fmin(cth, -1e-3))
        tan_theta = sth / cth_safe

        T = ca.SX(3, 3)
        T[0, 0] = 1
        T[0, 1] = sphi * tan_theta
        T[0, 2] = cphi * tan_theta
        T[1, 0] = 0
        T[1, 1] = cphi
        T[1, 2] = -sphi
        T[2, 0] = 0
        T[2, 1] = sphi / cth_safe
        T[2, 2] = cphi / cth_safe

        rpos_dot = -(R @ v_b)
        rpy_dot  = T @ w_b

        eps = float(cfg.mpc_uabs_eps)
        u_abs = ca.sqrt(u*u + eps*eps)
        u_sq  = u * u_abs

        Bdm = ca.DM(self.B)
        tau = Bdm @ u_sq

        prm = self._load_params(cfg)
        m = float(prm["mass"])
        I = prm["inertia"].astype(np.float64).reshape(3)
        added_mass = prm["added_mass"].astype(np.float64).reshape(6)
        Dlin  = prm["linear_damping"].astype(np.float64).reshape(6)
        Dquad = prm["quadratic_damping"].astype(np.float64).reshape(6)
        vol = float(prm["volume"])
        cobm = float(prm["coBM"])
        rho = float(cfg.water_density)
        g   = float(cfg.gravity)

        MRB = ca.diag(ca.vertcat(m, m, m, I[0], I[1], I[2]))
        MA  = ca.diag(ca.DM(added_mass.tolist()))
        M   = MRB + MA

        nu_abs = ca.sqrt(nu_b*nu_b + eps*eps)
        D = ca.diag(ca.DM(Dlin.tolist()) + ca.DM(Dquad.tolist()) * nu_abs)

        W = m * g
        Bbuoy = rho * g * vol
        d = cobm
        g_rest = ca.SX.zeros(6, 1)
        g_rest[0] = (W - Bbuoy) * sth
        g_rest[1] = -(W - Bbuoy) * sphi * cth
        g_rest[2] = -(W - Bbuoy) * cphi * cth
        g_rest[3] = -d * Bbuoy * cth * sphi
        g_rest[4] = -d * Bbuoy * sth
        g_rest[5] = 0.0

        nu_dot = ca.solve(M, (tau - D @ nu_b - g_rest))

        x_dot = ca.vertcat(rpos_dot, rpy_dot, nu_dot)
        x_next = x + self.dt * x_dot

        model.x = x
        model.u = u
        model.p_global = p_global
        model.disc_dyn_expr = x_next
        ocp.model = model

        # ✅ horizon / dims
        if hasattr(ocp.solver_options, "N_horizon"):
            ocp.solver_options.N_horizon = int(self.N)
        else:
            ocp.dims.N = int(self.N)

        ocp.dims.nx = int(self.nx)
        ocp.dims.nu = int(self.nu)

        if hasattr(ocp.dims, "np_global"):
            ocp.dims.np_global = int(np_global)
        else:
            ocp.dims.np = int(np_global)

        # ✅ tf는 반드시 float
        ocp.solver_options.tf = float(self.N * self.dt)

        # ✅ p_global 초기값(치수 일치) 반드시
        if hasattr(ocp, "p_global_values"):
            ocp.p_global_values = np.zeros((np_global,), dtype=np.float64)
        else:
            ocp.parameter_values = np.zeros((np_global,), dtype=np.float64)

        # cost (EXTERNAL)
        yv = ca.vertcat(x, u)
        q_diag = p_global[0:n_cost]
        p_lin  = p_global[n_cost:2*n_cost]
        stage_cost = ca.dot(q_diag, ca.power(yv, 2)) + ca.dot(p_lin, yv)
        term_cost  = ca.dot(q_diag[0:self.nx], ca.power(x, 2)) + ca.dot(p_lin[0:self.nx], x)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = term_cost

        # bounds
        ocp.constraints.lbu = self.u_min.copy()
        ocp.constraints.ubu = self.u_max.copy()
        ocp.constraints.idxbu = np.arange(self.nu)

        if hasattr(ocp.constraints, "idxbx_0"):
            ocp.constraints.idxbx_0 = np.arange(self.nx)
            ocp.constraints.lbx_0 = np.zeros((self.nx,), dtype=np.float64)
            ocp.constraints.ubx_0 = np.zeros((self.nx,), dtype=np.float64)
        elif hasattr(ocp.constraints, "x0"):
            # 구버전 호환
            ocp.constraints.x0 = np.zeros((self.nx,), dtype=np.float64)
        else:
            # 최후의 방어(버전이 weird한 경우)
            ocp.constraints.idxbx = np.arange(self.nx)
            ocp.constraints.lbx = -1e9 * np.ones((self.nx,), dtype=np.float64)
            ocp.constraints.ubx =  1e9 * np.ones((self.nx,), dtype=np.float64)

        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.nlp_solver_type = "SQP"

        # ---- Stabilization knobs (reduce QP failures / ACADOS_MINSTEP) ----
        # Levenberg-Marquardt regularization (if available)
        # NOTE: Parametric sensitivities in acados_template require an *exact* Hessian.
        # Setting levenberg_marquardt > 0 makes the Hessian regularized (not exact) and triggers:
        #   ValueError: Parametric sensitivities are only correct if an exact Hessian is used!
        if hasattr(ocp.solver_options, "levenberg_marquardt"):
            lm = float(getattr(cfg, "mpc_levenberg_marquardt", 0.0))
            if getattr(ocp.solver_options, "with_solution_sens_wrt_params", False):
                lm = 0.0
            ocp.solver_options.levenberg_marquardt = lm
        # HPIPM mode: ROBUST can be more stable than SPEED
        if hasattr(ocp.solver_options, "qp_solver_mode"):
            ocp.solver_options.qp_solver_mode = str(getattr(cfg, "mpc_qp_solver_mode", "ROBUST"))
        # Iteration limits (names vary by acados version)
        if hasattr(ocp.solver_options, "qp_solver_iter_max"):
            ocp.solver_options.qp_solver_iter_max = int(getattr(cfg, "mpc_qp_iter_max", 200))
        if hasattr(ocp.solver_options, "nlp_solver_max_iter"):
            ocp.solver_options.nlp_solver_max_iter = int(getattr(cfg, "mpc_sqp_max_iter", 20))
        if hasattr(ocp.solver_options, "globalization"):
            ocp.solver_options.globalization = str(getattr(cfg, "mpc_globalization", "MERIT_BACKTRACKING"))
        if hasattr(ocp.solver_options, "globalization_min_step"):
            ocp.solver_options.globalization_min_step = float(getattr(cfg, "mpc_globalization_min_step", 1e-10))

        if hasattr(ocp.solver_options, "with_solution_sens_wrt_params"):
            ocp.solver_options.with_solution_sens_wrt_params = True
        if hasattr(ocp, "ensure_solution_sensitivities_available"):
            ocp.ensure_solution_sensitivities_available()
        if hasattr(ocp.solver_options, "qp_solver_cond_ric_alg"):
            ocp.solver_options.qp_solver_cond_ric_alg = 0

        # codegen dir
        codegen_dir = os.path.abspath(getattr(cfg, "acados_codegen_dir", "./acados_codegen"))
        os.makedirs(codegen_dir, exist_ok=True)

        json_file = getattr(cfg, "acados_json", None)
        if json_file is None:
            json_file = os.path.join(codegen_dir, "acados_ocp.json")

        solver = AcadosOcpSolver(
            ocp,
            json_file=json_file,
            build=True,
            generate=True,
            verbose=False
        )
        return solver

    def solve_and_sens(self, x0_np: np.ndarray, p_np: np.ndarray,need_grad_p: bool = True):
        """
        Solve MPC with acados and (optionally) return du0/dp_global.
    
        Important:
          - This assumes your OCP actually defines an initial-state constraint for stage 0
            (idxbx_0/lbx_0/ubx_0 or x0). If not, setting lbx/ubx will have dimension 0 and
            you MUST fix _build_solver().
        """
        # ✅ lazy-build solver if you implemented it (recommended to avoid deepcopy/pickle issues)
        if hasattr(self, "_ensure_solver"):
            self._ensure_solver()
        solver = self.solver
    
        # -------------------------
        # shapes / sanity
        # -------------------------
        x0_np = np.asarray(x0_np, dtype=np.float64).reshape(self.nx)
        p_np = np.asarray(p_np, dtype=np.float64).reshape(-1)
    
        expected_np_global = 2 * (self.nx + self.nu)
        if p_np.size != expected_np_global:
            raise ValueError(
                f"p_global size mismatch: got {p_np.size}, expected {expected_np_global} "
                f"(= 2*(nx+nu) with nx={self.nx}, nu={self.nu})"
            )
        np_global = expected_np_global

        # -------------------------
        # quick NaN/Inf guard (prevents solver numeric blow-ups that often lead to ACADOS_MINSTEP)
        # -------------------------
        self._n_solves += 1
        if not (np.all(np.isfinite(x0_np)) and np.all(np.isfinite(p_np))):
            self._n_bad_inputs += 1
            u0 = (
                self._last_u0.copy()
                if getattr(self.cfg, "mpc_fail_action", "zero") == "previous"
                else np.zeros((self.nu,), dtype=np.float64)
            )
            sens = np.zeros((self.nu, np_global), dtype=np.float64) if need_grad_p else None
            return u0, sens

        # -------------------------
        # warm-start from previous solution (helps reduce MINSTEP / early QP failures)
        # -------------------------
        if getattr(self.cfg, "mpc_warm_start", True) and getattr(self, "_last_x", None) is not None:
            try:
                # Shift previous trajectory by 1 step (receding horizon).
                for stage in range(self.N):
                    x_guess = self._last_x[min(stage + 1, self.N)]
                    u_guess = self._last_u[min(stage + 1, self.N - 1)]
                    solver.set(stage, "x", x_guess)
                    solver.set(stage, "u", u_guess)
                solver.set(self.N, "x", self._last_x[self.N])
            except Exception:
                pass
    
        # -------------------------
        # set p_global
        # -------------------------
        if hasattr(solver, "set_p_global_and_precompute_dependencies"):
            solver.set_p_global_and_precompute_dependencies(p_np)
        elif hasattr(solver, "set_p_global"):
            solver.set_p_global(p_np)
        else:
            # very old fallback: try stage parameter field names
            for stage in range(self.N + 1):
                try:
                    solver.set(stage, "p_global", p_np)
                except Exception:
                    solver.set(stage, "p", p_np)
    
        # -------------------------
        # set x0 constraint at stage 0
        # -------------------------
        set_ok = False
        last_err = None
    
        # prefer constraints_set if available
        if hasattr(solver, "constraints_set"):
            try:
                solver.constraints_set(0, "lbx", x0_np)
                solver.constraints_set(0, "ubx", x0_np)
                set_ok = True
            except Exception as e:
                last_err = e
                set_ok = False
    
        # fallback: direct set
        if not set_ok:
            try:
                solver.set(0, "lbx", x0_np)
                solver.set(0, "ubx", x0_np)
                set_ok = True
            except Exception as e:
                last_err = e
                set_ok = False
    
        # last resort: warm start only (NOT a constraint)
        if not set_ok:
            try:
                solver.set(0, "x", x0_np)
                set_ok = True
            except Exception as e:
                last_err = e
                set_ok = False
    
        if not set_ok:
            # this is exactly your error case: lbx dimension is 0 because nbx_0 is 0
            raise RuntimeError(
                "Failed to apply initial-state constraint. "
                "Your solver reports lbx dim=0 (i.e., nbx_0=0). "
                "Fix _build_solver(): set stage-0 box constraints, e.g.\n"
                "  ocp.constraints.idxbx_0 = np.arange(nx)\n"
                "  ocp.constraints.lbx_0 = x0\n"
                "  ocp.constraints.ubx_0 = x0\n"
                "or (older acados_template) ocp.constraints.x0 = x0.\n"
                f"Last error: {last_err}"
            ) from last_err
    
        # -------------------------
        # solve
        # -------------------------
        status = solver.solve()
        if status != 0:
            self._n_fail += 1
            self._last_status = int(status)
            u0 = (
                self._last_u0.copy()
                if getattr(self.cfg, "mpc_fail_action", "zero") == "previous"
                else np.zeros((self.nu,), dtype=np.float64)
            )
            if not need_grad_p or (not getattr(self.cfg, "mpc_grad_on_fail", False)):
                sens = np.zeros((self.nu, np_global), dtype=np.float64) if need_grad_p else None
                return u0, sens
            sens = np.zeros((self.nu, np_global), dtype=np.float64)
            return u0, sens
    
        u0 = np.asarray(solver.get(0, "u"), dtype=np.float64).reshape(self.nu)
        self._last_u0 = u0.copy()

        # Save full trajectory for warm-starting next solve
        try:
            self._last_u = np.stack(
                [np.asarray(solver.get(i, "u"), dtype=np.float64).reshape(self.nu) for i in range(self.N)],
                axis=0,
            )
            self._last_x = np.stack(
                [np.asarray(solver.get(i, "x"), dtype=np.float64).reshape(self.nx) for i in range(self.N + 1)],
                axis=0,
            )
        except Exception:
            self._last_u = None
            self._last_x = None
    
        if not need_grad_p:
            return u0, None
    
        # -------------------------
        # sensitivity du/dp_global
        # -------------------------
        if not hasattr(solver, "eval_solution_sensitivity"):
            raise RuntimeError("This acados version does not expose eval_solution_sensitivity().")
    
        solver.eval_solution_sensitivity(with_respect_to="p_global")
    
        sens_u0 = None
        for key in ("sens_u", "sens_u0"):
            try:
                sens_u0 = solver.get(0, key)
                break
            except Exception:
                sens_u0 = None
    
        if sens_u0 is None:
            raise RuntimeError("Could not retrieve u sensitivity (expected solver.get(0,'sens_u')).")
    
        sens_u0 = np.asarray(sens_u0, dtype=np.float64).reshape(self.nu, np_global)
        return u0, sens_u0




# -------------------------
# MPC Actor head
# -------------------------
class MPCActor(nn.Module):
    def __init__(self, cfg: ACMPCConfig, obs_dim: int, action_low: torch.Tensor, action_high: torch.Tensor, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.action_dim = int(action_low.numel())
        self.action_low = action_low.detach().to(device).reshape(-1)
        self.action_high = action_high.detach().to(device).reshape(-1)

        # ✅ 누락 방지
        self.u_to_action = None

        if int(cfg.mpc_nu) != self.action_dim:
            cfg.mpc_nu = self.action_dim

        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, cfg.cost_hidden),
            nn.ReLU(),
            nn.Linear(cfg.cost_hidden, cfg.cost_hidden),
            nn.ReLU(),
        )

        n_cost = int(cfg.mpc_nx + cfg.mpc_nu)
        self.cost_map = NeuralCostMap(
            in_dim=cfg.cost_hidden,
            n_cost=n_cost,
            q_lb=cfg.cost_q_lb,
            q_ub=cfg.cost_q_ub,
            p_ub=cfg.cost_p_ub,
            hidden=cfg.cost_hidden,
        )
        self._init_cost_map(getattr(cfg, "cost_init", "none"))

        u_min = self.action_low.cpu().numpy().astype(np.float64)
        u_max = self.action_high.cpu().numpy().astype(np.float64)
        self.mpc = AcadosMPCLayer(cfg, u_min=u_min, u_max=u_max)

        self.actor_std = nn.Parameter(torch.full((self.action_dim,), float(getattr(cfg, "actor_log_std_init", 0.0))))

    def _init_cost_map(self, mode: str):
        mode = str(mode).strip().lower()
        if mode in ("none", "off", ""):
            return
        if mode not in ("hover",):
            raise ValueError(f"Unknown cost_init: {mode} (supported: none, hover)")

        # For Hover: prioritize position/orientation errors, regularize velocities,
        # keep action penalty small so the policy exits the thruster deadzone early.
        n_cost = int(self.cfg.mpc_nx + self.cfg.mpc_nu)
        q = torch.full((n_cost,), 1.0, dtype=torch.float32)
        q[0:3] = 50.0   # rpos
        q[3:6] = 5.0    # rpy
        q[6:9] = 2.0    # v_b
        q[9:12] = 0.5   # w_b
        q[self.cfg.mpc_nx:] = 0.05  # u

        q = q.clamp(min=float(self.cfg.cost_q_lb) + 1e-6, max=float(self.cfg.cost_q_ub) - 1e-6)
        q_sig = (q - float(self.cfg.cost_q_lb)) / (float(self.cfg.cost_q_ub) - float(self.cfg.cost_q_lb))
        q_sig = q_sig.clamp(1e-6, 1.0 - 1e-6)
        q_raw = torch.log(q_sig / (1.0 - q_sig))

        # p=0 initially
        p_raw = torch.zeros_like(q_raw)

        last = self.cost_map.net[-1]
        if not isinstance(last, nn.Linear):
            return
        with torch.no_grad():
            # Bias init; keep weights as-is so it can become state-dependent quickly.
            last.bias[:n_cost].copy_(q_raw.to(last.bias.device, dtype=last.bias.dtype))
            last.bias[n_cost:2 * n_cost].copy_(p_raw.to(last.bias.device, dtype=last.bias.dtype))

    def _obs_to_x0(self, obs2: torch.Tensor) -> torch.Tensor:
        if self.cfg.mpc_state_mode != "hover_obs":
            return obs2[:, : self.cfg.mpc_nx]

        rpos = obs2[:, 0:3]
        quat_raw = obs2[:, 3:7]
        vel_w = obs2[:, 7:13]

        # ✅ quat order 처리 (auto 지원)
        # Prefer alloc-npz provided quat order (train.py saved), fallback to cfg/auto-guess.
        order = getattr(self.mpc, "quat_order", None) or self.cfg.mpc_quat_order
        if order == "auto":
            order = _guess_quat_order_from_obs(quat_raw)

        if order == "wxyz":
            q_wxyz = quat_raw
        elif order == "xyzw":
            q_wxyz = _quat_xyzw_to_wxyz(quat_raw)
        else:
            raise ValueError(f"Unknown mpc_quat_order: {self.cfg.mpc_quat_order} (use xyzw/wxyz/auto)")

        # Normalize quaternion to avoid NaNs and Euler singularities from non-unit quats.


        q_norm = torch.linalg.norm(q_wxyz, dim=-1, keepdim=True)


        q_wxyz = q_wxyz / (q_norm + 1e-8)



        rpy = _quat_to_euler_zyx_from_wxyz(q_wxyz)


        # Clamp pitch away from +-pi/2 to avoid tan()/1/cos() blow-up in Euler-rate mapping.


        pitch = rpy[:, 1].clamp(min=-math.pi / 2 + 1e-3, max=math.pi / 2 - 1e-3)


        rpy = torch.stack([rpy[:, 0], pitch, rpy[:, 2]], dim=-1)

        v_w = vel_w[:, 0:3]
        w_w = vel_w[:, 3:6]
        v_b = _quat_wxyz_rotate_inverse(q_wxyz, v_w)
        w_b = _quat_wxyz_rotate_inverse(q_wxyz, w_w)

        x0 = torch.cat([rpos, rpy, v_b, w_b], dim=-1)  # 12
        return x0

    def forward(self, obs: torch.Tensor):
        orig_shape = obs.shape[:-1]
        obs2 = obs.reshape(-1, obs.shape[-1])

        feat = self.feature_net(obs2)
        p_global = self.cost_map(feat)

        # Optional: make rpy components track target rpy by shifting the linear term.
        if bool(getattr(self.cfg, "mpc_track_target_rpy", False)) and bool(getattr(self.cfg, "obs_has_target_quat", False)):
            try:
                tdim = int(getattr(self.cfg, "obs_time_encoding_dim", 0) or 0)
                # Hover appends target quat right before optional time encoding.
                if tdim > 0:
                    target_quat = obs2[:, -tdim - 4 : -tdim]
                else:
                    target_quat = obs2[:, -4:]
                # Normalize and convert to rpy (roll, pitch, yaw).
                qn = torch.linalg.norm(target_quat, dim=-1, keepdim=True)
                q_wxyz = target_quat / (qn + 1e-8)
                target_rpy = _quat_to_euler_zyx_from_wxyz(q_wxyz)

                n_cost = int(self.cfg.mpc_nx + self.cfg.mpc_nu)
                q_diag = p_global[:, :n_cost]
                p_lin = p_global[:, n_cost:]
                # For cost: q*(rpy - rpy_t)^2 = q*rpy^2 + (-2*q*rpy_t)*rpy + const
                p_lin[:, 3:6] = -2.0 * q_diag[:, 3:6] * target_rpy
                p_global = torch.cat([q_diag, p_lin], dim=-1)
            except Exception:
                pass

        x0 = self._obs_to_x0(obs2)
        u0 = self.mpc(x0, p_global)

        loc = self.u_to_action(u0) if self.u_to_action is not None else u0

        loc = torch.max(torch.min(loc, self.action_high), self.action_low)
        loc = loc.reshape(orig_shape + (self.action_dim,))

        log_std = self.actor_std
        log_std = log_std.clamp(
            min=float(getattr(self.cfg, "actor_log_std_min", -2.0)),
            max=float(getattr(self.cfg, "actor_log_std_max", 1.0)),
        )
        scale = torch.exp(log_std).view(*(1,) * (loc.dim() - 1), -1).expand_as(loc)
        return loc, scale


# -------------------------
# PPO policy
# -------------------------
class ACMPCPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: ACMPCConfig,
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

        if self.cfg.use_mpc_head:
            if not _HAS_ACADOS:
                raise RuntimeError("cfg.use_mpc_head=True but acados/casadi not available.")

            obs_key = ("agents", "observation")
            obs_dim = int(observation_spec[obs_key].shape[-1])

            low, high = _extract_action_bounds(action_spec, self.device, self.action_dim)

            actor_core = MPCActor(
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
        else:
            class Actor(nn.Module):
                def __init__(self, action_dim: int) -> None:
                    super().__init__()
                    self.actor_mean = nn.LazyLinear(action_dim)
                    self.actor_std = nn.Parameter(torch.zeros(action_dim))

                def forward(self, features: torch.Tensor):
                    loc = self.actor_mean(features)
                    scale = torch.exp(self.actor_std).expand_as(loc)
                    return loc, scale

            if self.cfg.priv_actor:
                intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
                actor_module = TensorDictSequential(
                    TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                    TensorDictModule(
                        nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                        [("agents", "intrinsics")], ["context"]
                    ),
                    CatTensors(["feature", "context"], "feature"),
                    TensorDictModule(
                        nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim)),
                        ["feature"], ["loc", "scale"]
                    )
                )
            else:
                actor_module = TensorDictModule(
                    nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim)),
                    [("agents", "observation")], ["loc", "scale"]
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

        if not self.cfg.use_mpc_head:
            self.actor(fake_input)
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
