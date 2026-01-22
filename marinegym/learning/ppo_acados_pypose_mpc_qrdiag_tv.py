import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor

from marinegym.utils.torch import euler_to_quaternion, normalize, quat_rotate_inverse

from .modules.distributions import IndependentNormal
from .ppo.common import GAE
from .utils.valuenorm import ValueNorm1

# Optional acados imports
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


@dataclass
class PPOAcadosPyPoseMPCQRDiagTVConfig:
    name: str = "ppo_acados_pypose_mpc_qrdiag_tv"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    priv_actor: bool = False
    priv_critic: bool = False
    checkpoint_path: Optional[str] = None

    entropy_coef: float = 0.001
    clip_param: float = 0.1
    actor_std: float = 0.15

    # Observation layout hints (populated by scripts/train.py).
    obs_has_target_quat: bool = False
    obs_time_encoding_dim: int = 0

    # MPC dims/settings (populated by scripts/train.py / task config).
    mpc_dt: float = 0.05
    mpc_horizon: int = 15
    mpc_nu: int = 6

    # Dynamics params (populated by scripts/train.py).
    mpc_param_yaml: Optional[str] = None
    mpc_alloc_npz: Optional[str] = None
    mpc_mass: Optional[float] = None
    mpc_inertia: Optional[List[float]] = None  # [xx, yy, zz]

    water_density: float = 997.0
    gravity: float = 9.81
    max_thruster_force: float = 40.0

    terminal_weight_mult: float = 10.0

    # acados codegen/cache paths (populated by scripts/train.py).
    acados_codegen_dir: str = "./acados_codegen"
    acados_json: Optional[str] = None

    # Solver knobs
    mpc_sqp_max_iter: int = 20
    mpc_qp_iter_max: int = 200
    mpc_uabs_eps: float = 1e-6
    # Regularize mass matrix inverse: inv(MRB + MA + eps*I). Helps avoid NaNs if params make MRB+MA ill-conditioned.
    mpc_massmat_eps: float = 1e-6
    # SQP globalization / robustness knobs (useful when seeing ACADOS_MINSTEP).
    mpc_qp_solver_mode: str = "ROBUST"
    mpc_globalization: str = "MERIT_BACKTRACKING"
    mpc_globalization_min_step: float = 1e-12
    # Hessian approximation: "EXACT" can be fragile with nonlinear dynamics; try "GAUSS_NEWTON" if MINSTEP occurs.
    mpc_hessian_approx: str = "EXACT"
    # NLP solver type: "SQP" (default) or "SQP_RTI" (often more robust / faster).
    mpc_nlp_solver_type: str = "SQP"

    # Input clipping (avoid numeric blow-ups / NaNs in dynamics).
    x0_clip_pos: float = 20.0
    x0_clip_vel: float = 20.0
    x0_clip_omega: float = 20.0

    # Learnable diagonal cost bounds (positive).
    wx_lb: float = 1e-3
    wx_ub: float = 1e2
    wu_lb: float = 1e-4
    wu_ub: float = 1.0
    weights_log_scale: bool = True
    cost_hidden: int = 256

    # Optional init (populated by scripts/train.py).
    wx_init: Optional[List[float]] = None  # (13,)
    wu_init: Optional[List[float]] = None  # (nu,)


cs = ConfigStore.instance()
cs.store("ppo_acados_pypose_mpc_qrdiag_tv", node=PPOAcadosPyPoseMPCQRDiagTVConfig, group="algo")
cs.store(
    "ppo_acados_pypose_mpc_qrdiag_tv_priv",
    node=PPOAcadosPyPoseMPCQRDiagTVConfig(priv_actor=True, priv_critic=True),
    group="algo",
)
cs.store(
    "ppo_acados_pypose_mpc_qrdiag_tv_priv_critic",
    node=PPOAcadosPyPoseMPCQRDiagTVConfig(priv_critic=True),
    group="algo",
)


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


def _inv_sigmoid(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = y.clamp(eps, 1.0 - eps)
    return torch.log(y) - torch.log1p(-y)


def _map_raw_to_positive(
    raw: torch.Tensor,
    lb: float,
    ub: float,
    *,
    log_scale: bool,
) -> torch.Tensor:
    if ub <= lb:
        raise ValueError(f"invalid bounds: lb={lb} ub={ub}")
    s = torch.sigmoid(raw)
    if log_scale:
        log_lb = math.log(lb)
        log_ub = math.log(ub)
        return torch.exp(log_lb + (log_ub - log_lb) * s)
    return lb + (ub - lb) * s


def _load_uav_params(cfg: PPOAcadosPyPoseMPCQRDiagTVConfig) -> dict:
    if cfg.mpc_param_yaml is None:
        raise ValueError("cfg.mpc_param_yaml is required (set by scripts/train.py).")
    if cfg.mpc_alloc_npz is None:
        raise ValueError("cfg.mpc_alloc_npz is required (set by scripts/train.py).")
    if cfg.mpc_mass is None:
        raise ValueError("cfg.mpc_mass is required (set by scripts/train.py).")
    if cfg.mpc_inertia is None or len(cfg.mpc_inertia) != 3:
        raise ValueError("cfg.mpc_inertia must be [xx, yy, zz] (set by scripts/train.py).")

    with open(cfg.mpc_param_yaml, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    hydro = params.get("hydro_coef", None)
    if hydro is None:
        raise ValueError(f"hydro_coef not found in {cfg.mpc_param_yaml}")

    alloc = np.load(cfg.mpc_alloc_npz)
    if "B" not in alloc:
        raise ValueError(f"'B' not found in {cfg.mpc_alloc_npz}")
    B = np.asarray(alloc["B"], dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != 6:
        raise ValueError(f"thruster allocation B must have shape (6,nu). got {B.shape}")

    inertia_xx, inertia_yy, inertia_zz = [float(x) for x in cfg.mpc_inertia]
    return {
        "name": params.get("name", "uav"),
        "mass": float(cfg.mpc_mass),
        "inertia": {"xx": inertia_xx, "yy": inertia_yy, "zz": inertia_zz},
        "hydro_coef": hydro,
        "thruster_allocation": B,
        "volume": float(params.get("volume", 0.0)),
        "coBM": float(params.get("coBM", 0.0)),
        "rho": float(cfg.water_density),
        "g": float(cfg.gravity),
        "max_thruster_force": float(cfg.max_thruster_force),
    }


class _NeuralDiagCostMapHorizonNoP(nn.Module):
    def __init__(self, cfg: PPOAcadosPyPoseMPCQRDiagTVConfig, *, horizon: int, nx: int, nu: int):
        super().__init__()
        self.cfg = cfg
        self.horizon = int(horizon)
        self.nx = int(nx)
        self.nu = int(nu)
        n_cost = self.nx + self.nu

        out_dim = self.horizon * n_cost
        hidden = int(cfg.cost_hidden)
        self.trunk = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.head = nn.Linear(hidden, out_dim)

        wx0 = torch.as_tensor(cfg.wx_init, dtype=torch.float32) if cfg.wx_init is not None else None
        wu0 = torch.as_tensor(cfg.wu_init, dtype=torch.float32) if cfg.wu_init is not None else None
        if wx0 is None:
            wx0 = torch.full((self.nx,), 1.0, dtype=torch.float32)
        if wu0 is None:
            wu0 = torch.full((self.nu,), 0.02, dtype=torch.float32)
        if int(wx0.numel()) != self.nx:
            raise ValueError(f"wx_init must have length {self.nx}. got {int(wx0.numel())}")
        if int(wu0.numel()) != self.nu:
            raise ValueError(f"wu_init must have length {self.nu}. got {int(wu0.numel())}")

        wx0 = wx0.clamp(min=cfg.wx_lb, max=cfg.wx_ub)
        wu0 = wu0.clamp(min=cfg.wu_lb, max=cfg.wu_ub)
        if cfg.weights_log_scale:
            sx0 = (torch.log(wx0) - math.log(cfg.wx_lb)) / (math.log(cfg.wx_ub) - math.log(cfg.wx_lb))
            su0 = (torch.log(wu0) - math.log(cfg.wu_lb)) / (math.log(cfg.wu_ub) - math.log(cfg.wu_lb))
        else:
            sx0 = (wx0 - cfg.wx_lb) / (cfg.wx_ub - cfg.wx_lb)
            su0 = (wu0 - cfg.wu_lb) / (cfg.wu_ub - cfg.wu_lb)

        raw0 = torch.cat([_inv_sigmoid(sx0), _inv_sigmoid(su0)], dim=-1)  # (n_cost,)
        raw0 = raw0.repeat(self.horizon)  # (horizon*n_cost,)
        with torch.no_grad():
            self.head.bias.copy_(raw0)

    def forward(self, obs_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_cost = self.nx + self.nu
        feat = self.trunk(obs_flat)
        raw = self.head(feat).view(obs_flat.shape[0], self.horizon, n_cost)

        raw_x = raw[..., : self.nx]
        raw_u = raw[..., self.nx :]
        w_x = _map_raw_to_positive(
            raw_x,
            float(self.cfg.wx_lb),
            float(self.cfg.wx_ub),
            log_scale=bool(self.cfg.weights_log_scale),
        )
        w_u = _map_raw_to_positive(
            raw_u,
            float(self.cfg.wu_lb),
            float(self.cfg.wu_ub),
            log_scale=bool(self.cfg.weights_log_scale),
        )
        return w_x, w_u


def _ca_cross(a, b):
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _ca_smooth_abs(x, eps: float):
    # Differentiable |x| approximation to avoid SQP/EXACT-Hessian NaNs near 0:
    # |x| ~= sqrt(x^2 + eps)
    return ca.sqrt(ca.power(x, 2) + float(eps))


def _ca_quat_to_rot_wxyz(q):
    # q = [w,x,y,z]
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = ca.SX(3, 3)
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - z * w)
    R[0, 2] = 2 * (x * z + y * w)
    R[1, 0] = 2 * (x * y + z * w)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - x * w)
    R[2, 0] = 2 * (x * z - y * w)
    R[2, 1] = 2 * (y * z + x * w)
    R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _ca_omega_mat(w):
    p, q, r = w[0], w[1], w[2]
    o = 0.0 * p
    return ca.vertcat(
        ca.horzcat(o, -p, -q, -r),
        ca.horzcat(p, o, r, -q),
        ca.horzcat(q, -r, o, p),
        ca.horzcat(r, q, -p, o),
    )


class AcadosMPCLayerPyPoseTV:
    """
    acados MPC layer for PyPose-style underwater dynamics:
      x = [pos_err(3), quat(wxyz,4), vel_b(6)] (nx=13)
      u = thruster command in [-1,1] (nu)

    Cost uses stage parameters:
      p_stage = [w_x(13), w_u(nu), target_quat(4)]  (no linear term p).

    The horizon is time-varying via stage-wise parameters p_stage[t].
    """

    def __init__(self, cfg: PPOAcadosPyPoseMPCQRDiagTVConfig, *, u_min: np.ndarray, u_max: np.ndarray):
        if not _HAS_ACADOS:  # pragma: no cover
            raise RuntimeError("acados/casadi not available. Install acados + acados_template + casadi.")
        self.cfg = cfg
        self.dt = float(cfg.mpc_dt)
        self.N = int(cfg.mpc_horizon)
        self.nx = 13
        self.nu = int(cfg.mpc_nu)
        self.u_min = np.asarray(u_min, dtype=np.float64).reshape(self.nu)
        self.u_max = np.asarray(u_max, dtype=np.float64).reshape(self.nu)

        uav = _load_uav_params(cfg)
        self.B_alloc = np.asarray(uav["thruster_allocation"], dtype=np.float64)
        self.mass = float(uav["mass"])
        self.inertia = np.array([uav["inertia"]["xx"], uav["inertia"]["yy"], uav["inertia"]["zz"]], dtype=np.float64)
        hydro = uav["hydro_coef"]
        self.added_mass = np.asarray(hydro["added_mass"], dtype=np.float64).reshape(6)
        self.d_lin = np.asarray(hydro["linear_damping"], dtype=np.float64).reshape(6)
        self.d_quad = np.asarray(hydro["quadratic_damping"], dtype=np.float64).reshape(6)
        self.volume = float(uav.get("volume", 0.0))
        self.coBM = float(uav.get("coBM", 0.0))
        self.rho = float(uav.get("rho", cfg.water_density))
        self.g = float(uav.get("g", cfg.gravity))
        self.max_thruster_force = float(uav.get("max_thruster_force", cfg.max_thruster_force))
        self.terminal_weight_mult = float(cfg.terminal_weight_mult)

        self._solver: Optional[AcadosOcpSolver] = None
        self._n_fail = 0
        self._last_u0 = np.zeros((self.nu,), dtype=np.float64)

    def _ensure_solver(self):
        if self._solver is None:
            self._solver = self._build_solver(self.cfg)

    def _build_solver(self, cfg: PPOAcadosPyPoseMPCQRDiagTVConfig) -> AcadosOcpSolver:
        ocp = AcadosOcp()
        model = AcadosModel()
        model.name = "pypose_mpc_alloc_tv"

        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)

        n_cost = self.nx + self.nu
        # Stage parameter encodes a (fixed) stage index for selecting time-varying weights from p_global.
        np_stage = 1
        p = ca.SX.sym("p", np_stage)
        stage_idx = p[0]  # intended to be set to stage index (0..N-1), terminal uses N-1

        # Global parameters hold the entire horizon's diagonal weights plus the tracking target quaternion.
        # p_global = [w_cost_seq_flat (N*(nx+nu)), target_quat (4)]
        np_global = self.N * n_cost + 4
        p_global = ca.SX.sym("p_global", np_global)

        w_all = p_global[0 : self.N * n_cost]
        tq = p_global[self.N * n_cost : self.N * n_cost + 4]

        pos = x[0:3]
        quat = x[3:7]
        v = x[7:10]
        w = x[10:13]

        # normalize quaternion
        quat = quat / ca.sqrt(ca.dot(quat, quat) + 1e-12)
        tq = tq / ca.sqrt(ca.dot(tq, tq) + 1e-12)

        # align target quaternion sign (q ~ -q)
        dot = ca.dot(quat, tq)
        tq = ca.if_else(dot < 0, -tq, tq)

        R = _ca_quat_to_rot_wxyz(quat)
        pos_dot = R @ v
        quat_dot = 0.5 * (_ca_omega_mat(w) @ quat)

        # command [-1, 1] -> force [N]
        u_force = u * float(self.max_thruster_force)
        Bdm = ca.DM(self.B_alloc)
        tau_wrench = Bdm @ u_force  # (6,)

        nu = ca.vertcat(v, w)  # (6,)

        # inertia matrices
        MRB = ca.diag(ca.vertcat(self.mass, self.mass, self.mass, self.inertia[0], self.inertia[1], self.inertia[2]))
        MA = ca.diag(ca.DM(self.added_mass.tolist()))
        massmat_eps = float(getattr(cfg, "mpc_massmat_eps", 1e-6))
        Minv = ca.inv(MRB + MA + massmat_eps * ca.DM.eye(6))

        # coriolis from added mass (match pypose_mpc_controller)
        ab = MA @ nu
        ab_lin = ab[0:3]
        ab_ang = ab[3:6]
        nu_lin = nu[0:3]
        nu_ang = nu[3:6]
        coriolis = ca.vertcat(
            -_ca_cross(ab_lin, nu_ang),
            -(_ca_cross(ab_lin, nu_lin) + _ca_cross(ab_ang, nu_ang)),
        )

        # damping: (d_lin + d_quad*|nu|) * nu
        abs_eps = float(getattr(cfg, "mpc_uabs_eps", 1e-6))
        nu_abs = _ca_smooth_abs(nu, abs_eps)
        damping = ca.diag(ca.DM(self.d_lin.tolist()) + ca.DM(self.d_quad.tolist()) * nu_abs) @ nu

        # gravity / buoyancy (world -> body via R^T)
        buoy = float(self.rho) * float(self.g) * float(self.volume)
        f_g_world = ca.DM([0.0, 0.0, -self.mass * self.g])
        f_b_world = ca.DM([0.0, 0.0, buoy])
        f_g_body = R.T @ f_g_world
        f_b_body = R.T @ f_b_world
        r_cb = ca.DM([0.0, 0.0, -float(self.coBM)])
        tau_b = ca.vertcat(f_b_body, _ca_cross(r_cb, f_b_body))
        tau_g = ca.vertcat(f_g_body, ca.DM([0.0, 0.0, 0.0]))

        hydro_wrench = -coriolis - damping + tau_b + tau_g
        nu_dot = Minv @ (tau_wrench + hydro_wrench)
        v_dot = nu_dot[0:3]
        w_dot = nu_dot[3:6]

        x_dot = ca.vertcat(pos_dot, quat_dot, v_dot, w_dot)
        x_next = x + self.dt * x_dot
        # renormalize quaternion state after integration
        q_next = x_next[3:7]
        q_next = q_next / ca.sqrt(ca.dot(q_next, q_next) + 1e-12)
        x_next = ca.vertcat(x_next[0:3], q_next, x_next[7:13])

        model.x = x
        model.u = u
        model.p = p
        model.p_global = p_global
        model.disc_dyn_expr = x_next
        ocp.model = model

        # horizon / dims
        if hasattr(ocp.solver_options, "N_horizon"):
            ocp.solver_options.N_horizon = int(self.N)
        else:
            ocp.dims.N = int(self.N)
        ocp.dims.nx = int(self.nx)
        ocp.dims.nu = int(self.nu)
        ocp.dims.np = int(np_stage)
        if hasattr(ocp.dims, "np_global"):
            ocp.dims.np_global = int(np_global)
        else:
            # fallback for older template versions
            ocp.dims.np_global = int(np_global)
        ocp.solver_options.tf = float(self.N * self.dt)

        # default param values
        ocp.parameter_values = np.zeros((np_stage,), dtype=np.float64)
        ocp.p_global_values = np.zeros((np_global,), dtype=np.float64)

        # Build time-varying weights w_cost(tau) from p_global using a sharp soft-selection.
        # This approximates per-stage (horizon-indexed) weights while keeping param sensitivities w.r.t p_global.
        beta = 200.0
        denom = 0.0 * stage_idx + 1e-12
        w_cost = ca.SX.zeros(n_cost, 1)
        for t in range(self.N):
            dtau = stage_idx - float(t)
            a = ca.exp(-beta * dtau * dtau)
            denom = denom + a
            w_t = w_all[t * n_cost : (t + 1) * n_cost]
            w_cost = w_cost + a * w_t
        w_cost = w_cost / denom

        w_x = w_cost[0 : self.nx]
        w_u = w_cost[self.nx : self.nx + self.nu]

        # cost (EXTERNAL): 0.5*(w_x*||x-x_ref||^2 + w_u*||u||^2)
        x_ref = ca.vertcat(ca.DM.zeros(3, 1), tq, ca.DM.zeros(6, 1))
        dx = x - x_ref
        stage_cost = 0.5 * (ca.dot(w_x, ca.power(dx, 2)) + ca.dot(w_u, ca.power(u, 2)))
        term_cost = 0.5 * self.terminal_weight_mult * ca.dot(w_x, ca.power(dx, 2))

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
            ocp.constraints.x0 = np.zeros((self.nx,), dtype=np.float64)

        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        if hasattr(ocp.solver_options, "hessian_approx"):
            ocp.solver_options.hessian_approx = str(getattr(cfg, "mpc_hessian_approx", "EXACT"))
        if hasattr(ocp.solver_options, "nlp_solver_type"):
            ocp.solver_options.nlp_solver_type = str(getattr(cfg, "mpc_nlp_solver_type", "SQP"))
        if hasattr(ocp.solver_options, "nlp_solver_max_iter"):
            ocp.solver_options.nlp_solver_max_iter = int(getattr(cfg, "mpc_sqp_max_iter", 20))
        if hasattr(ocp.solver_options, "qp_solver_iter_max"):
            ocp.solver_options.qp_solver_iter_max = int(getattr(cfg, "mpc_qp_iter_max", 200))
        if hasattr(ocp.solver_options, "qp_solver_mode"):
            ocp.solver_options.qp_solver_mode = str(getattr(cfg, "mpc_qp_solver_mode", "ROBUST"))
        if hasattr(ocp.solver_options, "globalization"):
            ocp.solver_options.globalization = str(getattr(cfg, "mpc_globalization", "MERIT_BACKTRACKING"))
        if hasattr(ocp.solver_options, "globalization_min_step"):
            ocp.solver_options.globalization_min_step = float(getattr(cfg, "mpc_globalization_min_step", 1e-12))
        if hasattr(ocp.solver_options, "with_solution_sens_wrt_params"):
            ocp.solver_options.with_solution_sens_wrt_params = True
        if hasattr(ocp, "ensure_solution_sensitivities_available"):
            ocp.ensure_solution_sensitivities_available()

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
            verbose=False,
        )
        return solver

    def solve_and_sens(
        self,
        x0_np: np.ndarray,
        w_cost_seq_np: np.ndarray,
        target_quat_np: np.ndarray,
        *,
        need_grad_w: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._ensure_solver()
        solver = self._solver

        x0_np = np.asarray(x0_np, dtype=np.float64).reshape(self.nx)
        w_cost_seq_np = np.asarray(w_cost_seq_np, dtype=np.float64)
        target_quat_np = np.asarray(target_quat_np, dtype=np.float64).reshape(4)

        n_cost = self.nx + self.nu
        if w_cost_seq_np.shape != (self.N, n_cost):
            raise ValueError(f"w_cost_seq must have shape ({self.N},{n_cost}). got {w_cost_seq_np.shape}")

        # Clip x0 to avoid dynamics overflow (especially during resets / large errors).
        pos_clip = float(getattr(self.cfg, "x0_clip_pos", 0.0))
        vel_clip = float(getattr(self.cfg, "x0_clip_vel", 0.0))
        omg_clip = float(getattr(self.cfg, "x0_clip_omega", 0.0))
        if pos_clip > 0:
            x0_np[0:3] = np.clip(x0_np[0:3], -pos_clip, pos_clip)
        if vel_clip > 0:
            x0_np[7:10] = np.clip(x0_np[7:10], -vel_clip, vel_clip)
        if omg_clip > 0:
            x0_np[10:13] = np.clip(x0_np[10:13], -omg_clip, omg_clip)

        # Normalize quaternion; if degenerate, fall back to identity.
        q = x0_np[3:7]
        qn = float(np.linalg.norm(q))
        if not np.isfinite(qn) or qn < 1e-8:
            x0_np[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            x0_np[3:7] = q / qn

        # Clip weights to configured bounds (guards rare numeric overshoots).
        wx_lb = float(getattr(self.cfg, "wx_lb", 1e-6))
        wx_ub = float(getattr(self.cfg, "wx_ub", 1e2))
        wu_lb = float(getattr(self.cfg, "wu_lb", 1e-6))
        wu_ub = float(getattr(self.cfg, "wu_ub", 1.0))
        w_cost_seq_np[:, 0 : self.nx] = np.clip(w_cost_seq_np[:, 0 : self.nx], wx_lb, wx_ub)
        w_cost_seq_np[:, self.nx : self.nx + self.nu] = np.clip(
            w_cost_seq_np[:, self.nx : self.nx + self.nu], wu_lb, wu_ub
        )

        # Finite-value guard: prevents ACADOS_NAN_DETECTED from propagating due to bad inputs.
        if not (
            np.all(np.isfinite(x0_np))
            and np.all(np.isfinite(w_cost_seq_np))
            and np.all(np.isfinite(target_quat_np))
        ):
            u0 = self._last_u0.copy()
            sens = None
            if need_grad_w:
                sens = np.zeros((self.nu, self.N * n_cost), dtype=np.float64)
            return u0, sens

        # set stage parameters: tau (stage index)
        for stage in range(self.N):
            solver.set(stage, "p", np.asarray([float(stage)], dtype=np.float64))
        solver.set(self.N, "p", np.asarray([float(self.N - 1)], dtype=np.float64))

        # set p_global = [w_cost_seq_flat, target_quat]
        p_global = np.concatenate([w_cost_seq_np.reshape(-1), target_quat_np.reshape(-1)], axis=0)
        if hasattr(solver, "set_p_global_and_precompute_dependencies"):
            solver.set_p_global_and_precompute_dependencies(p_global)
        elif hasattr(solver, "set_p_global"):
            solver.set_p_global(p_global)
        else:
            for stage in range(self.N + 1):
                solver.set(stage, "p_global", p_global)

        # set x0 constraint at stage 0
        set_ok = False
        last_err = None
        if hasattr(solver, "constraints_set"):
            try:
                solver.constraints_set(0, "lbx", x0_np)
                solver.constraints_set(0, "ubx", x0_np)
                set_ok = True
            except Exception as e:
                last_err = e
                set_ok = False
        if not set_ok:
            try:
                solver.set(0, "lbx", x0_np)
                solver.set(0, "ubx", x0_np)
                set_ok = True
            except Exception as e:
                last_err = e
                set_ok = False
        if not set_ok:
            try:
                solver.set(0, "x", x0_np)
                set_ok = True
            except Exception as e:
                last_err = e
                set_ok = False
        if not set_ok:
            raise RuntimeError(
                "Failed to apply initial-state constraint. "
                "Your acados build may not expose stage-0 lbx/ubx. "
                f"Last error: {last_err}"
            ) from last_err

        status = solver.solve()
        if status != 0:
            self._n_fail += 1
            u0 = self._last_u0.copy()
            sens = None
            if need_grad_w:
                # return zeros so training can continue
                sens = np.zeros((self.nu, self.N * n_cost), dtype=np.float64)
            return u0, sens

        u0 = np.asarray(solver.get(0, "u"), dtype=np.float64).reshape(self.nu)
        self._last_u0 = u0.copy()

        if not need_grad_w:
            return u0, None

        if not hasattr(solver, "eval_solution_sensitivity"):
            raise RuntimeError("This acados version does not expose eval_solution_sensitivity().")

        # Sensitivity wrt global parameters.
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

        np_global = self.N * n_cost + 4
        sens_u0 = np.asarray(sens_u0, dtype=np.float64).reshape(self.nu, np_global)

        # keep only the cost weights portion (drop target_quat dims)
        sens_u0 = sens_u0[:, : self.N * n_cost]  # (nu, N*n_cost)

        return u0, sens_u0

    def __call__(self, x0: torch.Tensor, w_cost_seq: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        return AcadosMPCFunctionTV.apply(x0, w_cost_seq, target_quat, self)


class AcadosMPCFunctionTV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0: torch.Tensor, w_cost_seq: torch.Tensor, target_quat: torch.Tensor, mpc_layer: AcadosMPCLayerPyPoseTV):
        assert x0.dim() == 2
        assert w_cost_seq.dim() == 3
        assert target_quat.dim() == 2
        B = x0.shape[0]
        N = int(mpc_layer.N)
        n_cost = int(mpc_layer.nx + mpc_layer.nu)
        if w_cost_seq.shape[0] != B or w_cost_seq.shape[1] != N or w_cost_seq.shape[2] != n_cost:
            raise ValueError(f"w_cost_seq must have shape (B,{N},{n_cost}). got {tuple(w_cost_seq.shape)}")
        if target_quat.shape != (B, 4):
            raise ValueError(f"target_quat must have shape (B,4). got {tuple(target_quat.shape)}")

        x0_np = x0.detach().cpu().double().numpy()
        w_np = w_cost_seq.detach().cpu().double().numpy()
        tq_np = target_quat.detach().cpu().double().numpy()

        need_grad_w = w_cost_seq.requires_grad
        u0_out = np.zeros((B, mpc_layer.nu), dtype=np.float64)
        sens_out = np.zeros((B, mpc_layer.nu, N * n_cost), dtype=np.float64) if need_grad_w else None

        for i in range(B):
            u0_i, sens_i = mpc_layer.solve_and_sens(x0_np[i], w_np[i], tq_np[i], need_grad_w=need_grad_w)
            u0_out[i] = u0_i
            if need_grad_w and sens_out is not None and sens_i is not None:
                sens_out[i] = sens_i

        u0 = torch.from_numpy(u0_out).to(device=x0.device, dtype=x0.dtype)
        if need_grad_w:
            sens = torch.from_numpy(sens_out).to(device=x0.device, dtype=x0.dtype)  # (B, nu, N*n_cost)
        else:
            sens = torch.empty(0, device=x0.device, dtype=x0.dtype)
        ctx.save_for_backward(sens)
        ctx.N = N
        ctx.n_cost = n_cost
        return u0

    @staticmethod
    def backward(ctx, grad_u0: torch.Tensor):
        (sens,) = ctx.saved_tensors
        N = int(ctx.N)
        n_cost = int(ctx.n_cost)

        grad_x0 = None
        grad_tq = None
        grad_w = None
        if sens.numel() != 0:
            # grad_u0: (B,nu) ; sens: (B,nu,N*n_cost) -> grad_w_flat: (B, N*n_cost)
            grad_w_flat = torch.einsum("bi,bij->bj", grad_u0, sens)
            grad_w = grad_w_flat.view(grad_u0.shape[0], N, n_cost)
        return grad_x0, grad_w, grad_tq, None


class _AcadosMPCMeanActor(nn.Module):
    def __init__(self, cfg: PPOAcadosPyPoseMPCQRDiagTVConfig, *, nu: int, action_low: torch.Tensor, action_high: torch.Tensor, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.nu = int(nu)
        self.nx = 13

        self.action_low = action_low.detach().to(device).reshape(-1)
        self.action_high = action_high.detach().to(device).reshape(-1)

        self.cost_map = _NeuralDiagCostMapHorizonNoP(cfg, horizon=int(cfg.mpc_horizon), nx=self.nx, nu=self.nu).to(device)

        u_min = self.action_low.cpu().numpy().astype(np.float64)
        u_max = self.action_high.cpu().numpy().astype(np.float64)
        self.mpc = AcadosMPCLayerPyPoseTV(cfg, u_min=u_min, u_max=u_max)

    def _extract_target_quat(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs_dim = int(obs_flat.shape[-1])
        time_dim = int(getattr(self.cfg, "obs_time_encoding_dim", 0))
        has_target_quat = bool(getattr(self.cfg, "obs_has_target_quat", False))

        if has_target_quat:
            drone_state_dim = obs_dim - time_dim - 3 - 4
            if drone_state_dim <= 0 or (drone_state_dim + 3 + 4) > (obs_dim - time_dim):
                raise ValueError(f"invalid obs layout: obs_dim={obs_dim} time_dim={time_dim}")
            tq_start = drone_state_dim + 3
            tq = obs_flat[:, tq_start : tq_start + 4]
            return normalize(tq)

        drone_state_dim = obs_dim - time_dim - 3
        if drone_state_dim <= 0 or (drone_state_dim + 3) > (obs_dim - time_dim):
            raise ValueError(f"invalid obs layout: obs_dim={obs_dim} time_dim={time_dim}")
        rheading = obs_flat[:, drone_state_dim : drone_state_dim + 3]
        drone_heading = obs_flat[:, 13:16]
        target_heading = drone_heading + rheading
        yaw = torch.atan2(target_heading[:, 1], target_heading[:, 0])
        rpy = torch.zeros((obs_flat.shape[0], 3), device=obs_flat.device, dtype=obs_flat.dtype)
        rpy[:, 2] = yaw
        return euler_to_quaternion(rpy)

    def _obs_to_x0(self, obs_flat: torch.Tensor) -> torch.Tensor:
        rpos = obs_flat[:, 0:3]
        quat = normalize(obs_flat[:, 3:7])
        vel_w = obs_flat[:, 7:13]
        v_b = quat_rotate_inverse(quat, vel_w[:, 0:3])
        w_b = quat_rotate_inverse(quat, vel_w[:, 3:6])
        vel_b = torch.cat([v_b, w_b], dim=-1)
        pos_err = -rpos
        return torch.cat([pos_err, quat, vel_b], dim=-1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.squeeze(-2)
        batch_shape = obs.shape[:-1]
        obs_flat = obs.reshape(-1, obs.shape[-1])

        x0 = self._obs_to_x0(obs_flat)
        target_quat = self._extract_target_quat(obs_flat)

        w_x_seq, w_u_seq = self.cost_map(obs_flat)
        w_cost_seq = torch.cat([w_x_seq, w_u_seq], dim=-1)  # (B, N, nx+nu)

        u0 = self.mpc(x0, w_cost_seq, target_quat)
        loc = u0.view(*batch_shape, 1, self.nu).to(dtype=obs.dtype)
        scale = torch.full_like(loc, float(self.cfg.actor_std))
        return loc, scale


class PPOAcadosPyPoseMPCQRDiagTVPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOAcadosPyPoseMPCQRDiagTVConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        if self.cfg.priv_actor:
            raise NotImplementedError("priv_actor not supported for this algo (keeps actor limited to cost-map).")

        self.entropy_coef = float(cfg.entropy_coef)
        self.clip_param = float(cfg.clip_param)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        self.n_agents, self.action_dim = action_spec.shape[-2:]
        if self.n_agents != 1:
            raise ValueError(f"Expected n_agents=1. got {self.n_agents}")
        self.gae = GAE(0.99, 0.95)

        action_low, action_high = _extract_action_bounds(action_spec, device=device, action_dim=self.action_dim)

        actor_core = _AcadosMPCMeanActor(
            cfg, nu=self.action_dim, action_low=action_low, action_high=action_high, device=device
        )

        actor_module = TensorDictModule(
            actor_core,
            in_keys=[("agents", "observation")],
            out_keys=["loc", "scale"],
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
            [("agents", "observation")],
            ["state_value"],
        ).to(self.device)

        fake_input = observation_spec.zero()
        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")].unsqueeze(-1)
        values = tensordict["state_value"]

        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for _epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
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
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = -self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(-self.clip_param, self.clip_param)
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()

        explained_var = 1.0 - F.mse_loss(values, b_returns) / b_returns.var().clamp_min(1e-8)
        return TensorDict(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
                "explained_var": explained_var,
            },
            [],
        )


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    usable = (tensordict.shape[0] // num_minibatches) * num_minibatches
    perm = torch.randperm(usable, device=tensordict.device).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
