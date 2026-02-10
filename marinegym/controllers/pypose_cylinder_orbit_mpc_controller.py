"""
PyPose 기반 differentiable Cylinder-Orbit MPC (iLQR, Gauss-Newton cost).

`CylinderOrbitMPCController`(acados)의 cost 항을 그대로 PyTorch로 구현해
autograd가 가능한 MPC를 제공합니다.

State / Input 정의는 기존 MPC와 동일합니다.
  - state x: [pos(3), quat(wxyz,4), vel_b(6)] = 13
  - input u: thruster command (nu) in [-1, 1]
    (dynamics 내부에서 `max_thruster_force`를 곱해 실제 thruster force[N]로 변환)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .controller import ControllerBase
from marinegym.utils.torch import quaternion_to_rotation_matrix, normalize

try:
    import pypose as pp

    _PYPOSE_AVAILABLE = True
    _PYPOSE_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover
    pp = None
    _PYPOSE_AVAILABLE = False
    _PYPOSE_IMPORT_ERROR = e


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cross(a, b, dim=-1)


def _omega_mat(w: torch.Tensor) -> torch.Tensor:
    # w: (..., 3) in body frame
    p, q, r = w.unbind(dim=-1)
    o = torch.zeros_like(p)
    return torch.stack(
        [
            torch.stack([o, -p, -q, -r], dim=-1),
            torch.stack([p, o, r, -q], dim=-1),
            torch.stack([q, -r, o, p], dim=-1),
            torch.stack([r, q, -p, o], dim=-1),
        ],
        dim=-2,
    )


def _quat_to_rp(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q: (..., 4) in wxyz, assumes normalized-ish
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    return roll, pitch


@dataclass(frozen=True)
class _UAVParams:
    mass: float
    inertia_xx: float
    inertia_yy: float
    inertia_zz: float
    added_mass: torch.Tensor  # (6,)
    linear_damping: torch.Tensor  # (6,)
    quadratic_damping: torch.Tensor  # (6,)
    thruster_allocation: torch.Tensor  # (6, nu)
    volume: float
    coBM: float
    rho: float
    g: float
    max_thruster_force: float


class _UnderwaterVehicleNLS(pp.module.NLS if _PYPOSE_AVAILABLE else object):
    """Same dynamics as `mpc_controller.py` / `pypose_mpc_controller.py` (Euler integration)."""

    def __init__(self, dt: float, params: _UAVParams, *, dtype: torch.dtype):
        if not _PYPOSE_AVAILABLE:  # pragma: no cover
            raise ImportError("PyPose가 필요합니다.") from _PYPOSE_IMPORT_ERROR
        super().__init__()
        self.dt = float(dt)
        self.params = params

        mass = torch.as_tensor(params.mass, dtype=dtype)
        I = torch.diag(
            torch.as_tensor(
                [params.inertia_xx, params.inertia_yy, params.inertia_zz],
                dtype=dtype,
            )
        )
        M_rb = torch.block_diag(mass * torch.eye(3, dtype=dtype), I)
        A_added = torch.diag(params.added_mass.to(dtype=dtype))
        self.register_buffer("M_inv", torch.linalg.inv(M_rb + A_added))
        self.register_buffer("A_added", A_added)
        self.register_buffer("d_lin", params.linear_damping.to(dtype=dtype))
        self.register_buffer("d_quad", params.quadratic_damping.to(dtype=dtype))
        self.register_buffer("B_alloc", params.thruster_allocation.to(dtype=dtype))

        buoyancy_force = float(params.rho) * float(params.g) * float(params.volume)
        self.buoyancy_force = buoyancy_force
        self.register_buffer(
            "_f_g_world",
            torch.tensor([0.0, 0.0, -params.mass * params.g], dtype=dtype),
        )
        self.register_buffer(
            "_f_b_world",
            torch.tensor([0.0, 0.0, buoyancy_force], dtype=dtype),
        )
        self.register_buffer(
            "_r_cb",
            torch.tensor([0.0, 0.0, -params.coBM], dtype=dtype),
        )

    def state_transition(self, state: torch.Tensor, input: torch.Tensor, t=None) -> torch.Tensor:
        # state: (B, 13), input: (B, nu) thruster command in [-1,1]
        state = state.to(dtype=self.M_inv.dtype)
        input = input.to(dtype=self.M_inv.dtype)

        p = state[..., 0:3]
        q = normalize(state[..., 3:7])
        v = state[..., 7:10]
        w = state[..., 10:13]

        # command [-1, 1] -> force [N]
        u_force = input * float(self.params.max_thruster_force)
        tau = torch.einsum("ij,...j->...i", self.B_alloc, u_force)  # (..., 6)

        nu = torch.cat([v, w], dim=-1)  # (..., 6)

        # exogenous flow input (world-frame linear flow) for relative-velocity damping
        if t is None:
            flow_w = torch.zeros_like(v)
        else:
            flow_w = torch.as_tensor(t, dtype=v.dtype, device=v.device)
            if flow_w.shape == (3,):
                flow_w = flow_w.view(1, 3).expand(v.shape[0], 3)
            if flow_w.shape != v.shape:
                raise ValueError(f"flow_w(t) must have shape {tuple(v.shape)} or (3,), got {tuple(flow_w.shape)}")

        # Convert world flow to body frame with the predicted attitude q_k.
        R = quaternion_to_rotation_matrix(q)
        flow_b = torch.einsum("...ji,...j->...i", R, flow_w)
        nu_rel = torch.cat([v - flow_b, w], dim=-1)

        # coriolis
        ab = torch.einsum("ij,...j->...i", self.A_added, nu)
        ab_lin, ab_ang = ab[..., 0:3], ab[..., 3:6]
        nu_lin, nu_ang = nu[..., 0:3], nu[..., 3:6]
        coriolis = torch.cat(
            [
                -_cross(ab_lin, nu_ang),
                -(_cross(ab_lin, nu_lin) + _cross(ab_ang, nu_ang)),
            ],
            dim=-1,
        )

        # damping: (D_lin + D_quad*|nu|) nu  (D_* are diagonal)
        damping = (self.d_lin + self.d_quad * torch.abs(nu_rel)) * nu_rel

        # gravity/buoyancy in body frame
        f_g_body = torch.einsum("...ji,...j->...i", R, self._f_g_world.to(dtype=R.dtype, device=R.device))
        f_b_body = torch.einsum("...ji,...j->...i", R, self._f_b_world.to(dtype=R.dtype, device=R.device))
        r_cb = self._r_cb.to(dtype=R.dtype, device=R.device)
        tau_b = torch.cat([f_b_body, _cross(r_cb.expand_as(f_b_body), f_b_body)], dim=-1)
        tau_g = torch.cat([f_g_body, torch.zeros_like(f_g_body)], dim=-1)

        hydro_wrench = -coriolis - damping + tau_b + tau_g

        # kinematics
        p_dot = torch.einsum("...ij,...j->...i", R, v)
        q_dot = 0.5 * torch.einsum("...ij,...j->...i", _omega_mat(w), q)

        # dynamics: nu_dot = inv(M_rb + A) (tau + hydro)
        nu_dot = torch.einsum("ij,...j->...i", self.M_inv, tau + hydro_wrench)
        v_dot, w_dot = nu_dot[..., 0:3], nu_dot[..., 3:6]

        dt = self.dt
        p_next = p + dt * p_dot
        q_next = normalize(q + dt * q_dot)
        v_next = v + dt * v_dot
        w_next = w + dt * w_dot
        return torch.cat([p_next, q_next, v_next, w_next], dim=-1)

    def observation(self, state: torch.Tensor, input: torch.Tensor, t=None) -> torch.Tensor:
        return state


def _orbit_errors(
    x: torch.Tensor,
    *,
    center_w: torch.Tensor,
    radius: torch.Tensor,
    z: torch.Tensor,
    v_tan: torch.Tensor,
    dir_sign: torch.Tensor,
    yaw_offset: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Error vector used by acados `CylinderOrbitMPCController`.

    Returns: (..., 10)
      [radial_err, z_err, tan_err, v_rad, heading_err_x, heading_err_y, roll, pitch, w_x, w_y]
    """
    pos = x[..., 0:3]
    quat = normalize(x[..., 3:7])
    v_b = x[..., 7:10]
    w_b = x[..., 10:13]

    rel_xy = center_w[..., 0:2] - pos[..., 0:2]
    dist_xy = torch.sqrt((rel_xy.square()).sum(dim=-1, keepdim=True) + eps)  # (..., 1)
    h_des_xy = rel_xy / dist_xy  # (..., 2)

    co = torch.cos(yaw_offset)
    so = torch.sin(yaw_offset)
    h_tgt_xy = torch.stack(
        [
            co * h_des_xy[..., 0] - so * h_des_xy[..., 1],
            so * h_des_xy[..., 0] + co * h_des_xy[..., 1],
        ],
        dim=-1,
    )

    R = quaternion_to_rotation_matrix(quat)  # (..., 3, 3)
    h_act_xy = torch.stack([R[..., 0, 0], R[..., 1, 0]], dim=-1)
    h_act_xy = h_act_xy / (torch.linalg.norm(h_act_xy, dim=-1, keepdim=True) + eps)
    heading_err = h_act_xy - h_tgt_xy  # (..., 2)

    radial_err = dist_xy.squeeze(-1) - radius
    z_err = pos[..., 2] - z

    v_w = torch.einsum("...ij,...j->...i", R, v_b)
    v_xy = v_w[..., 0:2]
    t_des_xy = torch.stack([-h_des_xy[..., 1], h_des_xy[..., 0]], dim=-1) * dir_sign
    v_tan_act = (t_des_xy * v_xy).sum(dim=-1)
    v_rad = (h_des_xy * v_xy).sum(dim=-1)
    tan_err = v_tan_act - v_tan

    roll, pitch = _quat_to_rp(quat)
    w_x = w_b[..., 0]
    w_y = w_b[..., 1]

    return torch.stack(
        [
            radial_err,
            z_err,
            tan_err,
            v_rad,
            heading_err[..., 0],
            heading_err[..., 1],
            roll,
            pitch,
            w_x,
            w_y,
        ],
        dim=-1,
    )


class PyPoseCylinderOrbitMPCController(ControllerBase):
    def __init__(
        self,
        *,
        uav_params: dict,
        mpc_dtype: torch.dtype = torch.float32,
        dt: float = 0.05,
        horizon: int = 10,
        batch_size: int = 1,
        ilqr_iters: int = 5,
        ilqr_reg: float = 1e-3,
        terminal_weight_mult: float = 1.0,
        max_thruster_force: float = 40.0,
        q_radial: float = 50.0,
        q_z: float = 30.0,
        q_tan: float = 10.0,
        q_radial_speed: float = 5.0,
        q_heading: float = 30.0,
        q_roll: float = 30.0,
        q_pitch: float = 10.0,
        q_wxy: float = 0.5,
        r_u: float = 0.01,
    ):
        super().__init__()
        if not _PYPOSE_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "PyPoseCylinderOrbitMPCController를 사용하려면 pypose가 필요합니다."
            ) from _PYPOSE_IMPORT_ERROR

        B_np = np.asarray(uav_params["thruster_allocation"], dtype=np.float64)
        if B_np.ndim != 2 or B_np.shape[0] != 6:
            raise ValueError(f"thruster_allocation must have shape (6, nu). got {B_np.shape}")
        nu = int(B_np.shape[1])

        self.nu = nu
        self.nx = 13
        self.ne = 10
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.batch_size = int(batch_size)
        self.ilqr_iters = int(ilqr_iters)
        self.ilqr_reg = float(ilqr_reg)
        self.terminal_weight_mult = float(terminal_weight_mult)
        self.max_thruster_force = float(max_thruster_force)
        self.mpc_dtype = mpc_dtype

        hydro = uav_params["hydro_coef"]
        params = _UAVParams(
            mass=float(uav_params["mass"]),
            inertia_xx=float(uav_params["inertia"]["xx"]),
            inertia_yy=float(uav_params["inertia"]["yy"]),
            inertia_zz=float(uav_params["inertia"]["zz"]),
            added_mass=torch.as_tensor(hydro["added_mass"], dtype=mpc_dtype),
            linear_damping=torch.as_tensor(hydro["linear_damping"], dtype=mpc_dtype),
            quadratic_damping=torch.as_tensor(hydro["quadratic_damping"], dtype=mpc_dtype),
            thruster_allocation=torch.as_tensor(B_np, dtype=mpc_dtype),
            volume=float(uav_params.get("volume", 0.0)),
            coBM=float(uav_params.get("coBM", 0.0)),
            rho=float(uav_params.get("rho", 997.0)),
            g=float(uav_params.get("g", 9.81)),
            max_thruster_force=self.max_thruster_force,
        )
        self.system = _UnderwaterVehicleNLS(self.dt, params, dtype=mpc_dtype)

        w_err = torch.tensor(
            [
                float(q_radial),
                float(q_z),
                float(q_tan),
                float(q_radial_speed),
                float(q_heading),
                float(q_heading),
                float(q_roll),
                float(q_pitch),
                float(q_wxy),
                float(q_wxy),
            ],
            dtype=mpc_dtype,
        )
        # acados cost penalizes thruster *forces*; our decision variable is command in [-1,1]
        # => weight(cmd) = weight(force) * (max_thruster_force^2)
        w_u = torch.tensor([float(r_u) * (self.max_thruster_force**2)] * nu, dtype=mpc_dtype)
        self.register_buffer("_w_err", w_err)
        self.register_buffer("_w_u", w_u)

        self._u_warm: torch.Tensor | None = None
        self._last_solve_stats: dict[str, float | int | bool | str] = {}
        self._diag_accum: dict[str, float] = {}
        self._diag_count: int = 0
        self._prev_energy_mean: float | None = None
        self.requires_grad_(False)

    def _trajectory_cost(
        self,
        x_traj: torch.Tensor,
        u_traj: torch.Tensor,
        *,
        center_w: torch.Tensor,
        radius_t: torch.Tensor,
        z_t: torch.Tensor,
        v_tan_t: torch.Tensor,
        dir_sign: torch.Tensor,
        yaw_offset_t: torch.Tensor,
        w_err_seq: torch.Tensor | None,
        w_u_seq: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, nu = u_traj.shape
        dtype = x_traj.dtype
        device = x_traj.device

        x_flat = x_traj[:, :-1, :].reshape(B * T, self.nx)
        c_flat = center_w.unsqueeze(1).expand(B, T, center_w.shape[-1]).reshape(B * T, center_w.shape[-1])
        e = _orbit_errors(
            x_flat,
            center_w=c_flat,
            radius=radius_t,
            z=z_t,
            v_tan=v_tan_t,
            dir_sign=dir_sign,
            yaw_offset=yaw_offset_t,
        ).view(B, T, self.ne)

        if w_err_seq is None:
            w_err_b = self._w_err.to(device=device, dtype=dtype).view(1, 1, self.ne).expand(B, T, self.ne)
            w_err_T = self._w_err.to(device=device, dtype=dtype).view(1, self.ne).expand(B, self.ne)
        else:
            w_err_b = w_err_seq
            w_err_T = w_err_seq[:, -1, :]

        if w_u_seq is None:
            w_u_b = self._w_u.to(device=device, dtype=dtype).view(1, 1, nu).expand(B, T, nu)
        else:
            w_u_b = w_u_seq

        stage = 0.5 * (w_err_b * e.square()).sum(dim=(-1, -2)) + 0.5 * (w_u_b * u_traj.square()).sum(dim=(-1, -2))

        eT = _orbit_errors(
            x_traj[:, -1, :],
            center_w=center_w,
            radius=radius_t,
            z=z_t,
            v_tan=v_tan_t,
            dir_sign=dir_sign,
            yaw_offset=yaw_offset_t,
        )
        terminal = 0.5 * float(self.terminal_weight_mult) * (w_err_T * eT.square()).sum(dim=-1)
        return stage + terminal

    def _accumulate_diagnostics(self, stats: dict[str, float | int | bool | str]) -> None:
        self._last_solve_stats = stats
        self._diag_count += 1
        self._diag_accum["n_iters"] = self._diag_accum.get("n_iters", 0.0) + float(stats.get("n_iters", 0.0))
        self._diag_accum["converged"] = self._diag_accum.get("converged", 0.0) + (1.0 if bool(stats.get("converged", False)) else 0.0)
        for k in ("final_cost", "cost_reduction", "alpha", "lambda", "sat_ratio", "roll_abs", "pitch_abs", "v_rel_norm", "energy_spike"):
            self._diag_accum[k] = self._diag_accum.get(k, 0.0) + float(stats.get(k, 0.0))

    def get_and_reset_diagnostics(self) -> dict[str, float | str]:
        if self._diag_count <= 0:
            out = {}
            if self._last_solve_stats:
                out["ilqr_stop_reason"] = str(self._last_solve_stats.get("stop_reason", "none"))
            return out
        n = float(self._diag_count)
        out = {
            "ilqr_n_iters": self._diag_accum.get("n_iters", 0.0) / n,
            "ilqr_converged": self._diag_accum.get("converged", 0.0) / n,
            "ilqr_final_cost": self._diag_accum.get("final_cost", 0.0) / n,
            "ilqr_cost_reduction": self._diag_accum.get("cost_reduction", 0.0) / n,
            "ilqr_alpha": self._diag_accum.get("alpha", 0.0) / n,
            "ilqr_lambda": self._diag_accum.get("lambda", 0.0) / n,
            "sat_ratio": self._diag_accum.get("sat_ratio", 0.0) / n,
            "roll_abs": self._diag_accum.get("roll_abs", 0.0) / n,
            "pitch_abs": self._diag_accum.get("pitch_abs", 0.0) / n,
            "v_rel_norm": self._diag_accum.get("v_rel_norm", 0.0) / n,
            "energy_spike": self._diag_accum.get("energy_spike", 0.0) / n,
            "ilqr_stop_reason": str(self._last_solve_stats.get("stop_reason", "none")),
        }
        self._diag_accum = {}
        self._diag_count = 0
        return out

    def _dynamics(self, x: torch.Tensor, u: torch.Tensor, flow_w: torch.Tensor | None = None) -> torch.Tensor:
        return self.system.state_transition(x, u, flow_w)

    def _rollout(
        self,
        x0: torch.Tensor,
        u_traj: torch.Tensor,
        flow_w_seq: torch.Tensor | None = None,
        flow_w_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        xs = [x0]
        x = x0
        for t in range(u_traj.shape[1]):
            flow_wt = None
            if flow_w_fn is not None:
                flow_wt = flow_w_fn(x, t)
            elif flow_w_seq is not None:
                flow_wt = flow_w_seq[:, t, :]
            x = self._dynamics(x, u_traj[:, t, :], flow_wt)
            xs.append(x)
        return torch.stack(xs, dim=1)  # (B, T+1, nx)

    def _ilqr(
        self,
        x0: torch.Tensor,
        *,
        center_w: torch.Tensor,
        radius: float | torch.Tensor,
        z: float | torch.Tensor,
        v_tan: float | torch.Tensor,
        direction: float | torch.Tensor = 1.0,
        yaw_offset: float | torch.Tensor = 0.0,
        u_init: torch.Tensor,
        w_err_seq: torch.Tensor | None = None,
        w_u_seq: torch.Tensor | None = None,
        gamma_d: torch.Tensor | None = None,
        x_meas_seq: torch.Tensor | None = None,
        flow_w_seq: torch.Tensor | None = None,
        flow_w_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
        try:
            from torch.func import jacrev, vmap  # type: ignore
        except Exception:  # pragma: no cover
            from functorch import jacrev, vmap  # type: ignore

        B, T, nu = u_init.shape
        u = u_init

        dtype = x0.dtype
        device = x0.device

        # constants as tensors (enable optional autograd through them)
        radius_t = torch.as_tensor(radius, dtype=dtype, device=device)
        z_t = torch.as_tensor(z, dtype=dtype, device=device)
        v_tan_t = torch.as_tensor(v_tan, dtype=dtype, device=device)
        direction_t = torch.as_tensor(direction, dtype=dtype, device=device)
        dir_sign = torch.where(direction_t >= 0, torch.tensor(1.0, dtype=dtype, device=device), torch.tensor(-1.0, dtype=dtype, device=device))
        yaw_offset_t = torch.as_tensor(yaw_offset, dtype=dtype, device=device)

        w_err = self._w_err.to(device=device, dtype=dtype)  # (ne,)
        w_u = self._w_u.to(device=device, dtype=dtype)  # (nu,)

        def f_single(x, u, fb):
            x_next = self._dynamics(x.unsqueeze(0), u.unsqueeze(0), fb.unsqueeze(0)).squeeze(0)
            return x_next

        jac_x = jacrev(f_single, argnums=0)
        jac_u = jacrev(f_single, argnums=1)
        jac_x_b = vmap(jac_x, in_dims=(0, 0, 0))
        jac_u_b = vmap(jac_u, in_dims=(0, 0, 0))

        def e_single(x, c):
            return _orbit_errors(
                x,
                center_w=c,
                radius=radius_t,
                z=z_t,
                v_tan=v_tan_t,
                dir_sign=dir_sign,
                yaw_offset=yaw_offset_t,
            )

        jac_e = jacrev(e_single, argnums=0)  # (ne, nx)
        jac_e_b = vmap(jac_e, in_dims=(0, 0))
        e_b = vmap(e_single, in_dims=(0, 0))

        Iu = torch.eye(nu, device=device, dtype=dtype).unsqueeze(0).expand(B, nu, nu)

        w_err = self._w_err.to(device=device, dtype=dtype)
        w_u_force = self._w_u.to(device=device, dtype=dtype)
        # Allow time-varying weights (per-step). We interpret w_u_seq as "force-penalty" weights (like acados),
        # then scale to the decision-variable space (command in [-1,1]).
        if w_err_seq is not None:
            w_err_seq = w_err_seq.to(device=device, dtype=dtype)
            if w_err_seq.shape != (B, T, self.ne):
                raise ValueError(f"w_err_seq must have shape ({B},{T},{self.ne}). got {tuple(w_err_seq.shape)}")
        if w_u_seq is not None:
            w_u_seq = w_u_seq.to(device=device, dtype=dtype)
            if w_u_seq.shape != (B, T, nu):
                raise ValueError(f"w_u_seq must have shape ({B},{T},{nu}). got {tuple(w_u_seq.shape)}")
            w_u_seq = w_u_seq * float(self.max_thruster_force**2)

        gamma_d_b = None

        flow_seq = None
        if flow_w_seq is not None:
            flow_seq = flow_w_seq.to(device=device, dtype=dtype)
            if flow_seq.shape != (B, T, 3):
                raise ValueError(f"flow_w_seq must have shape ({B},{T},3). got {tuple(flow_seq.shape)}")
        if gamma_d is not None:
            gamma_d_b = torch.as_tensor(gamma_d, dtype=dtype, device=device).reshape(-1)
            if gamma_d_b.numel() == 1 and B > 1:
                gamma_d_b = gamma_d_b.expand(B)
            if gamma_d_b.numel() != B:
                raise ValueError(f"gamma_d must have shape ({B},) or scalar. got {tuple(gamma_d_b.shape)}")

        x_traj0 = self._rollout(x0, u, flow_seq, flow_w_fn)
        initial_cost = self._trajectory_cost(
            x_traj0,
            u,
            center_w=center_w,
            radius_t=radius_t,
            z_t=z_t,
            v_tan_t=v_tan_t,
            dir_sign=dir_sign,
            yaw_offset_t=yaw_offset_t,
            w_err_seq=w_err_seq,
            w_u_seq=w_u_seq,
        )
        last_alpha = 1.0
        stop_reason = "max_iters"
        converged = False
        n_iters = 0

        for it in range(self.ilqr_iters):
            n_iters = it + 1
            x_traj = self._rollout(x0, u, flow_seq, flow_w_fn)  # (B, T+1, nx)
            cost_before = self._trajectory_cost(
                x_traj,
                u,
                center_w=center_w,
                radius_t=radius_t,
                z_t=z_t,
                v_tan_t=v_tan_t,
                dir_sign=dir_sign,
                yaw_offset_t=yaw_offset_t,
                w_err_seq=w_err_seq,
                w_u_seq=w_u_seq,
            )
            if x_meas_seq is None:
                x_meas_traj = x_traj
            else:
                x_meas_traj = x_meas_seq.to(device=device, dtype=dtype)
                if x_meas_traj.shape != x_traj.shape:
                    raise ValueError(f"x_meas_seq must have shape {tuple(x_traj.shape)}. got {tuple(x_meas_traj.shape)}")

            # dynamics jacobians (flatten (B,T) for fewer kernel launches)
            x_flat = x_traj[:, :-1, :].reshape(B * T, self.nx)
            u_flat = u.reshape(B * T, nu)
            if flow_w_fn is not None:
                flow_list = []
                for tt in range(T):
                    flow_t = flow_w_fn(x_traj[:, tt, :], tt)
                    flow_t = torch.as_tensor(flow_t, dtype=dtype, device=device)
                    if flow_t.shape == (3,):
                        flow_t = flow_t.view(1, 3).expand(B, 3)
                    if flow_t.shape != (B, 3):
                        raise ValueError(f"flow_w_fn must return ({B},3) or (3,), got {tuple(flow_t.shape)} at t={tt}")
                    flow_list.append(flow_t)
                flow_flat = torch.stack(flow_list, dim=1).reshape(B * T, 3)
            elif flow_seq is None:
                flow_flat = torch.zeros(B * T, 3, device=device, dtype=dtype)
            else:
                flow_flat = flow_seq.reshape(B * T, 3)
            A = jac_x_b(x_flat, u_flat, flow_flat).view(B, T, self.nx, self.nx)
            Bu = jac_u_b(x_flat, u_flat, flow_flat).view(B, T, self.nx, nu)

            # cost derivatives (Gauss-Newton for error terms)
            c_flat = center_w.unsqueeze(1).expand(B, T, center_w.shape[-1]).reshape(B * T, center_w.shape[-1])
            e_traj = e_b(x_flat, c_flat).view(B, T, self.ne)
            Je_traj = jac_e_b(x_flat, c_flat).view(B, T, self.ne, self.nx)

            # terminal
            e_T = e_b(x_traj[:, -1, :], center_w)  # (B, ne)
            Je_T = jac_e_b(x_traj[:, -1, :], center_w)  # (B, ne, nx)

            if w_err_seq is None:
                w_err_b = w_err.view(1, 1, self.ne).expand(B, T, self.ne)
                w_err_T = w_err
            else:
                w_err_b = w_err_seq
                w_err_T = w_err_seq[:, -1, :]  # (B, ne)

            if w_u_seq is None:
                w_u_b = w_u_force.view(1, 1, nu).expand(B, T, nu)
            else:
                w_u_b = w_u_seq

            # stage derivatives
            l_x = torch.einsum("btex,bte->btx", Je_traj, w_err_b * e_traj)
            l_xx = torch.einsum("btex,bte,btey->btxy", Je_traj, w_err_b, Je_traj)
            l_u = w_u_b * u
            l_uu = torch.diag_embed(w_u_b)  # (B, T, nu, nu)
            l_ux = torch.zeros(B, T, nu, self.nx, device=device, dtype=dtype)

            if gamma_d_b is not None and x_meas_seq is not None:
                dt = float(self.dt)
                for t in range(T):
                    x_next_pred = x_traj[:, t + 1, :]
                    x_next_meas = x_meas_traj[:, t + 1, :]
                    d_hat = (x_next_pred - x_next_meas)[:, :3] / dt
                    weighted_d = gamma_d_b.view(B, 1) * d_hat

                    Jx_pos = A[:, t, :3, :] / dt
                    Ju_pos = Bu[:, t, :3, :] / dt

                    l_x[:, t, :] = l_x[:, t, :] + torch.einsum("bix,bi->bx", Jx_pos, weighted_d)
                    l_u[:, t, :] = l_u[:, t, :] + torch.einsum("biu,bi->bu", Ju_pos, weighted_d)
                    l_xx[:, t, :, :] = l_xx[:, t, :, :] + torch.einsum("bix,biy->bxy", Jx_pos, Jx_pos) * gamma_d_b.view(B, 1, 1)
                    l_uu[:, t, :, :] = l_uu[:, t, :, :] + torch.einsum("biu,biv->buv", Ju_pos, Ju_pos) * gamma_d_b.view(B, 1, 1)
                    l_ux[:, t, :, :] = torch.einsum("biu,bix->bux", Ju_pos, Jx_pos) * gamma_d_b.view(B, 1, 1)

            # terminal derivatives
            if w_err_T.ndim == 1:
                V_x = self.terminal_weight_mult * torch.einsum("bex,be->bx", Je_T, w_err_T * e_T)
                V_xx = self.terminal_weight_mult * torch.einsum("bex,e,bey->bxy", Je_T, w_err_T, Je_T)
            else:
                V_x = self.terminal_weight_mult * torch.einsum("bex,be->bx", Je_T, w_err_T * e_T)
                V_xx = self.terminal_weight_mult * torch.einsum("bex,be,bey->bxy", Je_T, w_err_T, Je_T)

            K = []
            k = []
            for t in reversed(range(T)):
                At = A[:, t, :, :]
                Bt = Bu[:, t, :, :]

                Q_x = l_x[:, t, :] + torch.einsum("bij,bj->bi", At.transpose(-1, -2), V_x)
                Q_u = l_u[:, t, :] + torch.einsum("bik,bk->bi", Bt.transpose(-1, -2), V_x)

                Q_xx = l_xx[:, t, :, :] + At.transpose(-1, -2) @ V_xx @ At
                Q_ux = l_ux[:, t, :, :] + Bt.transpose(-1, -2) @ V_xx @ At
                Q_uu = l_uu[:, t, :, :] + Bt.transpose(-1, -2) @ V_xx @ Bt

                Q_uu_reg = Q_uu + self.ilqr_reg * Iu

                Kt = torch.linalg.solve(Q_uu_reg, -Q_ux)
                kt = torch.linalg.solve(Q_uu_reg, -Q_u.unsqueeze(-1)).squeeze(-1)

                V_x = Q_x + torch.einsum("bij,bj->bi", Kt.transpose(-1, -2), Q_u) + torch.einsum(
                    "bij,bj->bi", Q_ux.transpose(-1, -2), kt
                ) + torch.einsum("bij,bjk,bk->bi", Kt.transpose(-1, -2), Q_uu, kt)

                V_xx = (
                    Q_xx
                    + Kt.transpose(-1, -2) @ Q_uu @ Kt
                    + Kt.transpose(-1, -2) @ Q_ux
                    + Q_ux.transpose(-1, -2) @ Kt
                )

                K.append(Kt)
                k.append(kt)

            K = torch.stack(list(reversed(K)), dim=1)  # (B, T, nu, nx)
            k = torch.stack(list(reversed(k)), dim=1)  # (B, T, nu)

            # forward update (alpha=1): keep controller behavior aligned with the original implementation.
            # We still compute cost reduction for diagnostics, but we do not early-stop on no-descent.
            x = x0
            new_u = []
            for t in range(T):
                dx = x - x_traj[:, t, :]
                ut = u[:, t, :] + k[:, t, :] + torch.einsum("bij,bj->bi", K[:, t, :, :], dx)
                ut = ut.clamp(-1.0, 1.0)
                new_u.append(ut)
                if flow_w_fn is not None:
                    flow_wt = flow_w_fn(x, t)
                else:
                    flow_wt = None if flow_seq is None else flow_seq[:, t, :]
                x = self._dynamics(x, ut, flow_wt)
            u = torch.stack(new_u, dim=1)
            last_alpha = 1.0

            x_after = self._rollout(x0, u, flow_seq, flow_w_fn)
            cost_after = self._trajectory_cost(
                x_after,
                u,
                center_w=center_w,
                radius_t=radius_t,
                z_t=z_t,
                v_tan_t=v_tan_t,
                dir_sign=dir_sign,
                yaw_offset_t=yaw_offset_t,
                w_err_seq=w_err_seq,
                w_u_seq=w_u_seq,
            )
            rel_red = ((cost_before.mean() - cost_after.mean()) / cost_before.mean().abs().clamp_min(1e-9)).item()
            if rel_red < 1e-4:
                converged = True
                stop_reason = "small_reduction"
                break

        final_x = self._rollout(x0, u, flow_seq, flow_w_fn)
        final_cost = self._trajectory_cost(
            final_x,
            u,
            center_w=center_w,
            radius_t=radius_t,
            z_t=z_t,
            v_tan_t=v_tan_t,
            dir_sign=dir_sign,
            yaw_offset_t=yaw_offset_t,
            w_err_seq=w_err_seq,
            w_u_seq=w_u_seq,
        )
        if not torch.isfinite(final_cost).all():
            stop_reason = "nonfinite_cost"
            converged = False
        elif not converged and stop_reason == "max_iters":
            converged = bool((final_cost.mean() < initial_cost.mean()).item())
            if not converged:
                stop_reason = "max_iters_no_improve"

        stats = {
            "n_iters": int(n_iters),
            "converged": bool(converged),
            "stop_reason": str(stop_reason),
            "final_cost": float(final_cost.mean().detach().item()),
            "cost_reduction": float((initial_cost.mean() - final_cost.mean()).detach().item()),
            "alpha": float(last_alpha),
            "lambda": float(self.ilqr_reg),
        }
        return u, stats

    def compute(
        self,
        root_state: torch.Tensor,
        *,
        center_w: torch.Tensor,
        radius: float,
        z: float,
        v_tan: float,
        direction: float = 1.0,
        yaw_offset: float = 0.0,
        w_err_seq: torch.Tensor | None = None,
        w_u_seq: torch.Tensor | None = None,
        gamma_d: torch.Tensor | None = None,
        x_meas_seq: torch.Tensor | None = None,
        skip_gamma_d_without_x_meas_seq: bool = True,
        flow_w: torch.Tensor | None = None,
        flow_w_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x0 = root_state.reshape(-1, root_state.shape[-1])
        c = center_w.reshape(-1, center_w.shape[-1])
        B = int(x0.shape[0])
        if c.shape[0] == 1 and B > 1:
            c = c.expand(B, 3)
        if B != self.batch_size:
            self.batch_size = B
            self._u_warm = None

        dtype = self.mpc_dtype
        device = x0.device
        x0d = x0.to(dtype=dtype)
        cd = c.to(dtype=dtype)

        if self._u_warm is None or self._u_warm.shape != (B, self.horizon, self.nu):
            u_init = torch.zeros(B, self.horizon, self.nu, dtype=dtype, device=device)
        else:
            u_init = self._u_warm.to(device=device, dtype=dtype)


        flow_w_seq = None
        if flow_w is not None:
            flow_w_now = torch.as_tensor(flow_w, dtype=dtype, device=device)
            if flow_w_now.ndim == 1 and flow_w_now.numel() == 3:
                flow_w_now = flow_w_now.view(1, 3).expand(B, 3)
            elif flow_w_now.ndim == 2 and flow_w_now.shape == (B, 3):
                pass
            else:
                raise ValueError(f"flow_w must have shape (3,) or ({B},3). got {tuple(flow_w_now.shape)}")
            flow_w_seq = flow_w_now.view(B, 1, 3).expand(B, self.horizon, 3)

        gamma_d_eff = gamma_d
        if skip_gamma_d_without_x_meas_seq and x_meas_seq is None:
            gamma_d_eff = None

        u_traj, solve_stats = self._ilqr(
            x0d,
            center_w=cd,
            radius=radius,
            z=z,
            v_tan=v_tan,
            direction=direction,
            yaw_offset=yaw_offset,
            u_init=u_init,
            w_err_seq=w_err_seq,
            w_u_seq=w_u_seq,
            gamma_d=gamma_d_eff,
            x_meas_seq=x_meas_seq,
            flow_w_seq=flow_w_seq,
            flow_w_fn=flow_w_fn,
        )
        self._u_warm = torch.cat([u_traj[:, 1:, :], torch.zeros_like(u_traj[:, :1, :])], dim=1).detach()

        sat_ratio = float((u_traj.abs() >= 0.99).float().mean().detach().item())
        q0 = normalize(x0d[:, 3:7])
        roll0, pitch0 = _quat_to_rp(q0)
        roll_abs = float(roll0.abs().mean().detach().item())
        pitch_abs = float(pitch0.abs().mean().detach().item())
        v_b0 = x0d[:, 7:10]
        if flow_w is None:
            flow_w0 = torch.zeros_like(v_b0)
        else:
            flow_w0 = torch.as_tensor(flow_w, dtype=dtype, device=device)
            if flow_w0.shape == (3,):
                flow_w0 = flow_w0.view(1, 3).expand(B, 3)
        R0 = quaternion_to_rotation_matrix(q0)
        flow_b0 = torch.einsum("...ji,...j->...i", R0, flow_w0)
        v_rel_norm = float((v_b0 - flow_b0).norm(dim=-1).mean().detach().item())
        energy_mean = float((u_traj.abs() ** 3).sum(dim=-1).mean().detach().item())
        if self._prev_energy_mean is None:
            energy_spike = 0.0
        else:
            energy_spike = max(0.0, energy_mean - float(self._prev_energy_mean))
        self._prev_energy_mean = energy_mean
        solve_stats.update({
            "sat_ratio": sat_ratio,
            "roll_abs": roll_abs,
            "pitch_abs": pitch_abs,
            "v_rel_norm": v_rel_norm,
            "energy_spike": float(energy_spike),
        })
        self._accumulate_diagnostics(solve_stats)

        u0 = u_traj[:, 0, :].to(dtype=root_state.dtype)
        return torch.clamp(u0, -1.0, 1.0)

    def process_rl_actions(self, actions) -> torch.Tensor:
        return actions
