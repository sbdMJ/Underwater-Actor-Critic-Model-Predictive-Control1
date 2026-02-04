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

    def __init__(self, dt: float, params: _UAVParams):
        if not _PYPOSE_AVAILABLE:  # pragma: no cover
            raise ImportError("PyPose가 필요합니다.") from _PYPOSE_IMPORT_ERROR
        super().__init__()
        self.dt = float(dt)
        self.params = params

        mass = torch.as_tensor(params.mass, dtype=torch.float64)
        I = torch.diag(
            torch.as_tensor(
                [params.inertia_xx, params.inertia_yy, params.inertia_zz],
                dtype=torch.float64,
            )
        )
        M_rb = torch.block_diag(mass * torch.eye(3, dtype=torch.float64), I)
        A_added = torch.diag(params.added_mass.to(dtype=torch.float64))
        self.register_buffer("M_inv", torch.linalg.inv(M_rb + A_added))
        self.register_buffer("A_added", A_added)
        self.register_buffer("d_lin", params.linear_damping.to(dtype=torch.float64))
        self.register_buffer("d_quad", params.quadratic_damping.to(dtype=torch.float64))
        self.register_buffer("B_alloc", params.thruster_allocation.to(dtype=torch.float64))

        buoyancy_force = float(params.rho) * float(params.g) * float(params.volume)
        self.buoyancy_force = buoyancy_force
        self.register_buffer(
            "_f_g_world",
            torch.tensor([0.0, 0.0, -params.mass * params.g], dtype=torch.float64),
        )
        self.register_buffer(
            "_f_b_world",
            torch.tensor([0.0, 0.0, buoyancy_force], dtype=torch.float64),
        )
        self.register_buffer(
            "_r_cb",
            torch.tensor([0.0, 0.0, -params.coBM], dtype=torch.float64),
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
        damping = (self.d_lin + self.d_quad * torch.abs(nu)) * nu

        # rotation (body -> world)
        R = quaternion_to_rotation_matrix(q)

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

        hydro = uav_params["hydro_coef"]
        params = _UAVParams(
            mass=float(uav_params["mass"]),
            inertia_xx=float(uav_params["inertia"]["xx"]),
            inertia_yy=float(uav_params["inertia"]["yy"]),
            inertia_zz=float(uav_params["inertia"]["zz"]),
            added_mass=torch.as_tensor(hydro["added_mass"], dtype=torch.float64),
            linear_damping=torch.as_tensor(hydro["linear_damping"], dtype=torch.float64),
            quadratic_damping=torch.as_tensor(hydro["quadratic_damping"], dtype=torch.float64),
            thruster_allocation=torch.as_tensor(B_np, dtype=torch.float64),
            volume=float(uav_params.get("volume", 0.0)),
            coBM=float(uav_params.get("coBM", 0.0)),
            rho=float(uav_params.get("rho", 997.0)),
            g=float(uav_params.get("g", 9.81)),
            max_thruster_force=self.max_thruster_force,
        )
        self.system = _UnderwaterVehicleNLS(self.dt, params)

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
            dtype=torch.float64,
        )
        # acados cost penalizes thruster *forces*; our decision variable is command in [-1,1]
        # => weight(cmd) = weight(force) * (max_thruster_force^2)
        w_u = torch.tensor([float(r_u) * (self.max_thruster_force**2)] * nu, dtype=torch.float64)
        self.register_buffer("_w_err", w_err)
        self.register_buffer("_w_u", w_u)

        self._u_warm: torch.Tensor | None = None
        self.requires_grad_(False)

    def _dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.system.state_transition(x, u, None)

    def _rollout(self, x0: torch.Tensor, u_traj: torch.Tensor) -> torch.Tensor:
        xs = [x0]
        x = x0
        for t in range(u_traj.shape[1]):
            x = self._dynamics(x, u_traj[:, t, :])
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
    ) -> torch.Tensor:
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

        def f_single(x, u):
            x_next = self._dynamics(x.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
            return x_next

        jac_x = jacrev(f_single, argnums=0)
        jac_u = jacrev(f_single, argnums=1)
        jac_x_b = vmap(jac_x, in_dims=(0, 0))
        jac_u_b = vmap(jac_u, in_dims=(0, 0))

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

        for _ in range(self.ilqr_iters):
            x_traj = self._rollout(x0, u)  # (B, T+1, nx)

            # dynamics jacobians
            A_list = []
            B_list = []
            for t in range(T):
                xt = x_traj[:, t, :]
                ut = u[:, t, :]
                A_list.append(jac_x_b(xt, ut))
                B_list.append(jac_u_b(xt, ut))
            A = torch.stack(A_list, dim=1)  # (B, T, nx, nx)
            Bu = torch.stack(B_list, dim=1)  # (B, T, nx, nu)

            # cost derivatives (Gauss-Newton for error terms)
            e_list = []
            Je_list = []
            for t in range(T):
                xt = x_traj[:, t, :]
                et = e_b(xt, center_w)
                Jet = jac_e_b(xt, center_w)  # (B, ne, nx)
                e_list.append(et)
                Je_list.append(Jet)
            e_traj = torch.stack(e_list, dim=1)  # (B, T, ne)
            Je_traj = torch.stack(Je_list, dim=1)  # (B, T, ne, nx)

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
                Q_ux = Bt.transpose(-1, -2) @ V_xx @ At
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

            # forward update (alpha=1)
            x = x0
            new_u = []
            for t in range(T):
                dx = x - x_traj[:, t, :]
                ut = u[:, t, :] + k[:, t, :] + torch.einsum("bij,bj->bi", K[:, t, :, :], dx)
                ut = ut.clamp(-1.0, 1.0)
                new_u.append(ut)
                x = self._dynamics(x, ut)
            u = torch.stack(new_u, dim=1)

        return u

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
    ) -> torch.Tensor:
        x0 = root_state.reshape(-1, root_state.shape[-1])
        c = center_w.reshape(-1, center_w.shape[-1])
        B = int(x0.shape[0])
        if c.shape[0] == 1 and B > 1:
            c = c.expand(B, 3)
        if B != self.batch_size:
            self.batch_size = B
            self._u_warm = None

        dtype = torch.float64
        device = x0.device
        x0d = x0.to(dtype=dtype)
        cd = c.to(dtype=dtype)

        if self._u_warm is None or self._u_warm.shape != (B, self.horizon, self.nu):
            u_init = torch.zeros(B, self.horizon, self.nu, dtype=dtype, device=device)
        else:
            u_init = self._u_warm.to(device=device, dtype=dtype)

        u_traj = self._ilqr(
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
        )
        self._u_warm = torch.cat([u_traj[:, 1:, :], torch.zeros_like(u_traj[:, :1, :])], dim=1).detach()

        u0 = u_traj[:, 0, :].to(dtype=root_state.dtype)
        return torch.clamp(u0, -1.0, 1.0)

    def process_rl_actions(self, actions) -> torch.Tensor:
        return actions
