"""
PyPose 기반 differentiable MPC (iLQR).

PyTorch 연산으로 iLQR(Iterative LQR)을 unroll해 MPC를 풉니다.
(CasADi/acados 없이 end-to-end autograd 가능)

이 컨트롤러는 HoverMPC(acados)와 동일한 상태/입력 정의를 사용합니다.
  - state x: [pos(3), quat(wxyz,4), vel_b(6)] = 13
  - input u: thruster command (nu) in [-1, 1]
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
        # state: (B, 13), input: (B, nu)
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

        # coriolis (match marinegym/controllers/mpc_controller.py)
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


class PyPoseMPCController(ControllerBase):
    def __init__(
        self,
        *,
        uav_params: dict,
        dt: float = 0.05,
        horizon: int = 20,
        batch_size: int = 1,
        ilqr_iters: int = 2,
        ilqr_reg: float = 1e-3,
        terminal_weight_mult: float = 10.0,
        max_thruster_force: float = 40.0,
    ):
        super().__init__()
        if not _PYPOSE_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "PyPoseMPCController를 사용하려면 pypose가 필요합니다."
            ) from _PYPOSE_IMPORT_ERROR

        B_np = np.asarray(uav_params["thruster_allocation"], dtype=np.float64)
        if B_np.ndim != 2 or B_np.shape[0] != 6:
            raise ValueError(f"thruster_allocation must have shape (6, nu). got {B_np.shape}")
        nu = int(B_np.shape[1])
        self.nu = nu
        self.nx = 13
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.batch_size = int(batch_size)
        self.max_thruster_force = float(max_thruster_force)
        self.ilqr_iters = int(ilqr_iters)
        self.ilqr_reg = float(ilqr_reg)
        self.terminal_weight_mult = float(terminal_weight_mult)

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

        q_pos = float(uav_params.get("mpc_q_pos", 50.0))
        q_quat = float(uav_params.get("mpc_q_quat", 5.0))
        q_vel = float(uav_params.get("mpc_q_vel", 2.0))
        q_omega = float(uav_params.get("mpc_q_omega", 0.5))
        r_u = float(uav_params.get("mpc_r_u", 0.01))

        w_state = torch.tensor(
            [q_pos] * 3 + [q_quat] * 4 + [q_vel] * 3 + [q_omega] * 3,
            dtype=torch.float64,
        )
        w_u = torch.tensor([r_u] * nu, dtype=torch.float64)
        w = torch.cat([w_state, w_u], dim=-1)  # (nx+nu,)
        self.register_buffer("_w", w)  # cost diag

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

    def _cost(
        self,
        x_traj: torch.Tensor,
        u_traj: torch.Tensor,
        x_ref: torch.Tensor,
        *,
        w_x: torch.Tensor | None = None,
        w_u: torch.Tensor | None = None,
        w_x_seq: torch.Tensor | None = None,
        w_u_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # stage costs t=0..T-1 + terminal cost at T
        B, Tp1, nx = x_traj.shape
        _B, T, nu = u_traj.shape
        if _B != B or Tp1 != T + 1 or nx != self.nx or nu != self.nu:
            raise ValueError(
                f"shape mismatch: x_traj={tuple(x_traj.shape)} u_traj={tuple(u_traj.shape)} "
                f"(expected x_traj=(B,T+1,{self.nx}) u_traj=(B,T,{self.nu}))"
            )

        if (w_x_seq is None) != (w_u_seq is None):
            raise ValueError("w_x_seq and w_u_seq must be provided together.")

        if w_x_seq is not None and w_u_seq is not None:
            w_x_seq = w_x_seq.to(device=x_traj.device, dtype=x_traj.dtype)
            w_u_seq = w_u_seq.to(device=x_traj.device, dtype=x_traj.dtype)
            if w_x_seq.shape != (B, T, self.nx):
                raise ValueError(f"w_x_seq must have shape ({B},{T},{self.nx}). got {tuple(w_x_seq.shape)}")
            if w_u_seq.shape != (B, T, self.nu):
                raise ValueError(f"w_u_seq must have shape ({B},{T},{self.nu}). got {tuple(w_u_seq.shape)}")
        else:
            if w_x is None or w_u is None:
                w = self._w.to(device=x_traj.device, dtype=x_traj.dtype)
                w_x = w[: self.nx]
                w_u = w[self.nx :]
            w_x = w_x.to(device=x_traj.device, dtype=x_traj.dtype)
            w_u = w_u.to(device=x_traj.device, dtype=x_traj.dtype)
            if w_x.ndim == 1:
                w_x_seq = w_x.view(1, 1, self.nx).expand(B, T, self.nx)
            elif w_x.ndim == 2:
                w_x_seq = w_x.view(B, 1, self.nx).expand(B, T, self.nx)
            else:
                raise ValueError(f"w_x must have ndim 1/2 when w_x_seq not provided. got {w_x.ndim}")
            if w_u.ndim == 1:
                w_u_seq = w_u.view(1, 1, self.nu).expand(B, T, self.nu)
            elif w_u.ndim == 2:
                w_u_seq = w_u.view(B, 1, self.nu).expand(B, T, self.nu)
            else:
                raise ValueError(f"w_u must have ndim 1/2 when w_u_seq not provided. got {w_u.ndim}")

        dx = x_traj[:, :-1, :] - x_ref[:, None, :]  # (B, T, nx)
        du = u_traj  # (B, T, nu)
        stage = 0.5 * (w_x_seq * dx.square()).sum(dim=-1) + 0.5 * (w_u_seq * du.square()).sum(dim=-1)  # (B, T)
        dxT = x_traj[:, -1, :] - x_ref  # (B, nx)
        terminal = 0.5 * self.terminal_weight_mult * (w_x_seq[:, -1, :] * dxT.square()).sum(dim=-1)  # (B,)
        return stage.sum(dim=-1) + terminal  # (B,)

    def _ilqr(
        self,
        x0: torch.Tensor,
        x_ref: torch.Tensor,
        u_init: torch.Tensor,
        w_x: torch.Tensor,
        w_u: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from torch.func import jacrev, vmap  # type: ignore
        except Exception:  # pragma: no cover
            from functorch import jacrev, vmap  # type: ignore

        B, T, nu = u_init.shape
        u = u_init

        def f_single(x, u):
            x_next = self._dynamics(x.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
            return x_next

        jac_x = jacrev(f_single, argnums=0)
        jac_u = jacrev(f_single, argnums=1)
        jac_x_b = vmap(jac_x, in_dims=(0, 0))
        jac_u_b = vmap(jac_u, in_dims=(0, 0))

        w_x = w_x.to(device=x0.device, dtype=x0.dtype)
        w_u = w_u.to(device=x0.device, dtype=x0.dtype)

        # Allow time-invariant (nx,) / (B,nx) or time-varying (B,T,nx).
        if w_x.ndim == 1:
            w_x = w_x.view(1, 1, self.nx).expand(B, T, self.nx)
        elif w_x.ndim == 2:
            w_x = w_x.view(B, 1, self.nx).expand(B, T, self.nx)
        elif w_x.ndim == 3:
            if w_x.shape != (B, T, self.nx):
                raise ValueError(f"w_x must have shape ({B},{T},{self.nx}). got {tuple(w_x.shape)}")
        else:
            raise ValueError(f"w_x must have ndim 1/2/3. got {w_x.ndim}")

        if w_u.ndim == 1:
            w_u = w_u.view(1, 1, nu).expand(B, T, nu)
        elif w_u.ndim == 2:
            w_u = w_u.view(B, 1, nu).expand(B, T, nu)
        elif w_u.ndim == 3:
            if w_u.shape != (B, T, nu):
                raise ValueError(f"w_u must have shape ({B},{T},{nu}). got {tuple(w_u.shape)}")
        else:
            raise ValueError(f"w_u must have ndim 1/2/3. got {w_u.ndim}")

        Iu = torch.eye(nu, device=x0.device, dtype=x0.dtype).unsqueeze(0).expand(B, nu, nu)

        for _ in range(self.ilqr_iters):
            x_traj = self._rollout(x0, u)  # (B, T+1, nx)
            A_list = []
            B_list = []
            for t in range(T):
                xt = x_traj[:, t, :]
                ut = u[:, t, :]
                A_list.append(jac_x_b(xt, ut))
                B_list.append(jac_u_b(xt, ut))
            A = torch.stack(A_list, dim=1)  # (B, T, nx, nx)
            Bu = torch.stack(B_list, dim=1)  # (B, T, nx, nu)

            # terminal derivatives
            w_x_T = w_x[:, -1, :]  # (B, nx)
            V_x = self.terminal_weight_mult * w_x_T * (x_traj[:, -1, :] - x_ref)  # (B, nx)
            V_xx = self.terminal_weight_mult * torch.diag_embed(w_x_T)  # (B, nx, nx)

            K = []
            k = []
            for t in reversed(range(T)):
                xt = x_traj[:, t, :]
                ut = u[:, t, :]

                w_x_t = w_x[:, t, :]  # (B, nx)
                w_u_t = w_u[:, t, :]  # (B, nu)

                l_x = w_x_t * (xt - x_ref)  # (B, nx)
                l_u = w_u_t * ut  # (B, nu)
                l_xx = torch.diag_embed(w_x_t)  # (B, nx, nx)
                l_uu = torch.diag_embed(w_u_t)  # (B, nu, nu)

                At = A[:, t, :, :]
                Bt = Bu[:, t, :, :]
                Q_x = l_x + torch.einsum("bij,bj->bi", At.transpose(-1, -2), V_x)
                Q_u = l_u + torch.einsum("bik,bk->bi", Bt.transpose(-1, -2), V_x)
                Q_xx = l_xx + At.transpose(-1, -2) @ V_xx @ At
                Q_ux = Bt.transpose(-1, -2) @ V_xx @ At
                Q_uu = l_uu + Bt.transpose(-1, -2) @ V_xx @ Bt

                Q_uu_reg = Q_uu + self.ilqr_reg * Iu
                # solve Q_uu * K = -Q_ux, Q_uu * k = -Q_u
                Kt = torch.linalg.solve(Q_uu_reg, -Q_ux)
                kt = torch.linalg.solve(Q_uu_reg, -Q_u.unsqueeze(-1)).squeeze(-1)

                V_x = Q_x + torch.einsum("bij,bj->bi", Kt.transpose(-1, -2), Q_u) + torch.einsum(
                    "bij,bj->bi", Q_ux.transpose(-1, -2), kt
                ) + torch.einsum("bij,bjk,bk->bi", Kt.transpose(-1, -2), Q_uu, kt)
                V_xx = Q_xx + Kt.transpose(-1, -2) @ Q_uu @ Kt + Kt.transpose(-1, -2) @ Q_ux + Q_ux.transpose(-1, -2) @ Kt

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
        target_pos: torch.Tensor,
        *,
        target_quat: torch.Tensor | None = None,
        target_vel_b: torch.Tensor | None = None,
        target_omega_b: torch.Tensor | None = None,
        w_x: torch.Tensor | None = None,
        w_u: torch.Tensor | None = None,
        w_x_seq: torch.Tensor | None = None,
        w_u_seq: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x0 = root_state.reshape(-1, root_state.shape[-1])
        tp = target_pos.reshape(-1, target_pos.shape[-1])
        B = x0.shape[0]
        if B != self.batch_size:
            self.batch_size = B
            self._u_warm = None

        dtype = torch.float64
        device = x0.device
        x0d = x0.to(dtype=dtype)
        tpd = tp.to(dtype=dtype)

        x_ref = x0d.clone()
        x_ref[..., 0:3] = tpd[..., 0:3]
        x_ref[..., 7:13] = 0.0

        if target_vel_b is not None:
            tv = target_vel_b.reshape(-1, target_vel_b.shape[-1]).to(dtype=dtype)
            x_ref[..., 7:10] = tv[..., 0:3]
        if target_omega_b is not None:
            tw = target_omega_b.reshape(-1, target_omega_b.shape[-1]).to(dtype=dtype)
            x_ref[..., 10:13] = tw[..., 0:3]

        if target_quat is not None:
            tq = target_quat.reshape(-1, target_quat.shape[-1]).to(dtype=dtype)
            tq = normalize(tq)
            # choose closest representation (q ~ -q)
            dot = (tq * x0d[..., 3:7]).sum(dim=-1, keepdim=True)
            tq = torch.where(dot < 0, -tq, tq)
            x_ref[..., 3:7] = tq

        if self._u_warm is None or self._u_warm.shape != (B, self.horizon, self.nu):
            u_init = torch.zeros(B, self.horizon, self.nu, dtype=dtype, device=device)
        else:
            u_init = self._u_warm.to(device=device, dtype=dtype)

        if (w_x_seq is None) != (w_u_seq is None):
            raise ValueError("w_x_seq and w_u_seq must be provided together.")

        if w_x_seq is not None and w_u_seq is not None:
            w_x_eff = w_x_seq.to(device=device, dtype=dtype)
            w_u_eff = w_u_seq.to(device=device, dtype=dtype)
        elif w_x is None or w_u is None:
            w = self._w.to(device=device, dtype=dtype)
            w_x_eff = w[: self.nx]
            w_u_eff = w[self.nx :]
        else:
            w_x_eff = w_x.to(device=device, dtype=dtype)
            w_u_eff = w_u.to(device=device, dtype=dtype)
        u_traj = self._ilqr(x0d, x_ref, u_init, w_x_eff, w_u_eff)
        self._u_warm = torch.cat([u_traj[:, 1:, :], torch.zeros_like(u_traj[:, :1, :])], dim=1).detach()

        u0 = u_traj[:, 0, :].to(dtype=root_state.dtype)
        return torch.clamp(u0, -1.0, 1.0)

    def process_rl_actions(self, actions) -> torch.Tensor:
        return actions
