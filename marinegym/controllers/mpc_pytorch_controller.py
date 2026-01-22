"""
Differentiable MPC using `locuslab/mpc.pytorch`.

State / input definition matches `MPCController` / HoverMPC:
  - x (13): [pos(3), quat(wxyz,4), vel_b(6)]
  - u (nu): thruster command in [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from torch import nn

try:
    from mpc.mpc import MPC, QuadCost, GradMethods  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "MPCPyTorchController를 사용하려면 `locuslab/mpc.pytorch`가 필요합니다. "
        "Isaac Sim Python에서 `python -m pip show mpc`로 설치 여부를 확인하세요."
    ) from e

from .controller import ControllerBase
from marinegym.utils.torch import quaternion_to_rotation_matrix, normalize


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cross(a, b, dim=-1)


def _omega_mat(w: torch.Tensor) -> torch.Tensor:
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


class _UnderwaterVehicleDx(nn.Module):
    def __init__(
        self,
        *,
        dt: float,
        max_thruster_force: float,
        M_inv: torch.Tensor,
        A_added: torch.Tensor,
        d_lin: torch.Tensor,
        d_quad: torch.Tensor,
        B_alloc: torch.Tensor,
        f_g_world: torch.Tensor,
        f_b_world: torch.Tensor,
        r_cb: torch.Tensor,
    ):
        super().__init__()
        self.dt = float(dt)
        self.max_thruster_force = float(max_thruster_force)
        self.register_buffer("M_inv", M_inv)
        self.register_buffer("A_added", A_added)
        self.register_buffer("d_lin", d_lin)
        self.register_buffer("d_quad", d_quad)
        self.register_buffer("B_alloc", B_alloc)
        self.register_buffer("f_g_world", f_g_world)
        self.register_buffer("f_b_world", f_b_world)
        self.register_buffer("r_cb", r_cb)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.M_inv.dtype)
        u = u.to(dtype=self.M_inv.dtype)

        p = x[..., 0:3]
        q = normalize(x[..., 3:7])
        v = x[..., 7:10]
        w = x[..., 10:13]

        u_force = u * float(self.max_thruster_force)
        tau = torch.einsum("ij,...j->...i", self.B_alloc, u_force)  # (..., 6)

        nu = torch.cat([v, w], dim=-1)  # (..., 6)
        ab = torch.einsum("ij,...j->...i", self.A_added, nu)
        ab_lin, ab_ang = ab[..., 0:3], ab[..., 3:6]
        nu_lin, nu_ang = nu[..., 0:3], nu[..., 3:6]
        coriolis = torch.cat(
            [-_cross(ab_lin, nu_ang), -(_cross(ab_lin, nu_lin) + _cross(ab_ang, nu_ang))],
            dim=-1,
        )

        damping = (self.d_lin + self.d_quad * torch.abs(nu)) * nu

        R = quaternion_to_rotation_matrix(q)
        f_g_body = torch.einsum("...ji,...j->...i", R, self.f_g_world.to(device=R.device, dtype=R.dtype))
        f_b_body = torch.einsum("...ji,...j->...i", R, self.f_b_world.to(device=R.device, dtype=R.dtype))
        r_cb = self.r_cb.to(device=R.device, dtype=R.dtype)
        tau_b = torch.cat([f_b_body, _cross(r_cb.expand_as(f_b_body), f_b_body)], dim=-1)
        tau_g = torch.cat([f_g_body, torch.zeros_like(f_g_body)], dim=-1)

        hydro_wrench = -coriolis - damping + tau_b + tau_g

        p_dot = torch.einsum("...ij,...j->...i", R, v)
        q_dot = 0.5 * torch.einsum("...ij,...j->...i", _omega_mat(w), q)

        nu_dot = torch.einsum("ij,...j->...i", self.M_inv, tau + hydro_wrench)
        v_dot, w_dot = nu_dot[..., 0:3], nu_dot[..., 3:6]

        dt = self.dt
        p_next = p + dt * p_dot
        q_next = normalize(q + dt * q_dot)
        v_next = v + dt * v_dot
        w_next = w + dt * w_dot
        return torch.cat([p_next, q_next, v_next, w_next], dim=-1)


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


class MPCPyTorchController(ControllerBase):
    def __init__(
        self,
        *,
        uav_params: dict,
        dt: float = 0.05,
        horizon: int = 15,
        batch_size: int = 1,
        ilqr_iters: int = 6,
        ilqr_reg: float = 1e-3,
        terminal_weight_mult: float = 10.0,
        backprop: bool = True,
        eps: float = 1e-3,
        exit_unconverged: bool = False,
        detach_unconverged: bool | None = None,
        max_thruster_force: float = 40.0,
    ):
        super().__init__()

        B_np = np.asarray(uav_params["thruster_allocation"], dtype=np.float64)
        if B_np.ndim != 2 or B_np.shape[0] != 6:
            raise ValueError(f"thruster_allocation must have shape (6, nu). got {B_np.shape}")
        self.nu = int(B_np.shape[1])
        self.nx = 13
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.batch_size = int(batch_size)
        self.max_thruster_force = float(max_thruster_force)
        self.terminal_weight_mult = float(terminal_weight_mult)

        hydro = uav_params["hydro_coef"]
        params = _UAVParams(
            mass=float(uav_params["mass"]),
            inertia_xx=float(uav_params["inertia"]["xx"]),
            inertia_yy=float(uav_params["inertia"]["yy"]),
            inertia_zz=float(uav_params["inertia"]["zz"]),
            added_mass=torch.as_tensor(hydro["added_mass"], dtype=torch.float32),
            linear_damping=torch.as_tensor(hydro["linear_damping"], dtype=torch.float32),
            quadratic_damping=torch.as_tensor(hydro["quadratic_damping"], dtype=torch.float32),
            thruster_allocation=torch.as_tensor(B_np, dtype=torch.float32),
            volume=float(uav_params.get("volume", 0.0)),
            coBM=float(uav_params.get("coBM", 0.0)),
            rho=float(uav_params.get("rho", 997.0)),
            g=float(uav_params.get("g", 9.81)),
            max_thruster_force=self.max_thruster_force,
        )

        mass = torch.as_tensor(params.mass, dtype=torch.float32)
        I = torch.diag(torch.as_tensor([params.inertia_xx, params.inertia_yy, params.inertia_zz], dtype=torch.float32))
        M_rb = torch.block_diag(mass * torch.eye(3, dtype=torch.float32), I)
        A_added = torch.diag(params.added_mass)

        self.register_buffer("M_inv", torch.linalg.inv(M_rb + A_added))
        self.register_buffer("A_added", A_added)
        self.register_buffer("d_lin", params.linear_damping)
        self.register_buffer("d_quad", params.quadratic_damping)
        self.register_buffer("B_alloc", params.thruster_allocation)
        self.register_buffer("_f_g_world", torch.tensor([0.0, 0.0, -params.mass * params.g], dtype=torch.float32))
        self.register_buffer("_f_b_world", torch.tensor([0.0, 0.0, params.rho * params.g * params.volume], dtype=torch.float32))
        self.register_buffer("_r_cb", torch.tensor([0.0, 0.0, -params.coBM], dtype=torch.float32))

        q_pos = float(uav_params.get("mpc_q_pos", 50.0))
        q_quat = float(uav_params.get("mpc_q_quat", 5.0))
        q_vel = float(uav_params.get("mpc_q_vel", 2.0))
        q_omega = float(uav_params.get("mpc_q_omega", 0.5))
        r_u = float(uav_params.get("mpc_r_u", 0.02))

        w_x = torch.tensor([q_pos] * 3 + [q_quat] * 4 + [q_vel] * 3 + [q_omega] * 3, dtype=torch.float32)
        w_u = torch.tensor([r_u] * self.nu, dtype=torch.float32)
        self.register_buffer("w_x", w_x)
        self.register_buffer("w_u", w_u)
        self.terminal_weight_mult = float(terminal_weight_mult)

        n_sc = self.nx + self.nu
        diag = torch.cat([w_x, w_u], dim=-1)
        Q_stage = torch.diag(diag)  # (n_sc, n_sc)
        self.register_buffer("Q_stage", Q_stage)

        if detach_unconverged is None:
            detach_unconverged = bool(backprop)

        self.mpc = MPC(
            self.nx,
            self.nu,
            self.horizon,
            u_lower=-1.0,
            u_upper=1.0,
            u_init=None,  # warm-start는 매 step에서 self.mpc.u_init으로 주입
            lqr_iter=int(ilqr_iters),
            grad_method=GradMethods.AUTO_DIFF,
            verbose=0,
            eps=float(eps),
            n_batch=self.batch_size,
            exit_unconverged=bool(exit_unconverged),
            detach_unconverged=bool(detach_unconverged),
            backprop=bool(backprop),
            best_cost_eps=1e-4,
        )

        self._dx = _UnderwaterVehicleDx(
            dt=self.dt,
            max_thruster_force=self.max_thruster_force,
            M_inv=self.M_inv,
            A_added=self.A_added,
            d_lin=self.d_lin,
            d_quad=self.d_quad,
            B_alloc=self.B_alloc,
            f_g_world=self._f_g_world,
            f_b_world=self._f_b_world,
            r_cb=self._r_cb,
        )

        self._u_warm: torch.Tensor | None = None
        self.requires_grad_(False)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self._dx(x, u)

    def compute(
        self,
        root_state: torch.Tensor,
        target_pos: torch.Tensor,
        *,
        target_quat: torch.Tensor | None = None,
        target_vel_b: torch.Tensor | None = None,
        target_omega_b: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x0 = root_state.reshape(-1, root_state.shape[-1])
        tp = target_pos.reshape(-1, target_pos.shape[-1])
        B = x0.shape[0]
        if B != self.batch_size:
            raise ValueError(f"batch_size mismatch: controller={self.batch_size} input={B}")

        device = x0.device
        dtype = self.M_inv.dtype

        x0d = x0.to(device=device, dtype=dtype)
        tpd = tp.to(device=device, dtype=dtype)

        x_ref = x0d.clone()
        x_ref[..., 0:3] = tpd[..., 0:3]
        x_ref[..., 7:13] = 0.0

        if target_vel_b is not None:
            tv = target_vel_b.reshape(-1, target_vel_b.shape[-1]).to(device=device, dtype=dtype)
            x_ref[..., 7:10] = tv[..., 0:3]
        if target_omega_b is not None:
            tw = target_omega_b.reshape(-1, target_omega_b.shape[-1]).to(device=device, dtype=dtype)
            x_ref[..., 10:13] = tw[..., 0:3]

        if target_quat is not None:
            tq = target_quat.reshape(-1, target_quat.shape[-1]).to(device=device, dtype=dtype)
            tq = normalize(tq)
            dot = (tq * x0d[..., 3:7]).sum(dim=-1, keepdim=True)
            tq = torch.where(dot < 0, -tq, tq)
            x_ref[..., 3:7] = tq

        # locuslab/mpc.pytorch cost format:
        #   C: (T, B, n_tau, n_tau), c: (T, B, n_tau), where tau_t = [x_t, u_t]
        w_x = self.w_x.to(device=device, dtype=dtype)
        w_u = self.w_u.to(device=device, dtype=dtype)
        n_tau = self.nx + self.nu
        C = torch.zeros(self.horizon, B, n_tau, n_tau, device=device, dtype=dtype)
        c = torch.zeros(self.horizon, B, n_tau, device=device, dtype=dtype)

        Qx = torch.diag(w_x)
        Ru = torch.diag(w_u)
        C_stage = torch.block_diag(Qx, Ru)
        for t in range(self.horizon):
            Ct = C_stage
            if t == self.horizon - 1 and self.terminal_weight_mult != 1.0:
                Ct = torch.block_diag(self.terminal_weight_mult * Qx, Ru)
            C[t].copy_(Ct.unsqueeze(0).expand(B, -1, -1))
            c[t, :, 0 : self.nx] = -torch.einsum("ij,bj->bi", Ct[0 : self.nx, 0 : self.nx], x_ref)

        cost = QuadCost(C, c)

        if self._u_warm is None or self._u_warm.shape != (self.horizon, B, self.nu):
            u_init = torch.zeros(self.horizon, B, self.nu, device=device, dtype=dtype)
        else:
            u_init = self._u_warm.to(device=device, dtype=dtype)

        self.mpc.u_init = u_init.detach()
        _x_traj, u_traj, _costs = self.mpc(x0d, cost, self._dx)
        # warm-start shift: u_{t+1} <- u_t
        self._u_warm = torch.cat([u_traj[1:], torch.zeros_like(u_traj[:1])], dim=0).detach()

        u0 = u_traj[0].to(dtype=root_state.dtype)  # (B, nu)
        return torch.clamp(u0, -1.0, 1.0)

    def process_rl_actions(self, actions) -> torch.Tensor:
        return actions
