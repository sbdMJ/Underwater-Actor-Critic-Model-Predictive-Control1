"""
Cylinder-aware NMPC controller (acados).

Unlike waypoint tracking, this controller injects cylinder parameters directly
into the OCP as parameters and optimizes thruster forces to:
  - keep radial distance to cylinder center near a desired radius
  - keep a desired depth (z)
  - keep facing the cylinder center (heading alignment)
  - maintain a desired tangential speed around the cylinder (one-lap over period)
  - keep roll/pitch near 0 (optional weights)

Dependencies:
  - casadi
  - acados_template
"""

from __future__ import annotations

import numpy as np
import torch

from .controller import ControllerBase
from .mpc_controller import quat_to_rot_mat, _quat_to_rpy, _cross, omega_mat, get_thruster_allocation_matrix

try:
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    _ACADOS_AVAILABLE = True
    _ACADOS_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover
    ca = None
    AcadosModel = AcadosOcp = AcadosOcpSolver = None
    _ACADOS_AVAILABLE = False
    _ACADOS_IMPORT_ERROR = e


def setup_acados_ocp_cylinder_orbit(uav_params, dt=0.05, N=20):
    if not _ACADOS_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "CylinderOrbitMPCController를 사용하려면 casadi와 acados_template이 필요합니다."
        ) from _ACADOS_IMPORT_ERROR

    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = "bluerov_cylinder_orbit_mpc"

    nx = 13  # [pos(3), quat(4), lin_vel_b(3), ang_vel_b(3)]

    B_np = uav_params.get("thruster_allocation", None)
    if B_np is None:
        B_np = get_thruster_allocation_matrix()
    B_np = np.asarray(B_np, dtype=np.float64)
    if B_np.ndim != 2 or B_np.shape[0] != 6:
        raise ValueError(f"thruster_allocation must have shape (6, nu). got {B_np.shape}")
    nu = int(B_np.shape[1])

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)  # thruster forces (N)
    x_dot = ca.SX.sym("x_dot", nx)

    # Parameters:
    #   c_x,c_y,c_z, r_des, z_des, v_tan_des, dir_sign, yaw_offset
    np_p = 8
    p_par = ca.SX.sym("p", np_p)
    model.p = p_par
    cx, cy, cz, r_des, z_des, v_tan_des, dir_sign, yaw_offset = (
        p_par[0],
        p_par[1],
        p_par[2],
        p_par[3],
        p_par[4],
        p_par[5],
        p_par[6],
        p_par[7],
    )

    pos = x[0:3]
    quat = x[3:7]
    v_b = x[7:10]
    w_b = x[10:13]

    # Thruster allocation: tau = B * u (body wrench)
    B = ca.DM(B_np)
    tau = B @ u

    # Rigid body / hydrodynamics (match setup_acados_ocp in mpc_controller.py)
    mass = float(uav_params["mass"])
    inertia = uav_params["inertia"]
    I_mat = np.diag([float(inertia["xx"]), float(inertia["yy"]), float(inertia["zz"])])
    M_rb = ca.DM(np.block([[mass * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), I_mat]]))

    hydro_params = uav_params["hydro_coef"]
    A = ca.DM(np.diag(hydro_params["added_mass"]))
    D_lin = ca.DM(np.diag(hydro_params["linear_damping"]))
    D_quad = ca.DM(np.diag(hydro_params["quadratic_damping"]))

    nu_total = x[7:13]  # body-frame twist (6,)
    ab = A @ nu_total
    ab_lin, ab_ang = ab[0:3], ab[3:6]
    nu_lin, nu_ang = nu_total[0:3], nu_total[3:6]
    coriolis = ca.vertcat(-_cross(ab_lin, nu_ang), -(_cross(ab_lin, nu_lin) + _cross(ab_ang, nu_ang)))
    damping = (D_lin + D_quad @ ca.diag(ca.fabs(nu_total))) @ nu_total

    rho = float(uav_params.get("rho", 997.0))
    volume = float(uav_params.get("volume", 0.0))
    buoyancy_force = rho * 9.81 * volume
    com_offset = float(uav_params.get("coBM", 0.0))

    R = quat_to_rot_mat(quat)  # body -> world
    f_g_body = R.T @ ca.vertcat(0.0, 0.0, -mass * 9.81)
    f_b_body = R.T @ ca.vertcat(0.0, 0.0, buoyancy_force)
    r_cb = ca.vertcat(0.0, 0.0, -com_offset)
    tau_b = ca.vertcat(f_b_body, _cross(r_cb, f_b_body))
    tau_g = ca.vertcat(f_g_body, ca.vertcat(0.0, 0.0, 0.0))
    hydro_wrench = -coriolis -damping + tau_b + tau_g

    # Dynamics
    p_dot = R @ v_b
    q_dot = 0.5 * omega_mat(w_b) @ quat
    nu_dot = ca.inv(M_rb + A) @ (tau + hydro_wrench)
    v_dot, w_dot = nu_dot[0:3], nu_dot[3:6]
    model.f_expl_expr = ca.vertcat(p_dot, q_dot, v_dot, w_dot)
    model.x, model.u, model.xdot = x, u, x_dot

    # --- Cylinder orbit errors (world frame, derived from state + parameters) ---
    rel_xy = ca.vertcat(cx - pos[0], cy - pos[1])
    dist_xy = ca.sqrt(rel_xy[0] * rel_xy[0] + rel_xy[1] * rel_xy[1] + 1e-6)
    h_des_xy = rel_xy / dist_xy  # unit vector from vehicle to center (XY)

    # Apply yaw_offset to desired facing direction (accounts for model forward-axis mismatch).
    co = ca.cos(yaw_offset)
    so = ca.sin(yaw_offset)
    h_tgt_xy = ca.vertcat(co * h_des_xy[0] - so * h_des_xy[1], so * h_des_xy[0] + co * h_des_xy[1])

    # Actual heading in world is body x-axis: R @ [1,0,0] => first column of R
    h_act = ca.vertcat(R[0, 0], R[1, 0], R[2, 0])
    h_act_xy = ca.vertcat(h_act[0], h_act[1])
    h_act_xy = h_act_xy / ca.sqrt(h_act_xy[0] * h_act_xy[0] + h_act_xy[1] * h_act_xy[1] + 1e-6)

    heading_err = h_act_xy - h_tgt_xy  # (2,)

    radial_err = dist_xy - r_des
    z_err = pos[2] - z_des

    # Tangent direction (CCW if dir_sign>0): perpendicular to h_des_xy
    t_des_xy = ca.vertcat(-h_des_xy[1], h_des_xy[0]) * dir_sign
    v_w = p_dot  # world linear velocity
    v_xy = ca.vertcat(v_w[0], v_w[1])
    v_tan = t_des_xy.T @ v_xy
    v_rad = h_des_xy.T @ v_xy
    tan_err = v_tan - v_tan_des

    roll, pitch, _yaw = _quat_to_rpy(quat)
    w_x, w_y = w_b[0], w_b[1]

    # --- Cost ---
    # We use a nonlinear least-squares cost with y = [errors..., u]
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    y_err = ca.vertcat(
        radial_err,
        z_err,
        tan_err,
        v_rad,
        heading_err[0],
        heading_err[1],
        roll,
        pitch,
        w_x,
        w_y,
    )
    model.cost_y_expr = ca.vertcat(y_err, u)
    model.cost_y_expr_e = y_err

    # weights
    q_radial = float(uav_params.get("mpc_q_radial", 50.0))
    q_z = float(uav_params.get("mpc_q_z", 30.0))
    q_tan = float(uav_params.get("mpc_q_tan", 10.0))
    q_radial_speed = float(uav_params.get("mpc_q_radial_speed", 5.0))
    q_heading = float(uav_params.get("mpc_q_heading", 30.0))
    q_roll = float(uav_params.get("mpc_q_roll", 30.0))
    q_pitch = float(uav_params.get("mpc_q_pitch", 10.0))
    q_wxy = float(uav_params.get("mpc_q_wxy", 0.5))
    r_u = float(uav_params.get("mpc_r_u", 0.01))

    Q = np.diag(
        [
            q_radial,
            q_z,
            q_tan,
            q_radial_speed,
            q_heading,
            q_heading,
            q_roll,
            q_pitch,
            q_wxy,
            q_wxy,
        ]
    )
    Rw = np.diag([r_u] * nu)

    ocp.cost.W = np.block([[Q, np.zeros((Q.shape[0], nu))], [np.zeros((nu, Q.shape[0])), Rw]])
    ocp.cost.W_e = Q
    ocp.cost.yref = np.zeros(Q.shape[0] + nu)
    ocp.cost.yref_e = np.zeros(Q.shape[0])

    # --- Constraints ---
    max_thruster_force = float(uav_params.get("mpc_max_thruster_force", 40.0))
    ocp.constraints.lbu = np.array([-max_thruster_force] * nu)
    ocp.constraints.ubu = np.array([max_thruster_force] * nu)
    ocp.constraints.idxbu = np.arange(nu)

    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0 = np.zeros(nx)
    ocp.constraints.ubx_0 = np.zeros(nx)

    # --- Solver settings ---
    ocp.model = model
    ocp.dims.N = N
    ocp.dims.np = np_p
    ocp.parameter_values = np.zeros(np_p, dtype=np.float64)
    ocp.solver_options.tf = N * dt
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"

    return AcadosOcpSolver(ocp, json_file=f"{model.name}.json")


class CylinderOrbitMPCController(ControllerBase):
    def __init__(self, uav_params, dt=0.05, N=20):
        super().__init__()
        self.solver = setup_acados_ocp_cylinder_orbit(uav_params, dt, N)
        self.nx = self.solver.acados_ocp.dims.nx
        self.nu = self.solver.acados_ocp.dims.nu
        self.np = int(getattr(self.solver.acados_ocp.dims, "np", 0))
        self.N = N

        self.max_force_per_thruster = float(uav_params.get("mpc_max_thruster_force", 40.0))
        self._yref = np.zeros(int(self.solver.acados_ocp.dims.ny), dtype=np.float64)
        self._yref_e = np.zeros(int(self.solver.acados_ocp.dims.ny_e), dtype=np.float64)

        self._warn_count = 0
        self.requires_grad_(False)

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
    ) -> torch.Tensor:
        device = root_state.device
        x0_batch = root_state.reshape(-1, root_state.shape[-1])
        batch = int(x0_batch.shape[0])

        c_batch = center_w.reshape(-1, center_w.shape[-1])
        if c_batch.shape[0] == 1 and batch > 1:
            c_batch = c_batch.expand(batch, 3)

        commands = torch.zeros(batch, self.nu, device=device, dtype=root_state.dtype)

        for i in range(batch):
            x0 = x0_batch[i].detach().cpu().numpy().astype(np.float64, copy=False)
            c = c_batch[i].detach().cpu().numpy().astype(np.float64, copy=False)
            p_vec = np.array(
                [c[0], c[1], c[2], float(radius), float(z), float(v_tan), float(np.sign(direction) or 1.0), float(yaw_offset)],
                dtype=np.float64,
            )
            if self.np != 8:
                raise RuntimeError(f"Unexpected np={self.np} (expected 8)")

            # initial state
            self.solver.set(0, "lbx", x0)
            self.solver.set(0, "ubx", x0)

            # parameters + refs
            for k in range(self.N):
                self.solver.set(k, "p", p_vec)
                self.solver.set(k, "yref", self._yref)
            self.solver.set(self.N, "p", p_vec)
            self.solver.set(self.N, "yref", self._yref_e)

            status = self.solver.solve()
            if status != 0:
                if self._warn_count < 10:
                    print(f"[CylinderOrbitMPCController] acados solve 실패: status={status}")
                    self._warn_count += 1
                u_opt = np.zeros(self.nu, dtype=np.float64)
            else:
                u_opt = self.solver.get(0, "u")

            u_cmd = np.clip(u_opt / self.max_force_per_thruster, -1.0, 1.0)
            commands[i].copy_(torch.as_tensor(u_cmd, device=device, dtype=commands.dtype))

        return commands
