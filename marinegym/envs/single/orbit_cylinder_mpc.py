import torch

import omni.isaac.core.utils.prims as prim_utils
from pxr import Gf, UsdGeom

from marinegym.envs.single.hover_mpc import HoverMPC
from marinegym.utils.torch import quat_axis, euler_to_quaternion


class OrbitCylinderMPC(HoverMPC):
    """
    Cylinder-aware NMPC 태스크 (waypoint tracking 아님).

    - 원기둥 center/radius 정보를 MPC 내부 cost(파라미터)로 직접 넣어서
      *반경 유지* + *원기둥 바라보기* + *접선 속도 유지*를 동시에 최적화합니다.
    """

    def __init__(self, cfg, headless):
        # NOTE: IsaacEnv.__init__ (called inside super().__init__) will call `_design_scene()`.
        # Any attributes used inside `_design_scene()` must be defined before `super().__init__`.
        self.hide_target_vis = bool(cfg.task.get("hide_target_vis", True))
        self.orbit_yaw_offset = float(cfg.task.get("orbit_yaw_offset", 0.0))

        super().__init__(cfg, headless)

        # cylinder params (env-frame)
        center_cfg = cfg.task.get("cylinder_center", [0.0, 0.0, 1.5])
        self.cylinder_center = torch.tensor(center_cfg, device=self.device, dtype=torch.float32).view(1, 1, 3)
        self.cylinder_radius = float(cfg.task.get("cylinder_radius", 0.4))
        self.cylinder_height = float(cfg.task.get("cylinder_height", 3.0))

        # orbit params
        self.orbit_clearance = float(cfg.task.get("orbit_clearance", 1.0))
        self.orbit_radius = float(cfg.task.get("orbit_radius", self.cylinder_radius + self.orbit_clearance))
        self.orbit_z = float(cfg.task.get("orbit_z", float(self.cylinder_center[0, 0, 2].item())))
        self.orbit_period_steps = int(cfg.task.get("orbit_period_steps", self.max_episode_length))
        self.orbit_direction = 1.0 if float(cfg.task.get("orbit_direction", 1.0)) >= 0.0 else -1.0
        # desired tangential speed (m/s). Default makes ~1 lap over orbit_period_steps.
        self.orbit_v_tan = float(cfg.task.get("orbit_v_tan", 0.0))
        if self.orbit_v_tan <= 0.0:
            self.orbit_v_tan = float(2.0 * torch.pi * self.orbit_radius / (max(1, self.orbit_period_steps) * float(self.dt)))

    def _design_scene(self):
        global_paths = super()._design_scene()

        if self.hide_target_vis:
            try:
                target_prim = prim_utils.get_prim_at_path("/World/envs/env_0/target")
                UsdGeom.Imageable(target_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            except Exception:
                pass

        cylinder_cfg = self.cfg.task.get("cylinder", {})
        cylinder_prim_path = "/World/envs/env_0/Cylinder"
        center = self.cfg.task.get("cylinder_center", [0.0, 0.0, 1.5])

        prim = prim_utils.create_prim(
            prim_path=cylinder_prim_path,
            prim_type="Cylinder",
            translation=tuple(center),
        )
        geom = UsdGeom.Cylinder(prim)
        geom.GetRadiusAttr().Set(float(self.cfg.task.get("cylinder_radius", 0.4)))
        geom.GetHeightAttr().Set(float(self.cfg.task.get("cylinder_height", 3.0)))
        geom.GetAxisAttr().Set(UsdGeom.Tokens.z)

        color = cylinder_cfg.get("color", [0.9, 0.4, 0.1])
        if color is not None:
            geom.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])

        return global_paths

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # keep Hover observation stable: define a static "target" at cylinder center
        self.target_pos[env_ids] = self.cylinder_center.to(self.device).expand(len(env_ids), 1, 3)
        self.target_rot[env_ids] = euler_to_quaternion(torch.zeros(len(env_ids), 1, 3, device=self.device))
        self.target_heading[env_ids] = quat_axis(self.target_rot[env_ids].squeeze(1), 0).unsqueeze(1)

    def _pre_sim_step(self, tensordict):
        # 최신 상태 업데이트(자세/속도 포함)
        self.drone.get_state()

        # MPC는 body-frame v/w를 가정하므로 vel_b 사용
        root_state = torch.cat(
            [self.drone.pos, self.drone.rot, self.drone.vel_b], dim=-1
        ).squeeze(1)  # (num_envs, 13)

        # Cylinder-aware MPC: inject cylinder params directly (no waypoint)
        # NOTE: `root_state` position is in env-frame (see UnderwaterVehicle.get_state(env_frame=True)),
        # so the cylinder center must be provided in the same frame.
        center_env = self.cylinder_center.squeeze(0).expand(self.num_envs, 3)  # (num_envs, 3)
        cmds = self.controller.compute(
            root_state,
            center_w=center_env,
            radius=self.orbit_radius,
            z=self.orbit_z,
            v_tan=self.orbit_v_tan,
            direction=self.orbit_direction,
            yaw_offset=self.orbit_yaw_offset,
        ).unsqueeze(1)
        tensordict.set(("agents", "action"), cmds)
        self.effort = torch.abs(self.drone.apply_action(cmds))

    def _init_controller(self):
        from marinegym.controllers.cylinder_orbit_mpc_controller import CylinderOrbitMPCController
        from marinegym.controllers.thruster_allocation import compute_thruster_allocation_matrix_from_drone
        import numpy as np

        mass = float(self.drone.MASS_0.squeeze().item())
        inertia_xx, inertia_yy, inertia_zz = self.drone.INERTIA_0.squeeze().tolist()

        thrust_axis = int(self.cfg.task.get("thrust_axis", 0))
        if "thruster_allocation" in self.drone.params:
            B = np.asarray(self.drone.params["thruster_allocation"], dtype=np.float64)
        else:
            B = compute_thruster_allocation_matrix_from_drone(self.drone, thrust_axis=thrust_axis)

        uav_params = {
            "name": self.cfg.task.drone_model.name,
            "mass": mass,
            "inertia": {"xx": inertia_xx, "yy": inertia_yy, "zz": inertia_zz},
            "hydro_coef": self.drone.params["hydro_coef"],
            "thruster_allocation": B,
            "volume": float(self.drone.params.get("volume", 0.0)),
            "coBM": float(self.drone.params.get("coBM", 0.0)),
            "rho": float(self.drone.params.get("rho", 997.0)),
            "mpc_q_radial": float(self.cfg.task.get("mpc_q_radial", 50.0)),
            "mpc_q_z": float(self.cfg.task.get("mpc_q_z", 30.0)),
            "mpc_q_tan": float(self.cfg.task.get("mpc_q_tan", 10.0)),
            "mpc_q_radial_speed": float(self.cfg.task.get("mpc_q_radial_speed", 5.0)),
            "mpc_q_heading": float(self.cfg.task.get("mpc_q_heading", 30.0)),
            "mpc_q_roll": float(self.cfg.task.get("mpc_q_roll", 30.0)),
            "mpc_q_pitch": float(self.cfg.task.get("mpc_q_pitch", 10.0)),
            "mpc_q_wxy": float(self.cfg.task.get("mpc_q_wxy", 0.5)),
            "mpc_r_u": float(self.cfg.task.get("mpc_r_u", 0.01)),
            "mpc_max_thruster_force": float(self.cfg.task.get("mpc_max_thruster_force", 40.0)),
        }

        horizon = int(self.cfg.task.get("mpc_horizon", 20))
        dt = float(self.cfg.sim.dt)
        self.controller = CylinderOrbitMPCController(uav_params=uav_params, dt=dt, N=horizon)
