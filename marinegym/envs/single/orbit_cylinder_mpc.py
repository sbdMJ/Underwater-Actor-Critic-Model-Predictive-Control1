import math
import torch

import omni.isaac.core.utils.prims as prim_utils
from pxr import Gf, UsdGeom
from tensordict.tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec

from marinegym.envs.single.hover_mpc import HoverMPC
from marinegym.rewards.orbit_ppo import OrbitPPORewardCfg, compute_orbit_ppo_reward_terms
from marinegym.utils.torch import quat_axis, euler_to_quaternion


class OrbitCylinderMPC(HoverMPC):
    """
    Cylinder-aware NMPC 태스크.

    - 원기둥 center/radius 정보를 MPC 내부 cost(파라미터)로 직접 넣어서
      *반경 유지* + *원기둥 바라보기* + *접선 속도 유지*를 동시에 최적화합니다.
    - PyPose(iLQR) 모드에서는 `pypose_orbit_mode`에 따라
      (1) cylinder_cost(기존 cost 그대로) 또는
      (2) waypoint(원기둥 궤도 위 moving waypoint tracking)
      중 하나로 구동할 수 있습니다.
    """

    def __init__(self, cfg, headless):
        # NOTE: IsaacEnv.__init__ (called inside super().__init__) will call `_design_scene()`.
        # Any attributes used inside `_design_scene()` must be defined before `super().__init__`.
        self.hide_target_vis = bool(cfg.task.get("hide_target_vis", True))
        self.orbit_yaw_offset = float(cfg.task.get("orbit_yaw_offset", 0.0))
        self.reward_mode = str(cfg.task.get("reward_mode", "hover")).lower()
        self.use_pypose_mpc = bool(cfg.task.get("use_pypose_mpc", False))
        self.pypose_orbit_mode = str(cfg.task.get("pypose_orbit_mode", "cylinder_cost")).lower()
        self.orbit_waypoint_lookahead_steps = int(
            cfg.task.get("orbit_waypoint_lookahead_steps", cfg.task.get("pypose_orbit_waypoint_lookahead_steps", 0))
        )
        # Backward-compatible name.
        self.pypose_orbit_waypoint_lookahead_steps = self.orbit_waypoint_lookahead_steps

        control_mode = str(cfg.task.get("control_mode", "")).strip().lower()
        if control_mode in ("direct", "mpc"):
            cfg.task.use_internal_mpc = control_mode == "mpc"

        super().__init__(cfg, headless)

        flow_cfg = self.disturbances[self.mode]["flow"] if hasattr(self, "disturbances") else {}
        self.flow_update_mode = str(flow_cfg.get("update_mode", "reset")).strip().lower()
        self.flow_init_mode = str(flow_cfg.get("init_mode", "uniform")).strip().lower()
        self.flow_hybrid_prob = float(flow_cfg.get("hybrid_prob", 0.5))
        self.flow_tau = float(flow_cfg.get("tau", 0.0))
        self.flow_sigma = flow_cfg.get("sigma", getattr(self, "flow_velocity_gaussian_noise", 0.0))
        if getattr(self, "enable_flow", False):
            self._flow_sigma_tensor = torch.tensor(self.flow_sigma, device=self.device, dtype=torch.float32)
            self._flow_max_tensor = torch.tensor(self.max_flow_velocity, device=self.device, dtype=torch.float32)
            self._flow_hybrid_ou_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        self.current_vel = torch.zeros(self.num_envs, 3, device=self.device)
        current_cfg = self.cfg.env.get("current", {})
        self.current_enable = bool(current_cfg.get("enable", False))
        self.current_range = float(current_cfg.get("range", 0.0))

        orbit_target_mode = str(cfg.task.get("orbit_target_mode", "auto")).lower()
        if orbit_target_mode in ("auto", ""):
            # For RL training (use_internal_mpc=false):
            #   - hover reward wants a moving waypoint target in obs/reward.
            #   - orbit-cost reward does not need a moving target; keep target at center for stability.
            if self.reward_mode in ("orbit_cost", "orbit_ppo", "cylinder_cost", "cylinder_orbit_cost", "orbit"):
                orbit_target_mode = "center"
            else:
                orbit_target_mode = "waypoint" if not getattr(self, "_use_internal_mpc", True) else "center"
        self.orbit_target_mode = orbit_target_mode

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

        # Waypoint-mode phase (initialized on reset): theta0 per env.
        self._orbit_theta0 = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Deterministic spawn (train/eval): optionally force a fixed initial position and yaw
        # that faces the cylinder center in the horizontal plane.
        # Enable with `task.spawn_fixed=true`. If `task.spawn_fixed_pos` is not provided,
        # defaults to [cylinder_center.x + orbit_radius, cylinder_center.y, orbit_z].
        if bool(cfg.task.get("spawn_fixed", False)):
            center_cfg = cfg.task.get("cylinder_center", [0.0, 0.0, float(self.orbit_z)])
            center_vals = [float(x) for x in center_cfg]
            spawn_pos_cfg = cfg.task.get("spawn_fixed_pos", None)
            if spawn_pos_cfg is None:
                spawn_pos_cfg = [center_vals[0] + float(self.orbit_radius), center_vals[1], float(self.orbit_z)]
            self.spawn_fixed_pos = [float(x) for x in spawn_pos_cfg]
            face_point_cfg = cfg.task.get("spawn_face_point", center_vals)
            self.spawn_face_point = [float(x) for x in face_point_cfg]
            self.spawn_face_yaw_offset = float(cfg.task.get("spawn_face_yaw_offset", float(self.orbit_yaw_offset)))

        # Optional termination: if the vehicle stays nearly stationary (XY) for too long.
        # Useful to avoid "do nothing" local optima (e.g., earning only the effort reward).
        self.terminate_stationary_enable = bool(cfg.task.get("terminate_stationary_enable", False))
        self.terminate_stationary_xy_eps = float(cfg.task.get("terminate_stationary_xy_eps", 0.0))
        self.terminate_stationary_steps = int(cfg.task.get("terminate_stationary_steps", 0))
        self.terminate_stationary_grace_steps = int(cfg.task.get("terminate_stationary_grace_steps", 0))
        if (
            self.terminate_stationary_enable
            and self.terminate_stationary_steps > 0
            and self.terminate_stationary_xy_eps > 0.0
        ):
            self._stationary_steps_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            self._stationary_prev_xy = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)

        # Optional reward shaping: if the vehicle stays (almost) in place for too long, subtract a penalty.
        # This discourages "do nothing" local optima even when termination is disabled.
        self.reward_orbit_stationary_w = float(cfg.task.get("reward_orbit_w_stationary", 0.0))
        self.reward_orbit_stationary_xy_eps = float(
            cfg.task.get("reward_orbit_stationary_xy_eps", cfg.task.get("terminate_stationary_xy_eps", 0.0))
        )
        self.reward_orbit_stationary_grace_steps = int(
            cfg.task.get(
                "reward_orbit_stationary_grace_steps",
                cfg.task.get("terminate_stationary_grace_steps", 0),
            )
        )
        self.reward_orbit_stationary_start_steps = int(cfg.task.get("reward_orbit_stationary_start_steps", 0))
        self.reward_orbit_stationary_full_steps = int(cfg.task.get("reward_orbit_stationary_full_steps", 0))
        if self.reward_orbit_stationary_full_steps <= self.reward_orbit_stationary_start_steps:
            self.reward_orbit_stationary_full_steps = self.reward_orbit_stationary_start_steps + 1
        if self.reward_orbit_stationary_w != 0.0 and self.reward_orbit_stationary_xy_eps > 0.0:
            self._orbit_stationary_steps_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            self._orbit_stationary_prev_xy = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)

        # Optional reward shaping: penalize large actions more when orbit tracking is already good.
        # This gives the policy an incentive to increase learned w_u (r_u) in steady-state.
        self.reward_action_mag_weight = float(cfg.task.get("reward_action_mag_weight", 0.0))
        self.reward_action_mag_scale = float(cfg.task.get("reward_action_mag_scale", 1.0))
        self.reward_action_mag_gate_by_orbit = bool(cfg.task.get("reward_action_mag_gate_by_orbit", True))

        # --- orbit_ppo reward bookkeeping ---
        # Previous theta (for delta-theta progress reward) and previous action (for du penalty).
        # Stored per-env (agent dimension is always 1 for these single-agent tasks).
        self._orbit_prev_theta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        nu = int(self.drone.action_spec.shape[-1])
        self._orbit_prev_u_cmd = torch.zeros(self.num_envs, nu, device=self.device, dtype=torch.float32)
        # Optional: keep z_target fixed to the initial depth at reset.
        self._orbit_z_target_mode = str(cfg.task.get("reward_orbit_z_target_mode", "fixed")).strip().lower()
        self._orbit_z_target = torch.full((self.num_envs,), float(self.orbit_z), device=self.device, dtype=torch.float32)
        # Sparse lap bonus: add +bonus whenever the vehicle completes a full 2*pi orbit in the desired direction.
        self.reward_orbit_lap_bonus = float(cfg.task.get("reward_orbit_lap_bonus", 0.0))
        self._orbit_lap_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._orbit_lap_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

    @staticmethod
    def _is_waypoint_mode(mode: str) -> bool:
        return str(mode).lower() in ("waypoint", "moving_waypoint", "wp")

    def _set_specs(self):
        super()._set_specs()
        # Add extra per-term stats for orbit_ppo reward debugging / logging.
        # These are ignored by the PPO networks but get logged via scripts/train.py EpisodeStats.
        stats_spec = self.observation_spec["stats"]
        extra_keys = [
            "reward_step",
            "r_radius",
            "r_depth",
            "r_heading",
            "r_progress",
            "r_progress_raw",
            "progress_gate",
            "r_progress_back",
            "lap_bonus",
            "lap_count",
            "lap_progress",
            "r_att",
            "cos_yaw",
            "delta_theta",
            "v_tan",
            "v_rad",
            "delta_xy",
            "dist_xy",
            "e_r",
            "e_z",
            "roll",
            "pitch",
            "penalty_u",
            "penalty_du",
            "penalty_omega",
            "penalty_v_rad",
            "penalty_stationary",
            "stationary_steps",
            "terminate_orbit_ppo",
        ]
        leaf_shape = tuple(stats_spec.shape) + (1,)
        for k in extra_keys:
            if k not in stats_spec.keys():
                stats_spec[k] = UnboundedContinuousTensorSpec(leaf_shape, device=self.device)
        # Ensure spec device matches the env.
        stats_spec.to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _orbit_dtheta(self, *, dtype: torch.dtype, device) -> torch.Tensor:
        # dtheta per (env) step, based on desired tangential speed.
        r = float(self.orbit_radius)
        if r <= 1e-6:
            return torch.zeros((), device=device, dtype=dtype)
        dtheta = float(self.orbit_direction) * float(self.orbit_v_tan) * float(self.dt) / r
        return torch.as_tensor(dtheta, device=device, dtype=dtype)

    def _set_target_pose(self, env_ids: torch.Tensor, *, target_pos: torch.Tensor, target_quat: torch.Tensor):
        if env_ids.numel() == 0:
            return
        target_pos = target_pos.to(device=self.device, dtype=self.target_pos.dtype)
        target_quat = target_quat.to(device=self.device, dtype=self.target_rot.dtype)

        self.target_pos[env_ids, 0, :] = target_pos
        self.target_rot[env_ids, 0, :] = target_quat
        self.target_heading[env_ids, 0, :] = quat_axis(target_quat, 0)

        if self.hide_target_vis:
            return
        target_pos_world = self.target_pos[env_ids] + self.envs_positions[env_ids].unsqueeze(1)
        try:
            self.target_vis.set_world_poses(positions=target_pos_world.squeeze(1), indices=env_ids)
        except TypeError:
            self.target_vis.set_world_poses(positions=target_pos_world.squeeze(1), env_indices=env_ids)
        try:
            self.target_vis.set_world_poses(orientations=target_quat, indices=env_ids)
        except TypeError:
            self.target_vis.set_world_poses(orientations=target_quat, env_indices=env_ids)

    def _update_orbit_waypoint_target(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        dtype = self.target_pos.dtype
        device = self.device
        dtheta = self._orbit_dtheta(dtype=dtype, device=device)

        theta0 = self._orbit_theta0[env_ids].to(device=device, dtype=dtype)
        t = self.progress_buf[env_ids].to(device=device, dtype=dtype)
        theta = theta0 + t * dtheta
        theta_tgt = theta + float(self.orbit_waypoint_lookahead_steps) * dtheta
        theta_tgt = torch.remainder(theta_tgt, float(2.0 * torch.pi))

        center_env = self.cylinder_center.squeeze(0).expand(self.num_envs, 3)[env_ids].to(device=device, dtype=dtype)
        target_pos = center_env.clone()
        target_pos[:, 0] = target_pos[:, 0] + float(self.orbit_radius) * torch.cos(theta_tgt)
        target_pos[:, 1] = target_pos[:, 1] + float(self.orbit_radius) * torch.sin(theta_tgt)
        target_pos[:, 2] = float(self.orbit_z)

        to_center_xy = center_env[:, 0:2] - target_pos[:, 0:2]
        yaw = torch.atan2(to_center_xy[:, 1], to_center_xy[:, 0]) + float(self.orbit_yaw_offset)
        euler = torch.zeros((env_ids.shape[0], 3), device=device, dtype=dtype)
        euler[:, 2] = yaw
        target_quat = euler_to_quaternion(euler)

        self._set_target_pose(env_ids, target_pos=target_pos, target_quat=target_quat)

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
        if getattr(self, "enable_flow", False):
            num = int(env_ids.numel())
            flow = self.drone.flow_vels[env_ids, 0, :].clone()
            max_flow = self._flow_max_tensor.to(device=flow.device, dtype=flow.dtype)
            if self.flow_init_mode in ("uniform_signed", "signed"):
                flow = torch.empty((num, 6), device=flow.device, dtype=flow.dtype).uniform_(-1.0, 1.0)
                flow = flow * max_flow
            elif self.flow_init_mode in ("zero", "zeros"):
                flow = torch.zeros((num, 6), device=flow.device, dtype=flow.dtype)
            else:
                flow = torch.rand((num, 6), device=flow.device, dtype=flow.dtype) * max_flow
            self.drone.flow_vels[env_ids, 0, :] = flow
            if self.flow_update_mode == "hybrid":
                prob = float(max(0.0, min(1.0, self.flow_hybrid_prob)))
                self._flow_hybrid_ou_mask[env_ids] = torch.rand((num,), device=self.device) < prob
        if self.current_enable and self.training and self.current_range > 0.0:
            self.current_vel[env_ids] = torch.empty((env_ids.numel(), 3), device=self.device).uniform_(
                -self.current_range,
                self.current_range,
            )
        else:
            self.current_vel[env_ids] = 0.0

        # Reset orbit_ppo bookkeeping for the selected envs.
        try:
            pos = self.drone.pos[env_ids].squeeze(1)  # (E, 3) env-frame
            center = self.cylinder_center.squeeze(0).expand(self.num_envs, 3)[env_ids].to(device=pos.device, dtype=pos.dtype)
            rel = pos[:, 0:2] - center[:, 0:2]
            theta = torch.atan2(rel[:, 1], rel[:, 0])
            self._orbit_prev_theta[env_ids] = theta.to(dtype=self._orbit_prev_theta.dtype)
            self._orbit_prev_u_cmd[env_ids] = 0.0
            self._orbit_lap_progress[env_ids] = 0.0
            self._orbit_lap_count[env_ids] = 0
            if self._orbit_z_target_mode in ("initial", "init", "reset"):
                self._orbit_z_target[env_ids] = pos[:, 2].to(dtype=self._orbit_z_target.dtype)
            else:
                self._orbit_z_target[env_ids] = float(self.orbit_z)
        except Exception:
            pass
        if self.current_enable and hasattr(self, "drone") and hasattr(self.drone, "flow_vels"):
            base_flow = self.drone.flow_vels[env_ids, 0, :3]
            self.drone.flow_vels[env_ids, 0, :3] = base_flow + self.current_vel[env_ids].to(
                dtype=self.drone.flow_vels.dtype
            )
        # Initialize orbit phase so the moving waypoint starts near the current spawn pose.
        try:
            self.drone.get_state()
            pos = self.drone.pos[env_ids].squeeze(1)  # (E, 3) env-frame
            center = self.cylinder_center.to(self.device).squeeze(0).squeeze(0)  # (3,)
            rel = pos - center  # (E, 3)
            self._orbit_theta0[env_ids] = torch.atan2(rel[:, 1], rel[:, 0]).to(dtype=torch.float32)
        except Exception:
            self._orbit_theta0[env_ids] = 0.0

        # Reset stationary-termination buffers.
        if (
            getattr(self, "terminate_stationary_enable", False)
            and hasattr(self, "_stationary_steps_buf")
            and hasattr(self, "_stationary_prev_xy")
        ):
            try:
                self.drone.get_state()
                pos_xy = self.drone.pos[env_ids].squeeze(1)[:, 0:2]  # (E, 2)
                self._stationary_steps_buf[env_ids] = 0
                self._stationary_prev_xy[env_ids].copy_(pos_xy.to(dtype=self._stationary_prev_xy.dtype))
            except Exception:
                try:
                    self._stationary_steps_buf[env_ids] = 0
                except Exception:
                    pass

        # Reset stationary-penalty buffers (orbit_ppo only).
        if hasattr(self, "_orbit_stationary_steps_buf") and hasattr(self, "_orbit_stationary_prev_xy"):
            try:
                self.drone.get_state()
                pos_xy = self.drone.pos[env_ids].squeeze(1)[:, 0:2]  # (E, 2)
                self._orbit_stationary_steps_buf[env_ids] = 0
                self._orbit_stationary_prev_xy[env_ids].copy_(pos_xy.to(dtype=self._orbit_stationary_prev_xy.dtype))
            except Exception:
                try:
                    self._orbit_stationary_steps_buf[env_ids] = 0
                except Exception:
                    pass

        if self._is_waypoint_mode(getattr(self, "orbit_target_mode", "center")):
            # IsaacEnv resets progress_buf after _reset_idx; ensure t=0 here for correct initial waypoint.
            self.progress_buf[env_ids] = 0.0
            self._update_orbit_waypoint_target(env_ids)
            return

        # keep Hover observation stable: define a static "target" at cylinder center
        center_env = self.cylinder_center.squeeze(0).expand(self.num_envs, 3)[env_ids]
        euler = torch.zeros((env_ids.shape[0], 3), device=self.device, dtype=center_env.dtype)
        target_quat = euler_to_quaternion(euler)
        self._set_target_pose(env_ids, target_pos=center_env, target_quat=target_quat)

    def _compute_state_and_obs(self):
        if self._is_waypoint_mode(getattr(self, "orbit_target_mode", "center")):
            self._update_orbit_waypoint_target()
        td = super()._compute_state_and_obs()

        # If we keep the target at the center, make its heading dynamically point to the cylinder center
        # (so obs rheading is meaningful) when training with orbit-cost reward.
        if (
            self.reward_mode in ("orbit_cost", "orbit_ppo", "cylinder_cost", "cylinder_orbit_cost", "orbit")
            and not self._is_waypoint_mode(getattr(self, "orbit_target_mode", "center"))
        ):
            try:
                pos = self.drone_state[..., :3]  # (E, 1, 3) env-frame
                center_env = self.cylinder_center.expand(self.num_envs, 1, 3).to(device=pos.device, dtype=pos.dtype)
                rel_xy = center_env[..., 0:2] - pos[..., 0:2]
                dist = torch.norm(rel_xy, dim=-1, keepdim=True).clamp_min(1e-6)
                h_des_xy = rel_xy / dist

                yaw_offset = torch.as_tensor(float(self.orbit_yaw_offset), device=pos.device, dtype=pos.dtype)
                co = torch.cos(yaw_offset)
                so = torch.sin(yaw_offset)
                h_tgt_xy = torch.stack(
                    [
                        co * h_des_xy[..., 0] - so * h_des_xy[..., 1],
                        so * h_des_xy[..., 0] + co * h_des_xy[..., 1],
                    ],
                    dim=-1,
                )
                z0 = torch.zeros((*h_tgt_xy.shape[:-1], 1), device=pos.device, dtype=pos.dtype)
                h_tgt = torch.cat([h_tgt_xy, z0], dim=-1)
                h_tgt = h_tgt / (torch.norm(h_tgt, dim=-1, keepdim=True) + 1e-6)
                self.target_heading[:] = h_tgt

                yaw = torch.atan2(h_tgt_xy[..., 1], h_tgt_xy[..., 0])
                euler = torch.zeros((self.num_envs, 1, 3), device=pos.device, dtype=pos.dtype)
                euler[..., 2] = yaw
                self.target_rot[:] = euler_to_quaternion(euler)

                self.rpos = self.target_pos - self.drone_state[..., :3]
                self.rheading = self.target_heading - self.drone_state[..., 13:16]

                obs_parts = [self.rpos, self.drone_state[..., 3:], self.rheading]
                if self.include_target_quat_in_obs:
                    obs_parts.append(self.target_rot)
                if self.time_encoding:
                    t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
                    obs_parts.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
                td.set(("agents", "observation"), torch.cat(obs_parts, dim=-1))
            except Exception:
                pass

        if bool(self.cfg.task.get("include_cylinder_rel_in_obs", False)):
            try:
                obs = td[("agents", "observation")]
                center_env = self.cylinder_center.squeeze(0).expand(self.num_envs, 3)  # (E, 3)
                target_pos = self.target_pos.squeeze(1)  # (E, 3)
                center_rel = (center_env - target_pos).to(device=obs.device, dtype=obs.dtype).unsqueeze(1)  # (E, 1, 3)
                td.set(("agents", "observation"), torch.cat([obs, center_rel], dim=-1))
            except Exception:
                pass
        return td

    def _compute_reward_and_done(self):
        orbit_reward_modes = ("orbit_cost", "orbit_ppo", "cylinder_cost", "cylinder_orbit_cost", "orbit")
        if self.reward_mode not in orbit_reward_modes:
            return super()._compute_reward_and_done()

        try:
            from marinegym.controllers.pypose_cylinder_orbit_mpc_controller import _orbit_errors
        except Exception:
            return super()._compute_reward_and_done()

        # Use the same orbit error vector as the MPC cost (but the env reward weights are fixed).
        root_state = torch.cat([self.drone.pos, self.drone.rot, self.drone.vel_b], dim=-1).squeeze(1)  # (E, 13)
        dtype = root_state.dtype
        device = root_state.device

        center_env = self.cylinder_center.squeeze(0).expand(self.num_envs, 3).to(device=device, dtype=dtype)
        radius_t = torch.as_tensor(float(self.orbit_radius), device=device, dtype=dtype)
        z_t = torch.as_tensor(float(self.orbit_z), device=device, dtype=dtype)
        v_tan_t = torch.as_tensor(float(self.orbit_v_tan), device=device, dtype=dtype)
        dir_sign_t = torch.as_tensor(float(self.orbit_direction), device=device, dtype=dtype)
        yaw_offset_t = torch.as_tensor(float(self.orbit_yaw_offset), device=device, dtype=dtype)

        e = _orbit_errors(
            root_state,
            center_w=center_env,
            radius=radius_t,
            z=z_t,
            v_tan=v_tan_t,
            dir_sign=dir_sign_t,
            yaw_offset=yaw_offset_t,
        )  # (E, 10)

        if self.reward_mode == "orbit_ppo":
            # Full PPO reward shaping (radius/depth/heading/clockwise progress/attitude + penalties).
            # The actual term computation lives in `marinegym.rewards.orbit_ppo` for easier testing.
            nu = int(self.drone.action_spec.shape[-1])
            u_cmd = getattr(self, "_last_action_cmd", None)
            if u_cmd is None:
                u_cmd = getattr(self.drone, "throttle", None)
            if u_cmd is None:
                u_cmd = torch.zeros((self.num_envs, nu), device=device, dtype=dtype)
            if u_cmd.ndim >= 3:
                u_cmd = u_cmd.squeeze(1)
            u_cmd = u_cmd.to(device=device, dtype=dtype)

            u_prev = self._orbit_prev_u_cmd.to(device=device, dtype=dtype)
            theta_prev = self._orbit_prev_theta.to(device=device, dtype=dtype).view(self.num_envs, 1)

            # z_target can be fixed (cfg orbit_z) or "initial depth" (captured at reset).
            if self._orbit_z_target_mode in ("initial", "init", "reset"):
                z_target = self._orbit_z_target.to(device=device, dtype=dtype).view(self.num_envs, 1)
            else:
                z_target = z_t.view(1, 1).expand(self.num_envs, 1)

            cfg_reward = OrbitPPORewardCfg(
                sigma_r=float(self.cfg.task.get("reward_orbit_sigma_r", 0.20)),
                sigma_z=float(self.cfg.task.get("reward_orbit_sigma_z", 0.20)),
                sigma_roll=float(self.cfg.task.get("reward_orbit_sigma_roll", 0.35)),
                sigma_pitch=float(self.cfg.task.get("reward_orbit_sigma_pitch", 0.35)),
                theta_scale=float(self.cfg.task.get("reward_orbit_theta_scale", float(5.0 * torch.pi / 180.0))),
                v_tan_ref=float(self.cfg.task.get("reward_orbit_v_tan_ref", float(abs(self.orbit_v_tan)) if abs(self.orbit_v_tan) > 1e-6 else 0.5)),
                progress_mode=str(self.cfg.task.get("reward_orbit_progress_mode", "theta")),
                yaw_offset=float(self.cfg.task.get("reward_orbit_yaw_offset", float(self.orbit_yaw_offset))),
                progress_gate_mode=str(self.cfg.task.get("reward_orbit_progress_gate_mode", "none")),
                gate_heading_power=float(self.cfg.task.get("reward_orbit_gate_heading_power", 2.0)),
                gate_radius_power=float(self.cfg.task.get("reward_orbit_gate_radius_power", 1.0)),
                gate_heading_min=float(self.cfg.task.get("reward_orbit_gate_heading_min", 0.8)),
                gate_radius_min=float(self.cfg.task.get("reward_orbit_gate_radius_min", 0.8)),
                w_r=float(self.cfg.task.get("reward_orbit_w_radius", 1.0)),
                w_z=float(self.cfg.task.get("reward_orbit_w_depth", 1.0)),
                w_h=float(self.cfg.task.get("reward_orbit_w_heading", 0.5)),
                w_p=float(self.cfg.task.get("reward_orbit_w_progress", 0.5)),
                w_a=float(self.cfg.task.get("reward_orbit_w_att", 0.25)),
                w_back=float(self.cfg.task.get("reward_orbit_w_backwards", 0.0)),
                w_u=float(self.cfg.task.get("reward_orbit_w_u", 0.05)),
                w_du=float(self.cfg.task.get("reward_orbit_w_du", 0.10)),
                w_w=float(self.cfg.task.get("reward_orbit_w_omega", 0.02)),
                w_vrad=float(self.cfg.task.get("reward_orbit_w_v_rad", 0.0)),
                v_rad_ref=float(self.cfg.task.get("reward_orbit_v_rad_ref", 0.3)),
                r_min=float(self.cfg.task.get("reward_orbit_r_min", float(self.cylinder_radius))),
                r_max=float(self.cfg.task.get("reward_orbit_r_max", float(self.orbit_radius + 2.0))),
                rp_max_deg=float(self.cfg.task.get("reward_orbit_rp_max_deg", 45.0)),
                terminate_penalty=float(self.cfg.task.get("reward_orbit_terminate_penalty", -2.0)),
                clip_min=float(self.cfg.task.get("reward_orbit_clip_min", -5.0)),
                clip_max=float(self.cfg.task.get("reward_orbit_clip_max", 5.0)),
            )

            reward, terms = compute_orbit_ppo_reward_terms(
                p_world=root_state[:, 0:3],
                q_world_wxyz=root_state[:, 3:7],
                v_body=root_state[:, 7:10],
                w_body=root_state[:, 10:13],
                c_world=center_env,
                r_target=radius_t.view(1, 1).expand(self.num_envs, 1),
                z_target=z_target,
                orbit_direction=dir_sign_t.view(1, 1).expand(self.num_envs, 1),
                dt=float(self.dt),
                u_cmd=u_cmd,
                u_prev=u_prev,
                theta_prev=theta_prev,
                cfg=cfg_reward,
            )
            reward_orbit = reward.view(self.num_envs, 1)
            # Lap bonus: accumulate directed delta-theta and add a sparse bonus on each completed lap.
            lap_bonus_term = torch.zeros_like(reward_orbit)
            if float(getattr(self, "reward_orbit_lap_bonus", 0.0)) != 0.0:
                try:
                    progress_gate = terms.get("progress_gate", torch.ones_like(reward_orbit)).to(dtype=reward_orbit.dtype)
                    dtheta_dir = dir_sign_t.view(1, 1).expand(self.num_envs, 1) * terms["delta_theta"].to(
                        dtype=reward_orbit.dtype
                    )
                    lap_progress = self._orbit_lap_progress.to(device=device, dtype=reward_orbit.dtype).view(self.num_envs, 1)
                    lap_progress = (lap_progress + dtheta_dir).clamp_min(0.0)

                    two_pi = float(2.0 * torch.pi)
                    laps = torch.floor(lap_progress / two_pi).clamp_min(0.0)
                    lap_progress = lap_progress - laps * two_pi

                    laps_i = laps.to(dtype=torch.int32).view(self.num_envs)
                    self._orbit_lap_progress.copy_(lap_progress.view(self.num_envs).to(dtype=self._orbit_lap_progress.dtype))
                    self._orbit_lap_count.add_(laps_i.to(device=self._orbit_lap_count.device))

                    # Gate the lap bonus so we only reward a completed lap when radius+heading tracking are good.
                    lap_bonus_term = laps * float(self.reward_orbit_lap_bonus) * progress_gate
                    reward_orbit = reward_orbit + lap_bonus_term
                    terms["lap_bonus"] = lap_bonus_term
                    terms["lap_count"] = self._orbit_lap_count.to(device=device, dtype=reward_orbit.dtype).view(self.num_envs, 1)
                    terms["lap_progress"] = lap_progress
                except Exception:
                    pass
            # Update prev buffers for next step.
            try:
                self._orbit_prev_theta.copy_(terms["theta"].view(self.num_envs).to(dtype=self._orbit_prev_theta.dtype))
                self._orbit_prev_u_cmd.copy_(u_cmd.to(dtype=self._orbit_prev_u_cmd.dtype))
            except Exception:
                pass

            terminate_orbit_ppo = terms["terminate_orbit_ppo"].view(self.num_envs, 1).to(dtype=torch.bool)
        else:
            # Original orbit-cost reward: use the MPC error vector and fixed quadratic weights.
            q_radial = float(self.cfg.task.get("reward_q_radial", self.cfg.task.get("mpc_q_radial", 50.0)))
            q_z = float(self.cfg.task.get("reward_q_z", self.cfg.task.get("mpc_q_z", 30.0)))
            q_tan = float(self.cfg.task.get("reward_q_tan", self.cfg.task.get("mpc_q_tan", 10.0)))
            q_radial_speed = float(
                self.cfg.task.get("reward_q_radial_speed", self.cfg.task.get("mpc_q_radial_speed", 5.0))
            )
            q_heading = float(self.cfg.task.get("reward_q_heading", self.cfg.task.get("mpc_q_heading", 30.0)))
            q_roll = float(self.cfg.task.get("reward_q_roll", self.cfg.task.get("mpc_q_roll", 60.0)))
            q_pitch = float(self.cfg.task.get("reward_q_pitch", self.cfg.task.get("mpc_q_pitch", 60.0)))
            q_wxy = float(self.cfg.task.get("reward_q_wxy", self.cfg.task.get("mpc_q_wxy", 0.5)))

            w_err = torch.as_tensor(
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
                ],
                device=device,
                dtype=dtype,
            )

            orbit_cost = (w_err * e.square()).sum(dim=-1, keepdim=True)  # (E, 1)
            cost_scale = float(self.cfg.task.get("reward_orbit_cost_scale", 1.0))
            reward_orbit = 1.0 / (1.0 + cost_scale * orbit_cost)

        # For orbit_ppo, `reward_orbit` already includes all shaping and penalties.
        if self.reward_mode == "orbit_ppo":
            reward_effort = torch.zeros_like(reward_orbit)
            reward_action_smoothness = torch.zeros_like(reward_orbit)
            reward = reward_orbit
        else:
            reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
            reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)
            reward = reward_orbit + reward_effort + reward_action_smoothness  # (E, 1)

        # Extra action-magnitude reward term (optional).
        if self.reward_action_mag_weight != 0.0 and self.reward_action_mag_scale > 0.0:
            try:
                u_cmd = getattr(self, "_last_action_cmd", None)
                if u_cmd is None:
                    u_cmd = getattr(self.drone, "throttle", None)
                if u_cmd is not None:
                    # u_cmd: (E,1,nu) or (E,nu)
                    if u_cmd.ndim >= 3:
                        u_cmd = u_cmd.squeeze(1)
                    u_mag = torch.mean(torch.abs(u_cmd), dim=-1, keepdim=True)  # (E, 1)
                    gate = reward_orbit if bool(self.reward_action_mag_gate_by_orbit) else 1.0
                    reward_action_mag = float(self.reward_action_mag_weight) * gate * torch.exp(
                        -float(self.reward_action_mag_scale) * u_mag
                    )
                    reward = reward + reward_action_mag
            except Exception:
                pass

        # Optional: if we have been stationary (XY) for too long, subtract a penalty.
        penalty_stationary = torch.zeros_like(reward)
        stationary_steps_term = torch.zeros_like(reward)
        delta_xy_term = torch.zeros_like(reward)
        if (
            self.reward_mode == "orbit_ppo"
            and float(getattr(self, "reward_orbit_stationary_w", 0.0)) != 0.0
            and float(getattr(self, "reward_orbit_stationary_xy_eps", 0.0)) > 0.0
            and hasattr(self, "_orbit_stationary_steps_buf")
            and hasattr(self, "_orbit_stationary_prev_xy")
        ):
            try:
                pos_xy = root_state[:, 0:2]  # (E, 2)
                prev_xy = self._orbit_stationary_prev_xy.to(device=pos_xy.device, dtype=pos_xy.dtype)
                delta_xy = torch.linalg.norm(pos_xy - prev_xy, dim=-1)  # (E,)
                self._orbit_stationary_prev_xy.copy_(pos_xy.to(dtype=self._orbit_stationary_prev_xy.dtype))

                grace = int(getattr(self, "reward_orbit_stationary_grace_steps", 0))
                if grace > 0:
                    active = self.progress_buf >= float(grace)
                else:
                    active = torch.ones_like(delta_xy, dtype=torch.bool, device=delta_xy.device)
                still = active & (delta_xy < float(self.reward_orbit_stationary_xy_eps))

                steps = self._orbit_stationary_steps_buf.to(device=delta_xy.device)
                steps = torch.where(still, steps + 1, torch.zeros_like(steps))
                self._orbit_stationary_steps_buf.copy_(steps.to(dtype=self._orbit_stationary_steps_buf.dtype))

                start_steps = int(getattr(self, "reward_orbit_stationary_start_steps", 0))
                full_steps = int(getattr(self, "reward_orbit_stationary_full_steps", start_steps + 1))
                if full_steps <= start_steps:
                    full_steps = start_steps + 1
                denom = max(1, full_steps - start_steps)
                ratio = ((steps - start_steps).clamp(min=0).to(dtype=reward.dtype) / float(denom)).clamp(0.0, 1.0)

                penalty_stationary = ratio.view(self.num_envs, 1)
                stationary_steps_term = steps.to(dtype=reward.dtype).view(self.num_envs, 1)
                delta_xy_term = delta_xy.to(dtype=reward.dtype).view(self.num_envs, 1)

                reward = reward - float(self.reward_orbit_stationary_w) * penalty_stationary
                # Keep final reward bounded.
                reward = reward.clamp(
                    min=float(self.cfg.task.get("reward_orbit_clip_min", -5.0)),
                    max=float(self.cfg.task.get("reward_orbit_clip_max", 5.0)),
                )
            except Exception:
                pass
        if self.reward_mode == "orbit_ppo":
            try:
                terms["penalty_stationary"] = penalty_stationary
                terms["stationary_steps"] = stationary_steps_term
                terms["delta_xy"] = delta_xy_term
            except Exception:
                pass

        # Final reward clipping (orbit_ppo can add lap bonus / stationary penalty after core shaping).
        if self.reward_mode == "orbit_ppo":
            reward = reward.clamp(
                min=float(self.cfg.task.get("reward_orbit_clip_min", -5.0)),
                max=float(self.cfg.task.get("reward_orbit_clip_max", 5.0)),
            )

        # Termination: keep Hover's rules (z floor, far-away, NaN, timeout).
        try:
            distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)
        except Exception:
            distance = torch.abs(e[:, 0]).view(self.num_envs, 1)

        misbehave = (self.drone.pos[..., 2] < self.terminate_min_z) | (distance > self.terminate_distance)
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        if self.reward_mode == "orbit_ppo":
            try:
                terminated = terminated | terminate_orbit_ppo
            except Exception:
                pass

        # Optional: terminate if the vehicle is (almost) stationary for too long.
        if (
            getattr(self, "terminate_stationary_enable", False)
            and hasattr(self, "_stationary_steps_buf")
            and hasattr(self, "_stationary_prev_xy")
            and int(getattr(self, "terminate_stationary_steps", 0)) > 0
            and float(getattr(self, "terminate_stationary_xy_eps", 0.0)) > 0.0
        ):
            try:
                pos_xy = root_state[:, 0:2]  # (E, 2)
                prev_xy = self._stationary_prev_xy.to(device=pos_xy.device, dtype=pos_xy.dtype)
                delta_xy = torch.linalg.norm(pos_xy - prev_xy, dim=-1)  # (E,)
                # Update prev before early returns (stability across steps).
                self._stationary_prev_xy.copy_(pos_xy.to(dtype=self._stationary_prev_xy.dtype))

                grace = int(getattr(self, "terminate_stationary_grace_steps", 0))
                if grace > 0:
                    active = self.progress_buf >= float(grace)
                else:
                    active = torch.ones_like(delta_xy, dtype=torch.bool, device=delta_xy.device)

                still = active & (delta_xy < float(self.terminate_stationary_xy_eps))
                self._stationary_steps_buf[still] += 1
                self._stationary_steps_buf[~still] = 0

                stuck = (self._stationary_steps_buf >= int(self.terminate_stationary_steps)).view(self.num_envs, 1)
                terminated = terminated | stuck
            except Exception:
                pass

        radial_err = torch.abs(e[:, 0]).view(self.num_envs, 1)
        heading_err = e[:, 4:6]
        heading_alignment = (1.0 - 0.5 * heading_err.square().sum(dim=-1)).view(self.num_envs, 1)

        self.stats["pos_error"].lerp_(radial_err, (1 - self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1 - self.alpha))
        self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1 - self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1 - self.alpha))
        if self.reward_mode == "orbit_ppo":
            try:
                # terms are (E,1)
                self.stats["reward_step"].lerp_(reward, (1 - self.alpha))
                self.stats["r_radius"].lerp_(terms["r_radius"], (1 - self.alpha))
                self.stats["r_depth"].lerp_(terms["r_depth"], (1 - self.alpha))
                self.stats["r_heading"].lerp_(terms["r_heading"], (1 - self.alpha))
                self.stats["r_progress"].lerp_(terms["r_progress"], (1 - self.alpha))
                self.stats["r_progress_raw"].lerp_(terms.get("r_progress_raw", terms["r_progress"]), (1 - self.alpha))
                self.stats["progress_gate"].lerp_(terms.get("progress_gate", torch.ones_like(reward)), (1 - self.alpha))
                self.stats["r_progress_back"].lerp_(terms["r_progress_back"], (1 - self.alpha))
                self.stats["lap_bonus"].lerp_(terms.get("lap_bonus", torch.zeros_like(reward)), (1 - self.alpha))
                self.stats["lap_count"].lerp_(terms.get("lap_count", torch.zeros_like(reward)), (1 - self.alpha))
                self.stats["lap_progress"].lerp_(terms.get("lap_progress", torch.zeros_like(reward)), (1 - self.alpha))
                self.stats["r_att"].lerp_(terms["r_att"], (1 - self.alpha))
                self.stats["cos_yaw"].lerp_(terms["cos_yaw"], (1 - self.alpha))
                self.stats["delta_theta"].lerp_(terms["delta_theta"].abs(), (1 - self.alpha))
                self.stats["v_tan"].lerp_(terms["v_tan"], (1 - self.alpha))
                self.stats["v_rad"].lerp_(terms.get("v_rad", torch.zeros_like(reward)), (1 - self.alpha))
                self.stats["delta_xy"].lerp_(terms["delta_xy"], (1 - self.alpha))
                self.stats["dist_xy"].lerp_(terms["dist_xy"], (1 - self.alpha))
                self.stats["e_r"].lerp_(terms["e_r"], (1 - self.alpha))
                self.stats["e_z"].lerp_(terms["e_z"], (1 - self.alpha))
                self.stats["roll"].lerp_(terms["roll"].abs(), (1 - self.alpha))
                self.stats["pitch"].lerp_(terms["pitch"].abs(), (1 - self.alpha))
                self.stats["penalty_u"].lerp_(terms["penalty_u"], (1 - self.alpha))
                self.stats["penalty_du"].lerp_(terms["penalty_du"], (1 - self.alpha))
                self.stats["penalty_omega"].lerp_(terms["penalty_omega"], (1 - self.alpha))
                self.stats["penalty_v_rad"].lerp_(terms.get("penalty_v_rad", torch.zeros_like(reward)), (1 - self.alpha))
                self.stats["penalty_stationary"].lerp_(terms["penalty_stationary"], (1 - self.alpha))
                self.stats["stationary_steps"].lerp_(terms["stationary_steps"], (1 - self.alpha))
                self.stats["terminate_orbit_ppo"].lerp_(terms["terminate_orbit_ppo"], (1 - self.alpha))
            except Exception:
                pass
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "stats": self.stats.clone(),
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _update_flow_disturbance(self):
        if not getattr(self, "enable_flow", False):
            return
        mode = self.flow_update_mode
        if mode == "hybrid":
            if not bool(self._flow_hybrid_ou_mask.any()):
                return
        elif mode not in ("ou", "ornstein_uhlenbeck"):
            return
        tau = float(self.flow_tau)
        if tau <= 0.0:
            return
        dt = float(getattr(self, "dt", 0.0))
        if dt <= 0.0:
            return
        flow = self.drone.flow_vels[:, 0, :]
        max_flow = self._flow_max_tensor.to(device=flow.device, dtype=flow.dtype)
        sigma = self._flow_sigma_tensor.to(device=flow.device, dtype=flow.dtype)
        if mode == "hybrid":
            mask = self._flow_hybrid_ou_mask
            if mask.shape[0] != flow.shape[0]:
                mask = mask[: flow.shape[0]]
            flow_sel = flow[mask]
            noise = torch.randn_like(flow_sel) * sigma * math.sqrt(dt)
            flow_sel = flow_sel + (-flow_sel / tau) * dt + noise
            flow_sel = torch.clamp(flow_sel, min=-max_flow, max=max_flow)
            flow = flow.clone()
            flow[mask] = flow_sel
        else:
            noise = torch.randn_like(flow) * sigma * math.sqrt(dt)
            flow = flow + (-flow / tau) * dt + noise
            flow = torch.clamp(flow, min=-max_flow, max=max_flow)
        self.drone.flow_vels[:, 0, :] = flow

    def _pre_sim_step(self, tensordict):
        self._update_flow_disturbance()
        if not getattr(self, "_use_internal_mpc", True):
            try:
                self._last_action_cmd = tensordict[("agents", "action")].detach()
            except Exception:
                pass
            return super()._pre_sim_step(tensordict)

        # 최신 상태 업데이트(자세/속도 포함)
        self.drone.get_state()

        # MPC는 body-frame v/w를 가정하므로 vel_b 사용
        root_state = torch.cat(
            [self.drone.pos, self.drone.rot, self.drone.vel_b], dim=-1
        ).squeeze(1)  # (num_envs, 13)

        if self.use_pypose_mpc:
            if self._is_waypoint_mode(self.pypose_orbit_mode):
                # Moving waypoint tracking along the orbit circle (constant-z).
                dtype = root_state.dtype
                dtheta = self._orbit_dtheta(dtype=dtype, device=self.device)

                theta0 = self._orbit_theta0.to(device=self.device, dtype=dtype)
                t = self.progress_buf.to(device=self.device, dtype=dtype)
                theta = theta0 + t * dtheta
                theta_tgt = theta + float(self.orbit_waypoint_lookahead_steps) * dtheta
                theta_tgt = torch.remainder(theta_tgt, float(2.0 * torch.pi))

                center_env = self.cylinder_center.squeeze(0).expand(self.num_envs, 3).to(device=self.device, dtype=dtype)
                target_pos = center_env.clone()
                target_pos[:, 0] = target_pos[:, 0] + float(self.orbit_radius) * torch.cos(theta_tgt)
                target_pos[:, 1] = target_pos[:, 1] + float(self.orbit_radius) * torch.sin(theta_tgt)
                target_pos[:, 2] = float(self.orbit_z)

                to_center_xy = center_env[:, 0:2] - target_pos[:, 0:2]
                yaw = torch.atan2(to_center_xy[:, 1], to_center_xy[:, 0]) + float(self.orbit_yaw_offset)
                euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=dtype)
                euler[:, 2] = yaw
                target_quat = euler_to_quaternion(euler)

                cmds = self.controller.compute(root_state, target_pos, target_quat=target_quat).unsqueeze(1)
                tensordict.set(("agents", "action"), cmds)
                try:
                    self._last_action_cmd = cmds.detach()
                except Exception:
                    pass
                self.effort = torch.abs(self.drone.apply_action(cmds))
                return

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
            try:
                self._last_action_cmd = cmds.detach()
            except Exception:
                pass
            self.effort = torch.abs(self.drone.apply_action(cmds))
            return

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
        try:
            self._last_action_cmd = cmds.detach()
        except Exception:
            pass
        self.effort = torch.abs(self.drone.apply_action(cmds))

    def _init_controller(self):
        if self.use_pypose_mpc:
            return self._init_pypose_controller()

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

    def _init_pypose_controller(self):
        mode = str(getattr(self, "pypose_orbit_mode", self.cfg.task.get("pypose_orbit_mode", "cylinder_cost"))).lower()
        from marinegym.controllers.thruster_allocation import compute_thruster_allocation_matrix_from_drone
        import numpy as np

        mass = float(self.drone.MASS_0.squeeze().item())
        inertia_xx, inertia_yy, inertia_zz = self.drone.INERTIA_0.squeeze().tolist()

        thrust_axis = int(self.cfg.task.get("thrust_axis", 0))
        if getattr(self.drone, "thruster_allocation", None) is not None:
            B = np.asarray(self.drone.thruster_allocation.detach().cpu().numpy(), dtype=np.float64)
        elif "thruster_allocation" in self.drone.params:
            B = np.asarray(self.drone.params["thruster_allocation"], dtype=np.float64)
        else:
            B = compute_thruster_allocation_matrix_from_drone(self.drone, thrust_axis=thrust_axis)

        rho = float(self.drone.params.get("rho", self.drone.params.get("water_density", 997.0)))
        volume = float(self.drone.params.get("volume", 0.0))
        buoy_mode = str(self.drone.params.get("buoyancy_mode", "volume")).lower()
        if buoy_mode in ("neutral", "neutral_mass", "match_weight") and rho > 0.0:
            volume = float(mass / rho)

        hydro_coef = self.drone.params["hydro_coef"]
        if bool(self.drone.params.get("disable_hydrodynamics", False)):
            hydro_coef = {
                "added_mass": [0.0] * 6,
                "linear_damping": [0.0] * 6,
                "quadratic_damping": [0.0] * 6,
            }

        coBM = float(self.drone.params.get("coBM", 0.0))
        try:
            axis = getattr(self.drone, "_cobm_axis_unit", None)
            if axis is not None:
                axis = axis.detach().cpu().numpy().reshape(-1)
                if axis.shape[0] == 3 and not (
                    abs(float(axis[0])) < 1e-3 and abs(float(axis[1])) < 1e-3 and float(axis[2]) > 0.999
                ):
                    coBM = 0.0
        except Exception:
            pass

        uav_params = {
            "name": self.cfg.task.drone_model.name,
            "mass": mass,
            "inertia": {"xx": inertia_xx, "yy": inertia_yy, "zz": inertia_zz},
            "hydro_coef": hydro_coef,
            "thruster_allocation": B,
            "volume": volume,
            "coBM": coBM,
            "rho": rho,
            "g": 9.81,
        }

        horizon = int(self.cfg.task.get("pypose_mpc_horizon", self.cfg.task.get("mpc_horizon", 20)))
        ilqr_iters = int(self.cfg.task.get("pypose_mpc_ilqr_iters", 4))
        ilqr_reg = float(self.cfg.task.get("pypose_mpc_ilqr_reg", 1e-3))
        terminal_mult = float(self.cfg.task.get("pypose_mpc_terminal_weight_mult", 1.0))
        dt = float(self.cfg.sim.dt)
        max_thruster_force = float(self.cfg.task.get("pypose_max_thruster_force", self.cfg.task.get("mpc_max_thruster_force", 40.0)))

        if mode in ("waypoint", "moving_waypoint", "wp"):
            from marinegym.controllers.pypose_mpc_controller import PyPoseMPCController

            uav_params = dict(uav_params)
            uav_params.update(
                {
                    "mpc_q_pos": float(self.cfg.task.get("pypose_wp_mpc_q_pos", 50.0)),
                    "mpc_q_quat": float(self.cfg.task.get("pypose_wp_mpc_q_quat", 30.0)),
                    "mpc_q_vel": float(self.cfg.task.get("pypose_wp_mpc_q_vel", 2.0)),
                    "mpc_q_omega": float(self.cfg.task.get("pypose_wp_mpc_q_omega", 0.5)),
                    "mpc_r_u": float(self.cfg.task.get("pypose_wp_mpc_r_u", self.cfg.task.get("mpc_r_u", 0.01))),
                }
            )
            self.controller = PyPoseMPCController(
                uav_params=uav_params,
                dt=dt,
                horizon=horizon,
                batch_size=int(self.cfg.env.num_envs),
                ilqr_iters=ilqr_iters,
                ilqr_reg=ilqr_reg,
                terminal_weight_mult=terminal_mult,
                max_thruster_force=max_thruster_force,
            )
            self.controller.to(self.device)
            return

        from marinegym.controllers.pypose_cylinder_orbit_mpc_controller import PyPoseCylinderOrbitMPCController

        uav_params = dict(uav_params)
        uav_params.update(
            {
                "mpc_q_radial": float(self.cfg.task.get("mpc_q_radial", 50.0)),
                "mpc_q_z": float(self.cfg.task.get("mpc_q_z", 30.0)),
                "mpc_q_tan": float(self.cfg.task.get("mpc_q_tan", 10.0)),
                "mpc_q_radial_speed": float(self.cfg.task.get("mpc_q_radial_speed", 5.0)),
                "mpc_q_heading": float(self.cfg.task.get("mpc_q_heading", 30.0)),
                "mpc_q_roll": float(self.cfg.task.get("mpc_q_roll", 30.0)),
                "mpc_q_pitch": float(self.cfg.task.get("mpc_q_pitch", 10.0)),
                "mpc_q_wxy": float(self.cfg.task.get("mpc_q_wxy", 0.5)),
                "mpc_r_u": float(self.cfg.task.get("mpc_r_u", 0.01)),
            }
        )
        self.controller = PyPoseCylinderOrbitMPCController(
            uav_params=uav_params,
            dt=dt,
            horizon=horizon,
            batch_size=int(self.cfg.env.num_envs),
            ilqr_iters=ilqr_iters,
            ilqr_reg=ilqr_reg,
            terminal_weight_mult=terminal_mult,
            max_thruster_force=max_thruster_force,
            q_radial=float(self.cfg.task.get("mpc_q_radial", 50.0)),
            q_z=float(self.cfg.task.get("mpc_q_z", 30.0)),
            q_tan=float(self.cfg.task.get("mpc_q_tan", 10.0)),
            q_radial_speed=float(self.cfg.task.get("mpc_q_radial_speed", 5.0)),
            q_heading=float(self.cfg.task.get("mpc_q_heading", 30.0)),
            q_roll=float(self.cfg.task.get("mpc_q_roll", 30.0)),
            q_pitch=float(self.cfg.task.get("mpc_q_pitch", 10.0)),
            q_wxy=float(self.cfg.task.get("mpc_q_wxy", 0.5)),
            r_u=float(self.cfg.task.get("mpc_r_u", 0.01)),
        )
        self.controller.to(self.device)
