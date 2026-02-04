import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.prims.xform_prim_view import XFormPrimView
from omni.isaac.core.utils.viewports import set_camera_view

from marinegym.envs.isaac_env import AgentSpec, IsaacEnv
from marinegym.views import ArticulationView, RigidPrimView
from marinegym.utils.torch import euler_to_quaternion, quat_axis
from marinegym.utils.torch import quat_rotate, quat_mul

from tensordict.tensordict import TensorDict, TensorDictBase
from marinegym.utils.torchrl.specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from marinegym.robots.drone import UnderwaterVehicle

from ..utils import attach_payload

class Hover(IsaacEnv):
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reward_orientation_weight = float(cfg.task.get("reward_orientation_weight", 0.0))
        self.reward_orientation_scale = float(cfg.task.get("reward_orientation_scale", 1.0))
        self.time_encoding = cfg.task.time_encoding
        # If true, append target quaternion to observation (for pose-tracking policies).
        self.include_target_quat_in_obs = bool(cfg.task.get("include_target_quat_in_obs", False))
        self.mode = cfg.mode
        self.disturbances = cfg.task.get("disturbances", {})
        self.enable_payload = self.disturbances[self.mode]['payload']['enable_payload']
        self.enable_flow = self.disturbances[self.mode]['flow']['enable_flow']
        self.max_flow_velocity = self.disturbances[self.mode]['flow']['max_flow_velocity']
        self.flow_velocity_gaussian_noise = self.disturbances[self.mode]['flow']['flow_velocity_gaussian_noise']

        super().__init__(cfg, headless)

        self.drone.initialize()
        self._init_viewer_follow_camera()
        if self.enable_payload:
            payload_cfg = self.disturbances[self.mode]['payload']
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()

        # Target visualization is just a marker prim; do not wrap as an ArticulationView even if
        # the drone itself is an articulation.
        self.target_vis = XFormPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        if hasattr(self.target_vis, "initialize"):
            self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        init_pos_min = torch.tensor(cfg.task.get("init_pos_min", [-2.5, -2.5, 1.5]), device=self.device)
        init_pos_max = torch.tensor(cfg.task.get("init_pos_max", [2.5, 2.5, 2.5]), device=self.device)
        self.init_pos_dist = D.Uniform(
            init_pos_min,
            init_pos_max,
        )
        init_rpy_min = torch.tensor(cfg.task.get("init_rpy_min", [-0.2, -0.2, 0.0]), device=self.device) * torch.pi
        init_rpy_max = torch.tensor(cfg.task.get("init_rpy_max", [0.2, 0.2, 2.0]), device=self.device) * torch.pi
        self.init_rpy_dist = D.Uniform(
            init_rpy_min,
            init_rpy_max,
        )

        target_rpy_min_cfg = cfg.task.get("target_rpy_min", None)
        target_rpy_max_cfg = cfg.task.get("target_rpy_max", None)
        if target_rpy_min_cfg is not None and target_rpy_max_cfg is not None:
            target_rpy_min = torch.tensor(target_rpy_min_cfg, device=self.device) * torch.pi
            target_rpy_max = torch.tensor(target_rpy_max_cfg, device=self.device) * torch.pi
            self.target_rpy_dist = D.Uniform(target_rpy_min, target_rpy_max)
            self._fixed_target_rpy = None
        else:
            if bool(cfg.task.get("random_target_yaw", True)):
                self.target_rpy_dist = D.Uniform(
                    torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
                    torch.tensor([0.0, 0.0, 2.0], device=self.device) * torch.pi,
                )
                self._fixed_target_rpy = None
            else:
                self.target_rpy_dist = None
                self._fixed_target_rpy = torch.zeros(3, device=self.device)

        # target position (local frame of each env)
        self.random_target_pos = bool(cfg.task.get("random_target_pos", False))
        if self.random_target_pos:
            tmin = torch.tensor(cfg.task.get("target_pos_min", [-1.0, -1.0, 1.5]), device=self.device)
            tmax = torch.tensor(cfg.task.get("target_pos_max", [1.0, 1.0, 2.5]), device=self.device)
            self.target_pos_dist = D.Uniform(tmin, tmax)
            self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        else:
            self.target_pos = torch.tensor([[0.0, 0.0, 2.]], device=self.device).expand(self.num_envs, 1, 3).clone()

        # termination thresholds (override-friendly for MPC debugging)
        self.terminate_distance = float(cfg.task.get("terminate_distance", 4.0))
        self.terminate_min_z = float(cfg.task.get("terminate_min_z", 0.2))
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_rot = torch.zeros(self.num_envs, 1, 4, device=self.device)
        self.target_rot[..., 0] = 1.0
        self.alpha = 0.8

    def _init_viewer_follow_camera(self):
        self._viewer_follow_enabled = False
        self._viewer_follow_view = None
        self._viewer_follow_offset = None
        self._viewer_follow_forward = None
        self._viewer_follow_distance = None

        viewer_cfg = self.cfg.get("viewer", {})
        follow_cfg = viewer_cfg.get("follow", {}) if viewer_cfg is not None else {}
        if not follow_cfg:
            return

        enabled = follow_cfg.get("enable", follow_cfg.get("enabled", False))
        if not bool(enabled):
            return

        link = str(follow_cfg.get("link", "base_link"))
        offset = follow_cfg.get("offset", [0.25, 0.0, 0.05])
        forward = follow_cfg.get("forward", [1.0, 0.0, 0.0])
        distance = float(follow_cfg.get("distance", 2.0))

        follow_view = None
        if link in ("base", "base_link"):
            follow_view = getattr(self.drone, "base_link", None)
        else:
            if link.startswith("/"):
                prim_paths_expr = link
            else:
                prim_paths_expr = f"/World/envs/env_*/{self.drone.name}_*/{link}"
            try:
                follow_view = RigidPrimView(
                    prim_paths_expr,
                    reset_xform_properties=False,
                    shape=(-1, self.drone.n),
                )
                follow_view.initialize()
            except Exception as e:
                print(
                    f"[Hover] viewer.follow.link='{link}' prim lookup failed; "
                    f"fallback to base_link. ({type(e).__name__}: {e})"
                )
                follow_view = getattr(self.drone, "base_link", None)

        if follow_view is None:
            return

        self._viewer_follow_view = follow_view
        self._viewer_follow_offset = torch.tensor(offset, device=self.device, dtype=torch.float32)
        self._viewer_follow_forward = torch.tensor(forward, device=self.device, dtype=torch.float32)
        self._viewer_follow_distance = distance
        self._viewer_follow_enabled = True
        self._update_viewer_follow_camera()

    def _update_viewer_follow_camera(self):
        if not getattr(self, "_viewer_follow_enabled", False):
            return
        if not self.enable_viewport:
            return
        if self._viewer_follow_view is None:
            return

        pos_w, quat_w = self._viewer_follow_view.get_world_poses(clone=True)
        env_idx = int(self.central_env_idx)
        agent_idx = 0
        pos = pos_w[env_idx, agent_idx]
        quat = quat_w[env_idx, agent_idx]

        eye = pos + quat_rotate(quat.unsqueeze(0), self._viewer_follow_offset.unsqueeze(0)).squeeze(0)
        forward = quat_rotate(quat.unsqueeze(0), self._viewer_follow_forward.unsqueeze(0)).squeeze(0)
        target = eye + forward * self._viewer_follow_distance
        set_camera_view(
            eye=eye.detach().cpu().numpy(),
            target=target.detach().cpu().numpy(),
        )

    def _post_sim_step(self, tensordict: TensorDictBase):
        self._update_viewer_follow_camera()

    def _design_scene(self):
        import marinegym.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import UsdGeom

        drone_model_cfg = self.cfg.task.drone_model
        is_articulation = drone_model_cfg.get("is_articulation", None)
        if isinstance(is_articulation, str):
            low = is_articulation.strip().lower()
            if low in ("true", "1", "yes", "y", "t"):
                is_articulation = True
            elif low in ("false", "0", "no", "n", "f"):
                is_articulation = False
            else:
                raise ValueError(f"task.drone_model.is_articulation must be bool. got {is_articulation!r}")
        if is_articulation is None:
            # BlueROVHeavy는 (USD에 articulation root가 있다면) articulation로 스폰하는 편이
            # /{robot_root}/... 형태의 prim 구조를 유지하기 쉽습니다.
            if str(drone_model_cfg.name) in ("BlueROVHeavy", "bluerovheavy"):
                is_articulation = True
        self.drone, self.controller = UnderwaterVehicle.make(
            drone_model_cfg.name,
            drone_model_cfg.controller,
            is_articulation=is_articulation,
        )
        
        # Target visualization: use a simple marker prim (not a full robot USD) so it is always visible
        # and does not inherit complex physics/instancing behaviors.
        target_root = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            prim_type="Xform",
            translation=(0.0, 0.0, 2.0),
        )
        marker_path = "/World/envs/env_0/target/marker"
        marker = prim_utils.create_prim(
            prim_path=marker_path,
            prim_type="Sphere",
            translation=(0.0, 0.0, 0.0),
            attributes={"radius": float(self.cfg.task.get("target_marker_radius", 0.10))},
        )
        try:
            gprim = UsdGeom.Gprim(marker)
            if gprim and gprim.GetPrim().IsValid():
                # Bright red marker.
                gprim.CreateDisplayColorAttr().Set([(1.0, 0.1, 0.1)])
        except Exception:
            pass

        kit_utils.set_nested_collision_properties(
            target_root.GetPath(),
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_root.GetPath(),
            rigid_body_enabled=False,
            disable_gravity=True,
        )

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]
        if self.enable_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        # Observation is: rpos(3) + drone_state[3:](drone_state_dim-3) + rheading(3)
        # => drone_state_dim + 3
        observation_dim = drone_state_dim + 3
        if getattr(self, "include_target_quat_in_obs", False):
            observation_dim += 4

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
        # Optional: append (cylinder_center - target_pos) for cylinder-orbit tasks.
        if bool(self.cfg.task.get("include_cylinder_rel_in_obs", False)):
            observation_dim += 3

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        if self.enable_flow:
            self.drone.set_flow_velocities(env_ids, self.max_flow_velocity, self.flow_velocity_gaussian_noise)
        self.drone._reset_idx(env_ids, self.training)

        pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot_delta = euler_to_quaternion(rpy)

        # Optional model-specific rotation offset (in multiples of pi, like init_rpy_*).
        # Use this to correct assets that are authored with a different up-axis (e.g., -90deg around X).
        model_rpy_offset_cfg = self.cfg.task.get("model_rpy_offset", None)
        if model_rpy_offset_cfg is None:
            model_rpy_offset_cfg = getattr(self.drone, "model_rpy_offset_pi", [0.0, 0.0, 0.0])
        model_rpy_offset = torch.tensor(model_rpy_offset_cfg, device=self.device, dtype=rpy.dtype) * torch.pi
        rot_offset = euler_to_quaternion(model_rpy_offset.view(1, 1, 3)).expand_as(rot_delta)

        init_pose_mode = str(self.cfg.task.get("init_pose_mode", "absolute")).strip().lower()
        if init_pose_mode not in ("absolute", "delta_usd"):
            raise ValueError(f"task.init_pose_mode must be 'absolute' or 'delta_usd'. got {init_pose_mode!r}")

        # Base orientation captured right after spawn (USD-authored transform).
        base_rot = None
        if init_pose_mode == "delta_usd":
            try:
                _, base_rot_all = self.init_poses
                base_rot = base_rot_all[env_ids]
            except Exception:
                base_rot = None

        if base_rot is not None and base_rot.shape == rot_delta.shape:
            rot = quat_mul(base_rot, quat_mul(rot_offset, rot_delta))
        else:
            rot = quat_mul(rot_offset, rot_delta)

        # Some assets (notably BlueROVHeavy) can appear upside-down if their local-up axis is inverted
        # relative to the simulator's world-up. If the current orientation makes the body +Z point
        # downward, flip 180deg around the body X axis to restore uprightness.
        if str(self.cfg.task.drone_model.name) in ("BlueROVHeavy", "bluerovheavy"):
            up_w = quat_axis(rot.squeeze(1), axis=2).unsqueeze(1)
            inverted = up_w[..., 2] < 0.0
            if bool(inverted.any()):
                flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device, dtype=rot.dtype).view(1, 1, 4)
                rot_flipped = quat_mul(rot, flip.expand_as(rot))
                rot = torch.where(inverted.unsqueeze(-1), rot_flipped, rot)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self._update_viewer_follow_camera()

        if bool(self.cfg.task.get("debug_init_pose", False)) and not getattr(self, "_debug_init_pose_printed", False):
            try:
                env0 = int(env_ids[0].item())
                print(
                    "[Hover] init_pose debug:",
                    f"init_pose_mode={init_pose_mode}",
                    f"model_rpy_offset(pi)={model_rpy_offset_cfg}",
                    f"sampled_rpy(pi)={(rpy[0,0] / torch.pi).detach().cpu().tolist()}",
                )
                if base_rot is not None:
                    print(f"[Hover] base_rot(wxyz)={base_rot[0,0].detach().cpu().tolist()}")
                print(f"[Hover] final_rot(wxyz)={rot[0,0].detach().cpu().tolist()}")
            except Exception:
                pass
            self._debug_init_pose_printed = True

        if self.enable_payload:
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device),
                env_indices=env_ids, joint_indices=joint_indices)

            payload_mass = self.payload_mass_dist.sample(env_ids.shape+(1,)) * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        # randomize target position (per-env, local frame)
        if self.random_target_pos:
            target_pos = self.target_pos_dist.sample((*env_ids.shape, 1))
            self.target_pos[env_ids] = target_pos
        target_pos_world = self.target_pos[env_ids] + self.envs_positions[env_ids].unsqueeze(1)
        # Isaac view classes have slightly different kwarg names (indices vs env_indices).
        try:
            self.target_vis.set_world_poses(
                positions=target_pos_world.squeeze(1),
                indices=env_ids,
            )
        except TypeError:
            self.target_vis.set_world_poses(positions=target_pos_world.squeeze(1), env_indices=env_ids)

        if self.target_rpy_dist is not None:
            target_rpy = self.target_rpy_dist.sample((*env_ids.shape, 1))
        else:
            target_rpy = self._fixed_target_rpy.view(1, 1, 3).expand(*env_ids.shape, 1, 3)
        target_rot = euler_to_quaternion(target_rpy)
        self.target_rot[env_ids] = target_rot
        self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)
        # Visualize the actual target orientation for the envs being reset.
        target_rot_world = target_rot
        try:
            self.target_vis.set_world_poses(
                orientations=target_rot_world[:, 0],
                indices=env_ids,
            )
        except TypeError:
            self.target_vis.set_world_poses(orientations=target_rot_world[:, 0], env_indices=env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = torch.abs(self.drone.apply_action(actions))

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

        # relative position and heading
        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        obs = [self.rpos, self.drone_state[..., 3:], self.rheading]
        if self.include_target_quat_in_obs:
            obs.append(self.target_rot)
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # pose reward
        pos_error = torch.norm(self.rpos, dim=-1)
        heading_alignment = torch.sum(self.drone.heading * self.target_heading, dim=-1)

        distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)

        reward_pose = 0.5 * 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        # orientation (full quaternion) tracking reward, if enabled.
        reward_ori = 0.0
        if self.reward_orientation_weight > 0.0:
            q = self.drone_state[..., 3:7]
            qt = self.target_rot
            # align sign (q ~ -q)
            dot = torch.sum(q * qt, dim=-1, keepdim=True)
            qt = torch.where(dot < 0.0, -qt, qt)
            dot = torch.sum(q * qt, dim=-1).clamp(-1.0, 1.0)
            # 1 - dot^2 is a smooth orientation error in [0,1]
            ori_err = 1.0 - dot * dot
            reward_ori = self.reward_orientation_weight * 1.0 / (
                1.0 + torch.square(self.reward_orientation_scale * ori_err)
            )
        # uprightness
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        # spin reward
        spinnage = torch.square(self.drone.vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        assert reward_pose.shape == reward_up.shape == reward_spin.shape
        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            + reward_ori
            + reward_effort
            + reward_action_smoothness
        )

        misbehave = (self.drone.pos[..., 2] < self.terminate_min_z) | (distance > self.terminate_distance)
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["pos_error"].lerp_(pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
