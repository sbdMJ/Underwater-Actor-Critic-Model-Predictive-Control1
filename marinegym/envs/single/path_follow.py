import math

import torch

from marinegym.envs.single.hover import Hover
from marinegym.envs.isaac_env import AgentSpec
from marinegym.utils.torch import quat_rotate_inverse
from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec


# ~/isaac410/python.sh scripts/train.py   task=PathFollow algo=ppo headless=false enable_livestream=false



def _normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


class PathFollow(Hover):
    """
    PPO용 3D Path Following 태스크.

    - 액션: thruster(로터) 커맨드 [-1, 1] (Hover와 동일)
    - path: 각 env마다 랜덤 3D 폐곡선(loop polyline)
    - 관측: closest/ lookahead 벡터(Body-frame) + path tangent(Body-frame) + 진행도 + Hover drone_state(3:...)
    - 보상: cross-track(closest 거리) + 진행도(progress delta) + 접선 정렬(velocity-tangent) + effort/smoothness
    """

    def __init__(self, cfg, headless):
        # IsaacEnv.__init__ 내부에서 _set_specs()가 호출되므로,
        # spec에 영향을 주는 값은 super() 호출 전에 세팅해야 합니다.
        self.include_progress_sincos = bool(cfg.task.get("obs_progress_sincos", True))
        self.track_mode = str(cfg.task.get("track_mode", "trajectory")).lower()
        self.traj_future_points = int(cfg.task.get("traj_future_points", 3))

        super().__init__(cfg, headless)

        # visualization (debug_draw)
        self.draw_path = bool(cfg.task.get("draw_path", False))
        self.draw_path_mode = str(cfg.task.get("draw_path_mode", "central"))
        self.draw_path_color = tuple(cfg.task.get("draw_path_color", [1.0, 0.2, 0.2, 1.0]))
        self.draw_path_size = float(cfg.task.get("draw_path_size", 2.0))

        self.path_num_points = int(cfg.task.get("path_num_points", 96))
        self.path_harmonics = int(cfg.task.get("path_harmonics", 5))
        self.path_radius_xy = float(cfg.task.get("path_radius_xy", 1.8))
        self.path_z_amp = float(cfg.task.get("path_z_amp", 0.6))
        self.path_z_min = float(cfg.task.get("path_z_min", 0.6))
        self.path_z_max = float(cfg.task.get("path_z_max", 3.2))
        self.path_lookahead_steps = int(cfg.task.get("path_lookahead_steps", 6))
        self.traj_steps_per_point = int(cfg.task.get("traj_steps_per_point", 4))
        self.traj_lookahead_points = int(cfg.task.get("traj_lookahead_points", 1))
        self.traj_target_radius = float(cfg.task.get("traj_target_radius", 0.5))

        # reward weights
        self.r_cte_w = float(cfg.task.get("reward_cte_weight", 1.0))
        self.r_cte_scale = float(cfg.task.get("reward_cte_scale", 2.0))
        self.r_progress_w = float(cfg.task.get("reward_progress_weight", 0.7))
        self.r_align_w = float(cfg.task.get("reward_align_weight", 0.2))
        self.r_heading_w = float(cfg.task.get("reward_heading_weight", 0.2))
        self.r_heading_scale = float(cfg.task.get("reward_heading_scale", 2.0))
        self.r_roll_rate_w = float(cfg.task.get("reward_roll_rate_weight", 0.2))
        self.r_roll_rate_scale = float(cfg.task.get("reward_roll_rate_scale", 1.0))
        self.r_forward_speed_w = float(cfg.task.get("reward_forward_speed_weight", 0.3))
        self.r_forward_speed_scale = float(cfg.task.get("reward_forward_speed_scale", 1.0))
        self.r_stop_w = float(cfg.task.get("reward_stop_penalty_weight", 0.2))
        self.r_stop_speed_thresh = float(cfg.task.get("reward_stop_speed_threshold", 0.1))
        self.r_align_speed_min = float(cfg.task.get("reward_align_speed_min", 0.05))

        # termination thresholds
        self.terminate_cte = float(cfg.task.get("terminate_cte", 5.0))

        # observation design
        # (already set before super().__init__)

        # precompute Fourier basis (device-aware)
        P = max(32, self.path_num_points)
        H = max(1, self.path_harmonics)
        t = torch.linspace(0.0, 2.0 * math.pi, P, device=self.device, dtype=torch.float32, requires_grad=False)
        k = torch.arange(1, H + 1, device=self.device, dtype=torch.float32, requires_grad=False)
        self._fourier_sin = torch.sin(k[:, None] * t[None, :])  # (H, P)
        self._fourier_cos = torch.cos(k[:, None] * t[None, :])  # (H, P)
        self._fourier_w = (1.0 / (k**1.25)).to(torch.float32)  # (H,)

        # per-env path buffers (env-frame)
        self.path_points = torch.zeros(self.num_envs, P, 3, device=self.device, dtype=torch.float32)
        self.path_tangent = torch.zeros(self.num_envs, P, 3, device=self.device, dtype=torch.float32)
        self.path_seg_len = torch.zeros(self.num_envs, P, device=self.device, dtype=torch.float32)
        self.path_s = torch.zeros(self.num_envs, P + 1, device=self.device, dtype=torch.float32)
        self.path_total_len = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.path_lookahead_dist = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._closest_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._prev_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._progress_s = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)

        self.alpha = float(cfg.task.get("stats_ema_alpha", 0.8))

    def _should_draw(self) -> bool:
        # headless + no livestream에서는 debug draw가 의미 없음
        if bool(self.cfg.get("headless", False)) and (not bool(self.cfg.get("enable_livestream", False))):
            return False
        return bool(self.draw_path) and hasattr(self, "debug_draw")

    def _draw_env_ids(self) -> torch.Tensor:
        mode = str(self.draw_path_mode).lower()
        if mode == "all":
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if mode == "env0":
            return torch.zeros(1, device=self.device, dtype=torch.long)
        # default: central
        idx = int(getattr(self, "central_env_idx", 0))
        return torch.as_tensor([idx], device=self.device, dtype=torch.long)

    @torch.no_grad()
    def _redraw_paths(self) -> None:
        if not self._should_draw():
            return

        env_ids = self._draw_env_ids()
        if env_ids.numel() == 0:
            return

        try:
            self.debug_draw.clear()
        except Exception:
            return

        P = int(self.path_points.shape[1])
        for env_i in env_ids.detach().cpu().tolist():
            pts = self.path_points[env_i]  # (P, 3) in env-frame
            pts = torch.cat([pts, pts[:1]], dim=0)  # (P+1, 3) close loop
            pts_world = pts + self.envs_positions[env_i].to(dtype=pts.dtype)
            self.debug_draw.plot(pts_world, size=self.draw_path_size, color=self.draw_path_color)

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        base_obs_dim = (drone_state_dim - 3) + 3 + 3 + 3  # drone_state[3:], r_closest, r_lookahead, tangent
        if self.track_mode == "trajectory" and self.traj_future_points > 0:
            base_obs_dim += 3 * int(self.traj_future_points)
        if getattr(self, "include_progress_sincos", True):
            base_obs_dim += 2
        else:
            base_obs_dim += 1

        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec((1, base_obs_dim), device=self.device),
                        "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec(
            {"agents": CompositeSpec({"action": self.drone.action_spec.unsqueeze(0)})}
        ).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec(
            {"agents": CompositeSpec({"reward": UnboundedContinuousTensorSpec((1, 1))})}
        ).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

        stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "cte": UnboundedContinuousTensorSpec(1),
                "progress_delta": UnboundedContinuousTensorSpec(1),
                "tangent_align": UnboundedContinuousTensorSpec(1),
                "heading_align": UnboundedContinuousTensorSpec(1),
                "roll_rate": UnboundedContinuousTensorSpec(1),
                "forward_speed": UnboundedContinuousTensorSpec(1),
                "speed": UnboundedContinuousTensorSpec(1),
                "stop_penalty": UnboundedContinuousTensorSpec(1),
                "action_smoothness": UnboundedContinuousTensorSpec(1),
            }
        ).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    @torch.no_grad()
    def _sample_loop_paths(self, env_ids: torch.Tensor, centers: torch.Tensor) -> None:
        """
        centers: (E, 3) env-frame
        path_points[path_id]: (P, 3) env-frame
        """
        env_ids = env_ids.to(device=self.device)
        centers = centers.to(device=self.device, dtype=torch.float32)
        E = int(env_ids.numel())
        P = int(self.path_points.shape[1])
        H = int(self._fourier_w.shape[0])

        # random Fourier coefficients
        ax = torch.randn(E, H, device=self.device, dtype=torch.float32) * self._fourier_w[None, :]
        bx = torch.randn(E, H, device=self.device, dtype=torch.float32) * self._fourier_w[None, :]
        ay = torch.randn(E, H, device=self.device, dtype=torch.float32) * self._fourier_w[None, :]
        by = torch.randn(E, H, device=self.device, dtype=torch.float32) * self._fourier_w[None, :]
        az = torch.randn(E, H, device=self.device, dtype=torch.float32) * self._fourier_w[None, :]
        bz = torch.randn(E, H, device=self.device, dtype=torch.float32) * self._fourier_w[None, :]

        x_raw = ax @ self._fourier_sin + bx @ self._fourier_cos  # (E, P)
        y_raw = ay @ self._fourier_sin + by @ self._fourier_cos
        z_raw = az @ self._fourier_sin + bz @ self._fourier_cos

        def _norm01(v: torch.Tensor) -> torch.Tensor:
            m = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            return v / m

        x = _norm01(x_raw) * self.path_radius_xy
        y = _norm01(y_raw) * self.path_radius_xy
        z = _norm01(z_raw) * self.path_z_amp

        pts = torch.stack([x, y, z], dim=-1) + centers[:, None, :]
        pts[..., 2].clamp_(self.path_z_min, self.path_z_max)

        nxt = torch.roll(pts, shifts=-1, dims=1)
        tan = _normalize(nxt - pts)

        self.path_points[env_ids] = pts
        self.path_tangent[env_ids] = tan

        seg_len = (nxt - pts).norm(dim=-1).clamp_min(1e-6)
        s = torch.cumsum(seg_len, dim=1)
        s = torch.cat([torch.zeros(E, 1, device=self.device, dtype=torch.float32), s], dim=1)

        self.path_seg_len[env_ids] = seg_len
        self.path_s[env_ids] = s
        self.path_total_len[env_ids] = s[:, -1:].clone()
        self.path_lookahead_dist[env_ids] = self.path_total_len[env_ids] * (
            float(self.path_lookahead_steps) / float(P)
        )

        # reset progress indices
        self._closest_idx[env_ids] = 0
        self._prev_idx[env_ids] = 0
        self._progress_s[env_ids] = 0.0
        self._redraw_paths()

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self.drone.get_state()
        centers = self.drone.pos[env_ids].squeeze(1)  # env-frame (E, 3)
        self._sample_loop_paths(env_ids, centers)
        # initialize progress to closest point at reset
        pos = self.drone.pos[env_ids].squeeze(1)
        pts = self.path_points[env_ids]
        d2 = ((pts - pos[:, None, :]) ** 2).sum(dim=-1)
        idx = torch.argmin(d2, dim=-1)
        self._closest_idx[env_ids] = idx
        self._progress_s[env_ids, 0] = self.path_s[env_ids, idx]
        self.stats[env_ids] = 0.0

    def _sample_path_at_s(self, s_query: torch.Tensor):
        s_query = s_query.view(-1)
        total_len = self.path_total_len.view(-1).clamp_min(1e-6)
        s_mod = torch.remainder(s_query, total_len)

        s_table = self.path_s
        mask = s_table <= s_mod.unsqueeze(1)
        seg_idx = mask.sum(dim=1) - 1
        seg_idx = seg_idx.clamp(0, self.path_points.shape[1] - 1)

        env_ids = torch.arange(self.num_envs, device=self.device)
        s0 = s_table[env_ids, seg_idx]
        s1 = s_table[env_ids, seg_idx + 1]
        denom = (s1 - s0).clamp_min(1e-6)
        alpha = (s_mod - s0) / denom

        p0 = self.path_points[env_ids, seg_idx]
        p1 = self.path_points[env_ids, (seg_idx + 1) % self.path_points.shape[1]]
        pos = p0 + alpha.unsqueeze(1) * (p1 - p0)
        tangent = self.path_tangent[env_ids, seg_idx]
        return pos, tangent, seg_idx

    def _compute_path_quantities(self):
        # state in env-frame
        pos = self.drone_state[..., :3].squeeze(1)  # (N, 3)
        rot = self.drone_state[..., 3:7].squeeze(1)  # (N, 4)

        future_vecs_b = None
        r_future_w = None
        if self.track_mode == "trajectory":
            P = int(self.path_points.shape[1])
            steps_per = max(1, int(self.traj_steps_per_point))
            lookahead_pts = max(1, int(self.traj_lookahead_points))
            idx = ((self.progress_buf // steps_per) % P).to(torch.long)
            idx_next = (idx + lookahead_pts) % P

            gather_idx = idx[:, None, None].expand(-1, 1, 3)
            gather_idx_next = idx_next[:, None, None].expand(-1, 1, 3)
            ref_pos = self.path_points.gather(1, gather_idx).squeeze(1)
            lookahead = self.path_points.gather(1, gather_idx_next).squeeze(1)
            tangent = _normalize(lookahead - ref_pos)

            self._closest_idx = idx
            r_closest_w = ref_pos - pos
            r_lookahead_w = lookahead - pos
            cte = r_closest_w.norm(dim=-1, keepdim=True)  # (N, 1)

            delta_idx = (idx - self._prev_idx + P) % P
            progress_delta = (delta_idx.to(torch.float32) / float(P)).unsqueeze(-1)
            reached = (cte <= self.traj_target_radius).to(cte.dtype)
            progress_delta = progress_delta * reached
            self._prev_idx = idx

            future_n = max(0, int(self.traj_future_points))
            if future_n > 0:
                offsets = torch.arange(1, future_n + 1, device=self.device)
                idx_future = (idx[:, None] + offsets[None, :]) % P
                gather_idx_future = idx_future[..., None].expand(-1, -1, 3)
                future_pts = self.path_points.gather(1, gather_idx_future)
                r_future_w = future_pts - pos[:, None, :]
        elif self.track_mode == "closest":
            P = int(self.path_points.shape[1])
            diff = self.path_points - pos[:, None, :]
            d2 = (diff * diff).sum(dim=-1)
            idx = torch.argmin(d2, dim=-1)
            idx_la = (idx + self.path_lookahead_steps) % P

            gather_idx = idx[:, None, None].expand(-1, 1, 3)
            gather_idx_la = idx_la[:, None, None].expand(-1, 1, 3)
            ref_pos = self.path_points.gather(1, gather_idx).squeeze(1)
            lookahead = self.path_points.gather(1, gather_idx_la).squeeze(1)
            tangent = self.path_tangent.gather(1, gather_idx).squeeze(1)

            self._closest_idx = idx
            r_closest_w = ref_pos - pos
            r_lookahead_w = lookahead - pos
            cte = r_closest_w.norm(dim=-1, keepdim=True)

            delta_idx = (idx - self._prev_idx + P) % P
            delta_idx = torch.where(delta_idx > (P // 2), delta_idx - P, delta_idx)
            progress_delta = (delta_idx.to(torch.float32) / float(P)).unsqueeze(-1)
            self._prev_idx = idx
        else:
            progress_s = self._progress_s.squeeze(1)
            ref_pos, tangent, seg_idx = self._sample_path_at_s(progress_s)
            lookahead_s = progress_s + self.path_lookahead_dist.squeeze(1)
            lookahead, _tan_la, _seg_la = self._sample_path_at_s(lookahead_s)
            self._closest_idx = seg_idx

            r_closest_w = ref_pos - pos
            r_lookahead_w = lookahead - pos
            cte = r_closest_w.norm(dim=-1, keepdim=True)  # (N, 1)
            progress_delta = None

        # to body-frame for observation
        r_closest_b = quat_rotate_inverse(rot, r_closest_w)
        r_lookahead_b = quat_rotate_inverse(rot, r_lookahead_w)
        tangent_b = quat_rotate_inverse(rot, tangent)
        if self.track_mode == "trajectory" and r_future_w is not None:
            n_envs, n_pts, _ = r_future_w.shape
            rot_flat = rot.repeat_interleave(n_pts, dim=0)
            future_flat = quat_rotate_inverse(rot_flat, r_future_w.reshape(-1, 3))
            future_vecs_b = future_flat.view(n_envs, n_pts, 3)

        # velocity alignment with tangent (world-frame)
        vel_w = self.drone.vel_w.squeeze(1)[..., :3]  # (N, 3)
        speed = vel_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # (N, 1)
        v_dir = vel_w / speed
        align = (v_dir * tangent).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (N, 1)
        forward_speed = (vel_w * tangent).sum(dim=-1, keepdim=True)

        if self.track_mode not in ("trajectory", "closest"):
            dt = float(self.cfg.sim.dt) * float(self.cfg.sim.substeps)
            total_len = self.path_total_len.clamp_min(1e-6)
            progress_delta = (forward_speed * dt) / total_len
            self._progress_s[:, 0] = torch.remainder(
                self._progress_s[:, 0] + forward_speed.squeeze(1) * dt, total_len.squeeze(1)
            )

        # heading alignment with tangent (world-frame)
        heading_w = self.drone.heading.squeeze(1)  # (N, 3)
        heading_align = (heading_w * tangent).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        ang_vel_w = self.drone.vel_w.squeeze(1)[..., 3:6]
        ang_vel_b = quat_rotate_inverse(rot, ang_vel_w)
        roll_rate = ang_vel_b[..., 0:1]
        return (
            r_closest_b,
            r_lookahead_b,
            tangent_b,
            cte,
            progress_delta,
            align,
            heading_align,
            roll_rate,
            forward_speed,
            speed,
            future_vecs_b,
        )

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

        (
            r_closest_b,
            r_lookahead_b,
            tangent_b,
            cte,
            progress_delta,
            align,
            heading_align,
            roll_rate,
            forward_speed,
            _speed,
            future_vecs_b,
        ) = self._compute_path_quantities()

        if self.track_mode in ("trajectory", "closest"):
            P = int(self.path_points.shape[1])
            progress_ratio = (self._closest_idx.to(torch.float32) / float(P)).unsqueeze(-1)
        else:
            progress_ratio = (self._progress_s / self.path_total_len.clamp_min(1e-6)).clamp(0.0, 1.0)
        if self.include_progress_sincos:
            phase = progress_ratio * (2.0 * math.pi)
            phase = phase.squeeze(1)
            prog_feat = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)  # (N, 2)
        else:
            prog_feat = progress_ratio  # (N, 1)

        obs_parts = [
            r_closest_b.unsqueeze(1),
            r_lookahead_b.unsqueeze(1),
        ]
        if self.track_mode == "trajectory" and future_vecs_b is not None:
            obs_parts.append(future_vecs_b.reshape(future_vecs_b.shape[0], 1, -1))
        obs_parts.extend(
            [
                tangent_b.unsqueeze(1),
                prog_feat.unsqueeze(1),
                self.drone_state[..., 3:],
            ]
        )
        obs = torch.cat(obs_parts, dim=-1)

        # keep for reward/stat computation
        self._cte = cte
        self._progress_delta = progress_delta
        self._align = align
        self._heading_align = heading_align
        self._roll_rate = roll_rate
        self._forward_speed = forward_speed
        self._speed = _speed

        return TensorDict(
            {
                "agents": {"observation": obs, "intrinsics": self.drone.intrinsics},
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        cte = getattr(self, "_cte")
        progress_delta = getattr(self, "_progress_delta")
        align = getattr(self, "_align")
        heading_align = getattr(self, "_heading_align")
        roll_rate = getattr(self, "_roll_rate")
        forward_speed = getattr(self, "_forward_speed")
        speed = getattr(self, "_speed")

        # cross-track reward: high when close to path
        r_cte = torch.exp(-self.r_cte_scale * (cte * cte))

        # progress: encourage forward movement along the loop
        r_prog = progress_delta

        # tangent alignment: only meaningful when moving
        speed = self.drone.vel_w.squeeze(1)[..., :3].norm(dim=-1, keepdim=True)
        speed_gate = (speed > self.r_align_speed_min).to(cte.dtype)
        r_align = (align + 1.0) * 0.5 * speed_gate
        heading_error = 1.0 - heading_align
        r_heading = torch.exp(-self.r_heading_scale * (heading_error * heading_error))
        r_roll_rate = torch.exp(-self.r_roll_rate_scale * (roll_rate * roll_rate))
        r_forward_speed = torch.tanh(forward_speed / self.r_forward_speed_scale)
        speed_thresh = max(1e-6, float(self.r_stop_speed_thresh))
        stop_penalty = torch.clamp(1.0 - speed / speed_thresh, min=0.0)

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        reward = (
            self.r_cte_w * r_cte
            + self.r_progress_w * r_prog
            + self.r_align_w * r_align
            + self.r_heading_w * r_heading
            + self.r_roll_rate_w * r_roll_rate
            + self.r_forward_speed_w * r_forward_speed
            - self.r_stop_w * stop_penalty
            + reward_effort
            + reward_action_smoothness
        )

        terminated = (cte > self.terminate_cte) | (self.drone.pos[..., 2] < self.terminate_min_z)  # (N, 1)
        hasnan = torch.isnan(self.drone_state).any(-1)  # (N, 1)
        terminated = terminated | hasnan  # (N, 1)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # stats EMA
        self.stats["cte"].lerp_(cte, (1 - self.alpha))
        self.stats["progress_delta"].lerp_(progress_delta, (1 - self.alpha))
        self.stats["tangent_align"].lerp_(align, (1 - self.alpha))
        self.stats["heading_align"].lerp_(heading_align, (1 - self.alpha))
        self.stats["roll_rate"].lerp_(roll_rate.abs(), (1 - self.alpha))
        self.stats["forward_speed"].lerp_(forward_speed, (1 - self.alpha))
        self.stats["speed"].lerp_(speed, (1 - self.alpha))
        self.stats["stop_penalty"].lerp_(stop_penalty, (1 - self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1 - self.alpha))
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
