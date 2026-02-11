from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from marinegym.utils.torch import normalize, quaternion_to_rotation_matrix


def wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    two_pi = float(2.0 * math.pi)
    return (theta + math.pi).remainder(two_pi) - math.pi


def _quat_to_roll_pitch(q_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return roll, pitch from quaternion (wxyz)."""
    q = normalize(q_wxyz)
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    return roll, pitch


@dataclass(frozen=True)
class OrbitPPORewardCfg:
    # Gaussian-like shaping sigmas (in meters / radians).
    sigma_r: float = 0.20
    sigma_z: float = 0.20
    sigma_roll: float = 0.35
    sigma_pitch: float = 0.35

    # Progress shaping.
    theta_scale: float = float(5.0 * math.pi / 180.0)  # 5 deg
    v_tan_ref: float = 0.5  # m/s
    progress_mode: str = "theta"  # theta|vtan|both
    yaw_offset: float = 0.0  # radians (rotate desired heading in XY)

    # Progress gating (encourage "look at cylinder while orbiting").
    # - none: no gating
    # - mul: gate = r_heading^gate_heading_power * r_radius^gate_radius_power
    # - threshold: gate = 1 if r_heading>gate_heading_min AND r_radius>gate_radius_min else 0
    progress_gate_mode: str = "none"  # none|mul|threshold
    gate_heading_power: float = 2.0
    gate_radius_power: float = 1.0
    gate_heading_min: float = 0.8
    gate_radius_min: float = 0.8

    # Positive term weights.
    w_r: float = 1.0
    w_z: float = 1.0
    w_h: float = 0.5
    w_p: float = 0.5
    w_a: float = 0.25
    # Penalize progress in the wrong (opposite) orbit direction.
    w_back: float = 0.0

    # Tangential-speed penalty (discourage "slow/stationary" local optima).
    # penalty_vtan_shortfall = r_radius * ((max(0, v_tan_ref - v_tan) / v_tan_ref)^2 clipped)
    w_vtan_shortfall: float = 0.0

    # Penalty weights.
    w_u: float = 0.05
    w_du: float = 0.10
    w_w: float = 0.02
    w_vrad: float = 0.0
    v_rad_ref: float = 0.3  # m/s (for radial-speed penalty scaling)

    # Termination thresholds / penalties.
    r_min: float = 0.3
    r_max: float = 3.0
    rp_max_deg: float = 45.0
    terminate_penalty: float = -2.0

    # Reward clipping.
    clip_min: float = -5.0
    clip_max: float = 5.0


def compute_orbit_ppo_reward_terms(
    *,
    p_world: torch.Tensor,  # (E, 3)
    q_world_wxyz: torch.Tensor,  # (E, 4)
    v_body: torch.Tensor,  # (E, 3)
    w_body: torch.Tensor,  # (E, 3)
    c_world: torch.Tensor,  # (E, 3)
    r_target: torch.Tensor,  # (E, 1) or scalar
    z_target: torch.Tensor,  # (E, 1) or scalar
    orbit_direction: torch.Tensor,  # (E, 1) or scalar (+1 CCW, -1 CW)
    dt: float,
    u_cmd: torch.Tensor,  # (E, nu)
    u_prev: torch.Tensor,  # (E, nu)
    theta_prev: torch.Tensor,  # (E, 1)
    cfg: OrbitPPORewardCfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    PPO-shaped orbit reward terms.

    Coordinate/sign convention:
      - theta = atan2(y-cy, x-cx) increases for CCW motion in the standard XY plane.
      - `orbit_direction` should be +1 for CCW, -1 for CW.
      - We define progress as `orbit_direction * wrap_to_pi(theta - theta_prev)` so that it is
        positive when motion matches the desired direction. If your asset/world axes are different,
        verify this sign by printing `delta_theta` for known CW/CCW motion.
    """
    eps = 1e-6
    dtype = p_world.dtype
    device = p_world.device

    p_xy = p_world[:, 0:2]
    c_xy = c_world[:, 0:2]
    rel_xy = p_xy - c_xy  # center -> robot (XY)
    dist_xy = torch.linalg.norm(rel_xy, dim=-1, keepdim=True).clamp_min(eps)  # (E, 1)

    e_r = dist_xy - r_target.to(device=device, dtype=dtype)
    e_z = (p_world[:, 2:3] - z_target.to(device=device, dtype=dtype))

    sigma_r = max(float(cfg.sigma_r), eps)
    sigma_z = max(float(cfg.sigma_z), eps)
    r_radius = torch.exp(-((e_r.abs() / sigma_r).square()))
    r_depth = torch.exp(-((e_z.abs() / sigma_z).square()))

    # Heading alignment: project body forward to XY and compare with (center - pos) direction.
    R = quaternion_to_rotation_matrix(normalize(q_world_wxyz))  # (E, 3, 3)
    f_world = R[:, :, 0]  # body x-axis in world
    f_xy = normalize(torch.cat([f_world[:, 0:2], torch.zeros((f_world.shape[0], 1), device=device, dtype=dtype)], dim=-1))[:, 0:2]

    to_center_xy = normalize(
        torch.cat([c_xy - p_xy, torch.zeros((p_world.shape[0], 1), device=device, dtype=dtype)], dim=-1)
    )[:, 0:2]
    # Apply yaw_offset to the desired facing direction (accounts for model forward-axis mismatch).
    yaw_offset = float(getattr(cfg, "yaw_offset", 0.0))
    if abs(yaw_offset) > 1e-12:
        co = math.cos(yaw_offset)
        so = math.sin(yaw_offset)
        to_center_xy = torch.stack(
            [
                co * to_center_xy[:, 0] - so * to_center_xy[:, 1],
                so * to_center_xy[:, 0] + co * to_center_xy[:, 1],
            ],
            dim=-1,
        )
    cos_yaw = (f_xy * to_center_xy).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    r_heading = 0.5 * (cos_yaw + 1.0)

    # Progress reward (clockwise/CCW).
    theta = torch.atan2(rel_xy[:, 1], rel_xy[:, 0]).view(-1, 1)  # (E, 1)
    dtheta = wrap_to_pi(theta - theta_prev.to(device=device, dtype=dtype))
    dir_sign = orbit_direction.to(device=device, dtype=dtype).view(-1, 1)

    # theta-based progress.
    theta_scale = max(float(cfg.theta_scale), eps)
    progress_theta = (dir_sign * dtheta / theta_scale).clamp(0.0, 1.0)
    progress_theta_back = (-dir_sign * dtheta / theta_scale).clamp(0.0, 1.0)

    # tangential-speed-based progress.
    v_world = torch.einsum("bij,bj->bi", R, v_body.to(device=device, dtype=dtype))
    v_xy = v_world[:, 0:2]
    radial_hat_xy = rel_xy / dist_xy
    # CCW tangent is zhat x radial = (-ry, rx). Multiply by dir_sign to get desired direction.
    tan_hat_xy = torch.cat([-radial_hat_xy[:, 1:2], radial_hat_xy[:, 0:1]], dim=-1) * dir_sign
    v_tan = (v_xy * tan_hat_xy).sum(dim=-1, keepdim=True)
    v_tan_ref = max(float(cfg.v_tan_ref), eps)
    progress_vtan = (v_tan / v_tan_ref).clamp(0.0, 1.0)
    progress_vtan_back = (-v_tan / v_tan_ref).clamp(0.0, 1.0)

    # Tangential-speed shortfall penalty (applied only when v_tan < v_tan_ref).
    v_tan_err = (v_tan - v_tan_ref).abs()
    vtan_shortfall = ((v_tan_ref - v_tan) / v_tan_ref).clamp(min=0.0)
    # Clip to keep the penalty bounded even if v_tan goes strongly negative.
    penalty_vtan_shortfall_raw = vtan_shortfall.square().clamp(0.0, 4.0)
    # Gate by radius-tracking quality: if we're far from the desired radius, don't over-penalize tangential speed.
    penalty_vtan_shortfall = penalty_vtan_shortfall_raw * r_radius

    mode = str(cfg.progress_mode).strip().lower()
    if mode == "vtan":
        r_progress = progress_vtan
        r_progress_back = progress_vtan_back
    elif mode == "both":
        r_progress = 0.5 * (progress_theta + progress_vtan)
        r_progress_back = 0.5 * (progress_theta_back + progress_vtan_back)
    else:
        r_progress = progress_theta
        r_progress_back = progress_theta_back

    # Gate progress so the policy learns to keep looking at the cylinder while orbiting.
    r_progress_raw = r_progress
    gate_mode = str(getattr(cfg, "progress_gate_mode", "none")).strip().lower()
    if gate_mode == "threshold":
        progress_gate = ((r_heading > float(cfg.gate_heading_min)) & (r_radius > float(cfg.gate_radius_min))).to(dtype=dtype)
    elif gate_mode == "mul":
        progress_gate = (
            r_heading.clamp(0.0, 1.0).pow(float(cfg.gate_heading_power))
            * r_radius.clamp(0.0, 1.0).pow(float(cfg.gate_radius_power))
        )
    else:
        progress_gate = torch.ones_like(r_progress_raw)
    r_progress = r_progress_raw * progress_gate

    # Attitude stability.
    roll, pitch = _quat_to_roll_pitch(q_world_wxyz)
    sigma_roll = max(float(cfg.sigma_roll), eps)
    sigma_pitch = max(float(cfg.sigma_pitch), eps)
    r_att = torch.exp(-((roll.abs() / sigma_roll).square() + (pitch.abs() / sigma_pitch).square())).view(-1, 1)

    # Penalties.
    penalty_u = (u_cmd.to(device=device, dtype=dtype).square()).mean(dim=-1, keepdim=True)
    penalty_du = ((u_cmd.to(device=device, dtype=dtype) - u_prev.to(device=device, dtype=dtype)).square()).mean(dim=-1, keepdim=True)
    penalty_omega = (w_body.to(device=device, dtype=dtype).square()).sum(dim=-1, keepdim=True)
    v_rad = (v_xy * radial_hat_xy).sum(dim=-1, keepdim=True)
    v_rad_ref = max(float(getattr(cfg, "v_rad_ref", 0.3)), eps)
    penalty_v_rad = (v_rad / v_rad_ref).square().clamp(0.0, 1.0)

    # Weighted sum (positive) - penalties.
    reward = (
        float(cfg.w_r) * r_radius
        + float(cfg.w_z) * r_depth
        + float(cfg.w_h) * r_heading
        + float(cfg.w_p) * r_progress
        + float(cfg.w_a) * r_att
        - float(cfg.w_back) * r_progress_back
        - float(getattr(cfg, "w_vtan_shortfall", 0.0)) * penalty_vtan_shortfall
        - float(cfg.w_u) * penalty_u
        - float(cfg.w_du) * penalty_du
        - float(cfg.w_w) * penalty_omega
        - float(getattr(cfg, "w_vrad", 0.0)) * penalty_v_rad
    )

    # Termination conditions (computed here; env decides how to combine with other done logic).
    rp_max = float(cfg.rp_max_deg) * math.pi / 180.0
    bad_radius = (dist_xy < float(cfg.r_min)) | (dist_xy > float(cfg.r_max))
    bad_rp = (roll.abs().view(-1, 1) > rp_max) | (pitch.abs().view(-1, 1) > rp_max)
    terminate = bad_radius | bad_rp

    if float(cfg.terminate_penalty) != 0.0:
        reward = reward + float(cfg.terminate_penalty) * terminate.to(dtype=dtype)

    reward = reward.clamp(min=float(cfg.clip_min), max=float(cfg.clip_max))

    terms = {
        "reward": reward,
        "r_radius": r_radius,
        "r_depth": r_depth,
        "r_heading": r_heading,
        "cos_yaw": cos_yaw,
        "r_progress": r_progress,
        "r_progress_raw": r_progress_raw,
        "progress_gate": progress_gate,
        "r_progress_theta": progress_theta,
        "r_progress_vtan": progress_vtan,
        "r_progress_back": r_progress_back,
        "r_progress_back_theta": progress_theta_back,
        "r_progress_back_vtan": progress_vtan_back,
        "theta": theta,
        "delta_theta": dtheta,
        "v_tan": v_tan,
        "v_tan_err": v_tan_err,
        "v_tan_shortfall": vtan_shortfall,
        "v_rad": v_rad,
        "dist_xy": dist_xy,
        "e_r": e_r,
        "e_z": e_z,
        "roll": roll.view(-1, 1),
        "pitch": pitch.view(-1, 1),
        "r_att": r_att,
        "penalty_v_tan_shortfall": penalty_vtan_shortfall,
        "penalty_u": penalty_u,
        "penalty_du": penalty_du,
        "penalty_omega": penalty_omega,
        "penalty_v_rad": penalty_v_rad,
        "terminate_orbit_ppo": terminate.to(dtype=dtype),
    }
    return reward, terms
