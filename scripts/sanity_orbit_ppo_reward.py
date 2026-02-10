import sys
from pathlib import Path

import math

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marinegym.rewards.orbit_ppo import OrbitPPORewardCfg, compute_orbit_ppo_reward_terms, wrap_to_pi
from marinegym.utils.torch import euler_to_quaternion


def _summary(name: str, x: torch.Tensor):
    x = x.detach().cpu().float()
    print(f"{name:>18s}: shape={tuple(x.shape)} min={x.min().item(): .4f} max={x.max().item(): .4f} mean={x.mean().item(): .4f}")


def main():
    torch.manual_seed(0)
    B = 2048
    nu = 6
    device = torch.device("cpu")

    center = torch.tensor([0.0, 0.0, 1.5], device=device).view(1, 3).expand(B, 3)
    r_target = torch.full((B, 1), 1.0, device=device)
    z_target = torch.full((B, 1), 1.5, device=device)
    orbit_direction = torch.full((B, 1), -1.0, device=device)  # CW

    theta = (torch.rand(B, device=device) * 2.0 - 1.0) * math.pi
    r = r_target.view(-1) + 0.2 * torch.randn(B, device=device)
    r = r.clamp_min(0.1)
    x = center[:, 0] + r * torch.cos(theta)
    y = center[:, 1] + r * torch.sin(theta)
    z = z_target.view(-1) + 0.15 * torch.randn(B, device=device)
    p = torch.stack([x, y, z], dim=-1)

    # Small roll/pitch, random yaw.
    roll = 0.2 * torch.randn(B, device=device)
    pitch = 0.2 * torch.randn(B, device=device)
    yaw = (torch.rand(B, device=device) * 2.0 - 1.0) * math.pi
    q = euler_to_quaternion(torch.stack([roll, pitch, yaw], dim=-1))

    v_body = 0.5 * torch.randn(B, 3, device=device)
    w_body = 0.5 * torch.randn(B, 3, device=device)

    u_prev = torch.empty(B, nu, device=device).uniform_(-1.0, 1.0)
    u_cmd = (u_prev + 0.2 * torch.randn(B, nu, device=device)).clamp(-1.0, 1.0)

    # Make theta_prev consistent with desired CW direction for ~half the batch.
    theta_prev = theta.clone()
    theta_prev[: B // 2] = wrap_to_pi(theta_prev[: B // 2] + 0.1)  # makes dtheta negative => CW progress positive
    theta_prev[B // 2 :] = wrap_to_pi(theta_prev[B // 2 :] - 0.1)  # opposite direction => progress should drop
    theta_prev = theta_prev.view(B, 1)

    cfg = OrbitPPORewardCfg(
        sigma_r=0.2,
        sigma_z=0.2,
        sigma_roll=0.35,
        sigma_pitch=0.35,
        theta_scale=0.0872664626,
        v_tan_ref=0.5,
        progress_mode="both",
        progress_gate_mode="mul",
        gate_heading_power=2.0,
        gate_radius_power=1.0,
        w_r=1.0,
        w_z=1.0,
        w_h=0.5,
        w_p=0.5,
        w_a=0.25,
        w_back=0.5,
        w_u=0.05,
        w_du=0.10,
        w_w=0.02,
        w_vrad=0.2,
        v_rad_ref=0.3,
        r_min=0.3,
        r_max=3.0,
        rp_max_deg=45.0,
        terminate_penalty=-2.0,
        clip_min=-5.0,
        clip_max=5.0,
    )

    reward, terms = compute_orbit_ppo_reward_terms(
        p_world=p,
        q_world_wxyz=q,
        v_body=v_body,
        w_body=w_body,
        c_world=center,
        r_target=r_target,
        z_target=z_target,
        orbit_direction=orbit_direction,
        dt=0.05,
        u_cmd=u_cmd,
        u_prev=u_prev,
        theta_prev=theta_prev,
        cfg=cfg,
    )

    # Range sanity checks.
    for k in ("r_radius", "r_depth", "r_heading", "r_progress", "r_progress_back", "r_att"):
        v = terms[k]
        assert torch.isfinite(v).all(), f"{k} has non-finite values"
        assert (v >= 0.0).all() and (v <= 1.0 + 1e-6).all(), f"{k} out of [0,1]"
    for k in ("r_progress_raw", "progress_gate"):
        v = terms[k]
        assert torch.isfinite(v).all(), f"{k} has non-finite values"
        assert (v >= 0.0).all() and (v <= 1.0 + 1e-6).all(), f"{k} out of [0,1]"
    assert (terms["cos_yaw"] >= -1.0 - 1e-6).all() and (terms["cos_yaw"] <= 1.0 + 1e-6).all()
    assert (terms["delta_theta"] >= -math.pi - 1e-6).all() and (terms["delta_theta"] <= math.pi + 1e-6).all()
    for k in ("penalty_u", "penalty_du", "penalty_omega"):
        v = terms[k]
        assert (v >= -1e-8).all(), f"{k} should be non-negative"
    assert (terms["penalty_v_rad"] >= -1e-8).all() and (terms["penalty_v_rad"] <= 1.0 + 1e-6).all()
    assert (reward >= cfg.clip_min - 1e-6).all() and (reward <= cfg.clip_max + 1e-6).all()

    # CW sign check: first half should have higher theta-progress than second half (on average).
    p1 = terms["r_progress_theta"][: B // 2].mean().item()
    p2 = terms["r_progress_theta"][B // 2 :].mean().item()
    print(f"progress_theta mean (CW-consistent / opposite): {p1:.3f} / {p2:.3f}")
    b1 = terms["r_progress_back_theta"][: B // 2].mean().item()
    b2 = terms["r_progress_back_theta"][B // 2 :].mean().item()
    print(f"progress_back_theta mean (CW-consistent / opposite): {b1:.3f} / {b2:.3f}")

    _summary("reward", reward)
    for k in (
        "r_radius",
        "r_depth",
        "r_heading",
        "r_progress",
        "r_progress_raw",
        "progress_gate",
        "r_progress_back",
        "r_att",
        "penalty_u",
        "penalty_du",
        "penalty_omega",
        "penalty_v_rad",
    ):
        _summary(k, terms[k])

    print("OK")


if __name__ == "__main__":
    main()
