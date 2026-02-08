import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)

# ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py   task=Hover_PyPose_MPC algo=ppo_pypose_mpc_qrdiag_tv task.use_internal_mpc=false   headless=false enable_livestream=false env.num_envs=1   +eval.ckpt=/home/mjkim/MarineGym/wandb/offline-run-20260121_150344-p0j9ggir/files/checkpoint_final.pt +eval.steps=4000   +eval.print_every=200 +eval.print_weights_every=200 mode=evaluate
# ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py   task=OrbitCylinder_MPC algo=ppo_pypose_mpc_qrdiag_tv +task.use_internal_mpc=false   headless=false enable_livestream=false env.num_envs=1   +eval.ckpt=/path/to/checkpoint_final.pt +eval.steps=4000

# ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py task=OrbitCylinder_MPC algo=ppo_pypose_cylinder_mpc_werr_wu_tv task.reward_mode=orbit_cost task.orbit_target_mode=auto task.use_internal_mpc=false task.include_cylinder_rel_in_obs=false headless=false enable_livestream=false env.num_envs=1 mode=evaluate +eval.ckpt=/path/to/checkpoint_final.pt +eval.steps=4000 +eval.print_every=200 +eval.print_weights_every=200 +eval.video_path=/tmp/orbit_eval.mp4 +eval.render_interval=2


FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_prepare_pypose_mpc_cfg(cfg, base_env, *, algo_name: str, out_dir: Path):
    from marinegym.controllers.thruster_allocation import compute_thruster_allocation_matrix_from_drone

    algo = cfg.algo

    out_dir.mkdir(parents=True, exist_ok=True)
    alloc_path = out_dir / "thruster_allocation.npz"

    if not algo.get("mpc_alloc_npz", None):
        thrust_axis = int(cfg.task.get("thrust_axis", 0))
        B = compute_thruster_allocation_matrix_from_drone(base_env.drone, thrust_axis=thrust_axis)
        np.savez(alloc_path, B=B, quat_order="wxyz")
        algo.mpc_alloc_npz = str(alloc_path)

    if not algo.get("mpc_param_yaml", None):
        algo.mpc_param_yaml = base_env.drone.param_path

    try:
        algo.mpc_mass = float(base_env.drone.MASS_0.squeeze().item())
        algo.mpc_inertia = [float(x) for x in base_env.drone.INERTIA_0.squeeze().tolist()]
    except Exception:
        pass

    algo.mpc_dt = float(cfg.sim.dt)

    try:
        algo.obs_time_encoding_dim = int(getattr(base_env, "time_encoding_dim", 0)) if bool(cfg.task.time_encoding) else 0
    except Exception:
        algo.obs_time_encoding_dim = 0

    try:
        algo.obs_has_target_quat = bool(getattr(base_env, "include_target_quat_in_obs", False))
    except Exception:
        algo.obs_has_target_quat = False

    try:
        algo.mpc_nu = int(base_env.drone.action_spec.shape[-1])
    except Exception:
        algo.mpc_nu = int(getattr(algo, "mpc_nu", 6))

    # Align MPC rollout settings with task defaults unless explicitly overridden.
    if algo_name in ("ppo_pypose_mpc_qrdiag", "ppo_pypose_mpc_qrdiag_tv", "ppo_pypose_mpc_qrconst"):
        algo.mpc_horizon = int(cfg.task.get("mpc_horizon", algo.get("mpc_horizon", 15)))
        algo.mpc_ilqr_iters = int(cfg.task.get("mpc_ilqr_iters", algo.get("mpc_ilqr_iters", 6)))
        algo.max_thruster_force = float(cfg.task.get("max_thruster_force", algo.get("max_thruster_force", 40.0)))

    if algo_name == "ppo_pypose_cylinder_mpc_werr_wu_tv":
        algo.mpc_horizon = int(cfg.task.get("pypose_mpc_horizon", cfg.task.get("mpc_horizon", algo.get("mpc_horizon", 15))))
        algo.mpc_ilqr_iters = int(
            cfg.task.get("pypose_mpc_ilqr_iters", cfg.task.get("mpc_ilqr_iters", algo.get("mpc_ilqr_iters", 6)))
        )
        algo.mpc_ilqr_reg = float(cfg.task.get("pypose_mpc_ilqr_reg", algo.get("mpc_ilqr_reg", 1e-3)))
        algo.terminal_weight_mult = float(cfg.task.get("pypose_mpc_terminal_weight_mult", algo.get("terminal_weight_mult", 10.0)))
        algo.max_thruster_force = float(
            cfg.task.get(
                "pypose_max_thruster_force",
                cfg.task.get(
                    "mpc_max_thruster_force",
                    cfg.task.get("max_thruster_force", algo.get("max_thruster_force", 40.0)),
                ),
            )
        )

        algo.orbit_radius = float(cfg.task.get("orbit_radius", algo.get("orbit_radius", 1.4)))
        algo.orbit_direction = 1.0 if float(cfg.task.get("orbit_direction", 1.0)) >= 0.0 else -1.0
        algo.orbit_yaw_offset = float(cfg.task.get("orbit_yaw_offset", 0.0))

        orbit_v_tan = float(cfg.task.get("orbit_v_tan", 0.0))
        if orbit_v_tan <= 0.0:
            orbit_period_steps = int(cfg.task.get("orbit_period_steps", cfg.task.get("max_episode_length", 1)))
            dt = float(cfg.sim.dt)
            r = float(algo.orbit_radius)
            orbit_v_tan = float(2.0 * np.pi * r / (max(1, orbit_period_steps) * dt)) if r > 1e-6 else 0.0
        algo.orbit_v_tan = float(orbit_v_tan)

        orbit_target_mode = str(cfg.task.get("orbit_target_mode", "auto")).lower()
        if orbit_target_mode in ("auto", ""):
            reward_mode = str(cfg.task.get("reward_mode", "hover")).lower()
            if reward_mode in ("orbit_cost", "cylinder_cost", "cylinder_orbit_cost", "orbit"):
                orbit_target_mode = "center"
            else:
                orbit_target_mode = "waypoint" if not bool(cfg.task.get("use_internal_mpc", True)) else "center"
        if orbit_target_mode in ("waypoint", "moving_waypoint", "wp"):
            algo.orbit_z = 0.0
        else:
            center_cfg = cfg.task.get("cylinder_center", [0.0, 0.0, 0.0])
            algo.orbit_z = float(cfg.task.get("orbit_z", float(center_cfg[2]))) - float(center_cfg[2])

        algo.obs_has_cylinder_rel = bool(cfg.task.get("include_cylinder_rel_in_obs", False))

        if not algo.get("werr_init", None):
            q_radial = float(cfg.task.get("mpc_q_radial", 50.0))
            q_z = float(cfg.task.get("mpc_q_z", 30.0))
            q_tan = float(cfg.task.get("mpc_q_tan", 10.0))
            q_radial_speed = float(cfg.task.get("mpc_q_radial_speed", 5.0))
            q_heading = float(cfg.task.get("mpc_q_heading", 30.0))
            q_roll = float(cfg.task.get("mpc_q_roll", 60.0))
            q_pitch = float(cfg.task.get("mpc_q_pitch", 60.0))
            q_wxy = float(cfg.task.get("mpc_q_wxy", 0.5))
            algo.werr_init = [
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
        if not algo.get("wu_init", None):
            r_u = float(cfg.task.get("mpc_r_u", 0.01))
            algo.wu_init = [r_u] * int(algo.mpc_nu)

    if algo_name in ("ppo_pypose_mpc_qrdiag", "ppo_pypose_mpc_qrdiag_tv"):
        if not algo.get("wx_init", None):
            q_pos = float(cfg.task.get("mpc_q_pos", 50.0))
            q_quat = float(cfg.task.get("mpc_q_quat", 5.0))
            q_vel = float(cfg.task.get("mpc_q_vel", 2.0))
            q_omega = float(cfg.task.get("mpc_q_omega", 0.5))
            algo.wx_init = [q_pos] * 3 + [q_quat] * 4 + [q_vel] * 3 + [q_omega] * 3
        if not algo.get("wu_init", None):
            r_u = float(cfg.task.get("mpc_r_u", 0.02))
            algo.wu_init = [r_u] * int(algo.mpc_nu)

    if algo_name == "ppo_pypose_mpc_qrconst":
        if algo.get("q_pos_init", None) is None:
            algo.q_pos_init = float(cfg.task.get("mpc_q_pos", 50.0))
        if algo.get("q_quat_init", None) is None:
            algo.q_quat_init = float(cfg.task.get("mpc_q_quat", 5.0))
        if algo.get("q_vel_init", None) is None:
            algo.q_vel_init = float(cfg.task.get("mpc_q_vel", 2.0))
        if algo.get("q_omega_init", None) is None:
            algo.q_omega_init = float(cfg.task.get("mpc_q_omega", 0.5))
        if algo.get("r_u_init", None) is None:
            algo.r_u_init = float(cfg.task.get("mpc_r_u", 0.02))


def _find_pypose_tv_actor(policy) -> Optional[torch.nn.Module]:
    for m in policy.modules():
        if hasattr(m, "cost_map") and hasattr(m, "mpc"):
            # Heuristic: our actor has both cost_map + mpc.
            return m
    return None


def _load_policy_state(policy, checkpoint_path: Path, device):
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    # Some scripts may wrap the state dict in a dict (wandb artifacts, etc.)
    if isinstance(ckpt, dict):
        for key in ("policy", "policy_state_dict", "model", "state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like dict. type={type(ckpt)}")

    base_sd = policy.state_dict()
    merged = {}
    mismatched = []

    for key, value in ckpt.items():
        if key in base_sd:
            if hasattr(value, "shape") and hasattr(base_sd[key], "shape") and value.shape != base_sd[key].shape:
                mismatched.append((key, tuple(value.shape), tuple(base_sd[key].shape)))
                continue
            merged[key] = value

    missing = [key for key in base_sd.keys() if key not in merged]
    for key in missing:
        merged[key] = base_sd[key]

    if mismatched:
        print("[eval] shape mismatched keys -> use default init for them:")
        for key, s1, s2 in mismatched[:30]:
            print("  ", key, s1, "!=", s2)
        if len(mismatched) > 30:
            print("  ...", len(mismatched) - 30, "more")

    if missing:
        print("[eval] filled missing keys:", missing[:30], ("..." if len(missing) > 30 else ""))

    policy.load_state_dict(merged, strict=True)
    policy.eval()


def _summarize_weights(w_x_seq: torch.Tensor, w_u_seq: torch.Tensor):
    # w_x_seq: (B,H,13) w_u_seq: (B,H,nu)
    wx0 = w_x_seq[:, 0, :]
    wxL = w_x_seq[:, -1, :]
    wu0 = w_u_seq[:, 0, :]
    wuL = w_u_seq[:, -1, :]

    def _group(wx: torch.Tensor):
        return {
            "q_pos": float(wx[:, 0:3].mean().item()),
            "q_quat": float(wx[:, 3:7].mean().item()),
            "q_vel": float(wx[:, 7:10].mean().item()),
            "q_omega": float(wx[:, 10:13].mean().item()),
        }

    s0 = _group(wx0)
    sL = _group(wxL)
    s0["r_u"] = float(wu0.mean().item())
    sL["r_u"] = float(wuL.mean().item())
    return s0, sL


def _summarize_orbit_weights(w_err_seq: torch.Tensor, w_u_seq: torch.Tensor):
    # w_err_seq: (B,H,10) w_u_seq: (B,H,nu)
    we0 = w_err_seq[:, 0, :]
    weL = w_err_seq[:, -1, :]
    wu0 = w_u_seq[:, 0, :]
    wuL = w_u_seq[:, -1, :]

    def _group(we: torch.Tensor):
        return {
            "q_radial": float(we[:, 0].mean().item()),
            "q_z": float(we[:, 1].mean().item()),
            "q_tan": float(we[:, 2].mean().item()),
            "q_radial_speed": float(we[:, 3].mean().item()),
            "q_heading": float(we[:, 4:6].mean().item()),
            "q_roll": float(we[:, 6].mean().item()),
            "q_pitch": float(we[:, 7].mean().item()),
            "q_wxy": float(we[:, 8:10].mean().item()),
        }

    s0 = _group(we0)
    sL = _group(weL)
    s0["r_u"] = float(wu0.mean().item())
    sL["r_u"] = float(wuL.mean().item())
    return s0, sL


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Evaluate a PyPose-MPC PPO policy (e.g., ppo_pypose_mpc_qrdiag_tv) on a task.

    Example:
      python scripts/evaluate_pypose_mpc_ppo.py task=Hover_PyPose_MPC algo=ppo_pypose_mpc_qrdiag_tv \\
        task.use_internal_mpc=false headless=true enable_livestream=false env.num_envs=1 mode=evaluate \\
        +eval.ckpt=/path/to/run/checkpoints/checkpoint_final.pt +eval.steps=2000 +eval.print_every=200

        ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py   task=OrbitCylinder_MPC algo=ppo_pypose_cylinder_mpc_werr_wu_tv   task.reward_mode=orbit_cost task.orbit_target_mode=auto   task.use_internal_mpc=false task.include_cylinder_rel_in_obs=false   headless=false enable_livestream=false env.num_envs=1 mode=evaluate   +eval.ckpt=/path/to/checkpoint_final.pt +eval.steps=4000   +eval.print_every=200 +eval.print_weights_every=200   +eval.video_path=/tmp/orbit_eval.mp4 +eval.render_interval=2

    Trajectory logging (default on when mode=evaluate):
      - Saves `trajectory.npz` under Hydra's run dir (./outputs/...).
      - Saves plots by default:
          - `trajectory.png`         (|v_tan - v_tan_ref| colormap if available; else speed)
          - `trajectory_energy.png`  (energy proxy colormap, E = Σ|u_i|^3, if available)
          - `trajectory_energy_polar.png` (polar plot: orbit angle vs instantaneous power proxy, P_total = Σ|u_i|^3)
      - Disable with `+eval.save_traj=false`.
      - Override paths with `+eval.traj_path=/tmp/trajectory.npz +eval.traj_png_path=/tmp/trajectory.png`.
      - Override trajectory colormap key with `+eval.traj_color_key=speed` (or `v_tan_err`, `v_tan`, `energy`, ...).
      - Re-visualize later: `python scripts/visualize_trajectory.py /path/to/trajectory.npz`

    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # OrbitCylinderMPC: ensure we evaluate the *policy* (direct thruster actions),
    # not the env's internal MPC override (which is enabled when control_mode=mpc).
    if str(getattr(cfg, "task", {}).get("name", "")) == "OrbitCylinderMPC":
        cfg.task.control_mode = "direct"
        cfg.task.use_internal_mpc = False

    eval_cfg = cfg.get("eval", {}) or {}
    mode = str(cfg.get("mode", "")).lower()
    ckpt = eval_cfg.get("ckpt", None)
    if not ckpt:
        ckpt = str(Path.cwd() / "checkpoints" / "checkpoint_final.pt")
    ckpt_path = Path(str(ckpt)).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    steps = int(eval_cfg.get("steps", 2000))
    seed = int(eval_cfg.get("seed", cfg.get("seed", 0)))
    print_every = int(eval_cfg.get("print_every", 200))
    print_weights_every = int(eval_cfg.get("print_weights_every", print_every))
    render = bool(eval_cfg.get("render", not cfg.headless))
    render_interval = int(eval_cfg.get("render_interval", 2))
    video_path = str(eval_cfg.get("video_path", "") or "")

    save_traj = eval_cfg.get("save_traj", None)
    if save_traj is None:
        save_traj = mode == "evaluate"
    save_traj = bool(save_traj)
    traj_env_id = int(eval_cfg.get("traj_env_id", 0))
    traj_stride = max(1, int(eval_cfg.get("traj_stride", 1)))
    traj_out = eval_cfg.get("traj_path", None)
    traj_path = Path(str(traj_out)).expanduser() if traj_out else Path("trajectory.npz")
    if not traj_path.is_absolute():
        traj_path = Path.cwd() / traj_path

    traj_png_out = eval_cfg.get("traj_png_path", None)
    traj_png_path = Path(str(traj_png_out)).expanduser() if traj_png_out else traj_path.with_suffix(".png")
    if not traj_png_path.is_absolute():
        traj_png_path = Path.cwd() / traj_png_path
    plot_energy = bool(eval_cfg.get("plot_energy", eval_cfg.get("plot_traj_energy", True)))
    traj_energy_png_out = eval_cfg.get("traj_energy_png_path", None)
    traj_energy_png_path = (
        Path(str(traj_energy_png_out)).expanduser()
        if traj_energy_png_out
        else traj_png_path.with_name(traj_png_path.stem + "_energy" + traj_png_path.suffix)
    )
    if not traj_energy_png_path.is_absolute():
        traj_energy_png_path = Path.cwd() / traj_energy_png_path
    plot_energy_polar = bool(eval_cfg.get("plot_energy_polar", eval_cfg.get("plot_polar_energy", True)))
    traj_energy_polar_png_out = eval_cfg.get("traj_energy_polar_png_path", None)
    traj_energy_polar_png_path = (
        Path(str(traj_energy_polar_png_out)).expanduser()
        if traj_energy_polar_png_out
        else traj_png_path.with_name(traj_png_path.stem + "_energy_polar" + traj_png_path.suffix)
    )
    if not traj_energy_polar_png_path.is_absolute():
        traj_energy_polar_png_path = Path.cwd() / traj_energy_polar_png_path
    energy_polar_bin_deg = float(eval_cfg.get("energy_polar_bin_deg", 5.0))
    energy_polar_show_raw = bool(eval_cfg.get("energy_polar_show_raw", False))
    plot_traj = bool(eval_cfg.get("plot_traj", True))
    plot_heading_stride = int(eval_cfg.get("plot_heading_stride", 50))
    plot_arrow_len = float(eval_cfg.get("plot_arrow_len", 0.25))
    plot_show = bool(eval_cfg.get("plot_show", False))

    simulation_app = init_simulation_app(cfg)

    from marinegym.envs import IsaacEnv, register_tasks

    register_tasks()
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()
    env.set_seed(seed)
    base_env.enable_render(render)

    algo_name = str(cfg.algo.name).lower()
    if algo_name.startswith("ppo_pypose_mpc_") or algo_name == "ppo_pypose_cylinder_mpc_werr_wu_tv":
        _maybe_prepare_pypose_mpc_cfg(cfg, base_env, algo_name=algo_name, out_dir=Path.cwd() / algo_name)

    policy = ALGOS[algo_name](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )
    _load_policy_state(policy, ckpt_path, device="cpu")

    mpc_actor = _find_pypose_tv_actor(policy)

    td = env.reset()
    episode_stats = []
    frames = []

    pos_log = []
    heading_log = []
    speed_log = []
    vtan_log = []
    vtan_err_log = []
    u_log = []
    energy_log = []
    flow_log = []
    done_log = []
    step_log = []

    traj_nu = int(getattr(base_env.drone, "num_rotors", 0) or 0)
    if traj_nu <= 0:
        try:
            traj_nu = int(getattr(base_env.drone.action_spec, "shape", (0,))[-1])
        except Exception:
            traj_nu = 0

    # Reference tangential speed (used for v_tan_err colormap).
    v_tan_ref = float("nan")
    try:
        v_tan_ref = float(cfg.task.get("reward_orbit_v_tan_ref", float("nan")))
    except Exception:
        v_tan_ref = float("nan")
    if not np.isfinite(v_tan_ref):
        try:
            v_tan_ref = float(getattr(base_env, "orbit_v_tan", float("nan")))
        except Exception:
            v_tan_ref = float("nan")

    last_action_cmd = None

    def _maybe_log_traj(*, step: int, done: bool) -> None:
        if not save_traj:
            return
        if (int(step) % int(traj_stride)) != 0:
            return
        try:
            base_env.drone.get_state()
            pos_np = _to_numpy(base_env.drone.pos[traj_env_id, 0, :]).astype(np.float32, copy=False)
            pos_log.append(pos_np)
            heading_log.append(_to_numpy(base_env.drone.heading[traj_env_id, 0, :]).astype(np.float32, copy=False))
            vel_lin_w = None
            try:
                vel_lin_w = base_env.drone.vel_w[traj_env_id, 0, 0:3]
                speed_log.append(float(torch.linalg.norm(vel_lin_w).item()))
            except Exception:
                speed_log.append(float("nan"))

            # Tangential speed around the cylinder (signed along desired orbit direction).
            v_tan = float("nan")
            v_tan_err = float("nan")
            try:
                if vel_lin_w is not None and hasattr(base_env, "cylinder_center"):
                    center_t = getattr(base_env, "cylinder_center", None)
                    center_np = (
                        np.asarray(_to_numpy(center_t), dtype=np.float32).reshape(-1)[:3]
                        if center_t is not None
                        else np.zeros((3,), dtype=np.float32)
                    )
                    rel_xy = pos_np[0:2] - center_np[0:2]  # center -> robot (XY)
                    dist = float(np.linalg.norm(rel_xy))
                    if np.isfinite(dist) and dist > 1e-6:
                        radial_hat_xy = rel_xy / dist
                        dir_sign = float(getattr(base_env, "orbit_direction", 1.0))
                        dir_sign = 1.0 if dir_sign >= 0.0 else -1.0
                        tan_hat_xy = np.asarray([-radial_hat_xy[1], radial_hat_xy[0]], dtype=np.float32) * dir_sign
                        v_xy = np.asarray(_to_numpy(vel_lin_w[0:2]), dtype=np.float32).reshape(2,)
                        v_tan = float(tan_hat_xy[0] * v_xy[0] + tan_hat_xy[1] * v_xy[1])
                        if np.isfinite(v_tan_ref):
                            v_tan_err = float(abs(v_tan - float(v_tan_ref)))
            except Exception:
                v_tan = float("nan")
                v_tan_err = float("nan")
            vtan_log.append(float(v_tan))
            vtan_err_log.append(float(v_tan_err))

            try:
                flow = base_env.drone.flow_vels[traj_env_id, 0, 0:3]
                flow_log.append(_to_numpy(flow).astype(np.float32, copy=False))
            except Exception:
                flow_log.append(np.full((3,), np.nan, dtype=np.float32))

            u_cmd = getattr(base_env.drone, "throttle", None)
            if u_cmd is None:
                u_cmd = last_action_cmd
            u_vec = None
            energy = float("nan")
            if u_cmd is not None:
                try:
                    if torch.is_tensor(u_cmd):
                        u_t = u_cmd
                        if u_t.ndim >= 3:
                            u_t = u_t.squeeze(1)
                        u_env = u_t[int(traj_env_id)]
                        u_vec = u_env.detach().cpu().to(dtype=torch.float32).numpy()
                        energy = float((u_env.abs() ** 3).sum().item())
                    else:
                        u_np = np.asarray(u_cmd)
                        if u_np.ndim >= 3:
                            u_np = np.squeeze(u_np, axis=1)
                        u_env = np.asarray(u_np[int(traj_env_id)], dtype=np.float32)
                        u_vec = u_env
                        energy = float(np.sum(np.abs(u_env) ** 3))
                except Exception:
                    u_vec = None
                    energy = float("nan")
            if u_vec is None:
                if traj_nu > 0:
                    u_vec = np.full((traj_nu,), np.nan, dtype=np.float32)
                else:
                    u_vec = np.zeros((0,), dtype=np.float32)
            try:
                u_vec = np.asarray(u_vec, dtype=np.float32).reshape(-1)
                if traj_nu > 0 and u_vec.shape[0] != traj_nu:
                    if u_vec.shape[0] > traj_nu:
                        u_vec = u_vec[:traj_nu]
                    else:
                        u_vec = np.pad(u_vec, (0, traj_nu - u_vec.shape[0]), constant_values=np.nan)
            except Exception:
                u_vec = np.full((traj_nu,), np.nan, dtype=np.float32) if traj_nu > 0 else np.zeros((0,), dtype=np.float32)
            u_log.append(u_vec)
            energy_log.append(float(energy))
            done_log.append(bool(done))
            step_log.append(int(step))
        except Exception:
            return

    _maybe_log_traj(step=0, done=False)

    with set_exploration_type(ExplorationType.MODE):
        for t in range(steps):
            if print_every > 0 and t % print_every == 0:
                try:
                    r = td.get(("stats", "return"), None)
                    pe = td.get(("stats", "pos_error"), None)
                    if r is not None and pe is not None:
                        print(f"[eval] t={t} return={float(r.mean().item()):.4g} pos_error={float(pe.mean().item()):.4g}")
                except Exception:
                    pass

            if mpc_actor is not None and print_weights_every > 0 and t % print_weights_every == 0:
                try:
                    obs = td[("agents", "observation")].squeeze(-2)
                    obs_flat = obs.reshape(-1, obs.shape[-1])
                    with torch.no_grad():
                        w_a_seq, w_u_seq = mpc_actor.cost_map(obs_flat)
                    if int(w_a_seq.shape[-1]) == 13:
                        s0, sL = _summarize_weights(w_a_seq, w_u_seq)
                    else:
                        s0, sL = _summarize_orbit_weights(w_a_seq, w_u_seq)
                    print(f"[eval] t={t} w0={s0} wT={sL}")
                except Exception:
                    pass

            td = policy(td)
            try:
                last_action_cmd = td.get(("agents", "action"), None)
            except Exception:
                last_action_cmd = None
            td = env.step(td)["next"]

            if render and video_path and render_interval > 0 and (t % render_interval) == 0:
                try:
                    frames.append(env.render(mode="rgb_array"))
                except Exception:
                    pass

            done = td.get("done", None)
            done_val = False
            if done is not None:
                try:
                    done_mask_for_traj = done.squeeze(-1)
                    done_val = bool(done_mask_for_traj[traj_env_id].item())
                except Exception:
                    done_val = False
            _maybe_log_traj(step=t + 1, done=done_val)

            if done is not None and bool(done.any()):
                done_mask = done.squeeze(-1)
                try:
                    stats_done = td["stats"][done_mask].cpu()
                    episode_stats.extend(stats_done.unbind(0))
                except Exception:
                    pass

                td.set("_reset", done_mask)
                td = env.reset(td)
                _maybe_log_traj(step=t + 1, done=False)

            if not bool(cfg.headless):
                env.render()

    if episode_stats:
        stats_td = torch.stack(episode_stats).to_tensordict()
        summary = {k: float(v.float().mean().item()) for k, v in stats_td.items(True, True)}
        print(f"[eval] episodes={len(episode_stats)} mean_stats={summary}")
    else:
        print("[eval] No completed episodes during evaluation (increase eval.steps or reduce max_episode_length).")

    if video_path and frames:
        try:
            from torchvision.io import write_video

            fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps * max(1, render_interval))
            write_video(video_path, torch.as_tensor(np.stack(frames)), fps=fps)
            print(f"[eval] saved video: {video_path}")
        except Exception as e:
            print(f"[eval] failed to save video (install torchvision or disable eval.video_path): {e}")

    if save_traj and pos_log:
        try:
            pos_arr = np.stack(pos_log, axis=0).astype(np.float32, copy=False)
            heading_arr = np.stack(heading_log, axis=0).astype(np.float32, copy=False)
            speed_arr = np.asarray(speed_log, dtype=np.float32)
            vtan_arr = np.asarray(vtan_log, dtype=np.float32)
            vtan_err_arr = np.asarray(vtan_err_log, dtype=np.float32)
            u_arr = np.stack(u_log, axis=0).astype(np.float32, copy=False) if u_log else np.zeros((0, 0), dtype=np.float32)
            energy_arr = np.asarray(energy_log, dtype=np.float32)
            energy_total = float(np.nansum(energy_arr))
            flow_arr = np.stack(flow_log, axis=0).astype(np.float32, copy=False) if flow_log else np.zeros((0, 3), dtype=np.float32)
            done_arr = np.asarray(done_log, dtype=np.bool_)
            step_arr = np.asarray(step_log, dtype=np.int64)

            if flow_arr.shape[0] != pos_arr.shape[0]:
                if flow_arr.shape[0] > pos_arr.shape[0]:
                    flow_arr = flow_arr[: pos_arr.shape[0]]
                else:
                    pad = np.full((pos_arr.shape[0] - flow_arr.shape[0], 3), np.nan, dtype=np.float32)
                    flow_arr = np.concatenate([flow_arr, pad], axis=0) if flow_arr.size else pad

            current_vec_w = (
                np.nanmean(flow_arr, axis=0).astype(np.float32, copy=False)
                if flow_arr.size
                else np.full((3,), np.nan, dtype=np.float32)
            )

            center = _to_numpy(getattr(base_env, "cylinder_center", np.zeros((1, 1, 3), dtype=np.float32)))
            center = np.asarray(center, dtype=np.float32).reshape(-1)[:3]

            traj_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = traj_path.with_name(traj_path.stem + "_tmp" + traj_path.suffix)
            np.savez_compressed(
                tmp_path,
                step=step_arr,
                pos=pos_arr,
                heading=heading_arr,
                speed=speed_arr,
                v_tan=vtan_arr,
                v_tan_err=vtan_err_arr,
                u=u_arr,
                energy=energy_arr,
                energy_total=energy_total,
                flow_vel_w=flow_arr,
                current_vec_w=current_vec_w,
                done=done_arr,
                cylinder_center=center,
                cylinder_radius=float(getattr(base_env, "cylinder_radius", 0.0)),
                cylinder_height=float(getattr(base_env, "cylinder_height", 0.0)),
                orbit_radius=float(getattr(base_env, "orbit_radius", 0.0)),
                orbit_z=float(getattr(base_env, "orbit_z", 0.0)),
                orbit_v_tan=float(getattr(base_env, "orbit_v_tan", 0.0)),
                v_tan_ref=float(v_tan_ref) if np.isfinite(v_tan_ref) else np.nan,
                meta=np.array(
                    [
                        {
                            "task": str(getattr(getattr(cfg, "task", None), "name", "")),
                            "algo": str(getattr(getattr(cfg, "algo", None), "name", "")),
                            "ckpt": str(ckpt_path),
                            "dt": float(getattr(getattr(cfg, "sim", None), "dt", 0.0)),
                            "seed": int(seed),
                            "steps": int(steps),
                            "traj_env_id": int(traj_env_id),
                            "traj_stride": int(traj_stride),
                            "current_vec_w": current_vec_w.tolist() if np.isfinite(current_vec_w).any() else None,
                        }
                    ],
                    dtype=object,
                ),
            )
            tmp_path.replace(traj_path)
            print(f"[eval] saved trajectory: {traj_path.resolve()}")
            print(f"[eval] energy_total (Σ_t Σ_i |u_i|^3): {energy_total:.6g}")

            if plot_traj:
                try:
                    from visualize_trajectory import plot_power_polar, plot_trajectory_3d

                    traj_color_key = str(eval_cfg.get("traj_color_key", eval_cfg.get("traj_color", ""))).strip()
                    if not traj_color_key:
                        traj_color_key = (
                            "v_tan_err"
                            if (np.isfinite(v_tan_ref) and hasattr(base_env, "cylinder_center"))
                            else "speed"
                        )
                    plot_trajectory_3d(
                        traj_path=traj_path,
                        out_path=traj_png_path,
                        heading_stride=int(plot_heading_stride),
                        arrow_len=float(plot_arrow_len),
                        color_key=str(traj_color_key),
                        show=bool(plot_show),
                    )
                    print(f"[eval] saved trajectory plot: {traj_png_path.resolve()}")

                    if plot_energy:
                        plot_trajectory_3d(
                            traj_path=traj_path,
                            out_path=traj_energy_png_path,
                            heading_stride=int(plot_heading_stride),
                            arrow_len=float(plot_arrow_len),
                            color_key="energy",
                            show=bool(plot_show),
                        )
                        print(f"[eval] saved energy plot: {traj_energy_png_path.resolve()}")

                    if plot_energy_polar:
                        plot_power_polar(
                            traj_path=traj_path,
                            out_path=traj_energy_polar_png_path,
                            bin_deg=float(energy_polar_bin_deg),
                            show_raw=bool(energy_polar_show_raw),
                            show=bool(plot_show),
                        )
                        print(f"[eval] saved energy polar plot: {traj_energy_polar_png_path.resolve()}")
                except Exception as exc:
                    print(f"[eval] failed to plot trajectory: {exc}")
        except Exception as exc:
            print(f"[eval] failed to save trajectory: {exc}")

    simulation_app.close()


if __name__ == "__main__":
    main()
