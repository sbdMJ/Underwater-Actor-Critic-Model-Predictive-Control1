import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)

# ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py   task=Hover_PyPose_MPC algo=ppo_pypose_mpc_qrdiag_tv task.use_internal_mpc=false   headless=false enable_livestream=false env.num_envs=1   +eval.ckpt=/home/mjkim/MarineGym/wandb/offline-run-20260121_150344-p0j9ggir/files/checkpoint_final.pt +eval.steps=4000   +eval.print_every=200 +eval.print_weights_every=200 mode=evaluate
# ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py   task=OrbitCylinder_MPC algo=ppo_pypose_mpc_qrdiag_tv +task.use_internal_mpc=false   headless=false enable_livestream=false env.num_envs=1   +eval.ckpt=/path/to/checkpoint_final.pt +eval.steps=4000

# ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py task=OrbitCylinder_MPC algo=ppo_pypose_cylinder_mpc_werr_wu_tv task.reward_mode=orbit_cost task.orbit_target_mode=auto task.use_internal_mpc=false task.include_cylinder_rel_in_obs=false headless=false enable_livestream=false env.num_envs=1 mode=evaluate +eval.ckpt=/path/to/checkpoint_final.pt +eval.steps=4000 +eval.print_every=200 +eval.print_weights_every=200 +eval.video_path=/tmp/orbit_eval.mp4 +eval.render_interval=2


FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


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


def _build_observation_template(
    rpos: torch.Tensor,
    *,
    observation_dim: int,
    drone_state_dim: int,
    include_target_quat: bool,
    time_encoding_dim: int,
    include_cylinder_rel: bool,
    device: torch.device,
):
    obs = torch.zeros((rpos.shape[0], 1, observation_dim), device=device, dtype=rpos.dtype)
    obs[:, 0, 0:3] = rpos
    rheading_start = drone_state_dim
    obs[:, 0, rheading_start : rheading_start + 3] = 0.0
    idx = rheading_start + 3
    if include_target_quat:
        idx += 4
    if time_encoding_dim > 0:
        idx += time_encoding_dim
    if include_cylinder_rel:
        obs[:, 0, idx : idx + 3] = 0.0
    return obs


def _evaluate_value_grid(
    policy,
    *,
    base_env,
    grid_xy: tuple[np.ndarray, np.ndarray],
    grid_rz: tuple[np.ndarray, np.ndarray],
    observation_dim: int,
    drone_state_dim: int,
    include_target_quat: bool,
    time_encoding_dim: int,
    include_cylinder_rel: bool,
):
    xx, yy = grid_xy
    rr, zz = grid_rz

    rpos_xy = torch.from_numpy(
        np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    ).to(device=base_env.device, dtype=torch.float32)
    rpos_rz = torch.from_numpy(
        np.stack([rr.ravel(), np.zeros(rr.size), zz.ravel()], axis=1)
    ).to(device=base_env.device, dtype=torch.float32)

    obs_xy = _build_observation_template(
        -rpos_xy,
        observation_dim=observation_dim,
        drone_state_dim=drone_state_dim,
        include_target_quat=include_target_quat,
        time_encoding_dim=time_encoding_dim,
        include_cylinder_rel=include_cylinder_rel,
        device=base_env.device,
    )
    obs_rz = _build_observation_template(
        -rpos_rz,
        observation_dim=observation_dim,
        drone_state_dim=drone_state_dim,
        include_target_quat=include_target_quat,
        time_encoding_dim=time_encoding_dim,
        include_cylinder_rel=include_cylinder_rel,
        device=base_env.device,
    )

    intrinsics = None
    if ("agents", "intrinsics") in base_env.observation_spec.keys(True, True):
        intrinsics = base_env.drone.intrinsics

    def _run(obs: torch.Tensor) -> np.ndarray:
        batch = obs.shape[0]
        tensordict = TensorDict(
            {("agents", "observation"): obs},
            batch_size=[batch],
        )
        if intrinsics is not None:
            tensordict.set(
                ("agents", "intrinsics"),
                intrinsics.expand(batch, -1, -1),
            )
        with torch.no_grad():
            values = policy.critic(tensordict)["state_value"]
        return values.squeeze(-1).squeeze(-1).cpu().numpy()

    values_xy = _run(obs_xy).reshape(xx.shape)
    values_rz = _run(obs_rz).reshape(rr.shape)
    return values_xy, values_rz


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Evaluate a PyPose-MPC PPO policy (e.g., ppo_pypose_mpc_qrdiag_tv) on a task.

    Example:
      python scripts/evaluate_pypose_mpc_ppo.py task=Hover_PyPose_MPC algo=ppo_pypose_mpc_qrdiag_tv \\
        task.use_internal_mpc=false headless=true enable_livestream=false env.num_envs=1 mode=evaluate \\
        +eval.ckpt=/path/to/run/checkpoints/checkpoint_final.pt +eval.steps=2000 +eval.print_every=200

        ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py   task=OrbitCylinder_MPC algo=ppo_pypose_cylinder_mpc_werr_wu_tv   task.reward_mode=orbit_cost task.orbit_target_mode=auto   task.use_internal_mpc=false task.include_cylinder_rel_in_obs=false   headless=false enable_livestream=false env.num_envs=1 mode=evaluate   +eval.ckpt=/path/to/checkpoint_final.pt +eval.steps=4000   +eval.print_every=200 +eval.print_weights_every=200   +eval.video_path=/tmp/orbit_eval.mp4 +eval.render_interval=2

    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_cfg = cfg.get("eval", {}) or {}
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
    save_plot_data = bool(eval_cfg.get("save_plot_data", False))
    plot_data_path = str(eval_cfg.get("plot_data_path", "outputs/ac_mpc_eval_plot_data.npz"))
    plot_pred_stride = int(eval_cfg.get("plot_pred_stride", 10))
    plot_grid_size = int(eval_cfg.get("plot_grid_size", 150))

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
    actual_traj = []
    velocities = []
    mpc_preds = []

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
            td = env.step(td)["next"]

            if save_plot_data:
                try:
                    base_env.drone.get_state()
                    pos = base_env.drone.pos.squeeze(1)[0].detach().cpu().numpy()
                    vel = base_env.drone.vel_w.squeeze(1)[0, :3]
                    speed = float(torch.linalg.norm(vel).detach().cpu().item())
                    actual_traj.append(pos)
                    velocities.append(speed)
                except Exception:
                    pass

                if (
                    mpc_actor is not None
                    and (t % max(1, plot_pred_stride) == 0)
                    and hasattr(mpc_actor, "mpc")
                ):
                    try:
                        x_traj = mpc_actor.mpc.last_x_traj
                        if x_traj is not None:
                            pred = x_traj[0, :, :3].detach().cpu().numpy()
                            mpc_preds.append(pred)
                    except Exception:
                        pass

            if render and video_path and render_interval > 0 and (t % render_interval) == 0:
                try:
                    frames.append(env.render(mode="rgb_array"))
                except Exception:
                    pass

            done = td.get("done", None)
            if done is not None and bool(done.any()):
                done_mask = done.squeeze(-1)
                try:
                    stats_done = td["stats"][done_mask].cpu()
                    episode_stats.extend(stats_done.unbind(0))
                except Exception:
                    pass

                td.set("_reset", done_mask)
                td = env.reset(td)

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

    if save_plot_data and actual_traj:
        actual_traj_np = np.asarray(actual_traj)
        velocities_np = np.asarray(velocities)
        x_min, x_max = actual_traj_np[:, 0].min(), actual_traj_np[:, 0].max()
        y_min, y_max = actual_traj_np[:, 1].min(), actual_traj_np[:, 1].max()
        pad = max(1.0, 0.3 * float(cfg.task.get("orbit_radius", 2.0)))
        x_min, x_max = x_min - pad, x_max + pad
        y_min, y_max = y_min - pad, y_max + pad
        xs = np.linspace(x_min, x_max, plot_grid_size)
        ys = np.linspace(y_min, y_max, plot_grid_size)
        xx, yy = np.meshgrid(xs, ys)

        r_max = max(float(cfg.task.get("orbit_radius", 2.0)) + pad, np.sqrt(x_max**2 + y_max**2))
        zs = np.linspace(actual_traj_np[:, 2].min() - pad, actual_traj_np[:, 2].max() + pad, plot_grid_size)
        rs = np.linspace(0.0, r_max, plot_grid_size)
        rr, zz = np.meshgrid(rs, zs)

        observation_dim = int(env.observation_spec[("agents", "observation")].shape[-1])
        drone_state_dim = int(base_env.drone.state_spec.shape[-1])
        include_target_quat = bool(getattr(base_env, "include_target_quat_in_obs", False))
        time_encoding_dim = int(getattr(base_env, "time_encoding_dim", 0)) if bool(cfg.task.time_encoding) else 0
        include_cylinder_rel = bool(cfg.task.get("include_cylinder_rel_in_obs", False))

        values_xy, values_rz = _evaluate_value_grid(
            policy,
            base_env=base_env,
            grid_xy=(xx, yy),
            grid_rz=(rr, zz),
            observation_dim=observation_dim,
            drone_state_dim=drone_state_dim,
            include_target_quat=include_target_quat,
            time_encoding_dim=time_encoding_dim,
            include_cylinder_rel=include_cylinder_rel,
        )

        np.savez(
            plot_data_path,
            actual_traj=actual_traj_np,
            velocities=velocities_np,
            mpc_preds=np.asarray(mpc_preds, dtype=object),
            values_xy=values_xy,
            values_rz=values_rz,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            r_max=r_max,
            z_min=zs.min(),
            z_max=zs.max(),
            grid_size=plot_grid_size,
            cylinder_radius=float(cfg.task.get("cylinder_radius", 0.5)),
            target_orbit=float(cfg.task.get("orbit_radius", 2.0)),
            target_depth=float(cfg.task.get("orbit_z", actual_traj_np[:, 2].mean())),
        )
        print(f"[eval] saved plot data: {plot_data_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
