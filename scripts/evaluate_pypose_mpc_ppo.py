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
      - Saves `trajectory.npz` and `trajectory.png` under Hydra's run dir (./outputs/...). (speed is shown as colormap if available)
      - Disable with `+eval.save_traj=false`.
      - Override paths with `+eval.traj_path=/tmp/trajectory.npz +eval.traj_png_path=/tmp/trajectory.png`.
      - Re-visualize later: `python scripts/visualize_trajectory.py /path/to/trajectory.npz`

    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

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
    done_log = []
    step_log = []

    def _maybe_log_traj(*, step: int, done: bool) -> None:
        if not save_traj:
            return
        if (int(step) % int(traj_stride)) != 0:
            return
        try:
            base_env.drone.get_state()
            pos_log.append(_to_numpy(base_env.drone.pos[traj_env_id, 0, :]).astype(np.float32, copy=False))
            heading_log.append(_to_numpy(base_env.drone.heading[traj_env_id, 0, :]).astype(np.float32, copy=False))
            try:
                vel_lin_w = base_env.drone.vel_w[traj_env_id, 0, 0:3]
                speed_log.append(float(torch.linalg.norm(vel_lin_w).item()))
            except Exception:
                speed_log.append(float("nan"))
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
            done_arr = np.asarray(done_log, dtype=np.bool_)
            step_arr = np.asarray(step_log, dtype=np.int64)

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
                done=done_arr,
                cylinder_center=center,
                cylinder_radius=float(getattr(base_env, "cylinder_radius", 0.0)),
                cylinder_height=float(getattr(base_env, "cylinder_height", 0.0)),
                orbit_radius=float(getattr(base_env, "orbit_radius", 0.0)),
                orbit_z=float(getattr(base_env, "orbit_z", 0.0)),
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
                        }
                    ],
                    dtype=object,
                ),
            )
            tmp_path.replace(traj_path)
            print(f"[eval] saved trajectory: {traj_path.resolve()}")

            if plot_traj:
                try:
                    from visualize_trajectory import plot_trajectory_3d

                    plot_trajectory_3d(
                        traj_path=traj_path,
                        out_path=traj_png_path,
                        heading_stride=int(plot_heading_stride),
                        arrow_len=float(plot_arrow_len),
                        show=bool(plot_show),
                    )
                    print(f"[eval] saved trajectory plot: {traj_png_path.resolve()}")
                except Exception as exc:
                    print(f"[eval] failed to plot trajectory: {exc}")
        except Exception as exc:
            print(f"[eval] failed to save trajectory: {exc}")

    simulation_app.close()


if __name__ == "__main__":
    main()
