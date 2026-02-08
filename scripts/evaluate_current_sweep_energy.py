import os
import sys
from pathlib import Path

# Ensure this repo's `marinegym/` is importable when running via `python scripts/...py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MARINEGYM_ROOT", str(_REPO_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore registration)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_dir(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size != 3:
        raise ValueError(f"current_dir must be length-3. got shape={v.shape}")
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError(f"current_dir must be non-zero. got {v}")
    return (v / n).astype(np.float32, copy=False)


def _apply_constant_current(base_env, *, current_vec_w: np.ndarray, env_ids: torch.Tensor | None = None) -> None:
    """Force a constant world-frame current velocity on the drone (linear XYZ only)."""
    if base_env is None or not hasattr(base_env, "drone"):
        return
    drone = base_env.drone
    if not hasattr(drone, "flow_vels"):
        return

    if env_ids is None:
        try:
            env_ids = torch.arange(int(base_env.num_envs), device=drone.flow_vels.device)
        except Exception:
            env_ids = None

    v = torch.as_tensor(current_vec_w, device=drone.flow_vels.device, dtype=drone.flow_vels.dtype).view(1, 3)

    try:
        if hasattr(drone, "flow_noise_scale"):
            drone.flow_noise_scale[env_ids] = 0.0
    except Exception:
        pass
    try:
        if hasattr(drone, "max_flow_vel"):
            drone.max_flow_vel[env_ids] = 0.0
    except Exception:
        pass

    try:
        # flow_vels: (num_envs, 1, 6) by default.
        drone.flow_vels[env_ids, 0, 0:3] = v.expand(int(env_ids.numel()), 3)
        drone.flow_vels[env_ids, 0, 3:6] = 0.0
    except Exception:
        try:
            # Fallback: broadcast to all envs/robots if indexing differs.
            drone.flow_vels[..., 0:3] = v.view(1, 1, 3)
            drone.flow_vels[..., 3:6] = 0.0
        except Exception:
            return


def _compute_energy_step_from_throttle(u_cmd, *, env_id: int = 0) -> float:
    """Energy proxy: P_total(t) = Σ_i |u_i|^3."""
    if u_cmd is None:
        return float("nan")
    try:
        if torch.is_tensor(u_cmd):
            u_t = u_cmd
            if u_t.ndim >= 3:
                u_t = u_t.squeeze(1)
            u_env = u_t[int(env_id)]
            return float((u_env.abs() ** 3).sum().item())
        u_np = np.asarray(u_cmd)
        if u_np.ndim >= 3:
            u_np = np.squeeze(u_np, axis=1)
        u_env = np.asarray(u_np[int(env_id)], dtype=np.float32)
        return float(np.sum(np.abs(u_env) ** 3))
    except Exception:
        return float("nan")


def _plot_current_vs_energy(
    *,
    out_path: Path,
    current_speeds: np.ndarray,
    energy_mean: np.ndarray,
    energy_std: np.ndarray | None,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    plt.figure(figsize=(7.5, 4.8))
    if energy_std is not None:
        plt.errorbar(
            current_speeds,
            energy_mean,
            yerr=energy_std,
            fmt="o-",
            linewidth=2.0,
            markersize=5,
            capsize=3,
        )
    else:
        plt.plot(current_speeds, energy_mean, "o-", linewidth=2.0, markersize=5)

    plt.xlabel("Current velocity |v| [m/s]")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    plt.close()


FILE_PATH = os.path.dirname(__file__)


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Sweep current velocity and plot episodic energy consumption.

    Example:
      ~/isaac410/python.sh scripts/evaluate_current_sweep_energy.py \\
        task=OrbitCylinder_MPC_PPO algo=ppo headless=true enable_livestream=false env.num_envs=1 mode=evaluate \\
        +eval.ckpt=/path/to/checkpoint_final.pt +eval.steps=4000 \\
        +eval.current_speeds='[0.0, 0.05, 0.10, 0.15, 0.20]' +eval.current_dir='[1, 0, 0]' \\
        +eval.episodes_per_speed=3
    """

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_cfg = cfg.get("eval", {}) if hasattr(cfg, "get") else getattr(cfg, "eval", {})
    mode = str(cfg.get("mode", "")).lower() if hasattr(cfg, "get") else str(getattr(cfg, "mode", "")).lower()
    if mode != "evaluate":
        print(f"[sweep] mode={mode!r} (set mode=evaluate for default saving).")

    controller = str(eval_cfg.get("controller", "auto")).strip().lower()
    if controller in ("", "default"):
        controller = "auto"
    if controller == "auto":
        controller = "policy" if bool(eval_cfg.get("ckpt", "")) else "internal_mpc"
    if controller not in ("policy", "internal_mpc"):
        raise ValueError(f"eval.controller must be one of: auto, policy, internal_mpc. got {controller!r}")

    # Keep behavior consistent with scripts/train.py and scripts/evaluate.py:
    # - PPO on OrbitCylinderMPC uses *direct* thruster control (policy actions).
    # - internal MPC baseline uses *mpc* mode (env overrides actions).
    if str(getattr(cfg, "task", {}).get("name", "")) == "OrbitCylinderMPC":
        if controller == "policy":
            cfg.task.control_mode = "direct"
            cfg.task.use_internal_mpc = False
        if controller == "internal_mpc":
            cfg.task.control_mode = "mpc"
            cfg.task.use_internal_mpc = True

    ckpt = eval_cfg.get("ckpt", None)
    if controller == "policy" and not ckpt:
        raise ValueError("Missing checkpoint for policy eval: pass +eval.ckpt=/path/to/checkpoint_final.pt")

    steps = int(eval_cfg.get("steps", 4000))
    episodes_per_speed = int(eval_cfg.get("episodes_per_speed", 1))
    traj_env_id = int(eval_cfg.get("traj_env_id", 0))
    seed = int(eval_cfg.get("seed", cfg.get("seed", 0)))

    current_speeds_cfg = eval_cfg.get("current_speeds", None)
    if current_speeds_cfg is None:
        current_speeds_cfg = [0.0, 0.05, 0.10, 0.15, 0.20]
    current_speeds = np.asarray(current_speeds_cfg, dtype=np.float32).reshape(-1)
    current_dir = _normalize_dir(eval_cfg.get("current_dir", [1.0, 0.0, 0.0]))

    out_npz_cfg = eval_cfg.get("current_sweep_npz", None)
    out_png_cfg = eval_cfg.get("current_sweep_png", None)
    default_stub = "current_vs_energy" if controller == "policy" else "current_vs_energy_mpc"
    out_npz = Path(str(out_npz_cfg)).expanduser() if out_npz_cfg else Path.cwd() / f"{default_stub}.npz"
    out_png = Path(str(out_png_cfg)).expanduser() if out_png_cfg else Path.cwd() / f"{default_stub}.png"
    if not out_npz.is_absolute():
        out_npz = Path.cwd() / out_npz
    if not out_png.is_absolute():
        out_png = Path.cwd() / out_png

    simulation_app = init_simulation_app(cfg)

    try:
        from marinegym.envs import register_tasks

        register_tasks()
        policy = None
        env = None
        base_env = None
        dummy_action = None

        if controller == "policy":
            from evaluate import load_checkpoint  # reuse robust checkpoint loader

            policy, env = load_checkpoint(str(ckpt), cfg, cfg.algo)
        else:
            from marinegym.envs.isaac_env import IsaacEnv
            from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv

            env_class = IsaacEnv.REGISTRY[cfg.task.name]
            base_env = env_class(cfg, headless=cfg.headless)
            env = TransformedEnv(base_env, Compose(InitTracker())).eval()
            dummy_action = env.action_spec.zero()

        env.eval()
        env.set_seed(seed)
        env.enable_render(bool(eval_cfg.get("render", not bool(getattr(cfg, "headless", True)))))
        base_env = getattr(env, "base_env", None)
        try:
            task_control_mode = cfg.task.get("control_mode", None)
            task_use_internal = cfg.task.get("use_internal_mpc", None)
        except Exception:
            task_control_mode = getattr(getattr(cfg, "task", None), "control_mode", None)
            task_use_internal = getattr(getattr(cfg, "task", None), "use_internal_mpc", None)
        base_use_internal = getattr(base_env, "_use_internal_mpc", None)
        print(
            f"[sweep] controller={controller!r} seed={seed} control_mode={task_control_mode!r} task.use_internal_mpc={task_use_internal!r} "
            f"base_env._use_internal_mpc={base_use_internal!r}"
        )

        energy_means = []
        energy_stds = []
        energy_all = []

        with set_exploration_type(ExplorationType.MODE):
            for v in current_speeds.tolist():
                v_vec = float(v) * current_dir
                ep_energies = []
                for ep in range(max(1, int(episodes_per_speed))):
                    td = env.reset()
                    _apply_constant_current(base_env, current_vec_w=v_vec)
                    ep_energy = 0.0
                    last_action = None

                    for t in range(int(steps)):
                        if controller == "policy":
                            td = policy(td)
                            try:
                                last_action = td.get(("agents", "action"), None)
                            except Exception:
                                last_action = None
                            td = env.step(td)["next"]
                        else:
                            if hasattr(dummy_action, "items"):
                                td.update(dummy_action)
                            else:
                                td.set(("agents", "action"), dummy_action)
                            td_out = env.step(td)
                            try:
                                last_action = td_out.get(("agents", "action"), None)
                            except Exception:
                                last_action = None
                            td = td_out["next"]

                        # Energy proxy from the *applied* thruster command (after env.step()).
                        u_cmd = getattr(getattr(base_env, "drone", None), "throttle", None)
                        if u_cmd is None:
                            u_cmd = last_action
                        e_step = _compute_energy_step_from_throttle(u_cmd, env_id=int(traj_env_id))
                        if np.isfinite(e_step):
                            ep_energy += float(e_step)

                        done = td.get("done", None)
                        if done is not None and bool(done.any()):
                            # Reset envs that are done, then re-apply the same current.
                            try:
                                done_mask = done.squeeze(-1)
                                td.set("_reset", done_mask)
                                td = env.reset(td)
                            except Exception:
                                td = env.reset()
                            _apply_constant_current(base_env, current_vec_w=v_vec)
                            break

                    ep_energies.append(float(ep_energy))

                ep_arr = np.asarray(ep_energies, dtype=np.float32)
                energy_all.append(ep_arr)
                energy_means.append(float(np.mean(ep_arr)))
                energy_stds.append(float(np.std(ep_arr, ddof=1)) if ep_arr.size >= 2 else float("nan"))
                print(f"[sweep] |v|={float(v):.3f} m/s  episodes={ep_arr.size}  energy_mean={energy_means[-1]:.6g}")

        energy_means_arr = np.asarray(energy_means, dtype=np.float32)
        energy_stds_arr = np.asarray(energy_stds, dtype=np.float32)
        energy_total_label = "Σ_t Σ_i |u_i|^3"

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            current_speeds=current_speeds,
            current_dir=current_dir,
            steps=int(steps),
            episodes_per_speed=int(episodes_per_speed),
            energy_mean=energy_means_arr,
            energy_std=energy_stds_arr,
            energy_per_episode=np.array(energy_all, dtype=object),
            ylabel=energy_total_label,
            meta=np.array(
                [
                    {
                        "task": str(getattr(cfg, "task", {}).get("name", "")) if hasattr(cfg, "task") else "",
                        "algo": str(getattr(cfg, "algo", {}).get("name", "")) if hasattr(cfg, "algo") else "",
                        "controller": str(controller),
                        "ckpt": str(ckpt) if ckpt else "",
                        "seed": int(seed),
                    }
                ],
                dtype=object,
            ),
        )
        print(f"[sweep] saved results: {out_npz.resolve()}")

        _plot_current_vs_energy(
            out_path=out_png,
            current_speeds=current_speeds,
            energy_mean=energy_means_arr,
            energy_std=(energy_stds_arr if np.isfinite(energy_stds_arr).any() else None),
            ylabel=f"Episodic energy proxy ({energy_total_label})",
        )
        print(f"[sweep] saved plot: {out_png.resolve()}")
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
