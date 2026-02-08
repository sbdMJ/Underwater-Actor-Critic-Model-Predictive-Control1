import os
import sys
from pathlib import Path

# Ensure Hydra searchpath `${oc.env:MARINEGYM_ROOT}` resolves.
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


def _as_vec3(x, *, default: np.ndarray) -> np.ndarray:
    if x is None:
        return np.asarray(default, dtype=np.float32).reshape(3,)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected a 3-vector, got shape={arr.shape} value={x!r}")
    return arr.astype(np.float32, copy=False)


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


def _energy_step_from_throttle(u_cmd, *, env_id: int = 0) -> float:
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


def _get_weight_newton(base_env, *, env_id: int = 0) -> float:
    """Robot weight (N). Includes payload mass if available."""
    if base_env is None or not hasattr(base_env, "drone"):
        return float("nan")
    drone = base_env.drone
    g = float(getattr(drone, "_g_mag", 9.81))

    m_drone = float("nan")
    try:
        m = _to_numpy(getattr(drone, "total_masses", None))
        if m is not None:
            m_drone = float(np.asarray(m)[int(env_id)].reshape(-1)[0])
    except Exception:
        m_drone = float("nan")
    if not np.isfinite(m_drone):
        try:
            m = _to_numpy(getattr(drone, "masses", None))
            if m is not None:
                m_drone = float(np.asarray(m)[int(env_id)].reshape(-1)[0])
        except Exception:
            m_drone = float("nan")

    m_payload = 0.0
    if hasattr(base_env, "payload") and getattr(base_env, "payload", None) is not None:
        try:
            masses = _to_numpy(base_env.payload.get_masses())
            if masses is not None:
                m_payload = float(np.asarray(masses)[int(env_id)].reshape(-1)[0])
        except Exception:
            m_payload = 0.0

    if not np.isfinite(m_drone):
        return float("nan")
    return float(g * (float(m_drone) + float(m_payload)))


def _plot_bar_two_conditions(
    *,
    out_path: Path,
    nominal: dict,
    harsh: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    labels = ["Nominal", "Harsh"]
    means = [float(nominal["cot_mean"]), float(harsh["cot_mean"])]
    stds = [float(nominal["cot_std"]), float(harsh["cot_std"])]

    x = np.arange(len(labels))
    plt.figure(figsize=(5.8, 4.0))
    plt.bar(x, means, yerr=stds, capsize=4, color=["#4e79a7", "#e15759"], alpha=0.9)
    plt.xticks(x, labels)
    plt.ylabel("CoT = TotalEnergy / (Weight * Distance)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    plt.close()


FILE_PATH = os.path.dirname(__file__)


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Compute CoT (Cost of Transport) under two current conditions:
      - Nominal: no current
      - Harsh: strong current

    CoT = TotalEnergy / (Weight * Distance)

    Notes:
      - "TotalEnergy" here uses an energy *proxy* based on thruster commands:
          P_total(t) = Σ_i |u_i|^3
          TotalEnergy = Σ_t P_total(t) * dt_eff
      - Fix payload randomness by pinning the range:
          task.disturbances.evaluate.payload.enable_payload=true
          task.disturbances.evaluate.payload.mass='[m, m]'
          task.disturbances.evaluate.payload.z='[z, z]'

    Example:
      ~/isaac410/python.sh scripts/evaluate_cot_nominal_harsh.py \\
        task=OrbitCylinder_MPC_PPO algo=ppo headless=true enable_livestream=false env.num_envs=1 mode=evaluate \\
        +eval.ckpt=/path/to/checkpoint_final.pt +eval.steps=4000 +eval.episodes=5 \\
        +eval.harsh_speed=0.2 +eval.current_dir='[1,0,0]'

      # Internal MPC baseline (no checkpoint):
      ~/isaac410/python.sh scripts/evaluate_cot_nominal_harsh.py \\
        task=OrbitCylinder_MPC algo=ppo headless=true enable_livestream=false env.num_envs=1 mode=evaluate \\
        task.use_pypose_mpc=false +eval.controller=internal_mpc +eval.steps=4000 +eval.episodes=5 \\
        +eval.harsh_speed=0.2 +eval.current_dir='[1,0,0]'
    """

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_cfg = cfg.get("eval", {}) if hasattr(cfg, "get") else getattr(cfg, "eval", {})
    mode = str(cfg.get("mode", "")).lower() if hasattr(cfg, "get") else str(getattr(cfg, "mode", "")).lower()
    if mode != "evaluate":
        print(f"[cot] mode={mode!r} (set mode=evaluate for default saving).")

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

    steps_per_episode = int(eval_cfg.get("steps", 4000))
    episodes = int(eval_cfg.get("episodes", eval_cfg.get("num_episodes", 5)))
    traj_env_id = int(eval_cfg.get("traj_env_id", 0))
    seed = int(eval_cfg.get("seed", cfg.get("seed", 0)))

    current_dir = _normalize_dir(eval_cfg.get("current_dir", [1.0, 0.0, 0.0]))
    nominal_speed = float(eval_cfg.get("nominal_speed", eval_cfg.get("cot_nominal_speed", 0.0)))
    harsh_speed = float(eval_cfg.get("harsh_speed", eval_cfg.get("cot_harsh_speed", 0.2)))
    nominal_vec = _as_vec3(eval_cfg.get("nominal_current_vec", None), default=nominal_speed * current_dir)
    harsh_vec = _as_vec3(eval_cfg.get("harsh_current_vec", None), default=harsh_speed * current_dir)

    out_npz_cfg = eval_cfg.get("cot_npz", None)
    out_png_cfg = eval_cfg.get("cot_png", None)
    default_stub = "cot_nominal_harsh" if controller == "policy" else "cot_nominal_harsh_mpc"
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
            f"[cot] controller={controller!r} seed={seed} control_mode={task_control_mode!r} task.use_internal_mpc={task_use_internal!r} "
            f"base_env._use_internal_mpc={base_use_internal!r}"
        )

        dt_eff = float(getattr(base_env, "dt", float(getattr(cfg, "sim", {}).get("dt", 0.0))))
        substeps = int(getattr(base_env, "substeps", int(getattr(getattr(cfg, "sim", None), "substeps", 1) or 1)))
        dt_eff = float(dt_eff) * float(max(1, substeps))

        def _run_condition(*, name: str, current_vec_w: np.ndarray) -> dict:
            cot_list = []
            energy_list = []
            dist_list = []
            weight_list = []
            steps_list = []

            with set_exploration_type(ExplorationType.MODE):
                for ep in range(int(episodes)):
                    td = env.reset()
                    _apply_constant_current(base_env, current_vec_w=current_vec_w)

                    # Weight may change per reset if payload is randomized.
                    weight_n = _get_weight_newton(base_env, env_id=int(traj_env_id))

                    try:
                        base_env.drone.get_state()
                        pos_prev = base_env.drone.pos[int(traj_env_id), 0, :].detach().cpu()
                    except Exception:
                        pos_prev = None

                    energy_total = 0.0
                    dist_total = 0.0
                    last_action = None
                    steps_taken = 0

                    for t in range(int(steps_per_episode)):
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
                        steps_taken += 1

                        # Distance from positions after env.step().
                        try:
                            base_env.drone.get_state()
                            pos = base_env.drone.pos[int(traj_env_id), 0, :].detach().cpu()
                            if pos_prev is not None:
                                dist_total += float(torch.linalg.norm(pos - pos_prev).item())
                            pos_prev = pos
                        except Exception:
                            pass

                        # Energy proxy from applied thruster command (after env.step()).
                        u_cmd = getattr(getattr(base_env, "drone", None), "throttle", None)
                        if u_cmd is None:
                            u_cmd = last_action
                        p_step = _energy_step_from_throttle(u_cmd, env_id=int(traj_env_id))
                        if np.isfinite(p_step):
                            energy_total += float(p_step) * float(dt_eff)

                        done = td.get("done", None)
                        if done is not None:
                            try:
                                done_mask = done.squeeze(-1)
                                if bool(done_mask[int(traj_env_id)].item()):
                                    break
                            except Exception:
                                if bool(done.any()):
                                    break

                    denom = float(weight_n) * float(dist_total)
                    cot = float(energy_total / denom) if denom > 0.0 else float("nan")

                    cot_list.append(cot)
                    energy_list.append(float(energy_total))
                    dist_list.append(float(dist_total))
                    weight_list.append(float(weight_n))
                    steps_list.append(int(steps_taken))

                    print(
                        f"[cot] {name} ep={ep} steps={steps_taken} "
                        f"energy={energy_total:.6g} dist={dist_total:.6g} weight={weight_n:.6g} cot={cot:.6g}"
                    )

            cot_arr = np.asarray(cot_list, dtype=np.float64)
            energy_arr = np.asarray(energy_list, dtype=np.float64)
            dist_arr = np.asarray(dist_list, dtype=np.float64)
            weight_arr = np.asarray(weight_list, dtype=np.float64)
            steps_arr = np.asarray(steps_list, dtype=np.int64)

            return {
                "name": str(name),
                "current_vec_w": np.asarray(current_vec_w, dtype=np.float32).reshape(3,),
                "episodes": int(episodes),
                "steps_per_episode": int(steps_per_episode),
                "dt_eff": float(dt_eff),
                "cot": cot_arr,
                "energy_total": energy_arr,
                "distance": dist_arr,
                "weight": weight_arr,
                "steps_taken": steps_arr,
                "cot_mean": float(np.nanmean(cot_arr)),
                "cot_std": float(np.nanstd(cot_arr, ddof=1)) if int(cot_arr.size) >= 2 else float("nan"),
            }

        nominal = _run_condition(name="nominal", current_vec_w=nominal_vec)
        harsh = _run_condition(name="harsh", current_vec_w=harsh_vec)

        print(
            f"[cot] nominal: mean={nominal['cot_mean']:.6g} std={nominal['cot_std']:.6g} "
            f"(current={nominal_vec.tolist()})"
        )
        print(
            f"[cot] harsh:   mean={harsh['cot_mean']:.6g} std={harsh['cot_std']:.6g} "
            f"(current={harsh_vec.tolist()})"
        )
        try:
            ratio = float(harsh["cot_mean"]) / float(nominal["cot_mean"])
        except Exception:
            ratio = float("nan")
        print(f"[cot] harsh/nominal ratio: {ratio:.6g}")

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            nominal=nominal,
            harsh=harsh,
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
        print(f"[cot] saved results: {out_npz.resolve()}")

        try:
            _plot_bar_two_conditions(out_path=out_png, nominal=nominal, harsh=harsh)
            print(f"[cot] saved plot: {out_png.resolve()}")
        except Exception as exc:
            print(f"[cot] failed to plot: {exc}")
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
