import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from marinegym import init_simulation_app
from marinegym.learning import ALGOS
from marinegym.utils.torchrl import EpisodeStats, RenderCallback
from marinegym.utils.torchrl.transforms import (
    FromDiscreteAction,
    FromMultiDiscreteAction,
    ravel_composite,
)

os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))

FILE_PATH = os.path.dirname(__file__)


def _guess_task_algo_from_wandb_debug(run_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    debug_path = run_dir / "logs" / "debug.log"
    if not debug_path.exists():
        return None, None
    try:
        text = debug_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None

    task_match = re.search(r"'task\\.name':\\s*'([^']+)'", text)
    algo_match = re.search(r"'algo\\.name':\\s*'([^']+)'", text)
    task_name = task_match.group(1) if task_match else None
    algo_name = algo_match.group(1) if algo_match else None
    return task_name, algo_name


def _resolve_run_paths(eval_cfg: dict) -> Tuple[Optional[Path], Optional[Path]]:
    run_dir_raw = eval_cfg.get("run_dir", "") or ""
    run_dir = Path(run_dir_raw).expanduser() if run_dir_raw else None

    ckpt_raw = eval_cfg.get("ckpt", "") or ""
    ckpt_path = Path(ckpt_raw).expanduser() if ckpt_raw else None

    if ckpt_path is None and run_dir is not None:
        candidates = [
            run_dir / "files" / "checkpoints" / "checkpoint_500224.pt",
            run_dir / "checkpoint_final.pt",
            run_dir / "files" / "files" / "checkpoint_final.pt",
        ]
        for cand in candidates:
            if cand.exists():
                ckpt_path = cand
                break

    return run_dir, ckpt_path


def _ensure_mpc_files(cfg, base_env, run_dir: Optional[Path]):
    algo_name = str(cfg.algo.name).lower()
    if algo_name not in ("ac_mpc", "ac_mpc_pypose"):
        return

    from marinegym.controllers.thruster_allocation import (
        compute_thruster_allocation_matrix_from_drone,
    )

    if not cfg.algo.get("mpc_param_yaml", None):
        if hasattr(base_env, "drone") and hasattr(base_env.drone, "param_path"):
            cfg.algo.mpc_param_yaml = base_env.drone.param_path

    if not cfg.algo.get("mpc_alloc_npz", None):
        candidates: List[Path] = []
        if run_dir is not None:
            candidates.extend(
                [
                    run_dir / "files" / algo_name / "thruster_allocation.npz",
                    run_dir / algo_name / "thruster_allocation.npz",
                ]
            )
        for cand in candidates:
            if cand.exists():
                cfg.algo.mpc_alloc_npz = str(cand)
                break

    if not cfg.algo.get("mpc_alloc_npz", None):
        thrust_axis = int(cfg.task.get("thrust_axis", 0))
        B = compute_thruster_allocation_matrix_from_drone(
            base_env.drone, thrust_axis=thrust_axis
        )
        with tempfile.NamedTemporaryFile(prefix="marinegym_thruster_alloc_", suffix=".npz", delete=False) as tmp:
            np.savez(tmp.name, B=B, quat_order="wxyz")
            cfg.algo.mpc_alloc_npz = tmp.name

    cfg.algo.mpc_dt = float(cfg.sim.dt)

    try:
        cfg.algo.obs_time_encoding_dim = (
            int(getattr(base_env, "time_encoding_dim", 0)) if bool(cfg.task.time_encoding) else 0
        )
    except Exception:
        cfg.algo.obs_time_encoding_dim = 0

    try:
        cfg.algo.obs_has_target_quat = bool(
            getattr(base_env, "include_target_quat_in_obs", False)
        )
    except Exception:
        cfg.algo.obs_has_target_quat = False

    if hasattr(base_env, "drone"):
        cfg.algo.mpc_nu = int(base_env.drone.action_spec.shape[-1])
        try:
            cfg.algo.mpc_mass = float(base_env.drone.MASS_0.squeeze().item())
            cfg.algo.mpc_inertia = [
                float(x) for x in base_env.drone.INERTIA_0.squeeze().tolist()
            ]
        except Exception:
            pass
    else:
        try:
            action_leaf = base_env.action_spec[("agents", "action")]
        except Exception:
            action_leaf = base_env.action_spec
        cfg.algo.mpc_nu = int(action_leaf.shape[-1])

    cfg.algo.use_mpc_head = True


def _build_env(cfg):
    from marinegym.envs import IsaacEnv, register_tasks

    register_tasks()

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(
            base_env.observation_spec, ("agents", "observation_central")
        )
        transforms.append(transform)

    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    return base_env, env


def _load_policy_state(policy, checkpoint_path: Path, device):
    ckpt = torch.load(str(checkpoint_path), map_location=device)

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
            if (
                hasattr(value, "shape")
                and hasattr(base_sd[key], "shape")
                and value.shape != base_sd[key].shape
            ):
                mismatched.append((key, tuple(value.shape), tuple(base_sd[key].shape)))
                continue
            merged[key] = value

    missing = [key for key in base_sd.keys() if key not in merged]
    for key in missing:
        merged[key] = base_sd[key]

    if mismatched:
        print("[load_checkpoint] shape mismatched keys -> use default init for them:")
        for key, s1, s2 in mismatched[:30]:
            print("  ", key, s1, "!=", s2)
        if len(mismatched) > 30:
            print("  ...", len(mismatched) - 30, "more")

    if missing:
        print(
            "[load_checkpoint] filled missing keys:",
            missing[:30],
            ("..." if len(missing) > 30 else ""),
        )

    policy.load_state_dict(merged, strict=True)
    policy.eval()


@torch.no_grad()
def _evaluate_policy(env, base_env, policy, num_episodes, seed, render, render_interval):
    stats_keys = [
        key
        for key in base_env.observation_spec.keys(True, True)
        if isinstance(key, tuple) and key[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    base_env.enable_render(render)
    env.eval()
    env.set_seed(seed)

    render_cb = RenderCallback(interval=render_interval) if render else None

    while len(episode_stats) < num_episodes:
        with set_exploration_type(ExplorationType.MODE):
            traj = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_cb,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        episode_stats.add(traj)

    stats = episode_stats.pop()
    if stats.shape[0] > num_episodes:
        stats = stats[:num_episodes]

    return stats, render_cb


def _summarize_stats(stats):
    summary = {}
    for key, value in stats.items(True, True):
        if isinstance(key, tuple) and key and key[0] == "stats":
            label = "stats." + str(key[1])
        else:
            label = ".".join(key) if isinstance(key, tuple) else str(key)
        summary[label] = value.float().mean().item()
    return summary


@torch.no_grad()
def _debug_print_qp(env, policy, steps: int, horizon_steps: int, dims: int):
    if steps <= 0:
        return

    actor_td_module = getattr(policy, "actor", None)
    actor_td_module = getattr(actor_td_module, "module", None)
    actor_core = getattr(actor_td_module, "module", None)
    if actor_core is None or not (hasattr(actor_core, "feature_net") and hasattr(actor_core, "cost_map")):
        print("[eval.print_qp] Actor does not expose feature_net/cost_map; skipping.")
        return

    td = env.reset()
    for step_idx in range(steps):
        obs = td.get(("agents", "observation"), None)
        if obs is None:
            raise KeyError("Missing ('agents','observation') in tensordict.")

        obs2 = obs.reshape(-1, obs.shape[-1])
        feat = actor_core.feature_net(obs2)
        q_seq, p_seq = actor_core.cost_map(feat)

        q_first = q_seq.reshape(-1, q_seq.shape[-2], q_seq.shape[-1])[0]
        p_first = p_seq.reshape(-1, p_seq.shape[-2], p_seq.shape[-1])[0]

        hs = min(int(horizon_steps), int(q_first.shape[0]))
        dk = min(int(dims), int(q_first.shape[1]))
        print(f"[eval.print_qp][step {step_idx}] q_seq/p_seq (first sample): horizon={q_first.shape[0]} dim={q_first.shape[1]}")
        for t in range(hs):
            qv = q_first[t, :dk].detach().cpu().numpy()
            pv = p_first[t, :dk].detach().cpu().numpy()
            print(f"  t={t:02d} q[:{dk}]={np.array2string(qv, precision=5, separator=', ')}")
            print(f"  t={t:02d} p[:{dk}]={np.array2string(pv, precision=5, separator=', ')}")

        with set_exploration_type(ExplorationType.MODE):
            policy(td)
        td = env.step(td)["next"]

        done = td.get("done", None)
        if done is not None and bool(done.any()):
            reset_mask = done.squeeze(-1)
            td.set("_reset", reset_mask)
            td = env.reset(td)


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Example (evaluate the provided offline run):
      ~/isaac410/python.sh scripts/evaluate_acmpc_pypose.py \\
        +eval.run_dir=/home/mjkim/MarineGym/wandb/offline-run-20260113_133253-t7yfdx19 \\
        task=Hover algo=ac_mpc_pypose headless=true enable_livestream=false \\
        eval.num_episodes=10 env.num_envs=1
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_cfg = cfg.get("eval", {}) or {}
    run_dir, ckpt_path = _resolve_run_paths(eval_cfg)
    if ckpt_path is None:
        raise ValueError(
            "Missing checkpoint. Set +eval.ckpt=/path/to/checkpoint.pt or +eval.run_dir=/path/to/wandb/offline-run-*"
        )

    if run_dir is not None:
        guessed_task, guessed_algo = _guess_task_algo_from_wandb_debug(run_dir)
        if guessed_task and str(cfg.task.name) != guessed_task:
            print(f"[wandb] run task={guessed_task} (cfg.task.name={cfg.task.name})")
        if guessed_algo and str(cfg.algo.name) != guessed_algo:
            print(f"[wandb] run algo={guessed_algo} (cfg.algo.name={cfg.algo.name})")

    num_episodes = int(eval_cfg.get("num_episodes", 10))
    seed = int(eval_cfg.get("seed", 0))
    render = bool(eval_cfg.get("render", not cfg.headless))
    render_interval = int(eval_cfg.get("render_interval", 2))
    video_path = str(eval_cfg.get("video_path", "") or "")

    if num_episodes <= 0:
        raise ValueError("eval.num_episodes must be > 0")

    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    base_env, env = _build_env(cfg)
    _ensure_mpc_files(cfg, base_env, run_dir)

    policy = ALGOS[str(cfg.algo.name).lower()](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )
    _load_policy_state(policy, ckpt_path, base_env.device)

    if bool(eval_cfg.get("print_qp", False)):
        _debug_print_qp(
            env=env,
            policy=policy,
            steps=int(eval_cfg.get("print_qp_steps", 1)),
            horizon_steps=int(eval_cfg.get("print_qp_horizon_steps", 3)),
            dims=int(eval_cfg.get("print_qp_dims", 12)),
        )

    stats, render_cb = _evaluate_policy(
        env, base_env, policy, num_episodes, seed, render, render_interval
    )

    summary = _summarize_stats(stats)
    print(f"episodes: {stats.shape[0]}")
    for key in sorted(summary.keys()):
        print(f"{key}: {summary[key]:.6f}")

    if video_path and render_cb is not None:
        from torchvision.io import write_video

        frames = render_cb.get_video_array(axes="t h w c")
        if frames.size > 0:
            fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps * render_interval)
            write_video(video_path, torch.as_tensor(frames), fps=fps)
            print(f"saved video: {video_path}")


if __name__ == "__main__":
    main()
