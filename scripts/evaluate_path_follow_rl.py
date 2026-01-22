import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from torchrl.envs.utils import set_exploration_type, ExplorationType

from marinegym import init_simulation_app
from marinegym.learning import ALGOS
from marinegym.utils.torchrl import RenderCallback, EpisodeStats
from marinegym.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)

# ~/isaac410/python.sh scripts/evaluate_path_follow_rl.py   +eval.ckpt=/home/mjkim/MarineGym/wandb/offline-run-20260105_173236-v7x4sm9v/files/checkpoint_final.pt   task=PathFollow algo=ppo headless=false enable_livestream=false   +eval.num_episodes=10 env.num_envs=1 env.max_episode_length=3000


os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))

FILE_PATH = os.path.dirname(__file__)


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
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
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


def _load_policy_state(policy, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

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
        print("[load_checkpoint] shape mismatched keys -> use default init for them:")
        for key, s1, s2 in mismatched[:30]:
            print("  ", key, s1, "!=", s2)
        if len(mismatched) > 30:
            print("  ...", len(mismatched) - 30, "more")

    if missing:
        print("[load_checkpoint] filled missing keys:", missing[:30], ("..." if len(missing) > 30 else ""))

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


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Example:
      ~/isaac410/python.sh scripts/evaluate_path_follow.py \\
        +eval.ckpt=/path/to/checkpoint.pt task=PathFollow algo=ppo headless=false enable_livestream=false \\
        eval.num_episodes=10 env.num_envs=1
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_cfg = cfg.get("eval", {}) or {}
    ckpt_path = eval_cfg.get("ckpt", None)
    if not ckpt_path:
        raise ValueError("Missing checkpoint path. Set +eval.ckpt=/path/to/checkpoint.pt")

    num_episodes = int(eval_cfg.get("num_episodes", 10))
    seed = int(eval_cfg.get("seed", 0))
    render = bool(eval_cfg.get("render", not cfg.headless))
    render_interval = int(eval_cfg.get("render_interval", 2))
    video_path = eval_cfg.get("video_path", "")

    if num_episodes <= 0:
        raise ValueError("eval.num_episodes must be > 0")

    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    base_env, env = _build_env(cfg)
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )
    _load_policy_state(policy, ckpt_path, base_env.device)

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
