import logging
import os
import time
from pathlib import Path
import sys

# Ensure this repo's `marinegym/` is importable when running via `python scripts/evaluate.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import OmegaConf

from marinegym import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from marinegym.utils.torchrl import SyncDataCollector
from marinegym.utils.torchrl.transforms import (
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    # VelController,
    AttitudeController,
    RateController,
    History
)
from marinegym.utils.wandb import init_wandb
from marinegym.utils.torchrl import RenderCallback, EpisodeStats
from marinegym.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


import torch

os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))

def load_checkpoint(checkpoint_path, env_config, algo_config):
    from marinegym.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[env_config.task.name]
    base_env = env_class(env_config, headless=env_config.headless)

    transforms = [InitTracker()]
    env = TransformedEnv(base_env, Compose(*transforms)).eval()

    policy = ALGOS[algo_config.name.lower()](
        algo_config,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device
    )

    # -----------------------------
    # 1) checkpoint 로드 (+ unwrap)
    # -----------------------------
    ckpt = torch.load(checkpoint_path, map_location=base_env.device)

    # wandb/torch save 포맷이 dict wrapper인 경우가 많아서 처리
    if isinstance(ckpt, dict):
        for k in ("policy", "policy_state_dict", "model", "state_dict", "net"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like dict. type={type(ckpt)}")

    # -----------------------------------------
    # 2) 현재 policy state_dict 기준으로 정리
    #    - 없는 키는 추가
    #    - shape mismatch는 무시하고 기본값 사용
    # -----------------------------------------
    base_sd = policy.state_dict()
    merged = {}
    mismatched = []

    # checkpoint에서 가져올 수 있는 것만 가져오기
    for k, v in ckpt.items():
        if k in base_sd:
            if hasattr(v, "shape") and hasattr(base_sd[k], "shape") and v.shape != base_sd[k].shape:
                mismatched.append((k, tuple(v.shape), tuple(base_sd[k].shape)))
                continue
            merged[k] = v

    # 빠진 키는 policy 기본값으로 채우기 (여기가 "key 추가" 부분)
    missing = [k for k in base_sd.keys() if k not in merged]
    for k in missing:
        merged[k] = base_sd[k]

    if mismatched:
        print("[load_checkpoint] shape mismatched keys -> use default init for them:")
        for k, s1, s2 in mismatched[:30]:
            print("  ", k, s1, "!=", s2)
        if len(mismatched) > 30:
            print("  ...", len(mismatched) - 30, "more")

    print("[load_checkpoint] filled missing keys:", missing[:30], ("..." if len(missing) > 30 else ""))
    # strict=True로 이제 통과해야 정상
    policy.load_state_dict(merged, strict=True)
    policy.eval()

    return policy, env




# Evaluate the loaded model
def evaluate_model(env, policy, num_episodes, cfg):
    from torchrl.envs.utils import set_exploration_type, ExplorationType

    from marinegym.sensors.camera import Camera, PinholeCameraCfg
    from torchvision.io import write_video
    import dataclasses

    sim_dt = cfg.sim.dt

    results = []
    frames_vis = np.empty((0,cfg.viewer.resolution[1],cfg.viewer.resolution[0],3))
    env.eval()
    env.set_seed(0)
    env.enable_render(True)
    render_callback = RenderCallback(interval=1)
    max_steps = int(getattr(env.base_env, "max_episode_length", 0) or 0)
    try:
        if hasattr(cfg, "eval") and cfg.eval is not None and hasattr(cfg.eval, "steps") and int(cfg.eval.steps) > 0:
            max_steps = int(cfg.eval.steps)
    except Exception:
        pass
    for i in tqdm(range(num_episodes)):
        with set_exploration_type(ExplorationType.MODE):
            traj = env.rollout(
                max_steps=max_steps,
                policy=policy,
                auto_reset=True,
                break_when_any_done=False
            )
        results.append(traj["next", "stats"].cpu())
    return results

FILE_PATH = os.path.dirname(__file__)
@hydra.main(config_path=FILE_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if str(cfg.task.name) == "OrbitCylinderMPC" and str(cfg.algo.name).lower() == "ppo":
        cfg.task.control_mode = "direct"
        cfg.task.use_internal_mpc = False
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))
    from marinegym.envs import register_tasks
    register_tasks()
    policy, env = load_checkpoint(cfg.eval.ckpt, cfg, cfg.algo)
    eval_results = evaluate_model(env, policy, num_episodes=100, cfg=cfg)
    print(eval_results)

if __name__ == "__main__":
    main()
