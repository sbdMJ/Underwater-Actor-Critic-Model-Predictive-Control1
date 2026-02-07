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
def _maybe_compute_orbit_cost(cfg, base_env, u_cmd):
    try:
        from marinegym.controllers.pypose_cylinder_orbit_mpc_controller import _orbit_errors
    except Exception:
        return None

    try:
        base_env.drone.get_state()
        root_state = torch.cat([base_env.drone.pos, base_env.drone.rot, base_env.drone.vel_b], dim=-1).squeeze(1)
    except Exception:
        return None

    try:
        dtype = root_state.dtype
        device = root_state.device
        center_env = base_env.cylinder_center.squeeze(0).expand(root_state.shape[0], 3).to(device=device, dtype=dtype)
        e_k = _orbit_errors(
            root_state,
            center_w=center_env,
            radius=torch.as_tensor(float(getattr(base_env, "orbit_radius", 0.0)), device=device, dtype=dtype),
            z=torch.as_tensor(float(getattr(base_env, "orbit_z", 0.0)), device=device, dtype=dtype),
            v_tan=torch.as_tensor(float(getattr(base_env, "orbit_v_tan", 0.0)), device=device, dtype=dtype),
            dir_sign=torch.as_tensor(float(getattr(base_env, "orbit_direction", 1.0)), device=device, dtype=dtype),
            yaw_offset=torch.as_tensor(float(getattr(base_env, "orbit_yaw_offset", 0.0)), device=device, dtype=dtype),
        )
    except Exception:
        return None

    try:
        q = cfg.task
        q_radial = float(q.get("mpc_q_radial", 50.0))
        q_z = float(q.get("mpc_q_z", 30.0))
        q_tan = float(q.get("mpc_q_tan", 10.0))
        q_radial_speed = float(q.get("mpc_q_radial_speed", 5.0))
        q_heading = float(q.get("mpc_q_heading", 30.0))
        q_roll = float(q.get("mpc_q_roll", 60.0))
        q_pitch = float(q.get("mpc_q_pitch", 60.0))
        q_wxy = float(q.get("mpc_q_wxy", 0.5))
        w_err = torch.as_tensor(
            [
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
            ],
            device=e_k.device,
            dtype=e_k.dtype,
        ).view(1, 10)
        r_u = float(q.get("mpc_r_u", 0.01))
        max_thruster_force = float(q.get("pypose_max_thruster_force", q.get("mpc_max_thruster_force", 40.0)))
        if u_cmd is None:
            cost_u = torch.zeros((e_k.shape[0],), device=e_k.device, dtype=e_k.dtype)
        else:
            u_cmd = u_cmd.to(device=e_k.device, dtype=e_k.dtype)
            if u_cmd.ndim >= 3:
                u_cmd = u_cmd.squeeze(1)
            w_u = torch.full(
                (1, int(u_cmd.shape[-1])),
                float(r_u) * float(max_thruster_force**2),
                device=u_cmd.device,
                dtype=u_cmd.dtype,
            )
            cost_u = (w_u * u_cmd.square()).sum(dim=-1)
        cost_err = (w_err * e_k.square()).sum(dim=-1)
        cost = 0.5 * (cost_err + cost_u)
        return cost
    except Exception:
        return None


def evaluate_model(env, policy, num_episodes, cfg):
    from torchrl.envs.utils import set_exploration_type, ExplorationType

    results = []
    env.eval()
    env.set_seed(0)
    env.enable_render(True)
    max_steps = int(getattr(env.base_env, "max_episode_length", 0) or 0)
    try:
        if hasattr(cfg, "eval") and cfg.eval is not None and hasattr(cfg.eval, "steps") and int(cfg.eval.steps) > 0:
            max_steps = int(cfg.eval.steps)
    except Exception:
        pass
    print_every = 200
    try:
        if hasattr(cfg, "eval") and cfg.eval is not None and hasattr(cfg.eval, "print_every"):
            print_every = int(cfg.eval.print_every)
    except Exception:
        pass

    for _ in tqdm(range(num_episodes)):
        episode_stats = []
        td = env.reset()
        last_action = None
        with set_exploration_type(ExplorationType.MODE):
            for t in range(max_steps):
                if print_every > 0 and (t % print_every) == 0:
                    r = td.get(("stats", "return"), None)
                    pe = td.get(("stats", "pos_error"), None)
                    cost = _maybe_compute_orbit_cost(cfg, env.base_env, last_action)
                    if r is not None and pe is not None:
                        if cost is None:
                            print(
                                f"[eval] t={t} return={float(r.mean().item()):.4g} pos_error={float(pe.mean().item()):.4g}"
                            )
                        else:
                            print(
                                f"[eval] t={t} return={float(r.mean().item()):.4g} pos_error={float(pe.mean().item()):.4g} cost={float(cost.mean().item()):.4g}"
                            )

                td = policy(td)
                try:
                    last_action = td.get(("agents", "action"), None)
                except Exception:
                    last_action = None
                td = env.step(td)["next"]

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
        if episode_stats:
            stats_td = torch.stack(episode_stats).to_tensordict()
            results.append(stats_td)
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
