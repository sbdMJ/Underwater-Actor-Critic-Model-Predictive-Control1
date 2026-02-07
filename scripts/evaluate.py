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
from tqdm import tqdm
from omegaconf import OmegaConf

from marinegym import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType
from marinegym.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _coerce_path(p, *, default: Path) -> Path:
    if p is None:
        return default
    path = Path(str(p)).expanduser()
    if path.is_dir():
        return path / default.name
    return path


def _get_task_meta(cfg) -> dict:
    task_cfg = cfg.get("task", None)
    if task_cfg is None:
        return {}
    task_dict = OmegaConf.to_container(task_cfg, resolve=True) if not isinstance(task_cfg, dict) else dict(task_cfg)
    out: dict = {}
    for k in (
        "cylinder_center",
        "cylinder_radius",
        "cylinder_height",
        "orbit_radius",
        "orbit_z",
        "orbit_direction",
        "orbit_v_tan",
        "orbit_yaw_offset",
    ):
        if k in task_dict:
            out[k] = task_dict[k]
    return out


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
def evaluate_model(env, policy, cfg):
    eval_cfg = cfg.get("eval", None)
    if eval_cfg is None:
        eval_cfg = {}
    eval_cfg = OmegaConf.to_container(eval_cfg, resolve=True) if not isinstance(eval_cfg, dict) else dict(eval_cfg)

    base_env = getattr(env, "base_env", None)
    if base_env is None:
        base_env = env

    seed = int(eval_cfg.get("seed", cfg.get("seed", 0) or 0))
    steps = int(eval_cfg.get("steps", 0) or 0)
    if steps <= 0:
        steps = int(getattr(base_env, "max_episode_length", 0) or 0)
    if steps <= 0:
        steps = 4000

    # Trajectory logging defaults:
    # - enabled by default when `mode=evaluate` (common usage), but can be disabled via `+eval.save_traj=false`.
    mode = str(cfg.get("mode", "") or "").lower()
    save_traj = bool(eval_cfg.get("save_traj", mode == "evaluate"))
    save_plot = bool(eval_cfg.get("save_plot", save_traj))
    env_idx = int(eval_cfg.get("traj_env_idx", 0))
    agent_idx = int(eval_cfg.get("traj_agent_idx", 0))
    traj_stride = int(eval_cfg.get("traj_stride", 1) or 1)
    traj_stride = max(1, traj_stride)

    run_dir = Path.cwd()
    traj_out = _coerce_path(eval_cfg.get("traj_out", None), default=run_dir / "bluerov_traj.npz")
    plot_out = _coerce_path(eval_cfg.get("plot_out", None), default=run_dir / "bluerov_traj_3d.png")

    env.eval()
    env.set_seed(seed)
    try:
        base_env.enable_render(True)
    except Exception:
        pass

    # Pre-allocate (steps+1 includes t=0 right after reset).
    T = int(steps) + 1
    pos = np.zeros((T, 3), dtype=np.float32)
    quat = np.zeros((T, 4), dtype=np.float32)  # wxyz
    heading = np.zeros((T, 3), dtype=np.float32)
    target_pos = np.full((T, 3), np.nan, dtype=np.float32)
    target_heading = np.full((T, 3), np.nan, dtype=np.float32)
    done = np.zeros((T,), dtype=np.bool_)
    episode_id = np.zeros((T,), dtype=np.int64)
    step_id = np.arange(T, dtype=np.int64)

    ep = 0
    td = env.reset()
    # record t=0
    try:
        if hasattr(base_env, "drone"):
            pos[0] = _to_numpy(base_env.drone.pos[env_idx, agent_idx])
            quat[0] = _to_numpy(base_env.drone.rot[env_idx, agent_idx])
            heading[0] = _to_numpy(base_env.drone.heading[env_idx, agent_idx])
        if hasattr(base_env, "target_pos"):
            target_pos[0] = _to_numpy(base_env.target_pos[env_idx, 0])
        if hasattr(base_env, "target_heading"):
            target_heading[0] = _to_numpy(base_env.target_heading[env_idx, 0])
    except Exception:
        pass

    with set_exploration_type(ExplorationType.MODE):
        for t in tqdm(range(1, T), desc="Evaluating"):
            with torch.no_grad():
                td = policy(td)
            td = env.step(td)["next"]

            # record every step but allow thinning via traj_stride (still keeps array length T for easy indexing).
            if (t % traj_stride) == 0:
                try:
                    if hasattr(base_env, "drone"):
                        pos[t] = _to_numpy(base_env.drone.pos[env_idx, agent_idx])
                        quat[t] = _to_numpy(base_env.drone.rot[env_idx, agent_idx])
                        heading[t] = _to_numpy(base_env.drone.heading[env_idx, agent_idx])
                    if hasattr(base_env, "target_pos"):
                        target_pos[t] = _to_numpy(base_env.target_pos[env_idx, 0])
                    if hasattr(base_env, "target_heading"):
                        target_heading[t] = _to_numpy(base_env.target_heading[env_idx, 0])
                except Exception:
                    pass
            else:
                # carry-forward for non-recorded steps (so plotting works without gaps).
                pos[t] = pos[t - 1]
                quat[t] = quat[t - 1]
                heading[t] = heading[t - 1]
                target_pos[t] = target_pos[t - 1]
                target_heading[t] = target_heading[t - 1]

            episode_id[t] = ep
            done_t = td.get("done", None)
            done_any = False
            done_env = False
            if done_t is not None:
                try:
                    done_any = bool(done_t.any().item())
                except Exception:
                    done_any = False
                try:
                    done_env = bool(done_t[env_idx].squeeze(-1).item())
                except Exception:
                    done_env = done_any
                done[t] = done_env

            if done_any:
                if done_env:
                    ep += 1
                try:
                    reset_mask = done_t.squeeze(-1)
                    td.set("_reset", reset_mask)
                    td = env.reset(td)
                except Exception:
                    pass

            if not bool(cfg.get("headless", True)):
                try:
                    env.render()
                except Exception:
                    pass

    if save_traj:
        meta = _get_task_meta(cfg)
        sim_dt = float(cfg.sim.dt)
        substeps = int(getattr(cfg.sim, "substeps", 1) or 1)
        np.savez_compressed(
            str(traj_out),
            step=step_id,
            episode_id=episode_id,
            done=done,
            pos=pos,
            quat_wxyz=quat,
            heading=heading,
            target_pos=target_pos,
            target_heading=target_heading,
            task_name=str(cfg.task.name),
            algo_name=str(cfg.algo.name),
            ckpt=str(eval_cfg.get("ckpt", "")),
            dt=sim_dt,
            substeps=substeps,
            **meta,
        )
        print(f"[evaluate] saved trajectory: {traj_out.resolve()}")

        if save_plot:
            try:
                from scripts.visualize_trajectory import plot_trajectory_3d  # noqa: WPS433

                plot_trajectory_3d(
                    traj_path=traj_out,
                    out_path=plot_out,
                    heading_stride=int(eval_cfg.get("heading_stride", 50) or 50),
                    arrow_len=float(eval_cfg.get("heading_arrow_len", 0.25) or 0.25),
                    show=bool(eval_cfg.get("plot_show", False)),
                )
                print(f"[evaluate] saved plot: {plot_out.resolve()}")
            except Exception as e:
                print(f"[evaluate] failed to plot trajectory (run visualize_trajectory.py manually): {type(e).__name__}: {e}")

    return {
        "steps": int(steps),
        "episodes": int(ep + 1),
        "traj_path": str(traj_out) if save_traj else "",
        "plot_path": str(plot_out) if (save_traj and save_plot) else "",
    }

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
    summary = evaluate_model(env, policy, cfg=cfg)
    print(f"[evaluate] done. summary={summary}")
    try:
        simulation_app.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
