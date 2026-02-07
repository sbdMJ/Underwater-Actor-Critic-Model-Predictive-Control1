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


os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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

    eval_cfg = cfg.get("eval", {}) if hasattr(cfg, "get") else getattr(cfg, "eval", {})
    mode = str(cfg.get("mode", "")).lower() if hasattr(cfg, "get") else str(getattr(cfg, "mode", "")).lower()
    save_traj = eval_cfg.get("save_traj", None)
    if save_traj is None:
        save_traj = mode == "evaluate"
    save_traj = bool(save_traj)

    traj_env_id = int(eval_cfg.get("traj_env_id", 0))
    traj_stride = max(1, int(eval_cfg.get("traj_stride", 1)))
    traj_out = eval_cfg.get("traj_path", None)
    traj_path = Path(traj_out).expanduser() if traj_out else Path("trajectory.npz")
    if not traj_path.is_absolute():
        traj_path = Path.cwd() / traj_path

    pos_log = []
    heading_log = []
    speed_log = []
    done_log = []
    step_log = []
    global_step = 0

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
        if save_traj:
            try:
                env.base_env.drone.get_state()
                if (global_step % traj_stride) == 0:
                    pos_log.append(_to_numpy(env.base_env.drone.pos[traj_env_id, 0, :]).astype(np.float32, copy=False))
                    heading_log.append(
                        _to_numpy(env.base_env.drone.heading[traj_env_id, 0, :]).astype(np.float32, copy=False)
                    )
                    try:
                        vel_lin_w = env.base_env.drone.vel_w[traj_env_id, 0, 0:3]
                        speed_log.append(float(torch.linalg.norm(vel_lin_w).item()))
                    except Exception:
                        speed_log.append(float("nan"))
                    done_log.append(False)
                    step_log.append(int(global_step))
            except Exception:
                pass
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
                global_step += 1

                done = td.get("done", None)
                if save_traj:
                    try:
                        env.base_env.drone.get_state()
                        if (global_step % traj_stride) == 0:
                            pos_log.append(
                                _to_numpy(env.base_env.drone.pos[traj_env_id, 0, :]).astype(np.float32, copy=False)
                            )
                            heading_log.append(
                                _to_numpy(env.base_env.drone.heading[traj_env_id, 0, :]).astype(np.float32, copy=False)
                            )
                            try:
                                vel_lin_w = env.base_env.drone.vel_w[traj_env_id, 0, 0:3]
                                speed_log.append(float(torch.linalg.norm(vel_lin_w).item()))
                            except Exception:
                                speed_log.append(float("nan"))
                            done_val = False
                            if done is not None:
                                done_mask = done.squeeze(-1)
                                done_val = bool(done_mask[traj_env_id].item())
                            done_log.append(done_val)
                            step_log.append(int(global_step))
                    except Exception:
                        pass

                if done is not None and bool(done.any()):
                    done_mask = done.squeeze(-1)
                    try:
                        stats_done = td["stats"][done_mask].cpu()
                        episode_stats.extend(stats_done.unbind(0))
                    except Exception:
                        pass
                    td.set("_reset", done_mask)
                    td = env.reset(td)
                    if save_traj:
                        try:
                            env.base_env.drone.get_state()
                            if (global_step % traj_stride) == 0:
                                pos_log.append(
                                    _to_numpy(env.base_env.drone.pos[traj_env_id, 0, :]).astype(np.float32, copy=False)
                                )
                                heading_log.append(
                                    _to_numpy(env.base_env.drone.heading[traj_env_id, 0, :]).astype(np.float32, copy=False)
                                )
                                try:
                                    vel_lin_w = env.base_env.drone.vel_w[traj_env_id, 0, 0:3]
                                    speed_log.append(float(torch.linalg.norm(vel_lin_w).item()))
                                except Exception:
                                    speed_log.append(float("nan"))
                                done_log.append(False)
                                step_log.append(int(global_step))
                        except Exception:
                            pass
        if episode_stats:
            stats_td = torch.stack(episode_stats).to_tensordict()
            results.append(stats_td)

    saved_path = None
    if save_traj and pos_log:
        try:
            pos_arr = np.stack(pos_log, axis=0).astype(np.float32, copy=False)
            heading_arr = np.stack(heading_log, axis=0).astype(np.float32, copy=False)
            speed_arr = np.asarray(speed_log, dtype=np.float32)
            done_arr = np.asarray(done_log, dtype=np.bool_)
            step_arr = np.asarray(step_log, dtype=np.int64)

            base_env = getattr(env, "base_env", None)
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
                            "task": str(getattr(cfg, "task", {}).get("name", "")) if hasattr(cfg, "task") else "",
                            "algo": str(getattr(cfg, "algo", {}).get("name", "")) if hasattr(cfg, "algo") else "",
                            "ckpt": str(eval_cfg.get("ckpt", "")),
                            "dt": float(getattr(getattr(cfg, "sim", None), "dt", 0.0)),
                            "traj_env_id": int(traj_env_id),
                            "traj_stride": int(traj_stride),
                        }
                    ],
                    dtype=object,
                ),
            )
            tmp_path.replace(traj_path)
            saved_path = traj_path
            print(f"[eval] saved trajectory: {traj_path.resolve()}")
        except Exception as exc:
            print(f"[eval] failed to save trajectory: {exc}")

    return results, saved_path

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
    eval_cfg = cfg.get("eval", {}) if hasattr(cfg, "get") else getattr(cfg, "eval", {})
    num_episodes = int(eval_cfg.get("episodes", 1))
    eval_results, traj_path = evaluate_model(env, policy, num_episodes=num_episodes, cfg=cfg)
    print(eval_results)
    if traj_path is not None and bool(eval_cfg.get("plot_traj", True)):
        try:
            from visualize_trajectory import plot_trajectory_3d

            out_png = eval_cfg.get("traj_png_path", None)
            out_path = Path(out_png).expanduser() if out_png else traj_path.with_suffix(".png")
            if not out_path.is_absolute():
                out_path = Path.cwd() / out_path
            plot_trajectory_3d(
                traj_path=traj_path,
                out_path=out_path,
                heading_stride=int(eval_cfg.get("plot_heading_stride", 50)),
                arrow_len=float(eval_cfg.get("plot_arrow_len", 0.25)),
                show=bool(eval_cfg.get("plot_show", False)),
            )
            print(f"[eval] saved trajectory plot: {out_path.resolve()}")
        except Exception as exc:
            print(f"[eval] failed to plot trajectory: {exc}")

if __name__ == "__main__":
    main()
