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


def _get_launch_cwd() -> Path:
    """Return the directory where the Hydra job was launched from (not the run dir)."""
    try:
        from hydra.utils import get_original_cwd  # noqa: WPS433

        return Path(get_original_cwd())
    except Exception:
        return Path.cwd()


def load_checkpoint(checkpoint_path, env_config, algo_config):
    from marinegym.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[env_config.task.name]
    base_env = env_class(env_config, headless=env_config.headless)

    transforms = [InitTracker()]
    env = TransformedEnv(base_env, Compose(*transforms)).eval()

    algo_name = str(getattr(algo_config, "name", "")).lower()
    if "pypose" in algo_name:
        try:
            from evaluate_pypose_mpc_ppo import _maybe_prepare_pypose_mpc_cfg  # noqa: WPS433

            _maybe_prepare_pypose_mpc_cfg(env_config, base_env, algo_name=algo_name, out_dir=Path.cwd() / algo_name)
        except Exception as exc:
            print(f"[load_checkpoint] pypose MPC config prep skipped/failed: {exc}")

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
    base_env = getattr(env, "base_env", None)

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
        traj_path = _get_launch_cwd() / traj_path

    pos_log = []
    heading_log = []
    speed_log = []
    vtan_log = []
    vtan_err_log = []
    u_log = []
    energy_log = []
    flow_log = []
    done_log = []
    terminated_log = []
    step_log = []
    global_step = 0

    traj_nu = int(getattr(getattr(base_env, "drone", None), "num_rotors", 0) or 0)
    if traj_nu <= 0:
        try:
            traj_nu = int(getattr(getattr(base_env, "drone", None).action_spec, "shape", (0,))[-1])
        except Exception:
            traj_nu = 0

    # Reference tangential speed used for PPO-style orbit progress shaping (if available).
    v_tan_ref = float("nan")
    try:
        if hasattr(cfg, "task") and cfg.task is not None:
            v_tan_ref = float(cfg.task.get("reward_orbit_v_tan_ref", float("nan")))
    except Exception:
        v_tan_ref = float("nan")
    if not np.isfinite(v_tan_ref):
        try:
            v_tan_ref = float(getattr(base_env, "orbit_v_tan", float("nan")))
        except Exception:
            v_tan_ref = float("nan")

    def _maybe_log_traj(*, step: int, done: bool, terminated: bool = False) -> None:
        if not save_traj:
            return
        if (int(step) % int(traj_stride)) != 0:
            return
        if base_env is None or not hasattr(base_env, "drone"):
            return
        try:
            base_env.drone.get_state()
            pos_np = _to_numpy(base_env.drone.pos[traj_env_id, 0, :]).astype(np.float32, copy=False)
            pos_log.append(pos_np)
            heading_log.append(_to_numpy(base_env.drone.heading[traj_env_id, 0, :]).astype(np.float32, copy=False))
            vel_lin_w = None
            try:
                vel_lin_w = base_env.drone.vel_w[traj_env_id, 0, 0:3]
                speed_log.append(float(torch.linalg.norm(vel_lin_w).item()))
            except Exception:
                speed_log.append(float("nan"))

            # Tangential speed around the cylinder (signed along desired direction).
            v_tan = float("nan")
            v_tan_err = float("nan")
            try:
                if vel_lin_w is not None and hasattr(base_env, "cylinder_center"):
                    center_t = getattr(base_env, "cylinder_center", None)
                    center_np = (
                        np.asarray(_to_numpy(center_t), dtype=np.float32).reshape(-1)[:3]
                        if center_t is not None
                        else np.zeros((3,), dtype=np.float32)
                    )
                    rel_xy = pos_np[0:2] - center_np[0:2]  # center -> robot (XY)
                    dist = float(np.linalg.norm(rel_xy))
                    if np.isfinite(dist) and dist > 1e-6:
                        radial_hat_xy = rel_xy / dist
                        dir_sign = float(getattr(base_env, "orbit_direction", 1.0))
                        dir_sign = 1.0 if dir_sign >= 0.0 else -1.0
                        tan_hat_xy = np.asarray([-radial_hat_xy[1], radial_hat_xy[0]], dtype=np.float32) * dir_sign
                        v_xy = np.asarray(_to_numpy(vel_lin_w[0:2]), dtype=np.float32).reshape(2,)
                        v_tan = float(tan_hat_xy[0] * v_xy[0] + tan_hat_xy[1] * v_xy[1])
                        if np.isfinite(v_tan_ref):
                            v_tan_err = float(abs(v_tan - float(v_tan_ref)))
            except Exception:
                v_tan = float("nan")
                v_tan_err = float("nan")
            vtan_log.append(float(v_tan))
            vtan_err_log.append(float(v_tan_err))

            try:
                flow = base_env.drone.flow_vels[traj_env_id, 0, 0:3]
                flow_log.append(_to_numpy(flow).astype(np.float32, copy=False))
            except Exception:
                flow_log.append(np.full((3,), np.nan, dtype=np.float32))

            u_cmd = getattr(base_env.drone, "throttle", None)
            if u_cmd is None:
                u_cmd = getattr(base_env, "_last_action_cmd", None)
            u_vec = None
            energy = float("nan")
            if u_cmd is not None:
                try:
                    if torch.is_tensor(u_cmd):
                        u_t = u_cmd
                        if u_t.ndim >= 3:
                            u_t = u_t.squeeze(1)
                        u_env = u_t[int(traj_env_id)]
                        u_vec = u_env.detach().cpu().to(dtype=torch.float32).numpy()
                        energy = float((u_env.abs() ** 3).sum().item())
                    else:
                        u_np = np.asarray(u_cmd)
                        if u_np.ndim >= 3:
                            u_np = np.squeeze(u_np, axis=1)
                        u_env = np.asarray(u_np[int(traj_env_id)], dtype=np.float32)
                        u_vec = u_env
                        energy = float(np.sum(np.abs(u_env) ** 3))
                except Exception:
                    u_vec = None
                    energy = float("nan")
            if u_vec is None:
                if traj_nu > 0:
                    u_vec = np.full((traj_nu,), np.nan, dtype=np.float32)
                else:
                    u_vec = np.zeros((0,), dtype=np.float32)
            try:
                u_vec = np.asarray(u_vec, dtype=np.float32).reshape(-1)
                if traj_nu > 0 and u_vec.shape[0] != traj_nu:
                    if u_vec.shape[0] > traj_nu:
                        u_vec = u_vec[:traj_nu]
                    else:
                        u_vec = np.pad(u_vec, (0, traj_nu - u_vec.shape[0]), constant_values=np.nan)
            except Exception:
                u_vec = np.full((traj_nu,), np.nan, dtype=np.float32) if traj_nu > 0 else np.zeros((0,), dtype=np.float32)
            u_log.append(u_vec)
            energy_log.append(float(energy))

            done_log.append(bool(done))
            terminated_log.append(bool(terminated))
            step_log.append(int(step))
        except Exception:
            return

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
        _maybe_log_traj(step=int(global_step), done=False, terminated=False)
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
                done_val = False
                terminated_val = False
                if done is not None:
                    try:
                        done_mask_for_traj = done.squeeze(-1)
                        done_val = bool(done_mask_for_traj[traj_env_id].item())
                    except Exception:
                        done_val = False
                terminated = td.get("terminated", None)
                if terminated is not None:
                    try:
                        term_mask_for_traj = terminated.squeeze(-1)
                        terminated_val = bool(term_mask_for_traj[traj_env_id].item())
                    except Exception:
                        try:
                            terminated_val = bool(terminated.any())
                        except Exception:
                            terminated_val = False
                _maybe_log_traj(step=int(global_step), done=done_val, terminated=terminated_val)

                if done is not None and bool(done.any()):
                    done_mask = done.squeeze(-1)
                    try:
                        stats_done = td["stats"][done_mask].cpu()
                        episode_stats.extend(stats_done.unbind(0))
                    except Exception:
                        pass
                    td.set("_reset", done_mask)
                    td = env.reset(td)
                    _maybe_log_traj(step=int(global_step), done=False, terminated=False)
        if episode_stats:
            stats_td = torch.stack(episode_stats).to_tensordict()
            results.append(stats_td)

    saved_path = None
    if save_traj and pos_log:
        try:
            pos_arr = np.stack(pos_log, axis=0).astype(np.float32, copy=False)
            heading_arr = np.stack(heading_log, axis=0).astype(np.float32, copy=False)
            speed_arr = np.asarray(speed_log, dtype=np.float32)
            vtan_arr = np.asarray(vtan_log, dtype=np.float32)
            vtan_err_arr = np.asarray(vtan_err_log, dtype=np.float32)
            u_arr = np.stack(u_log, axis=0).astype(np.float32, copy=False) if u_log else np.zeros((0, 0), dtype=np.float32)
            energy_arr = np.asarray(energy_log, dtype=np.float32)
            energy_total = float(np.nansum(energy_arr))
            flow_arr = np.stack(flow_log, axis=0).astype(np.float32, copy=False) if flow_log else np.zeros((0, 3), dtype=np.float32)
            done_arr = np.asarray(done_log, dtype=np.bool_)
            terminated_arr = np.asarray(terminated_log, dtype=np.bool_)
            step_arr = np.asarray(step_log, dtype=np.int64)

            if terminated_arr.shape[0] != pos_arr.shape[0]:
                if terminated_arr.shape[0] > pos_arr.shape[0]:
                    terminated_arr = terminated_arr[: pos_arr.shape[0]]
                else:
                    terminated_arr = np.pad(
                        terminated_arr,
                        (0, pos_arr.shape[0] - terminated_arr.shape[0]),
                        mode="constant",
                        constant_values=False,
                    )

            if flow_arr.shape[0] != pos_arr.shape[0]:
                if flow_arr.shape[0] > pos_arr.shape[0]:
                    flow_arr = flow_arr[: pos_arr.shape[0]]
                else:
                    pad = np.full((pos_arr.shape[0] - flow_arr.shape[0], 3), np.nan, dtype=np.float32)
                    flow_arr = np.concatenate([flow_arr, pad], axis=0) if flow_arr.size else pad

            current_vec_w = (
                np.nanmean(flow_arr, axis=0).astype(np.float32, copy=False)
                if flow_arr.size
                else np.full((3,), np.nan, dtype=np.float32)
            )

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
                v_tan=vtan_arr,
                v_tan_err=vtan_err_arr,
                u=u_arr,
                energy=energy_arr,
                energy_total=energy_total,
                flow_vel_w=flow_arr,
                current_vec_w=current_vec_w,
                done=done_arr,
                terminated=terminated_arr,
                cylinder_center=center,
                cylinder_radius=float(getattr(base_env, "cylinder_radius", 0.0)),
                cylinder_height=float(getattr(base_env, "cylinder_height", 0.0)),
                orbit_radius=float(getattr(base_env, "orbit_radius", 0.0)),
                orbit_z=float(getattr(base_env, "orbit_z", 0.0)),
                orbit_v_tan=float(getattr(base_env, "orbit_v_tan", 0.0)),
                v_tan_ref=float(v_tan_ref) if np.isfinite(v_tan_ref) else np.nan,
                meta=np.array(
                    [
                        {
                            "task": str(getattr(cfg, "task", {}).get("name", "")) if hasattr(cfg, "task") else "",
                            "algo": str(getattr(cfg, "algo", {}).get("name", "")) if hasattr(cfg, "algo") else "",
                            "ckpt": str(eval_cfg.get("ckpt", "")),
                            "dt": float(getattr(getattr(cfg, "sim", None), "dt", 0.0)),
                            "traj_env_id": int(traj_env_id),
                            "traj_stride": int(traj_stride),
                            "current_vec_w": current_vec_w.tolist() if np.isfinite(current_vec_w).any() else None,
                        }
                    ],
                    dtype=object,
                ),
            )
            tmp_path.replace(traj_path)
            saved_path = traj_path
            print(f"[eval] saved trajectory: {traj_path.resolve()}")
            print(f"[eval] energy_total (Σ_t Σ_i |u_i|^3): {energy_total:.6g}")
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
            from visualize_trajectory import plot_power_polar, plot_trajectory_3d

            out_png = eval_cfg.get("traj_png_path", None)
            out_path = Path(out_png).expanduser() if out_png else traj_path.with_suffix(".png")
            if not out_path.is_absolute():
                out_path = _get_launch_cwd() / out_path
            traj_color_key = str(eval_cfg.get("traj_color_key", eval_cfg.get("traj_color", ""))).strip()
            if not traj_color_key:
                traj_color_key = "v_tan_err"
            plot_trajectory_3d(
                traj_path=traj_path,
                out_path=out_path,
                heading_stride=int(eval_cfg.get("plot_heading_stride", 50)),
                arrow_len=float(eval_cfg.get("plot_arrow_len", 0.25)),
                path_color_mode=str(eval_cfg.get("traj_color_mode", "termination")),
                color_key=str(traj_color_key),
                show=bool(eval_cfg.get("plot_show", False)),
            )
            print(f"[eval] saved trajectory plot: {out_path.resolve()}")

            plot_energy = bool(eval_cfg.get("plot_energy", eval_cfg.get("plot_traj_energy", True)))
            if plot_energy:
                out_energy_png = eval_cfg.get("traj_energy_png_path", None)
                energy_path = (
                    Path(out_energy_png).expanduser()
                    if out_energy_png
                    else out_path.with_name(out_path.stem + "_energy" + out_path.suffix)
                )
                if not energy_path.is_absolute():
                    energy_path = _get_launch_cwd() / energy_path
                plot_trajectory_3d(
                    traj_path=traj_path,
                    out_path=energy_path,
                    heading_stride=int(eval_cfg.get("plot_heading_stride", 50)),
                    arrow_len=float(eval_cfg.get("plot_arrow_len", 0.25)),
                    path_color_mode=str(eval_cfg.get("traj_color_mode", "termination")),
                    color_key="energy",
                    show=bool(eval_cfg.get("plot_show", False)),
                )
                print(f"[eval] saved energy plot: {energy_path.resolve()}")

            plot_energy_polar = bool(eval_cfg.get("plot_energy_polar", eval_cfg.get("plot_polar_energy", True)))
            if plot_energy_polar:
                out_energy_polar_png = eval_cfg.get("traj_energy_polar_png_path", None)
                energy_polar_path = (
                    Path(out_energy_polar_png).expanduser()
                    if out_energy_polar_png
                    else out_path.with_name(out_path.stem + "_energy_polar" + out_path.suffix)
                )
                if not energy_polar_path.is_absolute():
                    energy_polar_path = _get_launch_cwd() / energy_polar_path
                plot_power_polar(
                    traj_path=traj_path,
                    out_path=energy_polar_path,
                    bin_deg=float(eval_cfg.get("energy_polar_bin_deg", 5.0)),
                    show_raw=bool(eval_cfg.get("energy_polar_show_raw", False)),
                    show=bool(eval_cfg.get("plot_show", False)),
                )
                print(f"[eval] saved energy polar plot: {energy_polar_path.resolve()}")
        except Exception as exc:
            print(f"[eval] failed to plot trajectory: {exc}")

if __name__ == "__main__":
    main()
