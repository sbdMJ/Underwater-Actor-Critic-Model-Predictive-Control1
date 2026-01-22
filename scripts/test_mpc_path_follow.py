import os
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker
from torchrl.envs.transforms import TransformedEnv

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)
from marinegym.utils.torch import euler_to_quaternion, quat_axis


FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _generate_random_fourier_loop_3d(
    rng: np.random.Generator,
    *,
    num_points: int,
    harmonics: int,
    center: np.ndarray,
    radius_xy: float,
    z_amp: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """
    닫힌(smooth & periodic) 3D 경로를 생성합니다. (SciPy 없이 동작)

    - x,y: 랜덤 Fourier series (주기 2π)로 폐곡선 생성
    - z: Fourier series + clip
    """
    num_points = max(32, int(num_points))
    harmonics = max(1, int(harmonics))
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False, dtype=np.float64)

    def series(amplitude: float) -> np.ndarray:
        x = np.zeros_like(t)
        for k in range(1, harmonics + 1):
            a = rng.normal(0.0, 1.0)
            b = rng.normal(0.0, 1.0)
            x += (a * np.sin(k * t) + b * np.cos(k * t)) / (k**1.25)
        if np.max(np.abs(x)) > 1e-9:
            x = x / np.max(np.abs(x))
        return amplitude * x

    center = np.asarray(center, dtype=np.float64).reshape(3)
    x = series(radius_xy)
    y = series(radius_xy)
    z = series(z_amp)

    pts = np.stack([center[0] + x, center[1] + y, center[2] + z], axis=-1)
    pts[:, 2] = np.clip(pts[:, 2], z_min, z_max)
    return pts


@dataclass
class _PathLoop:
    points: np.ndarray  # (P, 3)
    seg_len: np.ndarray  # (P,)
    s: np.ndarray  # (P+1,)
    total: float

    @classmethod
    def from_points(cls, points: np.ndarray) -> "_PathLoop":
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (P, 3). got {points.shape}")
        if points.shape[0] < 4:
            raise ValueError("need at least 4 points for a loop")

        nxt = np.roll(points, shift=-1, axis=0)
        seg = np.linalg.norm(nxt - points, axis=1)
        seg = np.maximum(seg, 1e-6)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        return cls(points=points, seg_len=seg, s=s, total=float(s[-1]))

    def closest_index(self, pos: np.ndarray, last_idx: int, window: int) -> int:
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        p = self.points.shape[0]
        window = int(max(2, min(window, p // 2)))

        offsets = np.arange(-window, window + 1, dtype=np.int64)
        idxs = (int(last_idx) + offsets) % p
        d2 = np.sum((self.points[idxs] - pos[None, :]) ** 2, axis=1)
        return int(idxs[int(np.argmin(d2))])

    def sample_at_s(self, s_query: float) -> tuple[np.ndarray, np.ndarray, int]:
        if self.total <= 0.0:
            raise ValueError("invalid path length")
        p = self.points.shape[0]

        s_mod = float(s_query) % self.total
        seg_idx = int(np.searchsorted(self.s, s_mod, side="right") - 1)
        seg_idx = max(0, min(seg_idx, p - 1))

        s0 = float(self.s[seg_idx])
        s1 = float(self.s[seg_idx + 1])
        denom = max(1e-9, s1 - s0)
        alpha = (s_mod - s0) / denom

        p0 = self.points[seg_idx]
        p1 = self.points[(seg_idx + 1) % p]
        pos = p0 + alpha * (p1 - p0)
        tan = p1 - p0
        return pos, tan, seg_idx


@dataclass
class _LoopFollowerState:
    last_closest_idx: int = 0
    last_yaw: float = 0.0


def _update_targets_from_paths(
    *,
    base_env,
    paths: list[_PathLoop],
    states: list[_LoopFollowerState],
    lookahead_m: float,
    search_window: int,
    update_target_vis: bool,
):
    device = base_env.device
    num_envs = int(base_env.num_envs)
    env_ids = torch.arange(num_envs, device=device)

    base_env.drone.get_state()
    pos_now = base_env.drone.pos.squeeze(1)  # (num_envs, 3) in env-frame

    target_pos = torch.empty((num_envs, 1, 3), device=device, dtype=pos_now.dtype)
    target_rot = torch.empty((num_envs, 1, 4), device=device, dtype=pos_now.dtype)

    for i in range(num_envs):
        p_np = pos_now[i].detach().cpu().numpy()

        closest = paths[i].closest_index(p_np, states[i].last_closest_idx, window=search_window)
        states[i].last_closest_idx = closest
        s_closest = float(paths[i].s[closest])

        ref_np, tan_np, _ = paths[i].sample_at_s(s_closest + lookahead_m)

        yaw = float(np.arctan2(tan_np[1], tan_np[0]))
        if not np.isfinite(yaw):
            yaw = float(states[i].last_yaw)
        states[i].last_yaw = yaw

        target_pos[i, 0].copy_(torch.as_tensor(ref_np, device=device, dtype=target_pos.dtype))
        rpy = torch.zeros((3,), device=device, dtype=target_pos.dtype)
        rpy[2] = torch.as_tensor(yaw, device=device, dtype=target_pos.dtype)
        target_rot[i, 0].copy_(euler_to_quaternion(rpy))

    base_env.target_pos.copy_(target_pos)
    base_env.target_rot.copy_(target_rot)
    base_env.target_heading.copy_(quat_axis(target_rot.squeeze(1), axis=0).unsqueeze(1))

    if update_target_vis and hasattr(base_env, "target_vis"):
        target_pos_world = target_pos + base_env.envs_positions.unsqueeze(1)
        base_env.target_vis.set_world_poses(
            positions=target_pos_world, orientations=target_rot, env_indices=env_ids
        )


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    랜덤 3D 경로를 만들고 HoverMPC(MPCController)로 path following을 수행하는 데모.

    예)
      ~/isaac410/python.sh scripts/test_mpc_path_follow.py task=Hover_MPC headless=false enable_livestream=false env.num_envs=1

    주요 파라미터(override 가능)
      - path_num_points=400
      - path_harmonics=5
      - path_radius_xy=1.5
      - path_z_amp=0.4
      - path_lookahead=0.7
      - path_search_window=80
      - path_update_target_vis=true/false
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)

    from marinegym.envs import IsaacEnv, register_tasks

    register_tasks()

    # HoverMPC가 reset에서 target을 랜덤으로 샘플링해도 되지만,
    # path-following에서는 매 step 직접 갱신하므로 기본은 꺼둠.
    cfg.task.random_target_pos = bool(cfg.task.get("random_target_pos", False)) and bool(
        cfg.get("path_keep_random_target_on_reset", False)
    )

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()

    seed = int(cfg.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    print("[test_mpc_path_follow] seed:", seed)
    env.set_seed(seed)
    rng = np.random.default_rng(seed)

    path_num_points = _safe_int(cfg.get("path_num_points", 400), 400)
    path_harmonics = _safe_int(cfg.get("path_harmonics", 5), 5)
    path_radius_xy = _safe_float(cfg.get("path_radius_xy", 1.5), 1.5)
    path_z_amp = _safe_float(cfg.get("path_z_amp", 0.4), 0.4)
    path_z_min = _safe_float(cfg.get("path_z_min", 0.6), 0.6)
    path_z_max = _safe_float(cfg.get("path_z_max", 3.2), 3.2)
    lookahead_m = _safe_float(cfg.get("path_lookahead", 0.7), 0.7)
    search_window = _safe_int(cfg.get("path_search_window", 80), 80)
    update_target_vis = bool(cfg.get("path_update_target_vis", True))

    td = env.reset()

    # per-env loop path + follower state
    num_envs = int(env.base_env.num_envs)
    env.base_env.drone.get_state()
    centers = env.base_env.drone.pos.squeeze(1).detach().cpu().numpy()  # (num_envs, 3)
    paths: list[_PathLoop] = []
    states: list[_LoopFollowerState] = []
    for i in range(num_envs):
        pts = _generate_random_fourier_loop_3d(
            rng,
            num_points=path_num_points,
            harmonics=path_harmonics,
            center=centers[i],
            radius_xy=path_radius_xy,
            z_amp=path_z_amp,
            z_min=path_z_min,
            z_max=path_z_max,
        )
        paths.append(_PathLoop.from_points(pts))
        states.append(_LoopFollowerState())

    _update_targets_from_paths(
        base_env=env.base_env,
        paths=paths,
        states=states,
        lookahead_m=lookahead_m,
        search_window=search_window,
        update_target_vis=update_target_vis,
    )

    dummy_action = env.action_spec.zero()

    steps = _safe_int(cfg.get("mpc_path_test_steps", cfg.get("mpc_test_steps", 5000)), 5000)
    for _ in range(steps):
        _update_targets_from_paths(
            base_env=env.base_env,
            paths=paths,
            states=states,
            lookahead_m=lookahead_m,
            search_window=search_window,
            update_target_vis=update_target_vis and (not cfg.headless),
        )

        if hasattr(dummy_action, "items"):
            td.update(dummy_action)
        else:
            td.set(("agents", "action"), dummy_action)
        td = env.step(td)["next"]

        done = td.get("done", None)
        if done is not None and bool(done.any()):
            reset_mask = done.squeeze(-1)
            td.set("_reset", reset_mask)
            td = env.reset(td)

            # reset된 env만 새 경로 생성
            env_ids = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
            env.base_env.drone.get_state()
            centers = env.base_env.drone.pos.squeeze(1).detach().cpu().numpy()
            for j in env_ids.detach().cpu().tolist():
                pts = _generate_random_fourier_loop_3d(
                    rng,
                    num_points=path_num_points,
                    harmonics=path_harmonics,
                    center=centers[j],
                    radius_xy=path_radius_xy,
                    z_amp=path_z_amp,
                    z_min=path_z_min,
                    z_max=path_z_max,
                )
                paths[j] = _PathLoop.from_points(pts)
                states[j] = _LoopFollowerState()

        if not cfg.headless:
            env.render()

    simulation_app.close()


if __name__ == "__main__":
    main()

