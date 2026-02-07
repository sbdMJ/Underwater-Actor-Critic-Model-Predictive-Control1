import os
import math
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)


FILE_PATH = str(Path(__file__).resolve().parent)

os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


@dataclass
class _EpisodeEnd:
    episode_id: int
    step: int
    length: int
    terminated: bool
    truncated: bool
    success: bool


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _coerce_np_array(
    x,
    *,
    shape: tuple[int, ...],
    dtype,
    fill_value=0,
) -> np.ndarray:
    out = np.full(shape, fill_value=fill_value, dtype=dtype)
    if x is None:
        return out
    try:
        arr = np.asarray(_to_numpy(x), dtype=dtype)
        arr = np.squeeze(arr)
        return arr.reshape(shape)
    except Exception:
        try:
            flat = np.asarray(_to_numpy(x)).reshape(-1)
            flat = flat.astype(dtype, copy=False)
        except Exception:
            return out
        out_flat = out.reshape(-1)
        n = int(min(out_flat.size, flat.size))
        if n > 0:
            out_flat[:n] = flat[:n]
        return out


def _atomic_replace(src_path: Path, dst_path: Path) -> None:
    try:
        os.replace(str(src_path), str(dst_path))
    except Exception:
        try:
            if dst_path.exists():
                dst_path.unlink(missing_ok=True)
        except Exception:
            pass
        os.rename(str(src_path), str(dst_path))


def _maybe_import_orbit_errors():
    try:
        from marinegym.controllers.pypose_cylinder_orbit_mpc_controller import _orbit_errors  # noqa: WPS433

        return _orbit_errors
    except Exception:
        return None


def _plot_orbit_xy(
    *,
    out_path: Path,
    traj_xyz: np.ndarray,
    center: np.ndarray,
    orbit_radius: float,
    cylinder_radius: float,
    n_ref: int = 200,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    traj_xyz = np.asarray(traj_xyz)
    center = np.asarray(center).reshape(3,)

    theta = np.linspace(0.0, 2.0 * np.pi, max(50, int(n_ref)))
    ref_x = center[0] + orbit_radius * np.cos(theta)
    ref_y = center[1] + orbit_radius * np.sin(theta)
    cyl_x = center[0] + cylinder_radius * np.cos(theta)
    cyl_y = center[1] + cylinder_radius * np.sin(theta)

    plt.figure(figsize=(7, 7))
    plt.plot(ref_x, ref_y, "k--", linewidth=1.5, label="desired orbit (xy)")
    plt.plot(cyl_x, cyl_y, color="#f28e2b", linewidth=2.0, label="cylinder (xy)")
    plt.plot(traj_xyz[:, 0], traj_xyz[:, 1], color="#4e79a7", linewidth=1.5, label="trajectory")
    plt.scatter([traj_xyz[0, 0]], [traj_xyz[0, 1]], c="g", s=30, label="start")
    plt.scatter([traj_xyz[-1, 0]], [traj_xyz[-1, 1]], c="r", s=30, label="end")
    plt.scatter([center[0]], [center[1]], c="k", s=20, label="center")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    MPC 기반 HoverMPC 환경이 실제로 step을 돌 수 있는지 확인하는 간단한 스모크 테스트.

    예)
      python scripts/test_mpc_withCam.py task=Hover_MPC headless=false enable_livestream=false env.num_envs=1
      python scripts/test_mpc_withCam.py task=Hover_MPC camera.capture_interval=60

      ~/isaac410/python.sh scripts/test_mpc_withCam.py   task=OrbitCylinder_MPC headless=false enable_livestream=false env.num_envs=1   task.use_pypose_mpc=false   +camera.head_offset='[0.4,0,0.15]' env.max_episode_length=800

      # 1바퀴(2π)마다 orbit_z를 조금씩 내려가며 여러 바퀴(나선형) 돌기
      ~/isaac410/python.sh scripts/test_mpc_withCam.py task=OrbitCylinder_MPC headless=false enable_livestream=false env.num_envs=1 +layered_orbit.enable=true +layered_orbit.delta_z=0.05 +layered_orbit.laps=5 +layered_orbit.transition_steps=60

      # 1바퀴 도는거 diffMPC (pypose)
      ~/isaac410/python.sh scripts/test_mpc_withCam.py task=OrbitCylinder_MPC headless=false enable_livestream=false env.num_envs=1 task.use_pypose_mpc=true +camera.head_offset='[0.4,0,0.15]' task.pypose_orbit_mode=cylinder_cost

    Trajectory logging (default on):
      - Saves `trajectory.npz` and `trajectory.png` under Hydra's run dir (./outputs/...). (speed is shown as colormap if available)
      - Disable with `+traj_log.enable=false`.
      - Override paths with `+traj_log.traj_path=/tmp/trajectory.npz +traj_log.traj_png_path=/tmp/trajectory.png`.
      - Re-visualize later: `python scripts/visualize_trajectory.py /path/to/trajectory.npz`

    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)

    from marinegym.envs import IsaacEnv, register_tasks
    register_tasks()

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()
    # 기본은 매 실행마다 랜덤 시드로 두는 편이 디버깅에 유리함
    seed = int(cfg.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    print("[test_mpc] seed:", seed)
    env.set_seed(seed)

    # Optional: spawn a Replicator camera using `marinegym/sensors/camera.py`.
    camera = None
    camera_cfg = cfg.get("camera", None)
    if camera_cfg is None:
        camera_cfg = {}
    camera_cfg = OmegaConf.to_container(camera_cfg, resolve=True) if not isinstance(camera_cfg, dict) else camera_cfg
    if bool(camera_cfg.get("enable", True)):
        from marinegym.sensors.camera import Camera
        from marinegym.sensors.config import FisheyeCameraCfg, PinholeCameraCfg

        projection_type = str(camera_cfg.get("projection_type", "pinhole"))
        resolution = tuple(camera_cfg.get("resolution", (640, 480)))
        data_types = list(camera_cfg.get("data_types", ["rgb"]))
        sensor_tick = float(camera_cfg.get("sensor_tick", 0.0))
        if projection_type.startswith("fisheye"):
            cam_cfg = FisheyeCameraCfg(
                sensor_tick=sensor_tick,
                resolution=resolution,
                data_types=data_types,
            )
        else:
            cam_cfg = PinholeCameraCfg(
                sensor_tick=sensor_tick,
                resolution=resolution,
                data_types=data_types,
            )
        camera = Camera(cam_cfg)

        attach_to = str(camera_cfg.get("attach_to", "drone_head")).lower()
        all_envs = bool(camera_cfg.get("all_envs", False))

        if attach_to in ("drone_head", "head", "bluerov_head"):
            # Attach as a child prim under the vehicle base_link so it moves rigidly with the robot.
            # The offset/forward/target are specified in the base_link local frame (+X is forward).
            if not hasattr(base_env, "drone") or not hasattr(base_env.drone, "prim_paths"):
                raise RuntimeError("camera.attach_to=drone_head requires env.drone.prim_paths")

            drone_paths = list(base_env.drone.prim_paths)
            if not all_envs:
                idx = int(getattr(base_env, "central_env_idx", 0))
                drone_paths = [drone_paths[idx]]

            camera_paths = [f"{p}/base_link/Camera_0" for p in drone_paths]

            head_offset = camera_cfg.get("head_offset", (0.35, 0.0, 0.10))
            head_forward = camera_cfg.get("head_forward", (1.0, 0.0, 0.0))
            head_distance = float(camera_cfg.get("head_distance", 1.0))
            translation = tuple(float(v) for v in head_offset)
            target = tuple(float(translation[i] + head_forward[i] * head_distance) for i in range(3))
        else:
            translation = tuple(camera_cfg.get("translation", (4.0, 4.0, 4.0)))
            target = tuple(camera_cfg.get("target", (0.0, 0.0, 2.0)))

            if all_envs:
                camera_paths = [f"{p}/Camera_0" for p in base_env.envs_prim_paths]
            else:
                env_path = base_env.envs_prim_paths[int(base_env.central_env_idx)]
                camera_paths = [f"{env_path}/Camera_0"]

        camera.spawn(camera_paths, translations=translation, targets=target)
        if attach_to in ("drone_head", "head", "bluerov_head"):
            drone_name = getattr(base_env.drone, "name", "Drone")
            camera.initialize(prim_paths_expr=f"/World/envs/env_.*/{drone_name}_.*/base_link/Camera_.*")
        else:
            camera.initialize()

    td = env.reset()
    dummy_action = env.action_spec.zero()
    try:
        print("[test_mpc] target_pos[0]:", env.base_env.target_pos[0, 0].detach().cpu().tolist())
    except Exception:
        pass

    traj_cfg = cfg.get("traj_log", None)
    if traj_cfg is None:
        traj_cfg = {}
    traj_cfg = OmegaConf.to_container(traj_cfg, resolve=True) if not isinstance(traj_cfg, dict) else traj_cfg
    traj_enable = bool(traj_cfg.get("enable", True))
    traj_env_id = int(traj_cfg.get("env_id", traj_cfg.get("traj_env_id", 0)))
    traj_stride = max(1, int(traj_cfg.get("stride", traj_cfg.get("traj_stride", 1))))
    traj_out = traj_cfg.get("path", traj_cfg.get("traj_path", None))
    traj_path = Path(str(traj_out)).expanduser() if traj_out else Path.cwd() / "trajectory.npz"
    if not traj_path.is_absolute():
        traj_path = Path.cwd() / traj_path
    traj_plot = bool(traj_cfg.get("plot", traj_cfg.get("plot_traj", True)))
    traj_png_out = traj_cfg.get("png_path", traj_cfg.get("traj_png_path", None))
    traj_png_path = Path(str(traj_png_out)).expanduser() if traj_png_out else traj_path.with_suffix(".png")
    if not traj_png_path.is_absolute():
        traj_png_path = Path.cwd() / traj_png_path
    traj_plot_heading_stride = int(traj_cfg.get("plot_heading_stride", 50))
    traj_plot_arrow_len = float(traj_cfg.get("plot_arrow_len", 0.25))
    traj_plot_show = bool(traj_cfg.get("plot_show", False))

    traj_pos_log = []
    traj_heading_log = []
    traj_speed_log = []
    traj_done_log = []
    traj_step_log = []

    def _traj_maybe_append(*, step: int, done: bool) -> None:
        if not traj_enable:
            return
        if (int(step) % int(traj_stride)) != 0:
            return
        try:
            base_env.drone.get_state()
            pos = _to_numpy(base_env.drone.pos[int(traj_env_id), 0, :]).astype(np.float32, copy=False)
            heading = _to_numpy(base_env.drone.heading[int(traj_env_id), 0, :]).astype(np.float32, copy=False)
            try:
                vel_lin_w = base_env.drone.vel_w[int(traj_env_id), 0, 0:3]
                speed = float(torch.linalg.norm(vel_lin_w).item())
            except Exception:
                speed = float("nan")
            traj_pos_log.append(pos)
            traj_heading_log.append(heading)
            traj_speed_log.append(speed)
            traj_done_log.append(bool(done))
            traj_step_log.append(int(step))
        except Exception:
            return

    _traj_maybe_append(step=0, done=False)

    if camera is not None and bool(camera_cfg.get("capture_on_reset", True)):
        base_env.sim.render()
        imgs = camera.get_images()
        if "rgb" in imgs.keys():
            print("[test_mpc_withCam] camera.rgb:", tuple(imgs["rgb"].shape))

    layered_orbit_cfg = cfg.get("layered_orbit", None)
    if layered_orbit_cfg is None:
        layered_orbit_cfg = {}
    layered_orbit_cfg = (
        OmegaConf.to_container(layered_orbit_cfg, resolve=True)
        if not isinstance(layered_orbit_cfg, dict)
        else layered_orbit_cfg
    )
    layered_orbit_enable = bool(layered_orbit_cfg.get("enable", False))
    layered_orbit_delta_z = float(layered_orbit_cfg.get("delta_z", 0.1))
    layered_orbit_laps = int(layered_orbit_cfg.get("laps", -1))
    layered_orbit_transition_steps = int(layered_orbit_cfg.get("transition_steps", 0))
    if layered_orbit_enable:
        if not hasattr(base_env, "orbit_z"):
            print("[test_mpc_withCam] layered_orbit.enable=true but env has no orbit_z; disabling layered_orbit.")
            layered_orbit_enable = False
        else:
            steps_per_lap = int(getattr(base_env, "orbit_period_steps", 0))
            print(
                "[test_mpc_withCam] layered_orbit:",
                f"steps_per_lap={steps_per_lap}",
                f"delta_z={layered_orbit_delta_z}",
                f"laps={layered_orbit_laps}",
                f"transition_steps={layered_orbit_transition_steps}",
                f"start_z={float(getattr(base_env, 'orbit_z'))}",
            )
    else:
        steps_per_lap = 0

    steps = int(cfg.get("mpc_test_steps", 5000))
    print(f"[test_mpc_withCam] mpc_test_steps={steps} (set bigger to run longer; headless runs as fast as possible)")
    capture_interval = int(camera_cfg.get("capture_interval", 0)) if camera is not None else 0
    try:
        exp_log_cfg = cfg.get("exp_log", None)
        if exp_log_cfg is None:
            exp_log_cfg = {}
        exp_log_cfg = (
            OmegaConf.to_container(exp_log_cfg, resolve=True) if not isinstance(exp_log_cfg, dict) else exp_log_cfg
        )
        exp_log_enable = bool(exp_log_cfg.get("enable", False))
        exp_log_out_dir: Path | None = None
        exp_log_buffers: dict | None = None
        exp_log_n_envs: int | None = None
        exp_log_nu: int | None = None
        ep_id: torch.Tensor | None = None
        ep_len: torch.Tensor | None = None
        ep_ends: list[_EpisodeEnd] = []

        orbit_errors_fn = _maybe_import_orbit_errors() if exp_log_enable else None

        def _save_exp_buffers(*, do_plot: bool) -> None:
            if exp_log_out_dir is None or exp_log_buffers is None or exp_log_n_envs is None or exp_log_nu is None:
                return

            n_envs = int(exp_log_n_envs)
            nu_local = int(exp_log_nu)
            T = int(len(exp_log_buffers.get("step", [])))

            if T <= 0:
                step_arr = np.zeros((0,), dtype=np.int64)
                ep_arr = np.zeros((0, n_envs), dtype=np.int64)
                x_arr = np.zeros((0, n_envs, 13), dtype=np.float32)
                u_arr = np.zeros((0, n_envs, nu_local), dtype=np.float32)
                e_arr = np.zeros((0, n_envs, 10), dtype=np.float32)
                cost_arr = np.zeros((0, n_envs), dtype=np.float32)
                done_arr = np.zeros((0, n_envs), dtype=np.bool_)
                term_arr = np.zeros((0, n_envs), dtype=np.bool_)
                trunc_arr = np.zeros((0, n_envs), dtype=np.bool_)
                succ_arr = np.zeros((0, n_envs), dtype=np.bool_)
                fail_arr = np.zeros((0, n_envs), dtype=np.bool_)
            else:
                keys = [
                    "episode_id",
                    "x",
                    "u",
                    "orbit_err",
                    "cost",
                    "done",
                    "terminated",
                    "truncated",
                    "success",
                    "failure",
                ]
                for k in keys:
                    T = min(T, int(len(exp_log_buffers.get(k, []))))

                step_arr = np.asarray(exp_log_buffers["step"][:T], dtype=np.int64)
                ep_arr = np.stack(exp_log_buffers["episode_id"][:T], axis=0).astype(np.int64, copy=False)  # (T,E)
                x_arr = np.stack(exp_log_buffers["x"][:T], axis=0).astype(np.float32, copy=False)  # (T,E,13)
                u_arr = np.stack(exp_log_buffers["u"][:T], axis=0).astype(np.float32, copy=False)  # (T,E,nu)
                e_arr = np.stack(exp_log_buffers["orbit_err"][:T], axis=0).astype(np.float32, copy=False)  # (T,E,10)
                cost_arr = np.stack(exp_log_buffers["cost"][:T], axis=0).astype(np.float32, copy=False)  # (T,E)
                done_arr = np.stack(exp_log_buffers["done"][:T], axis=0).astype(np.bool_, copy=False)
                term_arr = np.stack(exp_log_buffers["terminated"][:T], axis=0).astype(np.bool_, copy=False)
                trunc_arr = np.stack(exp_log_buffers["truncated"][:T], axis=0).astype(np.bool_, copy=False)
                succ_arr = np.stack(exp_log_buffers["success"][:T], axis=0).astype(np.bool_, copy=False)
                fail_arr = np.stack(exp_log_buffers["failure"][:T], axis=0).astype(np.bool_, copy=False)

            meta = {
                "seed": int(seed),
                "dt": float(cfg.sim.dt),
                "substeps": int(cfg.sim.substeps),
                "max_episode_length": int(cfg.env.max_episode_length),
                "orbit_radius": float(getattr(base_env, "orbit_radius", 0.0)),
                "orbit_z": float(getattr(base_env, "orbit_z", 0.0)),
                "orbit_v_tan": float(getattr(base_env, "orbit_v_tan", 0.0)),
                "orbit_direction": float(getattr(base_env, "orbit_direction", 1.0)),
                "orbit_yaw_offset": float(getattr(base_env, "orbit_yaw_offset", 0.0)),
                "cylinder_center": _to_numpy(getattr(base_env, "cylinder_center", torch.zeros(1, 1, 3)))
                .reshape(-1)
                .tolist(),
                "cylinder_radius": float(getattr(base_env, "cylinder_radius", 0.0)),
                "cylinder_height": float(getattr(base_env, "cylinder_height", 0.0)),
                "mpc_q": {
                    "radial": float(cfg.task.get("mpc_q_radial", 0.0)),
                    "z": float(cfg.task.get("mpc_q_z", 0.0)),
                    "tan": float(cfg.task.get("mpc_q_tan", 0.0)),
                    "radial_speed": float(cfg.task.get("mpc_q_radial_speed", 0.0)),
                    "heading": float(cfg.task.get("mpc_q_heading", 0.0)),
                    "roll": float(cfg.task.get("mpc_q_roll", 0.0)),
                    "pitch": float(cfg.task.get("mpc_q_pitch", 0.0)),
                    "wxy": float(cfg.task.get("mpc_q_wxy", 0.0)),
                },
                "mpc_r_u": float(cfg.task.get("mpc_r_u", 0.0)),
            }

            if bool(exp_log_cfg.get("save_npz", True)):
                tmp_npz = exp_log_out_dir / "rollout_tmp.npz"
                np.savez_compressed(
                    tmp_npz,
                    step=step_arr,
                    episode_id=ep_arr,
                    x=x_arr,
                    u=u_arr,
                    orbit_err=e_arr,
                    cost=cost_arr,
                    done=done_arr,
                    terminated=term_arr,
                    truncated=trunc_arr,
                    success=succ_arr,
                    failure=fail_arr,
                    meta=np.array([meta], dtype=object),
                )
                _atomic_replace(tmp_npz, exp_log_out_dir / "rollout.npz")

            if bool(exp_log_cfg.get("save_csv", False)) and T > 0:
                import pandas as pd  # noqa: WPS433

                rows = []
                for ti in range(int(x_arr.shape[0])):
                    for ei in range(int(x_arr.shape[1])):
                        rows.append(
                            {
                                "t": int(step_arr[ti]),
                                "env": int(ei),
                                "episode_id": int(ep_arr[ti, ei]),
                                "x": float(x_arr[ti, ei, 0]),
                                "y": float(x_arr[ti, ei, 1]),
                                "z": float(x_arr[ti, ei, 2]),
                                "cost": float(cost_arr[ti, ei]),
                                "done": bool(done_arr[ti, ei]),
                                "terminated": bool(term_arr[ti, ei]),
                                "truncated": bool(trunc_arr[ti, ei]),
                                "success": bool(succ_arr[ti, ei]),
                                "failure": bool(fail_arr[ti, ei]),
                            }
                        )
                tmp_csv = exp_log_out_dir / "rollout_summary_tmp.csv"
                pd.DataFrame(rows).to_csv(tmp_csv, index=False)
                _atomic_replace(tmp_csv, exp_log_out_dir / "rollout_summary.csv")

            if ep_ends:
                try:
                    import pandas as pd  # noqa: WPS433

                    tmp_eps = exp_log_out_dir / "episodes_tmp.csv"
                    pd.DataFrame([e.__dict__ for e in ep_ends]).to_csv(tmp_eps, index=False)
                    _atomic_replace(tmp_eps, exp_log_out_dir / "episodes.csv")
                except Exception:
                    pass

            if do_plot and T > 0:
                try:
                    center = np.asarray(meta["cylinder_center"], dtype=np.float64)
                    traj_xyz = x_arr[:, 0, 0:3]
                    tmp_png = exp_log_out_dir / "trajectory_xy_tmp.png"
                    _plot_orbit_xy(
                        out_path=tmp_png,
                        traj_xyz=traj_xyz,
                        center=center,
                        orbit_radius=float(meta["orbit_radius"]),
                        cylinder_radius=float(meta["cylinder_radius"]),
                        n_ref=int(exp_log_cfg.get("plot_num_points", 200)),
                    )
                    _atomic_replace(tmp_png, exp_log_out_dir / "trajectory_xy.png")
                except Exception:
                    pass

        if exp_log_enable:
            root = Path(os.environ.get("MARINEGYM_ROOT", str(Path.cwd()))).expanduser()
            base_out = Path(str(exp_log_cfg.get("out_dir", root / "outputs" / "experiment_logs"))).expanduser()
            run_name = str(exp_log_cfg.get("run_name", "") or "")
            if not run_name:
                ts = time.strftime("%Y%m%d_%H%M%S")
                run_name = f"{cfg.task.name}_test_mpc_{ts}"
            exp_log_out_dir = base_out / run_name
            exp_log_out_dir.mkdir(parents=True, exist_ok=True)
            try:
                (exp_log_out_dir / "cfg_resolved.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
            except Exception:
                pass

            n_envs = int(getattr(base_env, "num_envs", 1))
            nu = int(getattr(base_env.drone, "action_spec", env.action_spec[("agents", "action")]).shape[-1])
            exp_log_n_envs = n_envs
            exp_log_nu = nu
            exp_log_buffers = {
                "step": [],
                "episode_id": [],
                "x": [],
                "u": [],
                "orbit_err": [],
                "cost": [],
                "done": [],
                "terminated": [],
                "truncated": [],
                "success": [],
                "failure": [],
            }
            ep_id = torch.zeros((n_envs,), device=base_env.device, dtype=torch.long)
            ep_len = torch.zeros((n_envs,), device=base_env.device, dtype=torch.long)
            # Create a placeholder file early so users can verify the output path while the sim is running.
            try:
                _save_exp_buffers(do_plot=False)
            except Exception:
                pass

        # Layered orbit state (spiral down by changing OrbitCylinderMPC.orbit_z each lap)
        lap_count = 0
        two_pi = 2.0 * math.pi
        transition = None  # {"start_z":..., "target_z":..., "steps":..., "left":...}
        flush_interval_steps = int(exp_log_cfg.get("flush_interval_steps", 0) or 0) if exp_log_enable else 0
        flush_plot = bool(exp_log_cfg.get("flush_plot", False)) if exp_log_enable else False

        step_idx = 0
        while True:
            if steps >= 0 and step_idx >= steps:
                break

            # Log x_k / orbit errors before stepping (state used by the controller).
            x_k = None
            e_k = None
            if exp_log_enable and exp_log_buffers is not None:
                try:
                    base_env.drone.get_state()
                    x_k = torch.cat([base_env.drone.pos, base_env.drone.rot, base_env.drone.vel_b], dim=-1).squeeze(1)
                except Exception:
                    x_k = None

                if x_k is not None and orbit_errors_fn is not None and hasattr(base_env, "cylinder_center"):
                    try:
                        dtype = x_k.dtype
                        device = x_k.device
                        center_env = base_env.cylinder_center.squeeze(0).expand(x_k.shape[0], 3).to(device=device, dtype=dtype)
                        e_k = orbit_errors_fn(
                            x_k,
                            center_w=center_env,
                            radius=torch.as_tensor(float(getattr(base_env, "orbit_radius", 0.0)), device=device, dtype=dtype),
                            z=torch.as_tensor(float(getattr(base_env, "orbit_z", 0.0)), device=device, dtype=dtype),
                            v_tan=torch.as_tensor(float(getattr(base_env, "orbit_v_tan", 0.0)), device=device, dtype=dtype),
                            dir_sign=torch.as_tensor(float(getattr(base_env, "orbit_direction", 1.0)), device=device, dtype=dtype),
                            yaw_offset=torch.as_tensor(float(getattr(base_env, "orbit_yaw_offset", 0.0)), device=device, dtype=dtype),
                        )
                    except Exception:
                        e_k = None

            if hasattr(dummy_action, "items"):
                td.update(dummy_action)
            else:
                td.set(("agents", "action"), dummy_action)
            td_out = env.step(td)

            # action after internal MPC override (if enabled)
            u_k = None
            try:
                u_k = td_out[("agents", "action")].squeeze(-2)
            except Exception:
                u_k = None

            td_next = td_out["next"]
            td = td_next

            done = td.get("done", None)
            done_mask_for_traj = None
            if done is not None:
                try:
                    done_mask_for_traj = done.squeeze(-1)
                except Exception:
                    done_mask_for_traj = None
            done_val_for_traj = False
            if done_mask_for_traj is not None:
                try:
                    done_val_for_traj = bool(done_mask_for_traj[int(traj_env_id)].item())
                except Exception:
                    done_val_for_traj = False
            _traj_maybe_append(step=int(step_idx) + 1, done=done_val_for_traj)

            if exp_log_enable and exp_log_buffers is not None and ep_id is not None and ep_len is not None:
                n_envs = int(ep_id.numel())

                try:
                    done = td_next.get("done", None)
                    terminated = td_next.get("terminated", None)
                    truncated = td_next.get("truncated", None)
                    if done is None:
                        done = torch.zeros((n_envs, 1), device=base_env.device, dtype=torch.bool)
                    if terminated is None:
                        terminated = torch.zeros_like(done, dtype=torch.bool)
                    if truncated is None:
                        truncated = torch.zeros_like(done, dtype=torch.bool)
                    done_mask = done.squeeze(-1)
                    term_mask = terminated.squeeze(-1)
                    trunc_mask = truncated.squeeze(-1)
                except Exception:
                    done_mask = torch.zeros((n_envs,), device=base_env.device, dtype=torch.bool)
                    term_mask = torch.zeros_like(done_mask, dtype=torch.bool)
                    trunc_mask = torch.zeros_like(done_mask, dtype=torch.bool)

                # stage cost: 0.5 * (e^T W e + u^T W_u u)
                cost = None
                try:
                    if x_k is not None and e_k is not None and u_k is not None:
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
                        max_thruster_force = float(
                            q.get("pypose_max_thruster_force", q.get("mpc_max_thruster_force", 40.0))
                        )
                        w_u = torch.full(
                            (1, int(u_k.shape[-1])),
                            float(r_u) * float(max_thruster_force**2),
                            device=u_k.device,
                            dtype=u_k.dtype,
                        )
                        cost_err = (w_err * e_k.square()).sum(dim=-1)
                        cost_u = (w_u * u_k.square()).sum(dim=-1)
                        cost = 0.5 * (cost_err + cost_u)
                except Exception:
                    cost = None
                if cost is None:
                    cost = torch.zeros((n_envs,), device=base_env.device, dtype=torch.float32)

                # update episode lens and end events
                try:
                    ep_len += 1
                    if bool(done_mask.any()):
                        for i in torch.nonzero(done_mask, as_tuple=False).flatten().tolist():
                            episode_id_i = int(ep_id[i].item())
                            length_i = int(ep_len[i].item())
                            terminated_i = bool(term_mask[i].item())
                            truncated_i = bool(trunc_mask[i].item())
                            success_i = (not terminated_i) and truncated_i
                            ep_ends.append(
                                _EpisodeEnd(
                                    episode_id=episode_id_i,
                                    step=int(step_idx),
                                    length=length_i,
                                    terminated=terminated_i,
                                    truncated=truncated_i,
                                    success=success_i,
                                )
                            )

                        ep_id = ep_id + done_mask.to(dtype=ep_id.dtype)
                        ep_len = torch.where(done_mask, torch.zeros_like(ep_len), ep_len)
                except Exception:
                    pass

                success = (~term_mask) & trunc_mask
                failure = term_mask

                exp_log_buffers["step"].append(int(step_idx))
                exp_log_buffers["episode_id"].append(_coerce_np_array(ep_id, shape=(n_envs,), dtype=np.int64))
                exp_log_buffers["x"].append(_coerce_np_array(x_k, shape=(n_envs, 13), dtype=np.float32))
                exp_log_buffers["u"].append(_coerce_np_array(u_k, shape=(n_envs, int(nu)), dtype=np.float32))
                exp_log_buffers["orbit_err"].append(_coerce_np_array(e_k, shape=(n_envs, 10), dtype=np.float32))
                exp_log_buffers["cost"].append(_coerce_np_array(cost, shape=(n_envs,), dtype=np.float32))
                exp_log_buffers["done"].append(
                    _coerce_np_array(done_mask, shape=(n_envs,), dtype=np.bool_, fill_value=False)
                )
                exp_log_buffers["terminated"].append(
                    _coerce_np_array(term_mask, shape=(n_envs,), dtype=np.bool_, fill_value=False)
                )
                exp_log_buffers["truncated"].append(
                    _coerce_np_array(trunc_mask, shape=(n_envs,), dtype=np.bool_, fill_value=False)
                )
                exp_log_buffers["success"].append(
                    _coerce_np_array(success, shape=(n_envs,), dtype=np.bool_, fill_value=False)
                )
                exp_log_buffers["failure"].append(
                    _coerce_np_array(failure, shape=(n_envs,), dtype=np.bool_, fill_value=False)
                )

                if flush_interval_steps > 0 and (step_idx % flush_interval_steps) == 0:
                    try:
                        _save_exp_buffers(do_plot=flush_plot)
                    except Exception:
                        pass

            if done is not None and bool(done.any()):
                reset_mask = done.squeeze(-1)
                td.set("_reset", reset_mask)
                td = env.reset(td)
                _traj_maybe_append(step=int(step_idx) + 1, done=False)
                try:
                    print("[test_mpc] target_pos[0]:", env.base_env.target_pos[0, 0].detach().cpu().tolist())
                except Exception:
                    pass

            if not cfg.headless:
                env.render()
            if camera is not None and capture_interval > 0 and (step_idx % capture_interval) == 0:
                base_env.sim.render()
                _ = camera.get_images()
            step_idx += 1

            if layered_orbit_enable:
                if transition is not None and transition["left"] > 0:
                    steps_total = max(1, int(transition["steps"]))
                    left = int(transition["left"])
                    frac = float(steps_total - left + 1) / float(steps_total)
                    base_env.orbit_z = float(
                        transition["start_z"] + (transition["target_z"] - transition["start_z"]) * frac
                    )
                    transition["left"] = left - 1
                    if transition["left"] <= 0:
                        base_env.orbit_z = float(transition["target_z"])
                        transition = None

                # Most reliable definition of "one lap" for this task: orbit_period_steps.
                if steps_per_lap > 0 and (step_idx % steps_per_lap) == 0:
                    lap_count += 1
                    next_z = float(getattr(base_env, "orbit_z")) - layered_orbit_delta_z
                    if layered_orbit_transition_steps > 0:
                        transition = {
                            "start_z": float(getattr(base_env, "orbit_z")),
                            "target_z": next_z,
                            "steps": int(layered_orbit_transition_steps),
                            "left": int(layered_orbit_transition_steps),
                        }
                    else:
                        base_env.orbit_z = next_z
                    print(f"[test_mpc_withCam] lap={lap_count} -> orbit_z={float(getattr(base_env, 'orbit_z')):.3f}")
                    if layered_orbit_laps > 0 and lap_count >= layered_orbit_laps:
                        print(f"[test_mpc_withCam] layered_orbit done: laps={layered_orbit_laps}")
                        break
    finally:
        if traj_enable and traj_pos_log:
            try:
                pos_arr = np.stack(traj_pos_log, axis=0).astype(np.float32, copy=False)
                heading_arr = np.stack(traj_heading_log, axis=0).astype(np.float32, copy=False)
                speed_arr = np.asarray(traj_speed_log, dtype=np.float32)
                done_arr = np.asarray(traj_done_log, dtype=np.bool_)
                step_arr = np.asarray(traj_step_log, dtype=np.int64)

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
                                "seed": int(seed),
                                "dt": float(getattr(getattr(cfg, "sim", None), "dt", 0.0)),
                                "traj_env_id": int(traj_env_id),
                                "traj_stride": int(traj_stride),
                                "steps": int(steps),
                            }
                        ],
                        dtype=object,
                    ),
                )
                _atomic_replace(tmp_path, traj_path)
                print(f"[test_mpc_withCam] saved trajectory: {traj_path.resolve()}")

                if traj_plot:
                    from visualize_trajectory import plot_trajectory_3d

                    plot_trajectory_3d(
                        traj_path=traj_path,
                        out_path=traj_png_path,
                        heading_stride=int(traj_plot_heading_stride),
                        arrow_len=float(traj_plot_arrow_len),
                        show=bool(traj_plot_show),
                    )
                    print(f"[test_mpc_withCam] saved trajectory plot: {traj_png_path.resolve()}")
            except Exception as e:
                print(f"[test_mpc_withCam] failed to save trajectory: {e}")

        if exp_log_enable and exp_log_out_dir is not None and exp_log_buffers is not None:
            try:
                _save_exp_buffers(do_plot=bool(exp_log_cfg.get("plot", True)))
                print(f"[test_mpc_withCam] saved experiment logs: {exp_log_out_dir}")
            except Exception as e:
                print(f"[test_mpc_withCam] failed to save experiment logs: {e}")
        simulation_app.close()


if __name__ == "__main__":
    main()
