import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)


FILE_PATH = str(Path(__file__).resolve().parent)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    PyPose 기반 differentiable MPC(iLQR) + Replicator camera 스모크 테스트.

    예)
      python scripts/test_pypose_mpc_withCam.py task=Hover_PyPose_MPC headless=false enable_livestream=false env.num_envs=1
      python scripts/test_pypose_mpc_withCam.py task=Hover_PyPose_MPC headless=true enable_livestream=false env.num_envs=1 camera.capture_interval=60

      # MPC differentiability(autograd) 간단 체크(시뮬 그래프가 아닌 MPC solve 자체의 gradient)
      python scripts/test_pypose_mpc_withCam.py task=Hover_PyPose_MPC headless=true enable_livestream=false env.num_envs=1 +diff.enable=true +diff.every=200
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

    seed = int(cfg.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    print("[test_pypose_mpc_withCam] seed:", seed)
    env.set_seed(seed)

    # Optional: spawn a Replicator camera using `marinegym/sensors/camera.py`.
    camera = None
    camera_cfg = cfg.get("camera", None) or {}
    camera_cfg = (
        OmegaConf.to_container(camera_cfg, resolve=True)
        if not isinstance(camera_cfg, dict)
        else camera_cfg
    )
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
            target = tuple(
                float(translation[i] + head_forward[i] * head_distance) for i in range(3)
            )
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
            camera.initialize(
                prim_paths_expr=f"/World/envs/env_.*/{drone_name}_.*/base_link/Camera_.*"
            )
        else:
            camera.initialize()

    td = env.reset()
    dummy_action = env.action_spec.zero()

    if camera is not None and bool(camera_cfg.get("capture_on_reset", True)):
        base_env.sim.render()
        imgs = camera.get_images()
        if "rgb" in imgs:
            print("[test_pypose_mpc_withCam] camera.rgb:", tuple(imgs["rgb"].shape))

    diff_cfg = cfg.get("diff", None) or {}
    diff_cfg = (
        OmegaConf.to_container(diff_cfg, resolve=True) if not isinstance(diff_cfg, dict) else diff_cfg
    )
    diff_enable = bool(diff_cfg.get("enable", False))
    diff_every = int(diff_cfg.get("every", 200))

    steps = int(cfg.get("mpc_test_steps", 300))
    print(f"[test_pypose_mpc_withCam] mpc_test_steps={steps}")
    capture_interval = int(camera_cfg.get("capture_interval", 0)) if camera is not None else 0

    try:
        for step_idx in range(steps if steps >= 0 else 2**31 - 1):
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

            if not cfg.headless:
                env.render()

            if camera is not None and capture_interval > 0 and (step_idx % capture_interval) == 0:
                base_env.sim.render()
                _ = camera.get_images()

            if diff_enable and diff_every > 0 and (step_idx % diff_every) == 0:
                if hasattr(base_env, "controller") and base_env.controller is not None:
                    base_env.drone.get_state()
                    root_state = torch.cat(
                        [base_env.drone.pos, base_env.drone.rot, base_env.drone.vel_b], dim=-1
                    ).squeeze(1)
                    target_pos = base_env.target_pos.squeeze(1)
                    target_quat = base_env.target_rot.squeeze(1)

                    root_state = root_state.detach().clone().requires_grad_(True)
                    target_pos = target_pos.detach().clone().requires_grad_(True)
                    target_quat = target_quat.detach()

                    u = base_env.controller.compute(root_state, target_pos, target_quat=target_quat)
                    loss = (u**2).mean()
                    loss.backward()

                    rs_g = root_state.grad
                    tp_g = target_pos.grad
                    dx_s = (
                        f"|dL/dx|={float(rs_g.norm().detach().cpu()):.6f}"
                        if rs_g is not None
                        else "|dL/dx|=None"
                    )
                    dp_s = (
                        f"|dL/dp*|={float(tp_g.norm().detach().cpu()):.6f}"
                        if tp_g is not None
                        else "|dL/dp*|=None"
                    )
                    print(
                        "[test_pypose_mpc_withCam][diff]",
                        f"step={step_idx}",
                        f"loss={float(loss.detach().cpu()):.6f}",
                        dx_s,
                        dp_s,
                    )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
