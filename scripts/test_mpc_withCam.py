import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)


FILE_PATH = os.path.dirname(__file__)

os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    MPC 기반 HoverMPC 환경이 실제로 step을 돌 수 있는지 확인하는 간단한 스모크 테스트.

    예)
      python scripts/test_mpc_withCam.py task=Hover_MPC headless=false enable_livestream=false env.num_envs=1
      python scripts/test_mpc_withCam.py task=Hover_MPC camera.capture_interval=60

      ~/isaac410/python.sh scripts/test_mpc_withCam.py task=OrbitCylinder_MPC headless=false enable_livestream=false env.num_envs=1 +camera.head_offset='[0.4,0,0.15]'
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

    if camera is not None and bool(camera_cfg.get("capture_on_reset", True)):
        base_env.sim.render()
        imgs = camera.get_images()
        if "rgb" in imgs.keys():
            print("[test_mpc_withCam] camera.rgb:", tuple(imgs["rgb"].shape))

    steps = int(cfg.get("mpc_test_steps", 5000))
    capture_interval = int(camera_cfg.get("capture_interval", 0)) if camera is not None else 0
    for step_idx in range(steps):
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
            try:
                print("[test_mpc] target_pos[0]:", env.base_env.target_pos[0, 0].detach().cpu().tolist())
            except Exception:
                pass
        if not cfg.headless:
            env.render()
        if camera is not None and capture_interval > 0 and (step_idx % capture_interval) == 0:
            base_env.sim.render()
            _ = camera.get_images()

    simulation_app.close()


if __name__ == "__main__":
    main()
