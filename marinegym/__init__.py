import os
import sys

import torch
from tensordict import TensorDict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")


def init_simulation_app(cfg):
    try:
        from isaacsim import SimulationApp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Isaac Sim(isaacsim)이 설치되어 있어야 시뮬레이터를 실행할 수 있습니다."
        ) from e
    # launch the simulator
    # config = {"headless": cfg["headless"], "anti_aliasing": 1}
    config = {
        "headless": cfg["headless"],
        "enable_livestream": cfg["enable_livestream"],
        "anti_aliasing": 1,
        "width": 1280,
        "height": 720,
        "window_width": 1920,
        "window_height": 1080,
        "renderer": "RayTracedLighting",
        "display_options": 3286,  # Set display options to show default grid
    }
    try:
        from omegaconf import OmegaConf

        sim_app_cfg = cfg.get("simulation_app", None)
        if sim_app_cfg is not None:
            sim_app_cfg = OmegaConf.to_container(sim_app_cfg, resolve=True)
            if isinstance(sim_app_cfg, dict):
                config.update({k: v for k, v in sim_app_cfg.items() if v is not None})
    except Exception:
        pass
    # load cheaper kit config in headless
    # if cfg.headless:
    #     app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    # else:
    #     app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
    original_argv = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]]
        simulation_app = SimulationApp(config, experience=app_experience)
    finally:
        sys.argv = original_argv
    
    if config['enable_livestream']:
        from omni.isaac.core.utils.extensions import enable_extension
        simulation_app.set_setting("/app/window/drawMouse", True)
        simulation_app.set_setting("/app/livestream/proto", "ws")
        simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        simulation_app.set_setting("/ngx/enabled", False)
        # Note: Only one livestream extension can be enabled at a time
        # Enable Native Livestream extension
        # Default App: Streaming Client from the Omniverse Launcher
        # enable_extension("omni.kit.livestream.native")

        # Enable WebSocket Livestream extension(Deprecated)
        # Default URL: http://localhost:8211/streaming/client/
        # enable_extension("omni.services.streamclient.websocket")

        # Enable WebRTC Livestream extension
        # Default URL: http://localhost:8211/streaming/webrtc-client/
        enable_extension("omni.services.streamclient.webrtc")
    
    return simulation_app


def _get_shapes(self: TensorDict):
    return {
        k: v.shape if isinstance(v, torch.Tensor) else v.shapes for k, v in self.items()
    }


def _get_devices(self: TensorDict):
    return {
        k: v.device if isinstance(v, torch.Tensor) else v.devices
        for k, v in self.items()
    }


TensorDict.shapes = property(_get_shapes)
TensorDict.devices = property(_get_devices)
