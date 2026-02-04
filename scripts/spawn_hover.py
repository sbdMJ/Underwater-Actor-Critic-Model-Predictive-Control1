import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)


FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Hover task에서 지정한 드론 모델을 "띄우기만" 하는 최소 스크립트.

    예)
      /home/mjkim/isaac410/python.sh scripts/spawn_hover.py \\
        task=Hover headless=false enable_livestream=false env.num_envs=1 \\
        task.drone_model.name=BlueROVHeavy spawn_duration_s=10.0
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

    td = env.reset()

    # Render once so the stage is visible immediately in non-headless mode.
    if not cfg.headless:
        env.render()

    drone = env.base_env.drone
    num_rotors = int(getattr(drone, "num_rotors", 0))
    dt = float(cfg.sim.dt)
    duration_s = float(cfg.get("spawn_duration_s", 5.0))
    steps = max(1, int(round(duration_s / dt)))

    print(
        "[spawn_hover] task:",
        cfg.task.name,
        "model:",
        cfg.task.drone_model.name,
        "dt:",
        dt,
        "duration_s:",
        duration_s,
        "steps:",
        steps,
        "num_rotors:",
        num_rotors,
    )

    if num_rotors > 0:
        action = torch.zeros(*drone.shape, num_rotors, device=drone.device, dtype=torch.float32)
        for _ in range(steps):
            td.set(("agents", "action"), action)
            td = env.step(td)["next"]
            if not cfg.headless:
                env.render()

    simulation_app.close()


if __name__ == "__main__":
    main()

