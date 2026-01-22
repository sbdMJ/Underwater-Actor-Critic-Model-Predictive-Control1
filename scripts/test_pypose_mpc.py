import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)


FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    PyPose 기반 differentiable MPC(iLQR) 환경이 실제로 step을 돌 수 있는지 확인하는 스모크 테스트.

    예)
      python scripts/test_pypose_mpc.py task=Hover_PyPose_MPC headless=false enable_livestream=false env.num_envs=1
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
    print("[test_pypose_mpc] seed:", seed)
    env.set_seed(seed)

    td = env.reset()
    dummy_action = env.action_spec.zero()

    steps = int(cfg.get("mpc_test_steps", 300))
    for _ in range(steps):
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

    simulation_app.close()


if __name__ == "__main__":
    main()
