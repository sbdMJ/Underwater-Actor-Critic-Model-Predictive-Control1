import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401


FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Evaluate PPO policy that outputs MPC Q/p on HoverMPCQP.

    Example:
      python scripts/evaluate_mpc_qp_ppo.py task=Hover_MPC_QP algo=ppo headless=true enable_livestream=false \\
        mode=evaluate env.num_envs=1 +eval.ckpt=checkpoints/checkpoint_final.pt +eval.steps=2000
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_cfg = cfg.get("eval", {}) or {}
    ckpt = eval_cfg.get("ckpt", None)
    if not ckpt:
        ckpt = str(Path.cwd() / "checkpoints" / "checkpoint_final.pt")
    ckpt_path = Path(ckpt).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    simulation_app = init_simulation_app(cfg)

    from marinegym.envs import IsaacEnv, register_tasks

    register_tasks()
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()
    env.set_seed(int(eval_cfg.get("seed", cfg.get("seed", 0))))

    algo_name = str(cfg.algo.name).lower()
    policy = ALGOS[algo_name](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )
    sd = torch.load(str(ckpt_path), map_location="cpu")
    policy.load_state_dict(sd, strict=False)
    policy.eval()

    steps = int(eval_cfg.get("steps", 20000))
    print_every = int(eval_cfg.get("print_every", 100))

    td = env.reset()

    with set_exploration_type(ExplorationType.MODE):
        for t in range(steps):
            td = policy(td)
            action_qp = td[("agents", "action")]
            if hasattr(base_env, "decode_action_to_qp"):
                q, p = base_env.decode_action_to_qp(action_qp)
                if t % print_every == 0:
                    q0 = q[0].detach().cpu().tolist()
                    p0 = p[0].detach().cpu().tolist()
                    print(f"[eval_mpc_qp_ppo] t={t} q[0]={q0} p[0]={p0}")

            td = env.step(td)["next"]
            if t % print_every == 0 and hasattr(base_env, "last_mpc_cmd"):
                try:
                    cmd0 = base_env.last_mpc_cmd[0, 0].detach().cpu().tolist()
                    print(f"[eval_mpc_qp_ppo] t={t} mpc_cmd[0]={cmd0}")
                except Exception:
                    pass
            done = td.get("done", None)
            if done is not None and bool(done.any()):
                reset_mask = done.squeeze(-1)
                td.set("_reset", reset_mask)
                td = env.reset(td)

            if not bool(cfg.headless):
                env.render()

    simulation_app.close()


if __name__ == "__main__":
    main()
