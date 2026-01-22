import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)
from marinegym.utils.torchrl import SyncDataCollector


FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


class RolloutReplayBuffer:
    """
    On-policy replay buffer (rollout buffer) for PPO.

    Stores the latest rollout tensordict and returns it for PPO updates.
    """

    def __init__(self):
        self._latest = None

    def add(self, rollout_td):
        self._latest = rollout_td

    def get(self):
        if self._latest is None:
            raise RuntimeError("RolloutReplayBuffer is empty.")
        return self._latest


def _episode_return_stats(batch):
    done = batch.get(("next", "done"), None)
    ret = batch.get(("next", "stats", "return"), None)
    if done is None or ret is None:
        return None

    done = done.squeeze(-1)
    ret = ret.squeeze(-1)
    if done.shape != ret.shape:
        return None

    ep_ret = ret[done]
    if ep_ret.numel() == 0:
        return {"episodes": 0, "mean": None, "last_mean": ret[-1].float().mean().item()}
    return {"episodes": int(ep_ret.numel()), "mean": ep_ret.float().mean().item(), "last_mean": ret[-1].float().mean().item()}


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Train PPO to output MPC Q/p (HoverMPCQP task).

    Example:
      python scripts/train_mpc_qp_ppo.py task=Hover_MPC_QP algo=ppo headless=true enable_livestream=false env.num_envs=8
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)

    from marinegym.envs import IsaacEnv, register_tasks

    register_tasks()
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(int(cfg.get("seed", 0)))

    algo_name = str(cfg.algo.name).lower()
    policy = ALGOS[algo_name](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = int(cfg.get("total_frames", 200_000))
    max_iters = int(cfg.get("max_iters", -1))
    if max_iters > 0:
        total_frames = max_iters * frames_per_batch

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    buffer = RolloutReplayBuffer()

    ckpt_dir = Path.cwd() / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(cfg.get("save_interval", 50))
    print_every = int(cfg.get("print_interval", 10))

    with set_exploration_type(ExplorationType.RANDOM):
        for it, batch in enumerate(collector):
            buffer.add(batch)
            info = policy.train_op(buffer.get())

            if print_every > 0 and it % print_every == 0:
                frames = (it + 1) * frames_per_batch
                total = int(total_frames) if int(total_frames) > 0 else -1
                pct = (100.0 * frames / total) if total > 0 else float("nan")
                rs = _episode_return_stats(batch) or {}
                ep_n = rs.get("episodes", 0)
                ep_mean = rs.get("mean", None)
                last_mean = rs.get("last_mean", None)
                ret_str = (
                    f"ep_return_mean={ep_mean:.4g} (n={ep_n})"
                    if ep_mean is not None
                    else f"batch_last_return_mean={last_mean:.4g}" if last_mean is not None else "return=NA"
                )
                print(
                    f"[train_mpc_qp_ppo] it={it} frames={frames} progress={pct:.2f}% {ret_str} "
                    f"policy_loss={info.get('policy_loss', float('nan')):.4g} "
                    f"value_loss={info.get('value_loss', float('nan')):.4g} "
                    f"entropy={info.get('entropy', float('nan')):.4g}"
                )

            if save_every > 0 and (it % save_every == 0) and it > 0:
                ckpt_path = ckpt_dir / f"checkpoint_{(it+1)*frames_per_batch}.pt"
                torch.save(policy.state_dict(), ckpt_path)

    torch.save(policy.state_dict(), ckpt_dir / "checkpoint_final.pt")
    simulation_app.close()


if __name__ == "__main__":
    main()
