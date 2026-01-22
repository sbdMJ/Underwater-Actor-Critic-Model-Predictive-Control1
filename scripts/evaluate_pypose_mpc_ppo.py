import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)

# ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py   task=Hover_PyPose_MPC algo=ppo_pypose_mpc_qrdiag_tv task.use_internal_mpc=false   headless=false enable_livestream=false env.num_envs=1   +eval.ckpt=/home/mjkim/MarineGym/wandb/offline-run-20260121_150344-p0j9ggir/files/checkpoint_final.pt +eval.steps=4000   +eval.print_every=200 +eval.print_weights_every=200 mode=evaluate



FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


def _maybe_prepare_pypose_mpc_cfg(cfg, base_env, *, algo_name: str, out_dir: Path):
    from marinegym.controllers.thruster_allocation import compute_thruster_allocation_matrix_from_drone

    algo = cfg.algo

    out_dir.mkdir(parents=True, exist_ok=True)
    alloc_path = out_dir / "thruster_allocation.npz"

    if not algo.get("mpc_alloc_npz", None):
        thrust_axis = int(cfg.task.get("thrust_axis", 0))
        B = compute_thruster_allocation_matrix_from_drone(base_env.drone, thrust_axis=thrust_axis)
        np.savez(alloc_path, B=B, quat_order="wxyz")
        algo.mpc_alloc_npz = str(alloc_path)

    if not algo.get("mpc_param_yaml", None):
        algo.mpc_param_yaml = base_env.drone.param_path

    try:
        algo.mpc_mass = float(base_env.drone.MASS_0.squeeze().item())
        algo.mpc_inertia = [float(x) for x in base_env.drone.INERTIA_0.squeeze().tolist()]
    except Exception:
        pass

    algo.mpc_dt = float(cfg.sim.dt)

    try:
        algo.obs_time_encoding_dim = int(getattr(base_env, "time_encoding_dim", 0)) if bool(cfg.task.time_encoding) else 0
    except Exception:
        algo.obs_time_encoding_dim = 0

    try:
        algo.obs_has_target_quat = bool(getattr(base_env, "include_target_quat_in_obs", False))
    except Exception:
        algo.obs_has_target_quat = False

    try:
        algo.mpc_nu = int(base_env.drone.action_spec.shape[-1])
    except Exception:
        algo.mpc_nu = int(getattr(algo, "mpc_nu", 6))

    # Align MPC rollout settings with task defaults unless explicitly overridden.
    if algo_name in ("ppo_pypose_mpc_qrdiag", "ppo_pypose_mpc_qrdiag_tv", "ppo_pypose_mpc_qrconst"):
        algo.mpc_horizon = int(cfg.task.get("mpc_horizon", algo.get("mpc_horizon", 15)))
        algo.mpc_ilqr_iters = int(cfg.task.get("mpc_ilqr_iters", algo.get("mpc_ilqr_iters", 6)))
        algo.max_thruster_force = float(cfg.task.get("max_thruster_force", algo.get("max_thruster_force", 40.0)))

    if algo_name in ("ppo_pypose_mpc_qrdiag", "ppo_pypose_mpc_qrdiag_tv"):
        if not algo.get("wx_init", None):
            q_pos = float(cfg.task.get("mpc_q_pos", 50.0))
            q_quat = float(cfg.task.get("mpc_q_quat", 5.0))
            q_vel = float(cfg.task.get("mpc_q_vel", 2.0))
            q_omega = float(cfg.task.get("mpc_q_omega", 0.5))
            algo.wx_init = [q_pos] * 3 + [q_quat] * 4 + [q_vel] * 3 + [q_omega] * 3
        if not algo.get("wu_init", None):
            r_u = float(cfg.task.get("mpc_r_u", 0.02))
            algo.wu_init = [r_u] * int(algo.mpc_nu)

    if algo_name == "ppo_pypose_mpc_qrconst":
        if algo.get("q_pos_init", None) is None:
            algo.q_pos_init = float(cfg.task.get("mpc_q_pos", 50.0))
        if algo.get("q_quat_init", None) is None:
            algo.q_quat_init = float(cfg.task.get("mpc_q_quat", 5.0))
        if algo.get("q_vel_init", None) is None:
            algo.q_vel_init = float(cfg.task.get("mpc_q_vel", 2.0))
        if algo.get("q_omega_init", None) is None:
            algo.q_omega_init = float(cfg.task.get("mpc_q_omega", 0.5))
        if algo.get("r_u_init", None) is None:
            algo.r_u_init = float(cfg.task.get("mpc_r_u", 0.02))


def _find_pypose_tv_actor(policy) -> Optional[torch.nn.Module]:
    for m in policy.modules():
        if hasattr(m, "cost_map") and hasattr(m, "mpc"):
            # Heuristic: our actor has both cost_map + mpc.
            return m
    return None


def _summarize_weights(w_x_seq: torch.Tensor, w_u_seq: torch.Tensor):
    # w_x_seq: (B,H,13) w_u_seq: (B,H,nu)
    wx0 = w_x_seq[:, 0, :]
    wxL = w_x_seq[:, -1, :]
    wu0 = w_u_seq[:, 0, :]
    wuL = w_u_seq[:, -1, :]

    def _group(wx: torch.Tensor):
        return {
            "q_pos": float(wx[:, 0:3].mean().item()),
            "q_quat": float(wx[:, 3:7].mean().item()),
            "q_vel": float(wx[:, 7:10].mean().item()),
            "q_omega": float(wx[:, 10:13].mean().item()),
        }

    s0 = _group(wx0)
    sL = _group(wxL)
    s0["r_u"] = float(wu0.mean().item())
    sL["r_u"] = float(wuL.mean().item())
    return s0, sL


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Evaluate a PyPose-MPC PPO policy (e.g., ppo_pypose_mpc_qrdiag_tv) on a task.

    Example:
      python scripts/evaluate_pypose_mpc_ppo.py task=Hover_PyPose_MPC algo=ppo_pypose_mpc_qrdiag_tv \\
        task.use_internal_mpc=false headless=true enable_livestream=false env.num_envs=1 mode=evaluate \\
        +eval.ckpt=/path/to/run/checkpoints/checkpoint_final.pt +eval.steps=2000 +eval.print_every=200
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_cfg = cfg.get("eval", {}) or {}
    ckpt = eval_cfg.get("ckpt", None)
    if not ckpt:
        ckpt = str(Path.cwd() / "checkpoints" / "checkpoint_final.pt")
    ckpt_path = Path(str(ckpt)).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    steps = int(eval_cfg.get("steps", 2000))
    seed = int(eval_cfg.get("seed", cfg.get("seed", 0)))
    print_every = int(eval_cfg.get("print_every", 200))
    print_weights_every = int(eval_cfg.get("print_weights_every", print_every))

    simulation_app = init_simulation_app(cfg)

    from marinegym.envs import IsaacEnv, register_tasks

    register_tasks()
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()
    env.set_seed(seed)

    algo_name = str(cfg.algo.name).lower()
    if algo_name.startswith("ppo_pypose_mpc_"):
        _maybe_prepare_pypose_mpc_cfg(cfg, base_env, algo_name=algo_name, out_dir=Path.cwd() / algo_name)

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

    mpc_actor = _find_pypose_tv_actor(policy)

    td = env.reset()
    episode_stats = []

    with set_exploration_type(ExplorationType.MODE):
        for t in range(steps):
            if print_every > 0 and t % print_every == 0:
                try:
                    r = td.get(("stats", "return"), None)
                    pe = td.get(("stats", "pos_error"), None)
                    if r is not None and pe is not None:
                        print(f"[eval] t={t} return={float(r.mean().item()):.4g} pos_error={float(pe.mean().item()):.4g}")
                except Exception:
                    pass

            if mpc_actor is not None and print_weights_every > 0 and t % print_weights_every == 0:
                try:
                    obs = td[("agents", "observation")].squeeze(-2)
                    obs_flat = obs.reshape(-1, obs.shape[-1])
                    with torch.no_grad():
                        w_x_seq, w_u_seq = mpc_actor.cost_map(obs_flat)
                    s0, sL = _summarize_weights(w_x_seq, w_u_seq)
                    print(f"[eval] t={t} w0={s0} wT={sL}")
                except Exception:
                    pass

            td = policy(td)
            td = env.step(td)["next"]

            done = td.get("done", None)
            if done is not None and bool(done.any()):
                done_mask = done.squeeze(-1)
                try:
                    stats_done = td["stats"][done_mask].cpu()
                    episode_stats.extend(stats_done.unbind(0))
                except Exception:
                    pass

                td.set("_reset", done_mask)
                td = env.reset(td)

            if not bool(cfg.headless):
                env.render()

    if episode_stats:
        stats_td = torch.stack(episode_stats).to_tensordict()
        summary = {k: float(v.float().mean().item()) for k, v in stats_td.items(True, True)}
        print(f"[eval] episodes={len(episode_stats)} mean_stats={summary}")
    else:
        print("[eval] No completed episodes during evaluation (increase eval.steps or reduce max_episode_length).")

    simulation_app.close()


if __name__ == "__main__":
    main()

