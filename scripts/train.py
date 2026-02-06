import logging
import os
import time
import json
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from marinegym import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from marinegym.utils.torchrl import SyncDataCollector
from marinegym.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from marinegym.utils.wandb import init_wandb
from marinegym.utils.torchrl import RenderCallback, EpisodeStats
from marinegym.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

# ~/isaac410/python.sh scripts/train.py task=Hover_PyPose_MPC algo=ppo_pypose_mpc_qrdiag_tv task.use_internal_mpc=false headless=true enable_livestream=false env.num_envs=10
# ~/isaac410/python.sh scripts/train.py   task=OrbitCylinder_MPC algo=ppo_pypose_cylinder_mpc_werr_wu_tv   task.reward_mode=orbit_cost task.orbit_target_mode=auto   task.use_internal_mpc=false task.include_cylinder_rel_in_obs=false   headless=false enable_livestream=false env.num_envs=1



os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_rank0 = rank == 0

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        # Use per-process single GPU.
        cfg.sim.device = f"cuda:{local_rank}"
        if cfg.get("simulation_app", None) is None:
            cfg.simulation_app = {}
        cfg.simulation_app["active_gpu"] = int(local_rank)
        cfg.simulation_app["physics_gpu"] = int(local_rank)
        cfg.simulation_app["max_gpu_count"] = 1
        cfg.simulation_app["multi_gpu"] = False

        # Avoid port conflicts / redundant rendering.
        if not is_rank0:
            cfg.enable_livestream = False

        ddp_env_mode = str(cfg.get("ddp_env_mode", "keep_total")).lower()
        if ddp_env_mode in ("keep_total", "total"):
            total_envs = int(cfg.env.num_envs)
            if total_envs % world_size != 0:
                raise ValueError(f"ddp_env_mode=keep_total requires env.num_envs divisible by WORLD_SIZE. got {total_envs=} {world_size=}")
            per_rank_envs = total_envs // world_size
            cfg.env.num_envs = per_rank_envs
            try:
                cfg.task.env.num_envs = per_rank_envs
            except Exception:
                pass

        ddp_total_frames_mode = str(cfg.get("ddp_total_frames_mode", "global")).lower()
        if ddp_total_frames_mode in ("global", "scale_down") and int(cfg.get("total_frames", -1)) > 0:
            total_frames = int(cfg.total_frames)
            if total_frames < world_size:
                raise ValueError(f"ddp_total_frames_mode=global requires total_frames >= WORLD_SIZE. got {total_frames=} {world_size=}")
            cfg.total_frames = total_frames // world_size

    if str(cfg.task.name) == "OrbitCylinderMPC" and str(cfg.algo.name).lower() == "ppo":
        cfg.task.control_mode = "direct"
        cfg.task.use_internal_mpc = False
    simulation_app = init_simulation_app(cfg)
    run = None
    run_dir = None
    if (not is_distributed) or is_rank0:
        run = init_wandb(cfg)
        run_dir = run.dir
        run_name = run.name
    else:
        run_name = None

    if is_distributed:
        obj_list = [run_dir, run_name]
        dist.broadcast_object_list(obj_list, src=0, device=torch.device(f"cuda:{local_rank}"))
        run_dir, run_name = obj_list

    if run is None:
        class _DummyRun:
            def __init__(self, *, name: str, dir: str):
                self.name = name
                self.dir = dir

            def log(self, *args, **kwargs):
                pass

            def log_artifact(self, *args, **kwargs):
                pass

        run = _DummyRun(name=str(run_name), dir=str(run_dir))

    setproctitle(f"{run.name}-rank{rank}" if is_distributed else run.name)
    if is_rank0:
        print(OmegaConf.to_yaml(cfg))

    metrics_f = None
    if is_rank0:
        metrics_path = Path(run.dir) / "metrics.jsonl"
        metrics_f = open(metrics_path, "a", encoding="utf-8")
    log_interval = int(cfg.get("log_interval", 50))
    print_float_metrics = bool(cfg.get("print_float_metrics", False))
    flush_interval_steps = int(cfg.get("flush_interval_steps", 0))
    if flush_interval_steps <= 0:
        try:
            flush_interval_steps = int(cfg.get("exp_log", {}).get("flush_interval_steps", 0))
        except Exception:
            flush_interval_steps = 0
    if flush_interval_steps <= 0:
        flush_interval_steps = log_interval

    from marinegym.envs import IsaacEnv, register_tasks
    register_tasks()

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    algo_name = str(cfg.algo.name).lower()
    if algo_name in (
        "ac_mpc",
        "ac_mpc_pypose",
        "ppo_pypose_mpc_qrdiag",
        "ppo_pypose_mpc_qrconst",
        "ppo_pypose_mpc_qrdiag_tv",
        "ppo_pypose_cylinder_mpc_werr_wu_tv",
        "ppo_acados_pypose_mpc_qrdiag_tv",
        "ppo_acados_hover_mpc_qrdiag",
    ):
        from marinegym.controllers.thruster_allocation import (
            compute_thruster_allocation_matrix_from_drone,
        )

        alloc_dir = Path(run.dir) / algo_name
        if is_rank0:
            alloc_dir.mkdir(parents=True, exist_ok=True)
        alloc_path = alloc_dir / "thruster_allocation.npz"

        if not cfg.algo.get("mpc_alloc_npz", None):
            cfg.algo.mpc_alloc_npz = str(alloc_path)
            if is_rank0:
                thrust_axis = int(cfg.task.get("thrust_axis", 0))
                B = compute_thruster_allocation_matrix_from_drone(
                    base_env.drone, thrust_axis=thrust_axis
                )
                np.savez(alloc_path, B=B, quat_order="wxyz")
        if is_distributed:
            dist.barrier()

        if not cfg.algo.get("mpc_param_yaml", None):
            cfg.algo.mpc_param_yaml = base_env.drone.param_path
        # Pass sim-true dynamics params when YAML doesn't include them.
        try:
            cfg.algo.mpc_mass = float(base_env.drone.MASS_0.squeeze().item())
            cfg.algo.mpc_inertia = [float(x) for x in base_env.drone.INERTIA_0.squeeze().tolist()]
        except Exception:
            pass

        cfg.algo.mpc_dt = float(cfg.sim.dt)
        # Help the MPC head parse observation layout.
        try:
            cfg.algo.obs_time_encoding_dim = int(getattr(base_env, "time_encoding_dim", 0)) if bool(cfg.task.time_encoding) else 0
        except Exception:
            cfg.algo.obs_time_encoding_dim = 0
        try:
            cfg.algo.obs_has_target_quat = bool(getattr(base_env, "include_target_quat_in_obs", False))
        except Exception:
            cfg.algo.obs_has_target_quat = False
        if hasattr(base_env, "drone"):
            cfg.algo.mpc_nu = int(base_env.drone.action_spec.shape[-1])
        else:
            try:
                action_leaf = base_env.action_spec[("agents", "action")]
            except Exception:
                action_leaf = base_env.action_spec
            cfg.algo.mpc_nu = int(action_leaf.shape[-1])
        # Hint for per-env MPC warm-start caches (used by acados MPC PPO variants).
        try:
            cfg.algo.env_num_envs = int(env.num_envs)
        except Exception:
            pass
        if algo_name in ("ac_mpc", "ac_mpc_pypose"):
            cfg.algo.use_mpc_head = True
            if not cfg.algo.get("acados_codegen_dir", None):
                cfg.algo.acados_codegen_dir = str(alloc_dir / "acados_codegen")

        if algo_name == "ppo_acados_pypose_mpc_qrdiag_tv":
            if not cfg.algo.get("acados_codegen_dir", None):
                cfg.algo.acados_codegen_dir = str(alloc_dir / "acados_codegen")

            cfg.algo.mpc_horizon = int(cfg.task.get("mpc_horizon", cfg.algo.get("mpc_horizon", 15)))
            cfg.algo.max_thruster_force = float(cfg.task.get("max_thruster_force", cfg.algo.get("max_thruster_force", 40.0)))

            if not cfg.algo.get("wx_init", None):
                q_pos = float(cfg.task.get("mpc_q_pos", 50.0))
                q_quat = float(cfg.task.get("mpc_q_quat", 5.0))
                q_vel = float(cfg.task.get("mpc_q_vel", 2.0))
                q_omega = float(cfg.task.get("mpc_q_omega", 0.5))
                cfg.algo.wx_init = [q_pos] * 3 + [q_quat] * 4 + [q_vel] * 3 + [q_omega] * 3

            if not cfg.algo.get("wu_init", None):
                r_u = float(cfg.task.get("mpc_r_u", 0.02))
                cfg.algo.wu_init = [r_u] * int(cfg.algo.mpc_nu)

        if algo_name == "ppo_acados_hover_mpc_qrdiag":
            if not cfg.algo.get("acados_codegen_dir", None):
                cfg.algo.acados_codegen_dir = str(alloc_dir / "acados_codegen")

            cfg.algo.mpc_horizon = int(cfg.task.get("mpc_horizon", cfg.algo.get("mpc_horizon", 20)))
            cfg.algo.max_thruster_force = float(cfg.task.get("max_thruster_force", cfg.algo.get("max_thruster_force", 40.0)))

            if not cfg.algo.get("wx_init", None):
                q_pos = float(cfg.task.get("mpc_q_pos", 50.0))
                q_quat = float(cfg.task.get("mpc_q_quat", 5.0))
                q_vel = float(cfg.task.get("mpc_q_vel", 2.0))
                q_omega = float(cfg.task.get("mpc_q_omega", 0.5))
                cfg.algo.wx_init = [q_pos] * 3 + [q_quat] * 4 + [q_vel] * 3 + [q_omega] * 3

            if not cfg.algo.get("wu_init", None):
                r_u = float(cfg.task.get("mpc_r_u", 0.01))
                cfg.algo.wu_init = [r_u] * int(cfg.algo.mpc_nu)

        if algo_name == "ppo_pypose_mpc_qrdiag":
            # Keep MPC rollout settings aligned with the task defaults unless overridden.
            cfg.algo.mpc_horizon = int(cfg.task.get("mpc_horizon", cfg.algo.get("mpc_horizon", 15)))
            cfg.algo.mpc_ilqr_iters = int(cfg.task.get("mpc_ilqr_iters", cfg.algo.get("mpc_ilqr_iters", 6)))
            cfg.algo.max_thruster_force = float(cfg.task.get("max_thruster_force", cfg.algo.get("max_thruster_force", 40.0)))

            if not cfg.algo.get("wx_init", None):
                q_pos = float(cfg.task.get("mpc_q_pos", 50.0))
                q_quat = float(cfg.task.get("mpc_q_quat", 5.0))
                q_vel = float(cfg.task.get("mpc_q_vel", 2.0))
                q_omega = float(cfg.task.get("mpc_q_omega", 0.5))
                cfg.algo.wx_init = [q_pos] * 3 + [q_quat] * 4 + [q_vel] * 3 + [q_omega] * 3

            if not cfg.algo.get("wu_init", None):
                r_u = float(cfg.task.get("mpc_r_u", 0.02))
                cfg.algo.wu_init = [r_u] * int(cfg.algo.mpc_nu)

        if algo_name == "ppo_pypose_mpc_qrdiag_tv":
            # Time-varying horizon weights (still inferred from task defaults for init).
            cfg.algo.mpc_horizon = int(cfg.task.get("mpc_horizon", cfg.algo.get("mpc_horizon", 15)))
            cfg.algo.mpc_ilqr_iters = int(cfg.task.get("mpc_ilqr_iters", cfg.algo.get("mpc_ilqr_iters", 6)))
            cfg.algo.max_thruster_force = float(cfg.task.get("max_thruster_force", cfg.algo.get("max_thruster_force", 40.0)))

            if not cfg.algo.get("wx_init", None):
                q_pos = float(cfg.task.get("mpc_q_pos", 50.0))
                q_quat = float(cfg.task.get("mpc_q_quat", 5.0))
                q_vel = float(cfg.task.get("mpc_q_vel", 2.0))
                q_omega = float(cfg.task.get("mpc_q_omega", 0.5))
                cfg.algo.wx_init = [q_pos] * 3 + [q_quat] * 4 + [q_vel] * 3 + [q_omega] * 3

            if not cfg.algo.get("wu_init", None):
                r_u = float(cfg.task.get("mpc_r_u", 0.02))
                cfg.algo.wu_init = [r_u] * int(cfg.algo.mpc_nu)

        if algo_name == "ppo_pypose_cylinder_mpc_werr_wu_tv":
            # Orbit-cylinder cost weights (w_err, w_u) for a differentiable PyPose iLQR MPC.
            cfg.algo.mpc_horizon = int(
                cfg.task.get("pypose_mpc_horizon", cfg.task.get("mpc_horizon", cfg.algo.get("mpc_horizon", 15)))
            )
            cfg.algo.mpc_ilqr_iters = int(
                cfg.task.get(
                    "pypose_mpc_ilqr_iters",
                    cfg.task.get("mpc_ilqr_iters", cfg.algo.get("mpc_ilqr_iters", 6)),
                )
            )
            cfg.algo.mpc_ilqr_reg = float(cfg.task.get("pypose_mpc_ilqr_reg", cfg.algo.get("mpc_ilqr_reg", 1e-3)))
            cfg.algo.terminal_weight_mult = float(
                cfg.task.get("pypose_mpc_terminal_weight_mult", cfg.algo.get("terminal_weight_mult", 10.0))
            )
            cfg.algo.max_thruster_force = float(
                cfg.task.get(
                    "pypose_max_thruster_force",
                    cfg.task.get(
                        "mpc_max_thruster_force",
                        cfg.task.get("max_thruster_force", cfg.algo.get("max_thruster_force", 40.0)),
                    ),
                )
            )

            cfg.algo.orbit_radius = float(cfg.task.get("orbit_radius", cfg.algo.get("orbit_radius", 1.4)))
            cfg.algo.orbit_direction = 1.0 if float(cfg.task.get("orbit_direction", 1.0)) >= 0.0 else -1.0
            cfg.algo.orbit_yaw_offset = float(cfg.task.get("orbit_yaw_offset", 0.0))

            orbit_v_tan = float(cfg.task.get("orbit_v_tan", 0.0))
            if orbit_v_tan <= 0.0:
                orbit_period_steps = int(cfg.task.get("orbit_period_steps", cfg.task.get("max_episode_length", 1)))
                dt = float(cfg.sim.dt)
                r = float(cfg.algo.orbit_radius)
                orbit_v_tan = float(2.0 * np.pi * r / (max(1, orbit_period_steps) * dt)) if r > 1e-6 else 0.0
            cfg.algo.orbit_v_tan = float(orbit_v_tan)

            # The actor uses target-at-origin coordinates from Hover obs (pos_rel = drone_pos - target_pos).
            # If the target is the moving waypoint (OrbitCylinderMPC auto mode + use_internal_mpc=false),
            # then z in that relative frame should be 0 because target_z == orbit_z.
            orbit_target_mode = str(cfg.task.get("orbit_target_mode", "auto")).lower()
            if orbit_target_mode in ("auto", ""):
                reward_mode = str(cfg.task.get("reward_mode", "hover")).lower()
                if reward_mode in ("orbit_cost", "cylinder_cost", "cylinder_orbit_cost", "orbit"):
                    orbit_target_mode = "center"
                else:
                    orbit_target_mode = "waypoint" if not bool(cfg.task.get("use_internal_mpc", True)) else "center"
            if orbit_target_mode in ("waypoint", "moving_waypoint", "wp"):
                cfg.algo.orbit_z = 0.0
            else:
                center_cfg = cfg.task.get("cylinder_center", [0.0, 0.0, 0.0])
                cfg.algo.orbit_z = float(cfg.task.get("orbit_z", float(center_cfg[2]))) - float(center_cfg[2])

            cfg.algo.obs_has_cylinder_rel = bool(cfg.task.get("include_cylinder_rel_in_obs", False))

            if not cfg.algo.get("werr_init", None):
                q_radial = float(cfg.task.get("mpc_q_radial", 50.0))
                q_z = float(cfg.task.get("mpc_q_z", 30.0))
                q_tan = float(cfg.task.get("mpc_q_tan", 10.0))
                q_radial_speed = float(cfg.task.get("mpc_q_radial_speed", 5.0))
                q_heading = float(cfg.task.get("mpc_q_heading", 30.0))
                q_roll = float(cfg.task.get("mpc_q_roll", 60.0))
                q_pitch = float(cfg.task.get("mpc_q_pitch", 60.0))
                q_wxy = float(cfg.task.get("mpc_q_wxy", 0.5))
                cfg.algo.werr_init = [
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
                ]

            if not cfg.algo.get("wu_init", None):
                r_u = float(cfg.task.get("mpc_r_u", 0.01))
                cfg.algo.wu_init = [r_u] * int(cfg.algo.mpc_nu)

        if algo_name == "ppo_pypose_mpc_qrconst":
            cfg.algo.mpc_horizon = int(cfg.task.get("mpc_horizon", cfg.algo.get("mpc_horizon", 15)))
            cfg.algo.mpc_ilqr_iters = int(cfg.task.get("mpc_ilqr_iters", cfg.algo.get("mpc_ilqr_iters", 6)))
            cfg.algo.max_thruster_force = float(cfg.task.get("max_thruster_force", cfg.algo.get("max_thruster_force", 40.0)))

            if cfg.algo.get("q_pos_init", None) is None:
                cfg.algo.q_pos_init = float(cfg.task.get("mpc_q_pos", 50.0))
            if cfg.algo.get("q_quat_init", None) is None:
                cfg.algo.q_quat_init = float(cfg.task.get("mpc_q_quat", 5.0))
            if cfg.algo.get("q_vel_init", None) is None:
                cfg.algo.q_vel_init = float(cfg.task.get("mpc_q_vel", 2.0))
            if cfg.algo.get("q_omega_init", None) is None:
                cfg.algo.q_omega_init = float(cfg.task.get("mpc_q_omega", 0.5))
            if cfg.algo.get("r_u_init", None) is None:
                cfg.algo.r_u_init = float(cfg.task.get("mpc_r_u", 0.02))

        # Keep W&B config in sync with runtime-updated fields (mpc_dt, mpc_alloc_npz, etc).
        if is_rank0:
            try:
                run.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
            except Exception:
                pass

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(int(cfg.seed) + rank if is_distributed else int(cfg.seed))

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        ddp_find_unused_parameters = bool(cfg.get("ddp_find_unused_parameters", False))
        try:
            if hasattr(policy, "actor") and isinstance(getattr(policy, "actor"), torch.nn.Module):
                policy.actor = DDP(
                    policy.actor,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=ddp_find_unused_parameters,
                )
            if hasattr(policy, "critic") and isinstance(getattr(policy, "critic"), torch.nn.Module):
                policy.critic = DDP(
                    policy.critic,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=ddp_find_unused_parameters,
                )
        except Exception as e:
            raise RuntimeError("Failed to wrap policy.actor/critic with DDP. Set ddp_find_unused_parameters=true if you suspect unused params.") from e

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
    save_every_frames = cfg.get("save_every_frames", -1)
    # Save checkpoints by training progress.
    # Default: every 10% of training (override with `save_every_percent`; set <=0 to disable).
    save_every_percent = float(cfg.get("save_every_percent", 0.1))

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys) if is_rank0 else None
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    stop_file = cfg.get("stop_file", None)
    if stop_file is None:
        stop_file = str(Path(run.dir) / "STOP")

    def _state_dict_for_checkpoint() -> dict:
        state_dict = policy.state_dict()
        if not is_distributed:
            return state_dict
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("actor.module."):
                k = "actor." + k[len("actor.module.") :]
            elif k.startswith("critic.module."):
                k = "critic." + k[len("critic.module.") :]
            stripped[k] = v
        return stripped

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE
    ):

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
            format="mp4"
        )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info

    # TorchRL's ProbabilisticActor is deterministic unless an exploration type is set.
    # Without this, actions stay near 0 at init which can fall into the thruster deadzone.
    with set_exploration_type(ExplorationType.RANDOM):
        ckpt_dir = Path(run.dir) / "checkpoints"
        if is_rank0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt_frames = -1
        next_ckpt_target_frames = None
        next_ckpt_target_iter = None

        if save_every_percent > 0:
            if total_frames > 0:
                # Align targets to frames_per_batch so we don't miss checkpoints due to batching.
                def _align(frames: int) -> int:
                    return int((int(frames) + frames_per_batch - 1) // frames_per_batch * frames_per_batch)

                next_ckpt_target_frames = _align(total_frames * save_every_percent)
            elif max_iters > 0:
                next_ckpt_target_iter = max(1, int(max_iters * save_every_percent))

        def _maybe_save_checkpoint(force: bool = False):
            nonlocal last_ckpt_frames, next_ckpt_target_frames, next_ckpt_target_iter
            if not is_rank0:
                return
            if save_interval <= 0 and save_every_frames <= 0 and not force:
                if save_every_percent <= 0:
                    return
            if not force:
                should_save = False
                if save_interval > 0 and i % save_interval == 0:
                    should_save = True
                if save_every_frames > 0:
                    if last_ckpt_frames < 0:
                        should_save = collector._frames >= save_every_frames
                    else:
                        should_save = (collector._frames - last_ckpt_frames) >= save_every_frames
                if save_every_percent > 0:
                    if next_ckpt_target_frames is not None:
                        should_save = should_save or (collector._frames >= next_ckpt_target_frames)
                    if next_ckpt_target_iter is not None:
                        should_save = should_save or (i >= next_ckpt_target_iter)
                if not should_save:
                    return
            try:
                ckpt_path = ckpt_dir / f"checkpoint_{collector._frames}.pt"
                torch.save(_state_dict_for_checkpoint(), ckpt_path)
                last_ckpt_frames = int(collector._frames)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")

                # Advance percentage-based targets after a successful save.
                if save_every_percent > 0:
                    if next_ckpt_target_frames is not None and total_frames > 0:
                        def _align(frames: int) -> int:
                            return int((int(frames) + frames_per_batch - 1) // frames_per_batch * frames_per_batch)

                        while next_ckpt_target_frames is not None and collector._frames >= next_ckpt_target_frames:
                            next_ckpt_target_frames = _align(next_ckpt_target_frames + total_frames * save_every_percent)
                            if next_ckpt_target_frames > total_frames:
                                next_ckpt_target_frames = None
                    if next_ckpt_target_iter is not None and max_iters > 0:
                        while next_ckpt_target_iter is not None and i >= next_ckpt_target_iter:
                            next_ckpt_target_iter = int(next_ckpt_target_iter + max_iters * save_every_percent)
                            if next_ckpt_target_iter >= max_iters:
                                next_ckpt_target_iter = None
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        pbar = tqdm(collector, total=total_frames//frames_per_batch, disable=not is_rank0)
        env.train()
        for i, data in enumerate(pbar):
            if os.path.exists(stop_file):
                logging.info(f"Stop file detected: {stop_file}")
                break
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            if episode_stats is not None:
                episode_stats.add(data.to_tensordict())
                if len(episode_stats) >= base_env.num_envs:
                    stats = {
                        "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                        for k, v in episode_stats.pop().items(True, True)
                    }
                    info.update(stats)

            info.update(policy.train_op(data.to_tensordict()))
            # Action diagnostics (helps catch thruster deadzone / saturation).
            if is_rank0 and log_interval > 0 and i % log_interval == 0:
                try:
                    actions = None
                    for k in (("agents", "action"), "action", ("action",), ("collector", "action")):
                        try:
                            actions = data.get(k)
                            break
                        except Exception:
                            actions = None
                    if actions is None:
                        raise KeyError("action key not found")
                    info["train/action_abs_mean"] = actions.abs().mean().item()
                    # Thruster model has a real deadzone around ~0.075; we also track a wider "low action" band.
                    info["train/action_deadzone_frac"] = (actions.abs() < 0.08).float().mean().item()
                    info["train/action_low_frac"] = (actions.abs() < 0.2).float().mean().item()
                    info["train/action_sat_frac"] = (actions.abs() > 0.99).float().mean().item()
                except Exception:
                    pass
            # MPC solver diagnostics (acados failures often look like "robot doesn't move").
            if algo_name in ("ac_mpc", "ac_mpc_pypose"):
                try:
                    mpc_layer = None
                    mpc_actor = None
                    for m in policy.modules():
                        if m.__class__.__name__ == "AcadosMPCLayer":
                            mpc_layer = m
                        if m.__class__.__name__ in ("MPCActor", "MPCActorPyPose"):
                            mpc_actor = m
                        if m.__class__.__name__ == "ILQRMPCLayer":
                            mpc_layer = mpc_layer or m
                    if mpc_layer is not None and hasattr(mpc_layer, "get_and_reset_diagnostics"):
                        info.update({f"train/{k}": v for k, v in mpc_layer.get_and_reset_diagnostics().items()})
                    if mpc_actor is not None and hasattr(mpc_actor, "actor_std"):
                        info["train/actor_log_std_mean"] = float(mpc_actor.actor_std.detach().mean().item())
                        info["train/actor_log_std_min"] = float(mpc_actor.actor_std.detach().min().item())
                        info["train/actor_log_std_max"] = float(mpc_actor.actor_std.detach().max().item())
                    info["train/actor_param_count"] = float(
                        sum(p.numel() for p in policy.actor.parameters() if getattr(p, "requires_grad", False))
                    )
                except Exception:
                    pass

            if is_rank0 and eval_interval > 0 and i % eval_interval == 0:
                logging.info(f"Eval at {collector._frames} steps.")
                info.update(evaluate())
                env.train()
                base_env.train()

            _maybe_save_checkpoint()

            if is_rank0:
                run.log(info)
                if print_float_metrics and log_interval > 0 and i % log_interval == 0:
                    print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
                if metrics_f is not None:
                    try:
                        payload = {k: v for k, v in info.items() if isinstance(v, (int, float))}
                        metrics_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        if flush_interval_steps > 0 and i % flush_interval_steps == 0:
                            metrics_f.flush()
                    except Exception:
                        pass

                pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

            if max_iters > 0 and i >= max_iters - 1:
                break

    if is_rank0:
        logging.info(f"Final Eval at {collector._frames} steps.")
        info = {"env_frames": collector._frames}
        # info.update(evaluate())
        # run.log(info)

        try:
            ckpt_dir = Path(run.dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = str(Path(run.dir) / "checkpoint_final.pt")
            ckpt_path_copy = str(ckpt_dir / "checkpoint_final.pt")
            torch.save(_state_dict_for_checkpoint(), ckpt_path)
            # Convenience copy alongside intermediate checkpoints.
            try:
                torch.save(_state_dict_for_checkpoint(), ckpt_path_copy)
            except Exception:
                pass

            model_artifact = wandb.Artifact(
                f"{cfg.task.name}-{cfg.algo.name.lower()}-{cfg.task.drone_model.name}",
                type="model",
                description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
                metadata=dict(cfg))

            model_artifact.add_file(ckpt_path)
            wandb.save(ckpt_path)
            run.log_artifact(model_artifact)

            logging.info(f"Saved checkpoint to {str(ckpt_path)}")
        except AttributeError:
            logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        wandb.finish()
        try:
            if metrics_f is not None:
                metrics_f.close()
        except Exception:
            pass

    simulation_app.close()
    if is_distributed:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
