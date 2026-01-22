import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv

from marinegym import init_simulation_app
from marinegym.learning import ALGOS
from marinegym.utils.torchrl.transforms import (
    FromDiscreteAction,
    FromMultiDiscreteAction,
    ravel_composite,
)

os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))

FILE_PATH = os.path.dirname(__file__)


def _guess_task_algo_from_wandb_debug(run_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    debug_path = run_dir / "logs" / "debug.log"
    if not debug_path.exists():
        return None, None
    try:
        text = debug_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None

    task_match = re.search(r"'task\\.name':\\s*'([^']+)'", text)
    algo_match = re.search(r"'algo\\.name':\\s*'([^']+)'", text)
    task_name = task_match.group(1) if task_match else None
    algo_name = algo_match.group(1) if algo_match else None
    return task_name, algo_name


def _resolve_run_paths(eval_cfg: dict) -> Tuple[Optional[Path], Optional[Path]]:
    run_dir_raw = eval_cfg.get("run_dir", "") or ""
    run_dir = Path(run_dir_raw).expanduser() if run_dir_raw else None

    ckpt_raw = eval_cfg.get("ckpt", "") or ""
    ckpt_path = Path(ckpt_raw).expanduser() if ckpt_raw else None

    if ckpt_path is None and run_dir is not None:
        candidates = [
            run_dir / "files" / "checkpoints" / "checkpoint_500224.pt",
            run_dir / "checkpoint_final.pt",
            run_dir / "files" / "files" / "checkpoint_final.pt",
        ]
        for cand in candidates:
            if cand.exists():
                ckpt_path = cand
                break

    return run_dir, ckpt_path


def _ensure_mpc_files(cfg, base_env, run_dir: Optional[Path]):
    algo_name = str(cfg.algo.name).lower()
    if algo_name not in ("ac_mpc", "ac_mpc_pypose"):
        return

    from marinegym.controllers.thruster_allocation import (
        compute_thruster_allocation_matrix_from_drone,
    )

    if not cfg.algo.get("mpc_param_yaml", None):
        if hasattr(base_env, "drone") and hasattr(base_env.drone, "param_path"):
            cfg.algo.mpc_param_yaml = base_env.drone.param_path

    if not cfg.algo.get("mpc_alloc_npz", None):
        candidates: List[Path] = []
        if run_dir is not None:
            candidates.extend(
                [
                    run_dir / "files" / algo_name / "thruster_allocation.npz",
                    run_dir / algo_name / "thruster_allocation.npz",
                ]
            )
        for cand in candidates:
            if cand.exists():
                cfg.algo.mpc_alloc_npz = str(cand)
                break

    if not cfg.algo.get("mpc_alloc_npz", None):
        tmp_dir = Path(getattr(cfg, "tmp_dir", "")) if getattr(cfg, "tmp_dir", "") else Path("/tmp")
        alloc_dir = tmp_dir / "marinegym_eval" / algo_name
        alloc_dir.mkdir(parents=True, exist_ok=True)
        alloc_path = alloc_dir / "thruster_allocation.npz"

        thrust_axis = int(cfg.task.get("thrust_axis", 0))
        B = compute_thruster_allocation_matrix_from_drone(
            base_env.drone, thrust_axis=thrust_axis
        )
        np.savez(alloc_path, B=B, quat_order="wxyz")
        cfg.algo.mpc_alloc_npz = str(alloc_path)

    # Pass sim-true dynamics params (if available) to match train.py behavior.
    try:
        cfg.algo.mpc_mass = float(base_env.drone.MASS_0.squeeze().item())
        cfg.algo.mpc_inertia = [float(x) for x in base_env.drone.INERTIA_0.squeeze().tolist()]
    except Exception:
        pass

    cfg.algo.mpc_dt = float(cfg.sim.dt)
    try:
        cfg.algo.obs_time_encoding_dim = (
            int(getattr(base_env, "time_encoding_dim", 0)) if bool(cfg.task.time_encoding) else 0
        )
    except Exception:
        cfg.algo.obs_time_encoding_dim = 0
    try:
        cfg.algo.obs_has_target_quat = bool(getattr(base_env, "include_target_quat_in_obs", False))
    except Exception:
        cfg.algo.obs_has_target_quat = False

    try:
        cfg.algo.mpc_nu = int(base_env.drone.action_spec.shape[-1])
    except Exception:
        pass

    cfg.algo.use_mpc_head = True


def _build_env(cfg):
    from marinegym.envs import IsaacEnv, register_tasks

    register_tasks()
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(
            ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        )

    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    return base_env, env


def _load_policy_state(policy, checkpoint_path: Path, device):
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(ckpt, dict):
        for key in ("policy", "policy_state_dict", "model", "state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like dict. type={type(ckpt)}")

    base_sd = policy.state_dict()
    merged = {}
    for key, value in ckpt.items():
        if key in base_sd:
            if (
                hasattr(value, "shape")
                and hasattr(base_sd[key], "shape")
                and value.shape != base_sd[key].shape
            ):
                continue
            merged[key] = value
    for key in base_sd.keys():
        if key not in merged:
            merged[key] = base_sd[key]
    policy.load_state_dict(merged, strict=True)
    policy.eval()


def _get_actor_core(policy):
    def _is_core(m) -> bool:
        return m is not None and hasattr(m, "feature_net") and hasattr(m, "cost_map")

    # Common torchrl path: policy.actor (ProbabilisticActor) -> .module (TensorDictModule) -> .module (actor core)
    actor = getattr(policy, "actor", None)
    if actor is not None:
        td_module = getattr(actor, "module", None)
        core = getattr(td_module, "module", None)
        if _is_core(core):
            return core

    # Fallback: search module tree (robust to wrapper/version differences).
    for m in policy.modules():
        if _is_core(m):
            return m

    # Help debugging in Isaac python where prints are visible.
    parts = []
    try:
        parts.append(f"policy={type(policy)}")
        parts.append(f"actor={type(actor)}")
        parts.append(f"actor.module={type(getattr(actor, 'module', None))}")
        parts.append(f"actor.module.module={type(getattr(getattr(actor, 'module', None), 'module', None))}")
    except Exception:
        pass
    detail = " ".join(parts)
    raise AttributeError(
        "Policy actor does not expose feature_net/cost_map (expected MPCActorPyPose). " + detail
    )


def _set_acados_stage_weights(solver, stage: int, W: np.ndarray) -> bool:
    # AcadosOcpSolver APIs vary slightly; try the common ones.
    try:
        solver.set(stage, "W", W)
        return True
    except Exception:
        pass
    try:
        solver.cost_set(stage, "W", W)
        return True
    except Exception:
        return False


def _set_acados_stage_yref(solver, stage: int, yref: np.ndarray) -> None:
    solver.set(stage, "yref", yref)


def _set_acados_terminal_yref(solver, stage_N: int, yref_e: np.ndarray) -> None:
    solver.set(stage_N, "yref", yref_e)


def _solve_mpc_with_learned_qp(
    controller,
    root_state: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    q_seq: torch.Tensor,
    p_seq: torch.Tensor,
    nx_pred: int,
    warn_state: dict,
) -> torch.Tensor:
    device = root_state.device
    dtype = root_state.dtype

    solver = controller.solver
    nx = int(controller.nx)
    nu = int(controller.nu)
    ny = int(controller.ny)
    ny_e = int(controller.ny_e)
    N = int(controller.N)
    maxF = float(getattr(controller, "max_force_per_thruster", 40.0))

    # q_seq/p_seq: (B, Nq, nx_pred+nu)
    B = int(root_state.shape[0])
    Nq = int(q_seq.shape[1])
    n_cost = int(q_seq.shape[2])
    if n_cost != nx_pred + nu:
        raise ValueError(f"Expected cost dim nx_pred+nu={nx_pred+nu}, got {n_cost}")
    if nx_pred < 12:
        raise ValueError(f"nx_pred must be >= 12 for hover_obs mapping, got nx_pred={nx_pred}")
    if p_seq.shape != q_seq.shape:
        raise ValueError(f"q_seq and p_seq must match shape, got q_seq={tuple(q_seq.shape)} p_seq={tuple(p_seq.shape)}")
    if nx != 13:
        raise ValueError(f"Expected MPCController state dim nx=13, got nx={nx}")
    if ny != nx + nu:
        raise ValueError(f"Expected NONLINEAR_LS y=[x,u] so ny=nx+nu, got ny={ny}")
    if ny_e != nx:
        raise ValueError(f"Expected terminal yref_e dim ny_e=nx, got ny_e={ny_e}")

    cmds = torch.zeros((B, nu), device=device, dtype=dtype)

    # Move costs to CPU once (acados setter expects numpy).
    q_np = q_seq.detach().cpu().numpy().astype(np.float64, copy=False)
    p_np = p_seq.detach().cpu().numpy().astype(np.float64, copy=False)

    for i in range(B):
        x0 = root_state[i].detach().cpu().numpy().astype(np.float64, copy=False)
        tp = target_pos[i].detach().cpu().numpy().astype(np.float64, copy=False)
        tq = target_quat[i].detach().cpu().numpy().astype(np.float64, copy=False)

        # x(0) = x0
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        # Base reference: go to target pose, damp velocities.
        x_ref_base = x0.copy()
        x_ref_base[0:3] = tp[0:3]
        x_ref_base[7:13] = 0.0
        tq_norm = np.linalg.norm(tq)
        if tq_norm > 0:
            tq = tq / tq_norm
        if np.dot(tq, x0[3:7]) < 0:
            tq = -tq
        x_ref_base[3:7] = tq

        for k in range(N):
            idx = min(k, Nq - 1)
            qk = q_np[i, idx]
            pk = p_np[i, idx]

            qx_pred = qk[:nx_pred]
            qu_cmd = qk[nx_pred:]
            px_pred = pk[:nx_pred]
            pu_cmd = pk[nx_pred:]

            # Use only the hover_obs layout: rpos(3), rpy(3), v_b(3), w_b(3).
            qx_use = qx_pred[:12]
            px_use = px_pred[:12]

            qx_pred = np.maximum(qx_pred, 1e-6)
            qu_cmd = np.maximum(qu_cmd, 1e-9)

            # Map learned (rpos, rpy, v_b, w_b) weights to acados state (pos, quat, v_b, w_b).
            q_pos = qx_use[0:3]
            q_rpy = qx_use[3:6]
            q_vel = qx_use[6:9]
            q_omega = qx_use[9:12]
            q_quat = float(np.mean(q_rpy))

            qx = np.zeros(nx, dtype=np.float64)
            qx[0:3] = q_pos
            qx[3:7] = q_quat
            qx[7:10] = q_vel
            qx[10:13] = q_omega

            # Convert action weights from normalized cmd u in [-1,1] to force u in [-maxF,maxF].
            qu_force = qu_cmd / (maxF * maxF)
            W = np.zeros((ny, ny), dtype=np.float64)
            W[np.arange(nx), np.arange(nx)] = 2.0 * qx
            W[nx + np.arange(nu), nx + np.arange(nu)] = 2.0 * qu_force

            okW = _set_acados_stage_weights(solver, k, W)
            if (not okW) and (not warn_state.get("warned_W", False)):
                print("[eval_acmpc_pypose2] WARNING: acados solver does not allow setting stage W; using fixed weights.")
                warn_state["warned_W"] = True

            # Linear term -> shift references: yref = -p/(2q) (in the cost coordinates).
            x_ref = x_ref_base.copy()

            # rpos = target_pos - pos  => pos_ref = target_pos - rpos_ref
            rpos_ref = -px_use[0:3] / (2.0 * np.maximum(qx_use[0:3], 1e-6))
            x_ref[0:3] = tp[0:3] - rpos_ref

            # v_b and w_b are directly in acados state.
            v_ref = -px_use[6:9] / (2.0 * np.maximum(qx_use[6:9], 1e-6))
            w_ref = -px_use[9:12] / (2.0 * np.maximum(qx_use[9:12], 1e-6))
            x_ref[7:10] = v_ref
            x_ref[10:13] = w_ref

            # u_ref in force units.
            u_ref_cmd = -pu_cmd / (2.0 * qu_cmd)
            u_ref_force = np.clip(u_ref_cmd * maxF, -maxF, maxF)

            yref = np.zeros(ny, dtype=np.float64)
            yref[0:nx] = x_ref
            yref[nx:] = u_ref_force
            _set_acados_stage_yref(solver, k, yref)

        # Terminal reference (use last stage mapping).
        idx_e = min(N - 1, Nq - 1)
        qk = q_np[i, idx_e]
        pk = p_np[i, idx_e]
        qx_pred = qk[:nx_pred]
        px_pred = pk[:nx_pred]
        qx_use = np.maximum(qx_pred[:12], 1e-6)
        px_use = px_pred[:12]
        x_ref_e = x_ref_base.copy()
        rpos_ref = -px_use[0:3] / (2.0 * qx_use[0:3])
        x_ref_e[0:3] = tp[0:3] - rpos_ref
        v_ref = -px_use[6:9] / (2.0 * qx_use[6:9])
        w_ref = -px_use[9:12] / (2.0 * qx_use[9:12])
        x_ref_e[7:10] = v_ref
        x_ref_e[10:13] = w_ref
        _set_acados_terminal_yref(solver, N, x_ref_e[:ny_e])

        status = solver.solve()
        if status != 0:
            if warn_state.get("warned_solve", 0) < 10:
                print(f"[eval_acmpc_pypose2] acados solve failed: status={status}")
                warn_state["warned_solve"] = int(warn_state.get("warned_solve", 0)) + 1
            u_opt = np.zeros(nu, dtype=np.float64)
        else:
            u_opt = solver.get(0, "u")
        u_cmd = np.clip(u_opt / maxF, -1.0, 1.0)
        cmds[i].copy_(torch.as_tensor(u_cmd, device=device, dtype=dtype))

    return cmds


def _attach_learned_qp_controller(base_env, actor_core, cfg):
    if not hasattr(base_env, "controller"):
        raise AttributeError("Base env has no .controller (expected HoverMPC).")

    if base_env.controller.__class__.__name__ != "MPCController":
        raise TypeError(f"Expected base_env.controller to be MPCController, got {type(base_env.controller)}")

    controller = base_env.controller
    nx_pred = int(cfg.algo.get("mpc_nx", 12))
    warn_state: dict = {}

    def _pre_sim_step(self, tensordict):
        # Keep state up-to-date before building MPC initial condition.
        self.drone.get_state()

        obs = tensordict.get(("agents", "observation"))
        if obs is None:
            raise KeyError("Missing ('agents','observation') in tensordict.")
        obs2 = obs.reshape(-1, obs.shape[-1])

        feat = actor_core.feature_net(obs2)
        q_seq, p_seq = actor_core.cost_map(feat)

        root_state = torch.cat([self.drone.pos, self.drone.rot, self.drone.vel_b], dim=-1).squeeze(1)
        target_pos = self.target_pos.squeeze(1)
        target_quat = self.target_rot.squeeze(1)

        cmds = _solve_mpc_with_learned_qp(
            controller=controller,
            root_state=root_state,
            target_pos=target_pos,
            target_quat=target_quat,
            q_seq=q_seq,
            p_seq=p_seq,
            nx_pred=nx_pred,
            warn_state=warn_state,
        ).unsqueeze(1)

        tensordict.set(("agents", "action"), cmds)
        self.effort = torch.abs(self.drone.apply_action(cmds))

    base_env._pre_sim_step = _pre_sim_step.__get__(base_env, base_env.__class__)


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    Evaluate `ac_mpc_pypose`-learned Q/p by driving the HoverMPC (acados) controller weights/references.

    Example:
      ~/isaac410/python.sh scripts/evaluate_acmpc_pypose2.py \\
        +eval.run_dir=/home/mjkim/MarineGym/wandb/offline-run-*/ \\
        task=Hover_MPC algo=ac_mpc_pypose headless=true enable_livestream=false env.num_envs=1
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    eval_cfg = cfg.get("eval", {}) or {}
    run_dir, ckpt_path = _resolve_run_paths(eval_cfg)
    if ckpt_path is None:
        raise ValueError(
            "Missing checkpoint. Set +eval.ckpt=/path/to/checkpoint.pt or +eval.run_dir=/path/to/wandb/offline-run-*"
        )

    if run_dir is not None:
        guessed_task, guessed_algo = _guess_task_algo_from_wandb_debug(run_dir)
        if guessed_task and str(cfg.task.name) != guessed_task:
            print(f"[wandb] run task={guessed_task} (cfg.task.name={cfg.task.name})")
        if guessed_algo and str(cfg.algo.name) != guessed_algo:
            print(f"[wandb] run algo={guessed_algo} (cfg.algo.name={cfg.algo.name})")

    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    base_env, env = _build_env(cfg)
    _ensure_mpc_files(cfg, base_env, run_dir)

    algo_name = str(cfg.algo.name).lower()
    policy = ALGOS[algo_name](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )
    _load_policy_state(policy, ckpt_path, base_env.device)

    actor_core = _get_actor_core(policy).to(base_env.device)
    actor_core.eval()

    _attach_learned_qp_controller(base_env, actor_core, cfg)

    seed = int(eval_cfg.get("seed", cfg.get("seed", 0)))
    env.set_seed(seed)

    td = env.reset()
    dummy_action = env.action_spec.zero()
    steps = int(eval_cfg.get("steps", cfg.get("mpc_test_steps", 5000)))

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

        if not bool(cfg.headless):
            env.render()

    simulation_app.close()


if __name__ == "__main__":
    main()
