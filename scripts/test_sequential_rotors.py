import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker
from torchrl.envs.transforms import TransformedEnv

from marinegym import init_simulation_app
from marinegym.learning import ALGOS  # noqa: F401  (Hydra ConfigStore 등록용)
from marinegym.utils.torch import quat_axis, quat_rotate_inverse


FILE_PATH = os.path.dirname(__file__)
os.environ.setdefault("MARINEGYM_ROOT", str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base=None, config_path=FILE_PATH, config_name="train")
def main(cfg):
    """
    BlueROVHeavy의 각 rotor를 순서대로 지정 시간 동안 단독 구동하는 스크립트.
    또는 지정한 rotor 인덱스들을 동시에 계속(또는 지정 시간) 구동할 수도 있습니다.

    예)
      python scripts/test_sequential_rotors.py task=Hover headless=false enable_livestream=false env.num_envs=1 task.drone_model.name=BlueROVHeavy rotor_test_duration_s=5.0 rotor_test_throttle=0.5 rotor_test_rest_s=1.0
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # PhysX Direct GPU API(eENABLE_DIRECT_GPU_API) 환경에서는 articulation에 대해 teleport(set_world_poses)가 금지되어
    # reset 단계에서 에러가 날 수 있습니다. (PxArticulationLink::setGlobalPose)
    # 이 스크립트는 반드시 reset을 수행하므로, articulation 테스트 시에는 CPU 파이프라인으로 강제합니다.
    try:
        drone_name = str(cfg.task.drone_model.name)
        is_articulation = cfg.task.drone_model.get("is_articulation", None)
        if is_articulation is None and drone_name in ("BlueROVHeavy", "bluerovheavy"):
            is_articulation = True
        if bool(is_articulation) and bool(cfg.sim.get("use_gpu_pipeline", False)):
            print(
                "[test_sequential_rotors] WARN: sim.use_gpu_pipeline=true with articulation can crash on reset "
                "(PhysX Direct GPU API). Forcing CPU pipeline for this run."
            )
            cfg.sim.use_gpu_pipeline = False
            cfg.sim.use_gpu = False
            cfg.sim.device = "cpu"
    except Exception:
        pass

    simulation_app = init_simulation_app(cfg)

    from marinegym.envs import IsaacEnv, register_tasks

    register_tasks()

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()

    seed = int(cfg.get("seed", 0))
    env.set_seed(seed)

    td = env.reset()

    drone = env.base_env.drone
    num_rotors = int(getattr(drone, "num_rotors", 0))
    if num_rotors <= 0:
        raise RuntimeError("num_rotors를 확인할 수 없습니다.")

    print_alloc = bool(cfg.get("rotor_test_print_alloc", False))
    debug_wrench = bool(cfg.get("rotor_test_debug_wrench", False))
    print_rotor_axes = bool(cfg.get("rotor_test_print_rotor_axes", False))
    debug_wz = bool(cfg.get("rotor_test_debug_wz", False))

    duration_s = float(cfg.get("rotor_test_duration_s", 5.0))
    rest_s = float(cfg.get("rotor_test_rest_s", 0.0))
    throttle = float(cfg.get("rotor_test_throttle", 0.5))
    rotor_test_mode = str(cfg.get("rotor_test_mode", "sequential")).lower()
    rotor_test_indices = cfg.get("rotor_test_indices", None)

    dt = float(cfg.sim.dt)
    steps_on = None if duration_s < 0.0 else max(1, int(round(duration_s / dt)))
    steps_rest = max(0, int(round(rest_s / dt)))

    if abs(throttle) > 1.0:
        raise ValueError(f"rotor_test_throttle must be in [-1, 1]. got {throttle}")
    if steps_on is None and rotor_test_mode not in ("hold", "simultaneous") and rotor_test_indices is None:
        raise ValueError(
            "rotor_test_duration_s < 0 means 'run forever'. "
            "Set rotor_test_mode=hold (or provide rotor_test_indices) to use infinite run."
        )

    if rotor_test_indices is not None:
        try:
            rotor_test_indices = [int(x) for x in list(rotor_test_indices)]
        except Exception as e:
            raise ValueError(f"rotor_test_indices must be a list of ints. got {rotor_test_indices}") from e
        for rotor_idx in rotor_test_indices:
            if rotor_idx < 0 or rotor_idx >= num_rotors:
                raise ValueError(
                    f"rotor_test_indices contains out-of-range index {rotor_idx}. "
                    f"valid range: 0..{num_rotors-1}"
                )

    print(
        "[test_sequential_rotors] model:",
        cfg.task.drone_model.name,
        "num_rotors:",
        num_rotors,
        "dt:",
        dt,
        "steps_on:",
        steps_on if steps_on is not None else "inf",
        "steps_rest:",
        steps_rest,
        "throttle:",
        throttle,
        "mode:",
        rotor_test_mode,
    )
    if num_rotors == 8:
        print("[test_sequential_rotors] rotor indices: 0..7 (총 8개)")
    else:
        print(f"[test_sequential_rotors] rotor indices: 0..{num_rotors-1}")
    if rotor_test_indices is not None:
        print("[test_sequential_rotors] selected indices:", rotor_test_indices)

    if print_rotor_axes:
        try:
            if getattr(drone, "rotors_view", None) is None:
                raise RuntimeError("rotors_view is None (virtual rotors). Enable direct rotor prim mode to print axes.")
            rotor_pos_w, rotor_rot_wxyz = drone.rotors_view.get_world_poses(clone=True)
            base_pos_w, base_rot_wxyz = drone.base_link.get_world_poses(clone=True)
            rotor_rot_wxyz0 = rotor_rot_wxyz[0, 0]  # (R, 4)
            base_rot_wxyz0 = base_rot_wxyz[0, 0].unsqueeze(0).expand(rotor_rot_wxyz0.shape[0], 4)

            axes = ["x", "y", "z"]
            print("[test_sequential_rotors] rotor local axes in BODY frame (env0/agent0):")
            for i in range(num_rotors):
                row = []
                for axis_idx, axis_name in enumerate(axes):
                    d_w = quat_axis(rotor_rot_wxyz0[i].unsqueeze(0), axis=axis_idx).squeeze(0)
                    d_b = quat_rotate_inverse(base_rot_wxyz0[i].unsqueeze(0), d_w.unsqueeze(0)).squeeze(0)
                    row.append(f"{axis_name}={['%.3f'%v for v in d_b.tolist()]}")
                print(f"  - rotor_{i}: " + " ".join(row))
        except Exception as e:
            print(f"[test_sequential_rotors] WARN: failed to print rotor axes. ({type(e).__name__}: {e})")

    # Optional: print thruster allocation (body-frame) for debugging.
    if print_alloc:
        try:
            B = getattr(drone, "thruster_allocation", None)
            if B is None:
                from marinegym.controllers.thruster_allocation import (
                    compute_thruster_allocation_matrix_from_drone,
                )

                thrust_axis = int(getattr(drone, "params", {}).get("thrust_axis", 0))
                B_np = compute_thruster_allocation_matrix_from_drone(drone, thrust_axis=thrust_axis)
                B = torch.tensor(B_np, dtype=torch.float32)
            print(f"[test_sequential_rotors] thruster_allocation B shape={tuple(B.shape)}")
            # Print a compact per-rotor summary (direction + moment arm columns).
            for i in range(num_rotors):
                col = B[:, i]
                f = col[:3].tolist()
                m = col[3:].tolist()
                print(f"  - rotor_{i}: d={['%.3f'%x for x in f]} m={['%.3f'%x for x in m]}")
        except Exception as e:
            print(f"[test_sequential_rotors] WARN: failed to print allocation. ({type(e).__name__}: {e})")

    def _debug_predicted_wrench(action_tensor: torch.Tensor, tag: str):
        if not debug_wrench:
            return
        try:
            B = getattr(drone, "thruster_allocation", None)
            if B is None:
                from marinegym.controllers.thruster_allocation import (
                    compute_thruster_allocation_matrix_from_drone,
                )
                thrust_axis = int(getattr(drone, "params", {}).get("thrust_axis", 0))
                B_np = compute_thruster_allocation_matrix_from_drone(drone, thrust_axis=thrust_axis)
                B = torch.tensor(B_np, dtype=torch.float32, device=drone.device)
            else:
                B = B.to(device=drone.device, dtype=torch.float32)

            # Predict thruster-only wrench (exclude hydro/buoyancy) for env0/agent0 using current rotor state.
            cmds = action_tensor.to(device=drone.device, dtype=torch.float32)
            thrusts, _, _, _ = drone.rotors(
                cmds,
                drone.throttle,
                drone.rpm,
                tau_up=drone.tau_up,
                tau_down=drone.tau_down,
                directions=drone.directions,
                time_constants=drone.TIME_CONSTANTS_0,
                force_constants=drone.FORCE_CONSTANTS_0,
            )
            u = thrusts[0, 0].to(dtype=torch.float32)
            tau = torch.einsum("ij,j->i", B, u)
            print(
                f"[test_sequential_rotors] predicted wrench ({tag}) "
                f"F={tau[:3].detach().cpu().tolist()} M={tau[3:].detach().cpu().tolist()}"
            )
        except Exception as e:
            print(f"[test_sequential_rotors] WARN: wrench debug failed. ({type(e).__name__}: {e})")

    def _debug_measured_wz(action_tensor: torch.Tensor, tag: str, steps: int = 30):
        if not debug_wz:
            return
        nonlocal td
        try:
            wzs = []
            for _ in range(int(steps)):
                td.set(("agents", "action"), action_tensor)
                td = env.step(td)["next"]
                if hasattr(env, "base_env"):
                    drone_local = env.base_env.drone
                else:
                    drone_local = drone
                vel_w = drone_local.get_velocities(clone=True)
                # vel_w: (..., 6) = [vx,vy,vz, wx,wy,wz] in world
                wz = float(vel_w[0, 0, 5].detach().cpu().item())
                wzs.append(wz)
            mean_wz = sum(wzs) / max(1, len(wzs))
            print(f"[test_sequential_rotors] measured wz_world mean over {steps} steps ({tag}): {mean_wz:.6f}")
        except Exception as e:
            print(f"[test_sequential_rotors] WARN: wz debug failed. ({type(e).__name__}: {e})")

    def _step_with_action(action_tensor: torch.Tensor):
        nonlocal td
        td.set(("agents", "action"), action_tensor)
        td = env.step(td)["next"]
        if not cfg.headless:
            env.render()

    try:
        if rotor_test_mode in ("hold", "simultaneous") or rotor_test_indices is not None:
            indices = rotor_test_indices if rotor_test_indices is not None else list(range(num_rotors))
            action = torch.zeros(*drone.shape, num_rotors, device=drone.device, dtype=torch.float32)
            for rotor_idx in indices:
                action[..., rotor_idx] = throttle

            _debug_predicted_wrench(action, f"hold indices={indices} throttle={throttle:.3f}")
            _debug_measured_wz(action, f"hold indices={indices} throttle={throttle:.3f}")

            if steps_on is None:
                print(f"[test_sequential_rotors] hold rotors {indices} at {throttle:.3f} until Ctrl-C")
                while True:
                    _step_with_action(action)
            else:
                print(f"[test_sequential_rotors] run rotors {indices} for {duration_s:.2f}s")
                for _ in range(steps_on):
                    _step_with_action(action)

        else:
            for rotor_idx in range(num_rotors):
                print(f"[test_sequential_rotors] run rotor {rotor_idx} for {duration_s:.2f}s")

                action = torch.zeros(*drone.shape, num_rotors, device=drone.device, dtype=torch.float32)
                action[..., rotor_idx] = throttle
                _debug_predicted_wrench(action, f"sequential rotor={rotor_idx} throttle={throttle:.3f}")
                _debug_measured_wz(action, f"sequential rotor={rotor_idx} throttle={throttle:.3f}")

                for _ in range(steps_on):
                    _step_with_action(action)

                if steps_rest > 0:
                    action.zero_()
                    for _ in range(steps_rest):
                        _step_with_action(action)
    except KeyboardInterrupt:
        pass
    finally:
        action = torch.zeros(*drone.shape, num_rotors, device=drone.device, dtype=torch.float32)
        for _ in range(max(1, int(round(1.0 / dt)))):
            _step_with_action(action)
        simulation_app.close()


if __name__ == "__main__":
    main()
