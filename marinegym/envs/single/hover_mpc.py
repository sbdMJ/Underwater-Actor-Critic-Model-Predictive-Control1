import torch
import numpy as np

from marinegym.envs.single.hover import Hover


class HoverMPC(Hover):
    """
    Hover 태스크를 MPCController로 제어하는 변형.

    - RL action은 무시하고(있어도 덮어씀), 매 step에서 MPC로 스러스터 커맨드를 계산합니다.
    - 기존 Hover의 관측/보상 정의는 그대로 재사용합니다.
    """

    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self._use_internal_mpc = bool(self.cfg.task.get("use_internal_mpc", True))
        if self._use_internal_mpc:
            self._init_controller()

    def _init_controller(self):
        from marinegym.controllers.mpc_controller import MPCController
        from marinegym.controllers.thruster_allocation import (
            compute_thruster_allocation_matrix_from_drone,
        )

        mass = float(self.drone.MASS_0.squeeze().item())
        inertia_xx, inertia_yy, inertia_zz = self.drone.INERTIA_0.squeeze().tolist()

        thrust_axis = int(self.cfg.task.get("thrust_axis", 0))
        B_src = "computed"
        if getattr(self.drone, "thruster_allocation", None) is not None:
            B = np.asarray(self.drone.thruster_allocation.detach().cpu().numpy(), dtype=np.float64)
            B_src = "drone.thruster_allocation"
        elif "thruster_allocation" in self.drone.params:
            B = np.asarray(self.drone.params["thruster_allocation"], dtype=np.float64)
            B_src = "drone.params.thruster_allocation"
        else:
            B = compute_thruster_allocation_matrix_from_drone(self.drone, thrust_axis=thrust_axis)

        # --- Model parameter alignment ---
        # rho/volume: simulation may run in neutral-buoyancy mode (buoyancy cancels weight exactly).
        # Align MPC buoyancy with the sim by setting volume = mass/rho in that case.
        rho = float(self.drone.params.get("rho", self.drone.params.get("water_density", 997.0)))
        volume = float(self.drone.params.get("volume", 0.0))
        buoy_mode = str(self.drone.params.get("buoyancy_mode", "volume")).lower()
        if buoy_mode in ("neutral", "neutral_mass", "match_weight") and rho > 0.0:
            volume = float(mass / rho)

        # If sim hydrodynamics are disabled, keep MPC model consistent by zeroing hydro terms.
        hydro_coef = self.drone.params["hydro_coef"]
        if bool(self.drone.params.get("disable_hydrodynamics", False)):
            hydro_coef = {
                "added_mass": [0.0] * 6,
                "linear_damping": [0.0] * 6,
                "quadratic_damping": [0.0] * 6,
            }

        # MPC 모델은 coBM이 body Z축 방향이라고 가정합니다. 환경(시뮬)에서 cobm_axis가 다른 경우,
        # MPC 내부 모델과 토크 방향이 어긋날 수 있어 기본값은 0으로 둡니다(override 가능).
        coBM = float(self.drone.params.get("coBM", 0.0))
        try:
            axis = getattr(self.drone, "_cobm_axis_unit", None)
            if axis is not None:
                axis = axis.detach().cpu().numpy().reshape(-1)
                if axis.shape[0] == 3 and not (abs(float(axis[0])) < 1e-3 and abs(float(axis[1])) < 1e-3 and float(axis[2]) > 0.999):
                    coBM = 0.0
        except Exception:
            pass

        uav_params = {
            "name": self.cfg.task.drone_model.name,
            "mass": mass,
            "inertia": {"xx": inertia_xx, "yy": inertia_yy, "zz": inertia_zz},
            "hydro_coef": hydro_coef,
            "thruster_allocation": B,
            "volume": volume,
            "coBM": coBM,
            "rho": rho,
            "mpc_q_pos": float(self.cfg.task.get("mpc_q_pos", 50.0)),
            "mpc_q_quat": float(self.cfg.task.get("mpc_q_quat", 5.0)),
            "mpc_q_vel": float(self.cfg.task.get("mpc_q_vel", 2.0)),
            "mpc_q_omega": float(self.cfg.task.get("mpc_q_omega", 0.5)),
            "mpc_q_roll": float(self.cfg.task.get("mpc_q_roll", 0.0)),
            "mpc_q_pitch": float(self.cfg.task.get("mpc_q_pitch", 0.0)),
            "mpc_r_u": float(self.cfg.task.get("mpc_r_u", 0.01)),
            "max_thruster_force": float(self.cfg.task.get("max_thruster_force", 40.0)),
        }

        horizon = int(self.cfg.task.get("mpc_horizon", 20))
        dt = float(self.cfg.sim.dt)
        self.controller = MPCController(g=9.81, uav_params=uav_params, dt=dt, N=horizon)

        if bool(self.cfg.task.get("mpc_debug_print_B", False)):
            np.set_printoptions(precision=3, suppress=True)
            print("[HoverMPC] thrust_axis:", thrust_axis)
            print("[HoverMPC] B source:", B_src)
            print("[HoverMPC] B (6 x n):\n", B)

    def _pre_sim_step(self, tensordict):
        if not getattr(self, "_use_internal_mpc", True):
            return super()._pre_sim_step(tensordict)

        # 최신 상태 업데이트(자세/속도 포함)
        self.drone.get_state()

        # MPC는 body-frame v/w를 가정하므로 vel_b 사용
        root_state = torch.cat(
            [self.drone.pos, self.drone.rot, self.drone.vel_b], dim=-1
        ).squeeze(1)  # (num_envs, 13)
        target_pos = self.target_pos.squeeze(1)
        target_quat = self.target_rot.squeeze(1)

        cmds = self.controller.compute(root_state, target_pos, target_quat=target_quat).unsqueeze(1)  # (num_envs, 1, 6)
        tensordict.set(("agents", "action"), cmds)

        self.effort = torch.abs(self.drone.apply_action(cmds))
