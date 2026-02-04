import numpy as np
import torch

from marinegym.envs.single.hover import Hover


class HoverMPCPyTorch(Hover):
    """
    Hover 태스크를 mpc.pytorch 스타일 differentiable MPC(iLQR)로 제어하는 변형.

    - RL action은 무시하고(있어도 덮어씀), 매 step에서 MPC로 스러스터 커맨드를 계산합니다.
    - 기존 Hover의 관측/보상 정의는 그대로 재사용합니다.
    """

    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self._init_controller()

    def _init_controller(self):
        from marinegym.controllers.mpc_pytorch_controller import MPCPyTorchController
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

        rho = float(self.drone.params.get("rho", self.drone.params.get("water_density", 997.0)))
        volume = float(self.drone.params.get("volume", 0.0))
        buoy_mode = str(self.drone.params.get("buoyancy_mode", "volume")).lower()
        if buoy_mode in ("neutral", "neutral_mass", "match_weight") and rho > 0.0:
            volume = float(mass / rho)

        hydro_coef = self.drone.params["hydro_coef"]
        if bool(self.drone.params.get("disable_hydrodynamics", False)):
            hydro_coef = {
                "added_mass": [0.0] * 6,
                "linear_damping": [0.0] * 6,
                "quadratic_damping": [0.0] * 6,
            }

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
            "g": 9.81,
            "mpc_q_pos": float(self.cfg.task.get("mpc_q_pos", 50.0)),
            "mpc_q_quat": float(self.cfg.task.get("mpc_q_quat", 5.0)),
            "mpc_q_vel": float(self.cfg.task.get("mpc_q_vel", 2.0)),
            "mpc_q_omega": float(self.cfg.task.get("mpc_q_omega", 0.5)),
            "mpc_r_u": float(self.cfg.task.get("mpc_r_u", 0.02)),
        }

        horizon = int(self.cfg.task.get("mpc_horizon", 15))
        ilqr_iters = int(self.cfg.task.get("mpc_ilqr_iters", 6))
        ilqr_reg = float(self.cfg.task.get("mpc_ilqr_reg", 1e-3))
        terminal_mult = float(self.cfg.task.get("mpc_terminal_mult", 10.0))

        self.controller = MPCPyTorchController(
            uav_params=uav_params,
            dt=float(self.cfg.sim.dt),
            horizon=horizon,
            batch_size=int(self.cfg.env.num_envs),
            ilqr_iters=ilqr_iters,
            ilqr_reg=ilqr_reg,
            terminal_weight_mult=terminal_mult,
            backprop=bool(self.cfg.task.get("mpc_backprop", False)),
            eps=float(self.cfg.task.get("mpc_eps", 1e-3)),
            exit_unconverged=bool(self.cfg.task.get("mpc_exit_unconverged", False)),
            detach_unconverged=(
                None
                if self.cfg.task.get("mpc_detach_unconverged", None) is None
                else bool(self.cfg.task.get("mpc_detach_unconverged"))
            ),
            max_thruster_force=float(self.cfg.task.get("max_thruster_force", 40.0)),
        )
        self.controller.to(self.device)

        if bool(self.cfg.task.get("mpc_debug_print_B", False)):
            np.set_printoptions(precision=3, suppress=True)
            print("[HoverMPCPyTorch] thrust_axis:", thrust_axis)
            print("[HoverMPCPyTorch] B source:", B_src)
            print("[HoverMPCPyTorch] B (6 x n):\n", B)

    def _pre_sim_step(self, tensordict):
        self.drone.get_state()

        root_state = torch.cat([self.drone.pos, self.drone.rot, self.drone.vel_b], dim=-1).squeeze(1)  # (N, 13)
        target_pos = self.target_pos.squeeze(1)
        target_quat = self.target_rot.squeeze(1)

        cmds = self.controller.compute(root_state, target_pos, target_quat=target_quat).unsqueeze(1)
        tensordict.set(("agents", "action"), cmds)
        self.effort = torch.abs(self.drone.apply_action(cmds))
