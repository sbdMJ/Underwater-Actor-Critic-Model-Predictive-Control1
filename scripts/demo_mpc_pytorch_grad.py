"""
Isaac Sim 없이 locuslab/mpc.pytorch differentiable MPC gradient를 확인하는 데모.

실행(Isaac Python 권장):
  /home/mjkim/isaac410/python.sh scripts/demo_mpc_pytorch_grad.py
"""

import torch

from marinegym.controllers.mpc_pytorch_controller import MPCPyTorchController


def main():
    device = torch.device("cpu")
    nu = 6
    uav_params = {
        "mass": 12.0,
        "inertia": {"xx": 0.2, "yy": 0.2, "zz": 0.3},
        "hydro_coef": {
            "added_mass": [0.0] * 6,
            "linear_damping": [2.0] * 6,
            "quadratic_damping": [0.5] * 6,
        },
        "thruster_allocation": torch.eye(6, nu).numpy(),
        "volume": 0.012,
        "coBM": 0.02,
        "rho": 997.0,
        "g": 9.81,
        "mpc_q_pos": 50.0,
        "mpc_q_quat": 5.0,
        "mpc_q_vel": 2.0,
        "mpc_q_omega": 0.5,
        "mpc_r_u": 0.05,
    }

    ctrl = MPCPyTorchController(
        uav_params=uav_params,
        dt=0.05,
        horizon=6,
        batch_size=1,
        ilqr_iters=4,
        ilqr_reg=1e-3,
        terminal_weight_mult=10.0,
        backprop=True,
        max_thruster_force=40.0,
    ).to(device)
    ctrl.mpc.exit_unconverged = False
    ctrl.mpc.detach_unconverged = False
    ctrl.mpc.eps = 1e-3

    root_state = torch.zeros(1, 13, device=device)
    root_state[..., 3] = 1.0
    root_state.requires_grad_()

    target_pos = torch.tensor([[1.0, 0.0, 2.0]], device=device, requires_grad=True)
    target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    u = ctrl.compute(root_state, target_pos, target_quat=target_quat)
    loss = (u**2).sum()
    loss.backward()

    print("u:", u.detach().cpu().numpy().round(4))
    print("d(loss)/d(root_state)[pos]:", root_state.grad[0, 0:3].detach().cpu().numpy().round(6))
    print("d(loss)/d(target_pos):", target_pos.grad.detach().cpu().numpy().round(6))


if __name__ == "__main__":
    main()
