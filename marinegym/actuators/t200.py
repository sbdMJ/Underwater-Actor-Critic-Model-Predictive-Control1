import torch
import torch.nn as nn


class T200(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()

        # 기본 파라미터(초기값). from_module로 params 뽑아 쓰는 용도.
        self.force_constants = nn.Parameter(torch.as_tensor(rotor_config["force_constants"]).float())
        self.moment_constants = nn.Parameter(torch.as_tensor(rotor_config["moment_constants"]).float())
        self.max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"]).float()
        self.num_rotors = int(self.force_constants.numel())

        self.dt = float(dt)
        self.noise_scale = 0.002  # 지금은 미사용 (원하면 rpm_noise 추가 가능)

        # 상태(원래는 모듈 내부 상태였는데, vmap 때문에 외부에서 들고다니는 걸 권장)
        self.throttle = nn.Parameter(torch.zeros(self.num_rotors))
        self.rpm = nn.Parameter(torch.zeros(self.num_rotors))

        # 방향/시정수 등도 기본값을 파라미터로 들고는 있으나,
        # forward에서는 인자로 받은 값을 우선 사용하도록 작성.
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"]).float())
        self.tau_up = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        self.tau_down = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        self.time_constants = nn.Parameter(torch.as_tensor(rotor_config["time_constants"]).float())

        self.f = torch.square
        self.f_inv = torch.sqrt

        # 학습 안 할 거면 False 유지
        self.requires_grad_(False)

    def forward(
        self,
        cmds: torch.Tensor,
        throttle: torch.Tensor,
        rpm: torch.Tensor,
        *,
        tau_up: torch.Tensor,
        tau_down: torch.Tensor,
        directions: torch.Tensor,
        time_constants: torch.Tensor,
        force_constants: torch.Tensor,
    ):
        """
        cmds:      (..., num_rotors)  in [-1, 1]
        throttle:  (..., num_rotors)
        rpm:       (..., num_rotors)
        returns: thrusts, moments, throttle_new, rpm_new  with same batch shape
        """

        # dtype/device 정렬
        cmds = cmds.to(device=throttle.device, dtype=throttle.dtype)
        tau_up = tau_up.to(device=throttle.device, dtype=throttle.dtype)
        tau_down = tau_down.to(device=throttle.device, dtype=throttle.dtype)
        directions = directions.to(device=throttle.device, dtype=throttle.dtype)
        time_constants = time_constants.to(device=throttle.device, dtype=throttle.dtype)
        force_constants = force_constants.to(device=throttle.device, dtype=throttle.dtype)

        # 1) throttle 1차 지연
        target_throttle = torch.clamp(cmds, -1.0, 1.0)

        tau = torch.where(target_throttle > throttle, tau_up, tau_down)
        tau = torch.clamp(tau, 0.0, 1.0)

        throttle_new = throttle + tau * (target_throttle - throttle)

        # 2) throttle -> target rpm (너가 쓰던 piecewise 그대로)
        target_rpm = torch.where(
            throttle_new > 0.075,
            3.6599e3 * throttle_new + 3.4521e2,
            torch.where(
                throttle_new < -0.075,
                3.4944e3 * throttle_new - 4.3350e2,
                torch.zeros_like(throttle_new),
            ),
        )

        # 3) rpm 1차 지연
        alpha = torch.exp(-self.dt / time_constants)
        rpm_new = torch.clamp(alpha * rpm + (1.0 - alpha) * target_rpm, -3900.0, 3900.0)

        # 4) rpm -> thrust (너가 쓰던 식 유지)
        thrusts = force_constants / 4.4e-7 * 9.81 * torch.where(
            rpm_new > 0,
            4.7368e-07 * self.f(rpm_new) - 1.9275e-04 * rpm_new + 8.4452e-02,
            -3.8442e-07 * self.f(rpm_new) - 1.6186e-04 * rpm_new - 3.9139e-02,
        )

        # moment는 지금 0 처리 (원하면 KM 기반으로 넣어도 됨)
        moments = thrusts * (-directions) * 0.0

        return thrusts, moments, throttle_new, rpm_new
