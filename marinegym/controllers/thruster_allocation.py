import numpy as np
import torch

from marinegym.utils.torch import quat_axis, quat_rotate_inverse


def compute_thruster_allocation_matrix_from_views(
    *,
    base_pos_w: torch.Tensor,
    base_rot_wxyz: torch.Tensor,
    rotor_pos_w: torch.Tensor,
    rotor_rot_wxyz: torch.Tensor,
    thrust_axis: int = 0,
) -> np.ndarray:
    """
    Isaac Sim에서 읽은 base/rotor의 world pose로부터, body frame wrench 할당행렬 B를 계산한다.

    정의:
      - u_i: i번째 thruster의 "추력 크기"(스칼라, N)
      - d_i: body frame에서의 추력 방향(단위벡터)
      - r_i: body frame에서의 thruster 위치벡터
      - wrench tau = [F; M] = sum_i [d_i; r_i x d_i] * u_i
      - 따라서 B = [[d_1 ... d_n]; [r_1 x d_1 ... r_n x d_n]] (shape: 6 x n)

    반력토크(프로펠러 회전에 의한 yaw moment)는 포함하지 않는다.
    """
    if base_pos_w.shape != (3,) or base_rot_wxyz.shape != (4,):
        raise ValueError("base_pos_w는 (3,), base_rot_wxyz는 (4,) 이어야 합니다.")
    if rotor_pos_w.ndim != 2 or rotor_pos_w.shape[-1] != 3:
        raise ValueError("rotor_pos_w는 (n, 3) 이어야 합니다.")
    if rotor_rot_wxyz.ndim != 2 or rotor_rot_wxyz.shape[-1] != 4:
        raise ValueError("rotor_rot_wxyz는 (n, 4) 이어야 합니다.")
    if rotor_pos_w.shape[0] != rotor_rot_wxyz.shape[0]:
        raise ValueError("rotor_pos_w와 rotor_rot_wxyz의 n이 다릅니다.")

    n = rotor_pos_w.shape[0]

    # r_w: rotor position relative to base origin, expressed in world frame
    r_w = rotor_pos_w - base_pos_w.unsqueeze(0)

    # body-frame vectors
    base_rot = base_rot_wxyz.unsqueeze(0).expand(n, 4)
    r_b = quat_rotate_inverse(base_rot, r_w)

    # thrust direction: rotor local axis (default: +X) expressed in world, then to body
    d_w = quat_axis(rotor_rot_wxyz, axis=thrust_axis)
    d_b = quat_rotate_inverse(base_rot, d_w)
    d_b = torch.nn.functional.normalize(d_b, dim=-1)

    m_b = torch.cross(r_b, d_b, dim=-1)

    B = torch.cat([d_b, m_b], dim=-1).T  # (6, n)
    return B.detach().cpu().numpy().astype(np.float64, copy=False)


def compute_thruster_allocation_matrix_from_drone(
    drone,
    *,
    env_index: int = 0,
    agent_index: int = 0,
    thrust_axis: int = 0,
) -> np.ndarray:
    """
    `UnderwaterVehicle` 인스턴스(초기화 완료)로부터 B를 계산한다.
    """
    # rotor poses (E, A, R, 3/4)
    rotor_pos, rotor_rot = drone.rotors_view.get_world_poses(clone=True)

    # 기준 프레임은 실제로 힘/토크를 받는 base_link가 더 일관적임
    if hasattr(drone, "base_link"):
        base_pos, base_rot = drone.base_link.get_world_poses(clone=True)
    else:
        base_pos, base_rot = drone.get_world_poses(clone=True)

    rotor_pos_w = rotor_pos[env_index, agent_index]  # (R, 3)
    rotor_rot_wxyz = rotor_rot[env_index, agent_index]  # (R, 4)
    base_pos_w = base_pos[env_index, agent_index]  # (3,)
    base_rot_wxyz = base_rot[env_index, agent_index]  # (4,)

    return compute_thruster_allocation_matrix_from_views(
        base_pos_w=base_pos_w,
        base_rot_wxyz=base_rot_wxyz,
        rotor_pos_w=rotor_pos_w,
        rotor_rot_wxyz=rotor_rot_wxyz,
        thrust_axis=thrust_axis,
    )
