"""
환경 패키지 초기화.

중요: Isaac Sim의 `omni.*` 모듈은 `SimulationApp` 시작 전에는 import가 실패할 수 있어,
이 모듈에서는 task 환경(Hover 등)을 자동 import하지 않습니다.

task 환경을 사용하려면 `register_tasks()`를 `init_simulation_app()` 이후에 호출하세요.
"""

from .isaac_env import IsaacEnv


def register_tasks():
    # task 모듈 import 시 IsaacEnv.REGISTRY에 자동 등록됨
    from . import single  # noqa: F401


__all__ = ["IsaacEnv", "register_tasks"]
