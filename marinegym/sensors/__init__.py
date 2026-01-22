# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Sensors package.

Note: Isaac Sim Python modules (e.g. ``omni.*``) are typically available only after
``isaacsim.SimulationApp`` is initialized. Import concrete sensors from submodules
after calling ``marinegym.init_simulation_app()``.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["Camera", "Sonar", "SonarCfg"]


def __getattr__(name: str) -> Any:
    if name == "Camera":
        return importlib.import_module(".camera", __name__).Camera
    if name in ("Sonar", "SonarCfg"):
        mod = importlib.import_module(".sonar", __name__)
        return getattr(mod, name)
    raise AttributeError(name)

