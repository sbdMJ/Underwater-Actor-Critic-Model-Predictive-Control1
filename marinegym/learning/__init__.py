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


from .ppo import *
from .ppo_mpc import *
from .acmpc_pypose import *
from .ppo_pypose_mpc_qrdiag import *
from .ppo_pypose_mpc_qrconst import *
from .ppo_pypose_mpc_qrdiag_tv import *
from .ppo_acados_pypose_mpc_qrdiag_tv import *
from .ppo_acados_hover_mpc_qrdiag import *
from .ppo_pypose_cylinder_mpc_werr_wu_tv import *
ALGOS = {
    "ppo": PPOPolicy,
    "ac_mpc": ACMPCPolicy,
    "ac_mpc_pypose": ACMPCPyPosePolicy,
    "ppo_pypose_mpc_qrdiag": PPOPyposeMPCQRDiagPolicy,
    "ppo_pypose_mpc_qrconst": PPOPyposeMPCQRConstPolicy,
    "ppo_pypose_mpc_qrdiag_tv": PPOPyposeMPCQRDiagTVPolicy,
    "ppo_pypose_cylinder_mpc_werr_wu_tv": PPOPyposeCylinderMPCWErrWUTVPolicy,
    "ppo_acados_pypose_mpc_qrdiag_tv": PPOAcadosPyPoseMPCQRDiagTVPolicy,
    "ppo_acados_hover_mpc_qrdiag": PPOAcadosHoverMPCQRDiagPolicy,
}
