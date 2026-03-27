# Copyright 2026 Moore Threads
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Optional

import axinfra as ax

from . import types as mjmtp
from .cached_array import get_cached_array


def mul_m(
  m: mjmtp.Model,
  d: mjmtp.Data,
  res: ax.array2d(dtype=float), # type: ignore
  vec: ax.array2d(dtype=float), # type: ignore
  skip: Optional[ax.array] = None,
  M: Optional[ax.array] = None,
):
  """Multiply vectors by inertia matrix; optionally skip per world.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    res: Result: qM @ vec.
    vec: Input vector to multiply by qM.
    skip: Per-world bitmask to skip computing output.
    M: Input matrix: M @ vec.
  """
  check_skip = skip is not None
  skip = skip or get_cached_array('mul_m_''skip', 0, ax.bool)

  if M is None:
    M = d.qM

  if m.opt.is_sparse:
    ax.launch(
      "mul_m_sparse_diag___mul_m_sparse_diag",
      dim=(d.nworld, m.nv),
      inputs=[m.dof_Madr, M, vec, skip, check_skip],
      outputs=[res],
    )

    ax.launch(
      "mul_m_sparse_ij___mul_m_sparse_ij",
      dim=(d.nworld, m.qM_madr_ij.size),
      inputs=[m.qM_mulm_i, m.qM_mulm_j, m.qM_madr_ij, M, vec, skip, check_skip],
      outputs=[res],
    )

  else:
    for tile in m.qM_tiles:
      ax.launch(
        "mul_m_dense___mul_m_dense",
        dim=(d.nworld, tile.adr.size, tile.size),
        inputs=[
          M,
          tile.adr,
          vec,
          skip,
          check_skip,
          tile.size,
        ],
        outputs=[res],
        block_dim=tile.size,
      )


def apply_ft(m: mjmtp.Model, d: mjmtp.Data, ft: ax.array2d(dtype=ax.spatial_vector), qfrc: ax.array2d(dtype=float), flg_add: bool): # type: ignore
  ax.launch(
    kernel="_apply_ft",
    dim=(d.nworld, m.nv),
    inputs=[m.nbody, m.body_parentid, m.body_rootid, m.dof_bodyid, d.xipos, d.subtree_com, d.cdof, ft, flg_add],
    outputs=[qfrc],
  )


def xfrc_accumulate(m: mjmtp.Model, d: mjmtp.Data, qfrc: ax.array2d(dtype=float)): # type: ignore
  """Map applied forces at each body via Jacobians to dof space and accumulate.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    qfrc: Total applied force mapped to dof space.
  """
  apply_ft(m, d, d.xfrc_applied, qfrc, True)
