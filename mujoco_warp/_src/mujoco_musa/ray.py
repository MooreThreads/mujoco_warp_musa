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

from typing import Optional, Tuple

import axinfra as ax
from axinfra import vec6

from . import types as mjmtp


def rays(
  m: mjmtp.Model,
  d: mjmtp.Data,
  pnt: ax.array2d(dtype=ax.vec3), # type: ignore
  vec: ax.array2d(dtype=ax.vec3), # type: ignore
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: ax.array(dtype=int), # type: ignore
  dist: ax.array2d(dtype=ax.vec3), # type: ignore
  geomid: ax.array2d(dtype=int), # type: ignore
):
  ax.launch(
    "_ray",
    dim=(d.nworld, pnt.shape[1], m.block_dim.ray),
    inputs=[
      m.ngeom,
      m.nmeshface,
      m.body_weldid,
      m.geom_type,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_matid,
      m.geom_group,
      m.geom_size,
      m.geom_rgba,
      m.mesh_vertadr,
      m.mesh_faceadr,
      m.mesh_vert,
      m.mesh_face,
      m.hfield_size,
      m.hfield_nrow,
      m.hfield_ncol,
      m.hfield_adr,
      m.hfield_data,
      m.mat_rgba,
      d.geom_xpos,
      d.geom_xmat,
      pnt,
      vec,
      geomgroup,
      flg_static,
      bodyexclude,
    ],
    outputs=[
      dist,
      geomid,
    ],
    block_dim=m.block_dim.ray,
  )


def ray(
  m: mjmtp.Model,
  d: mjmtp.Data,
  pnt: ax.array2d(dtype=ax.vec3), # type: ignore
  vec: ax.array2d(dtype=ax.vec3), # type: ignore
  geomgroup: Optional[vec6] = None,
  flg_static: bool = True,
  bodyexclude: int = -1,
) -> Tuple[ax.array, ax.array]:
  """Returns the distance at which rays intersect with primitive geoms.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    pnt: Ray origin points.
    vec: Ray directions.
    geomgroup: Group inclusion/exclusion mask. If all are ax.inf, ignore.
    flg_static: If True, allows rays to intersect with static geoms.
    bodyexclude: Ignore geoms on specified body id (-1 to disable).

  Returns:
    Distances from ray origins to geom surfaces and IDs of intersected geoms (-1 if none).
  """
  assert pnt.shape[0] == 1
  assert pnt.shape[0] == vec.shape[0]

  if geomgroup is None:
    geomgroup = vec6(-1, -1, -1, -1, -1, -1)

  ray_bodyexclude = ax.empty(1, dtype=int)
  ray_bodyexclude.fill_(bodyexclude)
  ray_dist = ax.empty((d.nworld, 1), dtype=float)
  ray_geomid = ax.empty((d.nworld, 1), dtype=int)

  # geomgroup.
  rays(m, d, pnt, vec, geomgroup, flg_static, ray_bodyexclude, ray_dist, ray_geomid)

  return ray_dist, ray_geomid
