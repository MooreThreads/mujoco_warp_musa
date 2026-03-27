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

import warp as wp

from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.warp_util import copy_wp_array_batch_attrlist

from ..ray import *


def ray_copydata(
  m: Model,
  d: Data,
  pnt: wp.array2d(dtype=wp.vec3),
  vec: wp.array2d(dtype=wp.vec3),
  geomgroup: Optional[vec6] = None,
  flg_static: bool = True,
  bodyexclude: int = -1,
) -> Tuple[wp.array, wp.array]:
  """Returns the distance at which rays intersect with primitive geoms.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    pnt: Ray origin points.
    vec: Ray directions.
    geomgroup: Group inclusion/exclusion mask. If all are wp.inf, ignore.
    flg_static: If True, allows rays to intersect with static geoms.
    bodyexclude: Ignore geoms on specified body id (-1 to disable).

  Returns:
    Distances from ray origins to geom surfaces and IDs of intersected geoms (-1 if none).
  """
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "geom_xpos",
    "geom_xmat",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  pnt_ = ax.from_wp_array(pnt)
  vec_ = ax.from_wp_array(vec)
  geomgroup_ = vec6(list(geomgroup)) if geomgroup is not None else None

  ray_dist_, ray_geomid_ = ray(m_, d_, pnt_, vec_, geomgroup_, flg_static, bodyexclude)

  ray_dist = wp.empty((d.nworld, 1), dtype=float)
  ray_geomid = wp.empty((d.nworld, 1), dtype=int)
  ax.copy_wp_array(ray_dist, ray_dist_)
  ax.copy_wp_array(ray_geomid, ray_geomid_)

  return ray_dist, ray_geomid
