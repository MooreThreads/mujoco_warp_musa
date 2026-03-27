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

from ..support import *


def mul_m_copydata(
  m: Model,
  d: Data,
  res: wp.array2d(dtype=float),
  vec: wp.array2d(dtype=float),
  skip: Optional[wp.array] = None,
  M: Optional[wp.array] = None,
):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "qM",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)

  res_ = ax.from_wp_array(res)
  vec_ = ax.from_wp_array(vec)
  skip_ = ax.from_wp_array(skip) if skip is not None else None
  M_ = ax.from_wp_array(M) if M is not None else None

  mul_m(m_, d_, res_, vec_, skip_, M_)

  ax.copy_wp_array(res, res_)
