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

from ..derivative import *


def deriv_smooth_vel_copydata(m: Model, d: Data, qDeriv: wp.array2d(dtype=float)): # type: ignore
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "act",
    "ctrl",
    "actuator_moment",
    "qM",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)

  qDeriv_ = ax.from_wp_array(qDeriv)
  ax.copy_wp_array(d_.efc.Ma, d.efc.Ma)

  deriv_smooth_vel(m_, d_, qDeriv_)

  ax.copy_wp_array(qDeriv, qDeriv_)
