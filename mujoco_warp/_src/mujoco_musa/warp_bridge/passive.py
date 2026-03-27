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

from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.warp_util import copy_wp_array_batch_attrlist

from ..passive import *


def passive_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "qpos",
    "qvel",
    "ten_J",
    "ten_length",
    "ten_velocity",
    "flexvert_xpos",
    "flexedge_length",
    "flexedge_velocity",
    "qfrc_spring",
    "qfrc_damper",
    "qfrc_fluid",
    "xipos",
    "subtree_com",
    "cdof",
  ]
  outputs = [
    "qfrc_spring",
    "qfrc_damper",
    "qfrc_passive",
    "qfrc_gravcomp",
    "qfrc_fluid",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  passive(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)
