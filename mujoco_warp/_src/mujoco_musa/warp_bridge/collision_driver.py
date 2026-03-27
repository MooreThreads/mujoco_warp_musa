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

from ..collision_driver import *


def collision_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    #"naconmax",
    "geom_xpos",
    "geom_xmat",
    "collision_pair",
    "collision_pairid",
    "collision_worldid",
    "ncollision",
    "nacon",
    "contact.dist",
    "contact.pos",
    "contact.frame",
    "contact.includemargin",
    "contact.friction",
    "contact.solref",
    "contact.solreffriction",
    "contact.solimp",
    "contact.dim",
    "contact.geom",
    "contact.worldid",
    "contact.type",
    "contact.geomcollisionid",
  ]

  outputs = [
    "collision_pair",
    "collision_pairid",
    "collision_worldid",
    "nacon",
    "ncollision",
    "contact.dist",
    "contact.pos",
    "contact.frame",
    "contact.includemargin",
    "contact.friction",
    "contact.solref",
    "contact.solreffriction",
    "contact.solimp",
    "contact.dim",
    "contact.geom",
    "contact.worldid",
    "contact.type",
    "contact.geomcollisionid",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)

  collision(m_, d_)
  # collision_use_warp_sort(m,d)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def broadphase_copydata(m: Model, d: Data):
  #from mujoco_warp._src.collision_driver import sap_broadphase
  #from mujoco_warp._src.collision_driver import nxn_broadphase
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    #"naconmax",
    "geom_xpos",
    "geom_xmat",
    "collision_pair",
    "collision_pairid",
    "collision_worldid",
    "ncollision",
    "nacon",
    "contact.dist",
    "contact.pos",
    "contact.frame",
    "contact.includemargin",
    "contact.friction",
    "contact.solref",
    "contact.solreffriction",
    "contact.solimp",
    "contact.dim",
    "contact.geom",
    "contact.worldid",
    "contact.type",
    "contact.geomcollisionid",
  ]

  outputs = [
    "collision_pair",
    "collision_pairid",
    "collision_worldid",
    "nacon",
    "ncollision",
    "contact.dist",
    "contact.pos",
    "contact.frame",
    "contact.includemargin",
    "contact.friction",
    "contact.solref",
    "contact.solreffriction",
    "contact.solimp",
    "contact.dim",
    "contact.geom",
    "contact.worldid",
    "contact.type",
    "contact.geomcollisionid",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  if m.opt.broadphase == BroadphaseType.NXN:
    nxn_broadphase(m_, d_)
  else:
    sap_broadphase(m_, d_)
    #sap_broadphase_use_warp_sort(m,d)
  #collision(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)