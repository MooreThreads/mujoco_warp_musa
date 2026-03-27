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

from ..constraint import *


def make_constraint_copydata(m: Model, d: Data):
  mm = m.musa_model
  dm = d.musa_data

  inputs = [
    "ne",
    "nf",
    "nl",
    "nefc",
    "ne_connect",
    "ne_weld",
    "ne_jnt",
    "ne_ten",
    "ne_flex",
    "qvel",
    "eq_active",
    "xpos",
    "xmat",
    "site_xpos",
    "subtree_com",
    "cdof",
    "xquat",
    "qpos",
    "ten_J",
    "ten_length",
    "flexedge_J",
    "flexedge_length",
    "nacon",
    "contact.dist",
    "contact.dim",
    "contact.includemargin",
    "contact.worldid",
    "contact.geom",
    "contact.pos",
    "contact.frame",
    "contact.friction",
    "contact.solref",
    "contact.solimp",
    "contact.type",
  ]

  outputs_pyramidal = [
    "ne",
    "nf",
    "nl",
    "ne_connect",
    "ne_weld",
    "ne_jnt",
    "ne_ten",
    "ne_flex",
  ]

  outputs_elliptic = [
    "nefc",
    "contact.efc_address",
    "efc.type",
    "efc.id",
    "efc.J",
    "efc.pos",
    "efc.margin",
    "efc.D",
    "efc.vel",
    "efc.aref",
    "efc.frictionloss",
  ]

  copy_wp_array_batch_attrlist(dm, d, inputs)

  # copy input !
  ax.copy_wp_array(dm.contact.efc_address, d.contact.efc_address)
  ax.copy_wp_array(dm.efc.type, d.efc.type)
  ax.copy_wp_array(dm.efc.id, d.efc.id)
  ax.copy_wp_array(dm.efc.J, d.efc.J)
  ax.copy_wp_array(dm.efc.pos, d.efc.pos)
  ax.copy_wp_array(dm.efc.margin, d.efc.margin)
  ax.copy_wp_array(dm.efc.D, d.efc.D)
  ax.copy_wp_array(dm.efc.vel, d.efc.vel)
  ax.copy_wp_array(dm.efc.aref, d.efc.aref)
  ax.copy_wp_array(dm.efc.frictionloss, d.efc.frictionloss)

  make_constraint(mm, dm)

  copy_wp_array_batch_attrlist(d, dm, outputs_pyramidal)
  copy_wp_array_batch_attrlist(d, dm, outputs_elliptic)
