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

from ..sensor import *


def energy_pos_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "xipos",
    "qpos",
    "ten_length",
  ]
  outputs = [
    "energy"
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  energy_pos(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def energy_vel_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "qM",
    "qvel",
  ]
  outputs = [
    "energy"
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  energy_vel(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def sensor_pos_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "time",
    "energy",
    "qpos",
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "ximat",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
    "cam_xpos",
    "cam_xmat",
    "subtree_com",
    "ten_length",
    "actuator_length",
    "sensordata",
    "ne",
    "nf",
    "nl",
    "qM",
    "qvel",
    "nacon"
]

  inputs_efc = [
    "type",
    "id",
    "pos",
    "margin",
  ]

  inputs_contact = [
    "dist",
    "pos",
    "frame",
    "geom",
    "worldid",
    "type",
    "geomcollisionid",
  ]

  outputs = [
    "sensordata"
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  copy_wp_array_batch_attrlist(d_.efc, d.efc, inputs_efc)
  copy_wp_array_batch_attrlist(d_.contact, d.contact, inputs_contact)

  sensor_pos(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def sensor_vel_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "xfrc_applied",
    "qvel",
    "xpos",
    "xmat",
    "xipos",
    "ximat",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
    "cam_xpos",
    "cam_xmat",
    "subtree_com",
    "ten_velocity",
    "actuator_velocity",
    "cvel",
    "subtree_linvel",
    "subtree_angmom",
    "ne",
    "nf",
    "nl",
    "sensordata",
    "efc.type",
    "efc.id",
    "efc.vel",
  ]

  outputs = [
    "sensordata"
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)

  sensor_vel(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def sensor_acc_copydata(m: Model, d: Data):
  mm = m.musa_model
  dm = d.musa_data

  inputs = [
    "site_xpos",
    "site_xmat",
    "contact.pos",
    "contact.frame",
    "contact.dim",
    "contact.geom",
    "contact.efc_address",
    "contact.worldid",
    "efc.force",
    "nacon",
    "sensordata",
    "geom_xpos",
    "geom_xmat",
    "subtree_com",
    "cvel",
    "xpos",
    "xipos",
    "site_xpos",
    "site_xmat",
    "cam_xpos",
    "actuator_force",
    "qfrc_actuator",
    "cacc",
    "cfrc_int",
    "contact.dist",
    "contact.friction",
    "ne",
    "nf",
    "nl",
    "efc.type",
    "efc.id",
  ]

  outputs = [
    "sensordata"
  ]

  copy_wp_array_batch_attrlist(dm, d, inputs)

  sensor_acc(mm, dm)

  copy_wp_array_batch_attrlist(d, dm, outputs)
