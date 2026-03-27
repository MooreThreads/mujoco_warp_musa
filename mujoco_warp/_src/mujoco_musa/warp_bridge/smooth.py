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

from ..smooth import *


def factor_m_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    "qM",
    "qLD",
    "qLDiagInv",
  ]
  outputs = [
    "qLD",
    "qLDiagInv",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)

  factor_m(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def solve_m_copydata(m: Model, d: Data, x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)): # type: ignore
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    "qLD",
    "qLDiagInv",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)


  x_ = ax.from_wp_array(x)
  y_ = ax.from_wp_array(y)

  solve_m(m_, d_, x_, y_)

  ax.copy_wp_array(x, x_)


def factor_solve_i_copydata(
  m: Model,
  d: Data,
  M: wp.array3d(dtype=float), # type: ignore
  L: wp.array3d(dtype=float), # type: ignore
  D: wp.array2d(dtype=float), # type: ignore
  x: wp.array2d(dtype=float), # type: ignore
  y: wp.array2d(dtype=float), # type: ignore
):
  m_ = m.musa_model
  d_ = d.musa_data

  M_ = ax.from_wp_array(M)
  L_ = ax.from_wp_array(L)
  D_ = ax.from_wp_array(D)
  x_ = ax.from_wp_array(x)
  y_ = ax.from_wp_array(y)

  factor_solve_i(m_, d_, M_, L_, D_, x_, y_)

  ax.copy_wp_array(L, L_)
  ax.copy_wp_array(D, D_)
  ax.copy_wp_array(x, x_)


def rne_copydata(m: Model, d: Data, flg_acc: bool = False):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "cdof",
    "cacc",
    "qvel",
    "qacc",
    "cdof_dot",
    "cinert",
    "cvel",
    "cfrc_ext",
    "cfrc_int",
    "qfrc_bias",
  ]
  outputs = inputs
  copy_wp_array_batch_attrlist(d_, d, inputs)

  rne(m_, d_, flg_acc=flg_acc)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def crb_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "cdof",
    "crb",
    "cinert",
    "qM",
  ]
  outputs = [
    "crb",
    "qM",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)

  crb(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def com_vel_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "qvel",
    "cdof",
    "cvel",
  ]

  outputs = [
    "cvel",
    "cdof_dot",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  com_vel(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def subtree_vel_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "xipos",
    "ximat",
    "subtree_com",
    "cvel",
    "subtree_linvel",
    "subtree_bodyvel",
  ]
  outputs = [
    "subtree_linvel",
    "subtree_angmom",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  subtree_vel(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def kinematics_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "qvel",
    "cdof",
    "cvel",
    "qpos",
    "mocap_pos",
    "mocap_quat",
    "xpos",
    "xquat",
    "xmat",
  ]

  outputs = [
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "ximat",
    "xanchor",
    "xaxis",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)

  kinematics(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def transmission_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "qpos",
    "xquat",
    "site_xpos",
    "site_xmat",
    "subtree_com",
    "cdof",
    "ten_J",
    "ten_length",
    "nacon",
  ]

  inputs_contact = [
    "dist",
    "pos",
    "frame",
    "includemargin",
    "dim",
    "geom",
    "efc_address",
    "worldid",
  ]

  outputs = [
    "actuator_length",
    "actuator_moment",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  copy_wp_array_batch_attrlist(d_.contact, d.contact, inputs_contact)
  ax.copy_wp_array(d_.efc.J, d.efc.J)

  transmission(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def tendon_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "qpos",
    "site_xpos",
    "subtree_com",
    "cdof",
    "geom_xpos",
    "geom_xmat",
  ]
  outputs = [
    "ten_J",
    "ten_length",
    "ten_wrapadr",
    "ten_wrapnum",
    "wrap_obj",
    "wrap_xpos",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  tendon(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def com_pos_copydata(m: Model, d: Data):
  mm = m.musa_model
  dm = d.musa_data
  inputs = [
    "xipos",
    "subtree_com",
    "ximat",
    "xmat",
    "xanchor",
    "xaxis",
  ]

  outputs = [
    "subtree_com",
    "cinert",
    "cdof",
  ]
  copy_wp_array_batch_attrlist(dm, d, inputs)
  com_pos(mm, dm)
  copy_wp_array_batch_attrlist(d, dm, outputs)


def camlight_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "xpos",
    "xquat",
    "subtree_com",
    "cam_xpos",
    "cam_xmat",
    "light_xpos",
    "light_xdir",
  ]
  outputs = [
    "cam_xpos",
    "cam_xmat",
    "light_xpos",
    "light_xdir",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  camlight(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def flex_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    "xpos",
    "qvel",
    "subtree_com",
    "cdof",
    "flexvert_xpos",
    "flexedge_J",
    "flexedge_length",
    "flexedge_velocity",
  ]
  outputs = [
    "flexvert_xpos",
    "flexedge_J",
    "flexedge_length",
    "flexedge_velocity",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  flex(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def tendon_armature_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "ten_J",
    # "qM",
  ]
  outputs = [
    "qM",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  tendon_armature(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def tendon_bias_copydata(m: Model, d: Data, qfrc: wp.array2d(dtype=float)): # type: ignore
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    "site_xpos",
    "subtree_com",
    "cdof",
    "cvel",
    "cdof_dot",
    "qvel",
    "ten_J",
  ]

  qfrc_musa = ax.from_wp_array(qfrc)
  copy_wp_array_batch_attrlist(d_, d, inputs)

  tendon_bias(m_, d_, qfrc_musa)

  ax.copy_wp_array(qfrc, qfrc_musa)


def rne_postconstraint_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "xfrc_applied",
    "xipos",
    "subtree_com",
    "xpos",
    "xmat",
    "efc.id",
    "efc.force",
    "ne_connect",
    "ne_weld",
    "contact.pos",
    "contact.frame",
    "contact.friction",
    "contact.dim",
    "contact.geom",
    "contact.efc_address",
    "contact.worldid",
    "efc.force",
    "nacon",
    "qvel",
    "qacc",
    "cdof",
    "cdof_dot",
    "cacc",
    "cfrc_ext",
    "cfrc_int",
    ]

  outputs = [
    "cfrc_ext",
    "cacc",
    "cfrc_int"
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)

  rne_postconstraint(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)
