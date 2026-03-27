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

from ..forward import *


def fwd_actuation_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
      "act",
      "ctrl",
      "actuator_length",
      "actuator_velocity",
      "actuator_moment",
      "qfrc_gravcomp",
      "actuator_force",
      "qfrc_actuator",
      "act_dot",
  ]
  outputs = [
      "act_dot",
      "actuator_force",
      "qfrc_actuator"
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  fwd_actuation(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def fwd_acceleration_copydata(m: Model, d: Data, factorize: bool = False):
  """Add up all non-constraint forces, compute qacc_smooth.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize inertia matrix.
  """
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
     "qfrc_applied",
     "qfrc_bias",
     "qfrc_passive",
     "qfrc_actuator",
     "xipos",
     "subtree_com",
     "cdof",
     "xfrc_applied",
     "qfrc_smooth",
  ]
  outputs = [
    "qfrc_smooth",
    "qacc_smooth",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)
  fwd_acceleration(m_, d_, factorize)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def _advance_copydata(m: Model, d: Data, qacc: wp.array, qvel: Optional[wp.array] = None):
  """Advance state and time given activation derivatives and acceleration."""
  # TODO(team): can we assume static timesteps?
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "act",
    "act_dot",
    "qvel",
    "qpos",
    "nefc",
    "time",
    "nacon",
    "ncollision",
    "qacc",
  ]
  outputs = [
    "act",
    "qvel",
    "qpos",
    "time",
    "qacc_warmstart",
  ]
  copy_wp_array_batch_attrlist(d_, d, inputs)

  qacc_ = ax.from_wp_array(qacc)
  qvel_ = ax.from_wp_array(qvel) if qvel is not None else None

  # advance activations
  ax.launch(
    "_next_activation",
    dim=(d.nworld, m.na),
    inputs=[
      m_.opt.timestep,
      m_.actuator_dyntype,
      m_.actuator_actlimited,
      m_.actuator_dynprm,
      m_.actuator_actrange,
      d_.act,
      d_.act_dot,
      1.0,
      True,
    ],
    outputs=[d_.act],
  )

  ax.launch(
    "_next_velocity",
    dim=(d.nworld, m.nv),
    inputs=[m_.opt.timestep, d_.qvel, qacc_, 1.0],
    outputs=[d_.qvel],
  )

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  qvel_in = qvel_ or d_.qvel

  ax.launch(
    "_next_position",
    dim=(d.nworld, m.njnt),
    inputs=[m_.opt.timestep, m_.jnt_type, m_.jnt_qposadr, m_.jnt_dofadr, d_.qpos, qvel_in, 1.0],
    outputs=[d_.qpos],
  )

  ax.launch(
    "_next_time",
    dim=d.nworld,
    inputs=[m_.opt.timestep, d_.nefc, d_.time, d.nworld, d.naconmax, d.njmax, d_.nacon, d_.ncollision],
    outputs=[d_.time],
  )

  ax.copy(d_.qacc_warmstart, d_.qacc)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def euler_copydata(m: Model, d: Data):
  """Euler integrator, semi-implicit in velocity."""
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "act",
    "act_dot",
    "qvel",
    "qpos",
    "nefc",
    "time",
    "nacon",
    "ncollision",
    "qacc",
    "qM"
  ]
  outputs = [
    "act",
    "qvel",
    "qpos",
    "time",
    "qacc_warmstart",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  ax.copy_wp_array(d_.efc.Ma, d.efc.Ma)

  euler(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def implicit_copydata(m: Model, d: Data):
  m_ = m.musa_model
  d_ = d.musa_data
  inputs = [
    "act",
    "act_dot",
    "qvel",
    "qpos",
    "nefc",
    "time",
    "nacon",
    "ncollision",
    "qacc",
    "qM",
    "ctrl",
    "actuator_moment",
  ]
  outputs = [
    "act",
    "qvel",
    "qpos",
    "time",
    "qacc_warmstart",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  ax.copy_wp_array(d_.efc.Ma, d.efc.Ma)

  implicit(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def forward_copydata(m: Model, d: Data):
  """Advance simulation."""
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    "solver_niter",
    "ne",
    "nf",
    "nl",
    "nefc",
    "time",
    "energy",
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "qacc",
    "act_dot",
    "sensordata",
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
    "cam_xpos",
    "cam_xmat",
    "light_xpos",
    "light_xdir",
    "subtree_com",
    "cdof",
    "cinert",
    "flexvert_xpos",
    "flexedge_J",
    "flexedge_length",
    "ten_wrapadr",
    "ten_wrapnum",
    "ten_J",
    "ten_length",
    "wrap_obj",
    "wrap_xpos",
    "actuator_length",
    "actuator_moment",
    "crb",
    "qM",
    "qLD",
    "qLDiagInv",
    "flexedge_velocity",
    "ten_velocity",
    "actuator_velocity",
    "cvel",
    "cdof_dot",
    "qfrc_bias",
    "qfrc_spring",
    "qfrc_damper",
    "qfrc_gravcomp",
    "qfrc_fluid",
    "qfrc_passive",
    "subtree_linvel",
    "subtree_angmom",
    "actuator_force",
    "qfrc_actuator",
    "qfrc_smooth",
    "qacc_smooth",
    "qfrc_constraint",
    "qfrc_inverse",
    "cacc",
    "cfrc_int",
    "cfrc_ext",
  ]
  outputs = inputs

  copy_wp_array_batch_attrlist(d_, d, inputs)
  forward(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)


def step_copydata(m: Model, d: Data):
  """Advance simulation."""
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    "solver_niter",
    "ne",
    "nf",
    "nl",
    "nefc",
    "time",
    "energy",
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "qacc",
    "act_dot",
    "sensordata",
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
    "cam_xpos",
    "cam_xmat",
    "light_xpos",
    "light_xdir",
    "subtree_com",
    "cdof",
    "cinert",
    "flexvert_xpos",
    "flexedge_J",
    "flexedge_length",
    "ten_wrapadr",
    "ten_wrapnum",
    "ten_J",
    "ten_length",
    "wrap_obj",
    "wrap_xpos",
    "actuator_length",
    "actuator_moment",
    "crb",
    "qM",
    "qLD",
    "qLDiagInv",
    "flexedge_velocity",
    "ten_velocity",
    "actuator_velocity",
    "cvel",
    "cdof_dot",
    "qfrc_bias",
    "qfrc_spring",
    "qfrc_damper",
    "qfrc_gravcomp",
    "qfrc_fluid",
    "qfrc_passive",
    "subtree_linvel",
    "subtree_angmom",
    "actuator_force",
    "qfrc_actuator",
    "qfrc_smooth",
    "qacc_smooth",
    "qfrc_constraint",
    "qfrc_inverse",
    "cacc",
    "cfrc_int",
    "cfrc_ext",
  ]
  outputs = inputs

  copy_wp_array_batch_attrlist(d_, d, inputs)
  step(m_, d_)
  copy_wp_array_batch_attrlist(d, d_, outputs)
