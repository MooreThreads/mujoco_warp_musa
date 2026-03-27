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

from typing import Optional

import axinfra as ax

from . import collision_driver
from . import constraint
from . import derivative
from . import passive
from . import sensor
from . import smooth
from . import solver
from . import types as mjmtp
from .cached_array import get_cached_array
from .support import xfrc_accumulate
from .types import DisableBit
from .types import EnableBit
from .types import IntegratorType


def fwd_actuation(m: mjmtp.Model, d: mjmtp.Data):
  if not m.nu or (m.opt.disableflags & DisableBit.ACTUATION):
    d.act_dot.zero_()
    d.qfrc_actuator.zero_()
    return

  ax.launch(
    "_actuator_force",
    dim=(d.nworld, m.nu),
    inputs=[
    m.na,
    m.opt.timestep,
    m.actuator_dyntype,
    m.actuator_gaintype,
    m.actuator_biastype,
    m.actuator_actadr,
    m.actuator_actnum,
    m.actuator_ctrllimited,
    m.actuator_forcelimited,
    m.actuator_actlimited,
    m.actuator_dynprm,
    m.actuator_gainprm,
    m.actuator_biasprm,
    m.actuator_actearly,
    m.actuator_ctrlrange,
    m.actuator_forcerange,
    m.actuator_actrange,
    m.actuator_acc0,
    m.actuator_lengthrange,
    d.act,
    d.ctrl,
    d.actuator_length,
    d.actuator_velocity,
    m.opt.disableflags & DisableBit.CLAMPCTRL,
    ],
    outputs=[d.act_dot, d.actuator_force],
  )

  if m.ntendon:
    ten_actfrc = get_cached_array('fwd_actuation_''ten_actfrc', (d.nworld, m.ntendon), dtype=float)
    ten_actfrc.zero_()

    ax.launch(
      "_tendon_actuator_force",
      dim=(d.nworld, m.nu),
      inputs=[
      m.actuator_trntype, 
      m.actuator_trnid, 
      d.actuator_force],
      outputs=[ten_actfrc],
    )

    ax.launch(
      "_tendon_actuator_force_clamp",
      dim=(d.nworld, m.nu),
      inputs=[m.tendon_actfrclimited,
                m.tendon_actfrcrange, 
                m.actuator_trntype, 
                m.actuator_trnid, 
                ten_actfrc],
      outputs=[d.actuator_force],
    )

  ax.launch(
    "_qfrc_actuator",
    dim=(d.nworld, m.nv),
    inputs=[
      m.nu,
      m.ngravcomp,
      m.jnt_actfrclimited,
      m.jnt_actgravcomp,
      m.jnt_actfrcrange,
      m.dof_jntid,
      d.actuator_moment,
      d.qfrc_gravcomp,
      d.actuator_force,
    ],
    outputs=[d.qfrc_actuator],
  )


def fwd_velocity(m: mjmtp.Model, d: mjmtp.Data):
  """Velocity-dependent computations."""
  ax.launch(
    "_actuator_velocity__actuator_velocity",
    dim=(d.nworld, m.nu),
    inputs=[d.qvel, d.actuator_moment, m.nv],
    outputs=[d.actuator_velocity],
    # block_dim=m.block_dim.actuator_velocity,
  )

  # TODO(team): sparse version
  ax.launch(
    "_tendon_velocity__tendon_velocity",
    dim=(d.nworld, m.ntendon),
    inputs=[d.qvel, d.ten_J, m.nv],
    outputs=[d.ten_velocity],
    # block_dim=m.block_dim.tendon_velocity,
  )

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)
  smooth.tendon_bias(m, d, d.qfrc_bias)


def fwd_acceleration(m: mjmtp.Model, d: mjmtp.Data, factorize: bool = False):
  """Add up all non-constraint forces, compute qacc_smooth.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize inertia matrix.
  """
  ax.launch(
    "_qfrc_smooth",
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_applied, d.qfrc_bias, d.qfrc_passive, d.qfrc_actuator],
    outputs=[d.qfrc_smooth],
  )
  xfrc_accumulate(m, d, d.qfrc_smooth)

  if factorize:
    smooth.factor_solve_i(m, d, d.qM, d.qLD, d.qLDiagInv, d.qacc_smooth, d.qfrc_smooth)
  else:
    smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


def _advance(m_: mjmtp.Model, d_: mjmtp.Data, qacc_: ax.array, qvel_: Optional[ax.array] = None):
  """Advance state and time given activation derivatives and acceleration."""
  # advance activations
  ax.launch(
    "_next_activation",
    dim=(d_.nworld, m_.na),
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
    dim=(d_.nworld, m_.nv),
    inputs=[m_.opt.timestep, d_.qvel, qacc_, 1.0],
    outputs=[d_.qvel],
  )

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  qvel_in = qvel_ or d_.qvel

  ax.launch(
    "_next_position",
    dim=(d_.nworld, m_.njnt),
    inputs=[m_.opt.timestep, m_.jnt_type, m_.jnt_qposadr, m_.jnt_dofadr, d_.qpos, qvel_in, 1.0],
    outputs=[d_.qpos],
  )

  ax.launch(
    "_next_time",
    dim=d_.nworld,
    inputs=[m_.opt.timestep, d_.nefc, d_.time, d_.nworld, d_.naconmax, d_.njmax, d_.nacon, d_.ncollision],
    outputs=[d_.time],
  )
  ax.copy(d_.qacc_warmstart, d_.qacc)


# Integrators: EULER, RK4, IMPLICITFAST

# 1. EULER
def euler(m: mjmtp.Model, d: mjmtp.Data):
  """Euler integrator, semi-implicit in velocity."""
  # integrate damping implicitly
  if not m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER):
    qacc = get_cached_array("euler_""qacc", (d.nworld, m.nv), dtype=float)
    if m.opt.is_sparse:
      qM = get_cached_array("euler_""qM", d.qM.shape, dtype=d.qM.dtype)
      ax.copy(qM, d.qM)
      qLD = get_cached_array("euler_""qLD", (d.nworld, 1, m.nC), dtype=float)
      qLDiagInv = get_cached_array("euler_""qLDiagInv", (d.nworld, m.nv), dtype=float)
      ax.launch(
        "_euler_damp_qfrc_sparse",
        dim=(d.nworld, m.nv),
        inputs=[m.opt.timestep, m.dof_Madr, m.dof_damping],
        outputs=[qM],
      )
      smooth.factor_solve_i(m, d, qM, qLD, qLDiagInv, qacc, d.efc.Ma)
    else:
      for tile in m.qM_tiles:
        ax.launch(
          "_tile_euler_dense__euler_dense",
          dim=(d.nworld, tile.adr.size, tile.size),
          inputs=[m.dof_damping, m.opt.timestep, d.qM, d.efc.Ma, tile.adr, tile.size],
          outputs=[qacc],
          block_dim=tile.size,
        )
    _advance(m, d, qacc)
  else:
    _advance(m, d, d.qacc)


# 2. RK4
def _rk_accumulate(
  m: mjmtp.Model,
  d: mjmtp.Data,
  scale: float,
  qvel_rk: ax.array2d(dtype=float), # type: ignore
  qacc_rk: ax.array2d(dtype=float), # type: ignore
  act_dot_rk: Optional[ax.array] = None,
):
  """Computes one term of 1/6 k_1 + 1/3 k_2 + 1/3 k_3 + 1/6 k_4."""
  ax.launch(
    "_rk_accumulate_velocity_acceleration",
    dim=(d.nworld, m.nv),
    inputs=[d.qvel, d.qacc, scale],
    outputs=[qvel_rk, qacc_rk],
  )

  if m.na and act_dot_rk is not None:
    ax.launch(
      "_rk_accumulate_activation_velocity",
      dim=(d.nworld, m.na),
      inputs=[d.act_dot, scale],
      outputs=[act_dot_rk],
    )


def _rk_perturb_state(
  m: mjmtp.Model,
  d: mjmtp.Data,
  scale: float,
  qpos_t0: ax.array2d(dtype=float), # type: ignore
  qvel_t0: ax.array2d(dtype=float), # type: ignore
  act_t0: Optional[ax.array] = None,
):
  # position
  ax.launch(
    "_next_position",
    dim=(d.nworld, m.njnt),
    inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, qpos_t0, d.qvel, scale],
    outputs=[d.qpos],
  )

  # velocity
  ax.launch(
    "_next_velocity",
    dim=(d.nworld, m.nv),
    inputs=[m.opt.timestep, qvel_t0, d.qacc, scale],
    outputs=[d.qvel],
  )

  # activation
  if m.na and act_t0 is not None:
    ax.launch(
      "_next_activation",
      dim=(d.nworld, m.na),
      inputs=[m.opt.timestep, act_t0, d.act_dot, scale, False],
      outputs=[d.act],
    )


def rungekutta4(m: mjmtp.Model, d: mjmtp.Data):
  """Runge-Kutta explicit order 4 integrator."""
  # RK4 tableau
  A = [0.5, 0.5, 1.0]  # diagonal only
  B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

  qpos_t0 = get_cached_array("rungekutta4_""qpos_t0", d.qpos.shape, d.qpos.dtype)
  ax.copy(qpos_t0, d.qpos)
  qvel_t0 = get_cached_array("rungekutta4_""qvel_t0", d.qvel.shape, d.qvel.dtype)
  ax.copy(qvel_t0, d.qvel)
  qvel_rk = get_cached_array("rungekutta4_""qvel_rk", (d.nworld, m.nv), dtype=float)
  qvel_rk.zero_()
  qacc_rk = get_cached_array("rungekutta4_""qacc_rk", (d.nworld, m.nv), dtype=float)
  qacc_rk.zero_()

  if m.na:
    act_t0 = get_cached_array("rungekutta4_""act_t0", d.act.shape, d.act.dtype)
    ax.copy(act_t0, d.act)
    act_dot_rk = get_cached_array("rungekutta4_""act_dot_rk", (d.nworld, m.na), dtype=float)
    act_dot_rk.zero_()
  else:
    act_t0 = None
    act_dot_rk = None

  _rk_accumulate(m, d, B[0], qvel_rk, qacc_rk, act_dot_rk)

  for i in range(3):
    a, b = float(A[i]), B[i + 1]
    _rk_perturb_state(m, d, a, qpos_t0, qvel_t0, act_t0)
    forward(m, d)
    _rk_accumulate(m, d, b, qvel_rk, qacc_rk, act_dot_rk)

  ax.copy(d.qpos, qpos_t0)
  ax.copy(d.qvel, qvel_t0)

  if m.na:
    ax.copy(d.act, act_t0)
    ax.copy(d.act_dot, act_dot_rk)

  _advance(m, d, qacc_rk, qvel_rk)


# 3. IMPLICITFAST
def implicit(m: mjmtp.Model, d: mjmtp.Data):
  """Integrates fully implicit in velocity."""
  if ~(m.opt.disableflags | ~(DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER)):
    if m.opt.is_sparse:
      qDeriv = get_cached_array("implicit_""qDeriv", (d.nworld, 1, m.nM), dtype=float)
      qLD = get_cached_array("implicit_""qLD", (d.nworld, 1, m.nC), dtype=float)
    else:
      qDeriv = get_cached_array("implicit_""qDeriv", (d.nworld, m.nv, m.nv), dtype=float)
      qLD = get_cached_array("implicit_""qLD", (d.nworld, m.nv, m.nv), dtype=float)
    qLDiagInv = get_cached_array("implicit_""qLDiagInv", (d.nworld, m.nv), dtype=float)
    derivative.deriv_smooth_vel(m, d, qDeriv)
    qacc = get_cached_array("implicit_""qacc", (d.nworld, m.nv), dtype=float)
    smooth.factor_solve_i(m, d, qDeriv, qLD, qLDiagInv, qacc, d.efc.Ma)
    _advance(m, d, qacc)
  else:
    _advance(m, d, d.qacc)


def fwd_position(m: mjmtp.Model, d: mjmtp.Data, factorize: bool = True):
  """Position-dependent computations.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize interia matrix.
  """
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.flex(m, d)
  smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  if factorize:
    smooth.factor_m(m, d)
  if m.opt.run_collision_detection:
    collision_driver.collision(m, d)
  constraint.make_constraint(m, d)
  smooth.transmission(m, d)


def forward(m: mjmtp.Model, d: mjmtp.Data):
  """Forward dynamics."""
  energy = m.opt.enableflags & EnableBit.ENERGY

  fwd_position(m, d, factorize=False)
  d.sensordata.zero_()
  sensor.sensor_pos(m, d)
  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  fwd_actuation(m, d)
  fwd_acceleration(m, d, factorize=True)

  solver.solve(m, d)
  sensor.sensor_acc(m, d)


def step(m: mjmtp.Model, d: mjmtp.Data):
  """Advance simulation."""
  # TODO(team): mj_checkPos
  # TODO(team): mj_checkVel
  forward(m, d)
  # TODO(team): mj_checkAcc

  if m.opt.integrator == IntegratorType.EULER:
    euler(m, d)
  elif m.opt.integrator == IntegratorType.RK4:
    rungekutta4(m, d)
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    implicit(m, d)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")


def step1(m: mjmtp.Model, d: mjmtp.Data):
  """Advance simulation in two phases: before input is set by user."""
  energy = m.opt.enableflags & EnableBit.ENERGY
  # TODO(team): mj_checkPos
  # TODO(team): mj_checkVel
  fwd_position(m, d)
  sensor.sensor_pos(m, d)

  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)


def step2(m: mjmtp.Model, d: mjmtp.Data):
  """Advance simulation in two phases: after input is set by user."""
  fwd_actuation(m, d)
  fwd_acceleration(m, d)
  solver.solve(m, d)
  sensor.sensor_acc(m, d)
  # TODO(team): mj_checkAcc

  # integrate with Euler or implicitfast
  # TODO(team): implicit
  if m.opt.integrator == IntegratorType.IMPLICITFAST:
    implicit(m, d)
  else:
    # note: RK4 defaults to Euler
    euler(m, d)
