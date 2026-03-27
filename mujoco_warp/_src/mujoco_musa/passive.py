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

import axinfra as ax

from . import support
from . import types as mjmtp
from .cached_array import get_cached_array
from .types import DisableBit


def _fluid(m: mjmtp.Model, d: mjmtp.Data):
  fluid_applied = get_cached_array("_fluid_""fluid_applied", (d.nworld, m.nbody), dtype=ax.spatial_vector)

  ax.launch(
    "_fluid_force",
    dim=(d.nworld, m.nbody),
    inputs=[
      m.opt.density,
      m.opt.viscosity,
      m.opt.wind,
      m.body_rootid,
      m.body_geomnum,
      m.body_geomadr,
      m.body_mass,
      m.body_inertia,
      m.geom_type,
      m.geom_size,
      m.geom_fluid,
      m.body_fluid_ellipsoid,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.subtree_com,
      d.cvel,
    ],
    outputs=[fluid_applied],
  )

  support.apply_ft(m, d, fluid_applied, d.qfrc_fluid, False)


def passive(m: mjmtp.Model, d: mjmtp.Data):
  """Adds all passive forces."""
  dsbl_spring = m.opt.disableflags & DisableBit.SPRING
  dsbl_damper = m.opt.disableflags & DisableBit.DAMPER

  if dsbl_spring and dsbl_damper:
    d.qfrc_spring.zero_()
    d.qfrc_damper.zero_()
    d.qfrc_gravcomp.zero_()
    d.qfrc_fluid.zero_()
    d.qfrc_passive.zero_()
    return

  ax.launch(
    "_spring_damper_dof_passive",
    dim=(d.nworld, m.njnt),
    inputs=[
      m.opt.disableflags,
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_stiffness,
      m.dof_damping,
      d.qpos,
      d.qvel,
    ],
    outputs=[d.qfrc_spring, d.qfrc_damper],
  )

  if m.ntendon:
    ax.launch(
      "_spring_damper_tendon_passive",
      dim=(d.nworld, m.ntendon, m.nv),
      inputs=[
        m.tendon_stiffness,
        m.tendon_damping,
        m.tendon_lengthspring,
        d.ten_J,
        d.ten_length,
        d.ten_velocity,
        dsbl_spring,
        dsbl_damper,
      ],
      outputs=[
        d.qfrc_spring,
        d.qfrc_damper,
      ],
    )

  if not dsbl_spring:
    ax.launch(
      "_flex_elasticity",
      dim=(d.nworld, m.nflexelem),
      inputs=[
        m.opt.timestep,
        m.body_dofadr,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_elemedgeadr,
        m.flex_vertbodyid,
        m.flex_elem,
        m.flex_elemedge,
        m.flexedge_length0,
        m.flex_stiffness,
        m.flex_damping,
        d.flexvert_xpos,
        d.flexedge_length,
        d.flexedge_velocity,
        dsbl_damper,
      ],
      outputs=[d.qfrc_spring],
    )
  ax.launch(
    "_flex_bending",
    dim=(d.nworld, m.nflexedge),
    inputs=[
      m.body_dofadr,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_edgeadr,
      m.flex_vertbodyid,
      m.flex_edge,
      m.flex_edgeflap,
      m.flex_bending,
      d.flexvert_xpos,
    ],
    outputs=[d.qfrc_spring],
  )

  gravcomp = m.ngravcomp and not (m.opt.disableflags & DisableBit.GRAVITY)

  if gravcomp:
    d.qfrc_gravcomp.zero_()
    ax.launch(
      "_gravity_force",
      dim=(d.nworld, m.nbody - 1, m.nv),
      inputs=[
        m.opt.gravity,
        m.body_parentid,
        m.body_rootid,
        m.body_mass,
        m.body_gravcomp,
        m.dof_bodyid,
        d.xipos,
        d.subtree_com,
        d.cdof,
      ],
      outputs=[d.qfrc_gravcomp],
    )

  if m.opt.has_fluid:
    _fluid(m, d)

  ax.launch(
    "_qfrc_passive",
    dim=(d.nworld, m.nv),
    inputs=[
      m.opt.has_fluid,
      m.jnt_actgravcomp,
      m.dof_jntid,
      d.qfrc_spring,
      d.qfrc_damper,
      d.qfrc_gravcomp,
      d.qfrc_fluid,
      gravcomp,
    ],
    outputs=[
      d.qfrc_passive,
    ],
  )
