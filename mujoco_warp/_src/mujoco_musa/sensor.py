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
from axinfra import vec6

from . import smooth
from . import support
from . import types as mjmtp
from .cached_array import get_cached_array
from .types import DisableBit


def energy_pos(m: mjmtp.Model, d: mjmtp.Data):
  """Position-dependent energy (potential)."""
  ax.launch("_energy_pos_zero", dim=d.nworld, outputs=[d.energy])

  # init potential energy: -sum_i(body_i.mass * dot(gravity, body_i.pos))
  if not m.opt.disableflags & DisableBit.GRAVITY:
    ax.launch(
      "_energy_pos_gravity", dim=(d.nworld, m.nbody - 1), inputs=[m.opt.gravity, m.body_mass, d.xipos], outputs=[d.energy]
    )

  if not m.opt.disableflags & DisableBit.SPRING:
    # add joint-level springs
    ax.launch(
      "_energy_pos_passive_joint",
      dim=(d.nworld, m.njnt),
      inputs=[
        m.qpos_spring,
        m.jnt_type,
        m.jnt_qposadr,
        m.jnt_stiffness,
        d.qpos,
      ],
      outputs=[d.energy],
    )

    # add tendon-level springs
    if m.ntendon:
      ax.launch(
        "_energy_pos_passive_tendon",
        dim=(d.nworld, m.ntendon),
        inputs=[
          m.tendon_stiffness,
          m.tendon_lengthspring,
          d.ten_length,
        ],
        outputs=[d.energy],
      )


def energy_vel(m: mjmtp.Model, d: mjmtp.Data):
  """Velocity-dependent energy (kinetic)."""
  # kinetic energy: 0.5 * qvel.T @ M @ qvel

  # M @ qvel
  support.mul_m(m, d, d.efc.mv, d.qvel)

  ax.launch(
    "_energy_vel_kinetic__energy_vel_kinetic",
    dim=d.nworld,
    inputs=[d.qvel, d.efc.mv, m.nv],
    outputs=[d.energy],
    block_dim=1,
  )


def sensor_pos(m: mjmtp.Model, d: mjmtp.Data):
  """Compute position-dependent sensor values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return

  # rangefinder
  rangefinder_dist_ = get_cached_array("sensor_pos_""rangefinder_dist_", (d.nworld, m.nrangefinder), dtype=float)
  if m.sensor_rangefinder_adr.size > 0:
    rangefinder_pnt_ = get_cached_array("sensor_pos_""rangefinder_pnt_", (d.nworld, m.nrangefinder), dtype=ax.vec3)
    rangefinder_vec_ = get_cached_array("sensor_pos_""rangefinder_vec_", (d.nworld, m.nrangefinder), dtype=ax.vec3)
    rangefinder_geomid_ = get_cached_array("sensor_pos_""rangefinder_geomid_", (d.nworld, m.nrangefinder), dtype=int)

    # get position and direction
    ax.launch(
      "_sensor_rangefinder_init",
      dim=(d.nworld, m.sensor_rangefinder_adr.size),
      inputs=[m.sensor_objid, m.sensor_rangefinder_adr, d.site_xpos, d.site_xmat],
      outputs=[rangefinder_pnt_, rangefinder_vec_],
    )

    from .ray import rays
    # get distances
    rays(
      m,
      d,
      rangefinder_pnt_,
      rangefinder_vec_,
      # vec6(ax.inf, ax.inf, ax.inf, ax.inf, ax.inf, ax.inf),
      vec6(-1, -1, -1, -1, -1, -1),
      True,
      m.sensor_rangefinder_bodyid,
      rangefinder_dist_,
      rangefinder_geomid_,
    )

  if m.sensor_e_potential:
    energy_pos(m, d)

  if m.sensor_e_kinetic:
    energy_vel(m, d)

  # collision sensors (distance, normal, fromto)
  sensor_collision_ = get_cached_array("sensor_pos_""sensor_collision_", (d.nworld, m.nsensorcollision, 8, 7), dtype=float)
  sensor_collision_.fill_(1.0e32)
  if m.nsensorcollision:
    ax.launch(
      "_sensor_collision",
      dim=d.naconmax,
      inputs=[
        m.ngeom,
        m.nxn_pairid,
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.geom,
        d.contact.worldid,
        d.contact.type,
        d.contact.geomcollisionid,
        d.nacon,
      ],
      outputs=[sensor_collision_],
    )

  ax.launch(
    "_sensor_pos",
    dim=(d.nworld, m.sensor_pos_adr.size),
    inputs=[
      m.ngeom,
      m.opt.magnetic,
      m.body_geomnum,
      m.body_geomadr,
      m.body_iquat,
      m.jnt_qposadr,
      m.geom_type,
      m.geom_bodyid,
      m.geom_quat,
      m.site_type,
      m.site_bodyid,
      m.site_size,
      m.site_quat,
      m.cam_bodyid,
      m.cam_quat,
      m.cam_fovy,
      m.cam_resolution,
      m.cam_sensorsize,
      m.cam_intrinsic,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.nxn_pairid,
      m.sensor_pos_adr,
      m.rangefinder_sensor_adr,
      d.time,
      d.energy,
      d.qpos,
      d.xpos,
      d.xquat,
      d.xmat,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.subtree_com,
      d.ten_length,
      d.actuator_length,
      rangefinder_dist_,
      sensor_collision_,
    ],
    outputs=[d.sensordata],
  )

  # jointlimitpos and tendonlimitpos
  ax.launch(
    "_limit_pos",
    dim=(d.nworld, d.njmax, m.sensor_limitpos_adr.size),
    inputs=[
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_limitpos_adr,
      d.ne,
      d.nf,
      d.nl,
      d.efc.type,
      d.efc.id,
      d.efc.pos,
      d.efc.margin,
    ],
    outputs=[
      d.sensordata,
    ],
  )


def sensor_vel(m: mjmtp.Model, d: mjmtp.Data):
  """Compute velocity-dependent sensor values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return

  if m.sensor_subtree_vel:
    smooth.subtree_vel(m, d)

  ax.launch(
    "_sensor_vel",
    dim=(d.nworld, m.sensor_vel_adr.size),
    inputs=[
      m.body_rootid,
      m.jnt_dofadr,
      m.geom_bodyid,
      m.site_bodyid,
      m.cam_bodyid,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_vel_adr,
      d.qvel,
      d.xpos,
      d.xmat,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.subtree_com,
      d.ten_velocity,
      d.actuator_velocity,
      d.cvel,
      d.subtree_linvel,
      d.subtree_angmom,
    ],
    outputs=[d.sensordata],
  )

  ax.launch(
    "_limit_vel",
    dim=(d.nworld, d.njmax, m.sensor_limitvel_adr.size),
    inputs=[
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_limitvel_adr,
      d.ne,
      d.nf,
      d.nl,
      d.efc.type,
      d.efc.id,
      d.efc.vel,
    ],
    outputs=[
      d.sensordata,
    ],
  )


def sensor_acc(m: mjmtp.Model, d: mjmtp.Data):
  """Compute acceleration-dependent sensor values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return

  ax.launch(
    "_sensor_touch",
    dim=(d.naconmax, m.sensor_touch_adr.size),
    inputs=[
      m.opt.cone,
      m.geom_bodyid,
      m.site_type,
      m.site_bodyid,
      m.site_size,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_touch_adr,
      d.site_xpos,
      d.site_xmat,
      d.contact.pos,
      d.contact.frame,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.worldid,
      d.efc.force,
      d.nacon,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  ax.launch(
    "_sensor_tactile",
    dim=(d.naconmax, m.nsensortaxel),
    inputs=[
      m.body_rootid,
      m.body_weldid,
      m.oct_child,
      m.oct_aabb,
      m.oct_coeff,
      m.geom_type,
      m.geom_bodyid,
      m.geom_size,
      m.mesh_vertadr,
      m.mesh_normaladr,
      m.mesh_vert,
      m.mesh_normal,
      m.mesh_quat,
      m.sensor_objid,
      m.sensor_refid,
      m.sensor_dim,
      m.sensor_adr,
      m.plugin,
      m.plugin_attr,
      m.geom_plugin_index,
      m.taxel_vertadr,
      m.taxel_sensorid,
      d.geom_xpos,
      d.geom_xmat,
      d.subtree_com,
      d.cvel,
      d.contact.geom,
      d.contact.worldid,
      d.nacon,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  sensor_contact_nmatch = get_cached_array("sensor_acc_""sensor_contact_nmatch", (d.nworld, m.nsensorcontact), dtype=int)
  sensor_contact_matchid = get_cached_array("sensor_acc_""sensor_contact_matchid", (d.nworld, m.nsensorcontact, m.opt.contact_sensor_maxmatch), dtype=int)
  sensor_contact_direction = get_cached_array("sensor_acc_""sensor_contact_direction", (d.nworld, m.nsensorcontact, m.opt.contact_sensor_maxmatch), dtype=float)
  if m.nsensorcontact:
    sensor_contact_criteria = get_cached_array("sensor_acc_""sensor_contact_criteria", (d.nworld, m.nsensorcontact, m.opt.contact_sensor_maxmatch), dtype=float)
    # TODO(team): fill_ operations in one kernel?
    sensor_contact_nmatch.fill_(0)
    sensor_contact_matchid.fill_(-1)
    sensor_contact_criteria.fill_(1.0e32)
    sensor_contact_direction.zero_()

    ax.launch(
      "_contact_match",
      dim=(m.sensor_contact_adr.size, d.naconmax),
      inputs=[
        m.opt.cone,
        m.opt.contact_sensor_maxmatch,
        m.body_parentid,
        m.geom_bodyid,
        m.site_type,
        m.site_size,
        m.sensor_objtype,
        m.sensor_objid,
        m.sensor_reftype,
        m.sensor_refid,
        m.sensor_intprm,
        m.sensor_contact_adr,
        d.site_xpos,
        d.site_xmat,
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.friction,
        d.contact.dim,
        d.contact.geom,
        d.contact.efc_address,
        d.contact.worldid,
        d.contact.type,
        d.efc.force,
        d.njmax,
        d.nacon,
      ],
      outputs=[sensor_contact_nmatch, sensor_contact_matchid, sensor_contact_criteria, sensor_contact_direction],
    )

    ax.launch(
      "_contact_sort__contact_sort",
      dim=(d.nworld, m.sensor_contact_adr.size, m.opt.contact_sensor_maxmatch),
      inputs=[m.sensor_intprm, m.sensor_contact_adr, sensor_contact_nmatch, sensor_contact_matchid, sensor_contact_criteria, m.opt.contact_sensor_maxmatch],
      outputs=[sensor_contact_matchid],
      block_dim=m.opt.contact_sensor_maxmatch,
    )

  if m.sensor_rne_postconstraint:
    smooth.rne_postconstraint(m, d)

  ax.launch(
    "_sensor_acc",
    dim=(d.nworld, m.sensor_acc_adr.size),
    inputs=[
      m.opt.cone,
      m.body_rootid,
      m.jnt_dofadr,
      m.geom_bodyid,
      m.site_bodyid,
      m.cam_bodyid,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_intprm,
      m.sensor_dim,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_acc_adr,
      m.sensor_adr_to_contact_adr,
      d.xpos,
      d.xipos,
      d.geom_xpos,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.subtree_com,
      d.cvel,
      d.actuator_force,
      d.qfrc_actuator,
      d.cacc,
      d.cfrc_int,
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.force,
      d.njmax,
      d.nacon,
      sensor_contact_nmatch,
      sensor_contact_matchid,
      sensor_contact_direction,
    ],
    outputs=[d.sensordata],
  )

  ax.launch(
    "_tendon_actuator_force_sensor",
    dim=(d.nworld, m.sensor_tendonactfrc_adr.size, m.nu),
    inputs=[
      m.actuator_trntype,
      m.actuator_trnid,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_tendonactfrc_adr,
      d.actuator_force,
    ],
    outputs=[
      d.sensordata,
    ],
  )

  ax.launch(
    "_tendon_actuator_force_cutoff",
    dim=(d.nworld, m.sensor_tendonactfrc_adr.size),
    inputs=[
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_tendonactfrc_adr,
      d.sensordata,
    ],
    outputs=[d.sensordata],
  )

  ax.launch(
    "_limit_frc",
    dim=(d.nworld, d.njmax, m.sensor_limitfrc_adr.size),
    inputs=[
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_limitfrc_adr,
      d.ne,
      d.nf,
      d.nl,
      d.efc.type,
      d.efc.id,
      d.efc.force,
    ],
    outputs=[
      d.sensordata,
    ],
  )
