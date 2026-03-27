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

from . import types as mjmtp
from .cached_array import get_cached_array
from .types import DisableBit


def _factor_i_sparse(m: mjmtp.Model, d: mjmtp.Data, M: ax.array3d(dtype=float), L: ax.array3d(dtype=float), D: ax.array2d(dtype=float)): # type: ignore
  """Sparse L'*D*L factorization of inertia-like matrix M, assumed spd."""
  ax.launch("_copy_CSR", dim=(d.nworld, m.nC), inputs=[m.mapM2M, M], outputs=[L])

  for i in reversed(range(len(m.qLD_updates))):
    qLD_updates = m.qLD_updates[i]
    ax.launch("_qLD_acc", dim=(d.nworld, qLD_updates.size), inputs=[m.M_rownnz, m.M_rowadr, qLD_updates, L], outputs=[L])

  ax.launch("_qLDiag_div", dim=(d.nworld, m.nv), inputs=[m.M_rownnz, m.M_rowadr, L], outputs=[D])


def _factor_i_dense(m: mjmtp.Model, d: mjmtp.Data, M: ax.array3d(dtype=float), L: ax.array3d(dtype=float)): # type: ignore
  """Dense Cholesky factorization of inertia-like matrix M, assumed spd."""
  for tile in m.qM_tiles:
    ax.launch(
      "_tile_cholesky_factorize__cholesky_factorize",
      dim=(d.nworld, tile.adr.size, tile.size),
      inputs=[M, tile.adr, tile.size],
      outputs=[L],
      block_dim=tile.size,
    )


def _factor_solve_i_dense(
  m: mjmtp.Model,
  d: mjmtp.Data,
  M: ax.array3d(dtype=float), # type: ignore
  x: ax.array2d(dtype=float), # type: ignore
  y: ax.array2d(dtype=float), # type: ignore
  L: ax.array3d(dtype=float), # type: ignore
):
  for tile in m.qM_tiles:
    ax.launch(
      "_tile_cholesky_factorize_solve__cholesky_factorize_solve",
      dim=(d.nworld, tile.adr.size, tile.size),
      inputs=[M, y, tile.adr, tile.size],
      outputs=[x, L],
      block_dim=tile.size,
    )


def _solve_LD_sparse(
  m: mjmtp.Model,
  d: mjmtp.Data,
  L: ax.array3d(dtype=float), # type: ignore
  D: ax.array2d(dtype=float), # type: ignore
  x: ax.array2d(dtype=float), # type: ignore
  y: ax.array2d(dtype=float), # type: ignore
):
  """Computes sparse backsubstitution: x = inv(L'*D*L)*y."""
  ax.copy(x, y)
  for qLD_updates in reversed(m.qLD_updates):
    ax.launch("_solve_LD_sparse_x_acc_up", dim=(d.nworld, qLD_updates.size), inputs=[L, qLD_updates], outputs=[x])

  ax.launch("_solve_LD_sparse_qLDiag_mul", dim=(d.nworld, m.nv), inputs=[D], outputs=[x])

  for qLD_updates in m.qLD_updates:
    ax.launch("_solve_LD_sparse_x_acc_down", dim=(d.nworld, qLD_updates.size), inputs=[L, qLD_updates], outputs=[x])


def _solve_LD_dense(m: mjmtp.Model, d: mjmtp.Data, L: ax.array3d(dtype=float), x: ax.array2d(dtype=float), y: ax.array2d(dtype=float)): # type: ignore
  """Computes dense backsubstitution: x = inv(L'*L)*y."""
  for tile in m.qM_tiles:
    ax.launch(
      "_tile_cholesky_solve__cholesky_solve",
      dim=(d.nworld, tile.adr.size, tile.size),
      inputs=[L, y, tile.adr, tile.size],
      outputs=[x],
      block_dim=tile.size,
    )


def solve_LD(
  m: mjmtp.Model,
  d: mjmtp.Data,
  L: ax.array3d(dtype=float), # type: ignore
  D: ax.array2d(dtype=float), # type: ignore
  x: ax.array2d(dtype=float), # type: ignore
  y: ax.array2d(dtype=float), # type: ignore
):
  """Computes backsubstitution to solve a linear system of the form x = inv(L'*D*L) * y.

  L and D are the factors from the Cholesky factorization of the inertia matrix.

  This function dispatches to either a sparse or dense solver depending on Model options.

  Args:
    m: The model containing factorization and sparsity information.
    d: The data object containing workspace and factorization results.
    L: Lower-triangular factor from the factorization (sparse or dense).
    D: Diagonal factor from the factorization (only used for sparse).
    x: Output array for the solution.
    y: Input right-hand side array.
  """
  if m.opt.is_sparse:
    _solve_LD_sparse(m, d, L, D, x, y)
  else:
    _solve_LD_dense(m, d, L, x, y)


# 1. factor
def factor_m(m: mjmtp.Model, d: mjmtp.Data):
  """Factorization of inertia-like matrix M, assumed spd."""
  if m.opt.is_sparse:
    _factor_i_sparse(m, d, d.qM, d.qLD, d.qLDiagInv)
  else:
    _factor_i_dense(m, d, d.qM, d.qLD)


# 2. back substitution
def solve_m(m: mjmtp.Model, d: mjmtp.Data, x: ax.array2d(dtype=float), y: ax.array2d(dtype=float)): # type: ignore
  """Computes backsubstitution: x = qLD * y.

  Args:
    m: The model containing inertia and factorization information.
    d: The data object containing factorization results.
    x: Output array for the solution.
    y: Input right-hand side array.
  """
  solve_LD(m, d, d.qLD, d.qLDiagInv, x, y)


# 3. factor + back substitution
def factor_solve_i(
  m: mjmtp.Model,
  d: mjmtp.Data,
  M: ax.array3d(dtype=float), # type: ignore
  L: ax.array3d(dtype=float), # type: ignore
  D: ax.array2d(dtype=float), # type: ignore
  x: ax.array2d(dtype=float), # type: ignore
  y: ax.array2d(dtype=float), # type: ignore
):
  """Factorizes and solves the linear system: x = inv(L'*D*L) * y or x = inv(L'*L) * y.

  M is an inertia-like matrix and L, D are its Cholesky-like factors.

  This function first factorizes the matrix M (sparse or dense depending on model options),
  then solves the system for x given right-hand side y.

  Args:
    m: The model containing factorization and sparsity information.
    d: The data object containing workspace and factorization results.
    M: The inertia-like matrix to factorize.
    L: Output lower-triangular factor from the factorization (sparse or dense).
    D: Output diagonal factor from the factorization (only used for sparse).
    x: Output array for the solution.
    y: Input right-hand side array.
  """
  if m.opt.is_sparse:
    _factor_i_sparse(m, d, M, L, D)
    _solve_LD_sparse(m, d, L, D, x, y)
  else:
    _factor_solve_i_dense(m, d, M, x, y, L)


def _rne_cacc_world(m: mjmtp.Model, d: mjmtp.Data):
  if m.opt.disableflags & DisableBit.GRAVITY:
    d.cacc.zero_()
  else:
    ax.launch("_cacc_world", dim=[d.nworld], inputs=[m.opt.gravity], outputs=[d.cacc])


def _rne_cacc_forward(m: mjmtp.Model, d: mjmtp.Data, flg_acc: bool = False):
  for body_tree in m.body_tree:
    ax.launch(
      "_cacc",
      dim=(d.nworld, body_tree.size),
      inputs=[
        m.body_parentid,
        m.body_dofnum,
        m.body_dofadr,
        d.qvel,
        d.qacc,
        d.cdof,
        d.cdof_dot,
        d.cacc,
        body_tree,
        flg_acc,
      ],
      outputs=[d.cacc],
    )


def _rne_cfrc(m: mjmtp.Model, d: mjmtp.Data, flg_cfrc_ext: bool = False):
  ax.launch(
    "_cfrc", dim=[d.nworld, m.nbody - 1], inputs=[d.cinert, d.cvel, d.cacc, d.cfrc_ext, flg_cfrc_ext], outputs=[d.cfrc_int]
  )


def _rne_cfrc_backward(m: mjmtp.Model, d: mjmtp.Data):
  for body_tree in reversed(m.body_tree):
    ax.launch(
      "_cfrc_backward", dim=[d.nworld, body_tree.size], inputs=[m.body_parentid, d.cfrc_int, body_tree], outputs=[d.cfrc_int]
    )


def rne(m: mjmtp.Model, d: mjmtp.Data, flg_acc: bool = False):
  """Computes inverse dynamics using the recursive Newton-Euler algorithm.

  Computes the bias forces (`qfrc_bias`) and internal forces (`cfrc_int`) for the current state,
  including the effects of gravity and optionally joint accelerations.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    flg_acc: If True, includes joint accelerations in the computation.
  """
  _rne_cacc_world(m, d)
  _rne_cacc_forward(m, d, flg_acc=flg_acc)
  _rne_cfrc(m, d)
  _rne_cfrc_backward(m, d)
  ax.launch("_qfrc_bias", dim=[d.nworld, m.nv], inputs=[m.dof_bodyid, d.cdof, d.cfrc_int], outputs=[d.qfrc_bias])


def crb(m: mjmtp.Model, d: mjmtp.Data):
  """Computes composite rigid body inertias for each body and the joint-space inertia matrix.

  Accumulates composite rigid body inertias up the kinematic tree and computes the
  joint-space inertia matrix in either sparse or dense format, depending on model options.
  """
  ax.copy(d.crb, d.cinert)

  for i in reversed(range(len(m.body_tree))):
    body_tree = m.body_tree[i]
    ax.launch("_crb_accumulate", dim=(d.nworld, body_tree.size), inputs=[m.body_parentid, d.crb, body_tree], outputs=[d.crb])

  d.qM.zero_()
  if m.opt.is_sparse:
    ax.launch(
      "_qM_sparse",
      dim=(d.nworld, m.nv),
      inputs=[m.dof_bodyid, m.dof_parentid, m.dof_Madr, m.dof_armature, d.cdof, d.crb],
      outputs=[d.qM],
    )
  else:
    ax.launch(
      "_qM_dense", dim=(d.nworld, m.nv), inputs=[m.dof_bodyid, m.dof_parentid, m.dof_armature, d.cdof, d.crb], outputs=[d.qM]
    )


def com_vel(m: mjmtp.Model, d: mjmtp.Data):
  """Computes the spatial velocities (cvel) and the derivative `cdof_dot` for all bodies.

  Propagates velocities down the kinematic tree, updating the spatial velocity and
  derivative for each body.
  """
  ax.launch("_comvel_root", dim=(d.nworld, 6), inputs=[], outputs=[d.cvel])

  for body_tree in m.body_tree:
    ax.launch(
      "_comvel_level",
      dim=(d.nworld, body_tree.size),
      inputs=[m.body_parentid, m.body_jntnum, m.body_jntadr, m.body_dofadr, m.jnt_type, d.qvel, d.cdof, d.cvel, body_tree],
      outputs=[d.cvel, d.cdof_dot],
    )


def subtree_vel(m: mjmtp.Model, d: mjmtp.Data):
  """Computes subtree linear velocity and angular momentum.

  Computes the linear momentum and angular momentum for each subtree, accumulating
  contributions up the kinematic tree.
  """
  # bodywise quantities
  ax.launch(
    "_subtree_vel_forward",
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com, d.cvel],
    outputs=[d.subtree_linvel, d.subtree_angmom, d.subtree_bodyvel],
  )

  # sum body linear momentum recursively up the kinematic tree
  for body_tree in reversed(m.body_tree):
    ax.launch(
      "_linear_momentum",
      dim=[d.nworld, body_tree.size],
      inputs=[m.body_parentid, m.body_subtreemass, d.subtree_linvel, body_tree],
      outputs=[d.subtree_linvel],
    )

  for body_tree in reversed(m.body_tree):
    ax.launch(
      "_angular_momentum",
      dim=[d.nworld, body_tree.size],
      inputs=[
        m.body_parentid,
        m.body_mass,
        m.body_subtreemass,
        d.xipos,
        d.subtree_com,
        d.subtree_linvel,
        d.subtree_bodyvel,
        body_tree,
      ],
      outputs=[d.subtree_angmom],
    )


def kinematics(m: mjmtp.Model, d: mjmtp.Data):
  """Computes forward kinematics for all bodies, sites, geoms, and flexible elements.

  This function updates the global positions and orientations of all bodies, as well as the
  derived positions and orientations of geoms, sites, and flexible elements, based on the
  current joint positions and any attached mocap bodies.
  """
  ax.launch("_kinematics_root", dim=(d.nworld), inputs=[], outputs=[d.xpos, d.xquat, d.xmat, d.xipos, d.ximat])

  for i in range(1, len(m.body_tree)):
    body_tree = m.body_tree[i]
    ax.launch(
      "_kinematics_level",
      dim=(d.nworld, body_tree.size),
      inputs=[
        m.qpos0,
        m.body_parentid,
        m.body_mocapid,
        m.body_jntnum,
        m.body_jntadr,
        m.body_pos,
        m.body_quat,
        m.body_ipos,
        m.body_iquat,
        m.jnt_type,
        m.jnt_qposadr,
        m.jnt_pos,
        m.jnt_axis,
        d.qpos,
        d.mocap_pos,
        d.mocap_quat,
        d.xpos,
        d.xquat,
        d.xmat,
        body_tree,
      ],
      outputs=[d.xpos, d.xquat, d.xmat, d.xipos, d.ximat, d.xanchor, d.xaxis],
    )

  ax.launch(
    "_geom_local_to_global",
    dim=(d.nworld, m.ngeom),
    inputs=[m.body_rootid, m.body_weldid, m.body_mocapid, m.geom_bodyid, m.geom_pos, m.geom_quat, d.xpos, d.xquat],
    outputs=[d.geom_xpos, d.geom_xmat],
  )

  ax.launch(
    "_site_local_to_global",
    dim=(d.nworld, m.nsite),
    inputs=[m.site_bodyid, m.site_pos, m.site_quat, d.xpos, d.xquat],
    outputs=[d.site_xpos, d.site_xmat],
  )


def transmission(m: mjmtp.Model, d: mjmtp.Data):
  """Computes actuator/transmission lengths and moments.

  Updates the actuator length and moments for all actuators in the model, including joint
  and tendon transmissions.
  """
  ax.launch(
    "_transmission",
    dim=[d.nworld, m.nu],
    inputs=[
      m.nv,
      m.body_parentid,
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.dof_bodyid,
      m.dof_parentid,
      m.site_bodyid,
      m.site_quat,
      m.tendon_adr,
      m.tendon_num,
      m.wrap_type,
      m.wrap_objid,
      m.actuator_trntype,
      m.actuator_trnid,
      m.actuator_gear,
      m.actuator_cranklength,
      d.qpos,
      d.xquat,
      d.site_xpos,
      d.site_xmat,
      d.subtree_com,
      d.cdof,
      d.ten_J,
      d.ten_length,
    ],
    outputs=[d.actuator_length, d.actuator_moment],
  )

  if m.nacttrnbody:
    # compute moments
    ncon = get_cached_array('transmission_''ncon', (d.nworld, m.nacttrnbody), dtype=int)
    ncon.zero_()
    ax.launch(
      "_transmission_body_moment",
      dim=(m.nacttrnbody, d.naconmax, m.nv),
      inputs=[
        m.opt.cone,
        m.body_parentid,
        m.body_rootid,
        m.dof_bodyid,
        m.geom_bodyid,
        m.actuator_trnid,
        m.actuator_trntype_body_adr,
        d.subtree_com,
        d.cdof,
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.includemargin,
        d.contact.dim,
        d.contact.geom,
        d.contact.efc_address,
        d.contact.worldid,
        d.efc.J,
        d.nacon,
      ],
      outputs=[d.actuator_moment, ncon],
    )

    # scale moments
    ax.launch(
      "_transmission_body_moment_scale",
      dim=(d.nworld, m.nacttrnbody, m.nv),
      inputs=[m.actuator_trntype_body_adr, ncon],
      outputs=[d.actuator_moment],
    )


def tendon(m: mjmtp.Model, d: mjmtp.Data):
  """Computes tendon lengths and moments.

  Updates the tendon length and moment arrays for all tendons in the model, including joint,
  site, and geom tendons.
  """
  if not m.ntendon:
    return

  d.ten_length.zero_()
  d.ten_J.zero_()

  # Cartesian 3D points fro geom wrap points
  wrap_geom_xpos = get_cached_array('tendon_''wrap_geom_xpos', (d.nworld, m.nwrap), dtype=ax.spatial_vector)

  # process joint tendons
  ax.launch(
    "_joint_tendon",
    dim=(d.nworld, m.wrap_jnt_adr.size),
    inputs=[m.jnt_qposadr, m.jnt_dofadr, m.wrap_objid, m.wrap_prm, m.tendon_jnt_adr, m.wrap_jnt_adr, d.qpos],
    outputs=[d.ten_J, d.ten_length],
  )

  spatial_site = m.wrap_site_pair_adr.size > 0
  spatial_geom = m.wrap_geom_adr.size > 0

  if spatial_site or spatial_geom:
    d.wrap_xpos.zero_()
    d.wrap_obj.zero_()

  # process spatial site tendons
  ax.launch(
    "_spatial_site_tendon",
    dim=(d.nworld, m.wrap_site_pair_adr.size),
    inputs=[
      m.nv,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.site_bodyid,
      m.wrap_objid,
      m.tendon_site_pair_adr,
      m.wrap_site_pair_adr,
      m.wrap_pulley_scale,
      d.site_xpos,
      d.subtree_com,
      d.cdof,
    ],
    outputs=[d.ten_J, d.ten_length],
  )

  # process spatial geom tendons
  ax.launch(
    "_spatial_geom_tendon",
    dim=(d.nworld, m.wrap_geom_adr.size),
    inputs=[
      m.nv,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.geom_bodyid,
      m.geom_size,
      m.site_bodyid,
      m.wrap_type,
      m.wrap_objid,
      m.wrap_prm,
      m.tendon_geom_adr,
      m.wrap_geom_adr,
      m.wrap_pulley_scale,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.subtree_com,
      d.cdof,
    ],
    outputs=[d.ten_J, d.ten_length, wrap_geom_xpos],
  )

  if spatial_site or spatial_geom:
    ax.launch(
      "_spatial_tendon_wrap",
      dim=d.nworld,
      inputs=[m.ntendon, m.tendon_adr, m.tendon_num, m.wrap_type, m.wrap_objid, d.site_xpos, wrap_geom_xpos],
      outputs=[d.ten_wrapadr, d.ten_wrapnum, d.wrap_obj, d.wrap_xpos],
    )


def com_pos(m: mjmtp.Model, d: mjmtp.Data):
  """Computes subtree center of mass positions.

  Transforms inertia and motion to global frame centered at subtree CoM. Accumulates the
  mass-weighted positions up the kinematic tree, divides by total mass, and computes composite
  inertias and motion degrees of freedom in the subtree CoM frame.
  """
  ax.launch("_subtree_com_init", dim=(d.nworld, m.nbody), inputs=[m.body_mass, d.xipos], outputs=[d.subtree_com])

  for i in reversed(range(len(m.body_tree))):
    body_tree = m.body_tree[i]
    ax.launch(
      "_subtree_com_acc",
      dim=(d.nworld, body_tree.size),
      inputs=[m.body_parentid, d.subtree_com, body_tree],
      outputs=[d.subtree_com],
    )

  ax.launch("_subtree_div", dim=(d.nworld, m.nbody), inputs=[m.body_subtreemass, d.subtree_com], outputs=[d.subtree_com])
  ax.launch(
    "_cinert",
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
    outputs=[d.cinert],
  )
  ax.launch(
    "_cdof",
    dim=(d.nworld, m.njnt),
    inputs=[m.body_rootid, m.jnt_type, m.jnt_dofadr, m.jnt_bodyid, d.xmat, d.xanchor, d.xaxis, d.subtree_com],
    outputs=[d.cdof],
  )


def camlight(m: mjmtp.Model, d: mjmtp.Data):
  ax.launch(
    "_cam_local_to_global",
    dim=(d.nworld, m.ncam),
    inputs=[
      m.cam_mode,
      m.cam_bodyid,
      m.cam_targetbodyid,
      m.cam_pos,
      m.cam_quat,
      m.cam_poscom0,
      m.cam_pos0,
      m.cam_mat0,
      d.xpos,
      d.xquat,
      d.subtree_com,
    ],
    outputs=[d.cam_xpos, d.cam_xmat],
  )

  ax.launch(
    "_light_local_to_global",
    dim=(d.nworld, m.nlight),
    inputs=[
      m.light_mode,
      m.light_bodyid,
      m.light_targetbodyid,
      m.light_pos,
      m.light_dir,
      m.light_poscom0,
      m.light_pos0,
      m.light_dir0,
      d.xpos,
      d.xquat,
      d.subtree_com,
    ],
    outputs=[d.light_xpos, d.light_xdir],
  )


def flex(m: mjmtp.Model, d: mjmtp.Data):
  ax.launch("_flex_vertices", dim=(d.nworld, m.nflexvert), inputs=[m.flex_vertbodyid, d.xpos], outputs=[d.flexvert_xpos])

  ax.launch(
    "_flex_edges",
    dim=(d.nworld, m.nflexedge),
    inputs=[
      m.nv,
      m.body_parentid,
      m.body_rootid,
      m.body_dofadr,
      m.dof_bodyid,
      m.flex_vertadr,
      m.flex_vertbodyid,
      m.flex_edge,
      d.qvel,
      d.subtree_com,
      d.cdof,
      d.flexvert_xpos,
    ],
    outputs=[d.flexedge_J, d.flexedge_length, d.flexedge_velocity],
  )


def tendon_armature(m: mjmtp.Model, d: mjmtp.Data):
  ax.launch(
    "_tendon_armature",
    dim=(d.nworld, m.ntendon, m.nv),
    inputs=[m.opt.is_sparse, m.dof_parentid, m.dof_Madr, m.tendon_armature, d.ten_J],
    outputs=[d.qM],
  )


def tendon_bias(m: mjmtp.Model, d: mjmtp.Data, qfrc: ax.array2d(dtype=float)): # type: ignore
  """Add bias force due to tendon armature.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    qfrc: Force.
  """
  # time derivative of tendon Jacobian
  ten_Jdot = get_cached_array('tendon_bias_''ten_Jdot', (d.nworld, m.ntendon, m.nv), dtype=float)
  ten_Jdot.zero_()
  ax.launch(
    "_tendon_dot",
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.nv,
      m.body_parentid,
      m.body_rootid,
      m.jnt_type,
      m.jnt_dofadr,
      m.dof_bodyid,
      m.dof_jntid,
      m.site_bodyid,
      m.tendon_adr,
      m.tendon_num,
      m.tendon_armature,
      m.wrap_type,
      m.wrap_objid,
      m.wrap_prm,
      d.site_xpos,
      d.subtree_com,
      d.cdof,
      d.cvel,
      d.cdof_dot,
    ],
    outputs=[ten_Jdot],
  )

  # tendon bias force coefficients
  ten_bias_coef = get_cached_array('tendon_bias_''ten_bias_coef', (d.nworld, m.ntendon), dtype=float)
  ten_bias_coef.zero_()
  ax.launch(
    "_tendon_bias_coef",
    dim=(d.nworld, m.ntendon, m.nv),
    inputs=[m.tendon_armature, d.qvel, ten_Jdot],
    outputs=[ten_bias_coef],
  )

  ax.launch(
    "_tendon_bias_qfrc",
    dim=(d.nworld, m.ntendon, m.nv),
    inputs=[m.tendon_armature, d.ten_J, ten_bias_coef],
    outputs=[qfrc],
  )


def rne_postconstraint(m: mjmtp.Model, d: mjmtp.Data):
  """Computes the recursive Newton-Euler algorithm after constraints are applied.

  Computes `cacc`, `cfrc_ext`, and `cfrc_int`, including the effects of applied forces, equality
  constraints, and contacts.
  """
  ax.launch(
    "_cfrc_ext",
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, d.xfrc_applied, d.xipos, d.subtree_com],
    outputs=[d.cfrc_ext],
  )

  ax.launch(
    "_cfrc_ext_equality",
    dim=(d.nworld, m.neq),
    inputs=[
      m.body_rootid,
      m.site_bodyid,
      m.site_pos,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.eq_data,
      d.xpos,
      d.xmat,
      d.subtree_com,
      d.efc.id,
      d.efc.force,
      d.ne_connect,
      d.ne_weld,
    ],
    outputs=[d.cfrc_ext],
  )

  ax.launch(
    "_cfrc_ext_contact",
    dim=(d.naconmax,),
    inputs=[
      m.opt.cone,
      m.body_rootid,
      m.geom_bodyid,
      d.subtree_com,
      d.contact.pos,
      d.contact.frame,
      d.contact.friction,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.worldid,
      d.efc.force,
      d.njmax,
      d.nacon,
    ],
    outputs=[d.cfrc_ext],
  )

  _rne_cacc_world(m, d)
  _rne_cacc_forward(m, d, flg_acc=True)
  _rne_cfrc(m, d, flg_cfrc_ext=True)
  _rne_cfrc_backward(m, d)
