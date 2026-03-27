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
from .collision_convex import convex_narrowphase
from .collision_primitive import primitive_narrowphase
from .collision_sdf import sdf_narrowphase
from .types import BroadphaseType
from .types import DisableBit


def _zero_nacon_ncollision(m: mjmtp.Model, d: mjmtp.Data):
  ax.launch("_zero_nacon_ncollision", dim=1, outputs=[d.nacon, d.ncollision])


def nxn_broadphase(m: mjmtp.Model, d: mjmtp.Data):
  ax.launch("_nxn_broadphase__kernel",dim=(d.nworld, m.nxn_geom_pair_filtered.shape[0]),
    inputs=[
      m.geom_type,
      m.geom_aabb,
      m.geom_rbound,
      m.geom_margin,
      m.nxn_geom_pair_filtered,
      m.nxn_pairid_filtered,
      d.naconmax,
      d.geom_xpos,
      d.geom_xmat,
      m.opt.broadphase_filter,
      m.geom_aabb.shape[0],
      m.geom_margin.shape[0],
      m.geom_rbound.shape[0],
    ],
    outputs=[
      d.collision_pair,
      d.collision_pairid,
      d.collision_worldid,
      d.ncollision,
    ],)

  #copy_wp_array_batch_attrlist(d, d_, outputs)


def _narrowphase(m, d):
  convex_narrowphase(m, d)
  primitive_narrowphase(m, d)

  if m.has_sdf_geom:
    sdf_narrowphase(m, d)


def collision(m: mjmtp.Model, d: mjmtp.Data):
  _zero_nacon_ncollision(m,d)

  if d.naconmax == 0 or m.opt.disableflags & (DisableBit.CONSTRAINT | DisableBit.CONTACT):
    return

  if m.opt.broadphase == BroadphaseType.NXN:
    nxn_broadphase(m, d)
  else:
    # print("WARN: MUSA sap_broadphase is not implemented yet.Using nxn_broadphase")
    # nxn_broadphase(m, d)
    sap_broadphase(m, d)

  _narrowphase(m, d)


def sap_broadphase(m: mjmtp.Model, d: mjmtp.Data):
  nworldgeom_musa = d.nworld * m.ngeom

  # TODO(team): direction
  # random fixed direction
  # direction = wp.normalize(wp.vec3(0.5935, 0.7790, 0.1235))
  direction_musa = ax.vec3([0.6012657880783081, 0.7891929149627686, 0.12511596083641052])

  projection_lower_musa = get_cached_array('sap_broadphase_''projection_lower_musa', (d.nworld, m.ngeom, 2), dtype=float)
  projection_upper_musa = get_cached_array('sap_broadphase_''projection_upper_musa', (d.nworld, m.ngeom), dtype=float)
  sort_index_musa = get_cached_array('sap_broadphase_''sort_index_musa', (d.nworld, m.ngeom, 2), dtype=int)
  range_musa = get_cached_array('sap_broadphase_''range_musa', (d.nworld, m.ngeom), dtype=int)
  cumulative_sum_musa = get_cached_array('sap_broadphase_''cumulative_sum_musa', (d.nworld, m.ngeom), dtype=int)
  segmented_index_musa = get_cached_array('sap_broadphase_''segmented_index_musa', d.nworld + 1 if m.opt.broadphase == BroadphaseType.SAP_SEGMENTED or m.opt.broadphase == BroadphaseType.SAP_TILE else 0, dtype=int)

  ax.launch("_sap_project__sap_project",
    dim=(d.nworld, m.ngeom),
    inputs=[m.ngeom, m.geom_rbound, m.geom_margin, d.nworld, d.geom_xpos, direction_musa, m.opt.broadphase],
    outputs=[
      projection_lower_musa.reshape((-1, m.ngeom)),
      projection_upper_musa,
      sort_index_musa.reshape((-1, m.ngeom)),
      segmented_index_musa,
    ],
  )

  # if m.opt.broadphase == BroadphaseType.SAP_TILE:
  #   from ..collision_driver import _segmented_sort
  #   projection_lower = wp.empty((d.nworld, m.ngeom, 2), dtype=float)
  #   sort_index = wp.empty((d.nworld, m.ngeom, 2), dtype=int)
  #   axinfra.copy_wp_array(projection_lower, projection_lower_musa)
  #   axinfra.copy_wp_array(sort_index, sort_index_musa)

  #   wp.launch_tiled(
  #     kernel=_segmented_sort(m.ngeom),
  #     dim=d.nworld,
  #     inputs=[projection_lower.reshape((-1, m.ngeom)), sort_index.reshape((-1, m.ngeom))],
  #     outputs=[projection_lower.reshape((-1, m.ngeom)), sort_index.reshape((-1, m.ngeom))],
  #     block_dim=m.block_dim.segmented_sort,
  #   )
  #   axinfra.copy_wp_array(projection_lower_musa, projection_lower)
  #   axinfra.copy_wp_array(sort_index_musa, sort_index)
  # else:

  ax.utils.segmented_sort_pairs(
    projection_lower_musa.reshape((-1, m.ngeom)), sort_index_musa.reshape((-1, m.ngeom)), nworldgeom_musa, segmented_index_musa
  )

  ax.launch("_sap_range",
    dim=(d.nworld, m.ngeom),
    inputs=[m.ngeom, projection_lower_musa.reshape((-1, m.ngeom)), projection_upper_musa, sort_index_musa.reshape((-1, m.ngeom))],
    outputs=[range_musa],
  )

  # scan is used for load balancing among the threads
  ax.utils.array_scan(range_musa.reshape(-1), cumulative_sum_musa.reshape(-1), True)

  # estimate number of overlap checks
  # assumes each geom has 5 other geoms (batched over all worlds)

  nsweep_musa = 5 * nworldgeom_musa
  ax.launch(
    "_sap_broadphase__kernel",
    dim=nsweep_musa,
    inputs=[
      m.ngeom,
      m.geom_type,
      m.geom_aabb,
      m.geom_rbound,
      m.geom_margin,
      m.nxn_pairid,
      d.nworld,
      d.naconmax,
      d.geom_xpos,
      d.geom_xmat,
      sort_index_musa.reshape((-1, m.ngeom)),
      cumulative_sum_musa.reshape(-1),
      nsweep_musa,
      m.geom_aabb.shape[0],
      m.geom_margin.shape[0],
      m.geom_rbound.shape[0],
      m.opt.broadphase_filter,
    ],
    outputs=[d.collision_pair, d.collision_pairid, d.collision_worldid, d.ncollision],
  )

