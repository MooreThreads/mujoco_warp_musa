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

from . import math
from . import types as mjmtp
from .cached_array import get_cached_array
from .types import MJ_MAX_EPAFACES
from .types import MJ_MAX_EPAHORIZON
from .types import GeomType

_CONVEX_COLLISION_PAIRS = [
  (GeomType.HFIELD, GeomType.SPHERE),
  (GeomType.HFIELD, GeomType.CAPSULE),
  (GeomType.HFIELD, GeomType.ELLIPSOID),
  (GeomType.HFIELD, GeomType.CYLINDER),
  (GeomType.HFIELD, GeomType.BOX),
  (GeomType.HFIELD, GeomType.MESH),
  (GeomType.SPHERE, GeomType.ELLIPSOID),
  (GeomType.SPHERE, GeomType.MESH),
  (GeomType.CAPSULE, GeomType.ELLIPSOID),
  (GeomType.CAPSULE, GeomType.CYLINDER),
  (GeomType.CAPSULE, GeomType.MESH),
  (GeomType.ELLIPSOID, GeomType.ELLIPSOID),
  (GeomType.ELLIPSOID, GeomType.CYLINDER),
  (GeomType.ELLIPSOID, GeomType.BOX),
  (GeomType.ELLIPSOID, GeomType.MESH),
  (GeomType.CYLINDER, GeomType.CYLINDER),
  (GeomType.CYLINDER, GeomType.BOX),
  (GeomType.CYLINDER, GeomType.MESH),
  (GeomType.BOX, GeomType.MESH),
  (GeomType.MESH, GeomType.MESH),
]


def convex_narrowphase(m: mjmtp.Model, d: mjmtp.Data):
    # m_.geom_pair_type_count = m.geom_pair_type_count
    # d_.naconmax = d.naconmax
    # m_.opt.ccd_iterations = m.opt.ccd_iterations
    if not any(m.geom_pair_type_count[math.upper_trid_index(len(GeomType),g[0].value, g[1].value)] for g in _CONVEX_COLLISION_PAIRS):
        return

    # set to true to enable multiccd
    use_multiccd = False
    # nmaxpolygon = m.nmaxpolygon if use_multiccd else 0
    # nmaxmeshdeg = m.nmaxmeshdeg if use_multiccd else 0

    # epa_vert: vertices in EPA polytope in Minkowski space
    epa_vert_shape = (d.naconmax, 5 + m.opt.ccd_iterations)
    epa_vert = get_cached_array('collision_convex_''epa_vert', epa_vert_shape, ax.vec3)
    # epa_vert1: vertices in EPA polytope in geom 1 space
    epa_vert1 = get_cached_array('collision_convex_''epa_vert1', epa_vert_shape, ax.vec3)
    # epa_vert2: vertices in EPA polytope in geom 2 space
    epa_vert2 = get_cached_array('collision_convex_''epa_vert2', epa_vert_shape, ax.vec3)
    # epa_vert_index1: vertex indices in EPA polytope for geom 1
    epa_vert_index1 = get_cached_array('collision_convex_''epa_vert_index1', epa_vert_shape, int)
    # epa_vert_index2: vertex indices in EPA polytope for geom 2  (naconmax, 5 + CCDiter)
    epa_vert_index2 = get_cached_array('collision_convex_''epa_vert_index2', epa_vert_shape, int)
    # epa_face: faces of polytope represented by three indices
    epa_face_shape = (d.naconmax, 6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations)
    epa_face = get_cached_array('collision_convex_''epa_face', epa_face_shape, ax.vec3i)
    # epa_pr: projection of origin on polytope faces
    epa_pr = get_cached_array('collision_convex_''epa_pr', epa_face_shape, ax.vec3)
    # epa_norm2: epa_pr * epa_pr
    epa_norm2 = get_cached_array('collision_convex_''epa_norm2', epa_face_shape, float)
    # epa_index: index of face in polytope map
    epa_index = get_cached_array('collision_convex_''epa_index', epa_face_shape, int)
    # epa_map: status of faces in polytope
    epa_map = get_cached_array('collision_convex_''epa_map', epa_face_shape, int)
    # epa_horizon: index pair (i j) of edges on horizon
    epa_horizon_shape = (d.naconmax, 2 * MJ_MAX_EPAHORIZON)
    epa_horizon = get_cached_array('collision_convex_''epa_horizon', epa_horizon_shape, int)
    # # multiccd_polygon: clipped contact surface
    # multiccd_polygon_shape = (d.naconmax, 2 * nmaxpolygon)
    # multiccd_polygon = get_cached_array('collision_convex_''multiccd_polygon', multiccd_polygon_shape, ax.vec3)
    # # multiccd_clipped: clipped contact surface (intermediate)
    # multiccd_clipped = get_cached_array('collision_convex_''multiccd_clipped', multiccd_polygon_shape, ax.vec3)
    # # multiccd_pnormal: plane normal of clipping polygon
    # multiccd_pnormal_shape = (d.naconmax, nmaxpolygon)
    # multiccd_pnormal = get_cached_array('collision_convex_''multiccd_pnormal', multiccd_pnormal_shape, ax.vec3)
    # # multiccd_pdist: plane distance of clipping polygon
    # multiccd_pdist = get_cached_array('collision_convex_''multiccd_pdist', multiccd_pnormal_shape, float)
    # # multiccd_idx1: list of normal index candidates for Geom 1
    # multiccd_idx1_shape = (d.naconmax, nmaxmeshdeg)
    # multiccd_idx1 = get_cached_array('collision_convex_''multiccd_idx1', multiccd_idx1_shape, int)
    # # multiccd_idx2: list of normal index candidates for Geom 2
    # multiccd_idx2 = get_cached_array('collision_convex_''multiccd_idx2', multiccd_idx1_shape, int)
    # # multiccd_n1: list of normal candidates for Geom 1
    # multiccd_n1 = get_cached_array('collision_convex_''multiccd_n1', multiccd_idx1_shape, ax.vec3)
    # # multiccd_n2: list of normal candidates for Geom 1
    # multiccd_n2 = get_cached_array('collision_convex_''multiccd_n2', multiccd_idx1_shape, ax.vec3)
    # # multiccd_endvert: list of edge vertices candidates
    # multiccd_endvert = get_cached_array('collision_convex_''multiccd_endvert', multiccd_idx1_shape, ax.vec3)
    # # multiccd_face1: contact face
    # multiccd_face1 = get_cached_array('collision_convex_''multiccd_face1', multiccd_pnormal_shape, ax.vec3)
    # # multiccd_face2: contact face
    # multiccd_face2 = get_cached_array('collision_convex_''multiccd_face2', multiccd_pnormal_shape, ax.vec3)

    # MAXCONPAIR = 500
    # hfield_contact_shape = (d.naconmax, MAXCONPAIR)
    # hfield_contact_dist_in = get_cached_array('collision_convex_''hfield_contact_dist_in', hfield_contact_shape, float)
    # hfield_contact_pos_in = get_cached_array('collision_convex_''hfield_contact_pos_in', hfield_contact_shape, ax.vec3)
    # hfield_contact_normal_in = get_cached_array('collision_convex_''hfield_contact_normal_in', hfield_contact_shape, ax.vec3)

    for geom_pair in _CONVEX_COLLISION_PAIRS:
      g1 = geom_pair[0].value
      g2 = geom_pair[1].value
      if m.geom_pair_type_count[math.upper_trid_index(len(GeomType),g1,g2)]:
        ax.launch("ccd_kernel_builder__ccd_kernel",
        dim=d.naconmax,
        inputs=[
          m.opt.ccd_tolerance,
          m.geom_type,
          m.geom_condim,
          m.geom_dataid,
          m.geom_priority,
          m.geom_solmix,
          m.geom_solref,
          m.geom_solimp,
          m.geom_size,
          m.geom_aabb,
          m.geom_rbound,
          m.geom_friction,
          m.geom_margin,
          m.geom_gap,
          m.hfield_adr,
          m.hfield_nrow,
          m.hfield_ncol,
          m.hfield_size,
          m.hfield_data,
          m.mesh_vertadr,
          m.mesh_vertnum,
          m.mesh_vert,
          m.mesh_graphadr,
          m.mesh_graph,
          m.mesh_polynum,
          m.mesh_polyadr,
          m.mesh_polynormal,
          m.mesh_polyvertadr,
          m.mesh_polyvertnum,
          m.mesh_polyvert,
          m.mesh_polymapadr,
          m.mesh_polymapnum,
          m.mesh_polymap,
          m.pair_dim,
          m.pair_solref,
          m.pair_solreffriction,
          m.pair_solimp,
          m.pair_margin,
          m.pair_gap,
          m.pair_friction,
          d.naconmax,
          d.geom_xpos,
          d.geom_xmat,
          d.collision_pair,
          d.collision_pairid,
          d.collision_worldid,
          d.ncollision,
          epa_vert,
          epa_vert1,
          epa_vert2,
          epa_vert_index1,
          epa_vert_index2,
          epa_face,
          epa_pr,
          epa_norm2,
          epa_index,
          epa_map,
          epa_horizon,
          # multiccd_polygon,
          # multiccd_clipped,
          # multiccd_pnormal,
          # multiccd_pdist,
          # multiccd_idx1,
          # multiccd_idx2,
          # multiccd_n1,
          # multiccd_n2,
          # multiccd_endvert,
          # multiccd_face1,
          # multiccd_face2,
          m.opt.ccd_iterations,
          g1,
          g2,
          g1 == GeomType.HFIELD,
          use_multiccd,
          # hfield_contact_dist_in,
          # hfield_contact_pos_in,
          # hfield_contact_normal_in,
        ],
        outputs=[
          d.nacon,
          d.contact.dist,
          d.contact.pos,
          d.contact.frame,
          d.contact.includemargin,
          d.contact.friction,
          d.contact.solref,
          d.contact.solreffriction,
          d.contact.solimp,
          d.contact.dim,
          d.contact.geom,
          d.contact.worldid,
          d.contact.type,
          d.contact.geomcollisionid,
        ],)

