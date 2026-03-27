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
from .cached_array import make_array_cache
from .types import GeomType

_PRIMITIVE_COLLISIONS = [
  (GeomType.PLANE, GeomType.SPHERE),
  (GeomType.PLANE, GeomType.CAPSULE),
  (GeomType.PLANE, GeomType.ELLIPSOID),
  (GeomType.PLANE, GeomType.CYLINDER),
  (GeomType.PLANE, GeomType.BOX),
  (GeomType.PLANE, GeomType.MESH),
  (GeomType.SPHERE, GeomType.SPHERE),
  (GeomType.SPHERE, GeomType.CAPSULE),
  (GeomType.SPHERE, GeomType.CYLINDER),
  (GeomType.SPHERE, GeomType.BOX),
  (GeomType.CAPSULE, GeomType.CAPSULE),
  (GeomType.CAPSULE, GeomType.BOX),
  (GeomType.BOX, GeomType.BOX),
]

cached_geom_pair_type_count = None

def primitive_narrowphase(m: mjmtp.Model, d: mjmtp.Data):
  global cached_geom_pair_type_count
  if m.geom_pair_type_count != cached_geom_pair_type_count:
    cached_geom_pair_type_count = m.geom_pair_type_count
    _primitive_collisions_types = []
    for types in _PRIMITIVE_COLLISIONS:
      idx = math.upper_trid_index(len(GeomType), types[0].value, types[1].value)
      if m.geom_pair_type_count[idx] and types not in _primitive_collisions_types:
        _primitive_collisions_types.append(types[0].value)
        _primitive_collisions_types.append(types[1].value)
    make_array_cache("primitive_narrowphase_""_primitive_collisions_types_warp", ax.array(_primitive_collisions_types, dtype=int))

  _primitive_collisions_types_warp = get_cached_array("primitive_narrowphase_""_primitive_collisions_types_warp")

  ax.launch("_create_narrowphase_kernel___primitive_narrowphase",dim=d.naconmax,
  inputs=[
    m.geom_type,
    m.geom_condim,
    m.geom_dataid,
    m.geom_priority,
    m.geom_solmix,
    m.geom_solref,
    m.geom_solimp,
    m.geom_size,
    m.geom_friction,
    m.geom_margin,
    m.geom_gap,
    m.mesh_vertadr,
    m.mesh_vertnum,
    m.mesh_graphadr,
    m.mesh_vert,
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
    d.geom_xpos,
    d.geom_xmat,
    d.naconmax,
    d.collision_pair,
    d.collision_pairid,
    d.collision_worldid,
    d.ncollision,
    _primitive_collisions_types_warp
  ],
  outputs=[
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
    d.nacon,
  ],)
