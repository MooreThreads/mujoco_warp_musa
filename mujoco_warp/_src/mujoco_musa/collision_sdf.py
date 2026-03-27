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


def sdf_narrowphase(m: mjmtp.Model, d: mjmtp.Data):
    ax.launch("_sdf_narrowphase",dim=(m.opt.sdf_initpoints, d.naconmax),
    inputs=[
      m.nmeshface,
      m.oct_child,
      m.oct_aabb,
      m.oct_coeff,
      m.geom_type,
      m.geom_condim,
      m.geom_dataid,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_aabb,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_faceadr,
      m.mesh_graphadr,
      m.mesh_vert,
      m.mesh_face,
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
      m.plugin,
      m.plugin_attr,
      m.geom_plugin_index,
      d.geom_xpos,
      d.geom_xmat,
      d.naconmax,
      d.collision_pair,
      d.collision_pairid,
      d.collision_worldid,
      d.ncollision,
      m.opt.sdf_initpoints,
      m.opt.sdf_iterations,
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