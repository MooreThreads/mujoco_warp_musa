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

from ..solver import *
from ..solver import _linesearch
from ..solver import _update_constraint
from ..solver import _update_gradient


def _linesearch_copydata(m: Model, d: Data, cost: wp.array2d(dtype=float)): # type: ignore
  m_ = m.musa_model
  d_ = d.musa_data

  inputs = [
    "qfrc_smooth",
    "nacon",
    "qacc",
    "ne",
    "nf",
    "nefc",
  ]
  inputs_efc = [
    "Ma",
    "search",
    "gauss",
    "mv",
    "done",
    "type",
    "id",
    "D",
    "Jaref",
    "jv",
    "alpha",
    "frictionloss",
    "search_dot",
    "quad",
    "quad_gauss",
  ]
  inputs_contact = [
    "friction",
    "dim",
    "efc_address",
  ]

  outputs = ["qacc"]
  outputs_efc = [
    "quad_gauss",
    "quad",
    "Ma",
    "Jaref",
    "mv",
    "jv",
    "alpha",
  ]

  copy_wp_array_batch_attrlist(d_, d, inputs)
  copy_wp_array_batch_attrlist(d_.efc, d.efc, inputs_efc)
  copy_wp_array_batch_attrlist(d_.contact, d.contact, inputs_contact)
  cost_ = ax.from_wp_array(cost)

  _linesearch(m_, d_, cost_)

  copy_wp_array_batch_attrlist(d, d_, outputs)
  copy_wp_array_batch_attrlist(d.efc, d_.efc, outputs_efc)
  ax.copy_wp_array(cost, cost_)


def _update_constraint_copydata(m: Model, d: Data):
  inputs =[
    # contact
    "contact.friction",
    "contact.dist",
    "contact.dim",
    "contact.efc_address",
    "contact.includemargin",
    "contact.efc_address",
    "contact.worldid",
    # efc
    "efc.J",
    "efc.aref",
    "efc.done",
    "efc.cost",
    "efc.type",
    "efc.id",
    "efc.D",
    "efc.frictionloss",
    "efc.Jaref",
    "efc.force",
    "efc.Ma",
    "efc.state",
    "efc.grad",
    "efc.h",
    "efc.cholesky_L_tmp",
    "efc.cholesky_y_tmp",
    "efc.Mgrad",
    #
    "nefc",
    "qacc",
    "qM",
    "qacc",
    "ne",
    "nf",
    "nefc",
    "nacon",
    "qfrc_smooth",
    "qacc_smooth",
    "qfrc_constraint",
    "qLD",
    "qLDiagInv"
  ]

  outputs = [
    "solver_niter",
    "efc.search_dot",
    "efc.cost",
    "efc.done",
    "efc.Jaref",
    "efc.Ma",
    "efc.gauss",
    "efc.prev_cost",
    "efc.force",
    "efc.state",
    "qfrc_constraint",
    "efc.grad_dot",
    "efc.grad",
    "efc.h",
    "efc.Mgrad"
  ]

  m_ = m.musa_model
  d_ = d.musa_data

  copy_wp_array_batch_attrlist(d_, d, inputs)
  copy_wp_array_batch_attrlist(d_, d, outputs)

  _update_constraint(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def _update_gradient_copydata(m: Model, d: Data):
  inputs =[
    # contact
    "contact.friction",
    "contact.dist",
    "contact.dim",
    "contact.efc_address",
    "contact.includemargin",
    "contact.efc_address",
    "contact.worldid",
    # efc
    "efc.J",
    "efc.aref",
    "efc.done",
    "efc.cost",
    "efc.type",
    "efc.id",
    "efc.D",
    "efc.frictionloss",
    "efc.Jaref",
    "efc.force",
    "efc.Ma",
    "efc.state",
    "efc.grad",
    "efc.h",
    "efc.cholesky_L_tmp",
    "efc.cholesky_y_tmp",
    "efc.Mgrad",
    #
    "nefc",
    "qacc",
    "qM",
    "qacc",
    "ne",
    "nf",
    "nefc",
    "nacon",
    "qfrc_smooth",
    "qacc_smooth",
    "qfrc_constraint",
    "qLD",
    "qLDiagInv"
  ]

  outputs = [
    "solver_niter",
    "efc.search_dot",
    "efc.cost",
    "efc.done",
    "efc.Jaref",
    "efc.Ma",
    "efc.gauss",
    "efc.prev_cost",
    "efc.force",
    "efc.state",
    "qfrc_constraint",
    "efc.grad_dot",
    "efc.grad",
    "efc.h",
    "efc.Mgrad"
  ]

  m_ = m.musa_model
  d_ = d.musa_data

  copy_wp_array_batch_attrlist(d_, d, inputs)
  copy_wp_array_batch_attrlist(d_, d, outputs)

  _update_gradient(m_, d_, m, d)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def create_context_copydata(m: Model, d: Data, grad: bool = True):
  inputs =[
    # contact
    "contact.friction",
    "contact.dist",
    "contact.dim",
    "contact.efc_address",
    "contact.includemargin",
    "contact.efc_address",
    "contact.worldid",
    # efc
    "efc.J",
    "efc.aref",
    "efc.done",
    "efc.cost",
    "efc.type",
    "efc.id",
    "efc.D",
    "efc.frictionloss",
    "efc.Jaref",
    "efc.force",
    "efc.Ma",
    "efc.state",
    "efc.grad",
    "efc.h",
    "efc.cholesky_L_tmp",
    "efc.cholesky_y_tmp",
    "efc.Mgrad",
    #
    "nefc",
    "qacc",
    "qM",
    "qacc",
    "ne",
    "nf",
    "nefc",
    "nacon",
    "qfrc_smooth",
    "qacc_smooth",
    "qfrc_constraint",
    "qLD",
    "qLDiagInv"
  ]

  outputs = [
    "solver_niter",
    "efc.search_dot",
    "efc.cost",
    "efc.done",
    "efc.Jaref",
    "efc.Ma",
    "efc.gauss",
    "efc.prev_cost",
    "efc.force",
    "efc.state",
    "qfrc_constraint",
    "efc.grad_dot",
    "efc.grad",
    "efc.h",
    "efc.Mgrad"
  ]

  m_ = m.musa_model
  d_ = d.musa_data

  copy_wp_array_batch_attrlist(d_, d, inputs)
  copy_wp_array_batch_attrlist(d_, d, outputs)

  create_context(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs)


def solve_copydata(m: Model, d: Data):
  inputs_0 = [
    "qacc",
    "qacc_smooth",
    "qacc_warmstart",
    "solver_niter",
    "nsolving",
    "qfrc_smooth",
    "nacon",
    "qacc",
    "ne",
    "nf",
    "nefc",
    # efc
    "efc.Ma",
    "efc.search",
    "efc.gauss",
    "efc.mv",
    "efc.done",
    "efc.type",
    "efc.id",
    "efc.D",
    "efc.Jaref",
    "efc.jv",
    "efc.alpha",
    "efc.frictionloss",
    "efc.search_dot",
    "efc.quad",
    "efc.quad_gauss",
    "efc.prev_grad",
    "efc.prev_Mgrad",
    "efc.grad_dot",
    "efc.prev_cost",
    # contact
    "contact.friction",
    "contact.dim",
    "contact.efc_address",
  ]

  inputs_1 =[
    # contact
    "contact.friction",
    "contact.dist",
    "contact.dim",
    "contact.efc_address",
    "contact.includemargin",
    "contact.efc_address",
    "contact.worldid",
    # efc
    "efc.J",
    "efc.aref",
    "efc.done",
    "efc.cost",
    "efc.type",
    "efc.id",
    "efc.D",
    "efc.frictionloss",
    "efc.Jaref",
    "efc.force",
    "efc.Ma",
    "efc.state",
    "efc.grad",
    "efc.h",
    "efc.cholesky_L_tmp",
    "efc.cholesky_y_tmp",
    "efc.Mgrad",
    #
    "nefc",
    "qacc",
    "qM",
    "qacc_warmstart",
    "ne",
    "nf",
    "nefc",
    "nacon",
    "qfrc_smooth",
    "qacc_smooth",
    "qfrc_constraint",
    "qLD",
    "qLDiagInv"
  ]

  outputs_1 = [
    "solver_niter",
    "efc.search_dot",
    "efc.cost",
    "efc.done",
    "efc.Jaref",
    "efc.Ma",
    "efc.gauss",
    "efc.prev_cost",
    "efc.force",
    "efc.state",
    "qfrc_constraint",
    "efc.grad_dot",
    "efc.grad",
    "efc.h",
    "efc.Mgrad",
  ]

  outputs_0 = [
    "solver_niter",
    "nsolving",
    "qacc",
    # efc
    "efc.search",
    "efc.search_dot",
    "efc.quad_gauss",
    "efc.quad",
    "efc.Ma",
    "efc.Jaref",
    "efc.mv",
    "efc.jv",
    "efc.alpha",
    "efc.prev_grad",
    "efc.prev_Mgrad",
    "efc.beta",
  ]

  m_ = m.musa_model
  d_ = d.musa_data

  copy_wp_array_batch_attrlist(d_, d, inputs_0)
  copy_wp_array_batch_attrlist(d_, d, inputs_1)
  copy_wp_array_batch_attrlist(d_, d, outputs_0)
  copy_wp_array_batch_attrlist(d_, d, outputs_1)

  solve(m_, d_)

  copy_wp_array_batch_attrlist(d, d_, outputs_0)
  copy_wp_array_batch_attrlist(d, d_, outputs_1)
