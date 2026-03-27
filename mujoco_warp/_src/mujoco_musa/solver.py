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

from math import ceil

import axinfra as ax

from . import smooth
from . import support as mjmsp
from . import types as mjmtp
from .cached_array import get_cached_array


def _linesearch_iterative(m: mjmtp.Model, d: mjmtp.Data):
  """Iterative linesearch."""
  ax.launch(
    "linesearch_iterative",
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.impratio,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.ls_iterations,
      m.stat.meaninertia,
      d.ne,
      d.nf,
      d.nefc,
      d.contact.friction,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      d.efc.Jaref,
      d.efc.search_dot,
      d.efc.jv,
      d.efc.quad,
      d.efc.quad_gauss,
      d.efc.done,
      d.njmax,
    ],
    outputs=[d.efc.alpha],
  )


def _linesearch_parallel(m: mjmtp.Model, d: mjmtp.Data, cost: ax.array2d(dtype=float)): # type: ignore
  ax.launch(
    "linesearch_parallel_fused",
    dim=(d.nworld, m.opt.ls_iterations),
    inputs=[
      m.opt.impratio,
      m.opt.ls_iterations,
      m.opt.ls_parallel_min_step,
      d.ne,
      d.nf,
      d.nefc,
      d.contact.friction,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      d.efc.Jaref,
      d.efc.jv,
      d.efc.quad,
      d.efc.quad_gauss,
      d.efc.done,
      d.njmax,
      d.nacon,
    ],
    outputs=[cost],
  )

  ax.launch(
    "linesearch_parallel_best_alpha",
    dim=(d.nworld),
    inputs=[m.opt.ls_iterations, m.opt.ls_parallel_min_step, d.efc.done, cost],
    outputs=[d.efc.alpha],
  )


def _linesearch(m: mjmtp.Model, d: mjmtp.Data, cost: ax.array2d(dtype=float)): # type: ignore
  # mv = qM @ search
  mjmsp.mul_m(m, d, d.efc.mv, d.efc.search, skip=d.efc.done)

  # jv = efc_J @ search
  # TODO(team): is there a better way of doing batched matmuls with dynamic array sizes?

  # if we are only using 1 thread, it makes sense to do more dofs as we can also skip the
  # init kernel. For more than 1 thread, dofs_per_thread is lower for better load balancing.

  if m.nv > 50:
    dofs_per_thread = 20
  else:
    dofs_per_thread = 50

  threads_per_efc = ceil(m.nv / dofs_per_thread)
  # we need to clear the jv array if we're doing atomic adds.
  if threads_per_efc > 1:
    ax.launch(
      "linesearch_zero_jv",
      dim=(d.nworld, d.njmax),
      inputs=[d.nefc, d.efc.done],
      outputs=[d.efc.jv],
    )

  ax.launch(
    "linesearch_jv_fused__kernel",
    dim=(d.nworld, d.njmax, threads_per_efc),
    inputs=[d.nefc, d.efc.J, d.efc.search, d.efc.done, dofs_per_thread, m.nv],
    outputs=[d.efc.jv],
  )

  # prepare quadratics
  # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
  ax.launch(
    "linesearch_prepare_gauss",
    dim=(d.nworld),
    inputs=[m.nv, d.qfrc_smooth, d.efc.Ma, d.efc.search, d.efc.gauss, d.efc.mv, d.efc.done],
    outputs=[d.efc.quad_gauss],
  )

  # quad = [0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D]
  ax.launch(
    "linesearch_prepare_quad",
    dim=(d.nworld, d.njmax),
    inputs=[
      m.opt.impratio,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.Jaref,
      d.efc.jv,
      d.efc.done,
      d.nacon,
    ],
    outputs=[d.efc.quad],
  )

  if m.opt.ls_parallel:
    _linesearch_parallel(m, d, cost)
  else:
    _linesearch_iterative(m, d)

  ax.launch(
    "linesearch_qacc_ma",
    dim=(d.nworld, m.nv),
    inputs=[d.efc.search, d.efc.mv, d.efc.alpha, d.efc.done],
    outputs=[d.qacc, d.efc.Ma],
  )

  ax.launch(
    "linesearch_jaref",
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.efc.jv, d.efc.alpha, d.efc.done],
    outputs=[d.efc.Jaref],
  )


def _update_constraint(m: mjmtp.Model, d: mjmtp.Data):
  """Update constraint arrays after each solve iteration."""
  ax.launch(
    "update_constraint_init_cost",
    dim=(d.nworld),
    inputs=[d.efc.cost, d.efc.done],
    outputs=[d.efc.gauss, d.efc.cost, d.efc.prev_cost],
  )

  ax.launch(
    "update_constraint_efc",
    dim=(d.nworld, d.njmax),
    inputs=[
      m.opt.impratio,
      d.ne,
      d.nf,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      d.efc.Jaref,
      d.efc.done,
      d.nacon,
    ],
    outputs=[d.efc.force, d.efc.cost, d.efc.state],
  )

  # qfrc_constraint = efc_J.T @ efc_force
  ax.launch(
    "update_constraint_init_qfrc_constraint",
    dim=(d.nworld, m.nv),
    inputs=[d.nefc, d.efc.J, d.efc.force, d.efc.done, d.njmax],
    outputs=[d.qfrc_constraint],
  )

  # if we are only using 1 thread, it makes sense to do more dofs and skip the atomics.
  # For more than 1 thread, dofs_per_thread is lower for better load balancing.
  if m.nv > 50:
    dofs_per_thread = 20
  else:
    dofs_per_thread = 50

  threads_per_efc = ceil(m.nv / dofs_per_thread)

  # gauss = 0.5 * (Ma - qfrc_smooth).T @ (qacc - qacc_smooth)
  ax.launch(
    "update_constraint_gauss_cost__kernel",
    dim=(d.nworld, threads_per_efc),
    inputs=[d.qacc, d.qfrc_smooth, d.qacc_smooth, d.efc.Ma, d.efc.done, dofs_per_thread, m.nv],
    outputs=[d.efc.gauss, d.efc.cost],
  )


# TODO: remove, tmp warp array for update_gradient_cholesky_blocked when nv >= 32
class _tmp_wp_arr():
#   grad = None
#   h = None
#   done = None
#   cholesky_L_tmp = None
#   cholesky_y_tmp = None
#   Mgrad = None

#   @staticmethod
#   def get(d: mjmtp.Data):
#     def copy_from_ax(name: str):
#       wp_arr = getattr(_tmp_wp_arr, name)
#       ax_arr = getattr(d.efc, name)
#       if wp_arr is None or wp_arr.shape != ax_arr.shape:
#         wp_arr = wp.empty(shape=ax_arr.shape, dtype=ax.ax_type_to_wp_dtype(ax_arr.dtype))
#         setattr(_tmp_wp_arr, name, wp_arr)
#       ax.copy_wp_array(wp_arr, ax_arr)

#     copy_from_ax("grad")
#     copy_from_ax("h")
#     copy_from_ax("done")
#     copy_from_ax("cholesky_L_tmp")
#     copy_from_ax("cholesky_y_tmp")
#     copy_from_ax("Mgrad")
#     return _tmp_wp_arr.grad, _tmp_wp_arr.h, _tmp_wp_arr.done, _tmp_wp_arr.cholesky_L_tmp, _tmp_wp_arr.cholesky_y_tmp, _tmp_wp_arr.Mgrad
  pass


# internal musa func
def _update_gradient(m: mjmtp.Model, d: mjmtp.Data):
  # grad = Ma - qfrc_smooth - qfrc_constraint
  ax.launch("update_gradient_zero_grad_dot", dim=(d.nworld), inputs=[d.efc.done], outputs=[d.efc.grad_dot])

  ax.launch(
    "update_gradient_grad",
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, d.efc.done],
    outputs=[d.efc.grad, d.efc.grad_dot],
  )

  if m.opt.solver == mjmtp.SolverType.CG:
    smooth.solve_m(m, d, d.efc.Mgrad, d.efc.grad)
    # raise NotImplementedError("CG solver is not implemented yet.")
  elif m.opt.solver == mjmtp.SolverType.NEWTON:
    # h = qM + (efc_J.T * efc_D * active) @ efc_J
    if m.opt.is_sparse:
      num_blocks_ceil = ceil(m.nv / mjmtp.TILE_SIZE_JTDAJ_SPARSE)
      lower_triangle_dim = int(num_blocks_ceil * (num_blocks_ceil + 1) / 2)

      ax.launch(
        "update_gradient_JTDAJ_sparse_tiled__kernel",
        dim=(d.nworld, lower_triangle_dim, m.block_dim.update_gradient_JTDAJ_sparse),
        inputs=[
          d.nefc,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.efc.done,
          mjmtp.TILE_SIZE_JTDAJ_SPARSE,
          d.njmax
        ],
        outputs=[d.efc.h],
        block_dim=m.block_dim.update_gradient_JTDAJ_sparse,
      )

      ax.launch(
        "update_gradient_set_h_qM_lower_sparse",
        dim=(d.nworld, m.qM_fullm_i.size),
        inputs=[m.qM_fullm_i, m.qM_fullm_j, d.qM, d.efc.done],
        outputs=[d.efc.h],
      )
    else:
      nv_padded = d.efc.J.shape[2]
      ax.launch(
        "update_gradient_JTDAJ_dense_tiled__kernel",
        dim=(d.nworld, nv_padded),
        inputs=[
          d.nefc,
          d.qM,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.efc.done,
          mjmtp.TILE_SIZE_JTDAJ_DENSE,
          d.njmax,
          nv_padded
        ],
        outputs=[d.efc.h],
        block_dim=nv_padded,
      )

    if m.opt.cone == mjmtp.ConeType.ELLIPTIC:
      # Optimization: launching update_gradient_JTCJ with limited number of blocks on a GPU.
      # Profiling suggests that only a fraction of blocks out of the original
      # d.njmax blocks do the actual work. It aims to minimize #CTAs with no
      # effective work. It launches with #blocks that's proportional to the number
      # of SMs on the GPU. We can now query the SM count:
      # https://github.com/NVIDIA/warp/commit/f3814e7e5459e5fd13032cf0fddb3daddd510f30

      # make dim_block and nblocks_perblock static for update_gradient_JTCJ to allow
      # loop unrolling
      # if wp.get_device().is_cuda:
      #   sm_count = wp.get_device().sm_count

      #   # Here we assume one block has 256 threads. We use a factor of 6, which
      #   # can be changed in the future to fine-tune the perf. The optimal factor will
      #   # depend on the kernel's occupancy, which determines how many blocks can
      #   # simultaneously run on the SM. TODO: This factor can be tuned further.
      #   dim_block = ceil((sm_count * 6 * 256) / m.dof_tri_row.size)
      # else:
      #   # fall back for CPU
      #   dim_block = d.naconmax

      dim_block = d.naconmax
      nblocks_perblock = int((d.naconmax + dim_block - 1) / dim_block)

      ax.launch(
        "update_gradient_JTCJ",
        dim=(dim_block, m.dof_tri_row.size),
        inputs=[
          m.opt.impratio,
          m.dof_tri_row,
          m.dof_tri_col,
          d.contact.dist,
          d.contact.includemargin,
          d.contact.friction,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.efc.J,
          d.efc.D,
          d.efc.Jaref,
          d.efc.state,
          d.efc.done,
          d.naconmax,
          d.nacon,
          nblocks_perblock,
          dim_block,
        ],
        outputs=[d.efc.h],
      )

    # TODO(team): Define good threshold for blocked vs non-blocked cholesky
    if m.nv < 128:
      ax.launch(
        "update_gradient_cholesky__kernel",
        dim=(d.nworld, m.nv),
        inputs=[d.efc.grad, d.efc.h, d.efc.done, m.nv],
        outputs=[d.efc.Mgrad],
        block_dim=m.nv,
      )
    else:
      # TODO[Moore Threads]: blocked cholesky kernels implement
      # axinfra.launch(
      #   "update_gradient_cholesky_blocked__kernel",
      #   dim=d.nworld,
      #   inputs=[
      #     d.efc.grad.reshape(shape=(d.nworld, m.nv, 1)),
      #     d.efc.h,
      #     d.efc.done,
      #     m.nv,
      #     d.efc.cholesky_L_tmp,
      #     d.efc.cholesky_y_tmp.reshape(shape=(d.nworld, m.nv, 1)),
      #     16
      #   ],
      #   outputs=[d.efc.Mgrad.reshape(shape=(d.nworld, m.nv, 1))],
      #   block_dim=m.block_dim.update_gradient_cholesky,
      # )
      # axinfra.launch(
      #   "update_gradient_cholesky__kernel",
      #   dim=(d.nworld,m.nv),
      #   inputs=[d.efc.grad, d.efc.h, d.efc.done, m.nv],
      #   outputs=[d.efc.Mgrad],
      #   block_dim=m.nv,
      # )
      # from ..solver import update_gradient_cholesky_blocked

      # grad, h, done, cholesky_L_tmp, cholesky_y_tmp, Mgrad = _tmp_wp_arr.get(d)
      # wp.launch_tiled(
      #   update_gradient_cholesky_blocked(16),
      #   dim=d.nworld,
      #   inputs=[
      #     grad.reshape(shape=(d.nworld, m.nv, 1)),
      #     h,
      #     done,
      #     m.nv,
      #     cholesky_L_tmp,
      #     cholesky_y_tmp.reshape(shape=(d.nworld, m.nv, 1)),
      #   ],
      #   outputs=[Mgrad.reshape(shape=(d.nworld, m.nv, 1))],
      #   block_dim=m.block_dim.update_gradient_cholesky,
      # )
      # ax.copy_wp_array(d.efc.Mgrad, Mgrad)
      raise RuntimeError(f"nv = {m.nv} is too large for blocked cholesky")
  else:
    raise ValueError(f"Unknown solver type: {m.opt.solver}")


def _solver_iteration(
  m: mjmtp.Model,
  d: mjmtp.Data,
  step_size_cost: ax.array2d(dtype=float), # type: ignore
):
  _linesearch(m, d, step_size_cost)

  if m.opt.solver == mjmtp.SolverType.CG:
    ax.launch(
      "solve_prev_grad_Mgrad",
      dim=(d.nworld, m.nv),
      inputs=[d.efc.grad, d.efc.Mgrad, d.efc.done],
      outputs=[d.efc.prev_grad, d.efc.prev_Mgrad],
    )

  _update_constraint(m, d)
  _update_gradient(m, d)

  # polak-ribiere
  if m.opt.solver == mjmtp.SolverType.CG:
    ax.launch(
      "solve_beta",
      dim=d.nworld,
      inputs=[m.nv, d.efc.grad, d.efc.Mgrad, d.efc.prev_grad, d.efc.prev_Mgrad, d.efc.done],
      outputs=[d.efc.beta],
    )

  ax.launch("solve_zero_search_dot", dim=(d.nworld), inputs=[d.efc.done], outputs=[d.efc.search_dot])

  ax.launch(
    "solve_search_update",
    dim=(d.nworld, m.nv),
    inputs=[m.opt.solver, d.efc.Mgrad, d.efc.search, d.efc.beta, d.efc.done],
    outputs=[d.efc.search, d.efc.search_dot],
  )

  ax.launch(
    "solve_done",
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.tolerance,
      m.opt.iterations,
      m.stat.meaninertia,
      d.efc.grad_dot,
      d.efc.cost,
      d.efc.prev_cost,
      d.efc.done,
    ],
    outputs=[d.solver_niter, d.efc.done, d.nsolving],
  )


def create_context(m: mjmtp.Model, d: mjmtp.Data, grad: bool = True):
  # initialize some efc arrays
  ax.launch(
    "solve_init_efc",
    dim=(d.nworld),
    outputs=[d.solver_niter, d.efc.search_dot, d.efc.cost, d.efc.done],
  )

  # jaref = d.efc_J @ d.qacc - d.efc_aref
  ax.launch(
    "solve_init_jaref",
    dim=(d.nworld, d.njmax),
    inputs=[m.nv, d.nefc, d.qacc, d.efc.J, d.efc.aref],
    outputs=[d.efc.Jaref],
  )

  # Ma = qM @ qacc
  mjmsp.mul_m(m, d, d.efc.Ma, d.qacc, skip=d.efc.done)

  _update_constraint(m, d)

  if grad:
    _update_gradient(m, d)


def _solve(m: mjmtp.Model, d: mjmtp.Data):
  """Finds forces that satisfy constraints."""
  if not (m.opt.disableflags & mjmtp.DisableBit.WARMSTART):
    ax.copy(d.qacc, d.qacc_warmstart)
  else:
    ax.copy(d.qacc, d.qacc_smooth)

  # create context
  create_context(m, d, grad=True)

  # search = -Mgrad
  ax.launch(
    "solve_init_search",
    dim=(d.nworld, m.nv),
    inputs=[d.efc.Mgrad],
    outputs=[d.efc.search, d.efc.search_dot],
  )

  step_size_cost = get_cached_array("_solve_""step_size_cost", (d.nworld, m.opt.ls_iterations if m.opt.ls_parallel else 0), dtype=float)

  if m.opt.iterations != 0 and m.opt.graph_conditional:
    # Note: the iteration kernel (indicated by while_body) is repeatedly launched
    # as long as condition_iteration is not zero.
    # condition_iteration is a warp array of size 1 and type int, it counts the number
    # of worlds that are not converged, it becomes 0 when all worlds are converged.
    # When the number of iterations reaches m.opt.iterations, solver_niter
    # becomes zero and all worlds are marked as converged to avoid an infinite loop.
    # note: we only launch the iteration kernel if everything is not done
    #d.nsolving.fill_(d.nworld)
    #raise NotImplementedError("Graph conditional is not implemented")
    # axinfra.capture_while(
    #   d.nsolving,
    #   while_body=_solver_iteration,
    #   m=m,
    #   d=d,
    #   step_size_cost=step_size_cost,
    # )
    for _ in range(m.opt.iterations):
      _solver_iteration(m, d, step_size_cost)
  else:
    # This branch is mostly for when JAX is used as it is currently not compatible
    # with CUDA graph conditional.
    # It should be removed when JAX becomes compatible.
    for _ in range(m.opt.iterations):
      _solver_iteration(m, d, step_size_cost)


def solve(m: mjmtp.Model, d: mjmtp.Data):
  if d.njmax == 0 or m.nv == 0:
    ax.copy(d.qacc, d.qacc_smooth)
    d.solver_niter.fill_(0)
  else:
    _solve(m, d)
