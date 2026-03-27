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


from __future__ import annotations

import ctypes
import os
import platform

from axinfra import AxKernelLaunchParam

# from .types import *
from axinfra import array_t
from axinfra import launch_bounds_t
from axinfra._src.types import c_vec3
from axinfra._src.types import c_vec6

mjm_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

class MJM_CPythonKernel:
    """Warp runtime context."""
    def __init__(self):
        
        if not (platform.system() == "Linux" and platform.machine() == "x86_64"):
            raise RuntimeError("mujoco musa currently only supports Linux x86_64 systems.")
        print("Loading mujoco musa kernels...")

        bin_path = os.path.join(mjm_path, "mujoco_musa", "bin")
        warp_lib = os.path.join(bin_path, "libmujoco_musa_shared.so")
        self.core = self.load_dll(warp_lib)
        self.kernel_mapping = {}
        
        try:

            self.core._rk_accumulate_activation_velocity.argtypes = [array_t,ctypes.c_float,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._rk_accumulate_activation_velocity.restype = None
            self.kernel_mapping["_rk_accumulate_activation_velocity"] = self.core._rk_accumulate_activation_velocity

            self.core._actuator_velocity__actuator_velocity.argtypes = [array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._actuator_velocity__actuator_velocity.restype = None
            self.kernel_mapping["_actuator_velocity__actuator_velocity"] = self.core._actuator_velocity__actuator_velocity

            self.core._tendon_velocity__tendon_velocity.argtypes = [array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_velocity__tendon_velocity.restype = None
            self.kernel_mapping["_tendon_velocity__tendon_velocity"] = self.core._tendon_velocity__tendon_velocity

            self.core._actuator_force.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._actuator_force.restype = None
            self.kernel_mapping["_actuator_force"] = self.core._actuator_force

            self.core._tendon_actuator_force.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_actuator_force.restype = None
            self.kernel_mapping["_tendon_actuator_force"] = self.core._tendon_actuator_force

            self.core._tendon_actuator_force_clamp.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_actuator_force_clamp.restype = None
            self.kernel_mapping["_tendon_actuator_force_clamp"] = self.core._tendon_actuator_force_clamp

            self.core._qfrc_actuator.argtypes = [ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qfrc_actuator.restype = None
            self.kernel_mapping["_qfrc_actuator"] = self.core._qfrc_actuator

            self.core._qfrc_smooth.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qfrc_smooth.restype = None
            self.kernel_mapping["_qfrc_smooth"] = self.core._qfrc_smooth

            self.core._euler_damp_qfrc_sparse.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._euler_damp_qfrc_sparse.restype = None
            self.kernel_mapping["_euler_damp_qfrc_sparse"] = self.core._euler_damp_qfrc_sparse

            self.core._next_activation.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_float,ctypes.c_bool,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._next_activation.restype = None
            self.kernel_mapping["_next_activation"] = self.core._next_activation

            self.core._next_velocity.argtypes = [array_t,array_t,array_t,ctypes.c_float,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._next_velocity.restype = None
            self.kernel_mapping["_next_velocity"] = self.core._next_velocity

            self.core._next_position.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_float,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._next_position.restype = None
            self.kernel_mapping["_next_position"] = self.core._next_position

            self.core._next_time.argtypes = [array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._next_time.restype = None
            self.kernel_mapping["_next_time"] = self.core._next_time

            self.core._tile_euler_dense__euler_dense.argtypes = [array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tile_euler_dense__euler_dense.restype = None
            self.kernel_mapping["_tile_euler_dense__euler_dense"] = self.core._tile_euler_dense__euler_dense

            self.core._rk_accumulate_velocity_acceleration.argtypes = [array_t,array_t,ctypes.c_float,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._rk_accumulate_velocity_acceleration.restype = None
            self.kernel_mapping["_rk_accumulate_velocity_acceleration"] = self.core._rk_accumulate_velocity_acceleration

            self.core._ray.argtypes = [ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,c_vec6,ctypes.c_bool,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._ray.restype = None
            self.kernel_mapping["_ray"] = self.core._ray

            self.core._zero_nacon_ncollision.argtypes = [array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._zero_nacon_ncollision.restype = None
            self.kernel_mapping["_zero_nacon_ncollision"] = self.core._zero_nacon_ncollision

            self.core._sap_project__sap_project.argtypes = [ctypes.c_int,array_t,array_t,ctypes.c_int,array_t,c_vec3,ctypes.c_int,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sap_project__sap_project.restype = None
            self.kernel_mapping["_sap_project__sap_project"] = self.core._sap_project__sap_project

            self.core._segmented_sort__segmented_sort.argtypes = [array_t,array_t,ctypes.c_int,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._segmented_sort__segmented_sort.restype = None
            self.kernel_mapping["_segmented_sort__segmented_sort"] = self.core._segmented_sort__segmented_sort

            self.core._sap_range.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sap_range.restype = None
            self.kernel_mapping["_sap_range"] = self.core._sap_range

            self.core._sap_broadphase__kernel.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sap_broadphase__kernel.restype = None
            self.kernel_mapping["_sap_broadphase__kernel"] = self.core._sap_broadphase__kernel

            self.core._nxn_broadphase__kernel.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._nxn_broadphase__kernel.restype = None
            self.kernel_mapping["_nxn_broadphase__kernel"] = self.core._nxn_broadphase__kernel

            self.core.ccd_kernel_builder__ccd_kernel.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_bool,ctypes.c_bool,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.ccd_kernel_builder__ccd_kernel.restype = None
            self.kernel_mapping["ccd_kernel_builder__ccd_kernel"] = self.core.ccd_kernel_builder__ccd_kernel

            self.core._create_narrowphase_kernel___primitive_narrowphase.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._create_narrowphase_kernel___primitive_narrowphase.restype = None
            self.kernel_mapping["_create_narrowphase_kernel___primitive_narrowphase"] = self.core._create_narrowphase_kernel___primitive_narrowphase

            self.core._kinematics_root.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._kinematics_root.restype = None
            self.kernel_mapping["_kinematics_root"] = self.core._kinematics_root

            self.core._kinematics_level.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._kinematics_level.restype = None
            self.kernel_mapping["_kinematics_level"] = self.core._kinematics_level

            self.core._geom_local_to_global.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._geom_local_to_global.restype = None
            self.kernel_mapping["_geom_local_to_global"] = self.core._geom_local_to_global

            self.core._site_local_to_global.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._site_local_to_global.restype = None
            self.kernel_mapping["_site_local_to_global"] = self.core._site_local_to_global

            self.core._subtree_com_init.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._subtree_com_init.restype = None
            self.kernel_mapping["_subtree_com_init"] = self.core._subtree_com_init

            self.core._subtree_com_acc.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._subtree_com_acc.restype = None
            self.kernel_mapping["_subtree_com_acc"] = self.core._subtree_com_acc

            self.core._subtree_div.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._subtree_div.restype = None
            self.kernel_mapping["_subtree_div"] = self.core._subtree_div

            self.core._cinert.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cinert.restype = None
            self.kernel_mapping["_cinert"] = self.core._cinert

            self.core._cdof.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cdof.restype = None
            self.kernel_mapping["_cdof"] = self.core._cdof

            self.core._cam_local_to_global.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cam_local_to_global.restype = None
            self.kernel_mapping["_cam_local_to_global"] = self.core._cam_local_to_global

            self.core._light_local_to_global.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._light_local_to_global.restype = None
            self.kernel_mapping["_light_local_to_global"] = self.core._light_local_to_global

            self.core._flex_vertices.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._flex_vertices.restype = None
            self.kernel_mapping["_flex_vertices"] = self.core._flex_vertices

            self.core._flex_edges.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._flex_edges.restype = None
            self.kernel_mapping["_flex_edges"] = self.core._flex_edges

            self.core._joint_tendon.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._joint_tendon.restype = None
            self.kernel_mapping["_joint_tendon"] = self.core._joint_tendon

            self.core._spatial_site_tendon.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._spatial_site_tendon.restype = None
            self.kernel_mapping["_spatial_site_tendon"] = self.core._spatial_site_tendon

            self.core._spatial_geom_tendon.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._spatial_geom_tendon.restype = None
            self.kernel_mapping["_spatial_geom_tendon"] = self.core._spatial_geom_tendon

            self.core._spatial_tendon_wrap.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._spatial_tendon_wrap.restype = None
            self.kernel_mapping["_spatial_tendon_wrap"] = self.core._spatial_tendon_wrap

            self.core._crb_accumulate.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._crb_accumulate.restype = None
            self.kernel_mapping["_crb_accumulate"] = self.core._crb_accumulate

            self.core._qM_sparse.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qM_sparse.restype = None
            self.kernel_mapping["_qM_sparse"] = self.core._qM_sparse

            self.core._tendon_armature.argtypes = [ctypes.c_bool,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_armature.restype = None
            self.kernel_mapping["_tendon_armature"] = self.core._tendon_armature

            self.core._transmission.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._transmission.restype = None
            self.kernel_mapping["_transmission"] = self.core._transmission

            self.core._transmission_body_moment.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._transmission_body_moment.restype = None
            self.kernel_mapping["_transmission_body_moment"] = self.core._transmission_body_moment

            self.core._transmission_body_moment_scale.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._transmission_body_moment_scale.restype = None
            self.kernel_mapping["_transmission_body_moment_scale"] = self.core._transmission_body_moment_scale

            self.core._comvel_root.argtypes = [array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._comvel_root.restype = None
            self.kernel_mapping["_comvel_root"] = self.core._comvel_root

            self.core._comvel_level.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._comvel_level.restype = None
            self.kernel_mapping["_comvel_level"] = self.core._comvel_level

            self.core._cacc_world.argtypes = [array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cacc_world.restype = None
            self.kernel_mapping["_cacc_world"] = self.core._cacc_world

            self.core._cacc.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_bool,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cacc.restype = None
            self.kernel_mapping["_cacc"] = self.core._cacc

            self.core._cfrc.argtypes = [array_t,array_t,array_t,array_t,ctypes.c_bool,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cfrc.restype = None
            self.kernel_mapping["_cfrc"] = self.core._cfrc

            self.core._cfrc_backward.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cfrc_backward.restype = None
            self.kernel_mapping["_cfrc_backward"] = self.core._cfrc_backward

            self.core._qfrc_bias.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qfrc_bias.restype = None
            self.kernel_mapping["_qfrc_bias"] = self.core._qfrc_bias

            self.core._tendon_dot.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_dot.restype = None
            self.kernel_mapping["_tendon_dot"] = self.core._tendon_dot

            self.core._tendon_bias_coef.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_bias_coef.restype = None
            self.kernel_mapping["_tendon_bias_coef"] = self.core._tendon_bias_coef

            self.core._tendon_bias_qfrc.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_bias_qfrc.restype = None
            self.kernel_mapping["_tendon_bias_qfrc"] = self.core._tendon_bias_qfrc

            self.core._copy_CSR.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._copy_CSR.restype = None
            self.kernel_mapping["_copy_CSR"] = self.core._copy_CSR

            self.core._qLD_acc.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qLD_acc.restype = None
            self.kernel_mapping["_qLD_acc"] = self.core._qLD_acc

            self.core._qLDiag_div.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qLDiag_div.restype = None
            self.kernel_mapping["_qLDiag_div"] = self.core._qLDiag_div

            self.core._solve_LD_sparse_x_acc_up.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._solve_LD_sparse_x_acc_up.restype = None
            self.kernel_mapping["_solve_LD_sparse_x_acc_up"] = self.core._solve_LD_sparse_x_acc_up

            self.core._solve_LD_sparse_qLDiag_mul.argtypes = [array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._solve_LD_sparse_qLDiag_mul.restype = None
            self.kernel_mapping["_solve_LD_sparse_qLDiag_mul"] = self.core._solve_LD_sparse_qLDiag_mul

            self.core._solve_LD_sparse_x_acc_down.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._solve_LD_sparse_x_acc_down.restype = None
            self.kernel_mapping["_solve_LD_sparse_x_acc_down"] = self.core._solve_LD_sparse_x_acc_down

            self.core._cfrc_ext.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cfrc_ext.restype = None
            self.kernel_mapping["_cfrc_ext"] = self.core._cfrc_ext

            self.core._cfrc_ext_equality.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cfrc_ext_equality.restype = None
            self.kernel_mapping["_cfrc_ext_equality"] = self.core._cfrc_ext_equality

            self.core._cfrc_ext_contact.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._cfrc_ext_contact.restype = None
            self.kernel_mapping["_cfrc_ext_contact"] = self.core._cfrc_ext_contact

            self.core._qM_dense.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qM_dense.restype = None
            self.kernel_mapping["_qM_dense"] = self.core._qM_dense

            self.core._tile_cholesky_factorize_solve__cholesky_factorize_solve.argtypes = [array_t,array_t,array_t,ctypes.c_int,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tile_cholesky_factorize_solve__cholesky_factorize_solve.restype = None
            self.kernel_mapping["_tile_cholesky_factorize_solve__cholesky_factorize_solve"] = self.core._tile_cholesky_factorize_solve__cholesky_factorize_solve

            self.core._tile_cholesky_solve__cholesky_solve.argtypes = [array_t,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tile_cholesky_solve__cholesky_solve.restype = None
            self.kernel_mapping["_tile_cholesky_solve__cholesky_solve"] = self.core._tile_cholesky_solve__cholesky_solve

            self.core._tile_cholesky_factorize__cholesky_factorize.argtypes = [array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tile_cholesky_factorize__cholesky_factorize.restype = None
            self.kernel_mapping["_tile_cholesky_factorize__cholesky_factorize"] = self.core._tile_cholesky_factorize__cholesky_factorize

            self.core._subtree_vel_forward.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._subtree_vel_forward.restype = None
            self.kernel_mapping["_subtree_vel_forward"] = self.core._subtree_vel_forward

            self.core._linear_momentum.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._linear_momentum.restype = None
            self.kernel_mapping["_linear_momentum"] = self.core._linear_momentum

            self.core._angular_momentum.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._angular_momentum.restype = None
            self.kernel_mapping["_angular_momentum"] = self.core._angular_momentum

            self.core._zero_constraint_counts.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._zero_constraint_counts.restype = None
            self.kernel_mapping["_zero_constraint_counts"] = self.core._zero_constraint_counts

            self.core._efc_equality_connect.argtypes = [ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_equality_connect.restype = None
            self.kernel_mapping["_efc_equality_connect"] = self.core._efc_equality_connect

            self.core._efc_equality_weld.argtypes = [ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_equality_weld.restype = None
            self.kernel_mapping["_efc_equality_weld"] = self.core._efc_equality_weld

            self.core._efc_equality_joint.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_equality_joint.restype = None
            self.kernel_mapping["_efc_equality_joint"] = self.core._efc_equality_joint

            self.core._efc_equality_tendon.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_equality_tendon.restype = None
            self.kernel_mapping["_efc_equality_tendon"] = self.core._efc_equality_tendon

            self.core._efc_equality_flex.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_equality_flex.restype = None
            self.kernel_mapping["_efc_equality_flex"] = self.core._efc_equality_flex

            self.core._num_equality.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._num_equality.restype = None
            self.kernel_mapping["_num_equality"] = self.core._num_equality

            self.core._efc_friction_dof.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_friction_dof.restype = None
            self.kernel_mapping["_efc_friction_dof"] = self.core._efc_friction_dof

            self.core._efc_friction_tendon.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_friction_tendon.restype = None
            self.kernel_mapping["_efc_friction_tendon"] = self.core._efc_friction_tendon

            self.core._efc_limit_ball.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_limit_ball.restype = None
            self.kernel_mapping["_efc_limit_ball"] = self.core._efc_limit_ball

            self.core._efc_limit_slide_hinge.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_limit_slide_hinge.restype = None
            self.kernel_mapping["_efc_limit_slide_hinge"] = self.core._efc_limit_slide_hinge

            self.core._efc_limit_tendon.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_limit_tendon.restype = None
            self.kernel_mapping["_efc_limit_tendon"] = self.core._efc_limit_tendon

            self.core._efc_contact_pyramidal.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_contact_pyramidal.restype = None
            self.kernel_mapping["_efc_contact_pyramidal"] = self.core._efc_contact_pyramidal

            self.core._efc_contact_elliptic.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._efc_contact_elliptic.restype = None
            self.kernel_mapping["_efc_contact_elliptic"] = self.core._efc_contact_elliptic

            self.core._sensor_pos.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sensor_pos.restype = None
            self.kernel_mapping["_sensor_pos"] = self.core._sensor_pos

            self.core._limit_pos.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._limit_pos.restype = None
            self.kernel_mapping["_limit_pos"] = self.core._limit_pos

            self.core._sensor_vel.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sensor_vel.restype = None
            self.kernel_mapping["_sensor_vel"] = self.core._sensor_vel

            self.core._limit_vel.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._limit_vel.restype = None
            self.kernel_mapping["_limit_vel"] = self.core._limit_vel

            self.core._sensor_touch.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sensor_touch.restype = None
            self.kernel_mapping["_sensor_touch"] = self.core._sensor_touch

            self.core._sensor_tactile.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sensor_tactile.restype = None
            self.kernel_mapping["_sensor_tactile"] = self.core._sensor_tactile

            self.core._sensor_acc.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sensor_acc.restype = None
            self.kernel_mapping["_sensor_acc"] = self.core._sensor_acc

            self.core._tendon_actuator_force_sensor.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_actuator_force_sensor.restype = None
            self.kernel_mapping["_tendon_actuator_force_sensor"] = self.core._tendon_actuator_force_sensor

            self.core._tendon_actuator_force_cutoff.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._tendon_actuator_force_cutoff.restype = None
            self.kernel_mapping["_tendon_actuator_force_cutoff"] = self.core._tendon_actuator_force_cutoff

            self.core._limit_frc.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._limit_frc.restype = None
            self.kernel_mapping["_limit_frc"] = self.core._limit_frc

            self.core._energy_pos_zero.argtypes = [array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._energy_pos_zero.restype = None
            self.kernel_mapping["_energy_pos_zero"] = self.core._energy_pos_zero

            self.core._energy_pos_gravity.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._energy_pos_gravity.restype = None
            self.kernel_mapping["_energy_pos_gravity"] = self.core._energy_pos_gravity

            self.core._energy_pos_passive_joint.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._energy_pos_passive_joint.restype = None
            self.kernel_mapping["_energy_pos_passive_joint"] = self.core._energy_pos_passive_joint

            self.core._energy_vel_kinetic__energy_vel_kinetic.argtypes = [array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._energy_vel_kinetic__energy_vel_kinetic.restype = None
            self.kernel_mapping["_energy_vel_kinetic__energy_vel_kinetic"] = self.core._energy_vel_kinetic__energy_vel_kinetic

            self.core._energy_pos_passive_tendon.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._energy_pos_passive_tendon.restype = None
            self.kernel_mapping["_energy_pos_passive_tendon"] = self.core._energy_pos_passive_tendon

            self.core._sensor_rangefinder_init.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sensor_rangefinder_init.restype = None
            self.kernel_mapping["_sensor_rangefinder_init"] = self.core._sensor_rangefinder_init

            self.core._sensor_collision.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sensor_collision.restype = None
            self.kernel_mapping["_sensor_collision"] = self.core._sensor_collision

            self.core._contact_match.argtypes = [ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._contact_match.restype = None
            self.kernel_mapping["_contact_match"] = self.core._contact_match

            self.core._contact_sort__contact_sort.argtypes = [array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._contact_sort__contact_sort.restype = None
            self.kernel_mapping["_contact_sort__contact_sort"] = self.core._contact_sort__contact_sort

            self.core._spring_damper_dof_passive.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._spring_damper_dof_passive.restype = None
            self.kernel_mapping["_spring_damper_dof_passive"] = self.core._spring_damper_dof_passive

            self.core._spring_damper_tendon_passive.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._spring_damper_tendon_passive.restype = None
            self.kernel_mapping["_spring_damper_tendon_passive"] = self.core._spring_damper_tendon_passive

            self.core._flex_elasticity.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._flex_elasticity.restype = None
            self.kernel_mapping["_flex_elasticity"] = self.core._flex_elasticity

            self.core._flex_bending.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._flex_bending.restype = None
            self.kernel_mapping["_flex_bending"] = self.core._flex_bending

            self.core._fluid_force.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._fluid_force.restype = None
            self.kernel_mapping["_fluid_force"] = self.core._fluid_force

            self.core._qfrc_passive.argtypes = [ctypes.c_bool,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qfrc_passive.restype = None
            self.kernel_mapping["_qfrc_passive"] = self.core._qfrc_passive

            self.core._gravity_force.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._gravity_force.restype = None
            self.kernel_mapping["_gravity_force"] = self.core._gravity_force

            self.core._apply_ft.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_bool,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._apply_ft.restype = None
            self.kernel_mapping["_apply_ft"] = self.core._apply_ft

            self.core.mul_m_sparse_diag___mul_m_sparse_diag.argtypes = [array_t,array_t,array_t,array_t,ctypes.c_bool,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.mul_m_sparse_diag___mul_m_sparse_diag.restype = None
            self.kernel_mapping["mul_m_sparse_diag___mul_m_sparse_diag"] = self.core.mul_m_sparse_diag___mul_m_sparse_diag

            self.core.mul_m_sparse_ij___mul_m_sparse_ij.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_bool,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.mul_m_sparse_ij___mul_m_sparse_ij.restype = None
            self.kernel_mapping["mul_m_sparse_ij___mul_m_sparse_ij"] = self.core.mul_m_sparse_ij___mul_m_sparse_ij

            self.core.mul_m_dense___mul_m_dense.argtypes = [array_t,array_t,array_t,array_t,ctypes.c_bool,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.mul_m_dense___mul_m_dense.restype = None
            self.kernel_mapping["mul_m_dense___mul_m_dense"] = self.core.mul_m_dense___mul_m_dense

            self.core.contact_force_kernel.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,ctypes.c_bool,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.contact_force_kernel.restype = None
            self.kernel_mapping["contact_force_kernel"] = self.core.contact_force_kernel

            self.core.get_state___get_state.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.get_state___get_state.restype = None
            self.kernel_mapping["get_state___get_state"] = self.core.get_state___get_state

            self.core.set_state___set_state.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.set_state___set_state.restype = None
            self.kernel_mapping["set_state___set_state"] = self.core.set_state___set_state

            self.core.solve_init_efc.argtypes = [array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.solve_init_efc.restype = None
            self.kernel_mapping["solve_init_efc"] = self.core.solve_init_efc

            self.core.solve_init_jaref.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.solve_init_jaref.restype = None
            self.kernel_mapping["solve_init_jaref"] = self.core.solve_init_jaref

            self.core.update_constraint_init_cost.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_constraint_init_cost.restype = None
            self.kernel_mapping["update_constraint_init_cost"] = self.core.update_constraint_init_cost

            self.core.update_constraint_efc.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_constraint_efc.restype = None
            self.kernel_mapping["update_constraint_efc"] = self.core.update_constraint_efc

            self.core.update_constraint_init_qfrc_constraint.argtypes = [array_t,array_t,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_constraint_init_qfrc_constraint.restype = None
            self.kernel_mapping["update_constraint_init_qfrc_constraint"] = self.core.update_constraint_init_qfrc_constraint

            self.core.update_constraint_gauss_cost__kernel.argtypes = [array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_constraint_gauss_cost__kernel.restype = None
            self.kernel_mapping["update_constraint_gauss_cost__kernel"] = self.core.update_constraint_gauss_cost__kernel

            self.core.update_gradient_zero_grad_dot.argtypes = [array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_gradient_zero_grad_dot.restype = None
            self.kernel_mapping["update_gradient_zero_grad_dot"] = self.core.update_gradient_zero_grad_dot

            self.core.update_gradient_grad.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_gradient_grad.restype = None
            self.kernel_mapping["update_gradient_grad"] = self.core.update_gradient_grad

            self.core.update_gradient_JTDAJ_sparse_tiled__kernel.argtypes = [array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_gradient_JTDAJ_sparse_tiled__kernel.restype = None
            self.kernel_mapping["update_gradient_JTDAJ_sparse_tiled__kernel"] = self.core.update_gradient_JTDAJ_sparse_tiled__kernel

            self.core.update_gradient_set_h_qM_lower_sparse.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_gradient_set_h_qM_lower_sparse.restype = None
            self.kernel_mapping["update_gradient_set_h_qM_lower_sparse"] = self.core.update_gradient_set_h_qM_lower_sparse

            self.core.update_gradient_cholesky_blocked__kernel.argtypes = [array_t,array_t,array_t,ctypes.c_int,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_gradient_cholesky_blocked__kernel.restype = None
            self.kernel_mapping["update_gradient_cholesky_blocked__kernel"] = self.core.update_gradient_cholesky_blocked__kernel

            self.core.solve_init_search.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.solve_init_search.restype = None
            self.kernel_mapping["solve_init_search"] = self.core.solve_init_search

            self.core.linesearch_zero_jv.argtypes = [array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_zero_jv.restype = None
            self.kernel_mapping["linesearch_zero_jv"] = self.core.linesearch_zero_jv

            self.core.linesearch_jv_fused__kernel.argtypes = [array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_jv_fused__kernel.restype = None
            self.kernel_mapping["linesearch_jv_fused__kernel"] = self.core.linesearch_jv_fused__kernel

            self.core.linesearch_prepare_gauss.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_prepare_gauss.restype = None
            self.kernel_mapping["linesearch_prepare_gauss"] = self.core.linesearch_prepare_gauss

            self.core.linesearch_prepare_quad.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_prepare_quad.restype = None
            self.kernel_mapping["linesearch_prepare_quad"] = self.core.linesearch_prepare_quad

            self.core.linesearch_iterative.argtypes = [ctypes.c_int,array_t,array_t,array_t,ctypes.c_int,ctypes.c_float,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_iterative.restype = None
            self.kernel_mapping["linesearch_iterative"] = self.core.linesearch_iterative

            self.core.linesearch_qacc_ma.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_qacc_ma.restype = None
            self.kernel_mapping["linesearch_qacc_ma"] = self.core.linesearch_qacc_ma

            self.core.linesearch_jaref.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_jaref.restype = None
            self.kernel_mapping["linesearch_jaref"] = self.core.linesearch_jaref

            self.core.solve_zero_search_dot.argtypes = [array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.solve_zero_search_dot.restype = None
            self.kernel_mapping["solve_zero_search_dot"] = self.core.solve_zero_search_dot

            self.core.solve_search_update.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.solve_search_update.restype = None
            self.kernel_mapping["solve_search_update"] = self.core.solve_search_update

            self.core.solve_done.argtypes = [ctypes.c_int,array_t,ctypes.c_int,ctypes.c_float,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.solve_done.restype = None
            self.kernel_mapping["solve_done"] = self.core.solve_done

            self.core.update_gradient_JTDAJ_dense_tiled__kernel.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_gradient_JTDAJ_dense_tiled__kernel.restype = None
            self.kernel_mapping["update_gradient_JTDAJ_dense_tiled__kernel"] = self.core.update_gradient_JTDAJ_dense_tiled__kernel

            self.core.update_gradient_cholesky__kernel.argtypes = [array_t,array_t,array_t,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_gradient_cholesky__kernel.restype = None
            self.kernel_mapping["update_gradient_cholesky__kernel"] = self.core.update_gradient_cholesky__kernel

            self.core.update_gradient_JTCJ.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,ctypes.c_int,ctypes.c_int,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.update_gradient_JTCJ.restype = None
            self.kernel_mapping["update_gradient_JTCJ"] = self.core.update_gradient_JTCJ

            self.core.linesearch_parallel_fused.argtypes = [array_t,ctypes.c_int,ctypes.c_float,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_parallel_fused.restype = None
            self.kernel_mapping["linesearch_parallel_fused"] = self.core.linesearch_parallel_fused

            self.core.linesearch_parallel_best_alpha.argtypes = [ctypes.c_int,ctypes.c_float,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.linesearch_parallel_best_alpha.restype = None
            self.kernel_mapping["linesearch_parallel_best_alpha"] = self.core.linesearch_parallel_best_alpha

            self.core.solve_prev_grad_Mgrad.argtypes = [array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.solve_prev_grad_Mgrad.restype = None
            self.kernel_mapping["solve_prev_grad_Mgrad"] = self.core.solve_prev_grad_Mgrad

            self.core.solve_beta.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.solve_beta.restype = None
            self.kernel_mapping["solve_beta"] = self.core.solve_beta

            self.core.plane_convex_test.argtypes = [array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.plane_convex_test.restype = None
            self.kernel_mapping["plane_convex_test"] = self.core.plane_convex_test

            self.core._sdf_narrowphase.argtypes = [ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._sdf_narrowphase.restype = None
            self.kernel_mapping["_sdf_narrowphase"] = self.core._sdf_narrowphase

            self.core._qderiv_actuator_passive.argtypes = [ctypes.c_int,array_t,ctypes.c_int,ctypes.c_bool,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qderiv_actuator_passive.restype = None
            self.kernel_mapping["_qderiv_actuator_passive"] = self.core._qderiv_actuator_passive

            self.core._qderiv_tendon_damping.argtypes = [ctypes.c_int,array_t,ctypes.c_bool,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._qderiv_tendon_damping.restype = None
            self.kernel_mapping["_qderiv_tendon_damping"] = self.core._qderiv_tendon_damping

            self.core._geom_dist___ccd_kernel.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,ctypes.c_float,ctypes.c_bool,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core._geom_dist___ccd_kernel.restype = None
            self.kernel_mapping["_geom_dist___ccd_kernel"] = self.core._geom_dist___ccd_kernel

            self.core.reset_data__reset_xfrc_applied.argtypes = [array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.reset_data__reset_xfrc_applied.restype = None
            self.kernel_mapping["reset_data__reset_xfrc_applied"] = self.core.reset_data__reset_xfrc_applied

            self.core.reset_data__reset_qM.argtypes = [array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.reset_data__reset_qM.restype = None
            self.kernel_mapping["reset_data__reset_qM"] = self.core.reset_data__reset_qM

            self.core.reset_data__reset_mocap.argtypes = [array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.reset_data__reset_mocap.restype = None
            self.kernel_mapping["reset_data__reset_mocap"] = self.core.reset_data__reset_mocap

            self.core.reset_data__reset_contact.argtypes = [array_t,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.reset_data__reset_contact.restype = None
            self.kernel_mapping["reset_data__reset_contact"] = self.core.reset_data__reset_contact

            self.core.reset_data__reset_nworld.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,array_t,array_t,ctypes.c_int,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,array_t,launch_bounds_t,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p]
            self.core.reset_data__reset_nworld.restype = None
            self.kernel_mapping["reset_data__reset_nworld"] = self.core.reset_data__reset_nworld

        except AttributeError as e:
            raise RuntimeError(f"Setting C-types for {warp_lib} failed. It may need rebuilding.") from e
        
    def load_dll(self, dll_path):
        try:
            dll = ctypes.CDLL(dll_path, winmode=0)
        except OSError as e:
            raise RuntimeError(f"Failed to load the shared library '{dll_path}'") from e
        return dll

    @property
    def musa_kernel_mapping(self):
        return self.kernel_mapping

mjm_kernels = None

def init_mjm_kernels():
    global mjm_kernels
    if mjm_kernels is None:
        mjm_kernels = MJM_CPythonKernel()