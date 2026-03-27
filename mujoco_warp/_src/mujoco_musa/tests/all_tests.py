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

from absl.testing import absltest

import mujoco_warp.musa_api as mj_musa
from mujoco_warp._src.mujoco_musa.tests.broadphase_test import BroadphaseTest
from mujoco_warp._src.mujoco_musa.tests.collision_driver_test import CollisionTest
from mujoco_warp._src.mujoco_musa.tests.constraint_test import ConstraintTest
from mujoco_warp._src.mujoco_musa.tests.forward_test import ForwardTest
from mujoco_warp._src.mujoco_musa.tests.io_test import IOTest
from mujoco_warp._src.mujoco_musa.tests.passive_test import PassiveTest
from mujoco_warp._src.mujoco_musa.tests.ray_test import RayTest
from mujoco_warp._src.mujoco_musa.tests.sensor_test import SensorTest
from mujoco_warp._src.mujoco_musa.tests.smooth_test import SmoothTest
from mujoco_warp._src.mujoco_musa.tests.solver_test import SolverTest
from mujoco_warp._src.mujoco_musa.tests.support_test import SupportTest

if __name__ == "__main__":
  mj_musa.init()
  absltest.main()
