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
from absl.testing import absltest

from mujoco_warp._src.smooth_test import SmoothTest
from mujoco_warp._src.constraint_test import ConstraintTest
from mujoco_warp._src.passive_test import PassiveTest
from mujoco_warp._src.solver_test import SolverTest
from mujoco_warp._src.forward_test import ForwardTest
from mujoco_warp._src.sensor_test import SensorTest
from mujoco_warp._src.ray_test import RayTest

from mujoco_warp.fsa_dump_warp import patched_launch, patched_launch_tile
from mujoco_warp import config

use_fsa_dump = False

if __name__ == "__main__":
  if use_fsa_dump:
    config.use_musa = False
    wp.launch = patched_launch
    wp.launch_tiled = patched_launch_tile

  wp.init()
  import mujoco_warp.musa_api as mj_musa
  mj_musa.init()
  absltest.main()
