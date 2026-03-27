# Modifications Copyright 2026 Moore Threads
# Copyright 2025 The Newton Developers
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

"""Tests for sensor functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import DisableBit
from mujoco_warp import test_data

# tolerance for difference between MuJoCo and MJWarp calculations - mostly
# due to float precision
_TOLERANCE = 5e-5

# from mujoco_warp.fsa_dump_warp import patched_launch, patched_launch_tile
# wp.launch = patched_launch
# wp.launch_tiled = patched_launch_tile


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SensorTest(parameterized.TestCase):
  @parameterized.product(
    type0=["sphere", "capsule", "ellipsoid", "cylinder"],  # TODO(team): box
    type1=["sphere", "capsule", "ellipsoid", "cylinder", "box"],
    type2=["sphere", "capsule", "ellipsoid", "cylinder", "box"],
    type3=["sphere", "capsule", "ellipsoid", "cylinder", "box"],
  )
  def test_sensor_collision(self, type0, type1, type2, type3):
    """Tests collision sensors: distance, normal, fromto."""
    _MJCF = f"""
      <mujoco>
        <worldbody>
          <body name="obj0">
            <geom name="obj0" type="{type0}" size=".1 .1 .1" euler="1 2 3"/>
          </body>
          <body name="obj1" pos="0 0 1">
            <geom name="obj1" type="{type1}" size=".1 .1 .1" euler="-1 2 -1"/>
          </body>
          <body name="objobj" pos="0 0 -1">
            <geom name="objobj0" pos=".01 0 0.005" type="{type2}" size=".09 .09 .09" euler="2 1 3"/>
            <geom name="objobj1" pos="-.01 0 -0.0025" type="{type3}" size=".11 .11 .11" euler="3 1 2"/>
          </body>
        </worldbody>
        <sensor>
          <!-- distance geom-geom -->
          <distance geom1="obj0" geom2="obj1" cutoff="0"/>
          <distance geom1="obj0" geom2="obj1" cutoff="10"/>
          <distance geom1="obj1" geom2="obj0" cutoff="0"/>
          <distance geom1="obj1" geom2="obj0" cutoff="10"/>
          <distance body1="obj0" body2="obj1" cutoff="10"/>
          <distance body1="obj1" body2="obj0" cutoff="10"/>
          <distance body1="obj0" body2="obj1" cutoff="10"/>
          <distance body1="obj1" body2="obj0" cutoff="10"/>
          <distance geom1="obj0" body2="obj1" cutoff="10"/>
          <distance body1="obj1" geom2="obj0" cutoff="10"/>

          <!-- normal geom-geom -->
          <normal geom1="obj0" geom2="obj1" cutoff="0"/>
          <normal geom1="obj0" geom2="obj1" cutoff="10"/>
          <normal geom1="obj1" geom2="obj0" cutoff="0"/>
          <normal geom1="obj1" geom2="obj0" cutoff="10"/>
          <normal body1="obj0" body2="obj1" cutoff="10"/>
          <normal body1="obj1" body2="obj0" cutoff="10"/>
          <normal body1="obj0" body2="obj1" cutoff="10"/>
          <normal body1="obj1" body2="obj0" cutoff="10"/>
          <normal geom1="obj0" body2="obj1" cutoff="10"/>
          <normal body1="obj1" geom2="obj0" cutoff="10"/>

          <!-- fromto geom-geom -->
          <fromto geom1="obj0" geom2="obj1" cutoff="0"/>
          <fromto geom1="obj0" geom2="obj1" cutoff="10"/>
          <fromto geom1="obj1" geom2="obj0" cutoff="0"/>
          <fromto geom1="obj1" geom2="obj0" cutoff="10"/>
          <fromto body1="obj0" body2="obj1" cutoff="10"/>
          <fromto body1="obj1" body2="obj0" cutoff="10"/>
          <fromto body1="obj0" body2="obj1" cutoff="10"/>
          <fromto body1="obj1" body2="obj0" cutoff="10"/>
          <fromto geom1="obj0" body2="obj1" cutoff="10"/>
          <fromto body1="obj1" geom2="obj0" cutoff="10"/>

          <!-- distance geom body -->
          <distance geom1="obj0" body2="objobj" cutoff="0"/>
          <distance geom1="obj0" body2="objobj" cutoff="10"/>
          <distance body1="objobj" geom2="obj0" cutoff="0"/>
          <distance body1="objobj" geom2="obj0" cutoff="10"/>
          <distance body1="obj0" body2="objobj" cutoff="10"/>
          <distance body1="objobj" body2="obj0" cutoff="10"/>
          <distance body1="obj0" body2="objobj" cutoff="10"/>
          <distance body1="objobj" body2="obj0" cutoff="10"/>
          <distance geom1="obj0" body2="objobj" cutoff="10"/>
          <distance body1="objobj" geom2="obj0" cutoff="10"/>

          <!-- normal geom body -->
          <normal geom1="obj0" body2="objobj" cutoff="0"/>
          <normal geom1="obj0" body2="objobj" cutoff="10"/>
          <normal body1="objobj" geom2="obj0" cutoff="0"/>
          <normal body1="objobj" geom2="obj0" cutoff="10"/>
          <normal body1="obj0" body2="objobj" cutoff="10"/>
          <normal body1="objobj" body2="obj0" cutoff="10"/>
          <normal body1="obj0" body2="objobj" cutoff="10"/>
          <normal body1="objobj" body2="obj0" cutoff="10"/>
          <normal geom1="obj0" body2="objobj" cutoff="10"/>
          <normal body1="objobj" geom2="obj0" cutoff="10"/>

          <!-- fromto geom body -->
          <fromto geom1="obj0" body2="objobj" cutoff="0"/>
          <fromto geom1="obj0" body2="objobj" cutoff="10"/>
          <fromto body1="objobj" geom2="obj0" cutoff="0"/>
          <fromto body1="objobj" geom2="obj0" cutoff="10"/>
          <fromto body1="obj0" body2="objobj" cutoff="10"/>
          <fromto body1="objobj" body2="obj0" cutoff="10"/>
          <fromto body1="obj0" body2="objobj" cutoff="10"/>
          <fromto body1="objobj" body2="obj0" cutoff="10"/>
          <fromto geom1="obj0" body2="objobj" cutoff="10"/>
          <fromto body1="objobj" geom2="obj0" cutoff="10"/>
        </sensor>
      </mujoco>
      """

    _, mjd, m, d = test_data.fixture(xml=_MJCF)

    d.sensordata.fill_(wp.inf)
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    mjw.sensor_pos(m, d)

    # print(d.sensordata.numpy()[0])
    # print(mjd.sensordata)
    _assert_eq(d.sensordata.numpy()[0], mjd.sensordata, "sensordata")

'''  @parameterized.parameters("sphere", "capsule", "ellipsoid", "cylinder", "box")
  def test_sensor_collision_plane(self, type_):
    """Tests collision sensors: distance, normal, fromto."""
    _MJCF = f"""
      <mujoco>
        <worldbody>
          <geom name="plane" type="plane" size="10 10 .01" euler="2 2 2"/>
          <body name="obj" pos="0 0 1">
            <geom name="obj" type="{type_}" size=".1 .1 .1" euler="1 2 3"/>
          </body>
        </worldbody>
        <sensor>
          <!-- distance geom-geom -->
          <distance geom1="plane" geom2="obj" cutoff="0"/>
          <distance geom1="plane" geom2="obj" cutoff="10"/>
          <distance geom1="obj" geom2="plane" cutoff="0"/>
          <distance geom1="obj" geom2="plane" cutoff="10"/>
          <distance geom1="plane" body2="obj" cutoff="10"/>

          <!-- normal geom-geom -->
          <normal geom1="plane" geom2="obj" cutoff="0"/>
          <normal geom1="plane" geom2="obj" cutoff="10"/>
          <normal geom1="obj" geom2="plane" cutoff="0"/>
          <normal geom1="obj" geom2="plane" cutoff="10"/>
          <normal geom1="plane" body2="obj" cutoff="10"/>

          <!-- fromto geom-geom -->
          <fromto geom1="plane" geom2="obj" cutoff="0"/>
          <fromto geom1="plane" geom2="obj" cutoff="10"/>
          <fromto geom1="obj" geom2="plane" cutoff="0"/>
          <fromto geom1="obj" geom2="plane" cutoff="10"/>
          <fromto geom1="plane" body2="obj" cutoff="10"/>
        </sensor>
      </mujoco>
      """

    _, mjd, m, d = test_data.fixture(xml=_MJCF)

    d.sensordata.fill_(wp.inf)
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    mjw.sensor_pos(m, d)

    _assert_eq(d.sensordata.numpy()[0], mjd.sensordata, "sensordata")

  @parameterized.parameters(0, 1)
  def test_insidesite(self, keyframe):
    _, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <site name="refsite" type="sphere" size=".1"/>
        <body name="body">
          <geom name="geom" type="sphere" size=".1"/>
          <site name="site"/>
          <camera name="camera"/>
          <joint type="slide" axis="1 0 0"/>
        </body>
      </worldbody>
      <sensor>
        <insidesite site="refsite" objtype="xbody" objname="body"/>
        <insidesite site="refsite" objtype="body" objname="body"/>
        <insidesite site="refsite" objtype="geom" objname="geom"/>
        <insidesite site="refsite" objtype="site" objname="site"/>
        <insidesite site="refsite" objtype="camera" objname="camera"/>
      </sensor>
      <keyframe>
        <key qpos="0"/>
        <key qpos="1"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=keyframe,
    )

    d.sensordata.fill_(wp.inf)
    mjw.forward(m, d)

    _assert_eq(d.sensordata.numpy()[0], mjd.sensordata, "sensordata")
'''

if __name__ == "__main__":
  wp.init()
  absltest.main()
