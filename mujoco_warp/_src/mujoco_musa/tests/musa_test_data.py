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

from typing import Any, Optional, Sequence, Tuple, Union

import mujoco
import numpy as np
from etils import epath

from .. import types as mjmtp
from ..io import override_model
from ..io import put_data
from ..io import put_model


def fixture(
  path: Optional[str] = None,
  xml: Optional[str] = None,
  keyframe: Optional[Union[int, str]] = None,
  qpos_noise: Optional[float] = None,
  qvel_noise: Optional[float] = None,
  ctrl_noise: Optional[float] = None,
  qfrc_noise: Optional[float] = None,
  xfrc_noise: Optional[float] = None,
  mocap_noise: Optional[float] = None,
  overrides: Union[dict[str, Any], Sequence[str]] = tuple(),
  nworld: int = 1,
) -> Tuple[mujoco.MjModel, mujoco.MjData, mjmtp.Model, mjmtp.Data]:
  """Loads MuJoCo MjModel / MjData and corresponding mjw.Model / mjw.Data.

  Args:
    path: path to the XML file to load, relative to the `test_data` root.
    xml: XML string to load (instead of fname).
    keyframe: if provided, the data will be reset to the specified keyframe.
    qpos_noise: initialize `qpos` with uniform noise with this magnitude.
    qvel_noise: initialize `qvel` with uniform noise with this magnitude.
    ctrl_noise: initialize `ctrl` with uniform noise with this magnitude.
    qfrc_noise: initialize `qfrc_applied` with uniform noise with this magnitude.
    xfrc_noise: initialize `xfrc_applied` with uniform noise with this magnitude.
    mocap_noise: initialize `mocap_pos` and `mocap_quat` will be initialized with uniform noise.
    overrides: a dict (or sequence of "foo=bar" strings) of model fields to override.
    nworld: number of worlds to create in mjw.Data.

  Returns:
    Tuple containing:
      mujoco.MjModel
      mujoco.MjData
      mjw.Model
      mjw.Data
  """
  np.random.seed(42)  # reduce test flakiness

  if path is not None:
    path = epath.resource_path("mujoco_warp") / "test_data" / path
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
  elif xml is not None:
    mjm = mujoco.MjModel.from_xml_string(xml)
  else:
    raise ValueError("either fname or xml must be provided")

  override_model(mjm, overrides)
  mjd = mujoco.MjData(mjm)
  if keyframe is not None:
    if isinstance(keyframe, str):
      keyframe = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)
  else:
    mujoco.mj_resetData(mjm, mjd)

  if qpos_noise:
    mjd.qpos[:] = np.random.uniform(-qpos_noise, qpos_noise, mjm.nq)
  if qvel_noise:
    mjd.qvel[:] = np.random.uniform(-qvel_noise, qvel_noise, mjm.nv)
  if ctrl_noise:
    mjd.ctrl[:] = np.random.uniform(-ctrl_noise, ctrl_noise, size=mjm.nu)
  if qfrc_noise:
    mjd.qfrc_applied[:] = np.random.uniform(-qfrc_noise, qfrc_noise, size=mjm.nv)
  if xfrc_noise:
    mjd.xfrc_applied[:] = np.random.uniform(-xfrc_noise, xfrc_noise, size=(mjm.nbody, 6))
  if mocap_noise:
    mjd.mocap_pos[:] = np.random.uniform(-mocap_noise, mocap_noise, size=(mjm.nmocap, 3))
    mjd.mocap_quat[:] = np.random.uniform(-mocap_noise, mocap_noise, size=(mjm.nmocap, 4))

  if qpos_noise or qvel_noise or ctrl_noise or qfrc_noise or xfrc_noise:
    mujoco.mj_step(mjm, mjd, 10)  # let dynamics get state significantly non-zero, non-identity rotations etc

  mujoco.mj_forward(mjm, mjd)
  mjd.qacc_warmstart = mjd.qacc
  m = put_model(mjm)
  override_model(m, overrides)

  d = put_data(mjm, mjd, nworld=nworld)

  return mjm, mjd, m, d

# from mujoco_warp import test_data as test_data_warp

# def fixture_wp(
#   path: Optional[str] = None,
#   xml: Optional[str] = None,
#   keyframe: Optional[Union[int, str]] = None,
#   qpos_noise: Optional[float] = None,
#   qvel_noise: Optional[float] = None,
#   ctrl_noise: Optional[float] = None,
#   qfrc_noise: Optional[float] = None,
#   xfrc_noise: Optional[float] = None,
#   mocap_noise: Optional[float] = None,
#   overrides: Union[dict[str, Any], Sequence[str]] = tuple(),
#   nworld: int = 1,
# ) -> Tuple[mujoco.MjModel, mujoco.MjData, mjmtp.Model, mjmtp.Data]:
#   mj_m, mj_d, m_wp, d_wp = test_data_warp.fixture(path, xml, keyframe, qpos_noise, qvel_noise, ctrl_noise, qfrc_noise, xfrc_noise, mocap_noise, overrides, nworld)
#   m = m_wp.musa_model
#   d = d_wp.musa_data
#   return mj_m, mj_d, m, d


# import dataclasses
# import axinfra as ax

# def field_equal(d1, d2, filed_name: str = ""):
#   equal = True
#   if isinstance(type(d1), ax.array) or type(d1) == ax.array:
#     equal = np.array_equal(d1.numpy(), d2.numpy())
#   elif dataclasses.is_dataclass(type(d1)):
#     equal = data_equal(d1, d2)
#   elif isinstance(d1, tuple):
#     equal = all(field_equal(d1[i], d2[i], str(i)) for i in range(len(d1)))
#   else:
#     equal = d1 == d2
#   if not equal:
#     print(f"data mismatch '{filed_name}': type={type(d1)}")
#     if isinstance(type(d1), ax.array) or type(d1) == ax.array:
#       print(d1)
#       print(d2)
#   return equal


# def data_equal(data1, data2):
#   all_equal = True
#   for f in dataclasses.fields(type(data1)):
#     d1 = getattr(data1, f.name)
#     d2 = getattr(data2, f.name)
#     equal = field_equal(d1, d2, f.name)
#     if not equal:
#       print(f"field mismatch '{f.name}': type={f.type}")
#       all_equal = False
#   return all_equal
