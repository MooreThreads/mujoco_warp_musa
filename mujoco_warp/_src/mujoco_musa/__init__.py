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

def init():
  import axinfra

  from . import mjm_kernels

  axinfra.init()
  mjm_kernels.init_mjm_kernels()
  axinfra.context.runtime.register_module("mujoco_warp_musa", mjm_kernels.mjm_kernels)


from . import collision_convex
from . import collision_driver
from . import collision_primitive
from . import collision_sdf
from . import constraint
from . import derivative
from . import forward
from . import io
from . import math
from . import passive
from . import ray
from . import sensor
from . import smooth
from . import solver
from . import support
from . import types
from .types import BiasType as BiasType
from .types import BroadphaseFilter as BroadphaseFilter
from .types import BroadphaseType as BroadphaseType
from .types import ConeType as ConeType
from .types import Constraint as Constraint
from .types import Contact as Contact
from .types import DisableBit as DisableBit
from .types import DynType as DynType
from .types import EnableBit as EnableBit
from .types import GainType as GainType
from .types import GeomType as GeomType
from .types import IntegratorType as IntegratorType
from .types import JointType as JointType
from .types import Option as Option
from .types import SolverType as SolverType
from .types import State as State
from .types import Statistic as Statistic
from .types import TrnType as TrnType
