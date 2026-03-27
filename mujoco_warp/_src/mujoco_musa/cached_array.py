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

array_cache = {}

# Helper function to get or create cached array
def get_cached_array(key: str, shape = None, dtype = None) -> ax.array:
  if shape == None or dtype == None:
    return array_cache[key] if key in array_cache else None
  if key not in array_cache or array_cache[key].shape != shape or array_cache[key].dtype != dtype:
    array_cache[key] = ax.empty(shape=shape, dtype=dtype)
  return array_cache[key]

def make_array_cache(key: str, arr: ax.array):
  cache = get_cached_array(key, arr.shape, arr.dtype)
  ax.copy(cache, arr)
  return cache
