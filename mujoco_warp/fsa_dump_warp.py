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

import inspect
import json
import os
from enum import IntFlag
import enum

from datetime import datetime
from typing import Any, List, Tuple, Optional, Callable

import numpy as np
import warp as wp

import mujoco_warp as mjw
import ctypes

wp.init()
# wp.config.mode = "debug"  # 再次确保 graph 不会自动捕获
# wp.config.disable_cuda_graphs = True

from mujoco_warp._src.types import GeomType


warp_type_registry = {
  # pythonType
  int: "Int32",
  float: "FloatType",
  str: "String",
  bool: "Bool",
  list: "List",
  dict: "Dict",
  tuple: "Tuple",
  # 标量类型
  wp.int8: "Int8",
  wp.int16: "Int16",
  wp.int32: "Int32",
  wp.int64: "Int64",
  wp.uint8: "UInt8",
  wp.uint16: "UInt16",
  wp.uint32: "UInt32",
  wp.uint64: "UInt64",
  wp.float16: "FP16",
  wp.float32: "FP32",
  wp.float64: "FP64",
  wp.bool: "Bool",
  wp.array: "Array",
  # 向量和矩阵
  wp.vec2i: "Vec2I",
  wp.vec3i: "Vec3I",
  wp.vec2: "Vec2",
  wp.vec3: "Vec3",
  wp.vec4: "Vec4",
  wp.mat22: "Mat2x2",
  wp.mat33: "Mat3x3",
  wp.mat44: "Mat4x4",
  wp.quat: "Quat",
  wp.spatial_vectorf: "SpatialVec",
  mjw._src.types.vec8i: "Vec8I",
  mjw._src.types.vec8f: "Vec8",
  mjw._src.types.vec5f: "Vec5",
  mjw._src.types.vec6f: "Vec6",
  mjw._src.types.vec10f: "Vec10",
  mjw._src.types.vec11f: "Vec11",
  mjw._src.types.DisableBit: "Int32",
}

_kernel_param_names = {}
_kernel_recodes = {"kernelCalls": []}
_file_name = None
_test_data_base = "F:/StudioWorks/mujoco_warp_test_data_base/"
_kernelCallstackId = 0
_dumpDenseData = True
_initLogger = False
_warrnings = []

import logging
logging.basicConfig(
    filename='__app.log',          # 日志文件名（会自动创建 .log 文件）
    filemode='w',                 # 'a' 为追加模式（默认），'w' 为每次运行覆盖
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
    level=logging.INFO)

def _get_param_names(kernel):
  #if kernel not in _kernel_param_names:
  sig = inspect.signature(kernel)
  names = []
  for name, param in sig.parameters.items():
    # 只取普通位置参数（Warp kernel 目前只支持这种）
    if param.kind == param.POSITIONAL_OR_KEYWORD:
      names.append(name)
    # 如果有 *args 之类会出错，但 Warp kernel 不会出现
  return names
    #_kernel_param_names[kernel] = names
  #return _kernel_param_names[kernel]


# 保存原始launch函数
_original_launch = wp.launch
_original_launch_tiled = wp.launch_tiled


def dim2List(dim):
  dimList=[]
  if isinstance(dim,(int, np.integer)):
    dimList.append(int(dim))
    return dimList
  for it in dim:
    dimList.append(int(it))
  return dimList


def pack_two_numbers(num1, num2):
  if not (0 <= num1 <= 10 and 0 <= num2 <= 10):
      raise ValueError("数字必须在 0~10 范围内")
  packed = (num1 << 4) | num2
  return packed


def mjwGeoType2int(geoType:GeomType):
  if geoType == GeomType.PLANE:
    return 1
  if geoType == GeomType.HFIELD:
    return 2
  if geoType == GeomType.SPHERE:
    return 3
  if geoType == GeomType.CAPSULE:
    return 4
  if geoType == GeomType.ELLIPSOID:
    return 5
  if geoType == GeomType.CYLINDER:
    return 6
  if geoType == GeomType.BOX:
    return 7
  if geoType == GeomType.MESH:
    return 8
  if geoType == GeomType.SDF:
    return 9
  return 0
  

def matchListData(input:list):
  ret = []
  typeKey = "Invalid"
  for item in input:
    if type(item) == tuple and len(item) == 2:
      if isinstance(item[0],GeomType) and isinstance(item[0],GeomType):
        ret.append(mjwGeoType2int(item[0]))
        ret.append(mjwGeoType2int(item[1]))
        #TODO : All branch test passed
        typeKey = "Int32"

  if len(ret) == 0:
    return None
  return ret,typeKey


def collectClosure(pyFunc,closurePrms,closureNames,externalInputs):
  if pyFunc.__closure__ is not None:
    for cell,varName in zip(pyFunc.__closure__,pyFunc.__code__.co_freevars):
      if varName in closureNames or varName in externalInputs:
        global _warrnings
        warn = f"  [WARRNING] : Ignore Param from main function {varName} "
        _warrnings.append(warn)
        logging.info(warn)
        continue
      content = cell.cell_contents

      if callable(content) and isinstance(content, wp._src.context.Function):
        logging.info(" ------------- callable Closure ------------- ")
        collectClosure(content.func,closurePrms,closureNames,externalInputs)
      else:
        closurePrms.append(content)
        closureNames.append(varName)
        logging.info(f"     -- closure vars : {varName}")


def patched_launch(kernel, dim, inputs=[], outputs=[], device=None,isTile=False, **kwargs):
  global _dumpDenseData
  global _initLogger


  param_names = _get_param_names(kernel)

  if len(param_names) != len(inputs+outputs):
    errorCode = f"---------------------------------\n"
    errorCode += f"Merge {kernel.__qualname__} : param_names ({len(param_names)}) != input+outputs ({len(inputs+outputs)})"
    errorCode += f"\nnames:{param_names}\nparms_inputs:{inputs} \nparms_outputs:{outputs}"

    errorCode += f"\nname_src:{inspect.signature(kernel).parameters.items()}"
    raise RuntimeError(errorCode)

  code = inspect.getsource(kernel.func)
  params = []
  paramDumpDatas = {"paramMap": {}}

  usedKernelName = kernel.__name__
  qualName = str(kernel.__qualname__)
  pyFunc = kernel.func

  if ".<locals>." in qualName:
    logging.info(f"    nestedKernel : {qualName}")
    usedKernelName = qualName.replace(".<locals>.","__")
    #externalParams
    for var, name in zip(pyFunc.__closure__, pyFunc.__code__.co_freevars):
      logging.info(f"         VarName: {name}  Type: {type(var.cell_contents)}  Value: {var.cell_contents}")

  kernelDesc = {
    "dim": dim2List(dim),
    "device": str(device),
    "originNames": param_names,
    "closureVars":pyFunc.__code__.co_freevars
  }

  # _test_data_base
  test_data_base_root = os.path.join(_test_data_base, usedKernelName)
  os.makedirs(test_data_base_root, exist_ok=True)

  global _kernelCallstackId
  dim_str = str(dim).replace(" ", "").replace("(", "").replace(")", "").replace(",", "x")
  test_IO_data_file_name = str(_kernelCallstackId) + "." + usedKernelName + "." + dim_str + ".IO"
  test_IO_data_json_path = os.path.join(test_data_base_root, test_IO_data_file_name + ".json")

  def _save_npy(dir, name, arr) -> str:
    os.makedirs(dir, exist_ok=True)
    data_path = os.path.join(dir, name + ".npy")
    np.save(data_path, arr)
    return data_path
  
  
  logging.info(f"Dump-Start {_kernelCallstackId} | {usedKernelName} ")

  #outObjSet = set()
  originKernelInputs = len(inputs)
  outputParamMap = {}
  idx=0
  
  #callable 
  closurePrms = []
  closureNames = []
  if pyFunc.__closure__ is not None:
    #closurePrms = [cell.cell_contents for cell in pyFunc.__closure__]
    #closureNames = pyFunc.__code__.co_freevars
    #logging.info(f" -- closure vars : {closureNames}")
    collectClosure(pyFunc,closurePrms,closureNames,param_names[:originKernelInputs])

  

  nInputParams = originKernelInputs + len(closurePrms)

  logging.info(f"  originKernelInputs:  {originKernelInputs}  closures: {len(closurePrms)} ")

  param_names[originKernelInputs:originKernelInputs] = closureNames

  all_params = inputs + closurePrms + outputs

  assert len(param_names) == len(all_params), (
      f"--------------------------\nMergeArgs : param_names ({len(param_names)}) != all_params ({len(all_params)})\nparam_names: {param_names} \nall_params: {all_params}"
  )
  
  for name, inputPrm in zip(param_names, all_params):
    prm = {}
    prm["name"] = name
    prm["isArray"] = False
    prm["ignore"] = False
    prm["ignoreDesc"] = ""
    prm["originType"] = str(type(inputPrm))
    prm["dtype"] = str(type(inputPrm)) #'int' object has no attribute 'dtype'
    prm["isOutput"] = False
    prm["externalFileSave"] = False

    bufferDataRaw = []
    if isinstance(inputPrm, wp.array):
      if _dumpDenseData == True:
        # cpu_array = inputPrm.to("cpu")
        # bufferDataRaw = cpu_array.numpy().tolist()
        base_dir = os.path.join(_test_data_base, usedKernelName)
        data_dir = os.path.join(base_dir, test_IO_data_file_name)
        save_path = _save_npy(data_dir, prm["name"] + ".input", inputPrm.numpy())
        bufferDataRaw = os.path.relpath(save_path, base_dir).replace("\\", "/")
        prm["externalFileSave"] = True
      prm["isArray"] = True
      prm["ndim"] = inputPrm.ndim
      if inputPrm.dtype in warp_type_registry:
        prm["dtype"] = warp_type_registry[inputPrm.dtype]
      else:
        prm["dtype"] = str(type(inputPrm.dtype))
        logging.info(f"        --------------------- ERROR TYPE : {inputPrm.dtype}")
    elif isinstance(inputPrm,list):
      matchedData = matchListData(inputPrm)
      if matchedData != None:
        prm["isArray"] = True
        bufferDataRaw = matchedData[0]
        prm["dtype"] = matchedData[1]
      else:
        bufferDataRaw = "List"
        prm["ignoreDesc"] = "unknow-list-format"
        prm["ignore"]  = True

    elif isinstance(inputPrm,mjw._src.types.TileSet):
      prm["dtype"] = "Int32"
      bufferDataRaw = inputPrm.size
    elif isinstance(inputPrm, enum.IntFlag):  # True（使用完整路径）
      prm["dtype"] = "Int32"
      bufferDataRaw = int(inputPrm)
      logging.info(f"         enum.IntFlag Var: {name} - {inputPrm}")
    elif callable(inputPrm):
      logging.info(f"         Function Var: {name} -\n          {inputPrm}")

    else:
      dtype = type(inputPrm)
      if dtype in warp_type_registry:
        prm["dtype"] = warp_type_registry[dtype]
        if isinstance(inputPrm, ctypes.Array):
          bufferDataRaw = list(inputPrm)
        else:
          bufferDataRaw = inputPrm
      else:
        prm["ignore"] = True
        prm["ignoreDesc"] = "unknow-type"
        bufferDataRaw = str(inputPrm)

    prm["shape"] = inputPrm.shape if isinstance(inputPrm, wp.array) else None

    if prm["ignore"] == False:
      if idx >= nInputParams:
        prm["isOutput"] = True
        logging.info(f"        OutParam:{name}")
        outputParamMap.update({name: [inputPrm, prm]})
      
      params.append(prm)
      paramDumpDatas["paramMap"].update({name: {"desc": prm, "data": bufferDataRaw}})
    # logging.info(f"TEST:{name} ====  {str(bufferDataRaw)}")
    # logging.info(f"TEST:{name} ====  {json.dumps(bufferDataRaw)}")
    idx+=1

  # ,"desc":kernelDesc
  paramDumpDatas.update({"desc": kernelDesc})

  # 带名字的记录
  record = {
    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
    "kernel": usedKernelName,
    "qualname":kernel.__qualname__,
    "kernelSrcFile": str(kernel.func.__code__.co_filename).replace("\\", "/"),
    "dim": dim2List(dim),
    "dimType":str(type(dim)),
    "device": str(device),
    "params": params,
    "code": code
    #"originNames": param_names
  }

  # 保存 json（漂亮缩进）
  os.makedirs("warp_json_records", exist_ok=True)

  global _file_name
  if _file_name == None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    _file_name = "warp_json_records/" + ts + "DUMP.json"
    print("_file_name:          ====================               ", _file_name)

  # 真正执行
  if isTile:
    result = _original_launch_tiled(kernel=kernel, dim=dim, inputs=inputs, outputs=outputs, device=device, **kwargs)
  else:
    result = _original_launch(kernel=kernel, dim=dim, inputs=inputs, outputs=outputs, device=device, **kwargs)

  # {name:{"desc":prm,"data":bufferDataRaw}}
  outBufferMap = {}
  for oname, oPrmBody in outputParamMap.items():
    if _dumpDenseData == True:
      # cpu_array = oPrmBody[0].to("cpu")
      base_dir = os.path.join(_test_data_base, usedKernelName)
      data_dir = os.path.join(base_dir, test_IO_data_file_name)
      save_path = _save_npy(data_dir, oname + ".output", oPrmBody[0].numpy())
      outBufferMap.update(
        {
          oname: {
            "desc": oPrmBody[1],
            "data": os.path.relpath(save_path, base_dir).replace("\\", "/"),
          }
        }
      )

  paramDumpDatas.update({"outputDataMap": outBufferMap})

  _kernelCallstackId += 1

  # 可选：再追加一次执行后的输出（直接覆盖同一个文件，方便看结果）
  # outputs_after = ["123"]
  # record["outputs_after"] = outputs_after
  _kernel_recodes["kernelCalls"].append(record)

  with open(_file_name, "w", encoding="utf-8") as f:
    json.dump(_kernel_recodes, f, indent=2, ensure_ascii=False)

  with open(test_IO_data_json_path, "w", encoding="utf-8") as f:
    # logging.info(str(paramDumpDatas))
    json.dump(paramDumpDatas, f, indent=2, ensure_ascii=False)

  
  return result


def patched_launch_tile(kernel, dim, inputs=[], outputs=[], device=None, **kwargs):
  return patched_launch(kernel, dim, inputs, outputs,isTile = True, device=device, **kwargs)


def print_state():
  global _warrnings
  logging.info("warnnings:"+"\n".join(_warrnings))