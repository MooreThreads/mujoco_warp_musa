import os

import setuptools
from setuptools.command import build_ext
from setuptools.command import install_scripts
from setuptools.command.egg_info import egg_info as _egg_info

REPO_DIR = os.path.abspath(os.path.dirname(__file__))
TEMP_BUILD_DIR = os.path.join(REPO_DIR,'build')
MUJOCO_MUSA_PATH = os.path.join(REPO_DIR,'dependencies','mujoco-musa')
WARP_MUSA_BIN_DIR = os.path.join(REPO_DIR,'mujoco_warp','_src','mujoco_musa','bin')

def build_musa_lib(repo_dir: str, build_dir: str, target_dir: str, update_submodule: bool = False):
    import multiprocessing
    import pathlib
    import shutil
    import subprocess

    if update_submodule:
      # 检查并拉取所有 submodule
      try:
          subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
      except Exception as e:
          print("拉取 submodule 失败，请确保已安装 git 并有网络连接。")
          raise e
    # /root/works/mt_verse/mujoco_warp_musa/dependencies/mujoco-musa
    build_temp = pathlib.Path(build_dir)
    build_temp.mkdir(parents=True, exist_ok=True)
    build_temp_source = pathlib.Path(repo_dir) / "dependencies" / "mujoco-musa"
    cmake_args = [
        "cmake",
        str(pathlib.Path(build_temp_source).resolve()),  # 源码根目录
        "-DUSE_MUSA=ON",
        "-B", str(build_temp.resolve()),
    ]
    subprocess.run(cmake_args)


    build_args = [
        "cmake",
        "--build", str(build_temp.resolve()),
        "--target", "mujoco_musa_shared",
        "--config", "Release",
        "--parallel", str(multiprocessing.cpu_count()),
    ]
    subprocess.run(build_args)
    print(build_args)
    so_src = build_temp / "libmujoco_musa_shared.so"
    so_dst = pathlib.Path(target_dir) / "libmujoco_musa_shared.so"
    if pathlib.Path(target_dir).exists() is False:
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"Copying {so_src} to {so_dst}")
    shutil.copy2(so_src, so_dst)

class CMakeExtension(setuptools.Extension):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def __init__(self, name):
    super().__init__(name, sources=[])

class BuildCMakeExtension(build_ext.build_ext):
  """Uses CMake to build extensions."""

  def run(self):
    env = os.environ
    run_build = not (env.get("SKIP_BUILD_MUSA_LIB", "").lower() in ("1", "true", "yes"))
    update_submodule = not (env.get("SKIP_UPDATE_SUBMODULE", "1").lower() in ("1", "true", "yes"))
    if run_build:
        build_musa_lib(REPO_DIR, TEMP_BUILD_DIR, WARP_MUSA_BIN_DIR, update_submodule)
    else:
        print("SKIP_BUILD_MUSA_LIB is set; skip building.")

class InstallScripts(install_scripts.install_scripts):
  """Strips file extension from executable scripts whose names end in `.py`."""

  def run(self):
    super().run()
    pass

class EggInfoWithBuildExt(_egg_info):
    def run(self):
        self.run_command('build_ext')  # 先执行 build_ext
        super().run()

if __name__ == '__main__':
    setuptools.setup(
        long_description="warp-musa based with binding mujoco warp",
        long_description_content_type='',
        package_data = {
            'mujoco_warp': ['_src/mujoco_musa/bin/*.so'],
        },
        include_package_data=True,
        cmdclass=dict(
            egg_info=EggInfoWithBuildExt,
            build_ext=BuildCMakeExtension,
            install_scripts=InstallScripts,
        ),
        ext_modules=[
            CMakeExtension('mujoco-musa'),
        ],
        scripts=[]
    )