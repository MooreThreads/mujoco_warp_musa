# MuJoCo MUSA

**MuJoCo MUSA** 是基于 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 扩展而来的 Python 包，它为 MuJoCo Warp 添加了 MUSA 计算后端，使 MuJoCo 物理引擎可以使用 MT MUSA 架构的 GPU 加速物理仿真。

**MuJoCo MUSA** 的接口与 MuJoCo Warp 保持一致，用户可以平滑地从 MuJoCo Warp 迁移到 **MuJoCo MUSA**，只需要在导入包时作如下修改，即可切换到 MUSA 后端。

```python
import mujoco_warp.musa_api as mjw
```

在仿真精度方面，**MuJoCo MUSA** 与 MuJoCo Warp CPU 版本保持了良好的计算一致性。以 `benchmark/humanoid/humanoid.xml` 为例，该测试中单个 step 的关节角度 (qpos) 与 CPU 版本的相对误差控制在 1e-5 以内，在 MuJoCo 自带 Viewer 中可视化对比结果如下：

(左：MuJoCo Warp CPU；右：MuJoCo MUSA)

![humanoid](benchmark/humanoid/humanoid.gif)

## 环境配置

### 环境要求

- GPU: MTT S4000/S5000
- MUSA SDK: 4.3.4
- muThrust
- Python >= 3.10
- CMake >= 3.18

### 本地安装

建议使用 [uv](https://github.com/astral-sh/uv) 进行项目部署，uv 的安装方法可参考 [uv 中文文档](https://uv.doczh.com/getting-started/)。

```bash
uv venv
uv sync
```

运行单元测试：

```bash
uv run ./mujoco_warp/_src/mujoco_musa/tests/all_test.py
```

## 使用方法

### API

MuJoCo MUSA API 与原版 mujoco_warp 保持一致：

```python
import mujoco_warp.musa_api as mjw
```

详细用法可以参考原版 [mujoco_warp 文档](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)。

### mujoco-viewer

使用 [MuJoCo native visualizer](https://mujoco.readthedocs.io/en/stable/programming/visualization.html) 交互式查看器展示 MuJoCo MUSA 的模拟结果：

```bash
uv run mujoco_warp/viewer_musa.py benchmark/humanoid/humanoid.xml
```

### 无图形环境下配置 mujoco-viewer

如果想在无图形加速的纯计算卡环境运行 mujoco-viewer，可以利用 Mesa 图形库在 CPU 上进行渲染。

#### 虚拟桌面配置

在 Ubuntu Server 22.04 上配置 Mesa 软件渲染 + X11 VNC 虚拟桌面，可以按照以下步骤操作：

```bash
# 安装相关软件包
apt install -y \
    xorg \
    xserver-xorg-video-dummy \
    mesa-utils \
    libgl1-mesa-dri \
    xvfb \
    x11vnc \
    fluxbox \
    xterm

# 使用 Xvfb 创建临时虚拟 framebuffer
Xvfb :1 -screen 0 1280x720x24 &
# 启动 fluxbox 窗口管理器
DISPLAY=:1 fluxbox &
# 启动 vnc server
x11vnc -forever -noxdamage -repeat -rfbport 5901 -shared -nopw -display :1 &
```

此时即可用 VNC 客户端连接服务器的 5901 端口访问虚拟桌面。

#### 运行 mujoco-viewer

```bash
export DISPLAY=:1
uv run mujoco_warp/viewer_musa.py benchmark/humanoid/humanoid.xml
```

## 开发者指南

### 安装 warp 可选依赖

安装 warp 后可支持原版 mujoco-warp kernel 与 MUSA kernel 的混合调试。

```bash
uv pip install -e ".[warp]"

# warp 版单元测试
uv run ./mujoco_warp/_src/all_test.py
```

### 单独编译 mujoco_musa lib 文件

```bash
rm -f build/CMakeCache.txt

# 增量编译
python build_lib.py
```

生成的库文件位于 ```mujoco_warp/_src/mujoco_musa/bin/libmujoco_musa_shared.so```

### 使用 pytest 运行单元测试

```bash
# 运行所有 MUSA 单元测试，并生成 html 测试报告
uv run --with pytest-html -m pytest mujoco_warp/_src/mujoco_musa/tests --html=report.html --self-contained-html
```

## 未来计划

当前版本 **MuJoCo MUSA** 基于 [此版本 MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp/tree/9fc294d86955a303619a254cefae809a41adb274) 修改而来，适配时尚未有正式版本号。

我们后续计划跟随 MuJoCo Warp 的发版进行同步升级。

## 许可证

**MuJoCo MUSA** 使用 Apache License, Version 2.0，详情请参见 [LICENSE](LICENSE) 文件。
