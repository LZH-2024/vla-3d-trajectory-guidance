# 基础环境兼容性说明

审计日期：2026-11-01。

官方来源：

- https://github.com/DAVIAN-Robotics/3D_HAMSTER
- https://huggingface.co/DAVIAN-Robotics/3D_HAMSTER
- https://github.com/huggingface/lerobot/tree/3f797de5952432284a7c77ad2760218cb74e2f26
- https://github.com/huggingface/lerobot/blob/3f797de5952432284a7c77ad2760218cb74e2f26/docs/source/libero.mdx
- https://pypi.org/project/hf-libero/0.1.4/
- https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/camera_utils.py
- https://huggingface.co/datasets/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6

## 审计对象

| 对象 | 固定版本 |
|---|---|
| 参考 Notebook | notebooks/setup_env_reference.ipynb |
| 3D HAMSTER | 97216a8493f46301bf569d398462b8bb21c458c5 |
| 3D HAMSTER checkpoint | ddc5987a56cdcb14e5e2297817612532e46e912b |
| LeRobot | 3f797de5952432284a7c77ad2760218cb74e2f26 |
| hf-libero | 0.1.4；wheel SHA-256 207f76e2f28bff30f78132223d8592fe8f64b1f8fd90ce7024948ada0d7e2c27 |
| robosuite | 1.4.0 |
| MuJoCo | 3.8.1 |
| LIBERO assets | 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 |

参考 Notebook SHA-256：8ACF825F25B5ED0386E4B2BE5EF91E841D2506B2A81DF0FC4A9FFA3E6D9109BE。

参考 Notebook 已保存的输出表明，它曾在 Colab T4 上用 Python 3.12.13、Torch 2.11.0+cu128 完成基础 CUDA 矩阵乘法。此后，用户已在 G4 RTX PRO 6000 Blackwell 服务器完整运行 01/02；本地检查保存输出确认环境、模型加载和官方六例通过。服务器生成的原始 outputs 尚未下载，因此本地还没有对应 JSON/CSV/overlay 文件副本。

## 约束对照

| 依赖 | 参考固定值 | 3D HAMSTER 官方要求 | 当前 LeRobot 官方要求 | 静态判断 |
|---|---:|---:|---:|---|
| Python | 3.12.13 | >=3.10 | >=3.12 | 两边满足 |
| Torch | 2.11.0+cu128 | >=2.1 | >=2.7,<2.12 | 两边满足 |
| Torchvision | 0.26.0+cu128 | >=0.16 | >=0.22,<0.27 | 两边满足 |
| NumPy | 2.0.2 | >=1.24 | >=2.0,<2.3 | 两边满足 |
| TorchCodec | 0.11.0+cu128 | 未要求 | dataset extra 为 >=0.3,<0.12（Linux x86_64） | 可保留，但 Planner 不依赖它 |
| NVIDIA NPP | 12.4.1.87 | 未直接约束 | 未直接约束 | 作为参考 GPU 栈保留 |
| Transformers | Planner 固定 4.57.6 | >=4.57,<4.58 | LeRobot LIBERO / SmolVLA extra 间接要求 >=5.4,<5.6；probe 固定 5.5.4 | 冲突，无交集 |
| Hugging Face Hub | Planner 解析到 <1.0 | >=0.34,<1.0 | >=1.6,<2.0 | 冲突，无交集 |

因此，Python/Torch 层面可以复用参考配置，但应用层依赖必须分环境。

## 3D HAMSTER 安装细节

官方 README 的推理安装方式是 pip install -e .。setup.py 会安装 merged checkpoint 推理所需依赖，并把 PEFT 只放在 merge extra 中；仓库 requirements.txt 另外列出了 peft>=0.15。官方模型卡明确说明发布的是 merged、自包含 checkpoint，普通推理不需要 PEFT，因此本项目严格跟随 README 的默认安装，不启用 merge extra。

官方 perf extra 只增加 xformers。它不是正确性依赖，并可能触发 Torch 重新解析，所以默认不安装。只有基线推理已成功且重新核对完整版本记录后，才考虑单独测试。

审计的 LeRobot commit 对应项目版本 0.6.2；SmolVLA 官方安装入口是项目的 smolvla extra。

## 环境方案

### /content/.venv_planner

用途：3D HAMSTER 高层推理。

- Python 3.12.13；
- 保留参考 Torch、Torchvision、TorchCodec、NumPy、NVIDIA NPP；
- Transformers 固定为 4.57.6；
- 安装固定 commit 的官方 hamster3d 包；
- 默认不安装可选 xformers，避免它重新解析 Torch；
- 用 uv pip check 和精确版本脚本阻止静默漂移。

### /content/.venv

当前用途：LIBERO observation、固定 LeRobot 与后续 SmolVLA Stage C 冒烟。

- Python 3.12.13；
- 保留参考 Torch、Torchvision、TorchCodec、NumPy、NVIDIA NPP；
- LeRobot 固定到审计 commit，项目版本 0.6.2；
- Transformers 5.5.4，Hugging Face Hub 1.33.0；
- hf-libero 0.1.4、robosuite 1.4.0、robomimic 0.2.0、MuJoCo 3.8.1；
- 当前以固定 commit 的 `[libero]` 环境为基础；真实 SmolVLA 冒烟仍需核对本地 policy 所需 extra，但不得改变受保护核心版本；
- 安装 editable checkout 时使用 uv `--no-sources`，避免 LeRobot 的开发期 `tool.uv.sources` 把参考 CUDA TorchCodec 静默改成 CPU wheel；实际依赖仍来自固定 commit 的标准项目 metadata；
- `MUJOCO_GL=egl` 由运行命令显式传入；
- 独立子进程固定 `MPLBACKEND=Agg`，防止继承 Notebook 内核的 `matplotlib_inline` backend；该 backend 只用于无界面导入与静态输出；
- `nvidia-npp-cu12` 的 wheel 把 `libnppicc.so.12` 放在虚拟环境内部；03 Notebook 会把该目录显式加入子进程 `LD_LIBRARY_PATH`，环境检查脚本也会在缺失时用同一解释器自动 re-exec；
- 用独立精确版本脚本阻止核心版本漂移。

不能复用 Planner 环境，也不能为了兼容 Planner 把 LeRobot 的 Transformers 强行降到 4.57。

## LIBERO assets 固定

hf-libero 0.1.4 的默认下载函数未传 `revision`，会读取 `lerobot/libero-assets` 的移动 HEAD。probe 在导入环境前显式下载并核对 revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`，然后把 hf-libero 的进程内 assets cache 指向该目录。这样场景资源不会因 Hub HEAD 后续变化而静默漂移。

`hf-libero` wheel、包内 BDDL 和 init state 由版本与 wheel SHA-256 固定；每次 probe 还会记录本次任务 BDDL 与 init state 文件哈希。

## 可复现策略

当前固定了：

1. uv、Python 和参考核心包版本；
2. 3D HAMSTER Git commit；
3. 3D HAMSTER checkpoint revision；
4. Planner Transformers patch 版本；
5. 用实际服务器解析结果生成 Planner 与 LIBERO 两份版本记录；
6. 固定 hf-libero wheel 哈希与 LIBERO assets revision。

上游 3D HAMSTER 只给出了部分范围依赖，没有官方完整 lockfile。本项目不会伪造一个未经 Colab 验证的全平台 lockfile。第一次成功运行后，应保留 outputs/environment/planner_environment.json，并可额外执行 uv pip freeze 形成该 Colab 镜像的冻结快照。

## 已确认与仍需确认

03 的服务器保存输出已经确认 `/content/.venv` 受保护版本、固定 LeRobot checkout、MuJoCo EGL、固定 assets、双相机 RGB-D、米制 depth 与标定。原始 outputs 尚未下载，本地证据仍以 Notebook 保存输出为准。

仍需在下一次 GPU 会话确认：

- 04 的真实单帧 3D HAMSTER 输出与几何可用性；
- 本地 SmolVLA policy 与固定 commit 的真实加载；
- E0/E1/E2 在真实 `SmolVLAPolicy` 上的 forward/backward、梯度、动作维度与显存。

这些由 05 Notebook 的 Stage A/B/C 分别执行；本地 CPU Mock 不能替代。
