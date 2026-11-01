# 基础环境兼容性说明

审计日期：2026-11-01。

官方来源：

- https://github.com/DAVIAN-Robotics/3D_HAMSTER
- https://huggingface.co/DAVIAN-Robotics/3D_HAMSTER
- https://github.com/huggingface/lerobot
- https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx

## 审计对象

| 对象 | 固定版本 |
|---|---|
| 参考 Notebook | notebooks/setup_env_reference.ipynb |
| 3D HAMSTER | 97216a8493f46301bf569d398462b8bb21c458c5 |
| 3D HAMSTER checkpoint | ddc5987a56cdcb14e5e2297817612532e46e912b |
| LeRobot | 3f797de5952432284a7c77ad2760218cb74e2f26 |

参考 Notebook SHA-256：8ACF825F25B5ED0386E4B2BE5EF91E841D2506B2A81DF0FC4A9FFA3E6D9109BE。

参考 Notebook 已保存的输出表明，它曾在 Colab T4 上用 Python 3.12.13、Torch 2.11.0+cu128 完成基础 CUDA 矩阵乘法。这个事实只说明附件中的参考栈曾运行成功，不等于本项目的新 Planner 环境已通过 GPU 验证。

## 约束对照

| 依赖 | 参考固定值 | 3D HAMSTER 官方要求 | 当前 LeRobot 官方要求 | 静态判断 |
|---|---:|---:|---:|---|
| Python | 3.12.13 | >=3.10 | >=3.12 | 两边满足 |
| Torch | 2.11.0+cu128 | >=2.1 | >=2.7,<2.12 | 两边满足 |
| Torchvision | 0.26.0+cu128 | >=0.16 | >=0.22,<0.27 | 两边满足 |
| NumPy | 2.0.2 | >=1.24 | >=2.0,<2.3 | 两边满足 |
| TorchCodec | 0.11.0+cu128 | 未要求 | dataset extra 为 >=0.3,<0.12（Linux x86_64） | 可保留，但 Planner 不依赖它 |
| NVIDIA NPP | 12.4.1.87 | 未直接约束 | 未直接约束 | 作为参考 GPU 栈保留 |
| Transformers | Planner 固定 4.57.6 | >=4.57,<4.58 | SmolVLA extra 间接要求 >=5.4,<5.6 | 冲突，无交集 |
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

用途：后续 LeRobot / SmolVLA。

本阶段不安装。后续应在固定 LeRobot commit 后重新生成独立环境记录。不能复用 Planner 环境，也不能为了兼容 Planner 把 LeRobot 的 Transformers 强行降到 4.57。

## 可复现策略

当前固定了：

1. uv、Python 和参考核心包版本；
2. 3D HAMSTER Git commit；
3. 3D HAMSTER checkpoint revision；
4. Planner Transformers patch 版本；
5. 用实际 Colab 解析结果生成版本记录。

上游 3D HAMSTER 只给出了部分范围依赖，没有官方完整 lockfile。本项目不会伪造一个未经 Colab 验证的全平台 lockfile。第一次成功运行后，应保留 outputs/environment/planner_environment.json，并可额外执行 uv pip freeze 形成该 Colab 镜像的冻结快照。

## 仍需运行时确认

- Python 3.12.13 下所有 3D HAMSTER 间接依赖是否都有 Colab Linux wheel；
- Torch 2.11.0+cu128 与模型自定义代码的真实加载；
- A100 上 BF16 加载显存；
- TorchCodec 与 NPP 的导入行为是否仍与参考 Notebook 一致；
- 六组示例的推理结果是否复现。

这些只能在 Colab GPU 上确认，本地静态审计不能替代。
