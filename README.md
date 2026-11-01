# VLA 3D Trajectory Guidance

本项目用于机器人学习课程大作业，当前目标是先稳定复现 **3D HAMSTER 官方高层规划器推理**，再判断其接入 LIBERO 与 SmolVLA 的可行性。

当前阶段只做：

- 3D HAMSTER 官方六组 RGB-D 示例的环境搭建、模型加载与推理复现；
- 原始预测、路点、静态可视化、耗时与显存记录；
- 深度单位、坐标范围和后续 LIBERO 接口的显式约定。

本阶段不训练模型，不修改 SmolVLA，不实现轨迹融合，也不编写完整论文。

## 当前结论

截至 2026-11-01，官方依赖存在明确冲突，不能放进同一个虚拟环境：

| 环境 | 用途 | 关键上游要求 |
|---|---|---|
| **/content/.venv_planner** | 3D HAMSTER 推理 | Transformers >=4.57,<4.58；Hugging Face Hub <1.0 |
| **/content/.venv** | 后续 LeRobot / SmolVLA | 当前 LeRobot 要求 Transformers >=5.4,<5.6；Hugging Face Hub >=1.6,<2.0 |

两者都能在约束层面使用 Python 3.12.13、Torch 2.11.0+cu128、Torchvision 0.26.0+cu128 和 NumPy 2.0.2，但 Transformers 与 Hub 版本没有交集。详见 [环境兼容性说明](docs/environment_compatibility.md)。

本项目固定：

- 3D HAMSTER Git commit：**97216a8493f46301bf569d398462b8bb21c458c5**
- 3D HAMSTER checkpoint revision：**ddc5987a56cdcb14e5e2297817612532e46e912b**
- LeRobot 审计 commit：**3f797de5952432284a7c77ad2760218cb74e2f26**
- Planner Transformers：**4.57.6**，位于官方允许范围内

官方依据：

- [3D HAMSTER 仓库](https://github.com/DAVIAN-Robotics/3D_HAMSTER)
- [3D HAMSTER 模型](https://huggingface.co/DAVIAN-Robotics/3D_HAMSTER)
- [LeRobot 仓库](https://github.com/huggingface/lerobot)
- [SmolVLA 官方说明](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx)

## 目录

    .
    ├── configs/                         # Colab 路径、上游 commit 和关键版本约束
    ├── docs/                            # 兼容性、接口设计和下一阶段清单
    ├── notebooks/
    │   ├── setup_env_reference.ipynb   # 用户提供的环境基准，保持不改
    │   ├── 01_planner_setup.ipynb      # Planner 环境、下载、导入与模型加载检查
    │   └── 02_planner_inference.ipynb  # 官方六例推理与结果展示
    ├── scripts/                         # 可重复执行的环境与推理入口
    ├── src/trajectory_guidance/         # 校验、坐标转换、可视化和接口契约
    ├── tests/                           # 不依赖 GPU 的基础测试
    └── outputs/                         # 生成结果；Git 默认忽略

## Colab 使用顺序

先在 Colab 中把本仓库放到：

    /content/vla-3d-trajectory-guidance

然后选择 A100 运行时，依次执行：

1. **notebooks/01_planner_setup.ipynb**
2. **notebooks/02_planner_inference.ipynb**

Notebook 默认内核不会切换到虚拟环境。所有安装、导入检查、模型加载和推理都由以下解释器显式执行：

    /content/.venv_planner/bin/python

01 Notebook 会：

1. 检查 GPU、驱动、显存与磁盘；
2. 安装 uv 0.13.6；
3. 创建并核对 Python 3.12.13 的 Planner 环境；
4. 安装并保护参考 GPU 栈；
5. 拉取固定 commit 的 3D HAMSTER；
6. 安装官方推理依赖；
7. 下载固定 revision 的公开 checkpoint；
8. 执行精确版本、CUDA 冒烟和可选模型加载检查；
9. 在 outputs/environment/ 写出版本记录。

02 Notebook 会一次加载模型并运行官方 sample_0 到 sample_5。每个样例保存：

- raw_prediction.json：原始模型文本、解析路点、校验结果、耗时和显存；
- raw_output.txt：模型原始文本；
- waypoints_uvd.csv：归一化图像坐标与米制深度；
- waypoints_camera_xyz.csv：有相机内参时生成的相机坐标系 XYZ；
- trajectory_overlay.png：二维轨迹叠加图；
- run_summary.json：六例汇总及是否全部通过。

## 坐标与深度约定

3D HAMSTER 的输出是 **[u, v, depth]**，不是直接的世界坐标 XYZ：

- u、v 使用 0 到 1000 的归一化图像坐标；
- depth 是米；
- 输入 RGB 与 depth 必须像素对齐；
- 官方预处理把最长边缩放到 640；
- 只有结合匹配分辨率的相机内参，才能反投影到相机坐标系；
- 外参方向和目标坐标系未核实前，不转换到 LIBERO 世界或机器人基座坐标系。

推理脚本会把空输出、NaN/Inf、u/v 越界、深度不在 0.01 到 10 米范围内视为失败。该范围来自官方几何输入的有效深度掩码。

## 本地基础测试

本地测试不下载模型，也不要求 GPU：

    python -m unittest discover -s tests

它只验证路点检查、深度检查与 UVD 到相机 XYZ 的几何转换。

## 已实现与待验证

已由代码实现：

- Git 项目骨架与忽略规则；
- 两套环境的隔离方案和核心版本保护；
- 固定上游 Git commit 与 checkpoint revision；
- 六组官方示例的批量推理、输出保存、校验、计时和显存统计；
- 相机坐标反投影与 LIBERO 设计接口；
- CPU 可执行的单元测试。

仍需用户在 Colab A100 上执行确认：

- Planner 依赖在实际 Colab 镜像中的完整解析与导入；
- 9B BF16 checkpoint 下载和真实模型加载；
- 六组官方样例的非空、合法轨迹结果；
- 实际推理耗时、峰值显存和图像质量。

本地没有 Colab GPU，本仓库不声称已完成上述真实 GPU 验证。

下一阶段见 [任务清单](docs/next_steps.md)。
