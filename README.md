# VLA 3D Trajectory Guidance

本项目研究 **3D HAMSTER 高层三维轨迹 + SmolVLA 低层动作专家 + 轻量轨迹条件**。当前目标不是直接开始大规模训练，而是先把数据、几何、条件接口和分阶段验收做稳，减少付费 GPU 被路径、格式或维度错误浪费的风险。

## 当前状态

截至 2026-11-01：

| 项目 | 状态 | 证据边界 |
|---|---|---|
| 3D HAMSTER 官方六例 | **已在服务器真实运行通过** | 01/02 Notebook 保存输出；六例均有合法轨迹 |
| LIBERO 单帧 RGB-D、K、外参、末端位姿 | **已在服务器真实运行通过** | 03 Notebook 保存输出 |
| LIBERO 单帧 3D HAMSTER | **已在服务器真实运行通过** | Stage B 保存 3 个 world XYZ 路点、几何诊断与 overlay |
| Mock Planner 全链路 | **CPU 测试通过** | 固定 UVD，非真实模型结果 |
| 统一轨迹缓存与因果读取 | **CPU 测试通过** | JSON/CSV/NPZ、SHA-256、Oracle 默认拒绝 |
| SmolVLA E1/E2 条件模块 | **CPU 小模型测试通过** | 维度、前后向、梯度、缺失回退；非真实 SmolVLA |
| 真实 SmolVLA E0/E1/E2 冒烟 | **已在服务器真实运行通过** | Stage C 完成真实 forward/backward/inference 与 adapter 梯度检查 |
| Stage D 单任务 optimizer/rollout | **代码已准备，待服务器 GPU** | 自动匹配真实 demonstration；每组仅 2 step + 1 rollout，不代表有效训练 |
| 正式 E0/E1/E2 实验 | **未开始** | 正式预算故意保持未设置 |

本地没有执行本次新增 Stage D、正式训练或闭环评估，也没有据此编造任务成功率。Stage C 保存的 loss 只用于有限值检查，不能横向解释模型质量。

## 固定环境与上游

两个环境必须隔离：

| 环境 | 用途 | 关键依赖 |
|---|---|---|
| `/content/.venv_planner` | 3D HAMSTER 推理 | Transformers 4.57.6；Hub `<1.0` |
| `/content/.venv` | LIBERO、LeRobot、SmolVLA | Transformers 5.5.4；Hub 1.33.0 |

共同受保护基础为 Python 3.12.13、Torch 2.11.0+cu128、Torchvision 0.26.0+cu128、TorchCodec 0.11.0+cu128、NumPy 2.0.2、NVIDIA NPP 12.4.1.87。不得为安装成功而静默改变这些版本。

固定点：

- 3D HAMSTER commit：`97216a8493f46301bf569d398462b8bb21c458c5`
- 3D HAMSTER checkpoint revision：`ddc5987a56cdcb14e5e2297817612532e46e912b`
- LeRobot 审计 commit：`3f797de5952432284a7c77ad2760218cb74e2f26`
- LIBERO assets revision：`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`

依赖冲突与版本依据见 [环境兼容性说明](docs/environment_compatibility.md)。上游参考：[3D HAMSTER](https://github.com/DAVIAN-Robotics/3D_HAMSTER)、[固定 LeRobot SmolVLA 源码](https://github.com/huggingface/lerobot/blob/3f797de5952432284a7c77ad2760218cb74e2f26/src/lerobot/policies/smolvla/modeling_smolvla.py)、[3DWay](https://github.com/ziqin-h/3DWay)。

## 项目结构

    .
    ├── configs/
    │   ├── experiments/                 # E0/E1/E2 的共享预算与条件开关
    │   └── gpu_acceptance_colab.json    # A→E 验收配置，D/E 默认未设置
    ├── docs/                            # 环境、数据格式、SmolVLA 与验收说明
    ├── notebooks/
    │   ├── setup_env_reference.ipynb    # 用户环境基准，不修改
    │   ├── 01_planner_setup.ipynb       # 已有真实服务器输出
    │   ├── 02_planner_inference.ipynb   # 已有真实服务器输出
    │   ├── 03_libero_observation_probe.ipynb # 已有真实服务器输出
    │   ├── 04_libero_planner_single_frame.ipynb
    │   └── 05_gpu_acceptance.ipynb
    ├── scripts/                         # Mock、检查、真实冒烟与分阶段入口
    ├── src/trajectory_guidance/         # 几何、缓存、条件、数据与恢复逻辑
    ├── tests/                           # CPU/Mock 单元测试
    └── outputs/                         # 生成结果，Git 忽略

## 离线开发入口

下面两条命令不下载模型、不需要 GPU：

    python scripts/run_mock_planner_validation.py
    python scripts/run_cpu_mock_experiments.py

第一条构造可控 RGB-D、K、`T_world_camera` 和固定 UVD，复用真实几何代码完成：

- UVD → camera XYZ → LIBERO world XYZ；
- JSON、CSV、NPZ、overlay 与统一缓存保存；
- 合法、空轨迹、U/V 越界、无效 depth 处理。

所有文件都标记 `mock=true`，写入 `outputs/mock_planner/`，不会混入真实预测。

第二条使用小型 Action Expert 形状替身测试 E0/E1/E2：

- 训练和推理走同一个条件 wrapper；
- 固定局部路点、valid mask、无轨迹精确零回退；
- 新增参数前向、反向与有限梯度；
- checkpoint 保存、恢复和动作输出维度。

它只验证计算和接口，不等于真实 SmolVLA 训练。

## 统一三维轨迹格式

每条记录是一个可跨虚拟环境交换的目录：

    <trajectory_id>/
    ├── metadata.json
    ├── trajectory.npz
    ├── waypoints.csv
    └── manifest.json

记录包含 episode/task/init state/observation、模型版本与代码 revision、输入 RGB SHA-256、语言指令、米制 XYZ、frame、顺序、valid mask、gripper event、K、`T_world_camera`、来源和可用时刻。加载时会验证所有文件 SHA-256、数组形状、顺序和 schema。

正常条件读取遵守：

    record.available_at_observation_index <= 当前 observation_index

包含专家未来轨迹的数据必须标为 Oracle / `uses_future_expert_information=true`，正常 loader 与 provider 默认拒绝。详见 [轨迹数据契约](docs/trajectory_data_contract.md)。

## SmolVLA 条件接口

在固定 LeRobot commit 中，SmolVLA 的 noisy action 与 timestep 在 Action Expert suffix 中经过 `action_time_mlp_out`，随后保留原 Flow Matching 目标和 MSE 损失。本项目不改第三方源码，而是包装已加载 policy，在该层输出处增加一个固定维度条件向量：

    当前 EEF + K 个局部 world XYZ + valid mask
        → EEF-relative / per-axis scale
        → 小型 MLP
        → Action Expert hidden condition

局部窗口按“当前 EEF 最近的有效路点 + 有序后继点”选择，并用单调 `min_progress_index` 防止交叉路径回跳。思路参考 3DWay 的局部 waypoint 选择，但没有复制 π0 实现。

- **E0**：原始 SmolVLA，不安装 wrapper。
- **E1**：局部三维路点 + MLP additive condition。
- **E2**：E1 + 几何异常 gain。

E2 第一版特征只有工作空间越界比例、相邻路点跳跃和当前 EEF 偏离。该值是可关闭的异常条件增益，**不是规划正确概率**。`torch.compile` 尚未真实验证，配置固定为 `false`。详见 [SmolVLA 条件设计](docs/smolvla_trajectory_conditioning.md)。

## 实验与恢复框架

`configs/experiments/` 共享相同 suite、task、init state、seed、batch size、学习率、smoke/pilot 预算，只改变条件开关。LIBERO dataset 固定为 `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`；本地 SmolVLA path 仍由 Stage C 证据自动解析，正式训练步数保持空值。

已实现并经过 CPU 测试：

- 轨迹与 action chunk 的 observation 时间索引对齐；
- 因果可用轨迹选择和缺失回退；
- 有限步训练循环、梯度裁剪、按步 checkpoint；
- config digest 校验和 checkpoint 恢复；
- 每 episode 评估结果、去重与汇总恢复；
- 通用闭环 episode plan 入口；失败写独立日志且不标记完成，下次可重试；
- 已预处理 tensor batch 的无 pickle NPZ + manifest 交换格式。

可先生成不加载模型的 run plan：

    python scripts/prepare_experiment_run.py --experiment E1 --phase smoke

它只固化 resolved config、digest、checkpoint/metrics/evaluation 路径，不会开始训练。`formal_steps` 当前为空，因此 formal plan 会主动拒绝生成。

Stage D 使用 `prepare_smolvla_stage_d_inputs.py` 将 Stage B 观测水平翻转到固定数据集的实际图像约定，再与全部同任务 demonstration 起点按语言、EEF 和双 RGB 视角做阈值匹配。只有唯一匹配通过，才生成带真实 `50×7` action target 的 batch。`run_stage_d_single_task_pilot.py` 随后让 E0/E1/E2 分别从同一固定 checkpoint 开始，使用相同 batch 和逐步 RNG seed 做 2 次 optimizer update，再各运行一次 task0/init0 闭环。此入口是 GPU plumbing gate；单 chunk、2 step 和单 rollout 都不是论文实验。

## 集中 GPU 验收

使用 [05_gpu_acceptance.ipynb](notebooks/05_gpu_acceptance.ipynb)，每阶段单独执行：

1. **A 环境与数据**：两个解释器、固定版本、CUDA、checkpoint revision、RGB-D 与标定。
2. **B 高层可行性**：执行一次 04 真实推理，保存图、world XYZ、诊断、耗时和统一缓存；结果明显不可用则停止。
3. **C SmolVLA 冒烟**：05 自动安装固定 `smolvla` extra、断点续传固定的 `lerobot/smolvla_libero` checkpoint，并由 B 的观测与 prediction 轨迹生成已预处理、因果检查的 smoke batch；随后检查 E0/E1/E2 真实 forward/backward、动作维度、梯度、耗时和显存，不做 optimizer step。
4. **D 单任务 pilot**：A/B/C 通过后自动准备真实匹配 batch，依次运行 E0/E1/E2 的 2-step optimizer smoke 与单次 rollout；无需填写路径。
5. **E 正式实验**：只在 D 通过并确定预算后填写命令。

`acceptance_ledger.json` 记录每次尝试、成功/失败、耗时、GPU 和 artifacts。前置阶段未通过时后续阶段拒绝启动；重复运行已通过阶段不会覆盖证据。Stage C 的 Notebook 单元会在执行前明确显示并固定下载来源；已存在的权重与 batch 会校验后复用。

## 持久化与成本

当前离线开发和短冒烟继续使用 `/content` 即可。开始长训练后，将 `RUN_DIR` 或 `VLA_PERSISTENT_ROOT` 指向已挂载的 Google Drive，用于权重缓存、轨迹、检查点、配置、评估与失败日志。大型文件始终由 `.gitignore` 排除，不提交 Git。

3D HAMSTER 和 SmolVLA 不要求同时驻留同一 Python 进程：高层轨迹通过文件缓存跨环境交换，可分别释放显存。

## 测试

    python -m unittest discover -s tests

测试覆盖原有几何/LIBERO 校验，以及轨迹缓存、局部选择、条件张量、门控、梯度、action 对齐、Mock artifacts、batch 交换、配置切换、checkpoint/evaluation 恢复和 fail-stop ledger。本次按用户要求未在本地重跑 CPU 测试；新增 Stage D 直接留给服务器 GPU 验收。

下一步状态见 [项目进度](docs/project_status.md) 和 [任务清单](docs/next_steps.md)。
