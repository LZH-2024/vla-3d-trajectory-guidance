# AGENTS.md

## 当前目标

当前主线是：在不占用付费 GPU 的前提下，完成 LIBERO 单帧规划、统一轨迹缓存、SmolVLA V1/V2 条件模块、CPU/Mock 测试与集中 GPU 验收准备。

本阶段不得执行正式训练或闭环评估，不得把 CPU Mock、静态检查或小模型结果写成真实 SmolVLA / LIBERO 实验结论。不要扩展到复杂融合模块或论文写作。

## 环境纪律

- 受保护版本来自 `configs/planner_colab.json` 与 `configs/libero_probe_colab.json`。
- 3D HAMSTER 只能使用 `/content/.venv_planner`。
- LeRobot / SmolVLA / LIBERO 只能使用 `/content/.venv`。
- Notebook 默认内核不是目标环境；安装、版本检查和模型命令必须显式调用相应环境的 `bin/python`。
- 不得静默升级或替换 Torch、Torchvision、TorchCodec、NumPy、NVIDIA NPP、Transformers。
- 不得把 3D HAMSTER 与 LeRobot 强装到同一环境。
- `xformers` 默认不安装；启用前必须说明 Torch 解析风险，并重新输出完整版本记录。
- 本地离线测试不得下载大模型或数据集，也不得调用 CUDA。

## 上游固定点

- 3D HAMSTER commit：`97216a8493f46301bf569d398462b8bb21c458c5`
- 3D HAMSTER checkpoint revision：`ddc5987a56cdcb14e5e2297817612532e46e912b`
- LeRobot 审计 commit：`3f797de5952432284a7c77ad2760218cb74e2f26`
- LIBERO assets revision：`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`

更新固定点前必须重新审计依赖、实际 SmolVLA 接口、兼容性文档和配置，并重新运行测试。

## 文件与 Notebook

- Notebook 只负责执行组织、参数设置和结果展示；复用逻辑放在 `scripts/` 或 `src/trajectory_guidance/`。
- `setup_env_reference.ipynb` 未经用户明确要求不得修改。
- 01、02、03 中已有真实服务器输出；不得删除、清空或伪造。04/05 尚未执行的单元保持无伪造输出。
- 所有生成数据写入 `outputs/`，不得提交权重、数据集、缓存、检查点或实验产物。
- Mock 输出必须同时标记 `mock=true` 和“非真实模型结果”，并与 prediction / Oracle 目录分离。

## 数据、坐标与因果约束

- Planner 输入 depth 必须是与 RGB 对齐的 camera z-depth（米）。不得假设 LIBERO 原始 depth buffer 已经是米。
- Planner 输出是 normalized U/V `[0,1000]` 加米制 depth，不是 world XYZ。
- 反投影前必须确认 K 与图像分辨率匹配；使用外参时记录矩阵方向、源 frame 和目标 frame。
- 统一缓存中的三维轨迹必须记录 episode/task/init state、observation index、模型 revision、输入哈希、语言、frame、单位、valid mask、gripper event、标定和来源。
- 正常训练/推理只可选择 `available_at_observation_index <= 当前 observation` 的轨迹。
- 包含专家未来信息的数据必须标记为 Oracle / diagnostic，正常 loader 默认拒绝；不得作为正常推理条件。

## SmolVLA 扩展边界

- E0：原始 SmolVLA；E1：固定局部 3D 路点 + 小 MLP；E2：E1 + 可关闭的几何异常门控。
- 保留原动作维度、Flow Matching 目标/损失、VLM 和 Action Expert 主结构。
- V1/V2 通过 wrapper 在固定 LeRobot 版本的 `model.action_time_mlp_out` 输出处加条件，不直接修改第三方源码。
- 有轨迹和无轨迹必须走同一训练/推理接口；缺失或全无效轨迹回退为精确零条件。
- 几何门控只能表述为异常条件 gain，不得称为校准后的规划正确概率。
- `torch.compile` 路径尚未验证，配置必须保持 `compile_model=false`。

## 验证要求

修改 Python 后至少运行：

    python -m unittest discover -s tests

修改 Notebook 后必须检查：

- JSON 可解析，所有代码单元语法可编译；
- 已有真实输出原样保留；新增 Notebook 不含伪造输出；
- Planner 模型命令显式使用 `/content/.venv_planner/bin/python`；
- SmolVLA / LIBERO 模型命令显式使用 `/content/.venv/bin/python`；
- 文档明确区分 CPU 测试、静态检查与待 GPU 验证。

集中 GPU 验收必须按 A→B→C→D→E 执行，ledger 前置阶段不通过时后续阶段拒绝启动：

- A：双环境、固定版本、checkpoint、缓存、RGB-D 与标定；
- B：04 的真实高层规划与几何目视验收；
- C：真实 SmolVLA E0/E1/E2 forward/backward/梯度/显存；
- D：单任务、少量训练与 rollout；
- E：正式实验。

不得因某阶段失败而自动继续消耗 GPU。D/E 预算在 C 通过前保持未设置。

## 持久化

当前离线开发和短冒烟默认写 `/content/.../outputs` 即可。开始长训练后再把 run root 指向已挂载的 Google Drive，并保存权重缓存、轨迹、检查点、配置、评估和失败日志；不要把大型文件提交 Git。
