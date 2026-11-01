# AGENTS.md

## 项目目标

3D HAMSTER 官方六例推理已经通过服务器运行验收。当前唯一主线是完成 **LIBERO 单帧 RGB-D observation probe**，核实深度、相机标定和末端位姿，为下一步单帧 Planner 可行性验证建立稳定基础。不要把本阶段扩展成模型训练、SmolVLA 修改、轨迹融合或论文写作任务。

## 环境纪律

- 参考环境的受保护版本来自 configs/planner_colab.json。
- 3D HAMSTER 只能使用 /content/.venv_planner。
- LIBERO probe 与后续 LeRobot / SmolVLA 只能使用 /content/.venv。
- Notebook 默认内核不是目标环境。安装、Python 执行和版本检查必须显式调用目标环境的 bin/python。
- 不得静默升级或替换 Torch、Torchvision、TorchCodec、NumPy、NVIDIA NPP、Transformers。
- 不得把 3D HAMSTER 与当前 LeRobot 强装到同一个环境。
- xformers 是可选优化，默认不安装；若要启用，必须先说明可能改变 Torch 解析结果，并重新输出完整版本记录。

## 上游固定点

- 3D HAMSTER commit：97216a8493f46301bf569d398462b8bb21c458c5
- Checkpoint revision：ddc5987a56cdcb14e5e2297817612532e46e912b
- LeRobot 审计 commit：3f797de5952432284a7c77ad2760218cb74e2f26
- hf-libero：0.1.4（wheel SHA-256：207f76e2f28bff30f78132223d8592fe8f64b1f8fd90ce7024948ada0d7e2c27）
- robosuite：1.4.0
- MuJoCo：3.8.1
- LIBERO assets revision：0b3ea86be5fe169d0fd036ae63d1070ec09e90f6

如需更新固定点，先重新核查官方依赖，更新兼容性说明和配置，再运行测试。不要只改 Notebook 中的一行版本号。

## 代码组织

- Notebook 只负责运行组织、参数设置和结果展示。
- 可复用逻辑放在 scripts/ 或 src/trajectory_guidance/。
- 生成数据写入 outputs/，不要提交模型权重、数据集、缓存或实验产物。
- setup_env_reference.ipynb 是用户提供的环境基准，除非用户明确要求，否则不要修改。
- 01_planner_setup.ipynb 和 02_planner_inference.ipynb 中已有服务器运行输出，按用户要求保留，不得为了格式检查清空。

## 数据与坐标

- Planner 输入深度必须明确转换为米，并与 RGB 对齐。
- Planner 输出约定是归一化 u/v 0 到 1000 加米制 depth。
- 不要把 UVD 称为世界坐标 XYZ。
- 使用相机内参反投影时，必须确认内参与图像分辨率匹配。
- 使用外参前必须记录矩阵方向、源坐标系和目标坐标系。
- 不得假设 LIBERO 默认深度已经是米。

## 验证

修改 Python 逻辑后至少运行：

    python -m unittest discover -s tests

修改 Notebook 后要检查：

- JSON 可解析；
- 不得清空已有服务器输出；新建但尚未运行的 Notebook 保持无伪造输出；
- Planner 模型相关命令显式使用 /content/.venv_planner/bin/python；
- LIBERO probe 的安装、检查和运行显式使用 /content/.venv/bin/python；
- 没有声称未经执行的 GPU 结果。

Colab 验收标准：

- 环境记录显示受保护版本全部匹配；
- CUDA 冒烟通过；
- checkpoint revision 匹配；
- 官方六例都完成；
- 每例输出非空、有限、u/v 在 0 到 1000、depth 在 0.01 到 10 米；
- 每例都有 JSON、CSV、overlay 和耗时/显存记录。

LIBERO probe 验收标准：

- /content/.venv 中受保护核心版本不漂移，LeRobot commit 与 hf-libero/robosuite/MuJoCo 版本匹配；
- MUJOCO_GL=egl，固定 assets revision 匹配；
- agentview 与 eye-in-hand 都得到对齐的 RGB、原始 depth buffer 和米制 depth；
- 原始 depth 在 0 到 1，米制 depth 非空、有限并在合理范围；
- K 与采集分辨率匹配，T_world_camera 和 T_camera_world 都被记录且互逆；
- 末端位置明确属于 LIBERO MuJoCo world frame，四元数顺序记录为 xyzw；
- 生成 manifest、NPY/NPZ、RGB、depth preview 和 calibration JSON；
- 本阶段不调用 Planner 或低层策略。
