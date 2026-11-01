# AGENTS.md

## 项目目标

当前唯一主线是完成 3D HAMSTER 官方推理复现，为后续 LIBERO 可行性验证建立稳定基础。不要把本阶段扩展成模型训练、SmolVLA 修改、轨迹融合或论文写作任务。

## 环境纪律

- 参考环境的受保护版本来自 configs/planner_colab.json。
- 3D HAMSTER 只能使用 /content/.venv_planner。
- 后续 LeRobot / SmolVLA 只能使用 /content/.venv。
- Notebook 默认内核不是目标环境。安装、Python 执行和版本检查必须显式调用目标环境的 bin/python。
- 不得静默升级或替换 Torch、Torchvision、TorchCodec、NumPy、NVIDIA NPP、Transformers。
- 不得把 3D HAMSTER 与当前 LeRobot 强装到同一个环境。
- xformers 是可选优化，默认不安装；若要启用，必须先说明可能改变 Torch 解析结果，并重新输出完整版本记录。

## 上游固定点

- 3D HAMSTER commit：97216a8493f46301bf569d398462b8bb21c458c5
- Checkpoint revision：ddc5987a56cdcb14e5e2297817612532e46e912b
- LeRobot 审计 commit：3f797de5952432284a7c77ad2760218cb74e2f26

如需更新固定点，先重新核查官方依赖，更新兼容性说明和配置，再运行测试。不要只改 Notebook 中的一行版本号。

## 代码组织

- Notebook 只负责运行组织、参数设置和结果展示。
- 可复用逻辑放在 scripts/ 或 src/trajectory_guidance/。
- 生成数据写入 outputs/，不要提交模型权重、数据集、缓存或实验产物。
- setup_env_reference.ipynb 是用户提供的环境基准，除非用户明确要求，否则不要修改。

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
- 所有代码单元格无已保存输出；
- 所有模型相关命令都显式使用 /content/.venv_planner/bin/python；
- 没有声称未经执行的 GPU 结果。

Colab 验收标准：

- 环境记录显示受保护版本全部匹配；
- CUDA 冒烟通过；
- checkpoint revision 匹配；
- 官方六例都完成；
- 每例输出非空、有限、u/v 在 0 到 1000、depth 在 0.01 到 10 米；
- 每例都有 JSON、CSV、overlay 和耗时/显存记录。
