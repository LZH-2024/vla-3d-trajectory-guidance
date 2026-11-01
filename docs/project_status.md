# 项目进度

## 已有真实服务器证据

- 01/02：3D HAMSTER 环境、checkpoint、CUDA 与官方六例已通过；Notebook 中保留真实输出。
- 03：LIBERO 双相机 RGB-D、米制 depth、K、外参、末端位姿和 EGL 已通过；Notebook 中保留真实输出。
- 04 / Stage B：LIBERO task0/init0 已生成 3 个 prediction world XYZ 路点；数值与坐标变换检查通过，overlay 已目视检查。部分路点与观测表面 depth 差超过 0.25 m，属于需在 rollout 中继续验证的几何诊断。
- Stage C：固定 `lerobot/smolvla_libero` 已完成 E0/E1/E2 真实 forward/backward/inference；动作维度均为 `(1, 50, 7)`，adapter/gate 梯度有限且非零。该次使用零 action Smoke target，loss 不可作为效果比较。
- Stage D：固定 demonstration 匹配、真实 action batch、E0/E1/E2 两步 optimizer、checkpoint 与 300-step 闭环均已完成；三组均 `success=false`、累计 reward 为 0。它只通过 plumbing gate，未通过任务效果验证。
- 原始服务器 `outputs/` 尚未下载，本地目前以保存的 Notebook 输出为证据。

## 本轮离线实现

- 04 增加可选 CPU Mock 入口；真实 Planner 路径增加统一 prediction cache。
- 轨迹 schema、原子保存、SHA-256 完整性、跨环境加载、因果可用时刻与 Oracle 防泄漏。
- 最近路点 + 有序后继的固定 K 局部窗口、mask、单调 progress 和相对归一化。
- SmolVLA E1 MLP condition wrapper；E2 可关闭 geometry anomaly gain。
- E0/E1/E2 可比较配置、action 时间对齐、有限步训练、checkpoint/config digest 恢复、逐 episode 评估失败日志与恢复。
- Mock Planner、Tiny Action Expert 和 tensor batch artifact。
- A→E fail-stop ledger、真实 Stage C forward/backward 入口和 05 Notebook。

上述模块已由 CPU/Mock 单元测试覆盖；不构成真实 SmolVLA、训练或闭环结果。

本轮本地验收：`python -m unittest discover -s tests` 共 62 项通过；独立 `run_cpu_mock_experiments.py` 的 E0/E1/E2 两步 Mock、梯度、checkpoint 与恢复均通过。该数字只描述当前代码测试，不是实验指标。

## 当前新增、待服务器 GPU 的部分

- 原始 E0（0 optimizer step）单次闭环，用于区分预训练策略行为与两步 smoke update 的影响；
- 已有四条 trace 的动作饱和、EEF 运动、reward、gate gain 与首末截图对比；
- 根据诊断结果决定先修 rollout/preprocessing，还是开始设计多 episode 的有效训练预算。

本地已完成 62 项 CPU 测试；未加载权重、未执行 GPU rollout，也未把 Mock/静态结果当作任务效果。

## 待 GPU 验证

1. 更新代码后只执行 05 的“Stage D 补充诊断”单元，不重跑 A/B/C/D。
2. 对比原始 E0 与两步后的 E0；再比较 E1/E2 的动作饱和、EEF 路径和 gate gain。
3. 检查四组首末截图，确认机器人是否移动、方向是否明显异常、gripper 是否长期饱和。
4. 不把四次单 rollout 或两步 loss 作为论文结论。
5. 先根据诊断修正接口或训练方案，再设计覆盖多个 demonstration/初始状态的 Stage E 预算。
