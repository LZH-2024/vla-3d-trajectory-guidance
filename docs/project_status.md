# 项目进度

## 已有真实服务器证据

- 01/02：3D HAMSTER 环境、checkpoint、CUDA 与官方六例已通过；Notebook 中保留真实输出。
- 03：LIBERO 双相机 RGB-D、米制 depth、K、外参、末端位姿和 EGL 已通过；Notebook 中保留真实输出。
- 04 / Stage B：LIBERO task0/init0 已生成 3 个 prediction world XYZ 路点；数值与坐标变换检查通过，overlay 已目视检查。部分路点与观测表面 depth 差超过 0.25 m，属于需在 rollout 中继续验证的几何诊断。
- Stage C：固定 `lerobot/smolvla_libero` 已完成 E0/E1/E2 真实 forward/backward/inference；动作维度均为 `(1, 50, 7)`，adapter/gate 梯度有限且非零。该次使用零 action Smoke target，loss 不可作为效果比较。
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

本轮本地验收：`python -m unittest discover -s tests` 共 50 项通过；独立 `run_cpu_mock_experiments.py` 的 E0/E1/E2 两步 Mock、梯度、checkpoint 与恢复均通过。该数字只描述当前代码测试，不是实验指标。

## 当前新增、待服务器 GPU 的部分

- `lerobot/libero` dataset revision 固定及真实 demonstration 初始帧筛选；
- Stage B probe 与 demonstration 的语言、EEF、双视角图像唯一匹配；
- 真实 action chunk 的预处理、无 padding 检查及 prediction trajectory 条件装配；
- E0/E1/E2 各 2-step optimizer smoke、trainable subset checkpoint 和 task0/init0 单次闭环；
- Stage D 自动 orchestration、失败日志和 ledger 汇总。

本次按用户要求不在本地运行 CPU 测试，新增部分直接交给服务器 GPU fail-stop 验收。

## 待 GPU 验证

1. 在已有 A/B/C 均 passed 的同一 `RUN_DIR` 中执行 05 的 Stage D 单元。
2. 先检查 observation match；不唯一或超阈值时停止，不做 optimizer step。
3. 检查 E0/E1/E2 的 2-step、checkpoint、rollout trace 与失败日志。
4. 不把三组 loss 或单次 rollout 成败作为论文结论。
5. D 管线通过后，再设计覆盖多个 demonstration/初始状态的真实训练集和 Stage E 预算。
