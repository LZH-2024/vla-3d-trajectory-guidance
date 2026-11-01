# 项目进度

## 已有真实服务器证据

- 01/02：3D HAMSTER 环境、checkpoint、CUDA 与官方六例已通过；Notebook 中保留真实输出。
- 03：LIBERO 双相机 RGB-D、米制 depth、K、外参、末端位姿和 EGL 已通过；Notebook 中保留真实输出。
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

## 只做了静态检查的部分

- 固定 LeRobot commit 的真实 `SmolVLAPolicy.from_pretrained` 装配；
- 真实 preprocessed batch 与 wrapper 的 forward/backward；
- Colab 上 A→E orchestration 的路径和环境调用；
- D/E 外部训练/rollout command（当前保持空值）。

## 待 GPU 验证

1. Stage A 重新确认两个环境与现有缓存。
2. Stage B 运行 04，目视检查真实 Planner 轨迹；不可用则停。
3. 准备一个因果、已预处理的 SmolVLA batch，运行 Stage C E0/E1/E2。
4. 根据 Stage C 的真实维度、显存和接口，补齐完整 LeRobot dataset/preprocessor 装配。
5. 仅在上述通过后配置 D 的单任务 pilot；D 通过再确定 E 的正式预算与 Google Drive 持久化。
