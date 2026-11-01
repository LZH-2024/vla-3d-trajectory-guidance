# 下一阶段任务清单

按顺序推进，前一项不通过就停止，不为小问题重复租用 GPU。

- [x] 3D HAMSTER 固定环境与官方六例真实复现。
- [x] LIBERO 单帧 RGB-D、米制 depth、K、外参和末端位姿真实验证。
- [x] 04 单帧 Planner 代码、Mock 全链路与统一轨迹缓存。
- [x] E0/E1/E2 配置、V1 MLP、V2 geometry gain、CPU 梯度与恢复测试。
- [x] 05 集中验收 Notebook 和 A→E fail-stop ledger。
- [x] **Stage A**：复核两个环境、checkpoint、probe 与标定。
- [x] **Stage B**：运行 04 真实 Planner；检查 overlay、world XYZ、异常跳跃、EEF 偏离与耗时。
- [x] B 的数值链路和 overlay 已检查；保留 surface-depth 诊断供闭环继续判断。
- [x] 05 自动准备固定 LeRobot preprocessor 输出的单个因果 smoke batch artifact，不使用未来专家轨迹，并明确标记 synthetic action/gripper 限制。
- [x] **Stage C**：真实 SmolVLA E0/E1/E2 forward/backward、adapter 梯度、动作维度与显存均通过。
- [x] 根据固定 LeRobot API 补齐 pinned dataset、真实 action/preprocessor 与闭环装配。
- [ ] **Stage D**：执行自动 observation match、E0/E1/E2 各 2 step 和各 1 次 rollout；检查 checkpoint 与结果记录。
- [ ] D 通过后确定正式训练步数、episode 数和 Google Drive 目录。
- [ ] **Stage E**：使用相同任务、示范、初始状态、动作设置和预算比较 E0/E1/E2。

当前不要执行正式训练、完整闭环或复杂融合模块。服务器原始 03/04 outputs 可在方便时归档，不阻塞本地开发。
