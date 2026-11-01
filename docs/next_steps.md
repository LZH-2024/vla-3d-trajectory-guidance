# 下一阶段任务清单

按顺序推进，前一项不通过就停止，不为小问题重复租用 GPU。

- [x] 3D HAMSTER 固定环境与官方六例真实复现。
- [x] LIBERO 单帧 RGB-D、米制 depth、K、外参和末端位姿真实验证。
- [x] 04 单帧 Planner 代码、Mock 全链路与统一轨迹缓存。
- [x] E0/E1/E2 配置、V1 MLP、V2 geometry gain、CPU 梯度与恢复测试。
- [x] 05 集中验收 Notebook 和 A→E fail-stop ledger。
- [ ] **Stage A**：在下一次 GPU 会话复核两个环境、checkpoint、probe 与标定。
- [ ] **Stage B**：运行 04 真实 Planner；检查 overlay、world XYZ、异常跳跃、EEF 偏离与耗时。
- [ ] 若 B 明显不可用：先分析 Planner/scene mismatch，不进入低层训练。
- [ ] 准备固定 LeRobot preprocessor 输出的单个因果 batch artifact，不使用未来专家轨迹。
- [ ] **Stage C**：检查真实 SmolVLA E0/E1/E2 forward/backward、adapter 梯度、动作维度与显存。
- [ ] 根据 C 的真实结果补齐 dataset/preprocessor 训练装配，避免提前猜接口。
- [ ] **Stage D**：只做单任务、少量 step 和少量 rollout；确认 checkpoint/恢复与结果记录。
- [ ] D 通过后确定正式训练步数、episode 数和 Google Drive 目录。
- [ ] **Stage E**：使用相同任务、示范、初始状态、动作设置和预算比较 E0/E1/E2。

当前不要执行正式训练、完整闭环或复杂融合模块。服务器原始 03/04 outputs 可在方便时归档，不阻塞本地开发。
