# 下一阶段任务清单

按顺序推进，前一项不通过时不要直接进入训练。

- [x] 在 Linux GPU 服务器完整执行 01_planner_setup.ipynb。
- [x] 确认受保护版本、CUDA 冒烟、固定 checkpoint 和模型加载通过。
- [x] 执行 02_planner_inference.ipynb，完成官方六例；六例均非空且合法。
- [ ] 从服务器下载并归档 Planner 原始 outputs；当前本地只有 Notebook 保存输出。
- [x] 实现 03_libero_observation_probe.ipynb、独立环境脚本和 CPU 单元测试。
- [x] 在服务器执行 03 Notebook：
  - 确认 `/content/.venv` 受保护版本与固定上游版本；
  - 确认 MuJoCo EGL 与固定 assets revision；
  - 导出 agentview 和 wrist 的 RGB、原始 depth buffer、米制 depth；
  - 核实 K、`T_world_camera`、`T_camera_world` 和末端 world pose；
  - 目视检查 RGB 与 depth preview 的方向和像素对齐；
  - 保存输出显示完整 manifest 验收通过；原始 artifacts 仍在服务器，暂不下载。
- [x] 实现 04_libero_planner_single_frame.ipynb、单帧桥接脚本和投影闭环测试。
- [ ] 在服务器执行 04 Notebook，完成当前唯一任务：
  - 自动选择最新有效 `agentview` probe，不重新下载数据；
  - 用 `/content/.venv_planner/bin/python` 复核 Planner 环境和 checkpoint provenance；
  - 生成非空合法 UVD，并保存 camera XYZ、LIBERO world XYZ 与 overlay；
  - 检查 UVD 投影闭环、刚体段长一致性、耗时和显存；
  - 目视判断轨迹是否落在合理目标区域；注意 scene depth 差值只是诊断，不是硬门槛。
- [ ] 再确定轨迹缓存 schema 和离线生成成本。
- [ ] 最后才讨论轻量轨迹条件接口、baseline、消融和训练预算。

当前不要训练，不要修改 SmolVLA，不要实现轨迹融合。
