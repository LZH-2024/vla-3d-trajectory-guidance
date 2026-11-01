# 下一阶段任务清单

按顺序推进，前一项不通过时不要直接进入训练。

1. 在 Colab A100 上完整执行 01_planner_setup.ipynb。
2. 保存环境记录，确认受保护版本、CUDA 冒烟和模型加载全部通过。
3. 执行 02_planner_inference.ipynb，完成官方六例。
4. 检查六例是否都有非空合法路点、overlay、JSON、CSV、耗时和峰值显存。
5. 若失败，先区分环境失败、checkpoint 失败、输入失败、解析失败或模型输出失败，再修复。
6. 六例稳定后，只做一个 LIBERO observation probe：
   - 导出一帧 RGB-D；
   - 核实 depth buffer 编码和米制转换；
   - 核实内外参、矩阵方向和末端坐标系；
   - 不训练策略。
7. 用单帧 LIBERO 数据调用 Planner，检查 UVD、camera XYZ 和 world/base XYZ 的几何合理性。
8. 确定轨迹缓存 schema 和离线生成成本。
9. 只有上述检查通过后，再在独立 /content/.venv 中固定 LeRobot / SmolVLA 环境。
10. 最后才讨论轻量轨迹条件接口、baseline、消融和训练预算。
