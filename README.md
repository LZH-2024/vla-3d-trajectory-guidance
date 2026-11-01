# VLA 3D Trajectory Guidance — Step by Step

本分支从精简工作区重新开始。旧工程保留在 `main`，本分支每次只实现一个经过讨论的小步骤。

当前唯一目标：先看懂并复现一次 3D HAMSTER 的三维轨迹预测（RGB-D + 指令 → UVD → XYZ），暂不接入 SmolVLA、训练或自动化框架。

不要在未经确认时新增后续阶段代码。需要 GPU 的步骤仅准备清楚的 Notebook，运行交由用户决定。
