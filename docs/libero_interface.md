# LIBERO 接口预留

本阶段只有契约设计，不导入 LIBERO，不修改 SmolVLA。

## 数据流

    LIBERO RGB-D observation
        -> 明确深度编码和米制缩放
        -> 保存 RGB、depth、K、外参约定、末端位置
        -> 3D HAMSTER 输出 normalized UVD
        -> K 反投影到 camera XYZ
        -> 经已核实方向的外参变换到 world/base XYZ
        -> 缓存轨迹和全部元数据
        -> 后续供 SmolVLA 训练或评估读取

## 必须显式记录

- RGB 与 depth 的 shape、时间同步和像素对齐关系；
- depth 原始编码、无效值、缩放到米的系数；
- 相机内参对应的图像分辨率；
- 外参矩阵表示 camera-to-world 还是 world-to-camera；
- 世界、机器人基座、末端执行器各自的 frame 名；
- 轨迹生成使用的模型 revision、代码 commit、指令和输入哈希；
- Planner 原始文本、UVD、camera XYZ、最终目标坐标和校验结果。

## 暂不做的假设

- 不假设 LIBERO depth 已经是米；
- 不假设 depth buffer 是线性相机深度；
- 不假设官方相机外参方向；
- 不假设 Planner 路点可直接作为机器人基座坐标；
- 不假设 SmolVLA 已有稳定的轨迹条件输入字段。

src/trajectory_guidance/libero_interface.py 定义了无模拟器依赖的数据结构与 Protocol。真正接入时先写一个最小 observation probe，保存一个 episode 的原始数值和标定，再决定转换代码。
