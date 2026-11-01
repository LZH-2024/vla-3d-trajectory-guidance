# LIBERO 接口预留

当前阶段已经实现单帧 observation probe，但仍不调用 Planner、不修改 SmolVLA。

## 数据流

    LIBERO RGB-D observation
        -> 明确深度编码和米制缩放
        -> 保存 RGB、depth、K、外参约定、末端位置
        -> 3D HAMSTER 输出 normalized UVD
        -> K 反投影到 camera XYZ
        -> 经已核实方向的外参变换到 world/base XYZ
        -> 缓存轨迹和全部元数据
        -> 后续供 SmolVLA 训练或评估读取

## 已核实的 LIBERO 约定

针对固定的 hf-libero 0.1.4 / robosuite 1.4.0：

- `OffScreenRenderEnv(camera_depths=True)` 才会返回 `*_depth`；LeRobot 当前封装默认只暴露 RGB，因此 probe 直接使用底层 `OffScreenRenderEnv`；
- robosuite 默认图像约定是 OpenGL；probe 在创建环境前显式设置 `IMAGE_CONVENTION="opencv"`，使 RGB 与 depth 都使用左上角原点、u 向右、v 向下；
- 原始 `*_depth` 是 `[0,1]` 的非线性 MuJoCo depth buffer，不是米；
- probe 调用 robosuite 官方 `get_real_depth_map`，使用场景 near/far 把它转换为 camera z-depth（米）；
- `get_camera_intrinsic_matrix` 返回与当前 256×256 渲染分辨率匹配的 K；
- `get_camera_extrinsic_matrix` 返回相机在 world 中的位姿，本项目命名为 `T_world_camera`；其逆命名为 `T_camera_world`；
- `robot0_eef_pos` 来自 MuJoCo 全局 site position，记录为 `libero_mujoco_world`；`robot0_eef_quat` 顺序为 `xyzw`。

不要把显示用 depth preview 当数值深度。后续几何只能读取 `*_depth_m.npy` 和对应 `*_calibration.json`。

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

`src/trajectory_guidance/libero_interface.py` 保留无模拟器依赖的数据结构与 Protocol。`src/trajectory_guidance/libero_probe.py` 和 `scripts/run_libero_observation_probe.py` 实现最小单帧探测，输出：

- `observation_manifest.json`；
- `observation_arrays.npz`；
- 双相机 RGB、原始 depth buffer、米制 depth 与 preview；
- 双相机 K、`T_world_camera`、`T_camera_world`；
- 稀疏 camera/world 点云样本；
- 末端 world position、`xyzw` quaternion 与 `T_world_eef`。

下一步只在 03 Notebook 通过后，用同一帧数据做 Planner 单帧调用与几何检查。
