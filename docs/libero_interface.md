# LIBERO 接口预留

单帧 observation probe 已在服务器通过。当前文件桥接既支持 04 的单帧 Planner，也把合法 world XYZ 写入统一轨迹缓存；离线阶段仍不运行 policy、不发送 simulator action。

## 数据流

    LIBERO RGB-D observation
        -> 明确深度编码和米制缩放
        -> 保存 RGB、depth、K、外参约定、末端位置
        -> 3D HAMSTER 输出 normalized UVD
        -> K 反投影到 camera XYZ
        -> 经已核实方向的外参变换到 world/base XYZ
        -> 统一 TrajectoryRecord（frame、单位、mask、event、标定、来源、因果时刻）
        -> 局部路点选择与 SmolVLA E1/E2 条件接口

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

`src/trajectory_guidance/libero_planner.py` 与 `scripts/run_libero_planner_single_frame.py` 现已实现同一帧的 Planner 调用准备、几何检查和 prediction cache：

- 自动选择最新的有效 probe；
- 复核 manifest、坐标约定、输入文件 SHA-256、depth、K 与 `T_world_camera`；
- 由 `/content/.venv_planner` 读取 `/content/.venv` 生成的文件，不混装依赖；
- 保存原始 UVD、camera XYZ、LIBERO world XYZ 和 2D overlay；
- 用 XYZ 重投影回 UVD，检查数值闭环和刚体段长不变性；
- 记录路点与场景表面 depth 差值，但不把场景 depth 覆盖到 Planner 路点上。
- 仅在全部校验通过时保存 source=`prediction` 的 JSON/CSV/NPZ 轨迹记录，并保留输入哈希、模型 revision 和相机标定。

04 的通过只能说明单帧输出在当前标定下可转换且数值自洽，不能证明低层控制可执行或 LIBERO 任务成功。

低层读取由 `trajectory_dataset.py` 执行因果筛选：轨迹必须在当前 observation 时刻已经可用；Oracle / 未来专家轨迹默认拒绝。完整格式见 [轨迹数据契约](trajectory_data_contract.md)。
