# 三维轨迹数据契约

## 目标

`TrajectoryRecord` 是 Planner、LIBERO 和低层策略之间的文件边界。它只依赖 NumPy/Pillow 项目的基础环境，不要求 3D HAMSTER 与 SmolVLA 同进程或同虚拟环境。

## 必填语义

- `trajectory_id`：路径安全且唯一。
- `source`：`prediction`、`mock`、`oracle` 或 `other`。
- `episode`：dataset、suite、task、episode、init state、source observation index。
- `model`：名称、模型版本/revision、代码 revision。
- `instruction` 与输入 RGB SHA-256。
- `waypoints_xyz_m`：`(N,3)`、有限、米制坐标；`coordinate_frame` 必须显式给出。
- `valid_mask`、连续 `waypoint_order` 和逐路点 gripper event。
- K、图像分辨率、`T_world_camera`、depth 单位和 frame 约定。
- `available_at_observation_index`：低层策略最早可使用该轨迹的时刻。
- `uses_future_expert_information`：是否包含当前 observation 之后的专家信息。

## 目录格式

    <trajectory_id>/
    ├── metadata.json      # 身份、模型、frame、标定、因果与 provenance
    ├── trajectory.npz     # XYZ、mask、order、gripper event；allow_pickle=False
    ├── waypoints.csv      # 人工审计与跨工具查看
    └── manifest.json      # 各文件 SHA-256 与 record digest

保存采用临时目录后原子 rename。加载会核查 schema、固定文件集、全部 SHA-256、record digest、trajectory ID、路点数、order、数组形状与记录自身校验。

## 因果规则

正常训练和推理只能读取：

    source in allowed_sources
    source_observation_index <= current_observation_index
    available_at_observation_index <= current_observation_index
    uses_future_expert_information == false

Provider 在同一 episode 中选择当前可用的最新记录。没有可用轨迹时，默认返回固定 K 的零坐标和全 false mask；训练和推理因此使用相同回退接口。

专家未来轨迹只允许用于 diagnostic / upper-bound：必须标为 `source=oracle` 和 `uses_future_expert_information=true`。普通 `load_trajectory_record` 与 `TrajectoryConditionProvider` 默认拒绝，只有明确同时打开 diagnostic 参数才能读取。

## 坐标规则

- 缓存的 3D 路点不保存为模糊的“XYZ”，必须同时记录 frame 和米制单位。
- 3D HAMSTER 原始 UVD 仍保存在原始预测 artifacts，不冒充 world XYZ。
- 当前 04 输出使用 `libero_mujoco_world`；尚未核实 world-to-base 时，不称为机器人基座动作。
- K 必须对应记录中的 RGB/depth 分辨率；`T_world_camera` 表示 `p_world=T_world_camera@p_camera`。

## Batch 交换

真实 SmolVLA Stage C 使用单独的 tensor batch artifact：

    <batch>/
    ├── batch_manifest.json
    └── batch_arrays.npz

它只接收非 object NumPy/Torch tensor，禁止 pickle，校验 SHA-256、key、shape 和 dtype。Stage C 还要求 metadata 明确标记：

- `preprocessed_for_smolvla=true`
- `causality_checked=true`

这避免把 raw dataset sample 或专家未来信息误送入低层策略。
