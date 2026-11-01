# SmolVLA 轨迹条件设计

## 固定代码依据

审计对象是 LeRobot commit `3f797de5952432284a7c77ad2760218cb74e2f26` 的 [SmolVLA 实现](https://github.com/huggingface/lerobot/blob/3f797de5952432284a7c77ad2760218cb74e2f26/src/lerobot/policies/smolvla/modeling_smolvla.py)。该版本在 Action Expert suffix 中将 noisy action 与 timestep 经 `action_time_mlp_out` 映射到 expert hidden；训练仍由原 `VLAFlowMatching.forward` 构造 Flow Matching 输入并计算原损失，采样也复用同一 suffix。

因此 V1/V2 不修改第三方库，而是给已加载 policy 安装 wrapper，在 `model.action_time_mlp_out` 的输出 `(B,T,H)` 上加 `(B,1,H)` 条件。原 action input/output projection、动作维度、VLM、Action Expert 主体、Flow Matching target 和 loss 都不替换。

## 局部路点

输入字段：

- `trajectory.local_waypoints_xyz_m`：`(B,K,3)`；
- `trajectory.local_waypoints_valid`：`(B,K)`；
- `trajectory.current_eef_xyz_m`：`(B,3)`。

从完整轨迹中选择当前 EEF 最近的有效点，再取它及其有序后继点，末尾不足 K 时零填充。rollout 可携带 `min_progress_index`，最近点搜索不能回到该索引之前，避免自交路径反向跳转。该局部窗口原则参考 [3DWay](https://github.com/ziqin-h/3DWay)，但实现独立于 π0，也不复制其网络。

路点先转换为 EEF-relative 坐标，再按每轴米制尺度归一化，与 mask bit 拼接并展平，通过两层小 MLP 得到 expert hidden condition。全缺失或全 invalid 时，condition 严格乘零，所以 E1/E2 回退到基础 policy 路径。

## E0 / E1 / E2

- **E0**：原始 SmolVLA，不创建 wrapper。
- **E1**：上述局部路点 MLP，gain 为 1。
- **E2**：在 E1 上乘可关闭的 geometry gain。

E2 的第一版 anomaly feature：

1. 有效路点落在配置 workspace 外的比例；
2. 相邻有效路点超过阈值的跳跃严重度；
3. 当前 EEF 到最近有效路点超过阈值的偏离严重度。

这些特征与 gain 都用于实验控制，不是校准的规划正确概率。配置中显式保存 `gate_is_calibrated_probability=false`。

## 接口与限制

- `forward`、`predict_action_chunk`、`select_action` 均由同一 wrapper 激活条件。
- wrapper 保留基础 policy 的 `config`，单独保存 `trajectory_config`。
- adapter 可单独取参数和 state dict，也可随完整 wrapper checkpoint 保存。
- 缺少 points/mask 时走零条件；只给其中一个会报错；有有效路点但没有 EEF 会报错。
- 当前 hook 未验证 `torch.compile`，实验配置强制 `compile_model=false`。
- CPU 测试使用 `TinySmolVLAPolicy`，只验证形状、梯度和恢复；真实 LeRobot forward/backward 必须在 Stage C 验收。
