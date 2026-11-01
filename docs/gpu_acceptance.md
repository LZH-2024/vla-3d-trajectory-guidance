# 集中 GPU 验收

入口：`notebooks/05_gpu_acceptance.ipynb` 或：

    python scripts/run_gpu_acceptance_stage.py \
      --stage A \
      --run-dir /content/vla-3d-trajectory-guidance/outputs/gpu_acceptance/<run_id>

每次只执行一个阶段。`acceptance_ledger.json` 要求前置阶段全部 `passed`，否则后续命令在加载模型前拒绝执行。失败会在 `stage_x/attempt_NNN/` 保存 `failure.json` 和日志；同一 run directory 可修复后重试，attempts 会增加且旧证据保留。已通过阶段不会覆盖。

验收入口会按目标解释器自动把其虚拟环境中的 `nvidia/npp/lib` 前置到子进程的 `LD_LIBRARY_PATH`，避免 Notebook 默认内核无法解析 `libnppicc.so.12`。`check_planner_environment.py` 直接运行时也会在导入 TorchCodec 前自检，并在需要时带正确路径重新执行一次；这不安装或更改任何受保护依赖版本。

## A：环境与数据

- `/content/.venv_planner/bin/python`：受保护版本、固定 3D HAMSTER checkout、CUDA smoke；
- `/content/.venv/bin/python`：受保护版本、固定 LeRobot checkout、LIBERO/MuJoCo import、CUDA smoke；
- checkpoint manifest revision 与本地 index；
- 最新或显式指定的 LIBERO probe；
- RGB/depth 文件哈希、米制 depth、K、`T_world_camera`。

A 不加载 9B Planner，也不下载内容。

## B：高层可行性

B 调用 `/content/.venv_planner/bin/python scripts/run_libero_planner_single_frame.py`，一次加载 3D HAMSTER。必须同时满足：

- Planner 输出非空、有限、U/V 和 depth 合法；
- UVD roundtrip 与刚体变换检查通过；
- 保存 UVD、camera/world XYZ、overlay、耗时、显存；
- 保存 source=`prediction` 的统一轨迹缓存。

自动校验只能确认数值与 frame 自洽。执行者还要目视 overlay 和 world XYZ；明显不可用时不要进入 C。

## C：真实 SmolVLA 冒烟

直接运行 05 的 Stage C 单元。它会按顺序：

1. 在现有 `/content/.venv` 中安装固定 LeRobot commit 的 `[libero,smolvla]` extras，并重新核对受保护版本；
2. 断点续传 `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`；
3. 固定并缓存 SmolVLM config/tokenizer/processor 支持文件，不重复下载其大权重；
4. 从已通过的 Stage B 自动定位双相机观测、EEF 和 prediction trajectory，生成已预处理、因果检查的 tensor batch artifact；
5. 自动设置两个 Stage C 路径并用 `/content/.venv/bin/python` 依次执行 E0/E1/E2。

检查内容：

- 真实 forward 与原 loss；
- backward 与 adapter 梯度；
- `predict_action_chunk`、动作维度和有限值；
- 各配置耗时、GPU 和显存峰值。

C 不做 optimizer step、checkpoint 更新或 rollout。smoke batch 中的零 action target 和缺失 gripper qpos 的零填充只用于真实模型接口/反向冒烟，metadata 明确禁止把它当成训练或评估数据。正式训练仍必须使用正确示范 action 和完整状态。

## D / E

D/E 的 command 和正式预算故意在配置中留空：

- D：C 通过后填写单任务、少 step、少初始状态的 pilot；
- E：D 通过并确定预算后填写 E0/E1/E2 正式命令。

不要在本轮离线开发中运行它们。

## 持久化

短验收继续写 `/content`。长训练开始后，把 Notebook 的 `RUN_DIR` 或 `VLA_PERSISTENT_ROOT` 指向已挂载 Google Drive。优先持久化 checkpoints、轨迹、config/版本、评估和失败日志；无需让 Planner 与 SmolVLA 同进程驻留。
