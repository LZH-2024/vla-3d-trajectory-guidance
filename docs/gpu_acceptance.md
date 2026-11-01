# 集中 GPU 验收

入口：`notebooks/05_gpu_acceptance.ipynb` 或：

    python scripts/run_gpu_acceptance_stage.py \
      --stage A \
      --run-dir /content/vla-3d-trajectory-guidance/outputs/gpu_acceptance/<run_id>

每次只执行一个阶段。`acceptance_ledger.json` 要求前置阶段全部 `passed`，否则后续命令在加载模型前拒绝执行。失败会在 `stage_x/attempt_NNN/` 保存 `failure.json` 和日志；同一 run directory 可修复后重试，attempts 会增加且旧证据保留。已通过阶段不会覆盖。

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

C 默认没有 policy 和 batch 路径，因此不会下载或自动启动。先设置：

    export VLA_SMOLVLA_POLICY_PATH=/content/models/<local-smolvla>
    export VLA_SMOLVLA_BATCH_ARTIFACT=/content/.../<preprocessed-batch>

随后用 `/content/.venv/bin/python` 一次加载本地 SmolVLA，依次执行 E0/E1/E2：

- 真实 forward 与原 loss；
- backward 与 adapter 梯度；
- `predict_action_chunk`、动作维度和有限值；
- 各配置耗时、GPU 和显存峰值。

C 不做 optimizer step、checkpoint 更新或 rollout。batch 必须已由固定版本的 policy preprocessor 处理，并带 causality metadata。

## D / E

D/E 的 command 和正式预算故意在配置中留空：

- D：C 通过后填写单任务、少 step、少初始状态的 pilot；
- E：D 通过并确定预算后填写 E0/E1/E2 正式命令。

不要在本轮离线开发中运行它们。

## 持久化

短验收继续写 `/content`。长训练开始后，把 Notebook 的 `RUN_DIR` 或 `VLA_PERSISTENT_ROOT` 指向已挂载 Google Drive。优先持久化 checkpoints、轨迹、config/版本、评估和失败日志；无需让 Planner 与 SmolVLA 同进程驻留。
