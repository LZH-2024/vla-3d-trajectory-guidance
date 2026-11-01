#!/usr/bin/env python3
"""Run one bounded real-data optimizer smoke or no-update diagnostic rollout.

This is Stage-D plumbing evidence, not a meaningful fine-tuning budget.  Each
E0/E1/E2 process starts from the same pinned SmolVLA checkpoint and uses the
same real demonstration chunk and per-step Flow-Matching RNG seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import Tensor

from trajectory_guidance.batch_artifact import load_batch_artifact
from trajectory_guidance.experiment import load_experiment_config
from trajectory_guidance.libero_pilot import image_hwc_float01, normalized_instruction
from trajectory_guidance.libero_assets import activate_libero_assets, validate_libero_assets
from trajectory_guidance.libero_probe import sha256_file
from trajectory_guidance.trajectory_conditioning import (
    TRAJECTORY_EEF_KEY,
    TRAJECTORY_MASK_KEY,
    TRAJECTORY_POINTS_KEY,
    SmolVLATrajectoryWrapper,
    TrajectoryConditioningConfig,
)
from trajectory_guidance.trajectory_record import load_trajectory_record
from trajectory_guidance.waypoint_selection import select_local_waypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--experiment", choices=["E0", "E1", "E2"], required=True)
    parser.add_argument("--assets-manifest", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _loss_from_output(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("SmolVLA forward did not return a loss tensor")


def _gpu_snapshot() -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "allocated_bytes": torch.cuda.memory_allocated(index),
        "reserved_bytes": torch.cuda.memory_reserved(index),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        "torch_cuda_build": torch.version.cuda,
    }


def _prepare_batch(batch_numpy: dict[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    batch_size: int | None = None
    for key, value in batch_numpy.items():
        tensor = torch.from_numpy(value).to(device=device)
        if tensor.ndim > 0:
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError(f"batch dimension mismatch for {key!r}")
        result[key] = tensor
    if batch_size != 1:
        raise ValueError(f"Stage-D pilot requires batch size 1, got {batch_size}")
    return result


def _save_trainable_checkpoint(
    runner: torch.nn.Module,
    destination: Path,
    *,
    experiment_id: str,
    source_policy: Path,
    config_digest: str,
    optimizer_steps: int,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in runner.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("SmolVLA exposes no trainable parameters")
    weights_path = destination / "trainable_state.safetensors"
    temporary = destination / ".trainable_state.safetensors.tmp"
    save_file(state, temporary)
    temporary.replace(weights_path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "trainable_parameter_subset_reloaded_on_pinned_base",
        "experiment_id": experiment_id,
        "source_policy": str(source_policy),
        "config_digest": config_digest,
        "optimizer_steps_completed": optimizer_steps,
        "tensor_count": len(state),
        "parameter_count": int(sum(tensor.numel() for tensor in state.values())),
        "weights": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "optimizer_state_saved": False,
        "resume_semantics": (
            "Completed experiment can be reused; interruption before this atomic checkpoint "
            "restarts the bounded optimizer smoke from the pinned base."
        ),
    }
    _write_json(destination / "checkpoint_manifest.json", manifest)
    return manifest


def _info_true(info: dict[str, Any], key: str) -> bool:
    value = info.get(key)
    if value is None:
        return False
    return bool(np.asarray(value, dtype=np.bool_).any())


def _save_snapshot(path: Path, image_value: Any) -> None:
    image = image_hwc_float01(image_value, name="rollout_agentview")
    Image.fromarray(np.round(image * 255.0).astype(np.uint8), mode="RGB").save(path)


def _rollout(
    *,
    runner: torch.nn.Module,
    base_policy: torch.nn.Module,
    wrapper: SmolVLATrajectoryWrapper | None,
    policy_config: Any,
    policy_path: Path,
    trajectory_dir: Path,
    settings: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    assets_dir: Path,
) -> dict[str, Any]:
    activate_libero_assets(assets_dir)
    import gymnasium as gym

    from lerobot.envs.libero import create_libero_envs
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.env_processor import LiberoProcessorStep

    if int(settings["target_init_state_id"]) != 0:
        raise ValueError("bounded Stage-D runner currently supports init_state_id=0 only")
    trajectory = load_trajectory_record(trajectory_dir, allowed_sources={"prediction"})
    if trajectory.uses_future_expert_information:
        raise ValueError("rollout trajectory contains future expert information")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config,
        str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    suite = str(settings["target_suite"])
    task_id = int(settings["target_task_id"])
    max_steps = int(settings["rollout_max_steps"])
    envs = create_libero_envs(
        task=suite,
        n_envs=1,
        gym_kwargs={
            "task_ids": [task_id],
            "obs_type": "pixels_agent_pos",
            "num_steps_wait": int(settings["num_steps_wait"]),
        },
        camera_name="agentview_image,robot0_eye_in_hand_image",
        init_states=True,
        env_cls=gym.vector.SyncVectorEnv,
        control_mode="relative",
        episode_length=max_steps,
    )
    env = envs[suite][task_id]
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    eef_positions: list[np.ndarray] = []
    gate_gains: list[float] = []
    success = False
    terminated_flag = False
    truncated_flag = False
    min_progress_index = 0
    base_policy.reset()
    rollout_started = time.perf_counter()
    try:
        observation, _ = env.reset(seed=int(settings["rollout_seed"]))
        task_description = getattr(env.envs[0], "task_description", trajectory.instruction)
        if normalized_instruction(task_description) != normalized_instruction(
            trajectory.instruction
        ):
            raise ValueError(
                "rollout task instruction does not match the cached prediction trajectory"
            )
        for step in range(max_steps):
            processed_observation = env_preprocessor(
                preprocess_observation(observation)
            )
            state = processed_observation.get("observation.state")
            if not isinstance(state, Tensor) or tuple(state.shape) != (1, 8):
                raise ValueError(
                    f"LIBERO rollout state must have shape (1, 8), got {getattr(state, 'shape', None)}"
                )
            current_eef = state[0, :3].detach().cpu().numpy().astype(np.float32)
            eef_positions.append(current_eef)
            if step % int(settings["snapshot_every_steps"]) == 0:
                _save_snapshot(
                    snapshots / f"step_{step:04d}.png",
                    processed_observation["observation.images.image"][0],
                )

            policy_input = dict(processed_observation)
            policy_input["task"] = [trajectory.instruction]
            policy_batch = preprocessor(policy_input)
            if wrapper is not None:
                window = select_local_waypoints(
                    trajectory.waypoints_xyz_m,
                    trajectory.valid_mask,
                    current_eef,
                    count=int(settings["num_waypoints"]),
                    min_progress_index=min_progress_index,
                )
                if window.start_index is not None:
                    min_progress_index = max(min_progress_index, window.start_index)
                policy_batch[TRAJECTORY_POINTS_KEY] = torch.from_numpy(
                    window.waypoints_xyz_m.astype(np.float32)
                ).unsqueeze(0).to(device)
                policy_batch[TRAJECTORY_MASK_KEY] = torch.from_numpy(
                    window.valid_mask
                ).unsqueeze(0).to(device)
                policy_batch[TRAJECTORY_EEF_KEY] = torch.from_numpy(
                    current_eef
                ).unsqueeze(0).to(device)

            with torch.no_grad():
                action = runner.select_action(policy_batch)
                action = postprocessor(action)
            if isinstance(action, Tensor):
                action_array = action.detach().cpu().numpy()
            else:
                action_array = np.asarray(action)
            action_array = np.asarray(action_array, dtype=np.float32)
            if action_array.shape == (7,):
                action_array = action_array[None, :]
            if action_array.shape != (1, 7) or not np.isfinite(action_array).all():
                raise ValueError(f"rollout action must be finite (1, 7), got {action_array.shape}")
            actions.append(action_array[0].copy())
            if wrapper is not None:
                gain = wrapper.diagnostics().get("gate_gain_not_probability")
                if isinstance(gain, Tensor):
                    gate_gains.append(float(gain.detach().float().mean().cpu()))

            observation, reward, terminated, truncated, info = env.step(action_array)
            rewards.append(float(np.asarray(reward, dtype=np.float64).reshape(-1)[0]))
            success = success or _info_true(info, "is_success")
            terminated_flag = bool(np.asarray(terminated, dtype=np.bool_).any())
            truncated_flag = bool(np.asarray(truncated, dtype=np.bool_).any())
            if success or terminated_flag or truncated_flag:
                break
    finally:
        env.close()
    rollout_seconds = time.perf_counter() - rollout_started
    np.savez_compressed(
        output_dir / "rollout_trace.npz",
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        eef_positions_xyz_m=np.asarray(eef_positions, dtype=np.float32),
        geometry_gate_gain_not_probability=np.asarray(gate_gains, dtype=np.float32),
    )
    return {
        "completed": True,
        "success": bool(success),
        "steps": len(actions),
        "sum_reward": float(sum(rewards)),
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "seconds": rollout_seconds,
        "trajectory_id": trajectory.trajectory_id,
        "trajectory_source": trajectory.source,
        "condition_uses_future_expert_information": False,
        "trace": str(output_dir / "rollout_trace.npz"),
        "snapshots": str(snapshots),
        "gate_gain_is_calibrated_probability": False,
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config or project_root / "configs" / "gpu_acceptance_colab.json"
    ).resolve()
    acceptance = json.loads(config_path.read_text(encoding="utf-8"))
    settings = acceptance["stage_d"]
    expected_python = Path(acceptance["smolvla_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")
    if platform.python_version() != "3.12.13":
        raise RuntimeError(f"unexpected Python version: {platform.python_version()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Stage-D experiment was not attempted")
    device = torch.device(settings["device"])
    if device.type != "cuda":
        raise ValueError("Stage D requires a CUDA device")
    torch.cuda.set_device(device.index or 0)

    inputs = json.loads(args.inputs.resolve().read_text(encoding="utf-8"))
    if inputs.get("valid") is not True:
        raise ValueError("Stage-D prepared inputs are invalid")
    policy_path = Path(inputs["policy_path"]).resolve()
    trajectory_dir = Path(inputs["trajectory_record"]).resolve()
    batch_numpy, batch_manifest = load_batch_artifact(inputs["batch_artifact"])
    metadata = batch_manifest.get("metadata", {})
    required_metadata = {
        "preprocessed_for_smolvla": True,
        "causality_checked": True,
        "condition_uses_future_expert_information": False,
        "real_demonstration_action_target": True,
        "synthetic_zero_action_target": False,
        "training_eligible_for_bounded_stage_d_pilot": True,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in required_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage-D batch metadata mismatch: {mismatches}")
    batch = _prepare_batch(batch_numpy, device)

    assets_manifest = json.loads(
        args.assets_manifest.resolve().read_text(encoding="utf-8")
    )
    if assets_manifest.get("valid") is not True:
        raise ValueError("Stage-D LIBERO asset manifest is invalid")
    libero_config_path = (
        project_root / acceptance["stage_a"]["libero_config"]
    ).resolve()
    libero_config = json.loads(libero_config_path.read_text(encoding="utf-8"))["libero"]
    if assets_manifest.get("repo_id") != libero_config["assets_repo_id"]:
        raise ValueError("Stage-D LIBERO asset repository does not match the pinned config")
    if assets_manifest.get("revision") != libero_config["assets_revision"]:
        raise ValueError("Stage-D LIBERO asset revision does not match the pinned config")
    assets_dir = validate_libero_assets(assets_manifest["assets_dir"])

    experiment = load_experiment_config(project_root, args.experiment)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    runtime_config = PreTrainedConfig.from_pretrained(
        str(policy_path), local_files_only=True
    )
    runtime_config.load_vlm_weights = False
    runtime_config.device = str(device)
    if getattr(runtime_config, "compile_model", False):
        raise ValueError("Stage-D wrapper path requires compile_model=False")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    load_started = time.perf_counter()
    base_policy = SmolVLAPolicy.from_pretrained(
        str(policy_path),
        config=runtime_config,
        local_files_only=True,
    ).to(device)
    runner: torch.nn.Module = base_policy
    wrapper: SmolVLATrajectoryWrapper | None = None
    if args.experiment != "E0":
        wrapper = SmolVLATrajectoryWrapper(
            base_policy,
            TrajectoryConditioningConfig.from_mapping(
                experiment["trajectory_conditioning"]
            ),
        )
        runner = wrapper
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    optimizer_steps = (
        int(settings["optimizer_steps"])
        if args.optimizer_steps is None
        else int(args.optimizer_steps)
    )
    if optimizer_steps < 0:
        raise ValueError("optimizer_steps must be non-negative")
    if optimizer_steps == 0 and args.experiment != "E0":
        raise ValueError("zero-update diagnostic rollout is supported only for E0")
    base_seed = int(experiment["training"]["seed"])
    losses: list[float] = []
    gradient_norms: list[float] = []
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] | None = None
    training_seconds = 0.0
    if optimizer_steps > 0:
        trainable = [
            parameter for parameter in runner.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("no trainable SmolVLA parameters were found")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(experiment["training"]["learning_rate"]),
            betas=tuple(getattr(runtime_config, "optimizer_betas", (0.9, 0.95))),
            eps=float(getattr(runtime_config, "optimizer_eps", 1e-8)),
            weight_decay=float(getattr(runtime_config, "optimizer_weight_decay", 1e-10)),
        )
        runner.train()
        training_started = time.perf_counter()
        for step in range(optimizer_steps):
            # Reset after adapter construction so E0/E1/E2 see identical FM noise/time RNG.
            torch.manual_seed(base_seed + step)
            torch.cuda.manual_seed_all(base_seed + step)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_from_output(runner(batch))
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Stage-D loss at step {step + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(experiment["training"]["gradient_clip_norm"])
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise RuntimeError(f"non-finite gradient norm at step {step + 1}")
            if wrapper is not None:
                adapter_gradients = [
                    parameter.grad for parameter in wrapper.adapter_parameters()
                ]
                if not adapter_gradients or any(
                    gradient is None for gradient in adapter_gradients
                ):
                    raise RuntimeError("trajectory adapter gradient is missing")
                if not all(
                    torch.isfinite(gradient).all() for gradient in adapter_gradients
                ):
                    raise RuntimeError("trajectory adapter gradient is non-finite")
            optimizer.step()
            losses.append(float(loss.detach().float().cpu()))
            gradient_norms.append(
                float(torch.as_tensor(gradient_norm).detach().float().cpu())
            )
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        checkpoint = _save_trainable_checkpoint(
            runner,
            output_dir / "checkpoint",
            experiment_id=args.experiment,
            source_policy=policy_path,
            config_digest=experiment["config_digest"],
            optimizer_steps=optimizer_steps,
        )
    runner.eval()
    rollout = _rollout(
        runner=runner,
        base_policy=base_policy,
        wrapper=wrapper,
        policy_config=runtime_config,
        policy_path=policy_path,
        trajectory_dir=trajectory_dir,
        settings=settings,
        output_dir=output_dir,
        device=device,
        assets_dir=assets_dir,
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "scope": (
            "pretrained_no_update_closed_loop_rollout_diagnostic"
            if optimizer_steps == 0
            else "bounded_real_demo_optimizer_smoke_and_one_closed_loop_rollout"
        ),
        "experiment_id": args.experiment,
        "source_policy": str(policy_path),
        "inputs": str(args.inputs.resolve()),
        "experiment_config_digest": experiment["config_digest"],
        "model_load_seconds": load_seconds,
        "optimizer_steps": optimizer_steps,
        "losses_not_cross_experiment_quality_metrics": losses,
        "gradient_norms": gradient_norms,
        "training_seconds": training_seconds,
        "checkpoint": checkpoint,
        "rollout": rollout,
        "libero_assets": {
            "repo_id": assets_manifest["repo_id"],
            "revision": assets_manifest["revision"],
            "assets_dir": str(assets_dir),
            "manifest": str(args.assets_manifest.resolve()),
        },
        "gpu": _gpu_snapshot(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "limitations": [
            (
                "This rollout uses the pinned pretrained E0 checkpoint without any optimizer update."
                if optimizer_steps == 0
                else "Two optimizer steps on one real chunk validate plumbing, not useful model training."
            ),
            "E0/E1/E2 losses are not task-quality comparisons.",
            "One rollout is not a success-rate estimate.",
            "A completed rollout may pass Stage D even when task success is false.",
        ],
    }
    _write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    if wrapper is not None:
        wrapper.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
