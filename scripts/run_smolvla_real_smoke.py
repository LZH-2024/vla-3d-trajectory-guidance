#!/usr/bin/env python3
"""Run fail-fast E0/E1/E2 forward/backward smoke checks on a local SmolVLA.

This entry never downloads a policy or dataset.  It requires a local policy
directory and an already preprocessed tensor batch artifact.
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

import torch
from torch import Tensor

from trajectory_guidance.batch_artifact import load_batch_artifact
from trajectory_guidance.experiment import (
    assert_experiments_comparable,
    load_experiment_config,
)
from trajectory_guidance.trajectory_conditioning import (
    TRAJECTORY_EEF_KEY,
    TRAJECTORY_MASK_KEY,
    TRAJECTORY_POINTS_KEY,
    SmolVLATrajectoryWrapper,
    TrajectoryConditioningConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--batch-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


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


def _loss_from_output(output: Any) -> tuple[Tensor, list[str]]:
    if isinstance(output, Tensor):
        return output, []
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        keys = sorted(output[1]) if len(output) > 1 and isinstance(output[1], dict) else []
        return output[0], keys
    raise TypeError("SmolVLA forward did not return a loss tensor or (loss, metrics)")


def _json_tensor(value: Tensor | None) -> Any:
    return None if value is None else value.detach().float().cpu().tolist()


def _prepare_batch(batch_numpy: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    batch: dict[str, Tensor] = {}
    batch_size: int | None = None
    for key, value in batch_numpy.items():
        tensor = torch.from_numpy(value).to(device=device)
        if tensor.ndim > 0:
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError(f"batch dimension mismatch for {key!r}")
        batch[key] = tensor
    if batch_size is None or batch_size <= 0:
        raise ValueError("batch artifact has no non-empty batch dimension")
    return batch


def _run_one(
    experiment: dict[str, Any],
    base_policy: torch.nn.Module,
    batch: dict[str, Tensor],
) -> dict[str, Any]:
    experiment_id = experiment["experiment_id"]
    runner: torch.nn.Module = base_policy
    wrapper: SmolVLATrajectoryWrapper | None = None
    if experiment_id != "E0":
        wrapper = SmolVLATrajectoryWrapper(
            base_policy,
            TrajectoryConditioningConfig.from_mapping(
                experiment["trajectory_conditioning"]
            ),
        )
        runner = wrapper

    base_policy.reset()
    runner.train()
    runner.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss, loss_metric_keys = _loss_from_output(runner(batch))
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError(f"{experiment_id} produced a non-finite scalar loss")
    loss.backward()
    torch.cuda.synchronize()
    forward_backward_seconds = time.perf_counter() - started

    adapter_gradients: list[dict[str, Any]] = []
    if wrapper is not None:
        for name, parameter in list(wrapper.condition_encoder.named_parameters()) + (
            list(wrapper.geometry_gate.named_parameters())
            if wrapper.geometry_gate is not None
            else []
        ):
            gradient = parameter.grad
            adapter_gradients.append(
                {
                    "name": name,
                    "present": gradient is not None,
                    "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
                    "nonzero": bool(gradient is not None and torch.count_nonzero(gradient) > 0),
                }
            )
        if not adapter_gradients or not all(item["finite"] for item in adapter_gradients):
            raise RuntimeError(f"{experiment_id} adapter gradients are missing or non-finite")
        if not any(item["nonzero"] for item in adapter_gradients):
            raise RuntimeError(f"{experiment_id} adapter gradients are all zero")

    runner.eval()
    base_policy.reset()
    torch.manual_seed(int(experiment["training"]["seed"]))
    torch.cuda.manual_seed_all(int(experiment["training"]["seed"]))
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.no_grad():
        action_chunk = runner.predict_action_chunk(batch)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    if not isinstance(action_chunk, Tensor) or action_chunk.ndim != 3:
        raise RuntimeError(f"{experiment_id} action chunk must have shape (B, T, A)")
    if not torch.isfinite(action_chunk).all():
        raise RuntimeError(f"{experiment_id} action chunk contains NaN or infinity")
    if "action" in batch and action_chunk.shape[-1] != batch["action"].shape[-1]:
        raise RuntimeError(
            f"{experiment_id} changed action dimension: "
            f"{action_chunk.shape[-1]} != {batch['action'].shape[-1]}"
        )

    diagnostics = wrapper.diagnostics() if wrapper is not None else {}
    result = {
        "experiment_id": experiment_id,
        "loss": float(loss.detach().float().cpu()),
        "loss_metric_keys": loss_metric_keys,
        "action_chunk_shape": list(action_chunk.shape),
        "forward_backward_seconds": forward_backward_seconds,
        "inference_seconds": inference_seconds,
        "adapter_gradients": adapter_gradients,
        "gate_mode": diagnostics.get("gate_mode", "none"),
        "gate_gain_not_probability": _json_tensor(
            diagnostics.get("gate_gain_not_probability")
        ),
        "geometry_anomaly_features": _json_tensor(
            diagnostics.get("geometry_anomaly_features")
        ),
        "gpu": _gpu_snapshot(),
    }
    if wrapper is not None:
        wrapper.close()
    return result


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    expected_python = Path("/content/.venv/bin/python")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(
            f"wrong interpreter: got {sys.executable}, expected {expected_python}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; real SmolVLA smoke was not attempted")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("real SmolVLA acceptance requires a CUDA device")
    torch.cuda.set_device(device.index or 0)

    policy_path = args.policy_path.resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"local SmolVLA policy directory is missing: {policy_path}")
    batch_numpy, batch_manifest = load_batch_artifact(args.batch_artifact.resolve())
    metadata = batch_manifest.get("metadata", {})
    if metadata.get("preprocessed_for_smolvla") is not True:
        raise ValueError("batch must be explicitly marked preprocessed_for_smolvla=true")
    if metadata.get("causality_checked") is not True:
        raise ValueError("batch must be explicitly marked causality_checked=true")
    for key in ("action", TRAJECTORY_POINTS_KEY, TRAJECTORY_MASK_KEY, TRAJECTORY_EEF_KEY):
        if key not in batch_numpy:
            raise KeyError(f"required Stage C batch key is missing: {key}")
    batch = _prepare_batch(batch_numpy, device)

    configs = [load_experiment_config(project_root, name) for name in ("E0", "E1", "E2")]
    assert_experiments_comparable(configs)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    load_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        str(policy_path), local_files_only=True
    ).to(device)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    if getattr(policy.config, "compile_model", False):
        raise ValueError("Stage C requires compile_model=False for the reviewed hook path")

    results = [_run_one(config, policy, batch) for config in configs]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "scope": "real_smolvla_forward_backward_smoke_no_optimizer_step_no_rollout",
        "lerobot_commit_expected": "3f797de5952432284a7c77ad2760218cb74e2f26",
        "policy_path": str(policy_path),
        "batch_artifact": str(args.batch_artifact.resolve()),
        "model_load_seconds": load_seconds,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "gpu_after_model_load": _gpu_snapshot(),
        "results": results,
        "all_passed": len(results) == 3,
        "limitations": [
            "No optimizer step or checkpoint update was performed.",
            "No LIBERO rollout or task-success measurement was performed.",
            "Geometry gate gains are anomaly-conditioned gains, not calibrated correctness probabilities.",
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
