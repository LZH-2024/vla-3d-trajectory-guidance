#!/usr/bin/env python3
"""Exercise E0/E1/E2 shapes, gradients, checkpoints, and resume on CPU mocks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from trajectory_guidance.experiment import (
    assert_experiments_comparable,
    load_experiment_config,
)
from trajectory_guidance.mock_smolvla import TinySmolVLAPolicy, make_tiny_batch
from trajectory_guidance.training import load_training_checkpoint, run_training_steps
from trajectory_guidance.trajectory_conditioning import (
    SmolVLATrajectoryWrapper,
    TrajectoryConditioningConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def _new_policy(config: dict) -> torch.nn.Module:
    base = TinySmolVLAPolicy()
    if config["experiment_id"] == "E0":
        return base
    return SmolVLATrajectoryWrapper(
        base,
        TrajectoryConditioningConfig.from_mapping(
            config["trajectory_conditioning"]
        ),
    )


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    project_root = args.project_root.resolve()
    configs = [load_experiment_config(project_root, name) for name in ("E0", "E1", "E2")]
    assert_experiments_comparable(configs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (args.output_dir or project_root / "outputs" / "cpu_mock" / timestamp).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for config in configs:
        torch.manual_seed(int(config["training"]["seed"]))
        policy = _new_policy(config)
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=float(config["training"]["learning_rate"])
        )
        batch = make_tiny_batch(
            num_waypoints=int(config["trajectory_conditioning"]["num_waypoints"]),
            include_trajectory=config["experiment_id"] != "E0",
            anomalous=config["experiment_id"] == "E2",
        )
        run_dir = output_root / config["experiment_id"]
        progress = run_training_steps(
            policy=policy,
            batches=[batch],
            optimizer=optimizer,
            run_dir=run_dir,
            config_digest=config["config_digest"],
            max_steps=args.steps,
            checkpoint_every_steps=1,
        )
        if progress.checkpoint_path is None:
            raise RuntimeError("mock training did not save a checkpoint")

        restored_policy = _new_policy(config)
        restored_optimizer = torch.optim.AdamW(
            restored_policy.parameters(),
            lr=float(config["training"]["learning_rate"]),
        )
        restored = load_training_checkpoint(
            progress.checkpoint_path,
            policy=restored_policy,
            optimizer=restored_optimizer,
            expected_config_digest=config["config_digest"],
        )
        action_chunk = restored_policy.predict_action_chunk(batch)
        adapter_gradients = []
        if isinstance(policy, SmolVLATrajectoryWrapper):
            adapter_gradients = [
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in policy.adapter_parameters()
            ]
        results.append(
            {
                "experiment_id": config["experiment_id"],
                "mock_steps": progress.step,
                "checkpoint": str(progress.checkpoint_path),
                "restored_step": restored.step,
                "action_chunk_shape": list(action_chunk.shape),
                "adapter_gradient_tensors_finite": adapter_gradients,
            }
        )

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "cpu_mock_only_no_real_smolvla_no_gpu",
        "mock": True,
        "all_passed": all(
            result["mock_steps"] == args.steps
            and result["restored_step"] == args.steps
            and all(result["adapter_gradient_tensors_finite"])
            for result in results
        ),
        "results": results,
        "warning": "This validates adapter plumbing only; it is not SmolVLA training or rollout evidence.",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
