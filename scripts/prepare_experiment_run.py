#!/usr/bin/env python3
"""Materialize a resumable E0/E1/E2 run plan without loading data or a model."""

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

from trajectory_guidance.experiment import (
    load_experiment_config,
    resolve_experiment_output_root,
    validate_experiment_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--experiment", choices=["E0", "E1", "E2"], required=True)
    parser.add_argument("--phase", choices=["smoke", "pilot", "formal"], required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--require-runnable",
        action="store_true",
        help="Require real dataset_repo_id and local pretrained_path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config = load_experiment_config(project_root, args.experiment)
    validate_experiment_config(config, require_runnable=args.require_runnable)
    step_key = f"{args.phase}_steps"
    step_budget = config["training"][step_key]
    if step_budget is None:
        raise ValueError(
            f"{step_key} is intentionally unset; determine it only after earlier acceptance stages"
        )
    output_root = args.run_dir.resolve() if args.run_dir else resolve_experiment_output_root(config) / args.phase
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_config.json"
    if resolved_path.is_file():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing.get("config_digest") != config["config_digest"]:
            raise ValueError("existing run directory has a different config digest")
        resumed = True
    else:
        resolved_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        resumed = False
    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_only_no_model_no_training_no_rollout",
        "experiment_id": config["experiment_id"],
        "phase": args.phase,
        "step_budget": step_budget,
        "config_digest": config["config_digest"],
        "resumed_existing_directory": resumed,
        "paths": {
            "run_dir": str(output_root),
            "resolved_config": str(resolved_path),
            "checkpoints": str(output_root / "checkpoints"),
            "metrics": str(output_root / "metrics.jsonl"),
            "evaluation": str(output_root / "evaluation"),
        },
        "next_gate": (
            "Stage C must pass before assembling a real trainer."
            if not args.require_runnable
            else "Real assembly still requires an explicitly reviewed dataset/preprocessor runner."
        ),
    }
    (output_root / "run_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
