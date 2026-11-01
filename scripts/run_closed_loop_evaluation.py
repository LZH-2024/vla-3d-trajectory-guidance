#!/usr/bin/env python3
"""Run a resumable episode plan using an explicitly supplied simulator runner."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.evaluation import load_episode_plan, run_episode_plan
from trajectory_guidance.experiment import EvaluationStore, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--experiment", choices=["E0", "E1", "E2"], required=True)
    parser.add_argument("--episode-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-module", required=True)
    parser.add_argument("--runner-callable", default="run_episode")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config = load_experiment_config(project_root, args.experiment)
    episodes = load_episode_plan(args.episode_plan.resolve())
    module = importlib.import_module(args.runner_module)
    runner = getattr(module, args.runner_callable, None)
    if not callable(runner):
        raise TypeError(
            f"runner {args.runner_module}:{args.runner_callable} is not callable"
        )
    summary = run_episode_plan(
        episodes,
        runner=runner,
        experiment_config=config,
        store=EvaluationStore(args.output_dir.resolve()),
        stop_on_error=not args.continue_on_error,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if not summary["failed_episode_ids"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
