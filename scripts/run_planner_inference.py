#!/usr/bin/env python3
"""CLI entry point for official-example 3D HAMSTER inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.planner_runner import run_official_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--examples-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (args.config or project_root / "configs" / "planner_colab.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    planner = config["planner"]

    expected_python = (Path(planner["env_dir"]) / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")

    model_path = (args.model_path or Path(planner["checkpoint_dir"])).resolve()
    examples_dir = (
        args.examples_dir
        or Path(planner["checkout_dir"]) / planner["examples_subdir"]
    ).resolve()
    output_dir = (
        args.output_dir
        or project_root / "outputs" / "planner" / "official_examples"
    ).resolve()

    summary = run_official_examples(
        model_path=model_path,
        examples_dir=examples_dir,
        output_dir=output_dir,
        sample_names=args.samples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        continue_on_error=args.continue_on_error,
        provenance={
            "planner_repo_url": planner["repo_url"],
            "planner_repo_commit": planner["repo_commit"],
            "model_id": planner["model_id"],
            "model_revision": planner["model_revision"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
