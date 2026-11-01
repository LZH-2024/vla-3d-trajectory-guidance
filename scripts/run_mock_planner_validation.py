#!/usr/bin/env python3
"""Run deterministic CPU-only planner artifact tests; never loads model weights."""

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

from trajectory_guidance.mock_planner import run_mock_suite
from trajectory_guidance.libero_probe import json_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir
        or project_root / "outputs" / "mock_planner" / timestamp
    ).resolve()
    summary = run_mock_suite(output_dir)
    print(
        json.dumps(
            json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False
        )
    )
    return 0 if summary["all_cases_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
