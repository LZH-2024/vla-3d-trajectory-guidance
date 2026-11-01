#!/usr/bin/env python3
"""Revalidate one saved LIBERO RGB-D/calibration artifact without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.libero_planner import load_libero_planner_input
from trajectory_guidance.libero_probe import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    planner_input = load_libero_planner_input(
        args.probe_dir.resolve(), camera_name=args.camera_name
    )
    payload = {
        "schema_version": 1,
        "valid": True,
        "scope": "saved_libero_planner_input_revalidation_no_model_no_gpu",
        "probe_dir": str(planner_input.probe_dir),
        "camera_name": planner_input.camera_name,
        "instruction": planner_input.instruction,
        "rgb_size_wh": list(planner_input.rgb.size),
        "depth_shape_hw": list(planner_input.depth_m.shape),
        "depth_validation": planner_input.depth_validation.to_dict(),
        "calibration_validation": planner_input.calibration_validation.to_dict(),
        "hashes": planner_input.hashes,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
