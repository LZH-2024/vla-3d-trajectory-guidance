#!/usr/bin/env python3
"""Download and record the pinned LIBERO assets before offline rollout."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.libero_assets import (
    REQUIRED_ASSET_PATHS,
    download_pinned_libero_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    acceptance_path = (
        args.config or project_root / "configs" / "gpu_acceptance_colab.json"
    ).resolve()
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    expected_python = Path(acceptance["smolvla_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")
    if platform.python_version() != "3.12.13":
        raise RuntimeError(f"unexpected Python version: {platform.python_version()}")

    libero_config_path = (
        project_root / acceptance["stage_a"]["libero_config"]
    ).resolve()
    libero = json.loads(libero_config_path.read_text(encoding="utf-8"))["libero"]
    assets_dir, resolved_revision = download_pinned_libero_assets(libero)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "repo_id": libero["assets_repo_id"],
        "revision": resolved_revision,
        "assets_dir": str(assets_dir),
        "required_paths": list(REQUIRED_ASSET_PATHS),
        "scope": "pinned_libero_assets_for_offline_stage_d_rollout",
    }
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
