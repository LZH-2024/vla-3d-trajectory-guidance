#!/usr/bin/env python3
"""Download the pinned public 3D HAMSTER checkpoint with the planner interpreter."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--record", type=Path)
    return parser.parse_args()


def assert_target_interpreter(expected_env: Path) -> None:
    expected = (expected_env / "bin" / "python").resolve()
    actual = Path(sys.executable).resolve()
    if actual != expected:
        raise RuntimeError(f"wrong interpreter: got {actual}, expected {expected}")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (args.config or project_root / "configs" / "planner_colab.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    planner = config["planner"]
    assert_target_interpreter(Path(planner["env_dir"]))

    from huggingface_hub import snapshot_download

    checkpoint_dir = (args.checkpoint_dir or Path(planner["checkpoint_dir"])).resolve()
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(checkpoint_dir.parent).free
    if free_bytes < 30 * 1024**3:
        print(
            f"WARNING: only {free_bytes / 1024**3:.1f} GiB free. "
            "The 9B BF16 checkpoint needs substantial disk space.",
            file=sys.stderr,
        )

    resolved = snapshot_download(
        repo_id=planner["model_id"],
        revision=planner["model_revision"],
        local_dir=checkpoint_dir,
    )
    required = checkpoint_dir / "model.safetensors.index.json"
    if not required.is_file():
        raise FileNotFoundError(f"checkpoint download is incomplete: missing {required}")

    files = [
        {
            "path": str(path.relative_to(checkpoint_dir)),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(checkpoint_dir.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    ]
    record = {
        "schema_version": 1,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": planner["model_id"],
        "requested_revision": planner["model_revision"],
        "resolved_path": str(Path(resolved).resolve()),
        "checkpoint_dir": str(checkpoint_dir),
        "files": files,
    }
    record_path = (
        args.record
        or project_root / "outputs" / "environment" / "planner_checkpoint_manifest.json"
    ).resolve()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
