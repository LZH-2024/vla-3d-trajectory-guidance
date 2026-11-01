#!/usr/bin/env python3
"""Add the pinned SmolVLA extra to the existing isolated LeRobot environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _run(command: Sequence[str]) -> None:
    print(f"\n$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run([str(part) for part in command], check=True)


def _capture(command: Sequence[str]) -> str:
    return subprocess.check_output([str(part) for part in command], text=True).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config or project_root / "configs" / "gpu_acceptance_colab.json"
    ).resolve()
    acceptance = json.loads(config_path.read_text(encoding="utf-8"))
    planner = json.loads(
        (project_root / acceptance["stage_a"]["planner_config"]).read_text(
            encoding="utf-8"
        )
    )
    libero = json.loads(
        (project_root / acceptance["stage_a"]["libero_config"]).read_text(
            encoding="utf-8"
        )
    )
    stage_c = acceptance["stage_c"]

    if platform.system() != "Linux":
        raise RuntimeError("SmolVLA bootstrap is intended for the Linux Colab/GPU runtime")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH")
    if planner["uv_version"] not in _capture([uv, "--version"]):
        raise RuntimeError("uv version differs from the protected reference")

    env_python = Path(acceptance["smolvla_python"])
    if not env_python.is_file():
        raise FileNotFoundError(f"SmolVLA interpreter is missing: {env_python}")
    actual_python = _capture(
        [env_python, "-c", "import platform; print(platform.python_version())"]
    )
    if actual_python != planner["python_version"]:
        raise RuntimeError(
            f"Python version mismatch: {actual_python} != {planner['python_version']}"
        )

    checkout = Path(libero["libero"]["checkout_dir"])
    expected_commit = libero["libero"]["repo_commit"]
    actual_commit = _capture(["git", "-C", checkout, "rev-parse", "HEAD"])
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"LeRobot checkout mismatch: {actual_commit} != {expected_commit}"
        )

    constraints = project_root / "configs" / "libero_constraints.txt"
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            env_python,
            "--torch-backend",
            "cu128",
            "--no-sources",
            "--constraint",
            constraints,
            "--editable",
            f"{checkout}[libero,smolvla]",
        ]
    )
    _run([uv, "pip", "check", "--python", env_python])

    expected_versions = {
        **planner["reference_core"],
        "transformers": libero["libero"]["transformers_version"],
        "huggingface-hub": libero["libero"]["huggingface_hub_version"],
        "hf-libero": libero["libero"]["hf_libero_version"],
        "robosuite": libero["libero"]["robosuite_version"],
        "robomimic": libero["libero"]["robomimic_version"],
        "mujoco": libero["libero"]["mujoco_version"],
        "lerobot": libero["libero"]["repo_version"],
        "accelerate": stage_c["accelerate_version"],
        "num2words": stage_c["num2words_version"],
    }
    version_code = (
        "import importlib.metadata,json; "
        f"names={list(expected_versions)!r}; "
        "print(json.dumps({name: importlib.metadata.version(name) for name in names}))"
    )
    actual_versions = json.loads(_capture([env_python, "-c", version_code]))
    mismatches = {
        name: {"expected": expected, "actual": actual_versions.get(name)}
        for name, expected in expected_versions.items()
        if actual_versions.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"protected dependency versions changed: {mismatches}")

    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "scope": "smolvla_stage_c_optional_dependencies_no_model_load",
        "python": {"executable": str(env_python), "version": actual_python},
        "lerobot_commit": actual_commit,
        "versions": actual_versions,
    }
    _write_json(args.output.resolve(), record)
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
