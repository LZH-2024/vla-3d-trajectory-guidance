#!/usr/bin/env python3
"""Orchestrate the automatic fail-stop Stage-D single-task GPU pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _run(command: list[str], log_path: Path, *, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _valid_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("valid") is True else None


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config or project_root / "configs" / "gpu_acceptance_colab.json"
    ).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Keep the virtual-environment entry point for child processes.  Resolving
    # this symlink produces uv's base CPython executable and drops .venv's
    # site-packages (for example NumPy).  Only use resolved paths for identity
    # validation of the current process.
    env_python = Path(config["smolvla_python"]).expanduser()
    expected_python = env_python.resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(
            f"wrong interpreter: got {sys.executable}, expected {env_python}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_summary = output_dir / "summary.json"
    reusable = _valid_summary(final_summary)
    if reusable is not None:
        print(json.dumps(reusable, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    subprocess_env = os.environ.copy()
    started = time.perf_counter()
    inputs_path = output_dir / "stage_d_inputs.json"
    if _valid_summary(inputs_path) is None:
        _run(
            [
                str(env_python),
                str(project_root / "scripts" / "prepare_smolvla_stage_d_inputs.py"),
                "--project-root",
                str(project_root),
                "--config",
                str(config_path),
                "--run-dir",
                str(args.run_dir.resolve()),
                "--output",
                str(inputs_path),
            ],
            output_dir / "prepare_inputs.log",
            env=subprocess_env,
        )
    prepared = _valid_summary(inputs_path)
    if prepared is None:
        raise RuntimeError("Stage-D input preparation did not produce valid evidence")

    results: list[dict[str, Any]] = []
    for experiment_id in ("E0", "E1", "E2"):
        experiment_dir = output_dir / experiment_id
        experiment_summary = experiment_dir / "summary.json"
        result = _valid_summary(experiment_summary)
        if result is None:
            _run(
                [
                    str(env_python),
                    str(project_root / "scripts" / "run_smolvla_stage_d_experiment.py"),
                    "--project-root",
                    str(project_root),
                    "--config",
                    str(config_path),
                    "--inputs",
                    str(inputs_path),
                    "--experiment",
                    experiment_id,
                    "--output-dir",
                    str(experiment_dir),
                ],
                output_dir / f"{experiment_id.lower()}.log",
                env=subprocess_env,
            )
            result = _valid_summary(experiment_summary)
        if result is None:
            raise RuntimeError(f"{experiment_id} did not produce a valid Stage-D summary")
        results.append(result)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "scope": "stage_d_bounded_single_task_real_data_optimizer_and_rollout_pilot",
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": str(inputs_path),
        "experiments": [
            {
                "experiment_id": result["experiment_id"],
                "optimizer_steps": result["optimizer_steps"],
                "training_seconds": result["training_seconds"],
                "rollout_completed": result["rollout"]["completed"],
                "rollout_success": result["rollout"]["success"],
                "rollout_steps": result["rollout"]["steps"],
                "summary": str(output_dir / result["experiment_id"] / "summary.json"),
            }
            for result in results
        ],
        "stage_pass_semantics": (
            "All three real-data optimizer and closed-loop pipelines completed without "
            "format/runtime errors; task success is recorded but not required for this plumbing gate."
        ),
        "not_formal_experiment": True,
    }
    _write_json(final_summary, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
