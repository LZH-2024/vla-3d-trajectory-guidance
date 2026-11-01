#!/usr/bin/env python3
"""Run a no-update E0 rollout and summarize all Stage-D traces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.rollout_diagnostics import summarize_rollout_trace
from trajectory_guidance.runtime_environment import environment_with_venv_npp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    payload = _load_json(path)
    return payload if payload.get("valid") is True else None


def _complete_rollout_summary(path: Path) -> dict[str, Any] | None:
    payload = _valid_summary(path)
    if payload is None or payload.get("rollout", {}).get("completed") is not True:
        return None
    trace = Path(payload["rollout"]["trace"]).expanduser()
    return payload if trace.is_file() else None


def _snapshot_record(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    paths = sorted(root.glob("step_*.png")) if root.is_dir() else []
    return {
        "directory": str(root),
        "count": len(paths),
        "first": str(paths[0]) if paths else None,
        "last": str(paths[-1]) if paths else None,
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config or project_root / "configs" / "gpu_acceptance_colab.json"
    ).resolve()
    config = _load_json(config_path)
    run_dir = args.run_dir.resolve()
    ledger_path = run_dir / "acceptance_ledger.json"
    ledger = _load_json(ledger_path)
    stage_d_ledger = ledger["stages"]["D"]
    if stage_d_ledger.get("status") != "passed":
        raise RuntimeError("Stage D must pass before running rollout diagnostics")
    stage_d_summary_path = Path(stage_d_ledger["artifacts"]["summary"]).resolve()
    stage_d = _load_json(stage_d_summary_path)
    if stage_d.get("valid") is not True:
        raise ValueError("recorded Stage-D summary is invalid")

    inputs_path = Path(stage_d["inputs"]).resolve()
    assets_manifest_path = Path(stage_d["libero_assets"]).resolve()
    if not inputs_path.is_file() or not assets_manifest_path.is_file():
        raise FileNotFoundError("Stage-D inputs or LIBERO asset manifest is missing")

    diagnostics_root = run_dir / "stage_d_diagnostics"
    baseline_dir = diagnostics_root / "E0_pretrained_no_update"
    baseline_summary_path = baseline_dir / "summary.json"
    baseline = _complete_rollout_summary(baseline_summary_path)
    if baseline is None:
        env_python = Path(config["smolvla_python"]).expanduser()
        command_env = environment_with_venv_npp(env_python)
        command_env.update(
            {
                str(key): str(value)
                for key, value in config["stage_d"].get("environment", {}).items()
            }
        )
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
                "E0",
                "--optimizer-steps",
                "0",
                "--assets-manifest",
                str(assets_manifest_path),
                "--output-dir",
                str(baseline_dir),
            ],
            diagnostics_root / "e0_pretrained_no_update.log",
            env=command_env,
        )
        baseline = _complete_rollout_summary(baseline_summary_path)
    if baseline is None or int(baseline.get("optimizer_steps", -1)) != 0:
        raise RuntimeError("no-update E0 baseline did not produce valid evidence")

    stage_d_summaries = {
        str(item["experiment_id"]): Path(item["summary"]).resolve()
        for item in stage_d["experiments"]
    }
    records: list[dict[str, Any]] = []
    summary_paths = {
        "E0_pretrained_no_update": baseline_summary_path,
        "E0_after_2_steps": stage_d_summaries["E0"],
        "E1_after_2_steps": stage_d_summaries["E1"],
        "E2_after_2_steps": stage_d_summaries["E2"],
    }
    for label, summary_path in summary_paths.items():
        summary = _complete_rollout_summary(summary_path)
        if summary is None:
            raise RuntimeError(f"rollout summary is incomplete: {label}")
        rollout = summary["rollout"]
        record = {
            "label": label,
            "experiment_id": summary["experiment_id"],
            "optimizer_steps": int(summary["optimizer_steps"]),
            "success": bool(rollout["success"]),
            "rollout_seconds": float(rollout["seconds"]),
            "summary": str(summary_path),
            "snapshots": _snapshot_record(rollout["snapshots"]),
            "trace_summary": summarize_rollout_trace(rollout["trace"]),
        }
        records.append(record)

    baseline_trace = records[0]["trace_summary"]
    for record in records[1:]:
        trace = record["trace_summary"]
        record["delta_vs_pretrained_e0"] = {
            "motion_l2_mean": float(
                trace["actions"]["motion_l2_mean"]
                - baseline_trace["actions"]["motion_l2_mean"]
            ),
            "action_saturation_fraction": float(
                trace["actions"]["fraction_abs_ge_0_95"]
                - baseline_trace["actions"]["fraction_abs_ge_0_95"]
            ),
            "motion_action_saturation_fraction": float(
                trace["actions"]["motion_fraction_abs_ge_0_95"]
                - baseline_trace["actions"]["motion_fraction_abs_ge_0_95"]
            ),
            "eef_net_displacement_m": float(
                trace["eef"]["net_displacement_m"]
                - baseline_trace["eef"]["net_displacement_m"]
            ),
            "eef_path_length_m": float(
                trace["eef"]["path_length_m"]
                - baseline_trace["eef"]["path_length_m"]
            ),
        }

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "scope": "stage_d_pretrained_baseline_and_rollout_trace_diagnostics",
        "source_stage_d_summary": str(stage_d_summary_path),
        "gpu_work": "one E0 rollout with zero optimizer updates; existing E0/E1/E2 traces are read only",
        "records": records,
        "success_count": int(sum(record["success"] for record in records)),
        "all_rewards_zero": bool(
            all(record["trace_summary"]["rewards"]["sum"] == 0.0 for record in records)
        ),
        "interpretation_limits": [
            "The no-update E0 rollout separates pretrained-policy behavior from the two-step optimizer smoke.",
            "One rollout per condition is diagnostic only and is not a success-rate estimate.",
            "Geometry gate gain is an uncalibrated condition value, not a correctness probability.",
        ],
    }
    output = (args.output or diagnostics_root / "summary.json").resolve()
    _write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
