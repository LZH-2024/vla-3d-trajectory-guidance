#!/usr/bin/env python3
"""Run one fail-stop GPU acceptance stage using explicit target interpreters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.experiment import RunLedger
from trajectory_guidance.runtime_environment import environment_with_venv_npp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list("ABCDE"), required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _run(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
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


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _find_probe(root: Path, camera_name: str) -> Path:
    candidates: list[tuple[str, Path]] = []
    for manifest_path in root.glob("*/observation_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        camera = manifest.get("cameras", {}).get(camera_name, {})
        if manifest.get("valid") is True and camera.get("valid") is True:
            candidates.append((str(manifest.get("created_at_utc", "")), manifest_path.parent))
    if not candidates:
        raise FileNotFoundError(
            f"no valid saved probe for camera {camera_name!r} under {root}"
        )
    return max(candidates, key=lambda item: (item[0], item[1].as_posix()))[1]


def _stage_a(
    project_root: Path, run_dir: Path, config: dict[str, Any], attempt: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = config["stage_a"]
    stage_dir = run_dir / "stage_a" / f"attempt_{attempt:03d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    planner_python = Path(config["planner_python"])
    smolvla_python = Path(config["smolvla_python"])
    for interpreter in (planner_python, smolvla_python):
        if not interpreter.is_file():
            raise FileNotFoundError(f"target interpreter is missing: {interpreter}")

    planner_subprocess_env = environment_with_venv_npp(planner_python)
    smolvla_subprocess_env = environment_with_venv_npp(smolvla_python)

    planner_env_output = stage_dir / "planner_environment.json"
    _run(
        [
            str(planner_python),
            str(project_root / "scripts" / "check_planner_environment.py"),
            "--project-root",
            str(project_root),
            "--config",
            str(_resolve_project_path(project_root, settings["planner_config"])),
            "--output",
            str(planner_env_output),
            "--gpu-test",
        ],
        stage_dir / "planner_environment.log",
        env=planner_subprocess_env,
    )

    libero_env_output = stage_dir / "libero_environment.json"
    libero_config = _resolve_project_path(project_root, settings["libero_config"])
    libero_config_payload = json.loads(libero_config.read_text(encoding="utf-8"))
    libero_settings = libero_config_payload["libero"]
    smolvla_subprocess_env.update(
        {
            "MUJOCO_GL": str(libero_settings["render_backend"]),
            "MPLBACKEND": str(libero_settings["matplotlib_backend"]),
            "LIBERO_CONFIG_PATH": str(libero_settings["config_dir"]),
        }
    )
    _run(
        [
            str(smolvla_python),
            str(project_root / "scripts" / "check_libero_environment.py"),
            "--project-root",
            str(project_root),
            "--config",
            str(libero_config),
            "--output",
            str(libero_env_output),
            "--gpu-test",
        ],
        stage_dir / "libero_environment.log",
        env=smolvla_subprocess_env,
    )

    probe_override = os.environ.get("VLA_LIBERO_PROBE_DIR") or settings.get("probe_dir")
    probe_dir = (
        Path(probe_override).expanduser().resolve()
        if probe_override
        else _find_probe(Path(settings["probe_output_root"]), settings["camera_name"])
    )
    input_output = stage_dir / "planner_input.json"
    _run(
        [
            str(planner_python),
            str(project_root / "scripts" / "check_libero_planner_input.py"),
            "--probe-dir",
            str(probe_dir),
            "--camera-name",
            str(settings["camera_name"]),
            "--output",
            str(input_output),
        ],
        stage_dir / "planner_input.log",
        env=planner_subprocess_env,
    )

    checkpoint_manifest_path = _resolve_project_path(
        project_root, settings["planner_checkpoint_manifest"]
    )
    if not checkpoint_manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {checkpoint_manifest_path}")
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    planner_config = json.loads(
        _resolve_project_path(project_root, settings["planner_config"]).read_text(
            encoding="utf-8"
        )
    )
    expected_revision = planner_config["planner"]["model_revision"]
    if checkpoint_manifest.get("requested_revision") != expected_revision:
        raise RuntimeError("3D HAMSTER checkpoint revision mismatch")
    checkpoint_dir = Path(settings["planner_checkpoint_dir"])
    if not (checkpoint_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"planner checkpoint is incomplete: {checkpoint_dir}")

    planner_environment = json.loads(planner_env_output.read_text(encoding="utf-8"))
    libero_environment = json.loads(libero_env_output.read_text(encoding="utf-8"))
    planner_input = json.loads(input_output.read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "valid": True,
        "stage": "A",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_dir": str(probe_dir),
        "checkpoint_revision": expected_revision,
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "planner_environment": str(planner_env_output),
        "libero_environment": str(libero_env_output),
        "planner_input": str(input_output),
    }
    _write_json(stage_dir / "summary.json", summary)
    metrics = {
        "planner_gpu": planner_environment.get("gpu"),
        "smolvla_gpu": libero_environment.get("gpu"),
        "input_depth_validation": planner_input.get("depth_validation"),
    }
    return metrics, {"summary": str(stage_dir / "summary.json")}


def _stage_b(
    project_root: Path, run_dir: Path, config: dict[str, Any], attempt: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger_state = json.loads(
        (run_dir / "acceptance_ledger.json").read_text(encoding="utf-8")
    )
    stage_a_summary_value = ledger_state["stages"]["A"]["artifacts"].get("summary")
    stage_a_summary_path = Path(stage_a_summary_value) if stage_a_summary_value else Path()
    if not stage_a_summary_path.is_file():
        raise FileNotFoundError("Stage A summary is missing")
    stage_a = json.loads(stage_a_summary_path.read_text(encoding="utf-8"))
    stage_dir = run_dir / "stage_b" / f"attempt_{attempt:03d}"
    output_dir = stage_dir / "planner_result"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(Path(config["planner_python"])),
            str(project_root / "scripts" / "run_libero_planner_single_frame.py"),
            "--project-root",
            str(project_root),
            "--config",
            str(
                _resolve_project_path(
                    project_root, config["stage_b"]["single_frame_config"]
                )
            ),
            "--probe-dir",
            stage_a["probe_dir"],
            "--output-dir",
            str(output_dir),
        ],
        stage_dir / "planner.log",
        env=environment_with_venv_npp(Path(config["planner_python"])),
    )
    summary_path = output_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("valid") is not True:
        raise RuntimeError("Stage B planner output is invalid")
    geometry = summary.get("geometry_validation") or {}
    if geometry.get("valid") is not True:
        raise RuntimeError("Stage B geometry validation failed")
    cache_record = summary.get("artifacts", {}).get("trajectory_cache_record")
    if not cache_record or not Path(cache_record).is_dir():
        raise RuntimeError("Stage B did not save a unified trajectory cache record")
    return (
        {
            "waypoint_count": summary["waypoint_count"],
            "timing_seconds": summary["timing_seconds"],
            "gpu": summary["gpu"],
            "geometry": geometry,
        },
        {
            "summary": str(summary_path),
            "trajectory_cache_record": cache_record,
        },
    )


def _stage_c(
    project_root: Path, run_dir: Path, config: dict[str, Any], attempt: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = config["stage_c"]
    policy_value = os.environ.get("VLA_SMOLVLA_POLICY_PATH") or settings.get("policy_path")
    batch_value = os.environ.get("VLA_SMOLVLA_BATCH_ARTIFACT") or settings.get("batch_artifact")
    if not policy_value or not batch_value:
        raise ValueError(
            "Stage C local policy_path and batch_artifact are unset; no download was attempted"
        )
    policy_path = Path(policy_value).expanduser().resolve()
    batch_path = Path(batch_value).expanduser().resolve()
    stage_dir = run_dir / "stage_c" / f"attempt_{attempt:03d}"
    output = stage_dir / "smolvla_smoke.json"
    _run(
        [
            str(Path(config["smolvla_python"])),
            str(project_root / "scripts" / "run_smolvla_real_smoke.py"),
            "--project-root",
            str(project_root),
            "--policy-path",
            str(policy_path),
            "--batch-artifact",
            str(batch_path),
            "--device",
            str(settings["device"]),
            "--output",
            str(output),
        ],
        stage_dir / "smolvla_smoke.log",
        env=environment_with_venv_npp(Path(config["smolvla_python"])),
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("all_passed") is not True:
        raise RuntimeError("Stage C SmolVLA smoke failed")
    return (
        {
            "model_load_seconds": result["model_load_seconds"],
            "gpu_after_model_load": result["gpu_after_model_load"],
            "experiments": result["results"],
        },
        {"summary": str(output)},
    )


def _configured_command_stage(
    stage: str, run_dir: Path, config: dict[str, Any], attempt: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = config[f"stage_{stage.lower()}"].get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(
            f"Stage {stage} command is intentionally unset; configure it only after prior stages pass"
        )
    formatted = [str(item).format(run_dir=str(run_dir)) for item in command]
    log = (
        run_dir
        / f"stage_{stage.lower()}"
        / f"attempt_{attempt:03d}"
        / "command.log"
    )
    _run(formatted, log)
    return {"command_completed": True}, {"log": str(log)}


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config or project_root / "configs" / "gpu_acceptance_colab.json"
    ).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("GPU acceptance config schema mismatch")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(run_dir / "acceptance_ledger.json")
    if ledger.state["stages"][args.stage]["status"] == "passed":
        print(f"Stage {args.stage} already passed; leaving recorded evidence unchanged.")
        return 0

    # Validate intentionally deferred stages before changing their ledger state.
    if args.stage == "C":
        settings = config["stage_c"]
        if not (os.environ.get("VLA_SMOLVLA_POLICY_PATH") or settings.get("policy_path")):
            raise ValueError("Stage C policy path is unset")
        if not (os.environ.get("VLA_SMOLVLA_BATCH_ARTIFACT") or settings.get("batch_artifact")):
            raise ValueError("Stage C batch artifact path is unset")
    if args.stage in {"D", "E"}:
        command = config[f"stage_{args.stage.lower()}"].get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"Stage {args.stage} command is intentionally unset")

    ledger.begin(args.stage)
    attempt = int(ledger.state["stages"][args.stage]["attempts"])
    try:
        if args.stage == "A":
            metrics, artifacts = _stage_a(project_root, run_dir, config, attempt)
        elif args.stage == "B":
            metrics, artifacts = _stage_b(project_root, run_dir, config, attempt)
        elif args.stage == "C":
            metrics, artifacts = _stage_c(project_root, run_dir, config, attempt)
        else:
            metrics, artifacts = _configured_command_stage(
                args.stage, run_dir, config, attempt
            )
        ledger.pass_stage(args.stage, metrics=metrics, artifacts=artifacts)
        print(f"Stage {args.stage} passed. Later stages remain manual and fail-stop.")
        return 0
    except Exception as exc:
        ledger.fail_stage(args.stage, f"{type(exc).__name__}: {exc}")
        failure_path = (
            run_dir
            / f"stage_{args.stage.lower()}"
            / f"attempt_{attempt:03d}"
            / "failure.json"
        )
        _write_json(
            failure_path,
            {
                "stage": args.stage,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
