#!/usr/bin/env python3
"""Run 3D HAMSTER once on a validated LIBERO RGB-D probe artifact."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.libero_planner import (
    build_trajectory_geometry,
    find_latest_valid_probe,
    load_libero_planner_input,
)
from trajectory_guidance.libero_probe import write_json
from trajectory_guidance.validation import validate_waypoints
from trajectory_guidance.visualization import save_trajectory_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--camera-name")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def gpu_snapshot(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "device_index": index,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "allocated_bytes": torch.cuda.memory_allocated(index),
        "reserved_bytes": torch.cuda.memory_reserved(index),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
    }


def write_uvd_csv(path: Path, waypoints: np.ndarray, actions: list[str | None]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "u_normalized", "v_normalized", "depth_m", "action"])
        for index, point in enumerate(waypoints):
            action = actions[index] if index < len(actions) else None
            writer.writerow(
                [index, float(point[0]), float(point[1]), float(point[2]), action or ""]
            )


def write_xyz_csv(
    path: Path,
    points: np.ndarray,
    actions: list[str | None],
    frame: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", f"x_{frame}_m", f"y_{frame}_m", f"z_{frame}_m", "action"])
        for index, point in enumerate(points):
            action = actions[index] if index < len(actions) else None
            writer.writerow(
                [index, float(point[0]), float(point[1]), float(point[2]), action or ""]
            )


def validate_checkpoint_record(
    project_root: Path,
    planner_config: dict[str, Any],
    model_path: Path,
) -> tuple[Path, dict[str, Any]]:
    record_path = project_root / "outputs" / "environment" / "planner_checkpoint_manifest.json"
    if not record_path.is_file():
        raise FileNotFoundError(
            f"checkpoint provenance record is missing: {record_path}; run 01_planner_setup.ipynb"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = planner_config["planner"]
    if record.get("repo_id") != expected["model_id"]:
        raise RuntimeError("checkpoint model id does not match planner_colab.json")
    if record.get("requested_revision") != expected["model_revision"]:
        raise RuntimeError("checkpoint revision does not match planner_colab.json")
    if Path(record.get("checkpoint_dir", "")).resolve() != model_path:
        raise RuntimeError("checkpoint manifest path does not match the requested model path")
    return record_path, record


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config or project_root / "configs" / "libero_planner_single_frame.json"
    ).resolve()
    bridge_config = json.loads(config_path.read_text(encoding="utf-8"))
    planner_config_path = (project_root / bridge_config["planner_config"]).resolve()
    probe_config_path = (project_root / bridge_config["probe_config"]).resolve()
    planner_config = json.loads(planner_config_path.read_text(encoding="utf-8"))
    probe_config = json.loads(probe_config_path.read_text(encoding="utf-8"))
    planner = planner_config["planner"]
    single = bridge_config["single_frame"]
    if Path(single["probe_output_root"]) != Path(probe_config["probe"]["output_root"]):
        raise RuntimeError("bridge and probe configs disagree on the probe output root")
    if list(single["accepted_depth_range_m"]) != list(
        probe_config["probe"]["accepted_depth_range_m"]
    ):
        raise RuntimeError("bridge and probe configs disagree on the accepted depth range")
    if single.get("offline", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    expected_python = (Path(planner["env_dir"]) / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")
    camera_name = args.camera_name or single["camera_name"]
    device = args.device or single["device"]
    max_new_tokens = (
        int(single["max_new_tokens"])
        if args.max_new_tokens is None
        else args.max_new_tokens
    )
    model_path = (args.model_path or Path(planner["checkpoint_dir"])).resolve()
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"planner checkpoint is incomplete: {model_path}")
    checkpoint_record_path, checkpoint_record = validate_checkpoint_record(
        project_root, planner_config, model_path
    )

    probe_root = Path(single["probe_output_root"])
    probe_dir = (
        args.probe_dir.resolve()
        if args.probe_dir
        else find_latest_valid_probe(probe_root, camera_name=camera_name)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir
        or Path(single["output_root"]) / f"{probe_dir.name}_{camera_name}_{timestamp}"
    ).resolve()
    ensure_empty_output(output_dir)

    try:
        min_depth_m, max_depth_m = map(
            float, single.get("accepted_depth_range_m", probe_config["probe"]["accepted_depth_range_m"])
        )
        planner_input = load_libero_planner_input(
            probe_dir,
            camera_name=camera_name,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )

        import torch
        from hamster3d.inference import Hamster3DPredictor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; planner model loading was not attempted")
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError(f"3D HAMSTER requires a CUDA device, got {device}")
        torch.cuda.set_device(requested_device.index or 0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        load_started = time.perf_counter()
        predictor = Hamster3DPredictor(str(model_path), device=device)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        model_load_gpu = gpu_snapshot(torch)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        result = predictor.predict(
            planner_input.rgb,
            planner_input.depth_m,
            planner_input.instruction,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        inference_gpu = gpu_snapshot(torch)

        raw_output = str(result["raw_output"])
        (output_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
        waypoints_list = [[float(value) for value in point] for point in result["waypoints"]]
        actions = list(result["actions"])
        output_validation = validate_waypoints(
            waypoints_list,
            actions,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        waypoints = (
            np.asarray(waypoints_list, dtype=np.float64).reshape(-1, 3)
            if waypoints_list
            else np.empty((0, 3), dtype=np.float64)
        )

        geometry = None
        if output_validation.valid:
            geometry = build_trajectory_geometry(
                waypoints,
                planner_input,
                max_roundtrip_error=float(single["max_projection_roundtrip_error"]),
            )

        artifacts: dict[str, str | None] = {
            "raw_output_txt": str(output_dir / "raw_output.txt"),
            "raw_prediction_json": str(output_dir / "raw_prediction.json"),
            "run_summary_json": str(output_dir / "run_summary.json"),
            "trajectory_overlay_png": None,
            "waypoints_uvd_csv": None,
            "waypoints_camera_xyz_csv": None,
            "waypoints_world_xyz_csv": None,
        }
        if len(waypoints):
            write_uvd_csv(output_dir / "waypoints_uvd.csv", waypoints, actions)
            save_trajectory_overlay(
                planner_input.rgb,
                waypoints_list,
                actions,
                output_dir / "trajectory_overlay.png",
            )
            artifacts["waypoints_uvd_csv"] = str(output_dir / "waypoints_uvd.csv")
            artifacts["trajectory_overlay_png"] = str(output_dir / "trajectory_overlay.png")
        if geometry is not None:
            write_xyz_csv(
                output_dir / "waypoints_camera_xyz.csv",
                geometry.camera_xyz_m,
                actions,
                "camera",
            )
            write_xyz_csv(
                output_dir / "waypoints_world_xyz.csv",
                geometry.world_xyz_m,
                actions,
                "world",
            )
            artifacts["waypoints_camera_xyz_csv"] = str(
                output_dir / "waypoints_camera_xyz.csv"
            )
            artifacts["waypoints_world_xyz_csv"] = str(output_dir / "waypoints_world_xyz.csv")

        resized_rgb = np.asarray(result["rgb_resized"])
        resized_depth = np.asarray(result["depth_resized"])
        preprocessing_valid = (
            resized_rgb.ndim == 3
            and resized_rgb.shape[2] == 3
            and resized_rgb.shape[:2] == resized_depth.shape
            and max(resized_rgb.shape[:2]) == 640
        )
        geometry_valid = geometry is not None and geometry.report.valid
        valid = bool(output_validation.valid and geometry_valid and preprocessing_valid)
        prediction = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "valid": valid,
            "scope": "single_frame_libero_planner_feasibility_no_policy_no_action_execution",
            "instruction": planner_input.instruction,
            "camera_name": camera_name,
            "coordinate_conventions": {
                "planner_output": "normalized_uv_0_1000_plus_camera_z_depth_m",
                "camera": "opencv_camera_x_right_y_down_z_forward",
                "world": "libero_mujoco_world",
                "extrinsics": "p_world=T_world_camera@p_camera",
                "world_xyz_is_not_robot_base_action": True,
            },
            "source_probe": {
                "directory": str(planner_input.probe_dir),
                "manifest": str(planner_input.manifest_path),
                "suite": planner_input.manifest["task"]["suite"],
                "task_id": planner_input.manifest["task"]["task_id"],
                "init_state_id": planner_input.manifest["task"]["init_state_id"],
                "seed": planner_input.manifest["task"]["seed"],
                "hashes": planner_input.hashes,
            },
            "input": {
                "rgb_path": str(planner_input.rgb_path),
                "depth_m_path": str(planner_input.depth_path),
                "calibration_path": str(planner_input.calibration_path),
                "robot_state_path": str(planner_input.robot_state_path),
                "original_rgb_size_wh": list(planner_input.rgb.size),
                "depth_shape_hw": list(planner_input.depth_m.shape),
                "depth_validation": planner_input.depth_validation.to_dict(),
                "calibration_validation": planner_input.calibration_validation.to_dict(),
                "intrinsics_K": planner_input.intrinsics,
                "T_world_camera": planner_input.t_world_camera,
                "eef_position_world_m": planner_input.eef_position_world_m,
            },
            "planner": {
                "repo_url": planner["repo_url"],
                "repo_commit": planner["repo_commit"],
                "model_id": planner["model_id"],
                "model_revision": planner["model_revision"],
                "model_path": str(model_path),
                "checkpoint_record": str(checkpoint_record_path),
                "checkpoint_record_revision": checkpoint_record["requested_revision"],
                "device": device,
                "max_new_tokens": max_new_tokens,
                "offline": bool(single.get("offline", True)),
            },
            "preprocessing": {
                "valid": preprocessing_valid,
                "method": "official_predictor_longest_edge_640_rgb_linear_depth_nearest",
                "uvd_to_pixel": "u_px=u_normalized/1000*width; v_px=v_normalized/1000*height",
                "rgb_resized_shape": list(resized_rgb.shape),
                "depth_resized_shape": list(resized_depth.shape),
                "normalized_uv_is_scale_invariant_under_uniform_resize": True,
            },
            "raw_output": raw_output,
            "waypoints_uvd": waypoints,
            "actions": actions,
            "output_validation": output_validation.to_dict(),
            "geometry_validation": geometry.report.to_dict() if geometry else None,
            "camera_xyz_m": geometry.camera_xyz_m if geometry else None,
            "world_xyz_m": geometry.world_xyz_m if geometry else None,
            "scene_depth_at_uv_m": geometry.scene_depth_at_uv_m if geometry else None,
            "planner_minus_scene_depth_m": (
                geometry.planner_minus_scene_depth_m if geometry else None
            ),
            "timing_seconds": {
                "model_load": model_load_seconds,
                "inference": inference_seconds,
            },
            "gpu": {
                "after_model_load": model_load_gpu,
                "after_inference": inference_gpu,
            },
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "artifacts": artifacts,
        }
        write_json(output_dir / "raw_prediction.json", prediction)
        summary = {
            "schema_version": 1,
            "valid": valid,
            "scope": prediction["scope"],
            "instruction": planner_input.instruction,
            "camera_name": camera_name,
            "source_probe_dir": str(planner_input.probe_dir),
            "output_dir": str(output_dir),
            "waypoint_count": int(len(waypoints)),
            "output_validation": output_validation.to_dict(),
            "geometry_validation": geometry.report.to_dict() if geometry else None,
            "preprocessing": prediction["preprocessing"],
            "timing_seconds": prediction["timing_seconds"],
            "gpu": prediction["gpu"],
            "planner_provenance": prediction["planner"],
            "artifacts": artifacts,
            "interpretation": (
                "geometry_feasibility_only; no policy, simulator action, or task-success claim"
            ),
        }
        write_json(output_dir / "run_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if valid else 2
    except Exception as exc:
        error = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "probe_dir": str(probe_dir),
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "error.json", error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
