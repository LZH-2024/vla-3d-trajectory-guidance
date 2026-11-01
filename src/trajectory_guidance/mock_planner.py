"""Deterministic CPU-only planner substitute for artifact and geometry tests."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from trajectory_guidance.libero_planner import (
    build_trajectory_geometry,
    load_libero_planner_input,
)
from trajectory_guidance.libero_probe import save_depth_preview, sha256_file, write_json
from trajectory_guidance.trajectory_record import (
    CameraCalibration,
    EpisodeIdentity,
    ModelIdentity,
    TrajectoryCache,
    TrajectoryRecord,
)
from trajectory_guidance.validation import validate_waypoints
from trajectory_guidance.visualization import save_trajectory_overlay


MOCK_MODEL_ID = "mock/fixed-uvd-v1"
MOCK_CASES: dict[str, dict[str, Any]] = {
    "valid": {
        "waypoints": [[350.0, 540.0, 0.90], [500.0, 500.0, 0.82], [650.0, 460.0, 0.74]],
        "actions": [None, "Close Gripper", None],
        "expected_valid": True,
    },
    "empty": {"waypoints": [], "actions": [], "expected_valid": False},
    "out_of_bounds": {
        "waypoints": [[-10.0, 500.0, 0.8], [1100.0, 500.0, 0.8]],
        "actions": [None, None],
        "expected_valid": False,
    },
    "invalid_depth": {
        "waypoints": [[500.0, 500.0, 0.0], [550.0, 500.0, 12.0]],
        "actions": [None, None],
        "expected_valid": False,
    },
}


def create_mock_probe(
    destination: str | Path,
    *,
    invalid_input_depth: bool = False,
) -> Path:
    """Create a tiny probe matching the real 03 artifact contract."""

    root = Path(destination).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"mock probe directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    height = width = 64
    u = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    v = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    rgb = np.stack(
        [
            np.broadcast_to(u, (height, width)),
            np.broadcast_to(v, (height, width)),
            np.full((height, width), 96, dtype=np.uint8),
        ],
        axis=-1,
    )
    depth = np.linspace(0.7, 1.1, height * width, dtype=np.float32).reshape(height, width)
    if invalid_input_depth:
        depth[0, 0] = np.nan
    intrinsics = np.array(
        [[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    t_world_camera = np.eye(4, dtype=np.float64)
    t_world_camera[:3, 3] = [0.25, -0.10, 0.80]
    eef_position = np.array([0.25, -0.10, 1.60], dtype=np.float64)

    rgb_path = root / "agentview_rgb.png"
    depth_path = root / "agentview_depth_m.npy"
    preview_path = root / "agentview_depth_preview.png"
    calibration_path = root / "agentview_calibration.json"
    robot_state_path = root / "robot_state.json"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    np.save(depth_path, depth, allow_pickle=False)
    save_depth_preview(depth, preview_path)
    write_json(
        calibration_path,
        {
            "camera_name": "agentview",
            "image_size": {"width": width, "height": height},
            "image_convention": "opencv_top_left_u_right_v_down",
            "depth_buffer": {"metric_depth_semantics": "camera_z_depth_m"},
            "camera_frame": "opencv_camera_x_right_y_down_z_forward",
            "world_frame": "libero_mujoco_world",
            "intrinsics_K": intrinsics,
            "T_world_camera": t_world_camera,
            "T_camera_world": np.linalg.inv(t_world_camera),
            "extrinsics_convention": "T_target_source; p_world=T_world_camera@p_camera",
        },
    )
    write_json(
        robot_state_path,
        {
            "frame": "libero_mujoco_world",
            "robot0_eef_pos_m": eef_position,
            "robot0_eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "T_world_eef": np.eye(4),
        },
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        # Keep the manifest marked valid even in the corruption case so the
        # consumer must independently revalidate the numeric depth array.
        "valid": True,
        "scope": "single_frame_libero_observation_probe_no_planner_no_policy",
        "task": {
            "suite": "mock_libero",
            "task_id": 0,
            "task_name": "mock_pick",
            "language_instruction": "pick up the mock cube",
            "init_state_id": 0,
            "seed": 0,
        },
        "coordinate_conventions": {
            "image": "opencv_top_left_u_right_v_down",
            "metric_depth": "camera_z_depth_m",
            "camera": "x_right_y_down_z_forward",
            "world": "libero_mujoco_world",
            "extrinsics": "T_target_source",
            "eef_quaternion": "xyzw",
        },
        "provenance": {
            "mock": True,
            "generator": MOCK_MODEL_ID,
            "no_real_model_or_simulator": True,
        },
        "robot_state_json": str(robot_state_path),
        "cameras": {
            "agentview": {
                "valid": True,
                "rgb": {
                    "shape": list(rgb.shape),
                    "dtype": str(rgb.dtype),
                    "path": str(rgb_path),
                    "sha256": sha256_file(rgb_path),
                },
                "artifacts": {
                    "depth_m_npy": str(depth_path),
                    "depth_preview_png": str(preview_path),
                    "calibration_json": str(calibration_path),
                },
            }
        },
    }
    write_json(root / "observation_manifest.json", manifest)
    return root


def _write_uvd_csv(
    path: Path, waypoints: np.ndarray, actions: list[str | None]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "u_normalized", "v_normalized", "depth_m", "action"])
        for index, point in enumerate(waypoints):
            action = actions[index] if index < len(actions) else None
            writer.writerow([index, *map(float, point), action or ""])


def run_mock_case(
    probe_dir: str | Path,
    output_dir: str | Path,
    case_name: str,
) -> dict[str, Any]:
    """Run one deterministic output case through the real geometry utilities."""

    if case_name not in MOCK_CASES:
        raise ValueError(f"unknown mock case: {case_name}")
    case = MOCK_CASES[case_name]
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"mock output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    planner_input = load_libero_planner_input(probe_dir, camera_name="agentview")
    waypoints_list = case["waypoints"]
    actions = list(case["actions"])
    validation = validate_waypoints(waypoints_list, actions)
    waypoints = (
        np.asarray(waypoints_list, dtype=np.float64).reshape(-1, 3)
        if waypoints_list
        else np.empty((0, 3), dtype=np.float64)
    )
    geometry = (
        build_trajectory_geometry(waypoints, planner_input)
        if validation.valid
        else None
    )
    actual_valid = bool(validation.valid and geometry is not None and geometry.report.valid)
    expected_valid = bool(case["expected_valid"])

    npz_path = destination / "mock_arrays.npz"
    np.savez_compressed(
        npz_path,
        waypoints_uvd=waypoints,
        actions=np.asarray([action or "none" for action in actions], dtype="U16"),
        camera_xyz_m=(
            geometry.camera_xyz_m if geometry is not None else np.empty((0, 3))
        ),
        world_xyz_m=(
            geometry.world_xyz_m if geometry is not None else np.empty((0, 3))
        ),
    )
    csv_path = destination / "waypoints_uvd.csv"
    _write_uvd_csv(csv_path, waypoints, actions)
    overlay_path: Path | None = None
    cache_path: Path | None = None
    if len(waypoints):
        overlay_path = save_trajectory_overlay(
            planner_input.rgb,
            waypoints_list,
            actions,
            destination / "mock_trajectory_overlay.png",
        )
    if actual_valid and geometry is not None:
        calibration = CameraCalibration.from_arrays(
            camera_name=planner_input.camera_name,
            image_width=planner_input.rgb.width,
            image_height=planner_input.rgb.height,
            intrinsics_k=planner_input.intrinsics,
            t_world_camera=planner_input.t_world_camera,
        )
        record = TrajectoryRecord(
            trajectory_id=f"mock-{case_name}",
            source="mock",
            episode=EpisodeIdentity(
                dataset="mock",
                suite="mock_libero",
                task_id="0",
                episode_id="mock-episode-0",
                init_state_id=0,
                observation_index=0,
            ),
            model=ModelIdentity(
                name=MOCK_MODEL_ID,
                version="1",
                code_revision="mock-only",
            ),
            instruction=planner_input.instruction,
            input_image_sha256=planner_input.hashes["rgb_sha256"],
            coordinate_frame="libero_mujoco_world",
            waypoints_xyz_m=geometry.world_xyz_m,
            valid_mask=np.ones(len(geometry.world_xyz_m), dtype=np.bool_),
            gripper_events=tuple(
                "close" if action and "close" in action.lower() else "none"
                for action in actions
            ),
            calibration=calibration,
            available_at_observation_index=0,
            provenance={"mock": True, "case": case_name},
        )
        cache_path = TrajectoryCache(destination / "trajectory_cache").save(record)

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "cpu_mock_planner_validation_no_real_model_no_gpu",
        "mock": True,
        "mock_model_id": MOCK_MODEL_ID,
        "case": case_name,
        "expected_valid": expected_valid,
        "actual_valid": actual_valid,
        "case_passed": actual_valid == expected_valid,
        "output_validation": validation.to_dict(),
        "geometry_validation": geometry.report.to_dict() if geometry else None,
        "raw_mock_output": {
            "waypoints_uvd": waypoints,
            "actions": actions,
            "camera_xyz_m": geometry.camera_xyz_m if geometry else None,
            "world_xyz_m": geometry.world_xyz_m if geometry else None,
        },
        "source_probe": str(Path(probe_dir).resolve()),
        "artifacts": {
            "prediction_json": str(destination / "mock_prediction.json"),
            "arrays_npz": str(npz_path),
            "waypoints_csv": str(csv_path),
            "overlay_png": str(overlay_path) if overlay_path else None,
            "trajectory_cache_record": str(cache_path) if cache_path else None,
        },
        "warning": "Mock results are format tests only and are not 3D HAMSTER results.",
    }
    write_json(destination / "mock_prediction.json", summary)
    return summary


def run_mock_suite(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"mock suite output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    probe = create_mock_probe(root / "mock_probe")
    cases = [
        run_mock_case(probe, root / "cases" / case_name, case_name)
        for case_name in MOCK_CASES
    ]
    suite = {
        "schema_version": 1,
        "scope": "cpu_mock_planner_validation_no_real_model_no_gpu",
        "mock": True,
        "all_cases_passed": all(case["case_passed"] for case in cases),
        "cases": cases,
        "mock_probe": str(probe),
    }
    write_json(root / "mock_suite_summary.json", suite)
    return suite
