"""Bridge one validated LIBERO observation to 3D HAMSTER geometry.

This module deliberately has no simulator or Torch dependency.  LIBERO writes
files with ``/content/.venv`` and the planner later reads those files with
``/content/.venv_planner``.  The file boundary keeps the conflicting model
environments isolated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from trajectory_guidance.geometry import (
    camera_xyz_to_normalized_uvd,
    normalized_uvd_to_camera_xyz,
)
from trajectory_guidance.libero_probe import (
    sha256_file,
    transform_points,
    validate_camera_calibration,
)
from trajectory_guidance.validation import ValidationReport, validate_depth_map


EXPECTED_CONVENTIONS = {
    "image": "opencv_top_left_u_right_v_down",
    "metric_depth": "camera_z_depth_m",
    "camera": "x_right_y_down_z_forward",
    "world": "libero_mujoco_world",
    "extrinsics": "T_target_source",
}


@dataclass(frozen=True)
class LiberoPlannerInput:
    probe_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    camera_name: str
    instruction: str
    rgb_path: Path
    depth_path: Path
    calibration_path: Path
    robot_state_path: Path
    rgb: Image.Image
    depth_m: np.ndarray
    intrinsics: np.ndarray
    t_world_camera: np.ndarray
    eef_position_world_m: np.ndarray
    depth_validation: ValidationReport
    calibration_validation: ValidationReport
    hashes: dict[str, str]


@dataclass(frozen=True)
class TrajectoryGeometry:
    camera_xyz_m: np.ndarray
    world_xyz_m: np.ndarray
    scene_depth_at_uv_m: np.ndarray
    planner_minus_scene_depth_m: np.ndarray
    report: ValidationReport


def _manifest_time(manifest: dict[str, Any], path: Path) -> tuple[float, str]:
    value = str(manifest.get("created_at_utc", ""))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp(), path.as_posix()
    except ValueError:
        return path.stat().st_mtime, path.as_posix()


def find_latest_valid_probe(output_root: str | Path, camera_name: str = "agentview") -> Path:
    """Return the newest valid probe directory containing the requested camera."""

    root = Path(output_root).expanduser().resolve()
    candidates: list[tuple[tuple[float, str], Path]] = []
    if root.is_dir():
        for manifest_path in root.glob("*/observation_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            camera = manifest.get("cameras", {}).get(camera_name, {})
            if manifest.get("valid") is True and camera.get("valid") is True:
                candidates.append((_manifest_time(manifest, manifest_path), manifest_path.parent))
    if not candidates:
        raise FileNotFoundError(
            f"no valid LIBERO probe with camera {camera_name!r} was found under {root}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def resolve_recorded_artifact(probe_dir: str | Path, recorded_path: str | Path) -> Path:
    """Resolve an artifact path, with a basename fallback for a moved probe folder."""

    root = Path(probe_dir).resolve()
    recorded = Path(recorded_path).expanduser()
    if recorded.is_file():
        return recorded.resolve()
    relative_candidate = root / recorded
    if not recorded.is_absolute() and relative_candidate.is_file():
        return relative_candidate.resolve()
    basename_candidate = root / recorded.name
    if basename_candidate.is_file():
        return basename_candidate.resolve()
    raise FileNotFoundError(f"probe artifact does not exist: {recorded_path}")


def load_libero_planner_input(
    probe_dir: str | Path,
    *,
    camera_name: str = "agentview",
    min_depth_m: float = 0.01,
    max_depth_m: float = 10.0,
) -> LiberoPlannerInput:
    """Load and revalidate the exact files consumed by the planner."""

    root = Path(probe_dir).expanduser().resolve()
    manifest_path = root / "observation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing probe manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("valid") is not True:
        raise ValueError(f"probe manifest is not valid: {manifest_path}")
    if manifest.get("scope") != "single_frame_libero_observation_probe_no_planner_no_policy":
        raise ValueError(f"unexpected probe scope: {manifest.get('scope')!r}")

    conventions = manifest.get("coordinate_conventions", {})
    mismatches = {
        key: {"expected": expected, "actual": conventions.get(key)}
        for key, expected in EXPECTED_CONVENTIONS.items()
        if conventions.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"probe coordinate convention mismatch: {mismatches}")

    camera_record = manifest.get("cameras", {}).get(camera_name)
    if not camera_record or camera_record.get("valid") is not True:
        raise ValueError(f"camera {camera_name!r} is missing or invalid in the probe manifest")
    instruction = str(manifest.get("task", {}).get("language_instruction", "")).strip()
    if not instruction:
        raise ValueError("probe task has no language instruction")

    rgb_path = resolve_recorded_artifact(root, camera_record["rgb"]["path"])
    depth_path = resolve_recorded_artifact(
        root, camera_record["artifacts"]["depth_m_npy"]
    )
    calibration_path = resolve_recorded_artifact(
        root, camera_record["artifacts"]["calibration_json"]
    )
    robot_state_path = resolve_recorded_artifact(root, manifest["robot_state_json"])

    rgb = Image.open(rgb_path).convert("RGB")
    depth_m = np.load(depth_path, allow_pickle=False)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    robot_state = json.loads(robot_state_path.read_text(encoding="utf-8"))
    image_size = calibration.get("image_size", {})
    width = int(image_size.get("width", -1))
    height = int(image_size.get("height", -1))
    if rgb.size != (width, height):
        raise ValueError(
            f"RGB/calibration size mismatch: rgb={rgb.size}, calibration={(width, height)}"
        )
    if calibration.get("camera_name") != camera_name:
        raise ValueError(
            f"calibration camera mismatch: {calibration.get('camera_name')!r} != {camera_name!r}"
        )
    if calibration.get("image_convention") != EXPECTED_CONVENTIONS["image"]:
        raise ValueError("calibration image convention does not match the probe contract")
    depth_semantics = calibration.get("depth_buffer", {}).get("metric_depth_semantics")
    if depth_semantics != EXPECTED_CONVENTIONS["metric_depth"]:
        raise ValueError(f"unexpected metric depth semantics: {depth_semantics!r}")
    expected_extrinsics = "T_target_source; p_world=T_world_camera@p_camera"
    if calibration.get("extrinsics_convention") != expected_extrinsics:
        raise ValueError("calibration extrinsics convention is not T_world_camera")

    intrinsics = np.asarray(calibration["intrinsics_K"], dtype=np.float64)
    t_world_camera = np.asarray(calibration["T_world_camera"], dtype=np.float64)
    depth_validation = validate_depth_map(
        depth_m,
        expected_shape=(height, width),
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    calibration_validation = validate_camera_calibration(
        intrinsics,
        t_world_camera,
        image_width=width,
        image_height=height,
    )
    if not depth_validation.valid:
        raise ValueError(f"metric depth validation failed: {depth_validation.errors}")
    if not calibration_validation.valid:
        raise ValueError(f"camera calibration validation failed: {calibration_validation.errors}")

    actual_rgb_hash = sha256_file(rgb_path)
    expected_rgb_hash = camera_record.get("rgb", {}).get("sha256")
    if expected_rgb_hash and actual_rgb_hash != expected_rgb_hash:
        raise ValueError(
            f"RGB sha256 mismatch: got {actual_rgb_hash}, expected {expected_rgb_hash}"
        )
    eef_position = np.asarray(robot_state.get("robot0_eef_pos_m"), dtype=np.float64)
    if eef_position.shape != (3,) or not np.isfinite(eef_position).all():
        raise ValueError(f"invalid world-frame end-effector position: {eef_position}")

    return LiberoPlannerInput(
        probe_dir=root,
        manifest_path=manifest_path,
        manifest=manifest,
        camera_name=camera_name,
        instruction=instruction,
        rgb_path=rgb_path,
        depth_path=depth_path,
        calibration_path=calibration_path,
        robot_state_path=robot_state_path,
        rgb=rgb,
        depth_m=np.asarray(depth_m, dtype=np.float32),
        intrinsics=intrinsics,
        t_world_camera=t_world_camera,
        eef_position_world_m=eef_position,
        depth_validation=depth_validation,
        calibration_validation=calibration_validation,
        hashes={
            "observation_manifest_sha256": sha256_file(manifest_path),
            "rgb_sha256": actual_rgb_hash,
            "depth_m_sha256": sha256_file(depth_path),
            "calibration_sha256": sha256_file(calibration_path),
            "robot_state_sha256": sha256_file(robot_state_path),
        },
    )


def _finite_range(values: np.ndarray) -> list[float] | None:
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return None
    return [float(finite.min()), float(finite.max())]


def build_trajectory_geometry(
    waypoints_uvd: np.ndarray,
    planner_input: LiberoPlannerInput,
    *,
    max_roundtrip_error: float = 1e-6,
) -> TrajectoryGeometry:
    """Unproject UVD, transform it to world, and run geometry-only checks."""

    waypoints = np.asarray(waypoints_uvd, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) == 0:
        raise ValueError("waypoints_uvd must be a non-empty array with shape (N, 3)")
    width, height = planner_input.rgb.size
    camera_xyz = normalized_uvd_to_camera_xyz(
        waypoints,
        planner_input.intrinsics,
        image_width=width,
        image_height=height,
    )
    world_xyz = transform_points(camera_xyz, planner_input.t_world_camera)
    roundtrip = camera_xyz_to_normalized_uvd(
        camera_xyz,
        planner_input.intrinsics,
        image_width=width,
        image_height=height,
    )
    roundtrip_abs_error = np.abs(roundtrip - waypoints)

    u_px = waypoints[:, 0] / 1000.0 * width
    v_px = waypoints[:, 1] / 1000.0 * height
    u_index = np.clip(np.rint(u_px).astype(np.int64), 0, width - 1)
    v_index = np.clip(np.rint(v_px).astype(np.int64), 0, height - 1)
    scene_depth = planner_input.depth_m[v_index, u_index].astype(np.float64)
    depth_delta = waypoints[:, 2] - scene_depth

    camera_segments = np.linalg.norm(np.diff(camera_xyz, axis=0), axis=1)
    world_segments = np.linalg.norm(np.diff(world_xyz, axis=0), axis=1)
    rigid_segment_error = (
        float(np.max(np.abs(camera_segments - world_segments)))
        if camera_segments.size
        else 0.0
    )
    max_error = float(roundtrip_abs_error.max())
    errors: list[str] = []
    warnings: list[str] = []
    if not np.isfinite(camera_xyz).all() or not np.isfinite(world_xyz).all():
        errors.append("camera/world trajectory contains NaN or infinity")
    if max_error > max_roundtrip_error:
        errors.append(
            f"UVD projection roundtrip error {max_error:.3e} exceeds {max_roundtrip_error:.3e}"
        )
    if rigid_segment_error > max_roundtrip_error:
        errors.append(
            f"rigid transform changed segment length by {rigid_segment_error:.3e} m"
        )
    if np.any(np.abs(depth_delta) > 0.25):
        warnings.append(
            "planner depth differs from the observed surface depth by more than 0.25 m at "
            "one or more pixels; this is diagnostic only because 3D waypoints may lie in free space"
        )

    eef_distances = np.linalg.norm(
        world_xyz - planner_input.eef_position_world_m[None, :], axis=1
    )
    report = ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        metrics={
            "waypoint_count": int(len(waypoints)),
            "projection_roundtrip_max_abs_error": max_error,
            "rigid_segment_length_max_abs_error_m": rigid_segment_error,
            "camera_xyz_range_m": {
                "x": _finite_range(camera_xyz[:, 0]),
                "y": _finite_range(camera_xyz[:, 1]),
                "z": _finite_range(camera_xyz[:, 2]),
            },
            "world_xyz_range_m": {
                "x": _finite_range(world_xyz[:, 0]),
                "y": _finite_range(world_xyz[:, 1]),
                "z": _finite_range(world_xyz[:, 2]),
            },
            "segment_length_range_m": _finite_range(world_segments),
            "trajectory_length_m": float(world_segments.sum()),
            "distance_to_probe_eef_range_m": _finite_range(eef_distances),
            "scene_depth_at_uv_range_m": _finite_range(scene_depth),
            "planner_minus_scene_depth_range_m": _finite_range(depth_delta),
            "scene_depth_sampling": "nearest_original_pixel_clamped_at_image_boundary",
            "scene_depth_is_diagnostic_only": True,
        },
    )
    return TrajectoryGeometry(
        camera_xyz_m=camera_xyz,
        world_xyz_m=world_xyz,
        scene_depth_at_uv_m=scene_depth,
        planner_minus_scene_depth_m=depth_delta,
        report=report,
    )
