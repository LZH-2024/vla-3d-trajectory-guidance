"""Dependency-light helpers for a single-frame LIBERO RGB-D probe.

The simulator-facing entry point lives in ``scripts/``.  This module keeps
array validation, calibration checks, projection, and artifact helpers easy to
unit test without installing LIBERO or requiring a GPU.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from trajectory_guidance.validation import ValidationReport


def squeeze_depth_map(depth: np.ndarray) -> np.ndarray:
    """Return an HxW depth map from LIBERO's HxWx1 or HxW representation."""

    array = np.asarray(depth)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"depth must have shape (H, W) or (H, W, 1), got {array.shape}")
    return array


def validate_depth_buffer(depth: np.ndarray) -> ValidationReport:
    """Validate the normalized, non-linear MuJoCo depth buffer before conversion."""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        array = squeeze_depth_map(depth)
    except ValueError as exc:
        return ValidationReport(False, [str(exc)], warnings, {"shape": list(np.shape(depth))})

    finite = np.isfinite(array)
    finite_fraction = float(finite.mean()) if array.size else 0.0
    if not array.size:
        errors.append("depth buffer is empty")
    if not finite.all():
        errors.append("depth buffer contains NaN or infinity")

    finite_values = array[finite]
    minimum = float(finite_values.min()) if finite_values.size else None
    maximum = float(finite_values.max()) if finite_values.size else None
    if finite_values.size and (minimum < 0.0 or maximum > 1.0):
        errors.append(f"MuJoCo depth buffer must be in [0, 1], got [{minimum}, {maximum}]")
    if finite_values.size and np.allclose(finite_values, finite_values[0]):
        warnings.append("depth buffer is constant; the rendered scene may be invalid")

    return ValidationReport(
        not errors,
        errors,
        warnings,
        {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite_fraction": finite_fraction,
            "normalized_min": minimum,
            "normalized_max": maximum,
        },
    )


def validate_camera_calibration(
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> ValidationReport:
    """Check an OpenCV-style K and a rigid ``T_world_camera`` transform."""

    errors: list[str] = []
    warnings: list[str] = []
    k = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)

    if image_width <= 0 or image_height <= 0:
        errors.append("image dimensions must be positive")
    if k.shape != (3, 3):
        errors.append(f"intrinsics must have shape (3, 3), got {k.shape}")
    if transform.shape != (4, 4):
        errors.append(f"camera_to_world must have shape (4, 4), got {transform.shape}")
    if errors:
        return ValidationReport(False, errors, warnings, {})

    if not np.isfinite(k).all():
        errors.append("intrinsics contain NaN or infinity")
    if not np.isfinite(transform).all():
        errors.append("camera_to_world contains NaN or infinity")

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        errors.append("intrinsics must have positive fx and fy")
    if not (0.0 <= cx <= image_width and 0.0 <= cy <= image_height):
        warnings.append("principal point lies outside the image bounds")
    if not np.allclose(k[2], [0.0, 0.0, 1.0], atol=1e-8):
        errors.append("intrinsics last row must be [0, 0, 1]")

    rotation = transform[:3, :3]
    rotation_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    rotation_det = float(np.linalg.det(rotation))
    if rotation_error > 1e-5:
        errors.append(f"camera rotation is not orthonormal (error={rotation_error:.3e})")
    if not np.isclose(rotation_det, 1.0, atol=1e-5):
        errors.append(f"camera rotation determinant must be +1, got {rotation_det:.8f}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        errors.append("camera_to_world last row must be [0, 0, 0, 1]")

    return ValidationReport(
        not errors,
        errors,
        warnings,
        {
            "image_width": image_width,
            "image_height": image_height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "rotation_orthonormal_error": rotation_error,
            "rotation_determinant": rotation_det,
        },
    )


def unproject_depth_sample(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    stride: int = 8,
    min_depth_m: float = 0.01,
    max_depth_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a strided metric z-depth map into OpenCV camera XYZ.

    Returns ``(pixels_uv, camera_xyz_m)``.  Pixel coordinates are ordered
    ``[u, v]``; camera axes are x right, y down, z forward.
    """

    depth = np.asarray(squeeze_depth_map(depth_m), dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if stride <= 0:
        raise ValueError("stride must be positive")

    v_grid, u_grid = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[v_grid, u_grid]
    valid = np.isfinite(z) & (z > min_depth_m) & (z < max_depth_m)
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    z = z[valid]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("intrinsics must have positive fx and fy")
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack((u, v)), np.column_stack((x, y, z))


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an Nx3 point array."""

    points = np.asarray(points_xyz, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (matrix @ homogeneous.T).T[:, :3]


def save_depth_preview(depth_m: np.ndarray, output_path: str | Path) -> Path:
    """Save a robust grayscale preview; numeric depth remains in the NPY artifact."""

    depth = np.asarray(squeeze_depth_map(depth_m), dtype=np.float64)
    finite_positive = np.isfinite(depth) & (depth > 0.0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if finite_positive.any():
        values = depth[finite_positive]
        low, high = np.percentile(values, [2.0, 98.0])
        if high <= low:
            high = low + 1e-6
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        preview[finite_positive] = np.round((1.0 - normalized[finite_positive]) * 255.0).astype(
            np.uint8
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="L").save(destination)
    return destination


def prepare_libero_config(config_dir: str | Path, datasets_dir: str | Path) -> Path:
    """Create LIBERO's non-interactive path config before importing ``libero.libero``."""

    distribution = importlib.metadata.distribution("hf-libero")
    benchmark_root = Path(distribution.locate_file("libero/libero")).resolve()
    destination_dir = Path(config_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(datasets_dir).expanduser().resolve()
    dataset_path.mkdir(parents=True, exist_ok=True)
    config_path = destination_dir / "config.yaml"
    entries = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": dataset_path,
        "assets": benchmark_root / "assets",
    }
    config_path.write_text(
        "".join(f"{key}: '{value.as_posix()}'\n" for key, value in entries.items()),
        encoding="utf-8",
    )
    return config_path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    """Convert NumPy-heavy metadata to strict JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return destination
