"""Coordinate conversions with explicit frame and unit assumptions."""

from __future__ import annotations

import numpy as np


def normalized_uvd_to_camera_xyz(
    waypoints_uvd: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Unproject 3D HAMSTER [u, v, depth] waypoints into the camera frame.

    The planner expresses u and v in the normalized 0..1000 image coordinate
    convention and depth in meters. Intrinsics must match the image dimensions
    supplied here. No extrinsic transform is applied.
    """

    points = np.asarray(waypoints_uvd, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("waypoints_uvd must have shape (N, 3)")
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    if not np.isfinite(camera_matrix).all() or fx <= 0 or fy <= 0:
        raise ValueError("intrinsics must be finite with positive fx and fy")

    u_px = points[:, 0] / 1000.0 * image_width
    v_px = points[:, 1] / 1000.0 * image_height
    depth_m = points[:, 2]

    x_m = (u_px - cx) * depth_m / fx
    y_m = (v_px - cy) * depth_m / fy
    return np.column_stack((x_m, y_m, depth_m))


def camera_xyz_to_normalized_uvd(
    points_camera_xyz: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Project OpenCV camera XYZ to 3D HAMSTER normalized UVD.

    This is the inverse of :func:`normalized_uvd_to_camera_xyz`.  It is used
    as a numerical consistency check after LIBERO waypoints are unprojected;
    it does not replace the depth predicted by the planner with scene depth.
    """

    points = np.asarray(points_camera_xyz, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera_xyz must have shape (N, 3)")
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not np.isfinite(points).all():
        raise ValueError("points_camera_xyz must be finite")
    if not np.isfinite(camera_matrix).all():
        raise ValueError("intrinsics must be finite")

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError("intrinsics must have positive fx and fy")

    depth_m = points[:, 2]
    if np.any(depth_m <= 0.0):
        raise ValueError("camera z-depth must be positive")
    u_px = fx * points[:, 0] / depth_m + cx
    v_px = fy * points[:, 1] / depth_m + cy
    u_normalized = u_px / image_width * 1000.0
    v_normalized = v_px / image_height * 1000.0
    return np.column_stack((u_normalized, v_normalized, depth_m))
