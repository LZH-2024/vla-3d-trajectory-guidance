"""Progress-aware local waypoint selection independent of any robot simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalWaypointWindow:
    waypoints_xyz_m: np.ndarray
    valid_mask: np.ndarray
    source_indices: np.ndarray
    start_index: int | None


def select_local_waypoints(
    waypoints_xyz_m: np.ndarray,
    valid_mask: np.ndarray,
    current_eef_xyz_m: np.ndarray,
    *,
    count: int,
    min_progress_index: int = 0,
) -> LocalWaypointWindow:
    """Select the closest causally current waypoint and its ordered successors.

    The closest point is searched only at or after ``min_progress_index``.  This
    allows a rollout to carry a monotonic progress index and prevents the local
    window from jumping backwards when a path crosses itself.
    """

    points = np.asarray(waypoints_xyz_m, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    eef = np.asarray(current_eef_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("waypoints_xyz_m must have shape (N, 3)")
    if mask.shape != (len(points),):
        raise ValueError("valid_mask shape does not match waypoints")
    if eef.shape != (3,) or not np.isfinite(eef).all():
        raise ValueError("current_eef_xyz_m must be a finite 3-vector")
    if count <= 0:
        raise ValueError("count must be positive")
    if min_progress_index < 0:
        raise ValueError("min_progress_index must be non-negative")

    selected_points = np.zeros((count, 3), dtype=np.float64)
    selected_mask = np.zeros(count, dtype=np.bool_)
    selected_indices = np.full(count, -1, dtype=np.int64)
    eligible = np.flatnonzero(mask & (np.arange(len(points)) >= min_progress_index))
    eligible = eligible[np.isfinite(points[eligible]).all(axis=1)]
    if not len(eligible):
        return LocalWaypointWindow(selected_points, selected_mask, selected_indices, None)

    distances = np.linalg.norm(points[eligible] - eef[None, :], axis=1)
    start_index = int(eligible[int(np.argmin(distances))])
    successors = eligible[eligible >= start_index][:count]
    selected_points[: len(successors)] = points[successors]
    selected_mask[: len(successors)] = True
    selected_indices[: len(successors)] = successors
    return LocalWaypointWindow(
        waypoints_xyz_m=selected_points,
        valid_mask=selected_mask,
        source_indices=selected_indices,
        start_index=start_index,
    )


def normalize_waypoints_relative_to_eef(
    waypoints_xyz_m: np.ndarray,
    valid_mask: np.ndarray,
    current_eef_xyz_m: np.ndarray,
    scale_m: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Return masked EEF-relative coordinates scaled per axis."""

    points = np.asarray(waypoints_xyz_m, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    eef = np.asarray(current_eef_xyz_m, dtype=np.float64)
    scale = np.asarray(scale_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("waypoints_xyz_m must have shape (N, 3)")
    if mask.shape != (len(points),):
        raise ValueError("valid_mask shape does not match waypoints")
    if eef.shape != (3,) or scale.shape != (3,):
        raise ValueError("current EEF and scale must be 3-vectors")
    if not np.isfinite(points).all() or not np.isfinite(eef).all():
        raise ValueError("waypoints and current EEF must be finite")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("scale must be finite and positive")
    normalized = (points - eef[None, :]) / scale[None, :]
    normalized[~mask] = 0.0
    return normalized
