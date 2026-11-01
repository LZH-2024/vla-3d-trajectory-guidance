"""Input and output validation for planner experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_depth_map(
    depth: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None = None,
    min_depth_m: float = 0.01,
    max_depth_m: float = 10.0,
) -> ValidationReport:
    """Validate an aligned metric-depth input without assuming every pixel is valid."""

    errors: list[str] = []
    warnings: list[str] = []
    array = np.asarray(depth)

    if array.ndim != 2:
        errors.append(f"depth map must be 2-D, got shape {array.shape}")
        return ValidationReport(False, errors, warnings, {"shape": list(array.shape)})

    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        errors.append(
            f"depth/RGB shape mismatch: depth={tuple(array.shape)}, rgb={tuple(expected_shape)}"
        )

    finite = np.isfinite(array)
    finite_fraction = float(finite.mean()) if array.size else 0.0
    if not finite.all():
        errors.append("depth map contains NaN or infinity")

    valid_mask = finite & (array > min_depth_m) & (array < max_depth_m)
    valid_fraction = float(valid_mask.mean()) if array.size else 0.0
    if valid_fraction == 0.0:
        errors.append("depth map has no valid metric depth in the accepted range")
    elif valid_fraction < 0.10:
        warnings.append("less than 10% of depth pixels are valid")

    valid_values = array[valid_mask]
    metrics: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_fraction": finite_fraction,
        "valid_fraction": valid_fraction,
        "accepted_depth_range_m": [min_depth_m, max_depth_m],
        "valid_min_m": float(valid_values.min()) if valid_values.size else None,
        "valid_max_m": float(valid_values.max()) if valid_values.size else None,
    }
    if array.dtype != np.float32:
        warnings.append(f"depth dtype is {array.dtype}; official API expects float32")

    return ValidationReport(not errors, errors, warnings, metrics)


def validate_waypoints(
    waypoints: Sequence[Sequence[float]] | np.ndarray,
    actions: Sequence[str | None] | None = None,
    *,
    min_depth_m: float = 0.01,
    max_depth_m: float = 10.0,
) -> ValidationReport:
    """Check non-empty [u, v, depth] output and official coordinate bounds."""

    errors: list[str] = []
    warnings: list[str] = []

    try:
        array = np.asarray(waypoints, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return ValidationReport(
            False,
            [f"waypoints cannot be converted to numbers: {exc}"],
            warnings,
            {"count": 0},
        )

    if array.size == 0:
        return ValidationReport(False, ["planner returned no waypoints"], warnings, {"count": 0})
    if array.ndim != 2 or array.shape[1] != 3:
        return ValidationReport(
            False,
            [f"waypoints must have shape (N, 3), got {array.shape}"],
            warnings,
            {"count": int(array.shape[0]) if array.ndim else 0},
        )

    finite_rows = np.isfinite(array).all(axis=1)
    if not finite_rows.all():
        bad_rows = np.flatnonzero(~finite_rows).tolist()
        errors.append(f"non-finite coordinates at waypoint indices {bad_rows}")

    uv_in_bounds = (
        (array[:, 0] >= 0.0)
        & (array[:, 0] <= 1000.0)
        & (array[:, 1] >= 0.0)
        & (array[:, 1] <= 1000.0)
    )
    if not uv_in_bounds.all():
        bad_rows = np.flatnonzero(~uv_in_bounds).tolist()
        errors.append(f"u/v outside normalized [0, 1000] at waypoint indices {bad_rows}")

    depth_in_bounds = (array[:, 2] > min_depth_m) & (array[:, 2] < max_depth_m)
    if not depth_in_bounds.all():
        bad_rows = np.flatnonzero(~depth_in_bounds).tolist()
        errors.append(
            f"depth outside ({min_depth_m}, {max_depth_m}) meters at waypoint indices {bad_rows}"
        )

    if actions is not None and len(actions) != len(array):
        errors.append(
            f"actions/waypoints length mismatch: actions={len(actions)}, waypoints={len(array)}"
        )

    def finite_range(values: np.ndarray) -> list[float] | None:
        finite_values = values[np.isfinite(values)]
        if not finite_values.size:
            return None
        return [float(finite_values.min()), float(finite_values.max())]

    metrics = {
        "count": int(len(array)),
        "u_range": finite_range(array[:, 0]),
        "v_range": finite_range(array[:, 1]),
        "depth_range_m": finite_range(array[:, 2]),
        "accepted_depth_range_m": [min_depth_m, max_depth_m],
    }
    return ValidationReport(not errors, errors, warnings, metrics)
