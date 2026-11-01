"""Utilities for reproducible 3D trajectory-planner experiments."""

from trajectory_guidance.geometry import (
    camera_xyz_to_normalized_uvd,
    normalized_uvd_to_camera_xyz,
)
from trajectory_guidance.validation import validate_depth_map, validate_waypoints

__all__ = [
    "camera_xyz_to_normalized_uvd",
    "normalized_uvd_to_camera_xyz",
    "validate_depth_map",
    "validate_waypoints",
]

__version__ = "0.1.0"
