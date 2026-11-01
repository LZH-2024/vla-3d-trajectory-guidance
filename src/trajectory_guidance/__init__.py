"""Utilities for reproducible 3D trajectory-planner experiments."""

from trajectory_guidance.geometry import normalized_uvd_to_camera_xyz
from trajectory_guidance.validation import validate_depth_map, validate_waypoints

__all__ = [
    "normalized_uvd_to_camera_xyz",
    "validate_depth_map",
    "validate_waypoints",
]

__version__ = "0.1.0"
