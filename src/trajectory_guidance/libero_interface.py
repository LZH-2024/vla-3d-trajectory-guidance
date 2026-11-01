"""Design-only contracts for the later LIBERO adapter.

This module intentionally imports no LIBERO or SmolVLA code. It makes units,
frames, and calibration metadata explicit before any simulator integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    intrinsics: np.ndarray
    extrinsics: np.ndarray | None
    extrinsics_convention: str | None
    image_frame: str
    target_frame: str | None = None


@dataclass(frozen=True)
class RGBDObservation:
    rgb: np.ndarray
    depth: np.ndarray
    calibration: CameraCalibration
    depth_unit: str
    depth_scale_to_meters: float | None
    end_effector_position: np.ndarray | None
    end_effector_frame: str | None
    episode_id: str
    step_index: int

    def metric_depth(self) -> np.ndarray:
        """Return meter-valued depth only when the upstream scale is known."""

        if self.depth_scale_to_meters is None:
            raise ValueError(
                "depth_scale_to_meters is unknown; inspect LIBERO's depth encoding before planning"
            )
        return np.asarray(self.depth, dtype=np.float32) * float(self.depth_scale_to_meters)


@dataclass(frozen=True)
class PlannerTrajectory:
    waypoints_uvd: np.ndarray
    gripper_actions: tuple[str | None, ...]
    coordinate_convention: Literal["normalized_uv_0_1000_depth_m"]
    camera_xyz_m: np.ndarray | None = None
    world_or_base_xyz_m: np.ndarray | None = None
    target_frame: str | None = None


class HighLevelPlanner(Protocol):
    def predict(self, observation: RGBDObservation, instruction: str) -> PlannerTrajectory:
        """Predict a high-level trajectory without invoking a low-level policy."""


class TrajectoryCache(Protocol):
    def put(
        self,
        observation: RGBDObservation,
        instruction: str,
        trajectory: PlannerTrajectory,
        metadata: dict[str, object],
    ) -> str:
        """Persist one trajectory and return a stable cache key."""
