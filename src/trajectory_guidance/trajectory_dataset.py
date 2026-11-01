"""Causal trajectory-to-action alignment and a generic dataset wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from trajectory_guidance.trajectory_conditioning import (
    TRAJECTORY_EEF_KEY,
    TRAJECTORY_MASK_KEY,
    TRAJECTORY_POINTS_KEY,
    TRAJECTORY_SOURCE_CODE_KEY,
)
from trajectory_guidance.trajectory_record import TrajectoryRecord
from trajectory_guidance.waypoint_selection import select_local_waypoints


SOURCE_CODES = {"missing": 0, "prediction": 1, "mock": 2, "oracle": 3, "other": 4}


def validate_action_time_alignment(
    observation_index: int,
    action_indices: Sequence[int] | np.ndarray,
    *,
    action_count: int | None = None,
) -> None:
    """Ensure an action chunk starts at the condition's observation time."""

    indices = np.asarray(action_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("action_indices must be a non-empty 1-D sequence")
    if int(indices[0]) != int(observation_index):
        raise ValueError(
            f"action chunk starts at {indices[0]}, expected observation index {observation_index}"
        )
    if len(indices) > 1 and not np.array_equal(np.diff(indices), np.ones(len(indices) - 1)):
        raise ValueError("action_indices must be contiguous and strictly increasing")
    if action_count is not None and len(indices) != action_count:
        raise ValueError(
            f"action index count {len(indices)} does not match action labels {action_count}"
        )


@dataclass(frozen=True)
class DatasetTrajectoryAdapterConfig:
    num_waypoints: int = 8
    episode_key: str = "episode_id"
    observation_index_key: str = "frame_index"
    current_eef_key: str = "observation.eef_position"
    action_key: str = "action"
    action_indices_key: str = "action_indices"
    progress_index_key: str | None = "trajectory_progress_index"
    require_action_indices: bool = True
    missing_trajectory_fallback: bool = True


class TrajectoryConditionProvider:
    """Select the newest causally available trajectory for each observation."""

    def __init__(
        self,
        records: Sequence[TrajectoryRecord],
        *,
        num_waypoints: int,
        allowed_sources: set[str] | None = None,
        allow_oracle: bool = False,
    ):
        if num_waypoints <= 0:
            raise ValueError("num_waypoints must be positive")
        self.num_waypoints = num_waypoints
        self.allowed_sources = (
            {"prediction"} if allowed_sources is None else set(allowed_sources)
        )
        self.allow_oracle = allow_oracle
        self.records = list(records)
        for record in self.records:
            report = record.validate()
            if not report.valid:
                raise ValueError(f"invalid trajectory record {record.trajectory_id}: {report.errors}")
            if record.source == "oracle" and not allow_oracle:
                raise ValueError("oracle record cannot enter the normal trajectory provider")
            if record.uses_future_expert_information and not allow_oracle:
                raise ValueError(
                    "future expert trajectory cannot enter the normal trajectory provider"
                )

    def _record_for_observation(
        self, episode_id: str, observation_index: int
    ) -> TrajectoryRecord | None:
        candidates = [
            record
            for record in self.records
            if record.episode.episode_id == episode_id
            and record.source in self.allowed_sources
            and record.episode.observation_index <= observation_index
            and record.available_at_observation_index <= observation_index
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda record: (
                record.episode.observation_index,
                record.available_at_observation_index,
                record.created_at_utc,
            ),
        )

    def condition_for_observation(
        self,
        *,
        episode_id: str,
        observation_index: int,
        current_eef_xyz_m: np.ndarray,
        min_progress_index: int = 0,
        missing_fallback: bool = True,
    ) -> dict[str, Any]:
        eef = np.asarray(current_eef_xyz_m, dtype=np.float64)
        if eef.shape != (3,) or not np.isfinite(eef).all():
            raise ValueError("current EEF must be a finite 3-vector")
        record = self._record_for_observation(episode_id, observation_index)
        if record is None:
            if not missing_fallback:
                raise KeyError(
                    f"no causally available trajectory for episode={episode_id!r}, "
                    f"observation={observation_index}"
                )
            return {
                TRAJECTORY_POINTS_KEY: np.zeros(
                    (self.num_waypoints, 3), dtype=np.float32
                ),
                TRAJECTORY_MASK_KEY: np.zeros(self.num_waypoints, dtype=np.bool_),
                TRAJECTORY_EEF_KEY: eef.astype(np.float32),
                TRAJECTORY_SOURCE_CODE_KEY: SOURCE_CODES["missing"],
                "trajectory.source_indices": np.full(
                    self.num_waypoints, -1, dtype=np.int64
                ),
                "trajectory.start_index": -1,
                "trajectory.id": None,
            }
        if record.source == "oracle" and not self.allow_oracle:
            raise ValueError("oracle trajectory rejected by causal provider")
        window = select_local_waypoints(
            record.waypoints_xyz_m,
            record.valid_mask,
            eef,
            count=self.num_waypoints,
            min_progress_index=min_progress_index,
        )
        return {
            TRAJECTORY_POINTS_KEY: window.waypoints_xyz_m.astype(np.float32),
            TRAJECTORY_MASK_KEY: window.valid_mask,
            TRAJECTORY_EEF_KEY: eef.astype(np.float32),
            TRAJECTORY_SOURCE_CODE_KEY: SOURCE_CODES[record.source],
            "trajectory.source_indices": window.source_indices,
            "trajectory.start_index": (
                -1 if window.start_index is None else window.start_index
            ),
            "trajectory.id": record.trajectory_id,
        }


def _scalar(value: Any, *, name: str) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    return array.reshape(-1)[0].item()


class TrajectoryConditionDataset:
    """Wrap any map-style dataset and add fixed local trajectory tensors.

    The wrapper is intentionally key-configurable because LeRobot dataset
    feature names depend on the dataset metadata.  It does not infer which
    state dimensions correspond to the end effector.
    """

    def __init__(
        self,
        base_dataset: Any,
        provider: TrajectoryConditionProvider,
        config: DatasetTrajectoryAdapterConfig | None = None,
    ):
        self.base_dataset = base_dataset
        self.provider = provider
        self.config = config or DatasetTrajectoryAdapterConfig(
            num_waypoints=provider.num_waypoints
        )
        if self.config.num_waypoints != provider.num_waypoints:
            raise ValueError("dataset adapter and provider disagree on num_waypoints")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.base_dataset[index]
        if not isinstance(original, Mapping):
            raise TypeError("base dataset samples must be mappings")
        sample = dict(original)
        episode_id = str(_scalar(sample[self.config.episode_key], name="episode_id"))
        observation_index = int(
            _scalar(
                sample[self.config.observation_index_key],
                name="observation_index",
            )
        )
        eef_value = sample[self.config.current_eef_key]
        if hasattr(eef_value, "detach"):
            eef_numpy = eef_value.detach().cpu().numpy()
        else:
            eef_numpy = np.asarray(eef_value)
        eef_numpy = np.asarray(eef_numpy, dtype=np.float64).reshape(-1)
        if eef_numpy.shape != (3,):
            raise ValueError(
                f"current EEF feature must contain exactly 3 values, got {eef_numpy.shape}"
            )

        action_indices = sample.get(self.config.action_indices_key)
        if action_indices is None:
            if self.config.require_action_indices:
                raise KeyError(
                    f"required action time index key is missing: {self.config.action_indices_key}"
                )
        else:
            if hasattr(action_indices, "detach"):
                action_indices = action_indices.detach().cpu().numpy()
            action_value = sample.get(self.config.action_key)
            action_count = None
            if action_value is not None:
                action_count = int(np.shape(action_value)[0])
            validate_action_time_alignment(
                observation_index,
                action_indices,
                action_count=action_count,
            )

        min_progress_index = 0
        if self.config.progress_index_key and self.config.progress_index_key in sample:
            min_progress_index = int(
                _scalar(
                    sample[self.config.progress_index_key],
                    name="trajectory_progress_index",
                )
            )
        condition = self.provider.condition_for_observation(
            episode_id=episode_id,
            observation_index=observation_index,
            current_eef_xyz_m=eef_numpy,
            min_progress_index=min_progress_index,
            missing_fallback=self.config.missing_trajectory_fallback,
        )

        use_torch = hasattr(eef_value, "detach")
        if use_torch:
            import torch

            device = eef_value.device
            sample[TRAJECTORY_POINTS_KEY] = torch.as_tensor(
                condition[TRAJECTORY_POINTS_KEY], dtype=torch.float32, device=device
            )
            sample[TRAJECTORY_MASK_KEY] = torch.as_tensor(
                condition[TRAJECTORY_MASK_KEY], dtype=torch.bool, device=device
            )
            sample[TRAJECTORY_EEF_KEY] = torch.as_tensor(
                condition[TRAJECTORY_EEF_KEY], dtype=torch.float32, device=device
            )
            sample[TRAJECTORY_SOURCE_CODE_KEY] = torch.tensor(
                condition[TRAJECTORY_SOURCE_CODE_KEY], dtype=torch.int64, device=device
            )
        else:
            for key in (
                TRAJECTORY_POINTS_KEY,
                TRAJECTORY_MASK_KEY,
                TRAJECTORY_EEF_KEY,
                TRAJECTORY_SOURCE_CODE_KEY,
            ):
                sample[key] = condition[key]
        return sample
