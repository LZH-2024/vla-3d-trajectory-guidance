import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from trajectory_guidance.trajectory_conditioning import (
    TRAJECTORY_MASK_KEY,
    TRAJECTORY_POINTS_KEY,
)
from trajectory_guidance.trajectory_dataset import (
    DatasetTrajectoryAdapterConfig,
    TrajectoryConditionDataset,
    TrajectoryConditionProvider,
    validate_action_time_alignment,
)
from trajectory_guidance.trajectory_record import (
    CameraCalibration,
    EpisodeIdentity,
    ModelIdentity,
    TrajectoryRecord,
)


def make_record(
    *, observation_index: int = 2, available_at: int = 3, source: str = "prediction"
) -> TrajectoryRecord:
    return TrajectoryRecord(
        trajectory_id=f"trajectory-{observation_index}-{available_at}-{source}",
        source=source,
        episode=EpisodeIdentity(
            "dataset", "suite", "task", "ep", 0, observation_index
        ),
        model=ModelIdentity("planner", "1", "commit"),
        instruction="move",
        input_image_sha256="b" * 64,
        coordinate_frame="world",
        waypoints_xyz_m=np.array(
            [[0.0, 0.0, 0.4], [0.1, 0.0, 0.4], [0.2, 0.0, 0.4]]
        ),
        valid_mask=np.ones(3, dtype=bool),
        gripper_events=("none", "none", "close"),
        calibration=CameraCalibration.from_arrays(
            camera_name="agentview",
            image_width=64,
            image_height=64,
            intrinsics_k=np.eye(3),
            t_world_camera=np.eye(4),
        ),
        available_at_observation_index=available_at,
    )


class TrajectoryDatasetTest(unittest.TestCase):
    def test_action_alignment_accepts_only_current_contiguous_chunk(self):
        validate_action_time_alignment(4, [4, 5, 6], action_count=3)
        with self.assertRaisesRegex(ValueError, "starts at"):
            validate_action_time_alignment(4, [5, 6])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_action_time_alignment(4, [4, 6])
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_action_time_alignment(4, [4, 5], action_count=3)

    def test_provider_hides_not_yet_available_trajectory(self):
        provider = TrajectoryConditionProvider([make_record()], num_waypoints=4)
        early = provider.condition_for_observation(
            episode_id="ep", observation_index=2, current_eef_xyz_m=np.zeros(3)
        )
        self.assertFalse(early[TRAJECTORY_MASK_KEY].any())
        current = provider.condition_for_observation(
            episode_id="ep", observation_index=3, current_eef_xyz_m=np.zeros(3)
        )
        self.assertTrue(current[TRAJECTORY_MASK_KEY].any())

    def test_oracle_rejected_by_normal_provider(self):
        with self.assertRaisesRegex(ValueError, "oracle"):
            TrajectoryConditionProvider(
                [make_record(source="oracle")],
                num_waypoints=4,
                allowed_sources={"oracle"},
            )

    def test_dataset_adds_condition_and_checks_action_indices(self):
        base = [
            {
                "episode_id": "ep",
                "frame_index": 3,
                "observation.eef_position": np.array([0.05, 0.0, 0.4]),
                "action": np.zeros((2, 7), dtype=np.float32),
                "action_indices": np.array([3, 4]),
            }
        ]
        provider = TrajectoryConditionProvider([make_record()], num_waypoints=2)
        dataset = TrajectoryConditionDataset(
            base,
            provider,
            DatasetTrajectoryAdapterConfig(num_waypoints=2),
        )
        sample = dataset[0]
        self.assertEqual(sample[TRAJECTORY_POINTS_KEY].shape, (2, 3))
        np.testing.assert_array_equal(sample[TRAJECTORY_MASK_KEY], [True, True])


if __name__ == "__main__":
    unittest.main()
