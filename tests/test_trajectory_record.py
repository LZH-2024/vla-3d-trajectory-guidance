import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from trajectory_guidance.trajectory_record import (
    CameraCalibration,
    EpisodeIdentity,
    ModelIdentity,
    TrajectoryRecord,
    load_trajectory_record,
    save_trajectory_record,
)


def make_record(
    *, source: str = "prediction", uses_future: bool = False
) -> TrajectoryRecord:
    return TrajectoryRecord(
        trajectory_id=f"record-{source}",
        source=source,
        episode=EpisodeIdentity(
            dataset="dataset",
            suite="suite",
            task_id="task-0",
            episode_id="episode-0",
            init_state_id=0,
            observation_index=3,
        ),
        model=ModelIdentity("planner", "v1", "abc123"),
        instruction="pick up the object",
        input_image_sha256="a" * 64,
        coordinate_frame="world",
        waypoints_xyz_m=np.array([[0.0, 0.1, 0.2], [0.1, 0.1, 0.2]]),
        valid_mask=np.array([True, True]),
        gripper_events=("none", "close"),
        calibration=CameraCalibration.from_arrays(
            camera_name="agentview",
            image_width=64,
            image_height=64,
            intrinsics_k=np.array([[80, 0, 32], [0, 80, 32], [0, 0, 1]]),
            t_world_camera=np.eye(4),
        ),
        available_at_observation_index=4,
        uses_future_expert_information=uses_future,
    )


class TrajectoryRecordTest(unittest.TestCase):
    def test_save_load_and_exchange_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "record"
            save_trajectory_record(make_record(), destination)
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                {"metadata.json", "trajectory.npz", "waypoints.csv", "manifest.json"},
            )
            loaded = load_trajectory_record(destination)
            np.testing.assert_allclose(loaded.waypoints_xyz_m, make_record().waypoints_xyz_m)
            self.assertEqual(loaded.gripper_events, ("none", "close"))
            metadata = json.loads((destination / "metadata.json").read_text("utf-8"))
            self.assertFalse(metadata["uses_future_expert_information"])
            self.assertFalse(
                metadata["causality_contract"][
                    "expert_future_trajectory_allowed_as_normal_input"
                ]
            )

    def test_checksum_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "record"
            save_trajectory_record(make_record(), destination)
            with (destination / "waypoints.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_trajectory_record(destination)

    def test_oracle_future_data_is_diagnostic_only(self):
        record = make_record(source="oracle", uses_future=True)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "oracle"
            save_trajectory_record(record, destination)
            with self.assertRaisesRegex(ValueError, "oracle trajectory rejected"):
                load_trajectory_record(destination)
            with self.assertRaisesRegex(ValueError, "future expert trajectory rejected"):
                load_trajectory_record(destination, allow_oracle=True)
            loaded = load_trajectory_record(
                destination,
                allow_oracle=True,
                allow_future_expert_information=True,
            )
            self.assertTrue(loaded.uses_future_expert_information)

    def test_future_expert_data_requires_oracle_source(self):
        report = make_record(uses_future=True).validate()
        self.assertFalse(report.valid)
        self.assertTrue(any("source='oracle'" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
