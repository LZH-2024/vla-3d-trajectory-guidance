import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from trajectory_guidance.libero_planner import load_libero_planner_input
from trajectory_guidance.mock_planner import create_mock_probe, run_mock_suite
from trajectory_guidance.trajectory_record import load_trajectory_record


class MockPlannerTest(unittest.TestCase):
    def test_full_mock_suite_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "suite"
            summary = run_mock_suite(output)
            self.assertTrue(summary["all_cases_passed"])
            self.assertEqual(
                {case["case"] for case in summary["cases"]},
                {"valid", "empty", "out_of_bounds", "invalid_depth"},
            )
            valid_dir = output / "cases" / "valid"
            for name in (
                "mock_prediction.json",
                "mock_arrays.npz",
                "waypoints_uvd.csv",
                "mock_trajectory_overlay.png",
            ):
                self.assertTrue((valid_dir / name).is_file(), name)
            prediction = json.loads(
                (valid_dir / "mock_prediction.json").read_text(encoding="utf-8")
            )
            self.assertTrue(prediction["mock"])
            self.assertIn("world_xyz_m", prediction["raw_mock_output"])
            with np.load(valid_dir / "mock_arrays.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["world_xyz_m"].shape, (3, 3))
            record = load_trajectory_record(
                valid_dir / "trajectory_cache" / "mock-valid",
                allowed_sources={"mock"},
            )
            self.assertEqual(record.source, "mock")

    def test_invalid_input_depth_probe_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            probe = create_mock_probe(Path(temporary) / "bad", invalid_input_depth=True)
            with self.assertRaisesRegex(ValueError, "metric depth validation failed"):
                load_libero_planner_input(probe)


if __name__ == "__main__":
    unittest.main()
