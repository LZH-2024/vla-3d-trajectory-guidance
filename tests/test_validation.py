from pathlib import Path
import json
import sys
import numpy as np
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_guidance.validation import validate_depth_map, validate_waypoints


class ValidationTests(unittest.TestCase):
    def test_valid_waypoints_pass(self) -> None:
        report = validate_waypoints(
            [[100.0, 200.0, 0.5], [900.0, 800.0, 1.2]],
            [None, "Close Gripper"],
        )

        self.assertTrue(report.valid)
        self.assertEqual(report.metrics["count"], 2)

    def test_empty_waypoints_fail(self) -> None:
        report = validate_waypoints([], [])

        self.assertFalse(report.valid)
        self.assertIn("no waypoints", report.errors[0])

    def test_invalid_coordinate_and_depth_fail(self) -> None:
        report = validate_waypoints(
            [[-1.0, 100.0, 0.0], [500.0, np.inf, 1.0]],
            [None, None],
        )

        self.assertFalse(report.valid)
        self.assertTrue(any("non-finite" in error for error in report.errors))
        self.assertTrue(any("outside normalized" in error for error in report.errors))
        self.assertTrue(any("depth outside" in error for error in report.errors))
        json.dumps(report.to_dict(), allow_nan=False)

    def test_depth_map_allows_masked_zero_background(self) -> None:
        depth = np.zeros((4, 4), dtype=np.float32)
        depth[1:3, 1:3] = 0.5

        report = validate_depth_map(depth, expected_shape=(4, 4))

        self.assertTrue(report.valid)
        self.assertEqual(report.metrics["valid_fraction"], 0.25)

    def test_depth_shape_mismatch_fails(self) -> None:
        report = validate_depth_map(
            np.ones((3, 4), dtype=np.float32),
            expected_shape=(4, 4),
        )

        self.assertFalse(report.valid)
        self.assertTrue(any("shape mismatch" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
