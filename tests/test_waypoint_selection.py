import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from trajectory_guidance.waypoint_selection import (
    normalize_waypoints_relative_to_eef,
    select_local_waypoints,
)


class WaypointSelectionTest(unittest.TestCase):
    def setUp(self):
        self.points = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )
        self.mask = np.array([True, True, False, True])

    def test_nearest_then_ordered_successors_with_padding(self):
        window = select_local_waypoints(
            self.points, self.mask, np.array([0.09, 0.0, 0.0]), count=4
        )
        self.assertEqual(window.start_index, 1)
        np.testing.assert_array_equal(window.source_indices, [1, 3, -1, -1])
        np.testing.assert_array_equal(window.valid_mask, [True, True, False, False])
        np.testing.assert_allclose(window.waypoints_xyz_m[2:], 0.0)

    def test_progress_index_prevents_backward_selection(self):
        window = select_local_waypoints(
            self.points,
            np.ones(4, dtype=bool),
            np.array([0.0, 0.0, 0.0]),
            count=2,
            min_progress_index=2,
        )
        self.assertEqual(window.start_index, 2)
        np.testing.assert_array_equal(window.source_indices, [2, 3])

    def test_no_valid_points_returns_masked_window(self):
        window = select_local_waypoints(
            self.points, np.zeros(4, dtype=bool), np.zeros(3), count=3
        )
        self.assertIsNone(window.start_index)
        self.assertFalse(window.valid_mask.any())
        np.testing.assert_array_equal(window.source_indices, [-1, -1, -1])

    def test_relative_normalization_masks_padding(self):
        normalized = normalize_waypoints_relative_to_eef(
            np.array([[0.2, 0.0, 0.5], [9.0, 9.0, 9.0]]),
            np.array([True, False]),
            np.array([0.1, 0.0, 0.4]),
            (0.5, 0.5, 0.5),
        )
        np.testing.assert_allclose(normalized[0], [0.2, 0.0, 0.2])
        np.testing.assert_allclose(normalized[1], 0.0)


if __name__ == "__main__":
    unittest.main()
