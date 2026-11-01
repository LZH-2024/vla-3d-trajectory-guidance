from pathlib import Path
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_guidance.libero_probe import (
    json_ready,
    squeeze_depth_map,
    transform_points,
    unproject_depth_sample,
    validate_camera_calibration,
    validate_depth_buffer,
)


class LiberoProbeTests(unittest.TestCase):
    def test_json_ready_converts_nested_numpy_values(self) -> None:
        payload = {
            "action": np.array([0.0, -1.0], dtype=np.float64),
            "count": np.int64(2),
            "nested": [np.float32(0.5)],
        }

        converted = json_ready(payload)

        self.assertEqual(converted["action"], [0.0, -1.0])
        self.assertEqual(converted["count"], 2)
        json.dumps(converted, allow_nan=False)

    def test_depth_buffer_accepts_libero_singleton_channel(self) -> None:
        depth = np.linspace(0.1, 0.9, 16, dtype=np.float32).reshape(4, 4, 1)

        report = validate_depth_buffer(depth)

        self.assertTrue(report.valid)
        self.assertEqual(report.metrics["shape"], [4, 4])

    def test_depth_buffer_rejects_values_outside_mujoco_range(self) -> None:
        report = validate_depth_buffer(np.array([[0.5, 1.2]], dtype=np.float32))

        self.assertFalse(report.valid)
        self.assertTrue(any("[0, 1]" in error for error in report.errors))

    def test_squeeze_depth_rejects_multichannel_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            squeeze_depth_map(np.zeros((4, 4, 3), dtype=np.float32))

    def test_camera_calibration_and_unprojection(self) -> None:
        intrinsics = np.array(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
        )
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = [1.0, 2.0, 3.0]

        report = validate_camera_calibration(
            intrinsics,
            camera_to_world,
            image_width=4,
            image_height=4,
        )
        pixels, camera_points = unproject_depth_sample(
            np.full((4, 4), 2.0, dtype=np.float32), intrinsics, stride=2
        )
        world_points = transform_points(camera_points, camera_to_world)

        self.assertTrue(report.valid)
        center_index = np.flatnonzero((pixels == [2.0, 2.0]).all(axis=1))[0]
        np.testing.assert_allclose(camera_points[center_index], [0.0, 0.0, 2.0])
        np.testing.assert_allclose(world_points[center_index], [1.0, 2.0, 5.0])

    def test_improper_rotation_is_rejected(self) -> None:
        transform = np.eye(4)
        transform[0, 0] = -1.0
        report = validate_camera_calibration(
            np.eye(3), transform, image_width=10, image_height=10
        )

        self.assertFalse(report.valid)
        self.assertTrue(any("determinant" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
