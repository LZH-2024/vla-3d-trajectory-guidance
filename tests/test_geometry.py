from pathlib import Path
import sys
import numpy as np
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_guidance.geometry import normalized_uvd_to_camera_xyz


class GeometryTests(unittest.TestCase):
    def test_normalized_center_unprojects_to_optical_axis(self) -> None:
        intrinsics = np.array(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ]
        )
        waypoint = np.array([[500.0, 500.0, 2.0]])

        xyz = normalized_uvd_to_camera_xyz(waypoint, intrinsics, 100, 100)

        np.testing.assert_allclose(xyz, [[0.0, 0.0, 2.0]])

    def test_invalid_intrinsics_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive fx and fy"):
            normalized_uvd_to_camera_xyz(
                np.array([[500.0, 500.0, 1.0]]),
                np.zeros((3, 3)),
                100,
                100,
            )


if __name__ == "__main__":
    unittest.main()
