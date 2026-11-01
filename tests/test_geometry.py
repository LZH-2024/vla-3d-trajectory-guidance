from pathlib import Path
import sys
import numpy as np
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_guidance.geometry import (
    camera_xyz_to_normalized_uvd,
    normalized_uvd_to_camera_xyz,
)


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

    def test_uvd_camera_roundtrip_preserves_official_normalization(self) -> None:
        intrinsics = np.array(
            [
                [309.0, 0.0, 128.0],
                [0.0, 309.0, 128.0],
                [0.0, 0.0, 1.0],
            ]
        )
        waypoints = np.array([[125.0, 750.0, 0.8], [900.0, 100.0, 1.2]])

        xyz = normalized_uvd_to_camera_xyz(waypoints, intrinsics, 256, 256)
        recovered = camera_xyz_to_normalized_uvd(xyz, intrinsics, 256, 256)

        np.testing.assert_allclose(recovered, waypoints, rtol=0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
