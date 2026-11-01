from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_guidance.libero_planner import (
    build_trajectory_geometry,
    find_latest_valid_probe,
    load_libero_planner_input,
)
from trajectory_guidance.libero_probe import sha256_file


class LiberoPlannerTests(unittest.TestCase):
    def _make_probe(self, root: Path, name: str, created_at: str) -> Path:
        probe = root / name
        probe.mkdir(parents=True)
        rgb_path = probe / "agentview_rgb.png"
        depth_path = probe / "agentview_depth_m.npy"
        calibration_path = probe / "agentview_calibration.json"
        robot_state_path = probe / "robot_state.json"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(rgb_path)
        np.save(depth_path, np.full((4, 4), 2.0, dtype=np.float32), allow_pickle=False)
        calibration_path.write_text(
            json.dumps(
                {
                    "camera_name": "agentview",
                    "image_size": {"width": 4, "height": 4},
                    "image_convention": "opencv_top_left_u_right_v_down",
                    "depth_buffer": {"metric_depth_semantics": "camera_z_depth_m"},
                    "camera_frame": "opencv_camera_x_right_y_down_z_forward",
                    "world_frame": "libero_mujoco_world",
                    "intrinsics_K": [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                    "T_world_camera": [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "extrinsics_convention": (
                        "T_target_source; p_world=T_world_camera@p_camera"
                    ),
                }
            ),
            encoding="utf-8",
        )
        robot_state_path.write_text(
            json.dumps({"robot0_eef_pos_m": [1.0, 2.0, 4.0]}), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "created_at_utc": created_at,
            "valid": True,
            "scope": "single_frame_libero_observation_probe_no_planner_no_policy",
            "task": {
                "suite": "libero_spatial",
                "task_id": 0,
                "init_state_id": 0,
                "seed": 0,
                "language_instruction": "pick up the bowl",
            },
            "coordinate_conventions": {
                "image": "opencv_top_left_u_right_v_down",
                "metric_depth": "camera_z_depth_m",
                "camera": "x_right_y_down_z_forward",
                "world": "libero_mujoco_world",
                "extrinsics": "T_target_source",
            },
            "robot_state_json": str(robot_state_path),
            "cameras": {
                "agentview": {
                    "valid": True,
                    "rgb": {"path": str(rgb_path), "sha256": sha256_file(rgb_path)},
                    "artifacts": {
                        "depth_m_npy": str(depth_path),
                        "calibration_json": str(calibration_path),
                    },
                }
            },
        }
        (probe / "observation_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return probe

    def test_latest_valid_probe_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_probe(root, "older", "2026-11-01T10:00:00+00:00")
            newest = self._make_probe(root, "newest", "2026-11-01T11:00:00+00:00")

            selected = find_latest_valid_probe(root)

            self.assertEqual(selected, newest)

    def test_probe_load_and_geometry_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            probe = self._make_probe(
                Path(temporary), "probe", "2026-11-01T11:00:00+00:00"
            )
            planner_input = load_libero_planner_input(probe)

            geometry = build_trajectory_geometry(
                np.array([[500.0, 500.0, 2.0], [750.0, 500.0, 2.0]]),
                planner_input,
            )

            self.assertTrue(geometry.report.valid)
            np.testing.assert_allclose(geometry.camera_xyz_m[0], [0.0, 0.0, 2.0])
            np.testing.assert_allclose(geometry.world_xyz_m[0], [1.0, 2.0, 5.0])
            self.assertLess(
                geometry.report.metrics["projection_roundtrip_max_abs_error"], 1e-10
            )

    def test_coordinate_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            probe = self._make_probe(
                Path(temporary), "probe", "2026-11-01T11:00:00+00:00"
            )
            manifest_path = probe / "observation_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["coordinate_conventions"]["metric_depth"] = "unknown"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "coordinate convention mismatch"):
                load_libero_planner_input(probe)


if __name__ == "__main__":
    unittest.main()
