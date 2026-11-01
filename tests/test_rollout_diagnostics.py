from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_guidance.rollout_diagnostics import summarize_rollout_trace


class RolloutDiagnosticsTests(unittest.TestCase):
    def test_trace_summary_reports_actions_eef_rewards_and_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.npz"
            np.savez_compressed(
                path,
                actions=np.array(
                    [
                        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                rewards=np.array([0.0, 1.0], dtype=np.float32),
                eef_positions_xyz_m=np.array(
                    [[0.0, 0.0, 0.0], [0.03, 0.04, 0.0]], dtype=np.float32
                ),
                geometry_gate_gain_not_probability=np.array(
                    [0.25, 0.75], dtype=np.float32
                ),
            )

            summary = summarize_rollout_trace(path)

            self.assertEqual(summary["steps"], 2)
            self.assertEqual(summary["actions"]["motion_fraction_abs_ge_0_95"], 0.0)
            self.assertAlmostEqual(summary["actions"]["motion_delta_l2_mean"], 0.1)
            self.assertAlmostEqual(summary["eef"]["net_displacement_m"], 0.05)
            self.assertAlmostEqual(summary["eef"]["path_length_m"], 0.05)
            self.assertEqual(summary["rewards"]["nonzero_count"], 1)
            self.assertAlmostEqual(
                summary["geometry_gate_gain_not_probability"]["mean"], 0.5
            )
            self.assertFalse(
                summary["geometry_gate_gain_not_probability"][
                    "is_calibrated_probability"
                ]
            )

    def test_empty_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.npz"
            np.savez_compressed(
                path,
                actions=np.empty((0, 7), dtype=np.float32),
                rewards=np.empty((0,), dtype=np.float32),
                eef_positions_xyz_m=np.empty((0, 3), dtype=np.float32),
                geometry_gate_gain_not_probability=np.empty((0,), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "empty"):
                summarize_rollout_trace(path)

    def test_partial_gate_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.npz"
            np.savez_compressed(
                path,
                actions=np.zeros((2, 7), dtype=np.float32),
                rewards=np.zeros((2,), dtype=np.float32),
                eef_positions_xyz_m=np.zeros((2, 3), dtype=np.float32),
                geometry_gate_gain_not_probability=np.ones((1,), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "gain length"):
                summarize_rollout_trace(path)


if __name__ == "__main__":
    unittest.main()
