import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from trajectory_guidance.batch_artifact import load_batch_artifact, save_batch_artifact


class BatchArtifactTest(unittest.TestCase):
    def test_roundtrip_preserves_keys_shapes_and_dtypes(self):
        batch = {
            "observation.state": np.zeros((2, 8), dtype=np.float32),
            "observation.language.tokens": np.ones((2, 5), dtype=np.int64),
            "trajectory.local_waypoints_valid": np.ones((2, 4), dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            save_batch_artifact(batch, root, metadata={"preprocessed": True})
            loaded, manifest = load_batch_artifact(root)
            self.assertEqual(set(loaded), set(batch))
            for key in batch:
                np.testing.assert_array_equal(loaded[key], batch[key])
            self.assertTrue(manifest["metadata"]["preprocessed"])

    def test_nonfinite_float_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                save_batch_artifact(
                    {"bad": np.array([np.nan], dtype=np.float32)},
                    Path(temporary) / "batch",
                )

    def test_checksum_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            save_batch_artifact({"x": np.ones(2)}, root)
            with (root / "batch_arrays.npz").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_batch_artifact(root)


if __name__ == "__main__":
    unittest.main()
