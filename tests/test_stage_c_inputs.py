from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_smolvla_stage_c_inputs import _ensure_batched_action


class StageCInputTests(unittest.TestCase):
    def test_unbatched_action_chunk_gets_one_batch_dimension(self) -> None:
        action = torch.zeros(50, 7)
        result = _ensure_batched_action(action, chunk_size=50, action_dim=7)
        self.assertEqual(tuple(result.shape), (1, 50, 7))
        self.assertEqual(result.data_ptr(), action.data_ptr())

    def test_already_batched_action_chunk_is_unchanged(self) -> None:
        action = torch.zeros(1, 50, 7)
        result = _ensure_batched_action(action, chunk_size=50, action_dim=7)
        self.assertIs(result, action)

    def test_invalid_action_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "preprocessed action shape"):
            _ensure_batched_action(
                torch.zeros(49, 7), chunk_size=50, action_dim=7
            )


if __name__ == "__main__":
    unittest.main()
