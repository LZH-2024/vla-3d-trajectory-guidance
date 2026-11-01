from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_guidance.libero_assets import (
    REQUIRED_ASSET_PATHS,
    validate_libero_assets,
)


class LiberoAssetsTests(unittest.TestCase):
    def test_complete_snapshot_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in REQUIRED_ASSET_PATHS:
                path = root / relative
                if path.suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("fixture", encoding="utf-8")
                else:
                    path.mkdir(parents=True, exist_ok=True)

            self.assertEqual(validate_libero_assets(root), root.resolve())

    def test_missing_required_scene_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "libero_tabletop_base_style.xml"):
                validate_libero_assets(temporary)


if __name__ == "__main__":
    unittest.main()
