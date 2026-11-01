from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_guidance.visualization import save_trajectory_overlay


class VisualizationTests(unittest.TestCase):
    def test_overlay_is_saved_with_original_dimensions(self) -> None:
        rgb = np.zeros((40, 60, 3), dtype=np.uint8)
        waypoints = [[100.0, 200.0, 0.5], [900.0, 800.0, 1.0]]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overlay.png"
            save_trajectory_overlay(rgb, waypoints, [None], output)

            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.size, (60, 40))


if __name__ == "__main__":
    unittest.main()
