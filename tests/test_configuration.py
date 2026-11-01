from pathlib import Path
import json
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.planner = json.loads(
            (PROJECT_ROOT / "configs" / "planner_colab.json").read_text(encoding="utf-8")
        )
        cls.libero = json.loads(
            (PROJECT_ROOT / "configs" / "libero_probe_colab.json").read_text(encoding="utf-8")
        )
        cls.constraints = (PROJECT_ROOT / "configs" / "libero_constraints.txt").read_text(
            encoding="utf-8"
        )

    def test_planner_and_libero_environments_are_isolated(self) -> None:
        self.assertNotEqual(
            self.planner["planner"]["env_dir"],
            self.libero["libero"]["env_dir"],
        )
        self.assertEqual(self.planner["planner"]["env_dir"], "/content/.venv_planner")
        self.assertEqual(self.libero["libero"]["env_dir"], "/content/.venv")

    def test_libero_constraints_match_configured_protected_versions(self) -> None:
        expected = dict(self.planner["reference_core"])
        expected.update(
            {
                "transformers": self.libero["libero"]["transformers_version"],
                "huggingface-hub": self.libero["libero"]["huggingface_hub_version"],
                "hf-libero": self.libero["libero"]["hf_libero_version"],
                "robosuite": self.libero["libero"]["robosuite_version"],
                "robomimic": self.libero["libero"]["robomimic_version"],
                "mujoco": self.libero["libero"]["mujoco_version"],
            }
        )
        for package, version in expected.items():
            with self.subTest(package=package):
                self.assertIn(f"{package}=={version}", self.constraints)


if __name__ == "__main__":
    unittest.main()
