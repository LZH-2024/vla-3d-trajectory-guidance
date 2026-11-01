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
        cls.bridge = json.loads(
            (PROJECT_ROOT / "configs" / "libero_planner_single_frame.json").read_text(
                encoding="utf-8"
            )
        )
        cls.gpu_acceptance = json.loads(
            (PROJECT_ROOT / "configs" / "gpu_acceptance_colab.json").read_text(
                encoding="utf-8"
            )
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
        self.assertIn("accelerate==1.14.0", self.constraints)
        self.assertIn("num2words==0.5.14", self.constraints)

    def test_single_frame_bridge_reuses_isolated_configs(self) -> None:
        self.assertEqual(self.bridge["planner_config"], "configs/planner_colab.json")
        self.assertEqual(self.bridge["probe_config"], "configs/libero_probe_colab.json")
        self.assertEqual(self.bridge["single_frame"]["camera_name"], "agentview")
        self.assertEqual(self.bridge["single_frame"]["device"], "cuda:0")
        self.assertTrue(self.bridge["single_frame"]["offline"])
        self.assertEqual(
            self.bridge["single_frame"]["probe_output_root"],
            self.libero["probe"]["output_root"],
        )
        self.assertEqual(
            self.bridge["single_frame"]["accepted_depth_range_m"],
            self.libero["probe"]["accepted_depth_range_m"],
        )

    def test_gpu_acceptance_is_isolated_and_formal_stages_are_opt_in(self) -> None:
        self.assertEqual(
            self.gpu_acceptance["planner_python"],
            "/content/.venv_planner/bin/python",
        )
        self.assertEqual(
            self.gpu_acceptance["smolvla_python"],
            "/content/.venv/bin/python",
        )
        self.assertIsNone(self.gpu_acceptance["stage_c"]["policy_path"])
        self.assertEqual(
            self.gpu_acceptance["stage_c"]["policy_repo_id"],
            "lerobot/smolvla_libero",
        )
        self.assertEqual(
            self.gpu_acceptance["stage_c"]["policy_revision"],
            "a24f1f380461b665158d52acd140152c79e8b36d",
        )
        self.assertEqual(
            self.gpu_acceptance["stage_c"]["vlm_revision"],
            "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        )
        self.assertIsNone(self.gpu_acceptance["stage_d"]["command"])
        self.assertIsNone(self.gpu_acceptance["stage_e"]["command"])
        self.assertFalse(self.gpu_acceptance["persistence"]["enabled"])


if __name__ == "__main__":
    unittest.main()
