import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StageDDiagnosticsTests(unittest.TestCase):
    def test_existing_rollouts_are_summarized_without_gpu_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            stage_d_root = run_dir / "stage_d" / "attempt_007" / "pilot"
            inputs = stage_d_root / "stage_d_inputs.json"
            assets = stage_d_root / "libero_assets.json"
            inputs.parent.mkdir(parents=True)
            inputs.write_text("{}", encoding="utf-8")
            assets.write_text("{}", encoding="utf-8")

            experiment_records = []
            for experiment_id in ("E0", "E1", "E2"):
                experiment_dir = stage_d_root / experiment_id
                summary = self._write_rollout(
                    experiment_dir,
                    experiment_id=experiment_id,
                    optimizer_steps=2,
                )
                experiment_records.append(
                    {"experiment_id": experiment_id, "summary": str(summary)}
                )

            stage_d_summary = stage_d_root / "summary.json"
            self._write_json(
                stage_d_summary,
                {
                    "valid": True,
                    "inputs": str(inputs),
                    "libero_assets": str(assets),
                    "experiments": experiment_records,
                },
            )
            self._write_json(
                run_dir / "acceptance_ledger.json",
                {
                    "stages": {
                        "D": {
                            "status": "passed",
                            "artifacts": {"summary": str(stage_d_summary)},
                        }
                    }
                },
            )
            self._write_rollout(
                run_dir / "stage_d_diagnostics" / "E0_pretrained_no_update",
                experiment_id="E0",
                optimizer_steps=0,
            )
            config = root / "config.json"
            self._write_json(config, {})
            output = root / "diagnostics.json"

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_stage_d_diagnostics.py"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--config",
                    str(config),
                    "--run-dir",
                    str(run_dir),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["valid"])
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual(result["records"][0]["optimizer_steps"], 0)
            self.assertIn("delta_vs_pretrained_e0", result["records"][1])

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_rollout(
        self,
        directory: Path,
        *,
        experiment_id: str,
        optimizer_steps: int,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / "rollout_trace.npz"
        np.savez_compressed(
            trace,
            actions=np.zeros((2, 7), dtype=np.float32),
            rewards=np.zeros((2,), dtype=np.float32),
            eef_positions_xyz_m=np.zeros((2, 3), dtype=np.float32),
            geometry_gate_gain_not_probability=(
                np.empty((0,), dtype=np.float32)
                if experiment_id == "E0"
                else np.ones((2,), dtype=np.float32)
            ),
        )
        summary = directory / "summary.json"
        self._write_json(
            summary,
            {
                "valid": True,
                "experiment_id": experiment_id,
                "optimizer_steps": optimizer_steps,
                "rollout": {
                    "completed": True,
                    "success": False,
                    "seconds": 1.0,
                    "trace": str(trace),
                    "snapshots": str(directory / "snapshots"),
                },
            },
        )
        return summary


if __name__ == "__main__":
    unittest.main()
