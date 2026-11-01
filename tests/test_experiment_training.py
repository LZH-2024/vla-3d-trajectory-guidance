import tempfile
import unittest
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from trajectory_guidance.experiment import (
    EvaluationStore,
    RunLedger,
    assert_experiments_comparable,
    load_experiment_config,
)
from trajectory_guidance.evaluation import EpisodeSpec, run_episode_plan
from trajectory_guidance.mock_smolvla import TinySmolVLAPolicy, make_tiny_batch
from trajectory_guidance.training import load_training_checkpoint, run_training_steps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentTrainingTest(unittest.TestCase):
    def test_experiment_switches_and_comparability(self):
        configs = [load_experiment_config(PROJECT_ROOT, name) for name in ("E0", "E1", "E2")]
        assert_experiments_comparable(configs)
        self.assertFalse(configs[0]["trajectory_conditioning"]["enabled"])
        self.assertEqual(configs[1]["trajectory_conditioning"]["gate_mode"], "none")
        self.assertEqual(configs[2]["trajectory_conditioning"]["gate_mode"], "geometry")

    def test_stage_ledger_stops_after_failure_until_retry_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = RunLedger(Path(temporary) / "ledger.json")
            with self.assertRaisesRegex(RuntimeError, "prerequisite"):
                ledger.begin("B")
            ledger.begin("A")
            ledger.fail_stage("A", "environment mismatch")
            with self.assertRaisesRegex(RuntimeError, "prerequisite"):
                ledger.begin("B")
            ledger.begin("A")
            ledger.pass_stage("A", metrics={"gpu": "mock"})
            ledger.begin("B")
            ledger.pass_stage("B")
            self.assertEqual(ledger.state["stages"]["A"]["attempts"], 2)

    def test_evaluation_records_resume_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = EvaluationStore(temporary)
            store.save_episode("task0-init0", {"success": True})
            self.assertEqual(store.completed_episode_ids(), {"task0-init0"})
            with self.assertRaises(FileExistsError):
                store.save_episode("task0-init0", {"success": False})
            self.assertEqual(store.summarize()["success_rate"], 1.0)

    def test_episode_runner_logs_failure_and_retries_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = EvaluationStore(temporary)
            episode = EpisodeSpec("task0-init0", "suite", 0, 0, 42, 10)
            attempts = {"count": 0}

            def flaky_runner(_episode, _config):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("mock rollout failure")
                return {"success": True, "steps": 4, "mock": True}

            with self.assertRaisesRegex(RuntimeError, "mock rollout failure"):
                run_episode_plan(
                    [episode],
                    runner=flaky_runner,
                    experiment_config={},
                    store=store,
                )
            self.assertEqual(store.completed_episode_ids(), set())
            self.assertEqual(len(list(store.failures_dir.glob("*.json"))), 1)
            summary = run_episode_plan(
                [episode],
                runner=flaky_runner,
                experiment_config={},
                store=store,
            )
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(attempts["count"], 2)

    def test_training_checkpoint_restores_exact_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = TinySmolVLAPolicy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
            batch = make_tiny_batch(include_trajectory=False)
            progress = run_training_steps(
                policy=policy,
                batches=[batch],
                optimizer=optimizer,
                run_dir=temporary,
                config_digest="digest",
                max_steps=2,
                checkpoint_every_steps=1,
            )
            restored_policy = TinySmolVLAPolicy()
            restored_optimizer = torch.optim.AdamW(restored_policy.parameters(), lr=1e-3)
            restored = load_training_checkpoint(
                progress.checkpoint_path,
                policy=restored_policy,
                optimizer=restored_optimizer,
                expected_config_digest="digest",
            )
            self.assertEqual(restored.step, 2)
            metrics_path = Path(temporary) / "metrics.jsonl"
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write('{"step": 99, "loss": 9.9, "forward_metrics": {}}\n')
            resumed = run_training_steps(
                policy=restored_policy,
                batches=[batch],
                optimizer=restored_optimizer,
                run_dir=temporary,
                config_digest="digest",
                max_steps=3,
                checkpoint_every_steps=1,
                start_step=restored.step,
            )
            self.assertEqual(resumed.step, 3)
            metric_steps = [
                int(json.loads(line)["step"])
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(metric_steps, [1, 2, 3])
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_training_checkpoint(
                    progress.checkpoint_path,
                    policy=restored_policy,
                    optimizer=restored_optimizer,
                    expected_config_digest="different",
                )


if __name__ == "__main__":
    unittest.main()
