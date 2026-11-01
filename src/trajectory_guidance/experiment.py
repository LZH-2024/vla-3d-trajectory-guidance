"""Comparable E0/E1/E2 configuration, stage ledger, and evaluation records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_FILES = {
    "E0": "e0_smolvla_base.json",
    "E1": "e1_smolvla_waypoints.json",
    "E2": "e2_smolvla_geometry_gate.json",
}
GPU_STAGES = ["A", "B", "C", "D", "E"]
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def config_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_experiment_config(
    project_root: str | Path,
    experiment_id: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    normalized_id = experiment_id.upper()
    if normalized_id not in EXPERIMENT_FILES:
        raise ValueError(f"unknown experiment {experiment_id!r}; expected E0, E1, or E2")
    config_root = root / "configs" / "experiments"
    common = json.loads((config_root / "common.json").read_text(encoding="utf-8"))
    override = json.loads(
        (config_root / EXPERIMENT_FILES[normalized_id]).read_text(encoding="utf-8")
    )
    config = _deep_merge(common, override)
    validate_experiment_config(config)
    config["config_digest"] = config_digest(config)
    return config


def validate_experiment_config(
    config: dict[str, Any], *, require_runnable: bool = False
) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("experiment schema_version must be 1")
    experiment_id = config.get("experiment_id")
    if experiment_id not in EXPERIMENT_FILES:
        raise ValueError("experiment_id must be E0, E1, or E2")
    upstream = config["upstream"]
    if upstream["lerobot_commit"] != "3f797de5952432284a7c77ad2760218cb74e2f26":
        raise ValueError("LeRobot commit changed without a compatibility re-audit")
    if upstream["env_dir"] != "/content/.venv":
        raise ValueError("SmolVLA must use /content/.venv")
    policy = config["policy"]
    condition = config["trajectory_conditioning"]
    if policy["compile_model"]:
        raise ValueError("V1/V2 hook has not been validated with torch.compile")
    if not policy["preserve_action_dimension"] or not policy["preserve_flow_matching_loss"]:
        raise ValueError("experiments must preserve the base action dimension and flow loss")
    expected_enabled = experiment_id in {"E1", "E2"}
    expected_gate = "geometry" if experiment_id == "E2" else "none"
    if bool(condition["enabled"]) != expected_enabled:
        raise ValueError(f"{experiment_id} trajectory enabled flag is inconsistent")
    if condition["gate_mode"] != expected_gate:
        raise ValueError(f"{experiment_id} gate mode must be {expected_gate!r}")
    if condition["injection_target"] != "model.action_time_mlp_out":
        raise ValueError("unreviewed SmolVLA injection target")
    if condition["gate_is_calibrated_probability"]:
        raise ValueError("geometry gate must not be represented as a calibrated probability")
    if condition["num_waypoints"] <= 0:
        raise ValueError("num_waypoints must be positive")
    data = config["data"]
    if data["allow_oracle_condition"]:
        raise ValueError("normal E0/E1/E2 configs must forbid oracle trajectory conditions")
    if not data["action_chunk_starts_at_observation"]:
        raise ValueError("action labels must be aligned to the condition observation")
    training = config["training"]
    if training["smoke_steps"] <= 0 or training["pilot_steps"] <= 0:
        raise ValueError("smoke and pilot steps must be positive")
    if require_runnable:
        if not data.get("dataset_repo_id"):
            raise ValueError("dataset_repo_id must be set before real training")
        if not policy.get("pretrained_path"):
            raise ValueError("policy.pretrained_path must be set before real SmolVLA loading")


def assert_experiments_comparable(configs: list[dict[str, Any]]) -> None:
    if {config["experiment_id"] for config in configs} != {"E0", "E1", "E2"}:
        raise ValueError("comparison requires exactly E0, E1, and E2")
    reference = configs[0]
    shared_paths = [
        ("data", "dataset_repo_id"),
        ("data", "suite"),
        ("data", "task_ids"),
        ("training", "seed"),
        ("training", "batch_size"),
        ("training", "learning_rate"),
        ("training", "smoke_steps"),
        ("training", "pilot_steps"),
        ("training", "formal_steps"),
        ("evaluation", "initial_state_ids"),
        ("evaluation", "episode_plan"),
        ("evaluation", "episodes_per_task_pilot"),
        ("evaluation", "episodes_per_task_formal"),
    ]
    for section, key in shared_paths:
        expected = reference[section][key]
        for config in configs[1:]:
            if config[section][key] != expected:
                raise ValueError(
                    f"experiment comparison mismatch at {section}.{key}: "
                    f"{expected!r} != {config[section][key]!r}"
                )


def resolve_experiment_output_root(config: dict[str, Any]) -> Path:
    runtime = config["runtime"]
    persistent = runtime.get("persistent_root") or os.environ.get("VLA_PERSISTENT_ROOT")
    root = Path(persistent) if persistent else Path(runtime["local_output_root"])
    return root.expanduser().resolve() / config["experiment_id"]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


class RunLedger:
    """Fail-stop stage state for concentrated GPU acceptance."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if self.path.is_file():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "host": {"platform": platform.platform(), "python": platform.python_version()},
                "stages": {
                    stage: {
                        "status": "pending",
                        "attempts": 0,
                        "started_at_utc": None,
                        "completed_at_utc": None,
                        "failure": None,
                        "metrics": {},
                        "artifacts": {},
                    }
                    for stage in GPU_STAGES
                },
            }
            self._save()

    def _save(self) -> None:
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.path, self.state)

    def begin(self, stage: str) -> None:
        if stage not in GPU_STAGES:
            raise ValueError(f"unknown GPU stage: {stage}")
        stage_index = GPU_STAGES.index(stage)
        incomplete = [
            previous
            for previous in GPU_STAGES[:stage_index]
            if self.state["stages"][previous]["status"] != "passed"
        ]
        if incomplete:
            raise RuntimeError(
                f"cannot start stage {stage}; prerequisite stages not passed: {incomplete}"
            )
        current = self.state["stages"][stage]
        if current["status"] == "passed":
            raise RuntimeError(f"stage {stage} already passed")
        current.update(
            {
                "status": "running",
                "attempts": int(current["attempts"]) + 1,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_at_utc": None,
                "failure": None,
            }
        )
        self._save()

    def pass_stage(
        self,
        stage: str,
        *,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        current = self.state["stages"][stage]
        if current["status"] != "running":
            raise RuntimeError(f"stage {stage} is not running")
        current.update(
            {
                "status": "passed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics or {},
                "artifacts": artifacts or {},
            }
        )
        self._save()

    def fail_stage(self, stage: str, failure: str) -> None:
        current = self.state["stages"][stage]
        if current["status"] != "running":
            raise RuntimeError(f"stage {stage} is not running")
        current.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure": str(failure),
            }
        )
        self._save()


class EvaluationStore:
    """Per-episode JSON records that naturally resume after interruption."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.episodes_dir = self.root / "episodes"
        self.failures_dir = self.root / "failures"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)

    def completed_episode_ids(self) -> set[str]:
        return {path.stem for path in self.episodes_dir.glob("*.json") if path.is_file()}

    def save_episode(self, episode_id: str, result: dict[str, Any]) -> Path:
        if not _SAFE_ID.fullmatch(episode_id):
            raise ValueError("episode_id is not path-safe")
        destination = self.episodes_dir / f"{episode_id}.json"
        if destination.exists():
            raise FileExistsError(f"episode result already exists: {destination}")
        payload = {
            "schema_version": 1,
            "episode_id": episode_id,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        _atomic_json(destination, payload)
        return destination

    def save_failure(self, episode_id: str, failure: dict[str, Any]) -> Path:
        if not _SAFE_ID.fullmatch(episode_id):
            raise ValueError("episode_id is not path-safe")
        attempt = len(list(self.failures_dir.glob(f"{episode_id}-attempt*.json"))) + 1
        destination = self.failures_dir / f"{episode_id}-attempt{attempt:03d}.json"
        payload = {
            "schema_version": 1,
            "episode_id": episode_id,
            "attempt": attempt,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            **failure,
        }
        _atomic_json(destination, payload)
        return destination

    def summarize(self) -> dict[str, Any]:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.episodes_dir.glob("*.json"))
        ]
        successes = sum(bool(record.get("success")) for record in records)
        summary = {
            "schema_version": 1,
            "episode_count": len(records),
            "success_count": successes,
            "success_rate": successes / len(records) if records else None,
            "episode_ids": [record["episode_id"] for record in records],
        }
        _atomic_json(self.root / "summary.json", summary)
        return summary
