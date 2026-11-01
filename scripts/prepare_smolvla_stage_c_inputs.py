#!/usr/bin/env python3
"""Prepare pinned SmolVLA weights and one explicit Stage C smoke batch."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from PIL import Image

from trajectory_guidance.batch_artifact import (
    load_batch_artifact,
    save_batch_artifact,
)
from trajectory_guidance.trajectory_conditioning import (
    TRAJECTORY_EEF_KEY,
    TRAJECTORY_MASK_KEY,
    TRAJECTORY_POINTS_KEY,
)
from trajectory_guidance.trajectory_record import load_trajectory_record
from trajectory_guidance.waypoint_selection import select_local_waypoints


VLM_SUPPORT_FILES = [
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _resolve_recorded_artifact(root: Path, value: str | Path) -> Path:
    recorded = Path(value).expanduser()
    if recorded.is_file():
        return recorded.resolve()
    relative = root / recorded
    if not recorded.is_absolute() and relative.is_file():
        return relative.resolve()
    basename = root / recorded.name
    if basename.is_file():
        return basename.resolve()
    raise FileNotFoundError(f"recorded artifact does not exist: {value}")


def _load_policy_image(probe_dir: Path, manifest: dict[str, Any], camera: str) -> torch.Tensor:
    camera_record = manifest.get("cameras", {}).get(camera)
    if not camera_record or camera_record.get("valid") is not True:
        raise ValueError(f"required Stage C camera is missing or invalid: {camera}")
    path = _resolve_recorded_artifact(probe_dir, camera_record["rgb"]["path"])
    rgb = np.asarray(Image.open(path).convert("RGB"))
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"unexpected {camera} RGB array: shape={rgb.shape}, dtype={rgb.dtype}")
    # The pinned LIBERO policy was trained on the dataset's 180-degree image convention.
    rgb = np.flip(rgb, axis=(0, 1)).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0


def _load_stage_b(run_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    ledger_path = run_dir / "acceptance_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    stage_b = ledger["stages"]["B"]
    if stage_b.get("status") != "passed":
        raise RuntimeError("Stage B must pass before Stage C inputs are prepared")
    summary_value = stage_b.get("artifacts", {}).get("summary")
    summary_path = Path(summary_value) if summary_value else Path()
    if not summary_path.is_file():
        raise FileNotFoundError("Stage B summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("valid") is not True:
        raise ValueError("Stage B summary is not valid")
    probe_dir = Path(summary["source_probe_dir"]).resolve()
    trajectory_dir = Path(summary["artifacts"]["trajectory_cache_record"]).resolve()
    return summary, probe_dir, trajectory_dir


def _download_policy(stage_c: dict[str, Any]) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    policy_dir = Path(stage_c["policy_cache_dir"]).resolve()
    print(
        f"Downloading/reusing {stage_c['policy_repo_id']} "
        f"@ {stage_c['policy_revision']} -> {policy_dir}",
        flush=True,
    )
    snapshot_download(
        repo_id=stage_c["policy_repo_id"],
        revision=stage_c["policy_revision"],
        local_dir=policy_dir,
    )
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
    ):
        if not (policy_dir / name).is_file():
            raise FileNotFoundError(f"SmolVLA checkpoint file is missing: {policy_dir / name}")

    # The model constructor and tokenizer use the VLM config/processor assets.  The
    # policy checkpoint already contains the actual VLM weights, so the large VLM
    # model file is intentionally not downloaded a second time.
    print(
        f"Caching VLM support files for {stage_c['vlm_repo_id']} "
        f"@ {stage_c['vlm_revision']}",
        flush=True,
    )
    vlm_snapshot = Path(
        snapshot_download(
            repo_id=stage_c["vlm_repo_id"],
            revision="main",
            allow_patterns=VLM_SUPPORT_FILES,
        )
    )
    if vlm_snapshot.name != stage_c["vlm_revision"]:
        raise RuntimeError(
            "VLM main revision changed: "
            f"{vlm_snapshot.name} != {stage_c['vlm_revision']}"
        )
    return policy_dir, vlm_snapshot


def _validate_reusable_batch(batch_dir: Path, stage_c: dict[str, Any]) -> bool:
    if not batch_dir.exists():
        return False
    _, manifest = load_batch_artifact(batch_dir)
    metadata = manifest.get("metadata", {})
    expected = {
        "preprocessed_for_smolvla": True,
        "causality_checked": True,
        "uses_future_expert_information": False,
        "policy_repo_id": stage_c["policy_repo_id"],
        "policy_revision": stage_c["policy_revision"],
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"existing Stage C batch metadata mismatch: {mismatches}")
    return True


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config or project_root / "configs" / "gpu_acceptance_colab.json"
    ).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage_c = config["stage_c"]
    expected_python = Path(config["smolvla_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")
    if platform.python_version() != "3.12.13":
        raise RuntimeError(f"unexpected Python version: {platform.python_version()}")

    run_dir = args.run_dir.resolve()
    stage_b, probe_dir, trajectory_dir = _load_stage_b(run_dir)
    policy_dir, vlm_snapshot = _download_policy(stage_c)
    output_root = args.output.resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)
    batch_dir = output_root / "smoke_batch"

    if not _validate_reusable_batch(batch_dir, stage_c):
        # Imports below require the optional SmolVLA dependencies installed by
        # bootstrap_smolvla_env.py.  Keeping them here leaves --help usable locally.
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from robosuite.utils.transform_utils import quat2axisangle

        manifest_path = probe_dir / "observation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("valid") is not True:
            raise ValueError("source LIBERO observation manifest is invalid")
        robot_state_path = _resolve_recorded_artifact(
            probe_dir, manifest["robot_state_json"]
        )
        robot_state = json.loads(robot_state_path.read_text(encoding="utf-8"))
        eef_position = np.asarray(robot_state["robot0_eef_pos_m"], dtype=np.float32)
        eef_quaternion = np.asarray(
            robot_state["robot0_eef_quat_xyzw"], dtype=np.float32
        )
        eef_axis_angle = np.asarray(
            quat2axisangle(eef_quaternion), dtype=np.float32
        )
        if eef_position.shape != (3,) or eef_axis_angle.shape != (3,):
            raise ValueError("invalid EEF state in the source probe")

        policy_config = SmolVLAConfig.from_pretrained(
            str(policy_dir), local_files_only=True
        )
        if policy_config.robot_state_feature is None or policy_config.action_feature is None:
            raise ValueError("SmolVLA LIBERO checkpoint has no state/action feature")
        if tuple(policy_config.robot_state_feature.shape) != (8,):
            raise ValueError("SmolVLA LIBERO checkpoint must use an 8-D state")
        if tuple(policy_config.action_feature.shape) != (7,):
            raise ValueError("SmolVLA LIBERO checkpoint must use a 7-D action")
        if int(policy_config.chunk_size) != 50:
            raise ValueError("unexpected SmolVLA action chunk size")

        # VLM support assets were resolved and checked above; no network fallback
        # is allowed while the actual preprocessor is constructed.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        preprocessor, _ = make_pre_post_processors(
            policy_config,
            str(policy_dir),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        gripper_qpos = np.zeros(2, dtype=np.float32)
        state = np.concatenate([eef_position, eef_axis_angle, gripper_qpos])
        raw_sample = {
            "observation.images.image": _load_policy_image(
                probe_dir, manifest, "agentview"
            ),
            "observation.images.image2": _load_policy_image(
                probe_dir, manifest, "robot0_eye_in_hand"
            ),
            "observation.state": torch.from_numpy(state),
            "action": torch.zeros(
                int(policy_config.chunk_size),
                int(policy_config.action_feature.shape[0]),
                dtype=torch.float32,
            ),
            "task": str(manifest["task"]["language_instruction"]),
        }
        processed = preprocessor(raw_sample)
        batch = {
            key: value
            for key, value in processed.items()
            if isinstance(value, torch.Tensor)
        }

        trajectory = load_trajectory_record(
            trajectory_dir, allowed_sources={"prediction"}
        )
        observation_index = int(trajectory.episode.observation_index)
        if trajectory.available_at_observation_index > observation_index:
            raise ValueError("trajectory is not causally available at this observation")
        window = select_local_waypoints(
            trajectory.waypoints_xyz_m,
            trajectory.valid_mask,
            eef_position,
            count=int(stage_c["num_waypoints"]),
        )
        if not window.valid_mask.any():
            raise ValueError("Stage B trajectory produced no valid local waypoint")
        batch[TRAJECTORY_POINTS_KEY] = torch.from_numpy(
            window.waypoints_xyz_m.astype(np.float32)
        ).unsqueeze(0)
        batch[TRAJECTORY_MASK_KEY] = torch.from_numpy(window.valid_mask).unsqueeze(0)
        batch[TRAJECTORY_EEF_KEY] = torch.from_numpy(eef_position).unsqueeze(0)

        for key, value in batch.items():
            if value.ndim > 0 and int(value.shape[0]) != 1:
                raise ValueError(
                    f"unexpected Stage C batch dimension for {key!r}: {tuple(value.shape)}"
                )
        save_batch_artifact(
            batch,
            batch_dir,
            metadata={
                "preprocessed_for_smolvla": True,
                "causality_checked": True,
                "uses_future_expert_information": False,
                "policy_repo_id": stage_c["policy_repo_id"],
                "policy_revision": stage_c["policy_revision"],
                "vlm_repo_id": stage_c["vlm_repo_id"],
                "vlm_revision": stage_c["vlm_revision"],
                "trajectory_source": trajectory.source,
                "trajectory_id": trajectory.trajectory_id,
                "observation_index": observation_index,
                "source_probe_dir": str(probe_dir),
                "smoke_only": True,
                "synthetic_zero_action_target": True,
                "gripper_qpos_zero_filled": True,
                "not_training_or_evaluation_data": True,
            },
        )

    _, batch_manifest = load_batch_artifact(batch_dir)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "scope": "stage_c_real_model_interface_smoke_inputs_not_training_data",
        "policy_path": str(policy_dir),
        "policy_repo_id": stage_c["policy_repo_id"],
        "policy_revision": stage_c["policy_revision"],
        "vlm_support_snapshot": str(vlm_snapshot),
        "vlm_revision": stage_c["vlm_revision"],
        "batch_artifact": str(batch_dir),
        "batch_metadata": batch_manifest["metadata"],
        "stage_b_summary": str(
            (Path(stage_b["output_dir"]) / "run_summary.json").resolve()
        ),
        "source_probe_dir": str(probe_dir),
        "trajectory_cache_record": str(trajectory_dir),
        "limitations": [
            "The zero action target is only for interface/backward smoke testing.",
            "The unavailable gripper qpos was zero-filled only for this smoke batch.",
            "This artifact is not training data and does not measure policy quality.",
        ],
    }
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
