#!/usr/bin/env python3
"""Prepare one real, causally conditioned LIBERO batch for bounded Stage D.

The script downloads only the pinned candidate demonstration episodes needed
to identify the Stage-B initial observation.  It refuses to train on a merely
same-task trajectory: the language, EEF position, and both RGB views must pass
the configured match thresholds first.
"""

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

from trajectory_guidance.batch_artifact import save_batch_artifact
from trajectory_guidance.libero_pilot import (
    MatchThresholds,
    image_hwc_float01,
    image_sha256,
    normalized_instruction,
    scalar_int,
    score_candidate_start,
    select_unique_candidate,
    validate_real_action_chunk,
)
from trajectory_guidance.trajectory_conditioning import (
    TRAJECTORY_EEF_KEY,
    TRAJECTORY_MASK_KEY,
    TRAJECTORY_POINTS_KEY,
)
from trajectory_guidance.trajectory_record import load_trajectory_record
from trajectory_guidance.waypoint_selection import select_local_waypoints


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
    candidates = [recorded]
    if not recorded.is_absolute():
        candidates.append(root / recorded)
    candidates.append(root / recorded.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"recorded artifact does not exist: {value}")


def _load_prior_evidence(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger_path = run_dir / "acceptance_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for stage in ("B", "C"):
        if ledger["stages"][stage]["status"] != "passed":
            raise RuntimeError(f"Stage {stage} must pass before Stage D input preparation")
    stage_b_path = Path(ledger["stages"]["B"]["artifacts"]["summary"])
    stage_c_path = Path(ledger["stages"]["C"]["artifacts"]["summary"])
    if not stage_b_path.is_file() or not stage_c_path.is_file():
        raise FileNotFoundError("Stage B/C evidence is missing")
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    stage_c = json.loads(stage_c_path.read_text(encoding="utf-8"))
    if stage_b.get("valid") is not True or stage_c.get("all_passed") is not True:
        raise ValueError("Stage B/C recorded evidence is not valid")
    return stage_b, stage_c


def _load_probe_image(
    probe_dir: Path, manifest: dict[str, Any], camera_name: str
) -> np.ndarray:
    camera = manifest.get("cameras", {}).get(camera_name)
    if not camera or camera.get("valid") is not True:
        raise ValueError(f"probe camera is missing or invalid: {camera_name}")
    image_path = _resolve_recorded_artifact(probe_dir, camera["rgb"]["path"])
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if rgb.shape != (256, 256, 3):
        raise ValueError(f"unexpected probe RGB shape for {camera_name}: {rgb.shape}")
    # LeRobot's LIBERO dataset and LiberoProcessorStep use this 180-degree convention.
    return np.flip(rgb, axis=(0, 1)).copy().astype(np.float32) / 255.0


def _episode_has_instruction(row: dict[str, Any], instruction: str) -> bool:
    tasks = row.get("tasks", [])
    if isinstance(tasks, str):
        tasks = [tasks]
    wanted = normalized_instruction(instruction)
    return any(normalized_instruction(str(task)) == wanted for task in tasks)


def _first_frame_indices(dataset: Any, episodes: list[int]) -> dict[int, int]:
    view = dataset.select_columns(["episode_index", "frame_index"])
    wanted = set(episodes)
    found: dict[int, int] = {}
    for relative_index in range(len(view)):
        row = view[relative_index]
        episode = scalar_int(row["episode_index"], name="episode_index")
        frame = scalar_int(row["frame_index"], name="frame_index")
        if episode in wanted and frame == 0 and episode not in found:
            found[episode] = relative_index
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"candidate episodes have no frame zero: {missing}")
    return found


def _rank_episodes_by_initial_eef(
    dataset: Any,
    episodes: list[int],
    target_eef_xyz_m: np.ndarray,
) -> list[dict[str, Any]]:
    """Rank all same-task episodes without decoding or downloading videos."""

    view = dataset.select_columns(
        ["episode_index", "frame_index", "observation.state"]
    )
    wanted = set(episodes)
    ranked: list[dict[str, Any]] = []
    for relative_index in range(len(view)):
        row = view[relative_index]
        episode = scalar_int(row["episode_index"], name="episode_index")
        frame = scalar_int(row["frame_index"], name="frame_index")
        if episode not in wanted or frame != 0:
            continue
        state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"episode {episode} has an invalid initial 8-D state")
        ranked.append(
            {
                "episode_index": episode,
                "eef_distance_m": float(
                    np.linalg.norm(state[:3] - np.asarray(target_eef_xyz_m))
                ),
            }
        )
    if len(ranked) != len(wanted):
        found = {int(item["episode_index"]) for item in ranked}
        raise ValueError(
            f"same-task episodes missing frame-zero state: {sorted(wanted - found)}"
        )
    return sorted(ranked, key=lambda item: (item["eef_distance_m"], item["episode_index"]))


def _save_comparison(
    destination: Path,
    target: np.ndarray,
    candidate: np.ndarray,
) -> None:
    target = image_hwc_float01(target, name="target")
    candidate = image_hwc_float01(candidate, name="candidate")
    difference = np.abs(target - candidate)
    display = np.concatenate([target, candidate, difference], axis=1)
    Image.fromarray(np.round(display * 255.0).astype(np.uint8), mode="RGB").save(
        destination
    )


def _ensure_batched(value: torch.Tensor, *, expected_tail: tuple[int, ...], key: str) -> torch.Tensor:
    if tuple(value.shape) == expected_tail:
        return value.unsqueeze(0)
    if tuple(value.shape) == (1, *expected_tail):
        return value
    raise ValueError(
        f"unexpected preprocessed shape for {key!r}: {tuple(value.shape)}; "
        f"expected {expected_tail} or {(1, *expected_tail)}"
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (
        args.config or project_root / "configs" / "gpu_acceptance_colab.json"
    ).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config["stage_d"]
    expected_python = Path(config["smolvla_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")
    if platform.python_version() != "3.12.13":
        raise RuntimeError(f"unexpected Python version: {platform.python_version()}")

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    output_root = output.parent
    output_root.mkdir(parents=True, exist_ok=True)
    stage_b, stage_c = _load_prior_evidence(run_dir)
    policy_path = Path(stage_c["policy_path"]).resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Stage C policy path is missing: {policy_path}")

    probe_dir = Path(stage_b["source_probe_dir"]).resolve()
    trajectory_dir = Path(stage_b["artifacts"]["trajectory_cache_record"]).resolve()
    trajectory = load_trajectory_record(trajectory_dir, allowed_sources={"prediction"})
    if trajectory.uses_future_expert_information:
        raise ValueError("Stage-D condition trajectory contains future expert information")
    if trajectory.available_at_observation_index > trajectory.episode.observation_index:
        raise ValueError("Stage-D trajectory was not available at its source observation")

    probe_manifest = json.loads(
        (probe_dir / "observation_manifest.json").read_text(encoding="utf-8")
    )
    robot_state_path = _resolve_recorded_artifact(
        probe_dir, probe_manifest["robot_state_json"]
    )
    robot_state = json.loads(robot_state_path.read_text(encoding="utf-8"))
    probe_eef = np.asarray(robot_state["robot0_eef_pos_m"], dtype=np.float32)
    probe_agentview = _load_probe_image(probe_dir, probe_manifest, "agentview")
    probe_wrist = _load_probe_image(
        probe_dir, probe_manifest, "robot0_eye_in_hand"
    )

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(settings["dataset_root"]).expanduser().resolve()
    metadata = LeRobotDatasetMetadata(
        settings["dataset_repo_id"],
        root=dataset_root,
        revision=settings["dataset_revision"],
    )
    candidate_episodes = metadata.filter_episodes(
        lambda row: _episode_has_instruction(row, trajectory.instruction)
    )
    if not candidate_episodes:
        raise ValueError("the pinned LIBERO dataset has no episode for the Stage-B instruction")

    # Download the small parquet rows for every same-task episode first, rank by
    # frame-zero EEF, and only then fetch videos for the closest candidates.
    tabular_dataset = LeRobotDataset(
        settings["dataset_repo_id"],
        root=dataset_root,
        episodes=candidate_episodes,
        revision=settings["dataset_revision"],
        download_videos=False,
        video_backend=settings["video_backend"],
    )
    eef_ranking = _rank_episodes_by_initial_eef(
        tabular_dataset, candidate_episodes, probe_eef
    )
    candidate_episodes = [
        int(item["episode_index"])
        for item in eef_ranking[: int(settings["candidate_episode_limit"])]
    ]

    chunk_size = int(settings["action_chunk_size"])
    action_dim = int(settings["action_dim"])
    delta_timestamps = {
        "action": [index / float(metadata.fps) for index in range(chunk_size)]
    }
    dataset = LeRobotDataset(
        settings["dataset_repo_id"],
        root=dataset_root,
        episodes=candidate_episodes,
        delta_timestamps=delta_timestamps,
        revision=settings["dataset_revision"],
        download_videos=True,
        video_backend=settings["video_backend"],
    )
    first_indices = _first_frame_indices(dataset, candidate_episodes)
    thresholds = MatchThresholds.from_mapping(settings["observation_match"])
    diagnostics: list[dict[str, Any]] = []
    samples: dict[int, dict[str, Any]] = {}
    for episode in candidate_episodes:
        sample = dataset[first_indices[episode]]
        samples[episode] = sample
        state = np.asarray(sample["observation.state"], dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"LIBERO state must have shape (8,), got {state.shape}")
        candidate_instruction = str(sample["task"])
        diagnostic = score_candidate_start(
            episode_index=episode,
            target_eef_xyz_m=probe_eef,
            candidate_eef_xyz_m=state[:3],
            target_agentview=probe_agentview,
            candidate_agentview=sample["observation.images.image"],
            target_wrist=probe_wrist,
            candidate_wrist=sample["observation.images.image2"],
            instruction_matches=(
                normalized_instruction(candidate_instruction)
                == normalized_instruction(trajectory.instruction)
            ),
            thresholds=thresholds,
        )
        diagnostic["task"] = candidate_instruction
        diagnostics.append(diagnostic)

    match_diagnostics_path = output_root / "observation_match.json"
    try:
        best = select_unique_candidate(diagnostics, thresholds)
    except Exception as exc:
        _write_json(
            match_diagnostics_path,
            {
                "schema_version": 1,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "thresholds": thresholds.__dict__,
                "eef_prefilter": eef_ranking,
                "candidates": diagnostics,
            },
        )
        raise

    selected_episode = int(best["episode_index"])
    selected = samples[selected_episode]
    selected_state = np.asarray(selected["observation.state"], dtype=np.float32).reshape(8)
    selected_agentview = image_hwc_float01(
        selected["observation.images.image"], name="selected_agentview"
    )
    selected_wrist = image_hwc_float01(
        selected["observation.images.image2"], name="selected_wrist"
    )
    validate_real_action_chunk(
        selected["action"],
        action_is_pad=selected["action_is_pad"],
        chunk_size=chunk_size,
        action_dim=action_dim,
    )
    _save_comparison(
        output_root / "agentview_stage_b_vs_dataset_vs_absdiff.png",
        probe_agentview,
        selected_agentview,
    )
    _save_comparison(
        output_root / "wrist_stage_b_vs_dataset_vs_absdiff.png",
        probe_wrist,
        selected_wrist,
    )
    _write_json(
        match_diagnostics_path,
        {
            "schema_version": 1,
            "valid": True,
            "match_method": "thresholded_language_eef_and_two_view_rgb_not_exact_hash",
            "thresholds": thresholds.__dict__,
            "eef_prefilter": eef_ranking,
            "selected": dict(best),
            "candidates": diagnostics,
        },
    )

    # Policy/tokenizer assets were fixed and cached by Stage C.  Disable network
    # fallback only after the pinned dataset episode has been materialized.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    policy_config = SmolVLAConfig.from_pretrained(
        str(policy_path), local_files_only=True
    )
    if tuple(policy_config.robot_state_feature.shape) != (8,):
        raise ValueError("pinned SmolVLA policy does not use an 8-D LIBERO state")
    if tuple(policy_config.action_feature.shape) != (action_dim,):
        raise ValueError("pinned SmolVLA policy action dimension changed")
    if int(policy_config.chunk_size) != chunk_size:
        raise ValueError("pinned SmolVLA policy action chunk size changed")
    preprocessor, _ = make_pre_post_processors(
        policy_config,
        str(policy_path),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    raw_sample = {
        "observation.images.image": selected["observation.images.image"],
        "observation.images.image2": selected["observation.images.image2"],
        "observation.state": selected["observation.state"],
        "action": selected["action"],
        "task": str(selected["task"]),
    }
    processed = preprocessor(raw_sample)
    batch = {
        key: value for key, value in processed.items() if isinstance(value, torch.Tensor)
    }
    if "action" not in batch:
        raise KeyError("SmolVLA preprocessor did not return the real action target")
    batch["action"] = _ensure_batched(
        batch["action"],
        expected_tail=(chunk_size, action_dim),
        key="action",
    )

    window = select_local_waypoints(
        trajectory.waypoints_xyz_m,
        trajectory.valid_mask,
        selected_state[:3],
        count=int(settings["num_waypoints"]),
    )
    if not window.valid_mask.any():
        raise ValueError("matched Stage-B trajectory has no valid local waypoint")
    batch[TRAJECTORY_POINTS_KEY] = torch.from_numpy(
        window.waypoints_xyz_m.astype(np.float32)
    ).unsqueeze(0)
    batch[TRAJECTORY_MASK_KEY] = torch.from_numpy(window.valid_mask).unsqueeze(0)
    batch[TRAJECTORY_EEF_KEY] = torch.from_numpy(selected_state[:3]).unsqueeze(0)
    for key, value in batch.items():
        if value.ndim > 0 and int(value.shape[0]) != 1:
            raise ValueError(f"Stage-D batch dimension mismatch for {key!r}: {value.shape}")

    batch_dir = output_root / "real_demo_batch"
    save_batch_artifact(
        batch,
        batch_dir,
        metadata={
            "preprocessed_for_smolvla": True,
            "causality_checked": True,
            "condition_uses_future_expert_information": False,
            "action_supervision_contains_future_chunk": True,
            "real_demonstration_action_target": True,
            "synthetic_zero_action_target": False,
            "training_eligible_for_bounded_stage_d_pilot": True,
            "not_formal_training_data": True,
            "single_initial_chunk_only": True,
            "dataset_repo_id": settings["dataset_repo_id"],
            "dataset_revision": settings["dataset_revision"],
            "dataset_episode_index": selected_episode,
            "observation_index": 0,
            "action_indices": list(range(chunk_size)),
            "task": str(selected["task"]),
            "policy_path": str(policy_path),
            "policy_revision": config["stage_c"]["policy_revision"],
            "trajectory_source": trajectory.source,
            "trajectory_id": trajectory.trajectory_id,
            "trajectory_record": str(trajectory_dir),
            "observation_match": dict(best),
            "agentview_decoded_sha256": image_sha256(selected_agentview),
            "wrist_decoded_sha256": image_sha256(selected_wrist),
        },
    )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "scope": "one_real_matched_libero_demonstration_chunk_for_bounded_gpu_pilot",
        "policy_path": str(policy_path),
        "batch_artifact": str(batch_dir),
        "dataset": {
            "repo_id": settings["dataset_repo_id"],
            "revision": settings["dataset_revision"],
            "root": str(dataset_root),
            "selected_episode_index": selected_episode,
            "frame_index": 0,
        },
        "trajectory_record": str(trajectory_dir),
        "observation_match": dict(best),
        "match_diagnostics": str(match_diagnostics_path),
        "limitations": [
            "This is one matched initial demonstration chunk for optimizer/rollout plumbing only.",
            "The observation match uses bounded EEF and decoded-image differences, not an exact file hash.",
            "The future action chunk is supervision only and is never used as trajectory condition input.",
        ],
    }
    _write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
