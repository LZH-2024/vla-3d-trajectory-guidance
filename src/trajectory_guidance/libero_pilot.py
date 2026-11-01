"""Fail-closed helpers for the bounded LIBERO Stage-D GPU pilot.

This module deliberately contains no LeRobot or simulator imports.  The GPU
scripts use it to compare a cached Stage-B observation with candidate LIBERO
demonstration starts before any optimizer step is allowed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def scalar_int(value: Any, *, name: str) -> int:
    """Read one integer from Python, NumPy, or Torch scalar-like values."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one value, got shape {array.shape}")
    return int(array.reshape(-1)[0])


def normalized_instruction(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def image_hwc_float01(value: Any, *, name: str) -> np.ndarray:
    """Convert one RGB image to finite HWC float32 in [0, 1]."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"{name} must be a 3-D RGB image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must have three RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if float(image.min()) < -1e-6 or float(image.max()) > 1.0 + 1e-6:
        raise ValueError(f"{name} is outside the expected [0, 1] range")
    return np.clip(image, 0.0, 1.0)


def image_sha256(image_hwc: np.ndarray) -> str:
    image = image_hwc_float01(image_hwc, name="image")
    quantized = np.round(image * 255.0).astype(np.uint8)
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class MatchThresholds:
    max_eef_distance_m: float
    max_agentview_mae: float
    max_wrist_mae: float
    min_score_margin: float

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "MatchThresholds":
        result = cls(
            max_eef_distance_m=float(source["max_eef_distance_m"]),
            max_agentview_mae=float(source["max_agentview_mae"]),
            max_wrist_mae=float(source["max_wrist_mae"]),
            min_score_margin=float(source["min_score_margin"]),
        )
        if (
            result.max_eef_distance_m <= 0.0
            or result.max_agentview_mae <= 0.0
            or result.max_wrist_mae <= 0.0
            or result.min_score_margin < 0.0
        ):
            raise ValueError("Stage-D match thresholds must be positive")
        return result


def score_candidate_start(
    *,
    episode_index: int,
    target_eef_xyz_m: np.ndarray,
    candidate_eef_xyz_m: np.ndarray,
    target_agentview: np.ndarray,
    candidate_agentview: np.ndarray,
    target_wrist: np.ndarray,
    candidate_wrist: np.ndarray,
    instruction_matches: bool,
    thresholds: MatchThresholds,
) -> dict[str, Any]:
    """Return transparent geometry/image diagnostics for one episode start."""

    target_eef = np.asarray(target_eef_xyz_m, dtype=np.float64).reshape(-1)
    candidate_eef = np.asarray(candidate_eef_xyz_m, dtype=np.float64).reshape(-1)
    if target_eef.shape != (3,) or candidate_eef.shape != (3,):
        raise ValueError("EEF positions must be finite 3-vectors")
    if not np.isfinite(target_eef).all() or not np.isfinite(candidate_eef).all():
        raise ValueError("EEF positions contain NaN or infinity")
    target_main = image_hwc_float01(target_agentview, name="target_agentview")
    candidate_main = image_hwc_float01(candidate_agentview, name="candidate_agentview")
    target_hand = image_hwc_float01(target_wrist, name="target_wrist")
    candidate_hand = image_hwc_float01(candidate_wrist, name="candidate_wrist")
    if target_main.shape != candidate_main.shape or target_hand.shape != candidate_hand.shape:
        raise ValueError("candidate and Stage-B image shapes do not match")

    eef_distance = float(np.linalg.norm(candidate_eef - target_eef))
    agentview_mae = float(np.mean(np.abs(candidate_main - target_main)))
    wrist_mae = float(np.mean(np.abs(candidate_hand - target_hand)))
    score = (
        eef_distance / thresholds.max_eef_distance_m
        + agentview_mae / thresholds.max_agentview_mae
        + wrist_mae / thresholds.max_wrist_mae
    )
    passed = bool(
        instruction_matches
        and eef_distance <= thresholds.max_eef_distance_m
        and agentview_mae <= thresholds.max_agentview_mae
        and wrist_mae <= thresholds.max_wrist_mae
    )
    return {
        "episode_index": int(episode_index),
        "instruction_matches": bool(instruction_matches),
        "eef_distance_m": eef_distance,
        "agentview_mae_0_to_1": agentview_mae,
        "wrist_mae_0_to_1": wrist_mae,
        "normalized_score": float(score),
        "passes_thresholds": passed,
    }


def select_unique_candidate(
    diagnostics: Sequence[Mapping[str, Any]], thresholds: MatchThresholds
) -> Mapping[str, Any]:
    passing = sorted(
        (item for item in diagnostics if item.get("passes_thresholds") is True),
        key=lambda item: (float(item["normalized_score"]), int(item["episode_index"])),
    )
    if not passing:
        raise ValueError("no LIBERO demonstration start matches the Stage-B observation")
    if len(passing) > 1:
        margin = float(passing[1]["normalized_score"]) - float(
            passing[0]["normalized_score"]
        )
        if margin < thresholds.min_score_margin:
            raise ValueError(
                "Stage-B observation match is ambiguous: "
                f"score margin {margin:.6f} < {thresholds.min_score_margin:.6f}"
            )
    return passing[0]


def validate_real_action_chunk(
    action: Any,
    *,
    action_is_pad: Any,
    chunk_size: int,
    action_dim: int,
) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (chunk_size, action_dim):
        raise ValueError(
            f"real action chunk must have shape {(chunk_size, action_dim)}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("real action chunk contains NaN or infinity")
    if hasattr(action_is_pad, "detach"):
        action_is_pad = action_is_pad.detach().cpu().numpy()
    padding = np.asarray(action_is_pad, dtype=np.bool_)
    if padding.shape != (chunk_size,):
        raise ValueError(
            f"action padding mask must have shape {(chunk_size,)}, got {padding.shape}"
        )
    if padding.any():
        raise ValueError("pilot action chunk contains padded future actions")
    return array
