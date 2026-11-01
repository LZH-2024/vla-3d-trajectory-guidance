"""Dependency-light summaries for saved Stage-D rollout traces."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_array(value: Any, *, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have shape (N, {shape_tail}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def summarize_rollout_trace(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path).expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"rollout trace is missing: {trace_path}")
    with np.load(trace_path, allow_pickle=False) as payload:
        required = {
            "actions",
            "rewards",
            "eef_positions_xyz_m",
            "geometry_gate_gain_not_probability",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"rollout trace is missing arrays: {missing}")
        actions = _finite_array(payload["actions"], shape_tail=(7,), name="actions")
        eef = _finite_array(
            payload["eef_positions_xyz_m"],
            shape_tail=(3,),
            name="eef_positions_xyz_m",
        )
        rewards = np.asarray(payload["rewards"], dtype=np.float32).reshape(-1)
        gains = np.asarray(
            payload["geometry_gate_gain_not_probability"], dtype=np.float32
        ).reshape(-1)
    if len(actions) == 0 or len(eef) == 0:
        raise ValueError("rollout trace is empty")
    if len(actions) != len(rewards) or len(actions) != len(eef):
        raise ValueError("rollout action/reward/EEF lengths do not match")
    if len(gains) not in {0, len(actions)}:
        raise ValueError(
            "geometry gain length must be zero or match the rollout action length"
        )
    if not np.isfinite(rewards).all() or not np.isfinite(gains).all():
        raise ValueError("rollout rewards or geometry gains contain NaN or infinity")

    absolute_actions = np.abs(actions)
    absolute_motion_actions = absolute_actions[:, :6]
    action_norms = np.linalg.norm(actions[:, :6], axis=1)
    action_delta_norms = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
    eef_steps = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    displacement_from_start = np.linalg.norm(eef - eef[0], axis=1)
    gain_summary = None
    if len(gains):
        gain_summary = {
            "count": int(len(gains)),
            "min": float(gains.min()),
            "max": float(gains.max()),
            "mean": float(gains.mean()),
            "std": float(gains.std()),
            "is_calibrated_probability": False,
        }
    return {
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "steps": int(len(actions)),
        "actions": {
            "shape": list(actions.shape),
            "mean_per_dim": actions.mean(axis=0).astype(float).tolist(),
            "mean_abs_per_dim": absolute_actions.mean(axis=0).astype(float).tolist(),
            "max_abs_per_dim": absolute_actions.max(axis=0).astype(float).tolist(),
            "motion_l2_mean": float(action_norms.mean()),
            "motion_l2_max": float(action_norms.max()),
            "fraction_abs_ge_0_95": float(np.mean(absolute_actions >= 0.95)),
            "motion_fraction_abs_ge_0_95": float(
                np.mean(absolute_motion_actions >= 0.95)
            ),
            "motion_delta_l2_mean": (
                float(action_delta_norms.mean()) if len(action_delta_norms) else 0.0
            ),
            "gripper_min": float(actions[:, 6].min()),
            "gripper_max": float(actions[:, 6].max()),
            "gripper_mean": float(actions[:, 6].mean()),
        },
        "eef": {
            "initial_xyz_m": eef[0].astype(float).tolist(),
            "final_xyz_m": eef[-1].astype(float).tolist(),
            "net_displacement_m": float(np.linalg.norm(eef[-1] - eef[0])),
            "path_length_m": float(eef_steps.sum()),
            "max_displacement_from_start_m": float(displacement_from_start.max()),
            "mean_step_m": float(eef_steps.mean()) if len(eef_steps) else 0.0,
        },
        "rewards": {
            "sum": float(rewards.sum()),
            "nonzero_count": int(np.count_nonzero(rewards)),
            "max": float(rewards.max()),
        },
        "geometry_gate_gain_not_probability": gain_summary,
    }
