"""Portable tensor-batch artifacts for cross-environment SmolVLA smoke tests."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trajectory_guidance.libero_probe import sha256_file


BATCH_ARTIFACT_SCHEMA_VERSION = 1


def _as_numpy(value: Any, key: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"batch value {key!r} has unsupported object dtype")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"batch value {key!r} contains NaN or infinity")
    return array


def save_batch_artifact(
    batch: Mapping[str, Any],
    destination: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a tensor-only, already preprocessed batch without pickle."""

    if not batch:
        raise ValueError("batch cannot be empty")
    root = Path(destination).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"batch artifact directory already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for index, (key, value) in enumerate(sorted(batch.items())):
        if not isinstance(key, str) or not key:
            raise ValueError("batch keys must be non-empty strings")
        storage_key = f"array_{index:04d}"
        array = _as_numpy(value, key)
        arrays[storage_key] = array
        entries.append(
            {
                "storage_key": storage_key,
                "batch_key": key,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )

    with tempfile.TemporaryDirectory(prefix=f".{root.name}.tmp-", dir=root.parent) as temporary:
        temporary_root = Path(temporary)
        arrays_path = temporary_root / "batch_arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        manifest = {
            "schema_version": BATCH_ARTIFACT_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format": "preprocessed_tensor_batch_npz_no_pickle",
            "arrays_sha256": sha256_file(arrays_path),
            "entries": entries,
            "metadata": metadata or {},
        }
        (temporary_root / "batch_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary_root.rename(root)
    return root


def load_batch_artifact(source: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(source).expanduser().resolve()
    manifest_path = root / "batch_manifest.json"
    arrays_path = root / "batch_arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"incomplete batch artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BATCH_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("batch artifact schema version mismatch")
    if sha256_file(arrays_path) != manifest.get("arrays_sha256"):
        raise ValueError("batch artifact checksum mismatch")
    batch: dict[str, np.ndarray] = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for entry in manifest.get("entries", []):
            storage_key = entry["storage_key"]
            batch_key = entry["batch_key"]
            if storage_key not in arrays:
                raise ValueError(f"batch array is missing: {storage_key}")
            value = arrays[storage_key].copy()
            if list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"]:
                raise ValueError(f"batch array metadata mismatch for {batch_key!r}")
            if batch_key in batch:
                raise ValueError(f"duplicate batch key: {batch_key!r}")
            batch[batch_key] = value
    if len(batch) != len(manifest.get("entries", [])):
        raise ValueError("batch artifact entry count mismatch")
    return batch, manifest
