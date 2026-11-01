"""Pinned LIBERO asset preparation shared by probes and GPU rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


REQUIRED_ASSET_PATHS = (
    "articulated_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
    "stable_hope_objects",
    "scenes",
    "scenes/libero_tabletop_base_style.xml",
)


def validate_libero_assets(destination: str | Path) -> Path:
    root = Path(destination).expanduser().resolve()
    missing = [relative for relative in REQUIRED_ASSET_PATHS if not (root / relative).exists()]
    if missing:
        raise RuntimeError(f"pinned LIBERO asset snapshot is incomplete; missing {missing}")
    return root


def download_pinned_libero_assets(
    libero_config: Mapping[str, Any],
) -> tuple[Path, str]:
    """Materialize and verify the exact configured Hugging Face asset revision."""

    from huggingface_hub import HfApi, snapshot_download

    repo_id = str(libero_config["assets_repo_id"])
    revision = str(libero_config["assets_revision"])
    info = HfApi().dataset_info(repo_id, revision=revision)
    if info.sha != revision:
        raise RuntimeError(f"asset revision resolved to {info.sha}, expected {revision}")
    destination = Path(libero_config["assets_dir"]).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
    )
    return validate_libero_assets(destination), info.sha


def activate_libero_assets(destination: str | Path) -> Path:
    """Point hf-libero's process-local cache at an already verified snapshot."""

    root = validate_libero_assets(destination)
    import libero.libero as libero_root

    libero_root._assets_path_cache = str(root)
    return root
