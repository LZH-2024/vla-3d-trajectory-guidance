#!/usr/bin/env python3
"""Validate the isolated LeRobot / LIBERO probe environment and record versions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.libero_probe import prepare_libero_config, write_json


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is not installed: {name}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-test", action="store_true")
    return parser.parse_args()


def gpu_info(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "index": index,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "torch_cuda_build": torch.version.cuda,
    }


def ensure_npp_loader_path() -> tuple[Path, list[str]]:
    """Re-exec once so the dynamic loader sees the venv's NPP directory."""

    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    npp_dir = Path(sys.prefix) / "lib" / python_dir / "site-packages" / "nvidia" / "npp" / "lib"
    libraries = sorted(str(path) for path in npp_dir.glob("libnppicc.so*"))
    if not libraries:
        raise RuntimeError(f"NVIDIA NPP library was not found under {npp_dir}")

    current = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [path for path in current.split(os.pathsep) if path]
    if str(npp_dir) not in paths:
        if os.environ.get("TRAJECTORY_GUIDANCE_NPP_REEXEC") == "1":
            raise RuntimeError(
                f"NPP directory is still absent from the loader path after re-exec: {npp_dir}"
            )
        updated_env = os.environ.copy()
        updated_env["LD_LIBRARY_PATH"] = str(npp_dir) + (os.pathsep + current if current else "")
        updated_env["TRAJECTORY_GUIDANCE_NPP_REEXEC"] = "1"
        print(f"Re-executing with NPP loader path: {npp_dir}", flush=True)
        os.execve(sys.executable, [sys.executable, *sys.argv], updated_env)
    return npp_dir, libraries


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (args.config or project_root / "configs" / "libero_probe_colab.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    core_config = json.loads(
        (project_root / config["protected_core_config"]).read_text(encoding="utf-8")
    )
    libero = config["libero"]
    core = core_config["reference_core"]

    print("[1/6] Checking target interpreter and render backend...", flush=True)

    expected_python = (Path(libero["env_dir"]) / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")
    if platform.python_version() != core_config["python_version"]:
        raise RuntimeError(
            f"Python version mismatch: got {platform.python_version()}, "
            f"expected {core_config['python_version']}"
        )
    if os.environ.get("MUJOCO_GL") != libero["render_backend"]:
        raise RuntimeError(
            f"MUJOCO_GL must be explicitly set to {libero['render_backend']!r}, "
            f"got {os.environ.get('MUJOCO_GL')!r}"
        )
    npp_dir, npp_libraries = ensure_npp_loader_path()

    expected_versions = {
        "torch": core["torch"],
        "torchvision": core["torchvision"],
        "torchcodec": core["torchcodec"],
        "numpy": core["numpy"],
        "nvidia-npp-cu12": core["nvidia-npp-cu12"],
        "transformers": libero["transformers_version"],
        "huggingface-hub": libero["huggingface_hub_version"],
        "hf-libero": libero["hf_libero_version"],
        "robosuite": libero["robosuite_version"],
        "robomimic": libero["robomimic_version"],
        "mujoco": libero["mujoco_version"],
        "lerobot": libero["repo_version"],
    }
    print("[2/6] Checking protected distribution versions...", flush=True)
    actual_versions = {name: distribution_version(name) for name in expected_versions}
    mismatches = {
        name: {"expected": expected_versions[name], "actual": actual}
        for name, actual in actual_versions.items()
        if actual != expected_versions[name]
    }
    if mismatches:
        raise RuntimeError(f"protected dependency versions changed: {mismatches}")
    print(json.dumps(actual_versions, ensure_ascii=False, indent=2), flush=True)

    print("[3/6] Checking the pinned LeRobot checkout...", flush=True)
    checkout_dir = Path(libero["checkout_dir"]).resolve()
    checkout_commit = subprocess.check_output(
        ["git", "-C", str(checkout_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if checkout_commit != libero["repo_commit"]:
        raise RuntimeError(
            f"LeRobot checkout mismatch: got {checkout_commit}, expected {libero['repo_commit']}"
        )

    os.environ["LIBERO_CONFIG_PATH"] = str(Path(libero["config_dir"]).resolve())
    libero_config_path = prepare_libero_config(libero["config_dir"], libero["datasets_dir"])

    print("[4/6] Importing MuJoCo, robosuite, LIBERO, and protected ML packages...", flush=True)
    import mujoco
    import numpy as np
    import robomimic
    import robosuite
    import torch
    import torchcodec
    import torchvision
    import transformers
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )
    print("[5/6] Imports passed; recording environment metadata...", flush=True)

    record: dict[str, Any] = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "platform": platform.platform(),
        "environment": {
            "MUJOCO_GL": os.environ["MUJOCO_GL"],
            "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LIBERO_CONFIG_PATH": os.environ["LIBERO_CONFIG_PATH"],
            "libero_config_file": str(libero_config_path),
            "npp_library_dir": str(npp_dir),
        },
        "npp_libraries": npp_libraries,
        "versions": actual_versions,
        "provenance": {
            "lerobot_repo_url": libero["repo_url"],
            "lerobot_repo_commit": checkout_commit,
            "hf_libero_wheel_sha256": libero["hf_libero_wheel_sha256"],
            "assets_repo_id": libero["assets_repo_id"],
            "assets_revision": libero["assets_revision"],
        },
        "imports": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torchcodec": torchcodec.__version__,
            "transformers": transformers.__version__,
            "robosuite": getattr(robosuite, "__version__", actual_versions["robosuite"]),
            "robomimic": getattr(robomimic, "__version__", actual_versions["robomimic"]),
            "mujoco": getattr(mujoco, "__version__", actual_versions["mujoco"]),
            "OffScreenRenderEnv": f"{OffScreenRenderEnv.__module__}.{OffScreenRenderEnv.__name__}",
            "camera_utilities": [
                get_real_depth_map.__name__,
                get_camera_intrinsic_matrix.__name__,
                get_camera_extrinsic_matrix.__name__,
            ],
        },
        "gpu": gpu_info(torch),
        "installed_distributions": {
            distribution.metadata.get("Name", "unknown"): distribution.version
            for distribution in sorted(
                importlib.metadata.distributions(),
                key=lambda item: (item.metadata.get("Name") or "").lower(),
            )
        },
    }
    try:
        record["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        record["nvidia_smi_error"] = str(exc)

    if args.gpu_test:
        print("[6/6] Running the CUDA smoke test...", flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; GPU smoke test was not run")
        left = torch.randn((512, 512), device="cuda")
        result = left @ left
        torch.cuda.synchronize()
        if not torch.isfinite(result).all().item():
            raise RuntimeError("GPU smoke test produced non-finite values")
        record["gpu_smoke_test"] = {"passed": True, "shape": list(result.shape)}

    output = (
        args.output
        or project_root / "outputs" / "environment" / "libero_environment.json"
    ).resolve()
    write_json(output, record)
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Environment record: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
