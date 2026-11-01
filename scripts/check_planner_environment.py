#!/usr/bin/env python3
"""Validate exact versions, CUDA, imports, and optionally 3D HAMSTER model loading."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is not installed: {name}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-test", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--allow-low-memory",
        action="store_true",
        help="Allow a model-load attempt below 24 GiB VRAM. This may OOM.",
    )
    return parser.parse_args()


def gpu_info(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "index": index,
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "torch_cuda_build": torch.version.cuda,
        "allocated_bytes": torch.cuda.memory_allocated(index),
        "reserved_bytes": torch.cuda.memory_reserved(index),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (args.config or project_root / "configs" / "planner_colab.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    planner = config["planner"]
    core = config["reference_core"]

    expected_python = (Path(planner["env_dir"]) / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")
    if platform.python_version() != config["python_version"]:
        raise RuntimeError(
            f"Python version mismatch: got {platform.python_version()}, "
            f"expected {config['python_version']}"
        )

    expected_versions = {
        "torch": core["torch"],
        "torchvision": core["torchvision"],
        "torchcodec": core["torchcodec"],
        "numpy": core["numpy"],
        "nvidia-npp-cu12": core["nvidia-npp-cu12"],
        "transformers": planner["transformers_version"],
    }
    actual_versions = {name: distribution_version(name) for name in expected_versions}
    mismatches = {
        name: {"expected": expected_versions[name], "actual": actual}
        for name, actual in actual_versions.items()
        if actual != expected_versions[name]
    }
    if mismatches:
        raise RuntimeError(f"protected dependency versions changed: {mismatches}")

    checkout_dir = Path(planner["checkout_dir"]).resolve()
    try:
        checkout_commit = subprocess.check_output(
            ["git", "-C", str(checkout_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect the pinned 3D HAMSTER checkout: {checkout_dir}") from exc
    if checkout_commit != planner["repo_commit"]:
        raise RuntimeError(
            f"3D HAMSTER checkout mismatch: got {checkout_commit}, expected {planner['repo_commit']}"
        )

    import numpy as np
    import torch
    import torchcodec
    import torchvision
    import transformers
    from hamster3d.inference import Hamster3DPredictor

    record: dict[str, Any] = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "platform": platform.platform(),
        "versions": actual_versions,
        "installed_distributions": {
            distribution.metadata.get("Name", "unknown"): distribution.version
            for distribution in sorted(
                importlib.metadata.distributions(),
                key=lambda item: (item.metadata.get("Name") or "").lower(),
            )
        },
        "provenance": {
            "planner_repo_url": planner["repo_url"],
            "planner_repo_commit": checkout_commit,
            "model_id": planner["model_id"],
            "model_revision": planner["model_revision"],
        },
        "imports": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torchcodec": torchcodec.__version__,
            "transformers": transformers.__version__,
            "Hamster3DPredictor": f"{Hamster3DPredictor.__module__}.{Hamster3DPredictor.__name__}",
        },
        "gpu": gpu_info(torch),
    }
    npp_dir = (
        Path(sys.prefix)
        / "lib"
        / "python3.12"
        / "site-packages"
        / "nvidia"
        / "npp"
        / "lib"
    )
    npp_libraries = sorted(str(path) for path in npp_dir.glob("libnppicc.so*"))
    if not npp_libraries:
        raise RuntimeError(f"NVIDIA NPP library was not found under {npp_dir}")
    record["npp_libraries"] = npp_libraries
    try:
        record["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        record["nvidia_smi_error"] = str(exc)

    if args.gpu_test:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; GPU smoke test was not run")
        torch.cuda.reset_peak_memory_stats()
        left = torch.randn((1024, 1024), device="cuda")
        right = left @ left
        torch.cuda.synchronize()
        if not torch.isfinite(right).all().item():
            raise RuntimeError("GPU smoke test produced non-finite values")
        record["gpu_smoke_test"] = {
            "passed": True,
            "shape": list(right.shape),
            "gpu": gpu_info(torch),
        }
        del left, right
        torch.cuda.empty_cache()

    if args.load_model:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; model load was not attempted")
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gib < 24 and not args.allow_low_memory:
            raise RuntimeError(
                f"only {total_gib:.1f} GiB VRAM is available; switch to A100 or pass "
                "--allow-low-memory to accept the OOM risk"
            )
        model_path = (args.model_path or Path(planner["checkpoint_dir"])).resolve()
        if not (model_path / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"checkpoint is incomplete or missing: {model_path}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictor = Hamster3DPredictor(str(model_path), device="cuda:0")
        torch.cuda.synchronize()
        record["model_load"] = {
            "passed": True,
            "model_path": str(model_path),
            "seconds": time.perf_counter() - started,
            "gpu": gpu_info(torch),
        }
        del predictor
        torch.cuda.empty_cache()

    output = (
        args.output
        or project_root / "outputs" / "environment" / "planner_environment.json"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
