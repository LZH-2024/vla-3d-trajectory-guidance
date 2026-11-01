"""Reusable runner for the six official 3D HAMSTER examples."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from trajectory_guidance.geometry import normalized_uvd_to_camera_xyz
from trajectory_guidance.validation import validate_depth_map, validate_waypoints
from trajectory_guidance.visualization import save_trajectory_overlay


@dataclass(frozen=True)
class OfficialExample:
    name: str
    rgb_path: Path
    depth_path: Path
    instruction_path: Path
    camera_path: Path | None


def discover_official_examples(examples_dir: str | Path) -> list[OfficialExample]:
    root = Path(examples_dir)
    examples: list[OfficialExample] = []
    for rgb_path in sorted(root.glob("sample_*_rgb.png")):
        name = rgb_path.name.removesuffix("_rgb.png")
        depth_path = root / f"{name}_depth.npy"
        instruction_path = root / f"{name}_instruction.txt"
        camera_path = root / f"{name}_camera.json"
        if depth_path.is_file() and instruction_path.is_file():
            examples.append(
                OfficialExample(
                    name=name,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    instruction_path=instruction_path,
                    camera_path=camera_path if camera_path.is_file() else None,
                )
            )
    return examples


def _distribution_versions() -> dict[str, str | None]:
    names = [
        "torch",
        "torchvision",
        "torchcodec",
        "numpy",
        "nvidia-npp-cu12",
        "transformers",
        "huggingface-hub",
        "accelerate",
        "hamster3d",
        "trajectory-guidance",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_snapshot(torch_module: Any) -> dict[str, Any]:
    torch = torch_module
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "available": True,
        "device_index": device,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return repr(float(value))
        if isinstance(value, np.generic):
            return value.item()
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_waypoints_csv(
    path: Path,
    waypoints: list[list[float]],
    actions: list[str | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "u_normalized", "v_normalized", "depth_m", "action"])
        for index, point in enumerate(waypoints):
            action = actions[index] if index < len(actions) else None
            writer.writerow([index, point[0], point[1], point[2], action or ""])


def _write_xyz_csv(path: Path, xyz: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "x_camera_m", "y_camera_m", "z_camera_m"])
        for index, point in enumerate(xyz):
            writer.writerow([index, float(point[0]), float(point[1]), float(point[2])])


def _load_camera(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_official_examples(
    *,
    model_path: str | Path,
    examples_dir: str | Path,
    output_dir: str | Path,
    sample_names: Iterable[str] | None = None,
    device: str = "cuda:0",
    max_new_tokens: int = 1024,
    continue_on_error: bool = False,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the official predictor once and run selected bundled examples."""

    import torch
    from hamster3d.inference import Hamster3DPredictor

    selected_names = set(sample_names or [])
    examples = discover_official_examples(examples_dir)
    if selected_names:
        examples = [example for example in examples if example.name in selected_names]
        missing = sorted(selected_names - {example.name for example in examples})
        if missing:
            raise FileNotFoundError(f"official examples not found: {missing}")
    if not examples:
        raise FileNotFoundError(f"no official examples found under {examples_dir}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; 3D HAMSTER model loading was not attempted")
    requested_device = torch.device(device)
    if requested_device.type != "cuda":
        raise ValueError(f"3D HAMSTER requires a CUDA device, got {device}")
    torch.cuda.set_device(requested_device.index or 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    load_started = time.perf_counter()
    predictor = Hamster3DPredictor(str(model_path), device=device)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    model_load_gpu = _gpu_snapshot(torch)

    run_summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(model_path).resolve()),
        "examples_dir": str(Path(examples_dir).resolve()),
        "device": device,
        "max_new_tokens": max_new_tokens,
        "provenance": provenance or {},
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "versions": _distribution_versions(),
        "model_load": {
            "seconds": model_load_seconds,
            "gpu": model_load_gpu,
        },
        "samples": [],
    }

    for example in examples:
        sample_dir = destination / example.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_record: dict[str, Any] = {
            "name": example.name,
            "instruction": example.instruction_path.read_text(encoding="utf-8").strip(),
            "source_files": {
                "rgb": str(example.rgb_path.resolve()),
                "depth": str(example.depth_path.resolve()),
                "instruction": str(example.instruction_path.resolve()),
                "camera": str(example.camera_path.resolve()) if example.camera_path else None,
            },
        }
        try:
            rgb = Image.open(example.rgb_path).convert("RGB")
            depth = np.load(example.depth_path, allow_pickle=False)
            depth_report = validate_depth_map(
                depth,
                expected_shape=(rgb.height, rgb.width),
            )
            if not depth_report.valid:
                raise ValueError("; ".join(depth_report.errors))
            depth = depth.astype(np.float32, copy=False)
            camera = _load_camera(example.camera_path)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            result = predictor.predict(
                rgb,
                depth,
                sample_record["instruction"],
                max_new_tokens=max_new_tokens,
            )
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_started
            (sample_dir / "raw_output.txt").write_text(result["raw_output"], encoding="utf-8")

            waypoints = [[float(value) for value in point] for point in result["waypoints"]]
            actions = list(result["actions"])
            output_report = validate_waypoints(waypoints, actions)
            prediction = {
                "schema_version": 1,
                "sample": example.name,
                "instruction": sample_record["instruction"],
                "raw_output": result["raw_output"],
                "waypoints_uvd": waypoints,
                "coordinate_convention": "normalized_uv_0_1000_depth_m",
                "actions": actions,
                "input_validation": depth_report.to_dict(),
                "output_validation": output_report.to_dict(),
                "timing": {"inference_seconds": inference_seconds},
                "gpu": _gpu_snapshot(torch),
                "camera": camera,
                "provenance": provenance or {},
            }
            _write_json(sample_dir / "raw_prediction.json", prediction)
            _write_waypoints_csv(sample_dir / "waypoints_uvd.csv", waypoints, actions)
            save_trajectory_overlay(
                result["rgb_resized"],
                waypoints,
                actions,
                sample_dir / "trajectory_overlay.png",
            )

            if camera and camera.get("intrinsics") and waypoints:
                resized_rgb = np.asarray(result["rgb_resized"])
                xyz = normalized_uvd_to_camera_xyz(
                    np.asarray(waypoints),
                    np.asarray(camera["intrinsics"]),
                    image_width=resized_rgb.shape[1],
                    image_height=resized_rgb.shape[0],
                )
                _write_xyz_csv(sample_dir / "waypoints_camera_xyz.csv", xyz)

            sample_record.update(
                {
                    "status": "ok" if output_report.valid else "invalid_output",
                    "inference_seconds": inference_seconds,
                    "output_validation": output_report.to_dict(),
                    "artifacts": {
                        "prediction_json": str(sample_dir / "raw_prediction.json"),
                        "waypoints_uvd_csv": str(sample_dir / "waypoints_uvd.csv"),
                        "waypoints_camera_xyz_csv": (
                            str(sample_dir / "waypoints_camera_xyz.csv")
                            if (sample_dir / "waypoints_camera_xyz.csv").is_file()
                            else None
                        ),
                        "overlay_png": str(sample_dir / "trajectory_overlay.png"),
                    },
                }
            )
        except Exception as exc:
            sample_record.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _write_json(sample_dir / "error.json", sample_record)
            run_summary["samples"].append(sample_record)
            _write_json(destination / "run_summary.json", run_summary)
            if not continue_on_error:
                raise
            continue

        run_summary["samples"].append(sample_record)
        _write_json(destination / "run_summary.json", run_summary)

    run_summary["all_valid"] = all(sample["status"] == "ok" for sample in run_summary["samples"])
    run_summary["completed_count"] = sum(
        sample["status"] == "ok" for sample in run_summary["samples"]
    )
    run_summary["requested_count"] = len(examples)
    _write_json(destination / "run_summary.json", run_summary)
    return run_summary
