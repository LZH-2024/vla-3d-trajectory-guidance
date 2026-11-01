#!/usr/bin/env python3
"""Capture and validate one calibrated LIBERO RGB-D observation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_guidance.libero_probe import (
    json_ready,
    prepare_libero_config,
    save_depth_preview,
    sha256_file,
    squeeze_depth_map,
    transform_points,
    unproject_depth_sample,
    validate_camera_calibration,
    validate_depth_buffer,
    write_json,
)
from trajectory_guidance.libero_assets import download_pinned_libero_assets
from trajectory_guidance.validation import validate_depth_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suite")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--init-state-id", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (args.config or project_root / "configs" / "libero_probe_colab.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    libero_config = config["libero"]
    probe_config = config["probe"]

    expected_python = (Path(libero_config["env_dir"]) / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: got {sys.executable}, expected {expected_python}")
    if os.environ.get("MUJOCO_GL") != libero_config["render_backend"]:
        raise RuntimeError(
            f"MUJOCO_GL must be explicitly set to {libero_config['render_backend']!r}, "
            f"got {os.environ.get('MUJOCO_GL')!r}"
        )
    os.environ["MPLBACKEND"] = libero_config["matplotlib_backend"]

    suite_name = args.suite or probe_config["suite"]
    task_id = probe_config["task_id"] if args.task_id is None else args.task_id
    init_state_id = (
        probe_config["init_state_id"] if args.init_state_id is None else args.init_state_id
    )
    seed = probe_config["seed"] if args.seed is None else args.seed
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir
        or Path(probe_config["output_root"])
        / f"{suite_name}_task{task_id:02d}_init{init_state_id:03d}_{timestamp}"
    ).resolve()
    ensure_empty_output(output_dir)

    os.environ["LIBERO_CONFIG_PATH"] = str(Path(libero_config["config_dir"]).resolve())
    libero_path_config = prepare_libero_config(
        libero_config["config_dir"], libero_config["datasets_dir"]
    )
    assets_dir, assets_revision = download_pinned_libero_assets(libero_config)

    # Imports below this point may initialize MuJoCo / EGL. The environment
    # variables and non-interactive LIBERO path config are already in place.
    import robosuite.macros as robosuite_macros
    import robosuite.utils.transform_utils as transform_utils

    robosuite_macros.IMAGE_CONVENTION = libero_config["image_convention"]
    if robosuite_macros.IMAGE_CONVENTION != "opencv":
        raise RuntimeError("the geometry probe requires robosuite IMAGE_CONVENTION='opencv'")

    from lerobot.envs.libero import get_libero_dummy_action, get_task_init_states
    from libero.libero import benchmark, get_libero_path
    import libero.libero as libero_root
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    # hf-libero otherwise downloads the moving HEAD of the asset dataset.
    # The module cache is pointed at the exact snapshot downloaded above.
    libero_root._assets_path_cache = str(assets_dir)

    suites = benchmark.get_benchmark_dict()
    if suite_name not in suites:
        raise ValueError(f"unknown LIBERO suite {suite_name!r}; available: {sorted(suites)}")
    task_suite = suites[suite_name]()
    if task_id < 0 or task_id >= len(task_suite.tasks):
        raise ValueError(f"task_id {task_id} is outside [0, {len(task_suite.tasks) - 1}]")
    task = task_suite.get_task(task_id)
    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    ).resolve()
    init_state_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / Path(task.init_states_file).name
    ).resolve()
    init_states = get_task_init_states(task_suite, task_id)
    if init_state_id < 0 or init_state_id >= len(init_states):
        raise ValueError(
            f"init_state_id {init_state_id} is outside [0, {len(init_states) - 1}]"
        )

    width = int(probe_config["image_width"])
    height = int(probe_config["image_height"])
    cameras = list(probe_config["camera_names"])
    min_depth_m, max_depth_m = map(float, probe_config["accepted_depth_range_m"])
    dummy_action = np.asarray(get_libero_dummy_action(), dtype=np.float64)
    env = None
    created_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_names=cameras,
            camera_heights=height,
            camera_widths=width,
            camera_depths=True,
            control_freq=int(probe_config["control_frequency_hz"]),
            hard_reset=True,
        )
        env.seed(seed)
        reset_started = time.perf_counter()
        observation = env.reset()
        observation = env.set_init_state(init_states[init_state_id])
        for _ in range(int(probe_config["settle_steps"])):
            observation, _, _, _ = env.step(dummy_action)
        reset_and_settle_seconds = time.perf_counter() - reset_started

        eef_position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        eef_quaternion_xyzw = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
        if eef_position.shape != (3,) or not np.isfinite(eef_position).all():
            raise RuntimeError(f"invalid robot0_eef_pos: {eef_position}")
        if eef_quaternion_xyzw.shape != (4,) or not np.isfinite(eef_quaternion_xyzw).all():
            raise RuntimeError(f"invalid robot0_eef_quat: {eef_quaternion_xyzw}")
        t_world_eef = transform_utils.pose2mat((eef_position, eef_quaternion_xyzw))

        camera_results: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {
            "robot0_eef_pos": eef_position,
            "robot0_eef_quat_xyzw": eef_quaternion_xyzw,
            "T_world_eef": t_world_eef,
        }
        all_valid = True
        for camera_name in cameras:
            image_key = f"{camera_name}_image"
            depth_key = f"{camera_name}_depth"
            if image_key not in observation or depth_key not in observation:
                raise RuntimeError(
                    f"missing RGB-D keys for {camera_name}: available={sorted(observation)}"
                )
            rgb = np.asarray(observation[image_key])
            raw_depth = squeeze_depth_map(np.asarray(observation[depth_key]))
            if rgb.shape != (height, width, 3):
                raise RuntimeError(f"{image_key} has shape {rgb.shape}, expected {(height, width, 3)}")
            if rgb.dtype != np.uint8:
                raise RuntimeError(f"{image_key} has dtype {rgb.dtype}, expected uint8")
            if raw_depth.shape != (height, width):
                raise RuntimeError(
                    f"{depth_key} has shape {raw_depth.shape}, expected {(height, width)}"
                )

            raw_report = validate_depth_buffer(raw_depth)
            metric_depth = np.asarray(
                squeeze_depth_map(get_real_depth_map(env.sim, raw_depth)), dtype=np.float32
            )
            metric_report = validate_depth_map(
                metric_depth,
                expected_shape=(height, width),
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
            intrinsics = get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
            t_world_camera = get_camera_extrinsic_matrix(env.sim, camera_name)
            t_camera_world = np.linalg.inv(t_world_camera)
            calibration_report = validate_camera_calibration(
                intrinsics,
                t_world_camera,
                image_width=width,
                image_height=height,
            )

            pixels_uv, camera_xyz_m = unproject_depth_sample(
                metric_depth,
                intrinsics,
                stride=int(probe_config["pointcloud_stride"]),
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
            world_xyz_m = transform_points(camera_xyz_m, t_world_camera)
            sample_rgb = rgb[pixels_uv[:, 1].astype(int), pixels_uv[:, 0].astype(int)]
            if len(camera_xyz_m) == 0:
                raise RuntimeError(f"{camera_name} produced no valid point-cloud samples")

            prefix = output_dir / camera_name
            rgb_path = Path(f"{prefix}_rgb.png")
            raw_depth_path = Path(f"{prefix}_depth_buffer.npy")
            metric_depth_path = Path(f"{prefix}_depth_m.npy")
            preview_path = Path(f"{prefix}_depth_preview.png")
            calibration_path = Path(f"{prefix}_calibration.json")
            pointcloud_path = Path(f"{prefix}_pointcloud_sample.npz")
            Image.fromarray(rgb, mode="RGB").save(rgb_path)
            np.save(raw_depth_path, raw_depth, allow_pickle=False)
            np.save(metric_depth_path, metric_depth, allow_pickle=False)
            save_depth_preview(metric_depth, preview_path)
            np.savez_compressed(
                pointcloud_path,
                pixels_uv=pixels_uv,
                camera_xyz_m=camera_xyz_m,
                world_xyz_m=world_xyz_m,
                rgb=sample_rgb,
            )

            extent = float(env.sim.model.stat.extent)
            near_m = float(env.sim.model.vis.map.znear * extent)
            far_m = float(env.sim.model.vis.map.zfar * extent)
            calibration_payload = {
                "camera_name": camera_name,
                "image_size": {"width": width, "height": height},
                "image_convention": "opencv_top_left_u_right_v_down",
                "depth_buffer": {
                    "encoding": "mujoco_nonlinear_normalized_0_1",
                    "near_m": near_m,
                    "far_m": far_m,
                    "conversion": "robosuite.utils.camera_utils.get_real_depth_map",
                    "metric_depth_semantics": "camera_z_depth_m",
                },
                "camera_frame": "opencv_camera_x_right_y_down_z_forward",
                "world_frame": "libero_mujoco_world",
                "intrinsics_K": intrinsics,
                "T_world_camera": t_world_camera,
                "T_camera_world": t_camera_world,
                "extrinsics_convention": "T_target_source; p_world=T_world_camera@p_camera",
                "validation": calibration_report.to_dict(),
            }
            write_json(calibration_path, calibration_payload)

            valid = raw_report.valid and metric_report.valid and calibration_report.valid
            all_valid = all_valid and valid
            camera_results[camera_name] = {
                "valid": valid,
                "rgb": {
                    "shape": list(rgb.shape),
                    "dtype": str(rgb.dtype),
                    "path": str(rgb_path),
                    "sha256": sha256_file(rgb_path),
                },
                "depth_buffer_validation": raw_report.to_dict(),
                "metric_depth_validation": metric_report.to_dict(),
                "calibration_validation": calibration_report.to_dict(),
                "pointcloud_sample_count": int(len(camera_xyz_m)),
                "artifacts": {
                    "depth_buffer_npy": str(raw_depth_path),
                    "depth_m_npy": str(metric_depth_path),
                    "depth_preview_png": str(preview_path),
                    "calibration_json": str(calibration_path),
                    "pointcloud_sample_npz": str(pointcloud_path),
                },
            }
            arrays[f"{camera_name}_rgb"] = rgb
            arrays[f"{camera_name}_depth_buffer"] = raw_depth
            arrays[f"{camera_name}_depth_m"] = metric_depth
            arrays[f"{camera_name}_K"] = intrinsics
            arrays[f"{camera_name}_T_world_camera"] = t_world_camera

        arrays_path = output_dir / "observation_arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        robot_state_path = write_json(
            output_dir / "robot_state.json",
            {
                "frame": "libero_mujoco_world",
                "robot0_eef_pos_m": eef_position,
                "robot0_eef_quat_xyzw": eef_quaternion_xyzw,
                "T_world_eef": t_world_eef,
                "source": "robosuite robot0_eef_pos / robot0_eef_quat observables",
            },
        )

        try:
            nvidia_smi = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.used",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            nvidia_smi = f"unavailable: {exc}"

        manifest = {
            "schema_version": 1,
            "created_at_utc": created_at,
            "valid": all_valid,
            "scope": "single_frame_libero_observation_probe_no_planner_no_policy",
            "task": {
                "suite": suite_name,
                "task_id": task_id,
                "task_name": task.name,
                "language_instruction": task.language,
                "bddl_file": str(bddl_path),
                "bddl_sha256": sha256_file(bddl_path),
                "init_state_id": init_state_id,
                "init_state_file": str(init_state_path),
                "init_state_sha256": sha256_file(init_state_path),
                "seed": seed,
                "settle_steps": int(probe_config["settle_steps"]),
                "dummy_action": dummy_action,
            },
            "coordinate_conventions": {
                "image": "opencv_top_left_u_right_v_down",
                "metric_depth": "camera_z_depth_m",
                "camera": "x_right_y_down_z_forward",
                "world": "libero_mujoco_world",
                "extrinsics": "T_target_source",
                "eef_quaternion": "xyzw",
            },
            "provenance": {
                "lerobot_repo_commit": libero_config["repo_commit"],
                "hf_libero_version": package_version("hf-libero"),
                "hf_libero_wheel_sha256": libero_config["hf_libero_wheel_sha256"],
                "robosuite_version": package_version("robosuite"),
                "mujoco_version": package_version("mujoco"),
                "assets_repo_id": libero_config["assets_repo_id"],
                "assets_revision": assets_revision,
                "libero_path_config": str(libero_path_config),
                "MUJOCO_GL": os.environ["MUJOCO_GL"],
                "MPLBACKEND": os.environ["MPLBACKEND"],
                "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            },
            "timing_seconds": {
                "reset_init_and_settle": reset_and_settle_seconds,
                "total": time.perf_counter() - started,
            },
            "nvidia_smi": nvidia_smi,
            "robot_state_json": str(robot_state_path),
            "observation_arrays_npz": str(arrays_path),
            "cameras": camera_results,
        }
        manifest_path = write_json(output_dir / "observation_manifest.json", manifest)
        print(
            json.dumps(
                json_ready(manifest),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        print(f"\nManifest: {manifest_path}")
        return 0 if all_valid else 2
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
