"""Versioned, environment-neutral storage for cached 3-D trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from trajectory_guidance.libero_probe import sha256_file
from trajectory_guidance.validation import ValidationReport


TRAJECTORY_SCHEMA_VERSION = 1
ALLOWED_SOURCES = {"prediction", "mock", "oracle", "other"}
ALLOWED_GRIPPER_EVENTS = {"none", "open", "close"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class EpisodeIdentity:
    dataset: str
    suite: str
    task_id: str
    episode_id: str
    init_state_id: int
    observation_index: int
    observation_timestamp_s: float | None = None


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    version: str
    code_revision: str


@dataclass(frozen=True)
class CameraCalibration:
    camera_name: str
    image_width: int
    image_height: int
    intrinsics_k: tuple[tuple[float, float, float], ...]
    t_world_camera: tuple[tuple[float, float, float, float], ...]
    image_convention: str = "opencv_top_left_u_right_v_down"
    camera_frame: str = "opencv_camera_x_right_y_down_z_forward"
    world_frame: str = "libero_mujoco_world"
    extrinsics_convention: str = "p_world=T_world_camera@p_camera"
    depth_semantics: str = "camera_z_depth_m"
    depth_unit: str = "m"

    @classmethod
    def from_arrays(
        cls,
        *,
        camera_name: str,
        image_width: int,
        image_height: int,
        intrinsics_k: np.ndarray,
        t_world_camera: np.ndarray,
        **kwargs: Any,
    ) -> "CameraCalibration":
        k = np.asarray(intrinsics_k, dtype=np.float64)
        transform = np.asarray(t_world_camera, dtype=np.float64)
        return cls(
            camera_name=camera_name,
            image_width=int(image_width),
            image_height=int(image_height),
            intrinsics_k=tuple(tuple(float(value) for value in row) for row in k),
            t_world_camera=tuple(
                tuple(float(value) for value in row) for row in transform
            ),
            **kwargs,
        )

    def intrinsics_array(self) -> np.ndarray:
        return np.asarray(self.intrinsics_k, dtype=np.float64)

    def t_world_camera_array(self) -> np.ndarray:
        return np.asarray(self.t_world_camera, dtype=np.float64)


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    source: str
    episode: EpisodeIdentity
    model: ModelIdentity
    instruction: str
    input_image_sha256: str
    coordinate_frame: str
    waypoints_xyz_m: np.ndarray
    valid_mask: np.ndarray
    gripper_events: tuple[str, ...]
    calibration: CameraCalibration
    available_at_observation_index: int
    uses_future_expert_information: bool = False
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = TRAJECTORY_SCHEMA_VERSION

    @property
    def waypoint_order(self) -> np.ndarray:
        return np.arange(len(self.waypoints_xyz_m), dtype=np.int64)

    def validate(self) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        points = np.asarray(self.waypoints_xyz_m)
        mask = np.asarray(self.valid_mask)

        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            errors.append(
                f"unsupported schema version {self.schema_version}; "
                f"expected {TRAJECTORY_SCHEMA_VERSION}"
            )
        if not self.trajectory_id or any(char in self.trajectory_id for char in "/\\"):
            errors.append("trajectory_id must be a non-empty path-safe identifier")
        if self.source not in ALLOWED_SOURCES:
            errors.append(f"unsupported trajectory source: {self.source!r}")
        if not self.instruction.strip():
            errors.append("language instruction is empty")
        if not _SHA256_PATTERN.fullmatch(self.input_image_sha256):
            errors.append("input_image_sha256 must be a lowercase 64-character SHA-256")
        if not self.coordinate_frame:
            errors.append("coordinate_frame is empty")
        if points.ndim != 2 or points.shape[1:] != (3,):
            errors.append(f"waypoints_xyz_m must have shape (N, 3), got {points.shape}")
        elif len(points) == 0:
            errors.append("trajectory must contain at least one waypoint")
        elif not np.isfinite(points).all():
            errors.append("trajectory contains NaN or infinity")
        if mask.shape != (len(points),):
            errors.append(f"valid_mask must have shape ({len(points)},), got {mask.shape}")
        elif mask.dtype != np.bool_:
            errors.append(f"valid_mask must use bool dtype, got {mask.dtype}")
        elif not mask.any():
            errors.append("trajectory has no valid waypoint")
        if len(self.gripper_events) != len(points):
            errors.append("gripper_events length does not match waypoints")
        invalid_events = sorted(set(self.gripper_events) - ALLOWED_GRIPPER_EVENTS)
        if invalid_events:
            errors.append(f"unsupported gripper events: {invalid_events}")
        if self.episode.init_state_id < 0 or self.episode.observation_index < 0:
            errors.append("init_state_id and observation_index must be non-negative")
        if self.available_at_observation_index < self.episode.observation_index:
            errors.append(
                "available_at_observation_index cannot precede the source observation"
            )
        if self.source == "oracle" and not self.uses_future_expert_information:
            warnings.append(
                "oracle source is marked causal; verify that it does not contain future expert information"
            )
        if self.uses_future_expert_information and self.source != "oracle":
            errors.append(
                "future expert information must use source='oracle' and remain diagnostic-only"
            )

        k = self.calibration.intrinsics_array()
        transform = self.calibration.t_world_camera_array()
        if k.shape != (3, 3) or not np.isfinite(k).all():
            errors.append("camera intrinsics must be a finite 3x3 matrix")
        elif k[0, 0] <= 0 or k[1, 1] <= 0:
            errors.append("camera intrinsics must have positive fx and fy")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            errors.append("T_world_camera must be a finite 4x4 matrix")
        elif not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            errors.append("T_world_camera has an invalid homogeneous last row")
        if self.calibration.image_width <= 0 or self.calibration.image_height <= 0:
            errors.append("calibration image dimensions must be positive")
        if self.calibration.depth_unit != "m":
            errors.append("trajectory cache depth unit must be meters")
        if self.source == "oracle":
            warnings.append(
                "oracle trajectories are diagnostic-only and are rejected by normal training loaders"
            )

        metrics = {
            "waypoint_count": int(len(points)),
            "valid_waypoint_count": int(mask.sum()) if mask.shape == (len(points),) else 0,
            "source": self.source,
            "coordinate_frame": self.coordinate_frame,
            "observation_index": self.episode.observation_index,
            "available_at_observation_index": self.available_at_observation_index,
            "uses_future_expert_information": self.uses_future_expert_information,
        }
        return ValidationReport(not errors, errors, warnings, metrics)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "created_at_utc": self.created_at_utc,
            "source": self.source,
            "episode": asdict(self.episode),
            "model": asdict(self.model),
            "instruction": self.instruction,
            "input_image_sha256": self.input_image_sha256,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_unit": "m",
            "waypoint_count": int(len(self.waypoints_xyz_m)),
            "available_at_observation_index": self.available_at_observation_index,
            "uses_future_expert_information": self.uses_future_expert_information,
            "calibration": asdict(self.calibration),
            "provenance": self.provenance,
            "causality_contract": {
                "condition_generated_from_observation_index": self.episode.observation_index,
                "condition_available_at_observation_index": self.available_at_observation_index,
                "expert_future_trajectory_allowed_as_normal_input": False,
                "uses_future_expert_information": self.uses_future_expert_information,
            },
        }


def _write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_waypoint_csv(path: Path, record: TrajectoryRecord) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["order", "x_m", "y_m", "z_m", "valid", "gripper_event", "coordinate_frame"]
        )
        for order, (point, valid, event) in enumerate(
            zip(
                record.waypoints_xyz_m,
                record.valid_mask,
                record.gripper_events,
                strict=True,
            )
        ):
            writer.writerow(
                [
                    order,
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(bool(valid)),
                    event,
                    record.coordinate_frame,
                ]
            )


def save_trajectory_record(record: TrajectoryRecord, destination: str | Path) -> Path:
    """Atomically save one record as JSON + CSV + NPZ + checksum manifest."""

    report = record.validate()
    if not report.valid:
        raise ValueError(f"invalid trajectory record: {report.errors}")
    output_dir = Path(destination).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"trajectory record directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        metadata_path = temporary_dir / "metadata.json"
        arrays_path = temporary_dir / "trajectory.npz"
        csv_path = temporary_dir / "waypoints.csv"
        _write_strict_json(metadata_path, record.metadata())
        np.savez_compressed(
            arrays_path,
            waypoints_xyz_m=np.asarray(record.waypoints_xyz_m, dtype=np.float64),
            valid_mask=np.asarray(record.valid_mask, dtype=np.bool_),
            waypoint_order=record.waypoint_order,
            gripper_events=np.asarray(record.gripper_events, dtype="U8"),
        )
        _write_waypoint_csv(csv_path, record)
        files = {
            name: sha256_file(temporary_dir / name)
            for name in ("metadata.json", "trajectory.npz", "waypoints.csv")
        }
        manifest = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "trajectory_id": record.trajectory_id,
            "hash_algorithm": "sha256",
            "files": files,
            "record_digest": hashlib.sha256(
                "".join(f"{name}:{digest}\n" for name, digest in sorted(files.items())).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
        _write_strict_json(temporary_dir / "manifest.json", manifest)
        temporary_dir.rename(output_dir)
    return output_dir


def _tuple_matrix(value: Iterable[Iterable[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def load_trajectory_record(
    source: str | Path,
    *,
    allow_oracle: bool = False,
    allow_future_expert_information: bool = False,
    allowed_sources: set[str] | None = None,
) -> TrajectoryRecord:
    """Load a record and verify every stored file before returning arrays."""

    root = Path(source).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"trajectory manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("trajectory manifest schema version mismatch")
    files = manifest.get("files", {})
    if set(files) != {"metadata.json", "trajectory.npz", "waypoints.csv"}:
        raise ValueError("trajectory manifest has an unexpected file set")
    for name, expected_hash in files.items():
        if not _SHA256_PATTERN.fullmatch(str(expected_hash)):
            raise ValueError(f"trajectory manifest has an invalid hash for {name}")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"trajectory cache file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"trajectory cache checksum mismatch for {name}")
    record_digest = hashlib.sha256(
        "".join(f"{name}:{digest}\n" for name, digest in sorted(files.items())).encode(
            "utf-8"
        )
    ).hexdigest()
    if record_digest != manifest.get("record_digest"):
        raise ValueError("trajectory manifest record digest mismatch")

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("trajectory_id") != manifest.get("trajectory_id"):
        raise ValueError("trajectory ID differs between manifest and metadata")
    with np.load(root / "trajectory.npz", allow_pickle=False) as arrays:
        waypoints = arrays["waypoints_xyz_m"].copy()
        valid_mask = arrays["valid_mask"].copy()
        waypoint_order = arrays["waypoint_order"].copy()
        events = tuple(str(item) for item in arrays["gripper_events"].tolist())
    expected_order = np.arange(len(waypoints), dtype=np.int64)
    if not np.array_equal(waypoint_order, expected_order):
        raise ValueError("stored waypoint_order is not contiguous from zero")
    if int(metadata.get("waypoint_count", -1)) != len(waypoints):
        raise ValueError("stored waypoint count differs between metadata and arrays")

    calibration_payload = metadata["calibration"]
    calibration = CameraCalibration(
        camera_name=calibration_payload["camera_name"],
        image_width=int(calibration_payload["image_width"]),
        image_height=int(calibration_payload["image_height"]),
        intrinsics_k=_tuple_matrix(calibration_payload["intrinsics_k"]),
        t_world_camera=_tuple_matrix(calibration_payload["t_world_camera"]),
        image_convention=calibration_payload["image_convention"],
        camera_frame=calibration_payload["camera_frame"],
        world_frame=calibration_payload["world_frame"],
        extrinsics_convention=calibration_payload["extrinsics_convention"],
        depth_semantics=calibration_payload["depth_semantics"],
        depth_unit=calibration_payload["depth_unit"],
    )
    record = TrajectoryRecord(
        schema_version=int(metadata["schema_version"]),
        trajectory_id=metadata["trajectory_id"],
        created_at_utc=metadata["created_at_utc"],
        source=metadata["source"],
        episode=EpisodeIdentity(**metadata["episode"]),
        model=ModelIdentity(**metadata["model"]),
        instruction=metadata["instruction"],
        input_image_sha256=metadata["input_image_sha256"],
        coordinate_frame=metadata["coordinate_frame"],
        waypoints_xyz_m=waypoints,
        valid_mask=valid_mask,
        gripper_events=events,
        calibration=calibration,
        available_at_observation_index=int(metadata["available_at_observation_index"]),
        uses_future_expert_information=bool(
            metadata.get("uses_future_expert_information", False)
        ),
        provenance=metadata.get("provenance", {}),
    )
    report = record.validate()
    if not report.valid:
        raise ValueError(f"loaded trajectory record is invalid: {report.errors}")
    if record.source == "oracle" and not allow_oracle:
        raise ValueError(
            "oracle trajectory rejected: normal training and inference must use causally available inputs"
        )
    if record.uses_future_expert_information and not allow_future_expert_information:
        raise ValueError(
            "future expert trajectory rejected: it is diagnostic-only and cannot be a normal condition"
        )
    if allowed_sources is not None and record.source not in allowed_sources:
        raise ValueError(
            f"trajectory source {record.source!r} is not in allowed sources {sorted(allowed_sources)}"
        )
    return record


class TrajectoryCache:
    """Small directory cache whose records are portable across virtual environments."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def save(self, record: TrajectoryRecord) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return save_trajectory_record(record, self.root / record.trajectory_id)

    def load(
        self,
        trajectory_id: str,
        *,
        allow_oracle: bool = False,
        allow_future_expert_information: bool = False,
    ) -> TrajectoryRecord:
        return load_trajectory_record(
            self.root / trajectory_id,
            allow_oracle=allow_oracle,
            allow_future_expert_information=allow_future_expert_information,
        )

    def record_directories(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            path.parent for path in self.root.glob("*/manifest.json") if path.is_file()
        )

    def load_all(
        self,
        *,
        allow_oracle: bool = False,
        allow_future_expert_information: bool = False,
    ) -> list[TrajectoryRecord]:
        return [
            load_trajectory_record(
                path,
                allow_oracle=allow_oracle,
                allow_future_expert_information=allow_future_expert_information,
            )
            for path in self.record_directories()
        ]
