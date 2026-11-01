"""Small trajectory-condition modules for a wrapped SmolVLA action expert.

The implementation intentionally does not import LeRobot.  It wraps an already
constructed policy instance and attaches one forward hook to the pinned
SmolVLA ``action_time_mlp_out`` projection.  The original action size, flow
matching target, loss computation, VLM, and Action Expert remain unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
from torch import Tensor, nn


TRAJECTORY_POINTS_KEY = "trajectory.local_waypoints_xyz_m"
TRAJECTORY_MASK_KEY = "trajectory.local_waypoints_valid"
TRAJECTORY_EEF_KEY = "trajectory.current_eef_xyz_m"
TRAJECTORY_SOURCE_CODE_KEY = "trajectory.source_code"


@dataclass(frozen=True)
class TrajectoryConditioningConfig:
    enabled: bool = True
    num_waypoints: int = 8
    encoder_hidden_dim: int = 128
    coordinate_scale_m: tuple[float, float, float] = (0.5, 0.5, 0.5)
    injection_scale: float = 1.0
    gate_mode: str = "none"
    gate_hidden_dim: int = 16
    workspace_min_m: tuple[float, float, float] = (-1.0, -1.0, -0.2)
    workspace_max_m: tuple[float, float, float] = (1.0, 1.0, 1.5)
    max_adjacent_jump_m: float = 0.35
    max_eef_deviation_m: float = 0.50

    @classmethod
    def from_mapping(cls, source: dict[str, Any]) -> "TrajectoryConditioningConfig":
        """Read the reviewed adapter fields from an experiment mapping."""

        return cls(
            enabled=bool(source["enabled"]),
            num_waypoints=int(source["num_waypoints"]),
            encoder_hidden_dim=int(source["encoder_hidden_dim"]),
            coordinate_scale_m=tuple(float(value) for value in source["coordinate_scale_m"]),
            injection_scale=float(source["injection_scale"]),
            gate_mode=str(source["gate_mode"]),
            gate_hidden_dim=int(source["gate_hidden_dim"]),
            workspace_min_m=tuple(float(value) for value in source["workspace_min_m"]),
            workspace_max_m=tuple(float(value) for value in source["workspace_max_m"]),
            max_adjacent_jump_m=float(source["max_adjacent_jump_m"]),
            max_eef_deviation_m=float(source["max_eef_deviation_m"]),
        )

    def validate(self) -> None:
        if self.num_waypoints <= 0:
            raise ValueError("num_waypoints must be positive")
        if self.encoder_hidden_dim <= 0 or self.gate_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if any(value <= 0.0 for value in self.coordinate_scale_m):
            raise ValueError("coordinate_scale_m must be positive")
        if self.injection_scale < 0.0:
            raise ValueError("injection_scale must be non-negative")
        if self.gate_mode not in {"none", "geometry"}:
            raise ValueError("gate_mode must be 'none' or 'geometry'")
        if any(
            low >= high
            for low, high in zip(
                self.workspace_min_m, self.workspace_max_m, strict=True
            )
        ):
            raise ValueError("workspace_min_m must be smaller than workspace_max_m")
        if self.max_adjacent_jump_m <= 0.0 or self.max_eef_deviation_m <= 0.0:
            raise ValueError("geometry thresholds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_geometry_anomaly_features(
    waypoints_xyz_m: Tensor,
    valid_mask: Tensor,
    current_eef_xyz_m: Tensor,
    config: TrajectoryConditioningConfig,
) -> Tensor:
    """Compute three bounded anomaly features; they are not probabilities.

    Output columns are workspace violation fraction, adjacent-jump severity,
    and current-EEF deviation severity.  Values lie in [0, 1].
    """

    if waypoints_xyz_m.ndim != 3 or waypoints_xyz_m.shape[-1] != 3:
        raise ValueError("waypoints_xyz_m must have shape (B, K, 3)")
    if valid_mask.shape != waypoints_xyz_m.shape[:2]:
        raise ValueError("valid_mask must have shape (B, K)")
    if current_eef_xyz_m.shape != (waypoints_xyz_m.shape[0], 3):
        raise ValueError("current_eef_xyz_m must have shape (B, 3)")
    mask = valid_mask.to(dtype=torch.bool)
    dtype = waypoints_xyz_m.dtype
    device = waypoints_xyz_m.device
    workspace_min = torch.tensor(
        config.workspace_min_m, dtype=dtype, device=device
    )
    workspace_max = torch.tensor(
        config.workspace_max_m, dtype=dtype, device=device
    )
    outside = (
        (waypoints_xyz_m < workspace_min[None, None, :])
        | (waypoints_xyz_m > workspace_max[None, None, :])
    ).any(dim=-1)
    valid_count = mask.sum(dim=1).clamp_min(1)
    outside_fraction = (outside & mask).sum(dim=1).to(dtype) / valid_count.to(dtype)

    if waypoints_xyz_m.shape[1] > 1:
        jumps = torch.linalg.vector_norm(
            waypoints_xyz_m[:, 1:] - waypoints_xyz_m[:, :-1], dim=-1
        )
        jump_mask = mask[:, 1:] & mask[:, :-1]
        jump_excess = torch.relu(jumps / config.max_adjacent_jump_m - 1.0)
        jump_excess = torch.where(jump_mask, jump_excess, torch.zeros_like(jump_excess))
        max_jump_excess = jump_excess.amax(dim=1)
        jump_severity = 1.0 - torch.exp(-max_jump_excess)
    else:
        jump_severity = torch.zeros(waypoints_xyz_m.shape[0], dtype=dtype, device=device)

    distances = torch.linalg.vector_norm(
        waypoints_xyz_m - current_eef_xyz_m[:, None, :], dim=-1
    )
    infinity = torch.full_like(distances, torch.inf)
    nearest = torch.where(mask, distances, infinity).amin(dim=1)
    has_valid = mask.any(dim=1)
    nearest = torch.where(has_valid, nearest, torch.zeros_like(nearest))
    deviation_excess = torch.relu(nearest / config.max_eef_deviation_m - 1.0)
    deviation_severity = 1.0 - torch.exp(-deviation_excess)
    return torch.stack(
        [outside_fraction, jump_severity, deviation_severity], dim=-1
    ).clamp(0.0, 1.0)


class GeometryValidityGate(nn.Module):
    """Optional gain for trajectory features, not a calibrated correctness score."""

    def __init__(self, config: TrajectoryConditioningConfig):
        super().__init__()
        self.config = config
        self.residual = nn.Sequential(
            nn.Linear(3, config.gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.gate_hidden_dim, 1),
        )
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        waypoints_xyz_m: Tensor,
        valid_mask: Tensor,
        current_eef_xyz_m: Tensor,
    ) -> tuple[Tensor, Tensor]:
        anomaly_features = compute_geometry_anomaly_features(
            waypoints_xyz_m,
            valid_mask,
            current_eef_xyz_m,
            self.config,
        )
        geometric_prior = torch.exp(-anomaly_features.sum(dim=-1, keepdim=True))
        learned_residual = 2.0 * torch.sigmoid(self.residual(anomaly_features))
        gain = (geometric_prior * learned_residual).clamp(0.0, 1.0)
        gain = gain * valid_mask.any(dim=1, keepdim=True).to(gain.dtype)
        return gain, anomaly_features


class TrajectoryConditionEncoder(nn.Module):
    """Encode a fixed local window into one Action-Expert condition vector."""

    def __init__(
        self,
        config: TrajectoryConditioningConfig,
        expert_hidden_size: int,
    ):
        super().__init__()
        config.validate()
        if expert_hidden_size <= 0:
            raise ValueError("expert_hidden_size must be positive")
        self.config = config
        input_dim = config.num_waypoints * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.encoder_hidden_dim, expert_hidden_size),
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        waypoints_xyz_m: Tensor,
        valid_mask: Tensor,
        current_eef_xyz_m: Tensor,
    ) -> Tensor:
        expected_points = (
            waypoints_xyz_m.shape[0],
            self.config.num_waypoints,
            3,
        )
        if tuple(waypoints_xyz_m.shape) != expected_points:
            raise ValueError(
                f"trajectory points must have shape {expected_points}, got {tuple(waypoints_xyz_m.shape)}"
            )
        if tuple(valid_mask.shape) != expected_points[:2]:
            raise ValueError("trajectory valid mask shape mismatch")
        if tuple(current_eef_xyz_m.shape) != (expected_points[0], 3):
            raise ValueError("current EEF shape mismatch")
        if not torch.isfinite(waypoints_xyz_m).all():
            raise ValueError("trajectory points contain NaN or infinity")
        if not torch.isfinite(current_eef_xyz_m).all():
            raise ValueError("current EEF contains NaN or infinity")

        mask = valid_mask.to(dtype=torch.bool)
        scale = torch.tensor(
            self.config.coordinate_scale_m,
            dtype=waypoints_xyz_m.dtype,
            device=waypoints_xyz_m.device,
        )
        relative = (waypoints_xyz_m - current_eef_xyz_m[:, None, :]) / scale
        relative = torch.where(mask[..., None], relative, torch.zeros_like(relative))
        features = torch.cat([relative, mask[..., None].to(relative.dtype)], dim=-1)
        encoded = self.mlp(features.flatten(start_dim=1))
        has_valid = mask.any(dim=1, keepdim=True).to(encoded.dtype)
        return encoded * has_valid * self.config.injection_scale


class SmolVLATrajectoryWrapper(nn.Module):
    """Add V1/V2 trajectory conditioning without editing LeRobot source files.

    The wrapper must be installed after loading the base SmolVLA policy.  It is
    intentionally rejected when ``torch.compile`` is enabled because dynamic
    hook state has not been validated with the pinned compiled path.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        config: TrajectoryConditioningConfig,
    ):
        super().__init__()
        config.validate()
        if not config.enabled:
            raise ValueError("do not wrap E0 when trajectory conditioning is disabled")
        base_config = getattr(base_policy, "config", None)
        if getattr(base_config, "compile_model", False):
            raise ValueError("trajectory wrapper requires SmolVLA compile_model=False")
        model = getattr(base_policy, "model", None)
        target = getattr(model, "action_time_mlp_out", None)
        hidden_size = getattr(target, "out_features", None)
        if target is None or hidden_size is None:
            raise TypeError(
                "base policy does not expose SmolVLA model.action_time_mlp_out.out_features"
            )

        self.base_policy = base_policy
        # LeRobot callers expect ``policy.config`` to remain the original
        # SmolVLA configuration.  Keep the adapter configuration separate.
        self.config = base_config
        self.trajectory_config = config
        self.condition_encoder = TrajectoryConditionEncoder(config, int(hidden_size))
        self.geometry_gate = (
            GeometryValidityGate(config) if config.gate_mode == "geometry" else None
        )
        try:
            target_device = next(target.parameters()).device
        except StopIteration as exc:
            raise TypeError("action_time_mlp_out has no parameters") from exc
        self.condition_encoder.to(device=target_device)
        if self.geometry_gate is not None:
            self.geometry_gate.to(device=target_device)
        self._active_condition: Tensor | None = None
        self._last_gate: Tensor | None = None
        self._last_anomaly_features: Tensor | None = None
        self._hook_handle = target.register_forward_hook(self._inject_condition_hook)

    def _inject_condition_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        if self._active_condition is None:
            return output
        if not isinstance(output, Tensor) or output.ndim != 3:
            raise RuntimeError("SmolVLA action_time_mlp_out output must have shape (B, T, H)")
        condition = self._active_condition.to(device=output.device, dtype=output.dtype)
        if condition.shape != (output.shape[0], output.shape[2]):
            raise RuntimeError(
                f"trajectory condition shape {tuple(condition.shape)} does not match "
                f"Action Expert output {tuple(output.shape)}"
            )
        return output + condition[:, None, :]

    @staticmethod
    def _reference_tensor(batch: dict[str, Any]) -> Tensor:
        for value in batch.values():
            if isinstance(value, Tensor) and value.ndim > 0:
                return value
        raise ValueError("batch contains no tensor from which to infer batch size/device")

    def _condition_from_batch(self, batch: dict[str, Any]) -> Tensor:
        reference = self._reference_tensor(batch)
        batch_size = reference.shape[0]
        points_value = batch.get(TRAJECTORY_POINTS_KEY)
        mask_value = batch.get(TRAJECTORY_MASK_KEY)
        eef_value = batch.get(TRAJECTORY_EEF_KEY)
        if (points_value is None) != (mask_value is None):
            raise ValueError("trajectory points and valid mask must be supplied together")
        if points_value is None:
            points = torch.zeros(
                batch_size,
                self.trajectory_config.num_waypoints,
                3,
                dtype=torch.float32,
                device=reference.device,
            )
            mask = torch.zeros(
                batch_size,
                self.trajectory_config.num_waypoints,
                dtype=torch.bool,
                device=reference.device,
            )
            eef = torch.zeros(batch_size, 3, dtype=torch.float32, device=reference.device)
        else:
            points = torch.as_tensor(points_value, device=reference.device, dtype=torch.float32)
            mask = torch.as_tensor(mask_value, device=reference.device, dtype=torch.bool)
            if eef_value is None:
                if mask.any():
                    raise ValueError("current EEF is required when valid trajectory points exist")
                eef = torch.zeros(batch_size, 3, dtype=torch.float32, device=reference.device)
            else:
                eef = torch.as_tensor(eef_value, device=reference.device, dtype=torch.float32)

        encoded = self.condition_encoder(points, mask, eef)
        if self.geometry_gate is not None:
            gate, anomaly_features = self.geometry_gate(points, mask, eef)
            encoded = encoded * gate
            self._last_gate = gate.detach()
            self._last_anomaly_features = anomaly_features.detach()
        else:
            self._last_gate = mask.any(dim=1, keepdim=True).to(encoded.dtype).detach()
            self._last_anomaly_features = None
        return encoded

    @contextmanager
    def _activate_batch_condition(self, batch: dict[str, Any]) -> Iterator[None]:
        if self._active_condition is not None:
            raise RuntimeError("trajectory wrapper does not support re-entrant policy calls")
        self._active_condition = self._condition_from_batch(batch)
        try:
            yield
        finally:
            self._active_condition = None

    def forward(self, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        with self._activate_batch_condition(batch):
            return self.base_policy(batch, *args, **kwargs)

    def predict_action_chunk(
        self, batch: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        with self._activate_batch_condition(batch):
            return self.base_policy.predict_action_chunk(batch, *args, **kwargs)

    def select_action(self, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        with self._activate_batch_condition(batch):
            return self.base_policy.select_action(batch, *args, **kwargs)

    def reset(self) -> None:
        reset = getattr(self.base_policy, "reset", None)
        if reset is not None:
            reset()

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.condition_encoder.parameters()
        if self.geometry_gate is not None:
            yield from self.geometry_gate.parameters()

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        """Return base and adapter parameters for LeRobot-style optimizers."""

        yield from self.parameters()

    def adapter_state_dict(self) -> dict[str, dict[str, Tensor]]:
        state = {"condition_encoder": self.condition_encoder.state_dict()}
        if self.geometry_gate is not None:
            state["geometry_gate"] = self.geometry_gate.state_dict()
        return state

    def load_adapter_state_dict(
        self, state: dict[str, dict[str, Tensor]], *, strict: bool = True
    ) -> None:
        self.condition_encoder.load_state_dict(state["condition_encoder"], strict=strict)
        if self.geometry_gate is not None:
            if "geometry_gate" not in state:
                raise KeyError("geometry_gate state is missing")
            self.geometry_gate.load_state_dict(state["geometry_gate"], strict=strict)

    def diagnostics(self) -> dict[str, Tensor | str | None]:
        return {
            "gate_mode": self.trajectory_config.gate_mode,
            "gate_gain_not_probability": self._last_gate,
            "geometry_anomaly_features": self._last_anomaly_features,
        }

    def close(self) -> None:
        self._hook_handle.remove()
