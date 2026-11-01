"""Tiny CPU model that exercises the SmolVLA adapter contract without LeRobot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from trajectory_guidance.trajectory_conditioning import (
    TRAJECTORY_EEF_KEY,
    TRAJECTORY_MASK_KEY,
    TRAJECTORY_POINTS_KEY,
)


@dataclass(frozen=True)
class TinySmolVLAConfig:
    compile_model: bool = False
    chunk_size: int = 4
    action_dim: int = 7
    expert_hidden_size: int = 16


class TinyActionExpert(nn.Module):
    """Only mirrors the named SmolVLA injection point and output shape."""

    def __init__(self, config: TinySmolVLAConfig):
        super().__init__()
        self.action_time_mlp_out = nn.Linear(
            config.expert_hidden_size, config.expert_hidden_size
        )
        self.action_out_proj = nn.Linear(config.expert_hidden_size, config.action_dim)

    def forward(self, expert_tokens: Tensor) -> Tensor:
        hidden = torch.tanh(self.action_time_mlp_out(expert_tokens))
        return self.action_out_proj(hidden)


class TinySmolVLAPolicy(nn.Module):
    """A deterministic shape/gradient double; it is not the real SmolVLA."""

    def __init__(self, config: TinySmolVLAConfig | None = None):
        super().__init__()
        self.config = config or TinySmolVLAConfig()
        self.model = TinyActionExpert(self.config)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        prediction = self.model(batch["expert_tokens"])
        target = batch["target_actions"]
        if prediction.shape != target.shape:
            raise ValueError(
                f"mock target shape {tuple(target.shape)} does not match {tuple(prediction.shape)}"
            )
        loss = torch.mean((prediction - target) ** 2)
        return loss, {"mock_action_mse": float(loss.detach())}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        return self.model(batch["expert_tokens"])

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any]) -> Tensor:
        return self.predict_action_chunk(batch)[:, 0]

    def reset(self) -> None:
        return None

    def get_optim_params(self):
        return self.parameters()


def make_tiny_batch(
    *,
    batch_size: int = 2,
    config: TinySmolVLAConfig | None = None,
    num_waypoints: int = 8,
    include_trajectory: bool = True,
    anomalous: bool = False,
    seed: int = 0,
) -> dict[str, Tensor]:
    """Build a repeatable CPU batch for E0/E1/E2 tests."""

    model_config = config or TinySmolVLAConfig()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch: dict[str, Tensor] = {
        "expert_tokens": torch.randn(
            batch_size,
            model_config.chunk_size,
            model_config.expert_hidden_size,
            generator=generator,
        ),
        "target_actions": torch.randn(
            batch_size,
            model_config.chunk_size,
            model_config.action_dim,
            generator=generator,
        ),
    }
    if include_trajectory:
        eef = torch.zeros(batch_size, 3, dtype=torch.float32)
        offsets = torch.linspace(0.0, 0.21, num_waypoints, dtype=torch.float32)
        points = torch.zeros(batch_size, num_waypoints, 3, dtype=torch.float32)
        points[:, :, 0] = offsets
        points[:, :, 2] = 0.4
        if anomalous:
            points[:, -1] = torch.tensor([4.0, 4.0, 4.0])
        batch[TRAJECTORY_POINTS_KEY] = points
        batch[TRAJECTORY_MASK_KEY] = torch.ones(
            batch_size, num_waypoints, dtype=torch.bool
        )
        batch[TRAJECTORY_EEF_KEY] = eef
    return batch
