"""Small resumable training-loop primitives used by CPU mocks and future GPU assembly."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainingProgress:
    step: int
    config_digest: str
    checkpoint_path: Path | None


def save_training_checkpoint(
    destination: str | Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config_digest: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": int(step),
        "config_digest": config_digest,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra or {},
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_training_checkpoint(
    source: str | Path,
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config_digest: str,
) -> TrainingProgress:
    path = Path(source).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("training checkpoint schema mismatch")
    if payload.get("config_digest") != expected_config_digest:
        raise ValueError("training checkpoint config digest mismatch")
    policy.load_state_dict(payload["policy_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return TrainingProgress(
        step=int(payload["step"]),
        config_digest=expected_config_digest,
        checkpoint_path=path,
    )


def _extract_loss(output: Any) -> tuple[Tensor, dict[str, Any]]:
    if isinstance(output, Tensor):
        return output, {}
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        metrics = output[1] if len(output) > 1 and isinstance(output[1], dict) else {}
        return output[0], metrics
    raise TypeError("policy forward must return a loss tensor or (loss, metrics)")


def _truncate_metrics_for_resume(path: Path, start_step: int) -> None:
    """Drop uncheckpointed tail records before resuming from an exact step."""

    if not path.is_file():
        return
    retained: list[str] = []
    seen_steps: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            step = int(record["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid metrics record at line {line_number}") from exc
        if step <= start_step:
            if step in seen_steps:
                raise ValueError(f"duplicate metrics record for step {step}")
            seen_steps.add(step)
            retained.append(json.dumps(record, ensure_ascii=False, allow_nan=False))
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".metrics.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write("".join(f"{line}\n" for line in retained))
    temporary.replace(path)


def run_training_steps(
    *,
    policy: nn.Module,
    batches: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    run_dir: str | Path,
    config_digest: str,
    max_steps: int,
    checkpoint_every_steps: int,
    start_step: int = 0,
    gradient_clip_norm: float | None = None,
) -> TrainingProgress:
    """Run a bounded loop and save exact-step checkpoints for interruption recovery."""

    if max_steps <= start_step:
        raise ValueError("max_steps must be greater than start_step")
    if checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")
    destination = Path(run_dir).expanduser().resolve()
    checkpoints = destination / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    _truncate_metrics_for_resume(metrics_path, start_step)
    policy.train()
    step = start_step
    latest_checkpoint: Path | None = None
    iterator = iter(batches)
    while step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("training batches iterable is empty") from exc
        optimizer.zero_grad(set_to_none=True)
        loss, forward_metrics = _extract_loss(policy(batch))
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError(f"training loss must be a finite scalar, got {loss}")
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip_norm)
        optimizer.step()
        step += 1
        record = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "forward_metrics": forward_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if step % checkpoint_every_steps == 0 or step == max_steps:
            latest_checkpoint = save_training_checkpoint(
                checkpoints / f"step_{step:08d}.pt",
                policy=policy,
                optimizer=optimizer,
                step=step,
                config_digest=config_digest,
            )
            latest_path = destination / "latest_checkpoint.json"
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination,
                prefix=".latest_checkpoint.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                latest_temporary = Path(handle.name)
                json.dump(
                    {"step": step, "path": str(latest_checkpoint)},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            latest_temporary.replace(latest_path)
    return TrainingProgress(step, config_digest, latest_checkpoint)
