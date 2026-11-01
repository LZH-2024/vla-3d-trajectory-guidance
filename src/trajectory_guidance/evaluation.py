"""Resumable closed-loop episode runner independent of a concrete simulator."""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from trajectory_guidance.experiment import EvaluationStore


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    suite: str
    task_id: int
    init_state_id: int
    seed: int
    max_steps: int

    def validate(self) -> None:
        if not _SAFE_ID.fullmatch(self.episode_id):
            raise ValueError("episode_id must be path-safe")
        if self.task_id < 0 or self.init_state_id < 0 or self.max_steps <= 0:
            raise ValueError("task/init IDs must be non-negative and max_steps positive")


def load_episode_plan(source: str | Path) -> list[EpisodeSpec]:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("episode plan schema mismatch")
    episodes = [EpisodeSpec(**entry) for entry in payload.get("episodes", [])]
    if not episodes:
        raise ValueError("episode plan is empty")
    identifiers: set[str] = set()
    for episode in episodes:
        episode.validate()
        if episode.episode_id in identifiers:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        identifiers.add(episode.episode_id)
    return episodes


EpisodeRunner = Callable[[EpisodeSpec, dict[str, Any]], dict[str, Any]]


def run_episode_plan(
    episodes: Sequence[EpisodeSpec],
    *,
    runner: EpisodeRunner,
    experiment_config: dict[str, Any],
    store: EvaluationStore,
    stop_on_error: bool = True,
) -> dict[str, Any]:
    """Run unfinished episodes and leave failures retryable on the next invocation."""

    completed = store.completed_episode_ids()
    attempted: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for episode in episodes:
        episode.validate()
        if episode.episode_id in completed:
            skipped.append(episode.episode_id)
            continue
        attempted.append(episode.episode_id)
        try:
            result = runner(episode, experiment_config)
            if not isinstance(result, dict):
                raise TypeError("episode runner must return a dictionary")
            if not isinstance(result.get("success"), bool):
                raise ValueError("episode result must contain boolean success")
            store.save_episode(episode.episode_id, result)
        except Exception as exc:
            failed.append(episode.episode_id)
            store.save_failure(
                episode.episode_id,
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            if stop_on_error:
                raise
    summary = store.summarize()
    return {
        **summary,
        "attempted_episode_ids": attempted,
        "skipped_completed_episode_ids": skipped,
        "failed_episode_ids": failed,
    }
