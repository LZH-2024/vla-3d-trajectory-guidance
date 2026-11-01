"""Dependency-light static visualizations for planner outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_trajectory_overlay(
    rgb: np.ndarray | Image.Image,
    waypoints: Sequence[Sequence[float]],
    actions: Sequence[str | None],
    output_path: str | Path,
) -> Path:
    """Save a 2-D overlay using the planner's normalized 0..1000 u/v convention."""

    image = rgb.convert("RGB") if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb))
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    points = [
        (
            int(round(float(point[0]) / 1000.0 * width)),
            int(round(float(point[1]) / 1000.0 * height)),
        )
        for point in waypoints
    ]

    if len(points) > 1:
        draw.line(points, fill=(255, 220, 0), width=3)

    for index, ((x, y), waypoint) in enumerate(zip(points, waypoints, strict=True)):
        action = actions[index] if index < len(actions) else None
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        normalized_action = (action or "").lower()
        if "close" in normalized_action:
            color = (255, 80, 80)
        elif "open" in normalized_action:
            color = (80, 160, 255)
        else:
            color = (70, 230, 120)
        radius = 7
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white")
        draw.text(
            (x + radius + 2, y - radius),
            f"{index}: {float(waypoint[2]):.3f} m",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            font=font,
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination
