#!/usr/bin/env python3
"""Create the pinned Colab planner environment without switching notebook kernels."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(list(command), check=True, env=env)


def capture(command: Sequence[str]) -> str:
    return subprocess.check_output(list(command), text=True).strip()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_python(env_python: Path, expected_version: str) -> None:
    actual = capture(
        [
            str(env_python),
            "-c",
            "import platform,sys; print(platform.python_version()); print(sys.executable)",
        ]
    )
    lines = actual.splitlines()
    if not lines or lines[0] != expected_version:
        raise RuntimeError(
            f"existing environment uses Python {lines[0] if lines else 'unknown'}, "
            f"expected {expected_version}; remove it explicitly or choose a new --env-dir"
        )
    print(f"Verified target interpreter: {lines[-1]} (Python {lines[0]})")


def ensure_checkout(checkout: Path, repo_url: str, commit: str) -> None:
    if checkout.exists() and not (checkout / ".git").is_dir():
        if any(checkout.iterdir()):
            raise RuntimeError(f"{checkout} exists and is not an empty Git checkout")
    checkout.mkdir(parents=True, exist_ok=True)

    if not (checkout / ".git").is_dir():
        run(["git", "-C", str(checkout), "init"])
        run(["git", "-C", str(checkout), "remote", "add", "origin", repo_url])
    else:
        origin = capture(["git", "-C", str(checkout), "remote", "get-url", "origin"])
        if origin.rstrip("/") != repo_url.rstrip("/"):
            raise RuntimeError(f"unexpected origin for {checkout}: {origin}")
        dirty = capture(["git", "-C", str(checkout), "status", "--porcelain"])
        if dirty:
            raise RuntimeError(f"{checkout} has local changes; refusing to replace its checkout")

    try:
        current = capture(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        current = ""
    if current != commit:
        run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", commit])
        run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"])
    resolved = capture(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if resolved != commit:
        raise RuntimeError(f"checkout resolved to {resolved}, expected {commit}")
    print(f"Pinned upstream checkout: {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-dir", type=Path)
    parser.add_argument("--checkout-dir", type=Path)
    parser.add_argument(
        "--with-perf",
        action="store_true",
        help="Install the optional xformers extra. Omitted by default to protect the Torch pin.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (args.config or project_root / "configs" / "planner_colab.json").resolve()
    config = load_config(config_path)
    planner = config["planner"]
    core = config["reference_core"]

    if platform.system() != "Linux":
        raise RuntimeError("This bootstrap is intended for a Linux Google Colab runtime")

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            f"uv is not on PATH. Install exactly uv=={config['uv_version']} in the notebook kernel first."
        )
    uv_version_output = capture([uv, "--version"])
    if config["uv_version"] not in uv_version_output:
        raise RuntimeError(
            f"uv version mismatch: got {uv_version_output}, expected {config['uv_version']}"
        )

    env_dir = (args.env_dir or Path(planner["env_dir"])).resolve()
    checkout_dir = (args.checkout_dir or Path(planner["checkout_dir"])).resolve()
    env_python = env_dir / "bin" / "python"
    constraints = project_root / "configs" / "planner_constraints.txt"

    run([uv, "python", "install", config["python_version"]])
    if env_python.is_file():
        verify_python(env_python, config["python_version"])
    else:
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                uv,
                "venv",
                "--managed-python",
                "--python",
                config["python_version"],
                str(env_dir),
            ]
        )
        verify_python(env_python, config["python_version"])

    core_requirements = [
        f"torch=={core['torch']}",
        f"torchvision=={core['torchvision']}",
        f"torchcodec=={core['torchcodec']}",
        f"numpy=={core['numpy']}",
        f"nvidia-npp-cu12=={core['nvidia-npp-cu12']}",
    ]
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(env_python),
            "--torch-backend",
            "cu128",
            "--constraint",
            str(constraints),
            *core_requirements,
        ]
    )

    ensure_checkout(checkout_dir, planner["repo_url"], planner["repo_commit"])
    planner_spec = f"{checkout_dir}[perf]" if args.with_perf else str(checkout_dir)
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(env_python),
            "--torch-backend",
            "cu128",
            "--constraint",
            str(constraints),
            "--editable",
            planner_spec,
        ]
    )
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(env_python),
            "--no-deps",
            "--editable",
            str(project_root),
        ]
    )
    run([uv, "pip", "check", "--python", str(env_python)])

    print("\nPlanner environment prepared.")
    print(f"Interpreter: {env_python}")
    print(f"3D HAMSTER: {checkout_dir} @ {planner['repo_commit']}")
    print("The notebook kernel was not changed. Invoke this interpreter explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
