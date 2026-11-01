#!/usr/bin/env python3
"""Prepare the pinned LeRobot / LIBERO probe environment in /content/.venv."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def run(command: Sequence[str]) -> None:
    print(f"\n$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(list(command), check=True)


def capture(command: Sequence[str]) -> str:
    return subprocess.check_output(list(command), text=True).strip()


def verify_python(env_python: Path, expected_version: str) -> None:
    output = capture(
        [str(env_python), "-c", "import platform,sys; print(platform.python_version()); print(sys.executable)"]
    )
    lines = output.splitlines()
    actual = lines[0] if lines else "unknown"
    if actual != expected_version:
        raise RuntimeError(
            f"existing environment uses Python {actual}, expected {expected_version}; "
            "remove or rename it explicitly before retrying"
        )
    print(f"Verified target interpreter: {lines[-1]} (Python {actual})")


def ensure_checkout(checkout: Path, repo_url: str, commit: str) -> None:
    if checkout.exists() and not (checkout / ".git").is_dir() and any(checkout.iterdir()):
        raise RuntimeError(f"{checkout} exists and is not an empty Git checkout")
    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").is_dir():
        run(["git", "-C", str(checkout), "init"])
        run(["git", "-C", str(checkout), "remote", "add", "origin", repo_url])
    else:
        origin = capture(["git", "-C", str(checkout), "remote", "get-url", "origin"])
        if origin.rstrip("/") != repo_url.rstrip("/"):
            raise RuntimeError(f"unexpected origin for {checkout}: {origin}")
        if capture(["git", "-C", str(checkout), "status", "--porcelain"]):
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
    print(f"Pinned LeRobot checkout: {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-dir", type=Path)
    parser.add_argument("--checkout-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = (args.config or project_root / "configs" / "libero_probe_colab.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    core_path = (project_root / config["protected_core_config"]).resolve()
    core_config = json.loads(core_path.read_text(encoding="utf-8"))
    libero = config["libero"]
    core = core_config["reference_core"]

    if platform.system() != "Linux":
        raise RuntimeError("This bootstrap is intended for a Linux Colab or GPU-server runtime")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            f"uv is not on PATH; install exactly uv=={core_config['uv_version']} before running this script"
        )
    uv_output = capture([uv, "--version"])
    if core_config["uv_version"] not in uv_output:
        raise RuntimeError(
            f"uv version mismatch: got {uv_output}, expected {core_config['uv_version']}"
        )

    env_dir = (args.env_dir or Path(libero["env_dir"])).resolve()
    planner_env = Path(core_config["planner"]["env_dir"]).resolve()
    if env_dir == planner_env:
        raise RuntimeError("LIBERO must not be installed into the planner environment")
    checkout_dir = (args.checkout_dir or Path(libero["checkout_dir"])).resolve()
    env_python = env_dir / "bin" / "python"
    constraints = project_root / "configs" / "libero_constraints.txt"

    run([uv, "python", "install", core_config["python_version"]])
    if env_python.is_file():
        verify_python(env_python, core_config["python_version"])
    else:
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                uv,
                "venv",
                "--managed-python",
                "--python",
                core_config["python_version"],
                str(env_dir),
            ]
        )
        verify_python(env_python, core_config["python_version"])

    protected_requirements = [
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
            *protected_requirements,
        ]
    )

    wheel_spec = f"{libero['hf_libero_wheel_url']}#sha256={libero['hf_libero_wheel_sha256']}"
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
            wheel_spec,
        ]
    )

    ensure_checkout(checkout_dir, libero["repo_url"], libero["repo_commit"])
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(env_python),
            "--torch-backend",
            "cu128",
            "--no-sources",
            "--constraint",
            str(constraints),
            "--editable",
            f"{checkout_dir}[libero]",
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

    print("\nLIBERO probe environment prepared.")
    print(f"Interpreter: {env_python}")
    print(f"LeRobot: {checkout_dir} @ {libero['repo_commit']}")
    print("Only the [libero] extra was installed; SmolVLA training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
