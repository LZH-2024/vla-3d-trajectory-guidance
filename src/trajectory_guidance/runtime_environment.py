"""Runtime environment helpers for isolated Colab virtual environments."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def find_venv_npp_library_dir(interpreter: str | Path) -> Path:
    """Return the NVIDIA NPP library directory belonging to ``interpreter``.

    The interpreter may be a symlink, so its lexical ``<venv>/bin/python`` path
    is deliberately retained instead of resolving it to the uv-managed base
    Python installation.
    """

    interpreter_path = Path(interpreter).expanduser()
    environment_root = interpreter_path.parent.parent
    candidates = sorted(
        path
        for path in (environment_root / "lib").glob(
            "python*/site-packages/nvidia/npp/lib"
        )
        if any(path.glob("libnppicc.so*"))
    )
    if not candidates:
        raise FileNotFoundError(
            "NVIDIA NPP runtime library libnppicc.so* was not found under "
            f"{environment_root / 'lib'}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "multiple NVIDIA NPP library directories were found for "
            f"{interpreter_path}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def environment_with_venv_npp(
    interpreter: str | Path,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a subprocess environment that can resolve the venv's NPP libs."""

    environment = dict(os.environ if base_environment is None else base_environment)
    npp_dir = str(find_venv_npp_library_dir(interpreter))
    current = environment.get("LD_LIBRARY_PATH", "")
    paths = [path for path in current.split(os.pathsep) if path]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [npp_dir, *(path for path in paths if path != npp_dir)]
    )
    return environment
