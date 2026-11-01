import os
from pathlib import Path
import tempfile
import unittest

from trajectory_guidance.runtime_environment import (
    environment_with_venv_npp,
    find_venv_npp_library_dir,
)


class RuntimeEnvironmentTests(unittest.TestCase):
    def _fake_environment(self, root: Path) -> tuple[Path, Path]:
        interpreter = root / ".venv_planner" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.touch()
        npp_dir = (
            root
            / ".venv_planner"
            / "lib"
            / "python3.12"
            / "site-packages"
            / "nvidia"
            / "npp"
            / "lib"
        )
        npp_dir.mkdir(parents=True)
        (npp_dir / "libnppicc.so.12").touch()
        return interpreter, npp_dir

    def test_finds_npp_from_lexical_venv_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            interpreter, npp_dir = self._fake_environment(Path(temporary))
            self.assertEqual(find_venv_npp_library_dir(interpreter), npp_dir)

    def test_prepends_and_deduplicates_loader_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            interpreter, npp_dir = self._fake_environment(Path(temporary))
            existing = os.pathsep.join(["/system/lib", str(npp_dir), "/driver/lib"])
            result = environment_with_venv_npp(
                interpreter,
                {"LD_LIBRARY_PATH": existing, "KEEP_ME": "yes"},
            )
            self.assertEqual(
                result["LD_LIBRARY_PATH"].split(os.pathsep),
                [str(npp_dir), "/system/lib", "/driver/lib"],
            )
            self.assertEqual(result["KEEP_ME"], "yes")

    def test_missing_npp_library_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            interpreter = Path(temporary) / ".venv" / "bin" / "python"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            with self.assertRaisesRegex(FileNotFoundError, "libnppicc"):
                find_venv_npp_library_dir(interpreter)


if __name__ == "__main__":
    unittest.main()
