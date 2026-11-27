#!/usr/bin/env python3
"""Marhaba ClipMind Universal Runner.
Can be invoked with any Python 3; automatically switches to the working environment with dependencies.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

KNOWN_VENVS = [
    Path(os.environ.get("CLIPMIND_PYTHON", "")) if os.environ.get("CLIPMIND_PYTHON") else Path(),
    HERE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    Path.home() / ".clipmind" / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    Path.home() / "Downloads" / "marhaba-clipmind" / "backend" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    HERE.parent / "marhaba-clipmind" / "backend" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    Path(sys.executable),
]


def find_working_python() -> str:
    try:
        import faster_whisper  # noqa: F401
        import cv2  # noqa: F401
        import scenedetect  # noqa: F401
        import psutil  # noqa: F401
        import pydantic  # noqa: F401
        import httpx  # noqa: F401
        return sys.executable
    except ImportError:
        pass

    for cand in KNOWN_VENVS:
        if cand and cand.is_file():
            cmd = [str(cand), "-c", "import faster_whisper, cv2, scenedetect, psutil, pydantic, httpx; print('OK')"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if res.returncode == 0 and "OK" in res.stdout:
                return str(cand)

    return sys.executable


def main():
    target_python = find_working_python()

    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{HERE}{os.pathsep}{existing_pp}" if existing_pp else str(HERE)
    env.setdefault("HF_HUB_VERBOSITY", "error")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    if Path(target_python).resolve() == Path(sys.executable).resolve():
        sys.path.insert(0, str(HERE))
        from clipmind.cli.main import main as cli_main
        cli_main()
    else:
        bootstrap = f"import sys; sys.path.insert(0, {repr(str(HERE))}); from clipmind.cli.main import main; main()"
        cmd = [target_python, "-c", bootstrap, *sys.argv[1:]]
        res = subprocess.run(cmd, env=env)
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
