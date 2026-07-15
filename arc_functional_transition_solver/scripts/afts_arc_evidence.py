"""Launch evidence commands without consulting adjacent timestamp-based pyc files."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    pycache_prefix = Path(tempfile.gettempdir()) / f"afts-arc-pycache-{uuid.uuid4().hex}"
    if pycache_prefix.exists():
        raise RuntimeError(f"fresh pycache prefix unexpectedly exists: {pycache_prefix}")

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("AFTS_M04A_TORCH_MODE", None)
    environment["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["AFTS_EVIDENCE_FRESH_SOURCE_LOADER"] = "1"
    environment["AFTS_EVIDENCE_LAUNCHER"] = str(Path(__file__).resolve())

    bootstrap = (
        "import sys; "
        "source_root=sys.argv.pop(1); "
        "pycache_prefix=sys.argv.pop(1); "
        "sys.path.append(source_root); "
        "sys.pycache_prefix=pycache_prefix; "
        "sys.dont_write_bytecode=True; "
        "import runpy; "
        "runpy.run_module('afts_arc',run_name='__main__',alter_sys=True)"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-S",
            "-c",
            bootstrap,
            str(source_root),
            str(pycache_prefix),
            *sys.argv[1:],
        ],
        cwd=Path.cwd(),
        env=environment,
        check=False,
    )
    if pycache_prefix.exists():
        raise RuntimeError(
            f"isolated pycache prefix was unexpectedly written: {pycache_prefix}"
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
