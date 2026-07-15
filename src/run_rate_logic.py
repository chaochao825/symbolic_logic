"""Run logic-discovery and MCR2/Booleanization experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rate_logic_experiments import run_logic_discovery, run_rate_reduction


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def git_state() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    seeds = range(1 if args.mode == "smoke" else 5)
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc)
    start_state = git_state()
    logic_rows: list[dict] = []
    rate_rows: list[dict] = []
    for seed in seeds:
        logic_rows.extend(run_logic_discovery(seed))
        rate_rows.extend(run_rate_reduction(seed))
    logic_path = RESULTS / "logic_discovery_results.csv"
    rate_path = RESULTS / "rate_reduction_results.csv"
    pd.DataFrame(logic_rows).to_csv(logic_path, index=False)
    pd.DataFrame(rate_rows).to_csv(rate_path, index=False)
    source_paths = [ROOT / "src" / name for name in ("logic_core.py", "rate_logic_experiments.py", "run_rate_logic.py")]
    metadata = {
        "mode": args.mode,
        "seeds": list(seeds),
        "elapsed_seconds": time.perf_counter() - started,
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "working_directory": str(Path.cwd()),
        "invocation": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "hash_mode": "sha256 after CRLF-to-LF normalization for tracked text",
        "requirements_lock_sha256": sha256(ROOT / "requirements-lock.txt"),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "git_at_start": start_state,
        "scope": "NumPy controlled proxy for MCR2 objective/gradient flow, finite gate/topology search, and Booleanization accounting; not full ReduNet or CRATE reproduction",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (logic_path, rate_path)},
    }
    metadata_path = RESULTS / "rate_logic_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"logic_rows": len(logic_rows), "rate_rows": len(rate_rows), **metadata}, indent=2))


if __name__ == "__main__":
    main()
