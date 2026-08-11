"""Write an immutable VARC query-blind cohort from ARC challenge JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402
from afts_arc.varc_blind import build_query_blind_tasks  # noqa: E402


def _load_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--source-cohort-id", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    challenges_path = Path(arguments.challenges).resolve()
    output_dir = Path(arguments.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace cohort directory: {output_dir}")
    blind_tasks, manifest = build_query_blind_tasks(
        challenges=_load_object(challenges_path),
        source_cohort_id=arguments.source_cohort_id,
        source_challenges_sha256=file_sha256(challenges_path),
    )
    task_dir = output_dir / "data" / "evaluation"
    for task_id in sorted(blind_tasks):
        atomic_write_json(task_dir / f"{task_id}.json", blind_tasks[task_id])
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "blind_cohort_id": manifest["blind_cohort_id"],
                "output_dir": str(output_dir),
                "task_count": len(blind_tasks),
            }
        )
    )


if __name__ == "__main__":
    main()
