"""Build deterministic Object–Program Workspace semantic controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json  # noqa: E402
from afts_arc.object_program_workspace_controls import (  # noqa: E402
    build_object_program_workspace_controls,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace control directory: {output_dir}")
    payload = build_object_program_workspace_controls(
        task_count=arguments.task_count
    )
    output_dir.mkdir(parents=True)
    for name in ("manifest", "challenges", "solutions", "visual_freeze"):
        atomic_write_json(output_dir / f"{name}.json", payload[name])
    print(
        json.dumps(
            {
                "cohort_id": payload["manifest"]["cohort_id"],
                "output_dir": str(output_dir),
                "task_count": arguments.task_count,
            }
        )
    )


if __name__ == "__main__":
    main()
