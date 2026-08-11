"""CLI for the demo-only stateful rewrite failure audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402
from afts_arc.stateful_rewrite_failure_audit import (  # noqa: E402
    audit_stateful_rewrite_freeze,
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--max-trials-per-node", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "afts_arc" / "stateful_scene.py",
        PROJECT_ROOT / "src" / "afts_arc" / "stateful_object_graph_rewrite.py",
        PROJECT_ROOT / "src" / "afts_arc" / "stateful_rewrite_failure_audit.py",
    )
    result = audit_stateful_rewrite_freeze(
        challenges=_read_object(args.challenges),
        freeze=_read_object(args.freeze),
        source_files={
            str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
                "\\", "/"
            ): file_sha256(path)
            for path in source_paths
        },
        max_trials_per_node=args.max_trials_per_node,
    )
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "audit_id": result["audit_id"],
                "metrics": result["metrics"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
