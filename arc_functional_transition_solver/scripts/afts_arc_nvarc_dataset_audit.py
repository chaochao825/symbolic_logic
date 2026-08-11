"""Audit a compiled NVARC dataset without opening query-label semantics."""

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
from afts_arc.nvarc_dataset_audit import (  # noqa: E402
    build_nvarc_dataset_boundary_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--global-batch-size", required=True, type=int)
    parser.add_argument("--dataset-seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    audit = build_nvarc_dataset_boundary_audit(
        data_dir=Path(arguments.data_dir),
        challenges_path=Path(arguments.challenges),
        epochs=arguments.epochs,
        global_batch_size=arguments.global_batch_size,
        dataset_seed=arguments.dataset_seed,
    )
    atomic_write_json(output, audit)
    print(
        json.dumps(
            {
                "audit_id": audit["audit_id"],
                "output": str(output),
                "status": audit["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
