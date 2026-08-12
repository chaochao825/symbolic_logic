"""Audit heterogeneous population evidence at the sealed family level."""

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
from afts_arc.population_family_audit import audit_population_families  # noqa: E402


def _load_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-result", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--anchor-provider", required=True)
    parser.add_argument("--recruited-provider", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    result_path = arguments.population_result.resolve()
    seal_path = arguments.seal.resolve()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    payload = audit_population_families(
        population_result=_load_object(result_path),
        seal=_load_object(seal_path),
        anchor_provider=arguments.anchor_provider,
        recruited_provider=arguments.recruited_provider,
        source_files={
            "population_result": file_sha256(result_path),
            "seal": file_sha256(seal_path),
        },
    )
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "audit_id": payload["audit_id"],
                "family_count": payload["family_count"],
                "family_signal_counts": payload["family_signal_counts"],
                "marginal_cluster_concentration": payload[
                    "marginal_cluster_concentration"
                ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
