"""Freeze the exact-coverage overlap of two scored candidate providers."""

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
from afts_arc.provider_overlap import build_provider_overlap  # noqa: E402


def _load_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-result", type=Path, required=True)
    parser.add_argument("--second-result", type=Path, required=True)
    parser.add_argument("--first-provider", required=True)
    parser.add_argument("--second-provider", required=True)
    parser.add_argument("--first-endpoint", required=True)
    parser.add_argument("--second-endpoint", required=True)
    parser.add_argument("--eligibility-amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    first_result = arguments.first_result.resolve()
    second_result = arguments.second_result.resolve()
    amendment_path = arguments.eligibility_amendment.resolve()
    amendment = _load_object(amendment_path)
    eligible_task_ids = amendment["eligible_task_ids"]
    if not isinstance(eligible_task_ids, list) or not all(
        isinstance(task_id, str) for task_id in eligible_task_ids
    ):
        raise TypeError("eligibility amendment task IDs must be an array of strings")
    overlap = build_provider_overlap(
        first_result=_load_object(first_result),
        second_result=_load_object(second_result),
        first_provider=arguments.first_provider,
        second_provider=arguments.second_provider,
        first_endpoint=arguments.first_endpoint,
        second_endpoint=arguments.second_endpoint,
        eligible_task_ids=eligible_task_ids,
        eligibility_amendment_id=str(amendment["amendment_id"]),
        source_files={
            "eligibility_amendment": file_sha256(amendment_path),
            "first_result": file_sha256(first_result),
            "second_result": file_sha256(second_result),
        },
    )
    atomic_write_json(output, overlap)
    print(
        json.dumps(
            {
                "coverage": overlap["coverage"],
                "output": str(output),
                "overlap_id": overlap["overlap_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
