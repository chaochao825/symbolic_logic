"""Freeze a collision-free ARC cohort eligibility amendment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.cohort_eligibility import build_collision_free_amendment  # noqa: E402
from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
)


def _load_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must contain an object with string keys")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--witnesses", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()

    output_root = arguments.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace artifact directory: {output_root}")
    source_paths = {
        "challenges": arguments.challenges.resolve(),
        "solutions": arguments.solutions.resolve(),
        "witnesses": arguments.witnesses.resolve(),
    }
    result = build_collision_free_amendment(
        challenges=_load_mapping(source_paths["challenges"]),
        solutions=_load_mapping(source_paths["solutions"]),
        witnesses=_load_mapping(source_paths["witnesses"]),
        source_paths=source_paths,
    )

    output_root.mkdir(parents=True)
    paths = {
        "challenges": output_root / "development_challenges.json",
        "solutions": output_root / "development_solutions.json",
        "witnesses": output_root / "development_witnesses.json",
    }
    for name in ("challenges", "solutions", "witnesses"):
        atomic_write_json(paths[name], result[name])
    amendment = result["amendment"]
    if not isinstance(amendment, dict):
        raise TypeError("amendment must be an object")
    amendment["output_artifacts"] = {
        name: {"path": paths[name].name, "sha256": file_sha256(paths[name])}
        for name in sorted(paths)
    }
    amendment = {"amendment_id": canonical_sha256(amendment), **amendment}
    atomic_write_json(output_root / "eligibility_amendment.json", amendment)
    print(
        json.dumps(
            {
                "amendment_id": amendment["amendment_id"],
                "eligible_tasks": amendment["counts"]["eligible_tasks"],
                "excluded_tasks": amendment["counts"]["excluded_tasks"],
                "output_root": str(output_root),
            }
        )
    )


if __name__ == "__main__":
    main()
