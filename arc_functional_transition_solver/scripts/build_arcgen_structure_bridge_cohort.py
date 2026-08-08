"""Build the frozen ARC-GEN cohort for the visual structure bridge gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path


SELECTION_SEED = "structured-visual-bridge-confirm-v1-20260809"
EXPECTED_ARCGEN_COMMIT = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
EXPECTED_ARC1_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"
FROZEN_FAMILIES = (
    "bae5c565",
    "9b30e358",
    "8dab14c2",
    "af726779",
    "87ab05b8",
    "9841fdad",
    "78e78cff",
    "14b8e18c",
    "6bcdb01e",
    "57edb29d",
    "e39e9282",
    "b5bb5719",
)
DEVELOPMENT_FAMILIES = frozenset(
    """2ccd9fef e45ef808 f18ec8cc 1d61978c 8fff9e47 f8f52ecc 2a28add5
    ac0c2ac3 b74ca5d1 b745798f 2b9ef948 f0f8a26d 37ce87bb aa62e3f4
    30f42897 22806e14 6350f1f4 1b59e163 a09f6c25 412b6263 cc9053aa
    230f2e48 b1986d4b 7e2bad24 880c1354 a2d730bd 20fb2937 252143c9
    470c91de d753a70b f0100645""".split()
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_grid(grid: object) -> list[list[int]]:
    if not isinstance(grid, list) or not grid:
        raise ValueError("generated grid must be a non-empty list")
    width: int | None = None
    normalized: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or not row:
            raise ValueError("generated row must be a non-empty list")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("generated grid must be rectangular")
        if any(type(cell) is not int or not 0 <= cell <= 9 for cell in row):
            raise ValueError("generated colors must be integers in [0, 9]")
        normalized.append(list(row))
    if len(normalized) > 30 or width is None or width > 30:
        raise ValueError("generated grid exceeds 30x30")
    return normalized


def pair_seed(family_id: str, pair_index: int) -> int:
    digest = hashlib.sha256(
        f"{SELECTION_SEED}\0{family_id}\0{pair_index}".encode("ascii")
    ).hexdigest()
    return int(digest[:16], 16)


def write_new_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arcgen_root", type=Path)
    parser.add_argument("arc1_training_dir", type=Path)
    parser.add_argument("arc1_evaluation_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"cohort output already exists: {args.output_dir}")
    if git_head(args.arcgen_root) != EXPECTED_ARCGEN_COMMIT:
        raise ValueError("ARC-GEN commit mismatch")
    arc1_root = args.arcgen_root / "external" / "ARC-AGI"
    if git_head(arc1_root) != EXPECTED_ARC1_COMMIT:
        raise ValueError("ARC-AGI submodule commit mismatch")

    sys.path.insert(0, str(args.arcgen_root.resolve()))
    task_list = importlib.import_module("task_list").task_list()
    arc1_ids = {
        path.stem
        for directory in (args.arc1_training_dir, args.arc1_evaluation_dir)
        for path in directory.glob("*.json")
    }
    generator_ids = {
        path.stem.removeprefix("task_")
        for path in (args.arcgen_root / "tasks").glob("task_*.py")
    }
    eligible = generator_ids - arc1_ids - DEVELOPMENT_FAMILIES
    ordered = tuple(
        sorted(
            eligible,
            key=lambda family_id: (
                hashlib.sha256(
                    f"{SELECTION_SEED}\0{family_id}".encode("ascii")
                ).hexdigest(),
                family_id,
            ),
        )
    )
    if ordered[: len(FROZEN_FAMILIES)] != FROZEN_FAMILIES:
        raise ValueError("frozen family selection does not replay")

    records: list[dict[str, object]] = []
    for family_id in FROZEN_FAMILIES:
        generator = task_list[family_id][0]
        pairs: list[dict[str, object]] = []
        seeds: list[int] = []
        for index in range(4):
            seed = pair_seed(family_id, index)
            random.seed(seed)
            generated = generator()
            if not isinstance(generated, dict) or set(generated) != {
                "input",
                "output",
            }:
                raise ValueError(f"malformed generated pair: {family_id}")
            pairs.append(
                {
                    "input": validate_grid(generated["input"]),
                    "output": validate_grid(generated["output"]),
                }
            )
            seeds.append(seed)
        if len({canonical_json(pair) for pair in pairs}) != len(pairs):
            raise ValueError(f"duplicate generated pair: {family_id}")
        gold = {"train": pairs[:3], "test": [pairs[3]]}
        blind = {
            "train": pairs[:3],
            "test": [{"input": pairs[3]["input"], "output": pairs[3]["input"]}],
        }
        gold_path = args.output_dir / "gold_data" / "training" / f"{family_id}.json"
        blind_path = (
            args.output_dir
            / "blind_data"
            / "ARC-AGI-2"
            / "data"
            / "evaluation"
            / f"{family_id}.json"
        )
        write_new_json(gold_path, gold)
        write_new_json(blind_path, blind)
        generator_path = args.arcgen_root / "tasks" / f"task_{family_id}.py"
        records.append(
            {
                "task_id": family_id,
                "family_id": family_id,
                "query_count": 1,
                "pair_seeds": seeds,
                "generator_sha256": file_sha256(generator_path),
                "gold_sha256": file_sha256(gold_path),
                "blind_sha256": file_sha256(blind_path),
            }
        )

    content = {
        "schema": "afts.visual-provider-cohort/v1",
        "selection": {
            "source_split": "ARC-GEN generated confirmation",
            "seed": SELECTION_SEED,
            "algorithm": "sha256(seed + NUL + family_id), ascending",
            "limit": len(FROZEN_FAMILIES),
            "episode_pair_count": 4,
            "train_pair_count": 3,
            "query_pair_count": 1,
            "eligible_generator_count": len(eligible),
            "arc1_task_count": len(arc1_ids),
            "development_family_count": len(DEVELOPMENT_FAMILIES),
        },
        "source": {
            "arcgen_commit": EXPECTED_ARCGEN_COMMIT,
            "arc1_submodule_commit": EXPECTED_ARC1_COMMIT,
            "arcgen_root": str(args.arcgen_root.resolve()),
        },
        "query_blind_contract": {
            "test_output_sentinel": "exact copy of the corresponding test input",
            "gold_output_present": False,
            "provider_may_read": "demonstration input/output pairs and test inputs only",
            "scoring_phase": "after content-addressed prediction and candidate freeze",
        },
        "blind_data_root": str(
            (args.output_dir / "blind_data" / "ARC-AGI-2").resolve()
        ),
        "gold_data_root": str((args.output_dir / "gold_data").resolve()),
        "tasks": records,
    }
    manifest = {"cohort_id": canonical_sha256(content), **content}
    write_new_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
