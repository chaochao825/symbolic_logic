"""Freeze a family-disjoint ARC-TGI cohort without exposing query gold to solvers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_json,
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)


COHORT_SCHEMA = "afts.arc-tgi-family-disjoint-cohort/v2"
PARTITION_SEED = "afts-arc-tgi-arcmini-family-split-20260810-v2"
EPISODE_SEED = "afts-arc-tgi-arcmini-episode-20260810-v1"
EXPECTED_FILE_COUNT = 180
EXPECTED_CANONICAL_FAMILY_COUNT = 170
DEVELOPMENT_FAMILY_COUNT = 50
CONFIRMATORY_FAMILY_COUNT = 100
RESERVE_FAMILY_COUNT = 18
MANUALLY_EXPOSED_FAMILIES = frozenset({"task2fJ984g27gSFKHfq53RTVH"})
QUARANTINED_FAMILIES = {
    "taskGqqVrV4CnAUNRgZGQZCUQK": {
        "selected_source_sha256": (
            "6050e25b52f8c4f669b7ddb0cdcc519ec25a1b5d42a59835292e0abe3353558e"
        ),
        "reason": "frozen episode output width exceeds ARC maximum 30",
    },
    "taskaasAJ4e5NPRnnWF5HTmp35": {
        "selected_source_sha256": (
            "3159f8fab3c1fdeeec3aebc91b4d4b021f655a8e586af573ff7c184491e02726"
        ),
        "reason": "frozen episode output width exceeds ARC maximum 30",
    },
}


def canonical_family_name(stem: str) -> str:
    """Merge upstream duplicate/revision suffixes into one conservative family."""

    return re.sub(r"(?:_1|_new|-new)$", "", stem)


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _normalize(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _normalize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("ARC-TGI metadata contains a non-finite float")
        return result
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("ARC-TGI metadata keys must be strings")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_normalize(item) for item in value]
        return sorted(items, key=canonical_json)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported ARC-TGI metadata type: {type(value).__name__}")


def _generator_sources(generator_root: Path) -> list[dict[str, object]]:
    files = sorted(generator_root.glob("*.py"))
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_FILE_COUNT} ARC-Mini files, found {len(files)}"
        )
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        grouped[canonical_family_name(path.stem)].append(path)
    if len(grouped) != EXPECTED_CANONICAL_FAMILY_COUNT:
        raise ValueError(
            "ARC-Mini canonical family count does not match the frozen contract"
        )
    rows = []
    for family_id, variants in sorted(grouped.items()):
        variant_rows = [
            {
                "relative_path": path.name,
                "source_sha256": file_sha256(path),
            }
            for path in variants
        ]
        selected = min(
            variant_rows,
            key=lambda row: (row["source_sha256"], row["relative_path"]),
        )
        family_source_id = canonical_sha256(
            {"family_id": family_id, "variants": variant_rows}
        )
        rows.append(
            {
                "family_id": family_id,
                "family_source_id": family_source_id,
                "selected_relative_path": selected["relative_path"],
                "selected_source_sha256": selected["source_sha256"],
                "variants": variant_rows,
                "manually_exposed_before_partition": (
                    family_id in MANUALLY_EXPOSED_FAMILIES
                ),
            }
        )
    return rows


def partition_generator_sources(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    if len(rows) != EXPECTED_CANONICAL_FAMILY_COUNT:
        raise ValueError("partition input does not contain exactly 170 families")
    quarantine = [
        {
            **dict(row),
            "quarantine_reason": QUARANTINED_FAMILIES[str(row["family_id"])]["reason"],
        }
        for row in rows
        if row["family_id"] in QUARANTINED_FAMILIES
    ]
    if {row["family_id"] for row in quarantine} != set(QUARANTINED_FAMILIES):
        raise ValueError("quarantine registry does not match generator sources")
    for row in quarantine:
        expected_sha256 = QUARANTINED_FAMILIES[str(row["family_id"])][
            "selected_source_sha256"
        ]
        if row["selected_source_sha256"] != expected_sha256:
            raise ValueError("quarantined generator source hash changed")
    eligible = [row for row in rows if row["family_id"] not in QUARANTINED_FAMILIES]
    exposed = [
        dict(row)
        for row in eligible
        if row["manually_exposed_before_partition"] is True
    ]
    if {row["family_id"] for row in exposed} != MANUALLY_EXPOSED_FAMILIES:
        raise ValueError("manual exposure registry does not match generator sources")
    remaining = [
        dict(row)
        for row in eligible
        if row["manually_exposed_before_partition"] is False
    ]
    remaining.sort(
        key=lambda row: hashlib.sha256(
            (
                PARTITION_SEED
                + "|"
                + str(row["family_id"])
                + "|"
                + str(row["family_source_id"])
            ).encode("utf-8")
        ).hexdigest()
    )
    exposed.sort(key=lambda row: str(row["family_id"]))
    development = exposed + remaining[: DEVELOPMENT_FAMILY_COUNT - len(exposed)]
    offset = DEVELOPMENT_FAMILY_COUNT - len(exposed)
    confirmatory = remaining[offset : offset + CONFIRMATORY_FAMILY_COUNT]
    reserve = remaining[offset + CONFIRMATORY_FAMILY_COUNT :]
    partitions = {
        "development": development,
        "confirmatory": confirmatory,
        "reserve": reserve,
        "quarantine": sorted(quarantine, key=lambda row: str(row["family_id"])),
    }
    expected_counts = {
        "development": DEVELOPMENT_FAMILY_COUNT,
        "confirmatory": CONFIRMATORY_FAMILY_COUNT,
        "reserve": RESERVE_FAMILY_COUNT,
        "quarantine": len(QUARANTINED_FAMILIES),
    }
    for name, expected in expected_counts.items():
        if len(partitions[name]) != expected:
            raise ValueError(f"{name} partition has the wrong family count")
    family_sets = {
        name: {str(row["family_id"]) for row in values}
        for name, values in partitions.items()
    }
    if family_sets["development"] & family_sets["confirmatory"]:
        raise ValueError("development and confirmatory families overlap")
    if family_sets["development"] & family_sets["reserve"]:
        raise ValueError("development and reserve families overlap")
    if family_sets["confirmatory"] & family_sets["reserve"]:
        raise ValueError("confirmatory and reserve families overlap")
    eligible_families = (
        family_sets["development"]
        | family_sets["confirmatory"]
        | family_sets["reserve"]
    )
    if eligible_families & family_sets["quarantine"]:
        raise ValueError("quarantined family entered an experiment partition")
    return partitions


def _load_generator_module(source_root: Path, source_path: Path) -> ModuleType:
    module_name = f"afts_arc_tgi_{file_sha256(source_path)}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import specification for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _generator_class(source_root: Path, source_path: Path) -> type:
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from Framework.arc_task_generator import ARCTaskGenerator

    module = _load_generator_module(source_root, source_path)
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, ARCTaskGenerator)
        and value is not ARCTaskGenerator
        and value.__module__ == module.__name__
    ]
    if len(classes) != 1:
        raise ValueError(f"generator file must define exactly one class: {source_path}")
    return classes[0]


def _grid(value: object, name: str) -> list[list[int]]:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional grid")
    if not 1 <= array.shape[0] <= 30 or not 1 <= array.shape[1] <= 30:
        raise ValueError(f"{name} has an invalid ARC shape")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if np.any(array < 0) or np.any(array > 9):
        raise ValueError(f"{name} has a color outside 0..9")
    return array.astype(np.uint8).tolist()


def _pair_list(value: object, name: str) -> list[dict[str, list[list[int]]]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    pairs = []
    for index, raw_pair in enumerate(value):
        if not isinstance(raw_pair, Mapping) or set(raw_pair) != {"input", "output"}:
            raise ValueError(f"{name}[{index}] has unexpected fields")
        pairs.append(
            {
                "input": _grid(raw_pair["input"], f"{name}[{index}].input"),
                "output": _grid(raw_pair["output"], f"{name}[{index}].output"),
            }
        )
    return pairs


def _generate_full_record(
    source_root: Path, source_path: Path, episode_seed: int
) -> dict[str, object]:
    generator_class = _generator_class(source_root, source_path)
    random.seed(episode_seed)
    np.random.seed(episode_seed % (2**32))
    generator = generator_class()
    task = generator.create_task()
    if not isinstance(task.data, Mapping) or set(task.data) != {"train", "test"}:
        raise ValueError("generated task data has unexpected fields")
    train = _pair_list(task.data["train"], "train")
    test = _pair_list(task.data["test"], "test")
    if not isinstance(task.input_reasoning_chain, list) or not all(
        isinstance(value, str) for value in task.input_reasoning_chain
    ):
        raise TypeError("input reasoning chain must be a list of strings")
    if not isinstance(task.transformation_reasoning_chain, list) or not all(
        isinstance(value, str) for value in task.transformation_reasoning_chain
    ):
        raise TypeError("transformation reasoning chain must be a list of strings")
    if not isinstance(task.code, str) or not task.code:
        raise TypeError("partial transformation witness must be a non-empty string")

    from Framework.execution import execute_transform_code

    partial_transform = execute_transform_code(task.code, generator)
    for split_name, pairs in (("train", train), ("test", test)):
        for pair_index, pair in enumerate(pairs):
            input_grid = np.asarray(pair["input"], dtype=np.uint8)
            expected = np.asarray(pair["output"], dtype=np.uint8)
            direct = np.asarray(
                generator.transform_input(input_grid.copy(), task.task_variables)
            )
            partial = np.asarray(partial_transform(generator, input_grid.copy()))
            if not np.array_equal(direct, expected):
                raise ValueError(
                    f"direct witness mismatch at {split_name}[{pair_index}]"
                )
            if not np.array_equal(partial, expected):
                raise ValueError(
                    f"partial witness mismatch at {split_name}[{pair_index}]"
                )
    return {
        "train": train,
        "test": test,
        "input_reasoning_chain": list(task.input_reasoning_chain),
        "transformation_reasoning_chain": list(task.transformation_reasoning_chain),
        "task_variables": _normalize(task.task_variables),
        "partial_transform_code": task.code,
    }


def split_full_record(
    full: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    train = full["train"]
    test = full["test"]
    if not isinstance(train, list) or not isinstance(test, list):
        raise TypeError("full task train/test fields must be lists")
    blind_test = []
    oracle_test = []
    for raw_pair in test:
        if not isinstance(raw_pair, Mapping):
            raise TypeError("test pair must be a mapping")
        blind_test.append({"input": raw_pair["input"]})
        oracle_test.append(raw_pair["output"])
    blind = {"train": train, "test": blind_test}
    oracle = {"test_outputs": oracle_test}
    witness = {
        "input_reasoning_chain": full["input_reasoning_chain"],
        "transformation_reasoning_chain": full["transformation_reasoning_chain"],
        "task_variables": full["task_variables"],
        "partial_transform_code": full["partial_transform_code"],
    }
    if any(set(pair) != {"input"} for pair in blind_test):
        raise ValueError("blind query payload contains an output field")
    return blind, oracle, witness


def _episode(
    arc_tgi_root: Path, generator_root: Path, source: Mapping[str, object]
) -> dict[str, object]:
    family_id = str(source["family_id"])
    source_path = generator_root / str(source["selected_relative_path"])
    episode_seed = _seed(
        EPISODE_SEED + "|" + family_id + "|" + str(source["family_source_id"])
    )
    first = _generate_full_record(arc_tgi_root, source_path, episode_seed)
    second = _generate_full_record(arc_tgi_root, source_path, episode_seed)
    if canonical_json(first) != canonical_json(second):
        raise ValueError(f"ARC-TGI replay is not deterministic: {family_id}")
    blind, oracle, witness = split_full_record(first)
    blind_sha256 = canonical_sha256(blind)
    episode_identity = {
        "family_id": family_id,
        "family_source_id": source["family_source_id"],
        "selected_source_sha256": source["selected_source_sha256"],
        "episode_seed": episode_seed,
        "blind_content_sha256": blind_sha256,
    }
    episode_id = canonical_sha256(episode_identity)
    task_id = f"arc_tgi_{family_id}_{episode_id[:16]}"
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "episode_seed": episode_seed,
        "family_id": family_id,
        "family_source_id": source["family_source_id"],
        "selected_relative_path": source["selected_relative_path"],
        "selected_source_sha256": source["selected_source_sha256"],
        "blind_content_sha256": blind_sha256,
        "oracle_sha256": canonical_sha256(oracle),
        "witness_sha256": canonical_sha256(witness),
        "replay_identical": True,
        "blind": blind,
        "oracle": oracle,
        "witness": witness,
    }


def _training_identifier_set(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise TypeError("NVARC training identifiers must be a list of strings")
    return {value.split("|||", maxsplit=1)[0] for value in raw if value != "<blank>"}


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _artifact_payloads(
    episodes: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    challenges = {str(episode["task_id"]): episode["blind"] for episode in episodes}
    solutions = {
        str(episode["task_id"]): episode["oracle"]["test_outputs"]  # type: ignore[index]
        for episode in episodes
    }
    witnesses = {str(episode["task_id"]): episode["witness"] for episode in episodes}
    return challenges, solutions, witnesses


def construct_cohort(args: argparse.Namespace) -> dict[str, object]:
    actual_revision = _git_revision(args.arc_tgi_root)
    if actual_revision != args.arc_tgi_commit:
        raise ValueError("ARC-TGI source revision does not match the frozen commit")
    generator_root = args.arc_tgi_root / "Generators" / "ARC-Mini"
    sources = _generator_sources(generator_root)
    training_ids = _training_identifier_set(args.training_identifiers)
    family_overlap = sorted(
        str(source["family_id"])
        for source in sources
        if source["family_id"] in training_ids
    )
    if family_overlap:
        raise ValueError("ARC-TGI family IDs overlap NVARC pretraining identifiers")
    partitions = partition_generator_sources(sources)
    generated: dict[str, list[dict[str, object]]] = {}
    for partition_name in ("development", "confirmatory"):
        generated[partition_name] = [
            _episode(args.arc_tgi_root, generator_root, source)
            for source in partitions[partition_name]
        ]
    development_ids = {episode["family_id"] for episode in generated["development"]}
    confirmatory_ids = {episode["family_id"] for episode in generated["confirmatory"]}
    if development_ids & confirmatory_ids:
        raise ValueError("generated development and confirmatory families overlap")
    all_blind_hashes = [
        str(episode["blind_content_sha256"])
        for values in generated.values()
        for episode in values
    ]
    if len(all_blind_hashes) != len(set(all_blind_hashes)):
        raise ValueError("generated cohort contains duplicate blind task content")

    args.output_root.mkdir(parents=True, exist_ok=True)
    artifact_rows: dict[str, dict[str, object]] = {}
    for partition_name in ("development", "confirmatory"):
        challenges, solutions, witnesses = _artifact_payloads(generated[partition_name])
        paths = {
            "challenges": args.output_root / f"{partition_name}_challenges.json",
            "solutions": args.output_root / f"{partition_name}_solutions.json",
            "witnesses": args.output_root / f"{partition_name}_witnesses.json",
        }
        payloads = {
            "challenges": challenges,
            "solutions": solutions,
            "witnesses": witnesses,
        }
        for name, path in paths.items():
            atomic_write_json(path, payloads[name])
        artifact_rows[partition_name] = {
            name: {
                "path": path.name,
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        }

    partition_manifest_content = {
        "schema": "afts.arc-tgi-family-partition/v1",
        "partition_seed": PARTITION_SEED,
        "manual_exposure_registry": sorted(MANUALLY_EXPOSED_FAMILIES),
        "structural_quarantine_registry": QUARANTINED_FAMILIES,
        "source_file_count": EXPECTED_FILE_COUNT,
        "canonical_family_count": EXPECTED_CANONICAL_FAMILY_COUNT,
        "variant_selection": "minimum_(source_sha256,relative_path)_per_canonical_family",
        "partitions": partitions,
    }
    partition_manifest = {
        "partition_id": canonical_sha256(partition_manifest_content),
        **partition_manifest_content,
    }
    partition_path = args.output_root / "partition_manifest.json"
    atomic_write_json(partition_path, partition_manifest)

    task_rows = {
        partition_name: [
            {
                key: episode[key]
                for key in (
                    "task_id",
                    "episode_id",
                    "episode_seed",
                    "family_id",
                    "family_source_id",
                    "selected_relative_path",
                    "selected_source_sha256",
                    "blind_content_sha256",
                    "oracle_sha256",
                    "witness_sha256",
                    "replay_identical",
                )
            }
            for episode in generated[partition_name]
        ]
        for partition_name in ("development", "confirmatory")
    }
    content = {
        "schema": COHORT_SCHEMA,
        "status": "frozen_before_solver_execution",
        "purpose": "mechanism_track_only",
        "arc_tgi_commit": actual_revision,
        "partition_id": partition_manifest["partition_id"],
        "partition_manifest_sha256": file_sha256(partition_path),
        "episode_seed_contract": EPISODE_SEED,
        "counts": {
            "development": len(generated["development"]),
            "confirmatory": len(generated["confirmatory"]),
            "reserve_unmaterialized": len(partitions["reserve"]),
            "quarantined_invalid_arc_shape": len(partitions["quarantine"]),
        },
        "nvarc_pretraining_family_id_overlap": family_overlap,
        "query_gold_boundary": {
            "challenge_files_contain_query_outputs": False,
            "solution_files_are_evaluator_only": True,
            "witness_files_are_evaluator_only": True,
            "solver_may_read_only_challenges": True,
            "query_gold_read_by_solver": False,
        },
        "family_disjoint": True,
        "instance_replay_identical": True,
        "artifacts": artifact_rows,
        "tasks": task_rows,
        "source_contract": {
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ): file_sha256(Path(__file__).resolve()),
            str(args.protocol.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ): file_sha256(args.protocol),
            "arc_tgi_framework_arc_task_generator": file_sha256(
                args.arc_tgi_root / "Framework" / "arc_task_generator.py"
            ),
            "arc_tgi_framework_execution": file_sha256(
                args.arc_tgi_root / "Framework" / "execution.py"
            ),
        },
        "runtime": runtime_metadata(("numpy", "shortuuid")),
    }
    return {"cohort_id": canonical_sha256(content), **content}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-tgi-root", type=Path, required=True)
    parser.add_argument("--arc-tgi-commit", required=True)
    parser.add_argument("--training-identifiers", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cohort = construct_cohort(args)
    atomic_write_json(args.output_root / "cohort_manifest.json", cohort)
    print(
        json.dumps(
            {
                "cohort_id": cohort["cohort_id"],
                "counts": cohort["counts"],
                "partition_id": cohort["partition_id"],
                "query_gold_boundary": cohort["query_gold_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
