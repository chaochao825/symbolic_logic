"""Build the frozen fresh-family cohort for visual trace repair."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.blind import BlindTask  # noqa: E402
from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
)
from afts_arc.grid import GridValidationError, as_grid, grid_to_lists  # noqa: E402
from afts_arc.hybrid.object_code import (  # noqa: E402
    enumerate_object_code_programs,
    execute_object_code_program,
    object_code_program_id,
    synthesize_object_code_programs,
)
from afts_arc.hybrid.scene_graph import (  # noqa: E402
    ScenePipelineProgram,
)
from afts_arc.hybrid.types import canonical_json  # noqa: E402
from afts_arc.task import ARCPair  # noqa: E402
from afts_arc.visual_provider_gate import (  # noqa: E402
    COHORT_SCHEMA as VISUAL_PROVIDER_COHORT_SCHEMA,
    _file_sha256,
    discover_previously_exposed_task_ids,
)
from afts_arc.visual_trace_repair import (  # noqa: E402
    scene_program_fields,
)


TRACE_REPAIR_COHORT_SCHEMA = "afts.visual-trace-repair-cohort/v1"
SELECTION_SEED = "visual-relational-trace-repair-v1-20260809"
EXPECTED_ARCGEN_COMMIT = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
EXPECTED_ARC1_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"
EXPECTED_ARC1_SPLIT_COUNT = 400
TASK_LIMIT = 12
INITIAL_PROGRAM_TRIALS = 512
MAX_NEAR_MISSES = 64
MINIMUM_NEAR_MISS_AGREEMENT = 0.5
MAX_SCENE_PROGRAMS = 100_000
FAMILY_SCAN_TIMEOUT_SECONDS = 30
SCAN_WORKERS = 8

EXPOSED_FAMILIES = frozenset(
    """2ccd9fef e45ef808 f18ec8cc 1d61978c 8fff9e47 f8f52ecc 2a28add5
    ac0c2ac3 b74ca5d1 b745798f 2b9ef948 f0f8a26d 37ce87bb aa62e3f4
    30f42897 22806e14 6350f1f4 1b59e163 a09f6c25 412b6263 cc9053aa
    230f2e48 b1986d4b 7e2bad24 880c1354 a2d730bd 20fb2937 252143c9
    470c91de d753a70b f0100645 bae5c565 9b30e358 8dab14c2 af726779
    87ab05b8 9841fdad 78e78cff 14b8e18c 6bcdb01e 57edb29d e39e9282
    b5bb5719""".split()
)


class GeneratedGridOutOfScopeError(ValueError):
    """A generated grid violates the public ARC size contract."""


def _git_head(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validated_arc1_ids(
    arcgen_root: Path,
    training_dir: Path,
    evaluation_dir: Path,
) -> set[str]:
    """Bind exclusions to the complete ARC-1 submodule at the locked commit."""

    arc1_data = (arcgen_root / "external" / "ARC-AGI" / "data").resolve()
    expected = (
        (arc1_data / "training").resolve(),
        (arc1_data / "evaluation").resolve(),
    )
    observed = (training_dir.resolve(), evaluation_dir.resolve())
    if observed != expected:
        raise ValueError("ARC-1 identity sources must be the locked ARC-GEN submodule")
    split_ids = tuple(
        {path.stem for path in directory.glob("*.json")} for directory in observed
    )
    if any(len(ids) != EXPECTED_ARC1_SPLIT_COUNT for ids in split_ids):
        raise ValueError("ARC-1 identity source is incomplete")
    if split_ids[0] & split_ids[1]:
        raise ValueError("ARC-1 training and evaluation identities overlap")
    return split_ids[0] | split_ids[1]


def _pair_seed(family_id: str, pair_index: int) -> int:
    digest = hashlib.sha256(
        f"{SELECTION_SEED}\0{family_id}\0{pair_index}".encode("ascii")
    ).hexdigest()
    return int(digest[:16], 16)


def _episode_id(family_id: str, kind: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}\0{family_id}\0{kind}".encode("ascii")
    ).hexdigest()[:12]


def _validate_grid(grid: object) -> list[list[int]]:
    normalized = grid_to_lists(as_grid(grid))
    if len(normalized) > 30 or len(normalized[0]) > 30:
        raise GeneratedGridOutOfScopeError(
            "generated grid exceeds ARC's 30x30 limit"
        )
    return normalized


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _blind_task(pairs: list[dict[str, list[list[int]]]]) -> BlindTask:
    return BlindTask.from_observations(
        train=tuple(
            ARCPair(as_grid(pair["input"]), as_grid(pair["output"]))
            for pair in pairs[:3]
        ),
        test_inputs=(as_grid(pairs[3]["input"]),),
    )


def _valid_query_execution(program: ScenePipelineProgram, task: BlindTask) -> bool:
    return all(
        (execution := execute_object_code_program(program, query)).ok
        and execution.output is not None
        for query in task.test_inputs
    )


def _demo_exact(program: ScenePipelineProgram, task: BlindTask) -> bool:
    return all(
        (execution := execute_object_code_program(program, pair.input)).ok
        and execution.output == pair.output
        for pair in task.train
    )


def _select_parent(
    task: BlindTask,
) -> tuple[
    tuple[
        ScenePipelineProgram,
        dict[str, object],
        tuple[str, ...],
        tuple[str, ...],
    ]
    | None,
    dict[str, object],
]:
    all_programs = enumerate_object_code_programs(task)
    initial_programs = tuple(all_programs[:INITIAL_PROGRAM_TRIALS])
    if len(initial_programs) != INITIAL_PROGRAM_TRIALS:
        return None, {
            "reason": "insufficient_initial_programs",
            "all_program_count": len(all_programs),
            "scene_program_count": sum(
                isinstance(program, ScenePipelineProgram) for program in all_programs
            ),
        }
    scene_programs = tuple(
        program for program in all_programs if isinstance(program, ScenePipelineProgram)
    )
    if len(scene_programs) > MAX_SCENE_PROGRAMS:
        return None, {
            "reason": "scene_grammar_construction_cap",
            "all_program_count": len(all_programs),
            "scene_program_count": len(scene_programs),
        }
    initial_ids = tuple(object_code_program_id(program) for program in initial_programs)
    synthesis = synthesize_object_code_programs(
        task,
        max_program_trials=INITIAL_PROGRAM_TRIALS,
        max_exact_programs=32,
        max_near_misses=MAX_NEAR_MISSES,
        minimum_near_miss_agreement=MINIMUM_NEAR_MISS_AGREEMENT,
        programs=initial_programs,
    )
    scene_records = tuple(
        (
            program,
            object_code_program_id(program),
            scene_program_fields(program),
        )
        for program in scene_programs
    )
    field_names = tuple(sorted(scene_records[0][2]))
    neighbor_index: dict[
        tuple[str, str], list[tuple[ScenePipelineProgram, str, dict[str, object]]]
    ] = {}
    for program, program_id, fields in scene_records:
        if tuple(sorted(fields)) != field_names:
            raise ValueError("scene programs have inconsistent flattened fields")
        for field in field_names:
            key = (
                field,
                canonical_json(
                    {name: fields[name] for name in field_names if name != field}
                ),
            )
            neighbor_index.setdefault(key, []).append((program, program_id, fields))
    initial_id_set = frozenset(initial_ids)
    for score in synthesis.near_miss_scores:
        parent = score.program
        if not isinstance(parent, ScenePipelineProgram) or not _valid_query_execution(
            parent, task
        ):
            continue
        parent_fields = scene_program_fields(parent)
        exact_edits: list[tuple[str, ScenePipelineProgram]] = []
        for field in field_names:
            key = (
                field,
                canonical_json(
                    {
                        name: parent_fields[name]
                        for name in field_names
                        if name != field
                    }
                ),
            )
            for candidate, candidate_id, fields in neighbor_index[key]:
                if (
                    candidate_id in initial_id_set
                    or fields[field] == parent_fields[field]
                ):
                    continue
                if _demo_exact(candidate, task) and _valid_query_execution(
                    candidate, task
                ):
                    exact_edits.append((field, candidate))
        if not exact_edits:
            continue
        parent_record = {
            "program": parent.to_json_dict(),
            "program_id": object_code_program_id(parent),
            "exact_demo_count": score.exact_demo_count,
            "shape_match_count": score.shape_match_count,
            "agreement": score.agreement,
            "mismatch_count": score.mismatch_count,
            "execution_valid": score.execution_valid,
            "functional_trace": list(parent.functional_trace),
        }
        repairable_slots = tuple(sorted({slot for slot, _ in exact_edits}))
        return (
            parent,
            parent_record,
            initial_ids,
            repairable_slots,
        ), {
            "reason": "retained",
            "all_program_count": len(all_programs),
            "scene_program_count": len(scene_programs),
            "near_miss_count": len(synthesis.near_miss_scores),
            "scene_near_miss_count": sum(
                isinstance(item.program, ScenePipelineProgram)
                for item in synthesis.near_miss_scores
            ),
        }
    return None, {
        "reason": "no_demo_exact_single_slot_neighbor",
        "all_program_count": len(all_programs),
        "scene_program_count": len(scene_programs),
        "near_miss_count": len(synthesis.near_miss_scores),
        "scene_near_miss_count": sum(
            isinstance(item.program, ScenePipelineProgram)
            for item in synthesis.near_miss_scores
        ),
    }


def _provider_episode(
    pairs: list[dict[str, list[list[int]]]],
    *,
    heldout_demo_index: int | None,
) -> dict[str, object]:
    if heldout_demo_index is None:
        train = pairs[:3]
        test_input = pairs[3]["input"]
    else:
        train = [pair for index, pair in enumerate(pairs[:3]) if index != heldout_demo_index]
        test_input = pairs[heldout_demo_index]["input"]
    return {
        "train": train,
        "test": [{"input": test_input, "output": test_input}],
    }


def _scan_family_worker(
    family_id: str,
    generator: Callable[[], object],
    connection: Connection,
) -> None:
    pairs: list[dict[str, list[list[int]]]] = []
    seeds = []
    for pair_index in range(4):
        seed = _pair_seed(family_id, pair_index)
        random.seed(seed)
        try:
            generated = generator()
        except Exception as error:
            connection.send(
                (
                    None,
                    None,
                    None,
                    None,
                    {
                        "reason": "generator_failure",
                        "pair_index": pair_index,
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    },
                )
            )
            connection.close()
            return
        if not isinstance(generated, dict) or set(generated) != {
            "input",
            "output",
        }:
            raise ValueError(f"malformed generated pair: {family_id}")
        try:
            pair = {
                "input": _validate_grid(generated["input"]),
                "output": _validate_grid(generated["output"]),
            }
        except (GeneratedGridOutOfScopeError, GridValidationError):
            connection.send(
                (
                    None,
                    None,
                    None,
                    None,
                    {"reason": "generated_grid_out_of_scope"},
                )
            )
            connection.close()
            return
        pairs.append(pair)
        seeds.append(seed)
    if len({json.dumps(pair, sort_keys=True) for pair in pairs}) != len(pairs):
        connection.send(
            (
                None,
                None,
                None,
                None,
                {"reason": "duplicate_generated_pair"},
            )
        )
        connection.close()
        return
    blind_task = _blind_task(pairs)
    selected, diagnostics = _select_parent(blind_task)
    connection.send((pairs, seeds, blind_task, selected, diagnostics))
    connection.close()


def _start_family_scan(
    context: multiprocessing.context.BaseContext,
    family_id: str,
    generator: Callable[[], object],
) -> tuple[Connection, multiprocessing.Process, float]:
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_scan_family_worker,
        args=(family_id, generator, sender),
    )
    process.start()
    sender.close()
    return receiver, process, time.monotonic()


def _scan_families_in_order(
    ordered: tuple[str, ...],
    task_list: dict[str, tuple[Callable[[], object], object]],
) -> Iterator[
    tuple[str, tuple[object, object, object, object, dict[str, object]]]
]:
    """Scan concurrently while yielding the preregistered family order."""

    context = multiprocessing.get_context("fork")
    active: dict[
        int, tuple[Connection, multiprocessing.Process, float]
    ] = {}
    completed: dict[
        int, tuple[object, object, object, object, dict[str, object]]
    ] = {}
    next_to_start = 0
    next_to_yield = 0
    try:
        while next_to_yield < len(ordered):
            while next_to_start < len(ordered) and len(active) < SCAN_WORKERS:
                family_id = ordered[next_to_start]
                active[next_to_start] = _start_family_scan(
                    context,
                    family_id,
                    task_list[family_id][0],
                )
                next_to_start += 1

            made_progress = False
            now = time.monotonic()
            for index in tuple(sorted(active)):
                receiver, process, started_at = active[index]
                if receiver.poll():
                    try:
                        result = receiver.recv()
                    except EOFError as error:
                        process.join()
                        receiver.close()
                        raise RuntimeError(
                            "family selection subprocess closed its result pipe: "
                            f"{ordered[index]} (exit code {process.exitcode})"
                        ) from error
                    process.join()
                    receiver.close()
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"family selection subprocess failed: {ordered[index]}"
                        )
                    completed[index] = result
                    del active[index]
                    made_progress = True
                    continue
                if not process.is_alive():
                    process.join()
                    receiver.close()
                    raise RuntimeError(
                        "family selection subprocess exited without a result: "
                        f"{ordered[index]} (exit code {process.exitcode})"
                    )
                if now - started_at >= FAMILY_SCAN_TIMEOUT_SECONDS:
                    process.terminate()
                    process.join()
                    receiver.close()
                    completed[index] = (
                        None,
                        None,
                        None,
                        None,
                        {
                            "reason": "family_scan_timeout",
                            "timeout_seconds": FAMILY_SCAN_TIMEOUT_SECONDS,
                        },
                    )
                    del active[index]
                    made_progress = True

            while next_to_yield in completed:
                family_id = ordered[next_to_yield]
                result = completed.pop(next_to_yield)
                next_to_yield += 1
                made_progress = True
                yield family_id, result
            if not made_progress:
                time.sleep(0.01)
    finally:
        for receiver, process, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
            receiver.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arcgen_root", type=Path)
    parser.add_argument("arc1_training_dir", type=Path)
    parser.add_argument("arc1_evaluation_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"cohort output already exists: {args.output_dir}")
    if _git_head(args.arcgen_root) != EXPECTED_ARCGEN_COMMIT:
        raise ValueError("ARC-GEN commit mismatch")
    arc1_root = args.arcgen_root / "external" / "ARC-AGI"
    if _git_head(arc1_root) != EXPECTED_ARC1_COMMIT:
        raise ValueError("ARC-AGI submodule commit mismatch")

    sys.path.insert(0, str(args.arcgen_root.resolve()))
    task_list = importlib.import_module("task_list").task_list()
    arc1_ids = _validated_arc1_ids(
        args.arcgen_root,
        args.arc1_training_dir,
        args.arc1_evaluation_dir,
    )
    generator_ids = {
        path.stem.removeprefix("task_")
        for path in (args.arcgen_root / "tasks").glob("task_*.py")
    }
    discovered_exposures = discover_previously_exposed_task_ids(PROJECT_ROOT)
    all_exposures = EXPOSED_FAMILIES | discovered_exposures
    eligible = generator_ids - arc1_ids - all_exposures
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

    records: list[dict[str, object]] = []
    provider_records: list[dict[str, object]] = []
    scanned = 0
    repairable_slot_counts: Counter[str] = Counter()
    selection_reason_counts: Counter[str] = Counter()
    scan_records: list[dict[str, object]] = []
    scan_iterator = _scan_families_in_order(ordered, task_list)
    try:
        for family_id, scan_result in scan_iterator:
            if len(records) == TASK_LIMIT:
                break
            scanned += 1
            print(f"scan {scanned}: {family_id}", file=sys.stderr, flush=True)
            pairs, seeds, blind_task, selected, scan_diagnostics = scan_result
            selection_reason_counts[scan_diagnostics["reason"]] += 1
            scan_records.append({"family_id": family_id, **scan_diagnostics})
            print(
                f"scan result {family_id}: {scan_diagnostics['reason']}",
                file=sys.stderr,
                flush=True,
            )
            if selected is None:
                continue
            if (
                not isinstance(pairs, list)
                or not isinstance(seeds, list)
                or not isinstance(blind_task, BlindTask)
            ):
                raise TypeError(
                    "retained family scan omitted generated task evidence"
                )
            _, parent_record, initial_program_ids, repairable_slots = selected
            repairable_slot_counts.update(repairable_slots)
            print(
                f"retain {len(records) + 1}/{TASK_LIMIT}: {family_id}",
                file=sys.stderr,
                flush=True,
            )

            gold = {"train": pairs[:3], "test": [pairs[3]]}
            blind = {
                "train": pairs[:3],
                "test": [
                    {"input": pairs[3]["input"], "output": pairs[3]["input"]}
                ],
            }
            gold_path = (
                args.output_dir / "gold_data" / "training" / f"{family_id}.json"
            )
            blind_path = args.output_dir / "blind_tasks" / f"{family_id}.json"
            _write_new(gold_path, gold)
            _write_new(blind_path, blind)

            episodes = []
            for kind, heldout in (
                ("query", None),
                ("lodo0", 0),
                ("lodo1", 1),
                ("lodo2", 2),
            ):
                episode_id = _episode_id(family_id, kind)
                episode_path = (
                    args.output_dir
                    / "provider_blind"
                    / "ARC-AGI-2"
                    / "data"
                    / "evaluation"
                    / f"{episode_id}.json"
                )
                _write_new(
                    episode_path,
                    _provider_episode(pairs, heldout_demo_index=heldout),
                )
                episode_record = {
                    "episode_id": episode_id,
                    "kind": kind,
                    "heldout_demo_index": heldout,
                    "blind_sha256": _file_sha256(episode_path),
                }
                episodes.append(episode_record)
                provider_records.append(
                    {
                        "task_id": episode_id,
                        "family_id": family_id,
                        "episode_kind": kind,
                        "heldout_demo_index": heldout,
                        "source_sha256": _file_sha256(blind_path),
                        "blind_sha256": _file_sha256(episode_path),
                        "query_count": 1,
                    }
                )

            generator_path = args.arcgen_root / "tasks" / f"task_{family_id}.py"
            records.append(
                {
                    "task_id": family_id,
                    "family_id": family_id,
                    "pair_seeds": seeds,
                    "generator_sha256": _file_sha256(generator_path),
                    "gold_sha256": _file_sha256(gold_path),
                    "blind_sha256": _file_sha256(blind_path),
                    "blind_task_id": blind_task.task_id,
                    "blind_content_sha256": blind_task.blind_content_sha256,
                    "parent": parent_record,
                    "initial_program_trial_count": INITIAL_PROGRAM_TRIALS,
                    "initial_program_ids": list(initial_program_ids),
                    "demo_exact_single_slot_repairable": True,
                    "repairable_slots": list(repairable_slots),
                    "episodes": episodes,
                }
            )
    finally:
        scan_iterator.close()
    scan_content = {
        "schema": "afts.visual-trace-repair-feasibility-scan/v1",
        "selection_seed": SELECTION_SEED,
        "source": {
            "arcgen_commit": EXPECTED_ARCGEN_COMMIT,
            "arc1_submodule_commit": EXPECTED_ARC1_COMMIT,
        },
        "query_gold_read": False,
        "varc_predictions_read": False,
        "requested_task_count": TASK_LIMIT,
        "retained_task_count": len(records),
        "scanned_family_count": scanned,
        "generator_family_count": len(generator_ids),
        "arc1_task_count": len(arc1_ids),
        "arc1_generator_overlap_count": len(generator_ids & arc1_ids),
        "explicit_exposed_family_count": len(EXPOSED_FAMILIES),
        "authored_exposure_count": len(discovered_exposures),
        "excluded_generator_exposure_count": len(generator_ids & all_exposures),
        "eligible_generator_count": len(eligible),
        "max_scene_programs": MAX_SCENE_PROGRAMS,
        "family_scan_timeout_seconds": FAMILY_SCAN_TIMEOUT_SECONDS,
        "scan_workers": SCAN_WORKERS,
        "selection_reason_counts": dict(sorted(selection_reason_counts.items())),
        "repairable_slot_counts": dict(sorted(repairable_slot_counts.items())),
        "families": scan_records,
    }
    _write_new(
        args.output_dir / "feasibility_scan.json",
        {"scan_id": canonical_sha256(scan_content), **scan_content},
    )
    if len(records) != TASK_LIMIT:
        raise ValueError(
            f"only {len(records)} repairable fresh families found after {scanned} scans"
        )

    provider_content = {
        "schema": VISUAL_PROVIDER_COHORT_SCHEMA,
        "selection": {
            "source_split": "ARC-GEN visual trace repair provider episodes",
            "seed": SELECTION_SEED,
            "algorithm": "four frozen query-blind episodes per retained family",
            "limit": len(provider_records),
            "family_count": len(records),
            "episode_count_per_family": 4,
        },
        "source": {
            "arcgen_commit": EXPECTED_ARCGEN_COMMIT,
            "arc1_submodule_commit": EXPECTED_ARC1_COMMIT,
        },
        "query_blind_contract": {
            "test_output_sentinel": "exact copy of the corresponding test input",
            "gold_output_present": False,
            "provider_may_read": "LODO/full demonstrations and one test input only",
            "scoring_phase": "after content-addressed repair-candidate freeze",
        },
        "blind_data_root": str(
            (args.output_dir / "provider_blind" / "ARC-AGI-2").resolve()
        ),
        "tasks": provider_records,
    }
    provider_manifest = {
        "cohort_id": canonical_sha256(provider_content),
        **provider_content,
    }
    _write_new(args.output_dir / "provider_manifest.json", provider_manifest)

    cohort_content = {
        "schema": TRACE_REPAIR_COHORT_SCHEMA,
        "selection": {
            "source_split": "ARC-GEN generated query-blind repair-opportunity cohort",
            "seed": SELECTION_SEED,
            "family_order": "sha256(seed + NUL + family_id), ascending",
            "retention": (
                "first 12 execution-valid scene-AST near misses with at least one "
                "content-novel one-existing-slot demo-exact edit"
            ),
            "task_limit": TASK_LIMIT,
            "eligible_generator_count": len(eligible),
            "scanned_family_count": scanned,
            "explicit_exposed_family_count": len(EXPOSED_FAMILIES),
            "authored_exposure_count": len(discovered_exposures),
            "excluded_generator_exposure_count": len(generator_ids & all_exposures),
            "arc1_task_count": len(arc1_ids),
            "initial_program_trials": INITIAL_PROGRAM_TRIALS,
            "max_near_misses": MAX_NEAR_MISSES,
            "minimum_near_miss_agreement": MINIMUM_NEAR_MISS_AGREEMENT,
            "max_scene_programs": MAX_SCENE_PROGRAMS,
            "family_scan_timeout_seconds": FAMILY_SCAN_TIMEOUT_SECONDS,
            "scan_workers": SCAN_WORKERS,
            "selection_reason_counts": dict(sorted(selection_reason_counts.items())),
        },
        "source": {
            "arcgen_commit": EXPECTED_ARCGEN_COMMIT,
            "arc1_submodule_commit": EXPECTED_ARC1_COMMIT,
            "arcgen_root": str(args.arcgen_root.resolve()),
        },
        "query_blind_contract": {
            "selection_reads": "demonstration outputs and query inputs",
            "selection_does_not_read": "query outputs or VARC predictions",
            "provider_test_output": "exact input-copy sentinel",
            "gold_release": "after byte-identical candidate replay",
        },
        "provider_cohort_id": provider_manifest["cohort_id"],
        "provider_manifest_sha256": _file_sha256(
            args.output_dir / "provider_manifest.json"
        ),
        "tasks": records,
    }
    manifest = {"cohort_id": canonical_sha256(cohort_content), **cohort_content}
    _write_new(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "cohort_id": manifest["cohort_id"],
                "provider_cohort_id": provider_manifest["cohort_id"],
                "task_ids": [record["task_id"] for record in records],
                "provider_episode_count": len(provider_records),
                "scanned_family_count": scanned,
                "repairable_slot_counts": dict(sorted(repairable_slot_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
