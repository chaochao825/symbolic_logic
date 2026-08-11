"""Build a query-gold-blind terminal failure matrix for relational delta v0.2."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.blind import BlindTask  # noqa: E402
from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    canonical_json,
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)
from afts_arc.grid import Grid, as_grid  # noqa: E402
from afts_arc.relational_delta import (  # noqa: E402
    RELATIONAL_DELTA_DSL_VERSION,
    RelationalDeltaProgram,
    enumerate_relational_delta_programs,
    execute_relational_delta,
    relational_delta_program_id,
    score_relational_delta_program,
)
from afts_arc.task import ARCPair  # noqa: E402


MATRIX_SCHEMA = "afts.relational-delta-terminal-failure-matrix/v1"
EXPECTED_FREEZE_ID = "85811f28942fc447d529be60e217d4cd4b5326c485ea901dda3d19b983fe32e5"
EXPECTED_COHORT_ID = "dfee67cf54613ba721e1fe46723967c95020715dff1a40500fe498055805e3ac"
EXPECTED_FAILURE_COUNT = 48
TERMINAL_CATEGORIES = (
    "canvas_incompatible",
    "parse_failure",
    "relation_missing",
    "ast_insufficient",
    "legal_but_inexact",
)
PARSE_FAILURE_REASONS = frozenset({"ambiguous_roles"})
RELATION_FAILURE_REASONS = frozenset({"no_unique_correspondence"})
AST_FAILURE_REASONS = frozenset(
    {
        "target_side_missing",
        "conflicting_target_colors",
        "delta_out_of_bounds",
        "target_overwrites_preserved_object",
        "topology_violation",
        "empty_effective_delta",
    }
)

Coordinate = tuple[int, int]


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _validate_content_id(payload: Mapping[str, object], field: str) -> None:
    body = dict(payload)
    declared = body.pop(field)
    if declared != canonical_sha256(body):
        raise ValueError(f"{field} does not match canonical content")


def _blind_task(payload: Mapping[str, object]) -> BlindTask:
    if set(payload) != {"train", "test"}:
        raise ValueError("blind task has missing or unknown fields")
    train = payload["train"]
    test = payload["test"]
    if not isinstance(train, list) or not train:
        raise ValueError("blind task requires demonstrations")
    if not isinstance(test, list) or not test:
        raise ValueError("blind task requires query inputs")
    for query in test:
        if not isinstance(query, Mapping) or set(query) != {"input", "output"}:
            raise ValueError("blind query must contain input and sentinel output")
        if query["output"] != query["input"]:
            raise ValueError("blind query output is not the input-copy sentinel")
    return BlindTask.from_observations(
        train=tuple(
            ARCPair(as_grid(pair["input"]), as_grid(pair["output"])) for pair in train
        ),
        test_inputs=tuple(as_grid(pair["input"]) for pair in test),
    )


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _delta(first: Grid, second: Grid) -> set[Coordinate]:
    if _shape(first) != _shape(second):
        raise ValueError("delta requires same-shape grids")
    return {
        (row, column)
        for row in range(len(first))
        for column in range(len(first[0]))
        if first[row][column] != second[row][column]
    }


def _component_count(cells: set[Coordinate]) -> int:
    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            row, column = frontier.pop()
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbor = row + dr, column + dc
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    return count


def _contains_subgrid(container: Grid, target: Grid) -> bool:
    container_height, container_width = _shape(container)
    target_height, target_width = _shape(target)
    if target_height > container_height or target_width > container_width:
        return False
    return any(
        all(
            container[top + row][left : left + target_width] == target[row]
            for row in range(target_height)
        )
        for top in range(container_height - target_height + 1)
        for left in range(container_width - target_width + 1)
    )


def _canvas_relation(source: Grid, target: Grid) -> str:
    if _shape(source) == _shape(target):
        return "same_canvas"
    if _contains_subgrid(source, target):
        return "output_is_input_crop"
    if _contains_subgrid(target, source):
        return "input_embeds_in_output"
    source_height, source_width = _shape(source)
    target_height, target_width = _shape(target)
    if target_height % source_height == 0 and target_width % source_width == 0:
        return "integer_canvas_expansion"
    if source_height % target_height == 0 and source_width % target_width == 0:
        return "integer_canvas_reduction"
    return "other_shape_change"


def _demo_features(task: BlindTask) -> list[dict[str, object]]:
    rows = []
    for demo_index, pair in enumerate(task.train):
        source_shape = _shape(pair.input)
        target_shape = _shape(pair.output)
        canvas_compatible = source_shape == target_shape
        changed_cells: set[Coordinate] | None = None
        transitions: Counter[str] = Counter()
        if canvas_compatible:
            changed_cells = _delta(pair.input, pair.output)
            transitions.update(
                f"{pair.input[row][column]}->{pair.output[row][column]}"
                for row, column in changed_cells
            )
        input_colors = sorted({cell for row in pair.input for cell in row})
        output_colors = sorted({cell for row in pair.output for cell in row})
        rows.append(
            {
                "demo_index": demo_index,
                "input_shape": list(source_shape),
                "output_shape": list(target_shape),
                "canvas_compatible": canvas_compatible,
                "canvas_relation": _canvas_relation(pair.input, pair.output),
                "changed_cell_count": (
                    len(changed_cells) if changed_cells is not None else None
                ),
                "changed_fraction": (
                    len(changed_cells) / (source_shape[0] * source_shape[1])
                    if changed_cells is not None
                    else None
                ),
                "delta_component_count": (
                    _component_count(changed_cells)
                    if changed_cells is not None
                    else None
                ),
                "color_transitions": dict(sorted(transitions.items())),
                "input_colors": input_colors,
                "output_colors": output_colors,
                "added_colors": sorted(set(output_colors) - set(input_colors)),
                "removed_colors": sorted(set(input_colors) - set(output_colors)),
            }
        )
    return rows


def _program_stage_reach(
    task: BlindTask,
    program: RelationalDeltaProgram,
) -> tuple[bool, bool, bool, tuple[str, ...]]:
    executions = tuple(
        execute_relational_delta(program, pair.input) for pair in task.train
    )
    reasons = tuple(
        execution.reason if execution.reason is not None else "ok"
        for execution in executions
    )
    known_reasons = (
        PARSE_FAILURE_REASONS | RELATION_FAILURE_REASONS | AST_FAILURE_REASONS
    )
    unknown = set(reasons) - known_reasons - {"ok"}
    if unknown:
        raise ValueError(f"unclassified relational-delta execution reasons: {unknown}")
    roles_on_all_demos = all(reason not in PARSE_FAILURE_REASONS for reason in reasons)
    relation_on_all_demos = all(
        reason not in PARSE_FAILURE_REASONS | RELATION_FAILURE_REASONS
        for reason in reasons
    )
    legal_on_all_demos = all(execution.ok for execution in executions)
    return roles_on_all_demos, relation_on_all_demos, legal_on_all_demos, reasons


def classify_terminal_failure(
    task: BlindTask,
    programs: Sequence[RelationalDeltaProgram],
) -> dict[str, object]:
    if not programs:
        raise ValueError("terminal classification requires at least one program")
    canvas_compatible = all(
        _shape(pair.input) == _shape(pair.output) for pair in task.train
    )
    reason_counts: Counter[str] = Counter()
    role_program_count = 0
    relation_program_count = 0
    legal_program_count = 0
    scores = []
    for program in programs:
        roles, relation, legal, reasons = _program_stage_reach(task, program)
        role_program_count += int(roles)
        relation_program_count += int(relation)
        legal_program_count += int(legal)
        reason_counts.update(reasons)
        scores.append(score_relational_delta_program(program, task))
    exact_programs = [score for score in scores if score.all_demo_exact]
    if exact_programs:
        raise ValueError("failure matrix received a demonstration-exact task")

    if not canvas_compatible:
        category = "canvas_incompatible"
    elif role_program_count == 0:
        category = "parse_failure"
    elif relation_program_count == 0:
        category = "relation_missing"
    elif legal_program_count == 0:
        category = "ast_insufficient"
    else:
        category = "legal_but_inexact"

    legal_scores = [score for score in scores if score.execution_valid]
    best_legal = (
        min(
            legal_scores,
            key=lambda score: (
                score.mismatch_count,
                -score.exact_demo_count,
                score.program.description_bits,
                relational_delta_program_id(score.program),
            ),
        )
        if legal_scores
        else None
    )
    identity_mismatch_count = None
    if canvas_compatible:
        identity_mismatch_count = sum(
            len(_delta(pair.input, pair.output)) for pair in task.train
        )
    best_payload = None
    if best_legal is not None:
        best_payload = {
            "program_id": relational_delta_program_id(best_legal.program),
            "program": best_legal.program.to_json_dict(),
            "exact_demo_count": best_legal.exact_demo_count,
            "mismatch_count": best_legal.mismatch_count,
            "improvement_over_identity": (
                identity_mismatch_count - best_legal.mismatch_count
                if identity_mismatch_count is not None
                else None
            ),
        }
    return {
        "primary_category": category,
        "canvas_compatible": canvas_compatible,
        "program_count": len(programs),
        "programs_reaching_roles_on_all_demos": role_program_count,
        "programs_reaching_relation_on_all_demos": relation_program_count,
        "programs_legal_on_all_demos": legal_program_count,
        "execution_reason_counts": dict(sorted(reason_counts.items())),
        "identity_mismatch_count": identity_mismatch_count,
        "best_legal_program": best_payload,
        "demonstrations": _demo_features(task),
    }


def _source_contract() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "afts_arc" / "relational_delta.py",
        PROJECT_ROOT
        / "notes"
        / "design"
        / "relational-delta-v0.2-failure-matrix-protocol.md",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
        for path in paths
    }


def construct_matrix(
    candidate_freeze_path: Path,
    cohort_manifest_path: Path,
) -> dict[str, object]:
    freeze = _load_object(candidate_freeze_path)
    manifest = _load_object(cohort_manifest_path)
    _validate_content_id(freeze, "freeze_id")
    if freeze["freeze_id"] != EXPECTED_FREEZE_ID:
        raise ValueError("candidate freeze ID mismatch")
    if freeze["cohort_id"] != EXPECTED_COHORT_ID:
        raise ValueError("candidate freeze cohort mismatch")
    if manifest["cohort_id"] != EXPECTED_COHORT_ID:
        raise ValueError("cohort manifest ID mismatch")
    if freeze["relational_delta_dsl_version"] != RELATIONAL_DELTA_DSL_VERSION:
        raise ValueError("relational delta DSL version mismatch")
    if freeze["query_gold_read"] is not False:
        raise ValueError("candidate freeze is not query blind")
    if freeze["public_evaluation_read"] is not False:
        raise ValueError("candidate freeze read public evaluation")
    if freeze["construction_replay_identical"] is not True:
        raise ValueError("candidate freeze is not replay-identical")
    frozen_module_hash = freeze["source_contract"]["src/afts_arc/relational_delta.py"]
    if frozen_module_hash != file_sha256(
        PROJECT_ROOT / "src" / "afts_arc" / "relational_delta.py"
    ):
        raise ValueError("relational delta source no longer matches candidate freeze")
    blind_root = Path(manifest["blind_task_root"])
    manifest_by_task = {task["task_id"]: task for task in manifest["tasks"]}
    rows = []
    for frozen_task in freeze["tasks"]:
        if frozen_task["full"]["demo_exact_program_count"] != 0:
            continue
        task_id = frozen_task["task_id"]
        task_manifest = manifest_by_task[task_id]
        if task_manifest["blind_sha256"] != frozen_task["blind_sha256"]:
            raise ValueError(f"manifest blind hash mismatch: {task_id}")
        blind_path = blind_root / f"{task_id}.json"
        if file_sha256(blind_path) != frozen_task["blind_sha256"]:
            raise ValueError(f"blind task hash mismatch: {task_id}")
        task = _blind_task(_load_object(blind_path))
        programs = enumerate_relational_delta_programs(task)
        if len(programs) != frozen_task["grammar_program_count"]:
            raise ValueError(f"grammar size no longer matches freeze: {task_id}")
        classification = classify_terminal_failure(task, programs)
        rows.append(
            {
                "task_id": task_id,
                "blind_sha256": frozen_task["blind_sha256"],
                "demo_count": frozen_task["demo_count"],
                "query_count": frozen_task["query_count"],
                "quality_parent_count": frozen_task["repair"]["quality_parent_count"],
                **classification,
            }
        )
    rows.sort(key=lambda row: row["task_id"])
    if len(rows) != EXPECTED_FAILURE_COUNT:
        raise ValueError("failure matrix does not contain exactly 48 tasks")
    category_counts = Counter(row["primary_category"] for row in rows)
    if set(category_counts) - set(TERMINAL_CATEGORIES):
        raise ValueError("failure matrix contains an unknown terminal category")
    if sum(category_counts.values()) != EXPECTED_FAILURE_COUNT:
        raise ValueError("terminal categories are not exhaustive")
    content = {
        "schema": MATRIX_SCHEMA,
        "status": "outcome_exposed_development_diagnostic",
        "candidate_freeze_id": freeze["freeze_id"],
        "candidate_freeze_sha256": file_sha256(candidate_freeze_path),
        "cohort_id": freeze["cohort_id"],
        "cohort_manifest_sha256": file_sha256(cohort_manifest_path),
        "relational_delta_dsl_version": RELATIONAL_DELTA_DSL_VERSION,
        "source_contract": _source_contract(),
        "query_gold_read": False,
        "public_evaluation_read": False,
        "classification_precedence": list(TERMINAL_CATEGORIES),
        "failure_task_count": len(rows),
        "category_counts": {
            category: category_counts[category] for category in TERMINAL_CATEGORIES
        },
        "runtime": runtime_metadata(),
        "tasks": rows,
    }
    return {"matrix_id": canonical_sha256(content), **content}


def _csv_text(matrix: Mapping[str, object]) -> str:
    fields = (
        "task_id",
        "primary_category",
        "canvas_compatible",
        "demo_count",
        "query_count",
        "program_count",
        "programs_reaching_roles_on_all_demos",
        "programs_reaching_relation_on_all_demos",
        "programs_legal_on_all_demos",
        "identity_mismatch_count",
        "best_legal_mismatch_count",
        "best_legal_improvement_over_identity",
        "quality_parent_count",
        "execution_reason_counts",
        "canvas_relations",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for task in matrix["tasks"]:
        best = task["best_legal_program"]
        writer.writerow(
            {
                "task_id": task["task_id"],
                "primary_category": task["primary_category"],
                "canvas_compatible": task["canvas_compatible"],
                "demo_count": task["demo_count"],
                "query_count": task["query_count"],
                "program_count": task["program_count"],
                "programs_reaching_roles_on_all_demos": task[
                    "programs_reaching_roles_on_all_demos"
                ],
                "programs_reaching_relation_on_all_demos": task[
                    "programs_reaching_relation_on_all_demos"
                ],
                "programs_legal_on_all_demos": task["programs_legal_on_all_demos"],
                "identity_mismatch_count": task["identity_mismatch_count"],
                "best_legal_mismatch_count": (
                    best["mismatch_count"] if best is not None else ""
                ),
                "best_legal_improvement_over_identity": (
                    best["improvement_over_identity"] if best is not None else ""
                ),
                "quality_parent_count": task["quality_parent_count"],
                "execution_reason_counts": canonical_json(
                    task["execution_reason_counts"]
                ),
                "canvas_relations": canonical_json(
                    [demo["canvas_relation"] for demo in task["demonstrations"]]
                ),
            }
        )
    return stream.getvalue()


def _write_new_text(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-freeze", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    first = construct_matrix(args.candidate_freeze, args.cohort_manifest)
    second = construct_matrix(args.candidate_freeze, args.cohort_manifest)
    if canonical_json(first) != canonical_json(second):
        raise ValueError("failure matrix construction is not replay-identical")
    atomic_write_json(args.output, first)
    _write_new_text(args.csv_output, _csv_text(first))
    print(json.dumps(first, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
