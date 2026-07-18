"""Bounded residual-directed repair policies.

Repairs are learned from demonstration residuals only, compose with an already
hard-verifiable parent rule, and must pass the same demo-exact/hard gate again.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence

from ..blind import BlindTask
from ..grid import Grid, as_grid
from .types import CandidateEvaluation, CandidateHypothesis, RepairReceipt, canonical_json


REPAIR_VERSION = "afts-residual-repair/v1"
LOCAL_SPECS: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("von_neumann_r1", ((0, 0), (-1, 0), (0, -1), (0, 1), (1, 0))),
    (
        "moore_r1",
        ((0, 0),)
        + tuple(
            (row, column)
            for row in (-1, 0, 1)
            for column in (-1, 0, 1)
            if (row, column) != (0, 0)
        ),
    ),
)


def _grid(value: object | None) -> Grid | None:
    if value is None:
        return None
    try:
        return as_grid(value.tolist() if hasattr(value, "tolist") else value)
    except (TypeError, ValueError):
        return None


def _parent_ready(
    evaluation: CandidateEvaluation,
) -> str | None:
    if evaluation.hypothesis.replay is None:
        return "non_replayable_parent"
    if not evaluation.hard_verified:
        return "unverified_parent"
    if evaluation.demo_exact:
        return "already_exact"
    if any(item is None for item in evaluation.demo_outputs):
        return "invalid_parent_execution"
    if any(not item.shape_match for item in evaluation.residuals):
        return "shape_mismatch"
    return None


def _composed_verifier(
    parent: CandidateHypothesis,
    replay,
    spec_replay,
    mapping_valid: bool,
):
    def verify(task: BlindTask) -> bool:
        if not mapping_valid or parent.hard_verifier is None:
            return False
        if parent.hard_verifier(task) is not True:
            return False
        for grid in tuple(pair.input for pair in task.train) + task.test_inputs:
            if _grid(replay(grid)) != _grid(spec_replay(grid)) or _grid(replay(grid)) is None:
                return False
        return True

    return verify


def global_color_map_repair(
    evaluation: CandidateEvaluation,
    task: BlindTask,
) -> tuple[tuple[CandidateHypothesis, ...], RepairReceipt]:
    parent = evaluation.hypothesis
    reason = _parent_ready(evaluation)
    if reason is not None:
        return (), RepairReceipt(parent.hypothesis_id, "global_color_map", "abstained", reason)

    targets: dict[int, set[int]] = defaultdict(set)
    for predicted, pair in zip(evaluation.demo_outputs, task.train):
        assert predicted is not None and pair.output is not None
        for predicted_row, target_row in zip(predicted, pair.output):
            for source, target in zip(predicted_row, target_row):
                targets[source].add(target)
    if any(len(values) != 1 for values in targets.values()):
        return (), RepairReceipt(
            parent.hypothesis_id,
            "global_color_map",
            "abstained",
            "ambiguous_color_mapping",
        )
    changes = {
        source: next(iter(values))
        for source, values in targets.items()
        if next(iter(values)) != source
    }
    if not changes:
        return (), RepairReceipt(parent.hypothesis_id, "global_color_map", "abstained", "no_change")

    repair_record = {
        "strategy": "global_color_map",
        "mapping": {str(source): target for source, target in sorted(changes.items())},
    }
    repair_json = canonical_json(repair_record)

    def replay(grid: Grid) -> object | None:
        assert parent.replay is not None
        base = _grid(parent.replay(grid))
        if base is None:
            return None
        return tuple(tuple(changes.get(cell, cell) for cell in row) for row in base)

    def spec_replay(grid: Grid) -> object | None:
        assert parent.replay is not None
        base = _grid(parent.replay(grid))
        if base is None:
            return None
        record = json.loads(repair_json)
        mapping = {int(source): int(target) for source, target in record["mapping"].items()}
        return tuple(tuple(mapping.get(cell, cell) for cell in row) for row in base)

    added_bits = 5 + 8 * len(changes)
    candidate = CandidateHypothesis.create(
        name=f"repair:color_map:{parent.hypothesis_id}",
        source="residual_repair",
        source_version=REPAIR_VERSION,
        route="residual_repair",
        description_bits=parent.description_bits + added_bits,
        verification_mode="replayable",
        functional_trace=(*parent.functional_trace, "residual_repair:global_color_map"),
        spec={
            "parent_hypothesis_id": parent.hypothesis_id,
            "repair": repair_record,
        },
        metadata={
            "repair_description_bits": added_bits,
            "query_support": parent.metadata.get("query_support", 1.0),
        },
        parent_hypothesis_ids=(parent.hypothesis_id,),
        replay=replay,
        hard_verifier=_composed_verifier(
            parent,
            replay,
            spec_replay,
            all(0 <= source <= 9 and 0 <= target <= 9 for source, target in changes.items()),
        ),
    )
    return (candidate,), RepairReceipt(
        parent.hypothesis_id,
        "global_color_map",
        "produced",
        None,
        (candidate.hypothesis_id,),
    )


def _patch(
    grid: Grid,
    row: int,
    column: int,
    offsets: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    height, width = len(grid), len(grid[0])
    return tuple(
        grid[row + row_delta][column + column_delta]
        if 0 <= row + row_delta < height and 0 <= column + column_delta < width
        else 10
        for row_delta, column_delta in offsets
    )


def local_transition_repair(
    evaluation: CandidateEvaluation,
    task: BlindTask,
    *,
    minimum_exception_support: int = 2,
) -> tuple[tuple[CandidateHypothesis, ...], RepairReceipt]:
    parent = evaluation.hypothesis
    reason = _parent_ready(evaluation)
    if reason is not None:
        return (), RepairReceipt(parent.hypothesis_id, "local_transition", "abstained", reason)
    if len(task.train) < 2:
        return (), RepairReceipt(
            parent.hypothesis_id,
            "local_transition",
            "abstained",
            "insufficient_demonstrations",
        )

    proposals: list[tuple[int, str, tuple[tuple[int, int], ...], dict[tuple[int, ...], int]]] = []
    for spec_name, offsets in LOCAL_SPECS:
        targets: dict[tuple[int, ...], set[int]] = defaultdict(set)
        support: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for demo_index, (predicted, pair) in enumerate(zip(evaluation.demo_outputs, task.train)):
            assert predicted is not None and pair.output is not None
            for row in range(len(predicted)):
                for column in range(len(predicted[0])):
                    pattern = _patch(predicted, row, column, offsets)
                    targets[pattern].add(pair.output[row][column])
                    support[pattern].add(demo_index)
        if any(len(values) != 1 for values in targets.values()):
            continue
        exceptions = {
            pattern: next(iter(values))
            for pattern, values in targets.items()
            if next(iter(values)) != pattern[0]
        }
        if not exceptions or any(
            len(support[pattern]) < minimum_exception_support for pattern in exceptions
        ):
            continue
        added_bits = 8 + 6 * len(offsets) + len(exceptions) * (4 * len(offsets) + 4)
        proposals.append((added_bits, spec_name, offsets, exceptions))
    if not proposals:
        return (), RepairReceipt(
            parent.hypothesis_id,
            "local_transition",
            "abstained",
            "no_high_support_local_rule",
        )
    added_bits, spec_name, offsets, exceptions = min(
        proposals,
        key=lambda item: (item[0], item[1]),
    )

    repair_record = {
        "strategy": "local_transition",
        "neighborhood": spec_name,
        "offsets": [list(offset) for offset in offsets],
        "entries": [
            {"pattern": list(pattern), "output": target}
            for pattern, target in sorted(exceptions.items())
        ],
        "minimum_exception_support": minimum_exception_support,
    }
    repair_json = canonical_json(repair_record)

    def apply_table(base: Grid, selected_offsets, selected_exceptions) -> list[list[int]]:
        output = [list(row) for row in base]
        for row in range(len(base)):
            for column in range(len(base[0])):
                pattern = _patch(base, row, column, selected_offsets)
                if pattern in selected_exceptions:
                    output[row][column] = selected_exceptions[pattern]
        return output

    def replay(grid: Grid) -> object | None:
        assert parent.replay is not None
        base = _grid(parent.replay(grid))
        if base is None:
            return None
        return apply_table(base, offsets, exceptions)

    def spec_replay(grid: Grid) -> object | None:
        assert parent.replay is not None
        base = _grid(parent.replay(grid))
        if base is None:
            return None
        record = json.loads(repair_json)
        selected_offsets = tuple(
            tuple(int(value) for value in offset) for offset in record["offsets"]
        )
        selected_exceptions = {
            tuple(int(value) for value in item["pattern"]): int(item["output"])
            for item in record["entries"]
        }
        return apply_table(base, selected_offsets, selected_exceptions)

    candidate = CandidateHypothesis.create(
        name=f"repair:local:{spec_name}:{parent.hypothesis_id}",
        source="residual_repair",
        source_version=REPAIR_VERSION,
        route="residual_repair",
        description_bits=parent.description_bits + added_bits,
        verification_mode="replayable",
        functional_trace=(*parent.functional_trace, f"residual_repair:{spec_name}"),
        spec={
            "parent_hypothesis_id": parent.hypothesis_id,
            "repair": repair_record,
        },
        metadata={
            "repair_description_bits": added_bits,
            "query_support": parent.metadata.get("query_support", 1.0),
        },
        parent_hypothesis_ids=(parent.hypothesis_id,),
        replay=replay,
        hard_verifier=_composed_verifier(
            parent,
            replay,
            spec_replay,
            all(
                len(pattern) == len(offsets)
                and all(0 <= value <= 10 for value in pattern)
                and 0 <= target <= 9
                for pattern, target in exceptions.items()
            ),
        ),
    )
    return (candidate,), RepairReceipt(
        parent.hypothesis_id,
        "local_transition",
        "produced",
        None,
        (candidate.hypothesis_id,),
    )


def generate_repairs(
    evaluations: Sequence[CandidateEvaluation],
    task: BlindTask,
    *,
    max_parents: int = 8,
    minimum_agreement: float = 0.5,
) -> tuple[tuple[CandidateHypothesis, ...], tuple[RepairReceipt, ...]]:
    parents = sorted(
        (
            item
            for item in evaluations
            if not item.demo_exact
            and item.hard_verified
            and item.hypothesis.replay is not None
            and item.agreement >= minimum_agreement
        ),
        key=lambda item: (
            item.residual_bits,
            item.hypothesis.description_bits,
            item.hypothesis.hypothesis_id,
        ),
    )[:max_parents]
    candidates: dict[str, CandidateHypothesis] = {}
    receipts: list[RepairReceipt] = []
    for parent in parents:
        for strategy in (global_color_map_repair, local_transition_repair):
            produced, receipt = strategy(parent, task)
            receipts.append(receipt)
            for candidate in produced:
                candidates[candidate.hypothesis_id] = candidate
    return (
        tuple(sorted(candidates.values(), key=lambda item: item.hypothesis_id)),
        tuple(receipts),
    )
