from __future__ import annotations

import pytest

from afts_arc.hybrid.control import FrozenActionBatch
from afts_arc.hybrid.metareasoning import (
    NativeBudgetLedger,
    NativeCostContract,
    NativeCostReservation,
    NativeCostVector,
    conditional_mutual_information,
)
from afts_arc.hybrid.types import CandidateHypothesis


def _grid_candidate() -> CandidateHypothesis:
    output = (((1,),),)
    return CandidateHypothesis.create(
        name="grid:test",
        source="test",
        source_version="test/v1",
        route="dsl_program",
        description_bits=1,
        verification_mode="grid_only",
        functional_trace=("test",),
        spec={"kind": "test"},
        explicit_demo_outputs=output,
        explicit_query_outputs=output,
    )


def test_native_cost_vector_is_canonical_and_additive() -> None:
    first = NativeCostVector.from_mapping(
        {"program": {"trials": 4}, "wall_seconds": 0.5},
        prefix="typed_dsl",
    )
    second = NativeCostVector.from_mapping(
        {"program": {"trials": 3}}, prefix="typed_dsl"
    )

    assert first.items == (
        ("typed_dsl.program.trials", 4.0),
        ("typed_dsl.wall_seconds", 0.5),
    )
    assert (first + second).to_mapping()["typed_dsl.program.trials"] == 7.0
    assert (
        first.normalized_total(
            {
                "typed_dsl.program.trials": 4.0,
                "typed_dsl.wall_seconds": 0.5,
            }
        )
        == 2.0
    )


def test_native_budget_ledger_rejects_undeclared_or_excess_work() -> None:
    limit = NativeCostVector.from_mapping({"trials": 10})
    ledger = NativeBudgetLedger(limit)
    ledger = ledger.charge(NativeCostVector.from_mapping({"trials": 6}))

    assert ledger.can_reserve(NativeCostVector.from_mapping({"trials": 4}))
    assert not ledger.can_reserve(NativeCostVector.from_mapping({"trials": 5}))
    assert not ledger.can_reserve(NativeCostVector.from_mapping({"tokens": 1}))
    with pytest.raises(ValueError, match="remaining budget"):
        ledger.charge(NativeCostVector.from_mapping({"trials": 5}))


def test_native_cost_contract_round_trips_with_content_id() -> None:
    contract = NativeCostContract(
        (
            NativeCostReservation(
                "typed_dsl",
                "synthesize",
                NativeCostVector.from_mapping({"program_trials": 10}),
            ),
        )
    )

    rebuilt = NativeCostContract.from_json_dict(contract.to_json_dict())
    assert rebuilt == contract
    assert rebuilt.contract_id == contract.contract_id


def test_frozen_batch_identity_carries_realized_native_cost() -> None:
    candidate = _grid_candidate()
    cheap = FrozenActionBatch(
        "synthesize",
        (candidate,),
        native_cost=NativeCostVector.from_mapping({"program_trials": 1}),
    )
    expensive = FrozenActionBatch(
        "synthesize",
        (candidate,),
        native_cost=NativeCostVector.from_mapping({"program_trials": 2}),
    )

    assert cheap.batch_id != expensive.batch_id
    assert cheap.to_json_dict()["native_cost"]["items"] == [["program_trials", 1.0]]


def test_conditional_mutual_information_handles_explained_dependence() -> None:
    x = (0, 0, 1, 1)
    y = (0, 0, 1, 1)

    assert conditional_mutual_information(x, y, (0, 0, 0, 0)) == pytest.approx(1.0)
    assert conditional_mutual_information(x, y, x) == pytest.approx(0.0)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_native_cost_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        NativeCostVector.from_mapping({"work": value})
