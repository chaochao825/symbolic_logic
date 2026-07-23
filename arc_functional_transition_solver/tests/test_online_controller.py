from __future__ import annotations

import unittest
from dataclasses import dataclass

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.hybrid import (
    BudgetVector,
    CandidateHypothesis,
    ControlAction,
    FixedSchedulePolicy,
    FrozenCandidatePoolProvider,
    OnlineControlConfig,
    OnlineFunctionalRouterSolver,
    ProviderResult,
    aggregate_control_metrics,
    evaluate_online_report_with_oracle,
)
from afts_arc.task import ARCPair, ARCTask


def _blind(
    train: tuple[tuple[list[list[int]], list[list[int]]], ...],
    queries: tuple[list[list[int]], ...],
) -> BlindTask:
    return BlindTask.from_observations(
        train=tuple(
            ARCPair(as_grid(source), as_grid(target)) for source, target in train
        ),
        test_inputs=tuple(as_grid(grid) for grid in queries),
    )


def _hypothesis(
    name: str,
    replay,
    *,
    route: str,
    bits: int = 8,
) -> CandidateHypothesis:
    return CandidateHypothesis.create(
        name=name,
        source="online_test",
        source_version="online-test/v1",
        route=route,
        description_bits=bits,
        verification_mode="replayable",
        functional_trace=(name,),
        spec={"name": name},
        replay=replay,
        hard_verifier=lambda task: True,
    )


def _budget(*, steps: int, provider_calls: int, repairs: int) -> BudgetVector:
    return BudgetVector(
        compute_units=2 * steps,
        controller_steps=steps,
        provider_calls=provider_calls,
        repair_attempts=repairs,
        candidate_slots=steps,
    )


@dataclass
class ResidualAwareCodeProvider:
    candidate: CandidateHypothesis
    expected_parent_id: str
    name: str = "residual-code"
    route: str = "code_llm"
    strict_budget_contract: bool = True
    max_control_calls: int = 1
    supports_residual_actions: bool = True

    def act(self, task, features, decision, blackboard, action):
        self.assertions = {
            "blind": isinstance(task, BlindTask),
            "operator": action.operator,
            "parent": action.parent_hypothesis_id,
            "evidence": action.evidence_signal_ids,
            "shape_signal": any(
                item.shape_mismatch_count for item in blackboard.residual_signals
            ),
        }
        if action.operator != "exception_resynthesize":
            raise AssertionError("shape residual was not compiled into the code action")
        if action.parent_hypothesis_id != self.expected_parent_id:
            raise AssertionError("cross-representation action lost its residual parent")
        return ProviderResult.ok(self.name, self.route, (self.candidate,))


@dataclass
class OverrunningStrictProvider:
    candidates: tuple[CandidateHypothesis, ...]
    name: str = "overrun"
    route: str = "dsl_program"
    strict_budget_contract: bool = True
    max_control_calls: int = 1

    def act(self, task, features, decision, blackboard, action):
        return ProviderResult.ok(self.name, self.route, self.candidates)


class OnlineControllerTests(unittest.TestCase):
    def test_residual_policy_beats_fixed_run_all_at_identical_ncu(self) -> None:
        task = _blind(
            (
                ([[0, 1], [1, 0]], [[0, 2], [2, 0]]),
                ([[1, 0]], [[2, 0]]),
            ),
            ([[0, 1]],),
        )
        near = _hypothesis("near-identity", lambda grid: grid, route="dsl_program")
        distractor = _hypothesis(
            "distractor",
            lambda grid: tuple(tuple(9 for _ in row) for row in grid),
            route="code_llm",
        )
        providers = (
            FrozenCandidatePoolProvider((near,), "near-pool", "dsl_program"),
            FrozenCandidatePoolProvider((distractor,), "distractor-pool", "code_llm"),
        )
        config = OnlineControlConfig(
            budget_limit=_budget(steps=2, provider_calls=2, repairs=1),
            provider_batch_size=1,
            max_selected_hypotheses=1,
        )
        residual = OnlineFunctionalRouterSolver(
            providers=providers,
            config=config,
        ).solve(task)
        fixed = OnlineFunctionalRouterSolver(
            providers=providers,
            config=config,
            policy=FixedSchedulePolicy(("near-pool", "distractor-pool")),
        ).solve(task)

        self.assertEqual(
            residual.states[-1].budget.limit, fixed.states[-1].budget.limit
        )
        self.assertEqual(
            residual.states[-1].budget.used.compute_units,
            fixed.states[-1].budget.used.compute_units,
        )
        self.assertEqual(residual.states[-1].budget.used.compute_units, 4)
        self.assertEqual(residual.states[-1].budget.used.candidate_slots, 2)
        self.assertEqual(fixed.states[-1].budget.used.candidate_slots, 2)
        self.assertTrue(residual.strict_budget_comparable)
        self.assertTrue(fixed.strict_budget_comparable)
        self.assertEqual(residual.status, "solved")
        self.assertEqual(fixed.status, "abstained")
        self.assertIn(
            "residual_repair:global_color_map",
            residual.selected[0].hypothesis.functional_trace,
        )
        residual_actors = [
            item.action.actor for item in residual.states[-1].action_results
        ]
        self.assertNotIn("distractor-pool", residual_actors)
        self.assertEqual(
            [item.action.kind for item in residual.states[-1].action_results],
            ["propose", "repair", "stop"],
        )
        lineage = residual.lineage_candidate_records(task_id="online-repair")
        self.assertEqual(len(lineage), 2)
        self.assertFalse(lineage[0].parent_candidate_ids)
        self.assertEqual(lineage[1].parent_candidate_ids, (lineage[0].candidate_id,))

    def test_localized_ambiguous_color_residual_selects_local_repair(self) -> None:
        source = [[0, 2, 0], [0, 1, 0], [0, 0, 1]]
        target = [[0, 2, 0], [0, 3, 0], [0, 0, 1]]
        task = _blind(
            ((source, target), (source, target)),
            (source,),
        )
        near = _hypothesis("local-near", lambda grid: grid, route="dsl_program")
        report = OnlineFunctionalRouterSolver(
            providers=(
                FrozenCandidatePoolProvider((near,), "local-near-pool", "dsl_program"),
            ),
            config=OnlineControlConfig(
                budget_limit=_budget(steps=2, provider_calls=1, repairs=1),
                provider_batch_size=1,
                max_selected_hypotheses=1,
            ),
        ).solve(task)
        repair = report.states[-1].action_results[1]
        self.assertEqual(repair.action.kind, "repair")
        self.assertEqual(repair.action.operator, "local_transition")
        self.assertEqual(report.status, "solved")
        self.assertIn(
            "residual_repair:von_neumann_r1",
            report.selected[0].hypothesis.functional_trace,
        )

    def test_shape_failure_becomes_typed_cross_representation_code_action(self) -> None:
        task = _blind(
            (([[3]], [[3, 3], [3, 3]]),),
            ([[4]],),
        )
        wrong_shape = _hypothesis(
            "wrong-shape",
            lambda grid: grid,
            route="dsl_program",
        )
        exact = _hypothesis(
            "scale-two",
            lambda grid: (
                (grid[0][0], grid[0][0]),
                (grid[0][0], grid[0][0]),
            ),
            route="code_llm",
        )
        code = ResidualAwareCodeProvider(exact, wrong_shape.hypothesis_id)
        report = OnlineFunctionalRouterSolver(
            providers=(
                FrozenCandidatePoolProvider(
                    (wrong_shape,), "shape-pool", "dsl_program"
                ),
                code,
            ),
            config=OnlineControlConfig(
                budget_limit=_budget(steps=2, provider_calls=2, repairs=0),
                provider_batch_size=1,
                max_selected_hypotheses=1,
            ),
        ).solve(task)
        self.assertEqual(report.status, "solved")
        self.assertTrue(code.assertions["blind"])
        self.assertEqual(code.assertions["operator"], "exception_resynthesize")
        self.assertEqual(code.assertions["parent"], wrong_shape.hypothesis_id)
        self.assertTrue(code.assertions["evidence"])
        self.assertTrue(code.assertions["shape_signal"])
        second = report.states[-1].action_results[1].action
        self.assertEqual(second.route, "code_llm")
        self.assertIn("cross_representation_control", second.reason_codes)

    def test_stop_state_action_and_report_are_deterministic(self) -> None:
        task = _blind((([[1]], [[1]]),), ([[2]],))
        identity = _hypothesis("identity", lambda grid: grid, route="dsl_program")
        provider = FrozenCandidatePoolProvider(
            (identity,), "identity-pool", "dsl_program"
        )
        config = OnlineControlConfig(
            budget_limit=_budget(steps=1, provider_calls=1, repairs=0),
            provider_batch_size=1,
            max_selected_hypotheses=1,
        )
        first = OnlineFunctionalRouterSolver(
            providers=(provider,), config=config
        ).solve(task)
        second = OnlineFunctionalRouterSolver(
            providers=(provider,), config=config
        ).solve(task)
        self.assertEqual(first.to_json_dict(), second.to_json_dict())
        terminal = first.states[-1]
        self.assertTrue(terminal.stopped)
        self.assertEqual(terminal.stop_reason, "pass_at_k_filled")
        self.assertEqual(terminal.action_results[-1].action.kind, "stop")
        self.assertEqual(terminal.action_results[-1].action.budget, BudgetVector())
        self.assertEqual(
            len({item.state_id for item in first.states}), len(first.states)
        )

    def test_frozen_pool_streams_multiple_content_addressed_batches(self) -> None:
        task = _blind((([[1]], [[1]]),), ([[2]],))
        identity = _hypothesis("batch-identity", lambda grid: grid, route="dsl_program")
        alternate = _hypothesis(
            "batch-alternate",
            lambda grid: tuple(
                tuple(3 if cell == 2 else cell for cell in row) for row in grid
            ),
            route="dsl_program",
            bits=9,
        )
        provider = FrozenCandidatePoolProvider(
            (identity, alternate), "batched-pool", "dsl_program"
        )
        report = OnlineFunctionalRouterSolver(
            providers=(provider,),
            config=OnlineControlConfig(
                budget_limit=_budget(steps=2, provider_calls=2, repairs=0),
                provider_batch_size=1,
                max_selected_hypotheses=2,
            ),
        ).solve(task)
        proposals = tuple(
            item
            for item in report.states[-1].action_results
            if item.action.kind == "propose"
        )
        self.assertEqual(len(proposals), 2)
        self.assertEqual(len(report.selected), 2)
        self.assertEqual(report.states[-1].stop_reason, "pass_at_k_filled")

    def test_controller_is_invariant_to_hidden_test_oracle(self) -> None:
        def source_task(output: list[list[int]]) -> ARCTask:
            return ARCTask(
                task_id="online_oracle_probe",
                train=(ARCPair(as_grid([[1]]), as_grid([[1]])),),
                test=(ARCPair(as_grid([[2]]), as_grid(output)),),
                source_path="fixture",
                source_sha256="0" * 64,
            )

        first_task = BlindTask.from_task(source_task([[3]]))
        second_task = BlindTask.from_task(source_task([[9]]))
        identity = _hypothesis("oracle-free", lambda grid: grid, route="dsl_program")
        provider = FrozenCandidatePoolProvider(
            (identity,), "oracle-free-pool", "dsl_program"
        )
        config = OnlineControlConfig(
            budget_limit=_budget(steps=1, provider_calls=1, repairs=0),
            provider_batch_size=1,
            max_selected_hypotheses=1,
        )
        first = OnlineFunctionalRouterSolver(
            providers=(provider,), config=config
        ).solve(first_task)
        second = OnlineFunctionalRouterSolver(
            providers=(provider,), config=config
        ).solve(second_task)
        self.assertEqual(first.to_json_dict(), second.to_json_dict())

    def test_strict_provider_overrun_is_rejected_and_full_slice_is_charged(
        self,
    ) -> None:
        task = _blind((([[1]], [[1]]),), ([[2]],))
        first = _hypothesis("first", lambda grid: grid, route="dsl_program")
        second = _hypothesis("second", lambda grid: grid, route="dsl_program", bits=9)
        report = OnlineFunctionalRouterSolver(
            providers=(OverrunningStrictProvider((first, second)),),
            config=OnlineControlConfig(
                budget_limit=_budget(steps=1, provider_calls=1, repairs=0),
                provider_batch_size=1,
                max_selected_hypotheses=1,
            ),
        ).solve(task)
        proposal = report.states[-1].action_results[0]
        self.assertEqual(proposal.status, "error")
        self.assertEqual(proposal.reason, "budget_contract_violation")
        self.assertEqual(proposal.actual_candidate_evaluations, 0)
        self.assertEqual(report.states[-1].budget.used.compute_units, 2)
        self.assertFalse(report.selected)

    def test_illegal_cross_route_operator_is_rejected_by_type_system(self) -> None:
        with self.assertRaisesRegex(ValueError, "not legal"):
            ControlAction.create(
                state_id="state",
                kind="propose",
                actor="dsl",
                route="dsl_program",
                operator="masked_inpaint",
                budget=BudgetVector(
                    compute_units=2,
                    controller_steps=1,
                    provider_calls=1,
                    candidate_slots=1,
                ),
            )

    def test_oracle_metrics_are_posthoc_and_measure_coverage_utilization(self) -> None:
        source = ARCTask(
            task_id="posthoc_metrics",
            train=(
                ARCPair(as_grid([[0, 1]]), as_grid([[0, 2]])),
                ARCPair(as_grid([[1, 0]]), as_grid([[2, 0]])),
            ),
            test=(ARCPair(as_grid([[1, 0, 1]]), as_grid([[2, 0, 2]])),),
            source_path="fixture",
            source_sha256="1" * 64,
        )
        task = BlindTask.from_task(source)
        near = _hypothesis("posthoc-near", lambda grid: grid, route="dsl_program")
        report = OnlineFunctionalRouterSolver(
            providers=(
                FrozenCandidatePoolProvider((near,), "posthoc-pool", "dsl_program"),
            ),
            config=OnlineControlConfig(
                budget_limit=_budget(steps=2, provider_calls=1, repairs=1),
                provider_batch_size=1,
                max_selected_hypotheses=1,
            ),
        ).solve(task)
        metrics = evaluate_online_report_with_oracle(report, source)
        self.assertTrue(metrics.oracle_covered)
        self.assertTrue(metrics.pass_at_k)
        self.assertEqual(metrics.oracle_coverage_utilization, 1.0)
        self.assertEqual(metrics.correct_repair_count, 1)
        aggregate = aggregate_control_metrics((metrics,))
        self.assertEqual(aggregate.pass_rate, 1.0)
        self.assertEqual(aggregate.oracle_coverage_utilization, 1.0)


if __name__ == "__main__":
    unittest.main()
