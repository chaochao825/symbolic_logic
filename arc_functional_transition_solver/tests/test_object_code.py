from __future__ import annotations

import unittest

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.hybrid import (
    BudgetVector,
    D4LabelCompletionProgram,
    FixedSchedulePolicy,
    FrozenCandidatePoolProvider,
    ObjectCodeProvider,
    ObjectCodeResidualCompiler,
    OnlineControlConfig,
    OnlineFunctionalRouterSolver,
    RoleStampProgram,
    diagnose_object_code_failure,
    enumerate_object_code_programs,
    evaluate_hypothesis,
    execute_object_code_program,
    make_object_code_hypothesis,
    object_code_program_from_json,
    synthesize_object_code_programs,
    typed_repair_frontier,
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


def _role_task() -> tuple[BlindTask, tuple[tuple[int, ...], ...]]:
    source = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0],
        [0, 8, 3, 8, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    target = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 8, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    query = [
        [0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 8, 3, 8, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    expected = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 8, 0, 0, 0, 0],
            [0, 8, 0, 8, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    return _blind(((source, target),), (query,)), expected


def _d4_task() -> BlindTask:
    source = [[0 for _ in range(12)] for _ in range(7)]
    for row, column in (
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (1, 7),
        (2, 7),
        (3, 7),
        (3, 8),
    ):
        source[row][column] = 4
    source[1][2] = 1
    source[4][1] = 3
    source[1][8] = 1
    target = [row[:] for row in source]
    target[4][7] = 3
    query = [row[:] for row in source]
    query[4][7] = 0
    return _blind(((source, target),), (query,))


def _persistent_role_task() -> BlindTask:
    source = [
        [0, 0, 0, 0, 0],
        [0, 3, 8, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ]
    target = [row[:] for row in source]
    target[3][4] = 8
    query = [
        [0, 0, 3, 8, 0],
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    return _blind(((source, target),), (query,))


def _budget(steps: int, slots: int = 4) -> BudgetVector:
    return BudgetVector(
        compute_units=steps * (1 + slots),
        controller_steps=steps,
        provider_calls=steps,
        candidate_slots=steps * slots,
    )


class ObjectCodeTests(unittest.TestCase):
    def test_role_stamp_synthesis_is_demo_exact_and_replays_query(self) -> None:
        task, expected = _role_task()
        result = synthesize_object_code_programs(task)
        self.assertGreater(result.program_trial_count, 0)
        self.assertTrue(result.exact_scores)
        predictions = tuple(
            execute_object_code_program(score.program, task.test_inputs[0]).output
            for score in result.exact_scores
        )
        self.assertIn(expected, predictions)
        self.assertTrue(
            any(
                isinstance(score.program, RoleStampProgram)
                for score in result.exact_scores
            )
        )

    def test_role_enumeration_keeps_persistent_anchors_for_copy_canvas(self) -> None:
        task = _persistent_role_task()
        programs = enumerate_object_code_programs(task)
        self.assertTrue(any(isinstance(item, RoleStampProgram) for item in programs))
        result = synthesize_object_code_programs(task)
        self.assertTrue(
            any(
                isinstance(score.program, RoleStampProgram)
                and score.program.canvas_mode == "copy"
                for score in result.exact_scores
            )
        )

    def test_d4_completion_transfers_only_missing_annotation(self) -> None:
        task = _d4_task()
        program = D4LabelCompletionProgram(0, 4, 4, 1)
        result = execute_object_code_program(program, task.train[0].input)
        self.assertTrue(result.ok)
        self.assertEqual(result.output, task.train[0].output)
        self.assertEqual(result.correspondence_count, 1)
        synthesis = synthesize_object_code_programs(task)
        self.assertTrue(
            any(
                isinstance(score.program, D4LabelCompletionProgram)
                for score in synthesis.exact_scores
            )
        )

    def test_program_and_candidate_identity_are_replayable(self) -> None:
        task, _ = _role_task()
        program = RoleStampProgram(0, 8, 3, 2, "identity", "blank")
        reconstructed = object_code_program_from_json(program.to_json_dict())
        self.assertEqual(program, reconstructed)
        first = make_object_code_hypothesis(program, demo_exact=True)
        second = make_object_code_hypothesis(reconstructed, demo_exact=True)
        self.assertEqual(first.hypothesis_id, second.hypothesis_id)
        evaluation = evaluate_hypothesis(first, task)
        self.assertTrue(evaluation.eligible)
        provisional = make_object_code_hypothesis(
            program,
            demo_exact=False,
            ast_holes=("target_anchor_color",),
        )
        provisional_evaluation = evaluate_hypothesis(provisional, task)
        self.assertTrue(provisional_evaluation.demo_exact)
        self.assertFalse(provisional_evaluation.eligible)
        self.assertEqual(
            provisional_evaluation.rejection_reason,
            "insufficient_query_support",
        )

    def test_d4_program_rejects_boolean_integer_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "connectivity"):
            D4LabelCompletionProgram(0, 4, True, 1)
        with self.assertRaisesRegex(ValueError, "attachment radius"):
            D4LabelCompletionProgram(0, 4, 4, True)

    def test_empty_repair_frontier_does_not_become_cold_restart(self) -> None:
        task, _ = _role_task()
        result = synthesize_object_code_programs(task, programs=())
        self.assertEqual(result.program_trial_count, 0)
        self.assertFalse(result.exact_scores)
        self.assertFalse(result.near_miss_scores)

    def test_partial_execution_failure_remains_a_natural_near_miss(self) -> None:
        task, _ = _role_task()
        second_input = [list(row) for row in task.train[0].input]
        second_input[4][4] = 0
        partial_task = BlindTask.from_observations(
            train=(
                task.train[0],
                ARCPair(as_grid(second_input), task.train[0].output),
            ),
            test_inputs=task.test_inputs,
        )
        program = RoleStampProgram(0, 8, 3, 2, "identity", "blank")
        result = synthesize_object_code_programs(
            partial_task,
            programs=(program,),
            max_near_misses=1,
            minimum_near_miss_agreement=0.2,
        )
        self.assertEqual(len(result.near_miss_scores), 1)
        self.assertEqual(result.near_miss_scores[0].exact_demo_count, 1)
        self.assertFalse(result.near_miss_scores[0].execution_valid)

    def test_failure_certificates_select_distinct_typed_actions(self) -> None:
        task, _ = _role_task()
        cases = (
            (
                make_object_code_hypothesis(
                    RoleStampProgram(0, 8, 3, 2, "identity", "crop"),
                    demo_exact=False,
                ),
                "canvas_reinfer",
            ),
            (
                make_object_code_hypothesis(
                    RoleStampProgram(0, 8, 3, 1, "identity", "blank"),
                    demo_exact=False,
                    ast_holes=("target_anchor_color",),
                ),
                "fill_ast_hole",
            ),
            (
                make_object_code_hypothesis(
                    RoleStampProgram(9, 8, 3, 2, "identity", "blank"),
                    demo_exact=False,
                ),
                "reparse_background",
            ),
            (
                make_object_code_hypothesis(
                    RoleStampProgram(0, 8, 3, 2, "rotate90", "blank"),
                    demo_exact=False,
                ),
                "object_rematch",
            ),
        )
        certificate_ids = set()
        for candidate, expected_action in cases:
            evaluation = evaluate_hypothesis(candidate, task)
            self.assertFalse(evaluation.demo_exact)
            certificate = diagnose_object_code_failure(task, evaluation)
            self.assertEqual(certificate.recommended_action, expected_action)
            self.assertEqual(
                certificate,
                diagnose_object_code_failure(task, evaluation),
            )
            certificate_ids.add(certificate.certificate_id)
        self.assertEqual(len(certificate_ids), len(cases))

        d4_task = _d4_task()
        d4_parent = make_object_code_hypothesis(
            D4LabelCompletionProgram(0, 1, 4, 1),
            demo_exact=False,
        )
        d4_evaluation = evaluate_hypothesis(d4_parent, d4_task)
        d4_certificate, d4_frontier = typed_repair_frontier(d4_task, d4_evaluation)
        self.assertEqual(d4_certificate.recommended_action, "object_rematch")
        self.assertEqual(
            d4_certificate.affected_slots,
            ("attachment_radius", "connectivity", "structure_color"),
        )
        self.assertTrue(d4_frontier)
        parent_fields = d4_parent.spec["object_code_program"]
        for program in d4_frontier:
            candidate_fields = program.to_json_dict()
            changed = {
                key
                for key in parent_fields
                if key not in {"object_code_dsl_version", "kind"}
                if parent_fields[key] != candidate_fields[key]
            }
            self.assertTrue(changed)
            self.assertTrue(changed <= set(d4_certificate.affected_slots))

    def test_ast_hole_repair_frontier_contains_exact_program(self) -> None:
        task, _ = _role_task()
        parent = make_object_code_hypothesis(
            RoleStampProgram(0, 8, 3, 1, "identity", "blank"),
            demo_exact=False,
            ast_holes=("target_anchor_color",),
        )
        evaluation = evaluate_hypothesis(parent, task)
        certificate, frontier = typed_repair_frontier(task, evaluation)
        self.assertEqual(certificate.recommended_action, "fill_ast_hole")
        self.assertTrue(frontier)
        scores = synthesize_object_code_programs(task, programs=frontier)
        self.assertTrue(scores.exact_scores)
        self.assertLess(len(frontier), len(enumerate_object_code_programs(task)))

    def test_typed_compiler_changes_online_action_and_recovers(self) -> None:
        task, _ = _role_task()
        wrong = make_object_code_hypothesis(
            RoleStampProgram(0, 8, 3, 2, "rotate90", "blank"),
            demo_exact=False,
        )
        seed = FrozenCandidatePoolProvider((wrong,), "seed-near-miss", "code_llm")
        object_provider = ObjectCodeProvider(max_exact_programs=8, max_near_misses=4)
        report = OnlineFunctionalRouterSolver(
            providers=(seed, object_provider),
            config=OnlineControlConfig(
                budget_limit=_budget(2),
                provider_batch_size=4,
                max_selected_hypotheses=1,
            ),
            policy=FixedSchedulePolicy(("seed-near-miss", "object_code_dsl")),
            compiler=ObjectCodeResidualCompiler(),
        ).solve(task)
        proposals = tuple(
            result
            for result in report.states[-1].action_results
            if result.action.kind == "propose"
        )
        self.assertEqual(proposals[0].action.operator, "open_hypothesis")
        self.assertEqual(proposals[1].action.operator, "object_rematch")
        self.assertEqual(proposals[1].action.parent_hypothesis_id, wrong.hypothesis_id)
        self.assertTrue(
            any(
                code.startswith("failure_certificate:")
                for code in proposals[1].action.reason_codes
            )
        )
        self.assertIsNotNone(proposals[1].provider_result)
        diagnostics = proposals[1].provider_result.diagnostics
        self.assertGreater(diagnostics["novel_frontier_count"], 0)
        self.assertTrue(diagnostics["frontier_changed"])
        self.assertEqual(
            diagnostics["frontier_change_basis"],
            "content_addressed_object_code_program_id",
        )
        self.assertEqual(report.status, "solved")

    def test_generation_is_invariant_to_hidden_query_output(self) -> None:
        blind, _ = _role_task()
        train = blind.train
        query = blind.test_inputs[0]

        def source(output: list[list[int]]) -> ARCTask:
            return ARCTask(
                task_id="hidden-output-probe",
                train=train,
                test=(ARCPair(query, as_grid(output)),),
                source_path="fixture",
                source_sha256="0" * 64,
            )

        first = BlindTask.from_task(source([[1]]))
        second = BlindTask.from_task(source([[9]]))
        self.assertEqual(first, second)
        first_result = ObjectCodeProvider().propose(first, None, None)
        second_result = ObjectCodeProvider().propose(second, None, None)
        self.assertEqual(
            tuple(item.hypothesis_id for item in first_result.candidates),
            tuple(item.hypothesis_id for item in second_result.candidates),
        )


if __name__ == "__main__":
    unittest.main()
