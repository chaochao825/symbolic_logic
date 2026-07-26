from __future__ import annotations

import unittest
from dataclasses import dataclass

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.hybrid import (
    BudgetVector,
    CandidateHypothesis,
    ControlAction,
    DeliberationSketch,
    FrozenCandidatePoolProvider,
    OnlineControlConfig,
    OnlineFunctionalRouterSolver,
    ProviderResult,
    StructuredDeliberationPolicy,
)
from afts_arc.hybrid.scene_dsl import execute_scene_rule, synthesize_scene_rules
from afts_arc.task import ARCPair


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


def _hypothesis(name: str, replay, *, route: str) -> CandidateHypothesis:
    return CandidateHypothesis.create(
        name=name,
        source="structured-test",
        source_version="structured-test/v1",
        route=route,
        description_bits=8,
        verification_mode="replayable",
        functional_trace=(name,),
        spec={"name": name},
        replay=replay,
        hard_verifier=lambda task: True,
    )


def _budget(steps: int) -> BudgetVector:
    return BudgetVector(
        compute_units=2 * steps,
        controller_steps=steps,
        provider_calls=steps,
        repair_attempts=0,
        candidate_slots=steps,
    )


@dataclass
class RootOnlySceneProvider:
    candidate: CandidateHypothesis
    name: str = "scene-root"
    route: str = "dsl_program"
    strict_budget_contract: bool = True
    max_control_calls: int = 1
    supports_residual_actions: bool = False
    parent_sensitive_operators: frozenset[str] = frozenset()

    def act(self, task, features, decision, blackboard, action):
        del task, features, decision, blackboard
        if action.operator != "synthesize" or action.parent_hypothesis_id is not None:
            raise AssertionError("root-only provider received a residual action")
        return ProviderResult.ok(self.name, self.route, (self.candidate,))


class StructuredDeliberationTests(unittest.TestCase):
    def test_scene_predicate_synthesis_finds_smallest_object_crop(self) -> None:
        task = _blind(
            (
                (
                    [[0, 2, 2, 0], [0, 0, 0, 3], [0, 0, 3, 3]],
                    [[2, 2]],
                ),
                (
                    [[4, 4, 0, 0], [0, 0, 0, 5], [0, 0, 5, 5]],
                    [[4, 4]],
                ),
            ),
            ([[0, 6, 6, 0], [7, 7, 7, 0], [7, 7, 7, 0]],),
        )
        result = synthesize_scene_rules(task)
        matching = tuple(
            rule
            for rule in result.rules
            if rule.selector == "smallest_area" and rule.action == "crop"
        )
        self.assertTrue(matching)
        replays = tuple(
            execute_scene_rule(rule, task.test_inputs[0]) for rule in matching
        )
        self.assertTrue(any(replay.ok for replay in replays))
        self.assertIn(as_grid([[6, 6]]), tuple(replay.output for replay in replays))
        self.assertGreater(result.rule_trial_count, 0)

    def test_root_only_representation_remains_legal_after_residual(self) -> None:
        task = _blind((([[1, 1]], [[2, 2]]),), ([[1, 1, 1]],))
        wrong = _hypothesis("wrong", lambda grid: grid, route="dsl_program")
        correct = _hypothesis(
            "correct",
            lambda grid: tuple(tuple(2 for _ in row) for row in grid),
            route="dsl_program",
        )
        report = OnlineFunctionalRouterSolver(
            providers=(
                FrozenCandidatePoolProvider((wrong,), "a-root", "dsl_program"),
                RootOnlySceneProvider(correct, name="z-scene-root"),
            ),
            config=OnlineControlConfig(
                budget_limit=_budget(2),
                provider_batch_size=1,
                max_selected_hypotheses=1,
            ),
            policy=StructuredDeliberationPolicy(
                name="diversity-test",
                use_grounding=False,
                use_adaptive_diversity=True,
                use_borderline_ucb=False,
            ),
        ).solve(task)
        proposals = tuple(
            result
            for result in report.states[-1].action_results
            if result.action.kind == "propose"
        )
        self.assertEqual(
            [item.action.actor for item in proposals],
            ["a-root", "z-scene-root"],
        )
        self.assertEqual(proposals[1].action.operator, "synthesize")
        self.assertIsNone(proposals[1].action.parent_hypothesis_id)
        self.assertEqual(report.status, "solved")

    def test_deliberation_sketch_is_content_addressed_and_oracle_free(self) -> None:
        task = _blind((([[1]], [[2]]),), ([[1, 1]],))
        wrong = _hypothesis("wrong", lambda grid: grid, route="dsl_program")
        report = OnlineFunctionalRouterSolver(
            providers=(FrozenCandidatePoolProvider((wrong,), "root", "dsl_program"),),
            config=OnlineControlConfig(
                budget_limit=_budget(1),
                provider_batch_size=1,
                max_selected_hypotheses=1,
            ),
        ).solve(task)
        first = DeliberationSketch.from_blackboard(report.states[-1])
        second = DeliberationSketch.from_blackboard(report.states[-1])
        self.assertEqual(first, second)
        self.assertEqual(first.sketch_id, second.sketch_id)
        serialized = first.to_json_dict()
        self.assertNotIn("test_output", canonical_text := str(serialized).lower())
        self.assertNotIn("oracle", canonical_text)
        self.assertIn(first.phase, {"explore", "refine"})

    def test_phase_control_has_a_true_same_model_ablation(self) -> None:
        task = _blind((([[1]], [[2]]),), ([[1]],))
        wrong = _hypothesis("wrong", lambda grid: grid, route="dsl_program")
        report = OnlineFunctionalRouterSolver(
            providers=(FrozenCandidatePoolProvider((wrong,), "root", "dsl_program"),),
            config=OnlineControlConfig(
                budget_limit=_budget(1),
                provider_batch_size=1,
                max_selected_hypotheses=1,
            ),
        ).solve(task)
        blackboard = report.states[1]
        signal = blackboard.residual_signals[0]
        repair = ControlAction.create(
            state_id=blackboard.state_id,
            kind="repair",
            actor="residual_repair",
            route="residual_repair",
            operator="global_color_map",
            parent_hypothesis_id=wrong.hypothesis_id,
            evidence_signal_ids=(signal.signal_id,),
            budget=BudgetVector(
                compute_units=2,
                controller_steps=1,
                repair_attempts=1,
                candidate_slots=1,
            ),
            priority=100,
        )
        proposal = ControlAction.create(
            state_id=blackboard.state_id,
            kind="propose",
            actor="root",
            route="dsl_program",
            operator="suffix_resynthesize",
            parent_hypothesis_id=wrong.hypothesis_id,
            evidence_signal_ids=(signal.signal_id,),
            budget=BudgetVector(
                compute_units=2,
                controller_steps=1,
                provider_calls=1,
                candidate_slots=1,
            ),
            priority=0,
        )
        dynamic = StructuredDeliberationPolicy(
            use_grounding=False,
            use_phase_control=True,
            use_adaptive_diversity=False,
            use_borderline_ucb=False,
        )
        static = StructuredDeliberationPolicy(
            use_grounding=False,
            use_phase_control=False,
            use_adaptive_diversity=False,
            use_borderline_ucb=False,
        )

        self.assertEqual(dynamic.select(blackboard, (repair, proposal)), repair)
        self.assertEqual(static.select(blackboard, (repair, proposal)), proposal)


if __name__ == "__main__":
    unittest.main()
