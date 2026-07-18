from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import dataclass

from afts_arc.blind import BlindTask
from afts_arc.grid import Grid, as_grid
from afts_arc.hybrid import (
    CandidateHypothesis,
    CodeModelProvider,
    DiffLogicHardProvider,
    FunctionalRouterConfig,
    FunctionalRouterSolver,
    ProviderResult,
    SparseCAProvider,
    evaluate_hypotheses,
    evaluate_hypothesis,
    global_color_map_repair,
    route_task,
)
from afts_arc.task import ARCPair, ARCTask


def _blind(
    train: tuple[tuple[list[list[int]], list[list[int]]], ...],
    queries: tuple[list[list[int]], ...],
) -> BlindTask:
    return BlindTask.from_observations(
        train=tuple(ARCPair(as_grid(source), as_grid(target)) for source, target in train),
        test_inputs=tuple(as_grid(grid) for grid in queries),
    )


def _hypothesis(
    name: str,
    replay,
    bits: int,
    *,
    route: str = "dsl_program",
    verifier=True,
    mode: str = "replayable",
) -> CandidateHypothesis:
    hard_verifier = None if verifier is None else (lambda task, result=verifier: result)
    return CandidateHypothesis.create(
        name=name,
        source="test_provider",
        source_version="test/v1",
        route=route,
        description_bits=bits,
        verification_mode=mode,
        functional_trace=(name,),
        spec={"name": name},
        replay=replay,
        hard_verifier=hard_verifier,
    )


@dataclass
class StaticProvider:
    candidates: tuple[CandidateHypothesis, ...]
    name: str = "static"
    route: str = "dsl_program"

    def propose(self, task, features, decision):
        if not isinstance(task, BlindTask):
            raise TypeError("oracle-bearing task reached provider")
        return ProviderResult.ok(self.name, self.route, self.candidates)


@dataclass
class InvalidCandidateProvider:
    name: str = "invalid"
    route: str = "dsl_program"

    def propose(self, task, features, decision):
        return ProviderResult.ok(self.name, self.route, (object(),))


@dataclass
class SpoofedResultProvider:
    candidate: CandidateHypothesis
    name: str = "spoofed"
    route: str = "dsl_program"

    def propose(self, task, features, decision):
        return ProviderResult.ok("another-provider", self.route, (self.candidate,))


def _content_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _rehash_difflogic_spec(spec: dict[str, object]) -> dict[str, object]:
    spec["hard_circuit_sha256"] = _content_sha256(spec["hard_circuit"])
    artifact_payload = {
        key: value for key, value in spec.items() if key != "artifact_sha256"
    }
    spec["artifact_sha256"] = _content_sha256(artifact_payload)
    return spec


def _difflogic_spec(task: BlindTask, *, horizon: object = 1) -> dict[str, object]:
    circuit = {
        "input_count": 1,
        "output_count": 1,
        "gates": [],
        "outputs": [0],
    }
    config = {
        "model_kind": "difflogic_hard_circuit",
        "state_encoding": "categorical10",
        "feature_count": 1,
        "max_steps": 2,
    }
    spec = {
        "source_commit": "1" * 40,
        "feature_schema_version": "fixture/features-v1",
        "config": config,
        "augmentation_policy": "none",
        "horizon": horizon,
        "hard_circuit": circuit,
        "hard_circuit_sha256": _content_sha256(circuit),
        "blind_content_sha256": task.blind_content_sha256,
    }
    return _rehash_difflogic_spec(spec)


class FunctionalRouterTests(unittest.TestCase):
    def test_router_prefers_dsl_for_shape_change_and_ca_for_local_same_shape(self) -> None:
        shape_task = _blind(
            (([[1, 2], [3, 4]], [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),),
            ([[5, 6], [7, 8]],),
        )
        features, decision = route_task(shape_task)
        self.assertTrue(features.shape_change)
        self.assertEqual(decision.route_order[0], "dsl_program")

        local_task = _blind(
            (
                ([[0, 1], [0, 0]], [[0, 2], [0, 0]]),
                ([[1, 0, 0]], [[2, 0, 0]]),
            ),
            ([[0, 1]],),
        )
        features, decision = route_task(local_task)
        self.assertTrue(features.compressible_local_transition)
        self.assertEqual(decision.route_order[0], "sparse_ca")
        self.assertEqual(decision.route_order[1], "difflogic_hard")

    def test_demo_exact_gate_and_mdl_break_ties_independent_of_provider_order(self) -> None:
        task = _blind((([[1, 0]], [[2, 0]]),), ([[1]], [[0, 1]]))
        wrong = _hypothesis("wrong-short", lambda grid: grid, 1)
        long_exact = _hypothesis(
            "long-exact",
            lambda grid: tuple(tuple(2 if cell == 1 else cell for cell in row) for row in grid),
            37,
        )
        short_exact = _hypothesis(
            "short-exact",
            lambda grid: tuple(tuple(2 if cell == 1 else cell for cell in row) for row in grid),
            11,
        )
        first = FunctionalRouterSolver(
            providers=(StaticProvider((long_exact, short_exact, wrong)),),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(task)
        second = FunctionalRouterSolver(
            providers=(StaticProvider((wrong, short_exact, long_exact)),),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(task)
        wrong_eval = next(item for item in first.initial_evaluations if item.hypothesis.name == "wrong-short")
        self.assertEqual(wrong_eval.rejection_reason, "demo_mismatch")
        self.assertEqual(first.selected[0].hypothesis.name, "short-exact")
        self.assertEqual(
            [item.hypothesis.hypothesis_id for item in first.selected],
            [item.hypothesis.hypothesis_id for item in second.selected],
        )
        records = first.candidate_records(task_id="multi_query")
        self.assertEqual([item.test_index for item in records], [0, 1])
        self.assertEqual(
            {item.generation_parameters_json for item in records},
            {records[0].generation_parameters_json},
        )

    def test_hard_route_rejects_missing_or_failed_certificate(self) -> None:
        task = _blind((([[1]], [[2]]),), ([[1]],))
        transform = lambda grid: tuple(tuple(2 if cell == 1 else cell for cell in row) for row in grid)
        missing = _hypothesis(
            "missing-circuit",
            transform,
            2,
            route="difflogic_hard",
            verifier=None,
            mode="hard_circuit",
        )
        failed = _hypothesis(
            "failed-circuit",
            transform,
            3,
            route="difflogic_hard",
            verifier=False,
            mode="hard_circuit",
        )
        report = FunctionalRouterSolver(
            providers=(StaticProvider((missing, failed), route="difflogic_hard"),),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(task)
        reasons = {item.hypothesis.name: item.rejection_reason for item in report.initial_evaluations}
        self.assertEqual(reasons["missing-circuit"], "missing_hard_verifier")
        self.assertEqual(reasons["failed-circuit"], "hard_verification_failed")
        self.assertFalse(report.selected)

    def test_optional_sources_abstain_without_loading_training_modules(self) -> None:
        sys.modules.pop("arc_difflogic_train", None)
        sys.modules.pop("afts_arc.m04a_train", None)
        task = _blind((([[1]], [[1]]),), ([[2]],))
        report = FunctionalRouterSolver().solve(task)
        by_provider = {item.provider: item for item in report.provider_results}
        self.assertEqual(by_provider["masked_diffusion"].status, "abstained")
        self.assertEqual(by_provider["code_llm"].status, "abstained")
        self.assertEqual(by_provider["difflogic_hard"].reason, "missing_hard_circuit")
        self.assertNotIn("arc_difflogic_train", sys.modules)
        self.assertNotIn("afts_arc.m04a_train", sys.modules)

    def test_color_map_repair_closes_residual_and_records_parent(self) -> None:
        task = _blind(
            (
                ([[0, 1], [1, 0]], [[0, 2], [2, 0]]),
                ([[1, 0, 0]], [[2, 0, 0]]),
            ),
            ([[0, 1]],),
        )
        identity = _hypothesis("identity", lambda grid: grid, 4)
        report = FunctionalRouterSolver(providers=(StaticProvider((identity,)),)).solve(task)
        receipt = next(item for item in report.repair_receipts if item.strategy == "global_color_map")
        self.assertEqual(receipt.status, "produced")
        repaired = next(
            item
            for item in report.repaired_evaluations
            if "residual_repair:global_color_map" in item.hypothesis.functional_trace
        )
        self.assertTrue(repaired.eligible)
        self.assertEqual(repaired.query_outputs, (((0, 2),),))
        self.assertEqual(repaired.hypothesis.parent_hypothesis_ids, (identity.hypothesis_id,))
        self.assertEqual(report.selected[0].hypothesis.hypothesis_id, repaired.hypothesis.hypothesis_id)
        record = report.candidate_records(task_id="repair_probe")[0]
        self.assertTrue(record.parent_candidate_ids)
        parent_records = json.loads(record.generation_parameters_json)["parent_candidate_records"]
        self.assertEqual(
            record.parent_candidate_ids,
            tuple(item["candidate_id"] for item in parent_records),
        )
        lineage = report.lineage_candidate_records(task_id="repair_probe")
        lineage_ids = {item.candidate_id for item in lineage}
        self.assertTrue(
            all(set(item.parent_candidate_ids).issubset(lineage_ids) for item in lineage)
        )
        self.assertEqual(lineage[-1].candidate_id, record.candidate_id)

    def test_color_map_repair_rejects_ambiguous_mapping(self) -> None:
        task = _blind((([[1, 1]], [[2, 1]]),), ([[1, 1]],))
        identity = _hypothesis("identity-ambiguous", lambda grid: grid, 4)
        evaluation = evaluate_hypothesis(identity, task)
        candidates, receipt = global_color_map_repair(evaluation, task)
        self.assertFalse(candidates)
        self.assertEqual(receipt.reason, "ambiguous_color_mapping")

    def test_solver_is_invariant_to_public_test_oracle(self) -> None:
        def task(output: list[list[int]]) -> ARCTask:
            return ARCTask(
                task_id="oracle_probe",
                train=(ARCPair(as_grid([[1]]), as_grid([[1]])),),
                test=(ARCPair(as_grid([[2]]), as_grid(output)),),
                source_path="fixture",
                source_sha256="0" * 64,
            )

        first = BlindTask.from_task(task([[3]]))
        second = BlindTask.from_task(task([[9]]))
        self.assertEqual(first, second)
        identity = _hypothesis("oracle-free-identity", lambda grid: grid, 4)
        first_report = FunctionalRouterSolver(
            providers=(StaticProvider((identity,)),),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(first)
        second_report = FunctionalRouterSolver(
            providers=(StaticProvider((identity,)),),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(second)
        self.assertEqual(first_report.to_json_dict(), second_report.to_json_dict())
        self.assertNotIn("output", first.to_json_dict()["test"][0])

    def test_sparse_ca_center_rule_and_compiled_codecs_are_hard_verified(self) -> None:
        sys.modules.pop("arc_difflogic_train", None)
        task = _blind(
            (
                ([[1, 0], [0, 1]], [[3, 0], [0, 3]]),
                ([[0, 1, 1]], [[0, 3, 3]]),
            ),
            ([[1, 0, 1]],),
        )
        report = FunctionalRouterSolver(
            providers=(SparseCAProvider(policies=("none",), max_rules_per_policy=1, max_programs=0),),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(task)
        center = next(
            item
            for item in report.initial_evaluations
            if item.hypothesis.spec.get("neighborhood") == "center"
        )
        self.assertTrue(center.eligible)
        self.assertTrue(center.hard_verified)
        self.assertEqual(center.query_outputs, (((3, 0, 3),),))
        self.assertEqual(center.hypothesis.metadata["query_support"], 1.0)
        self.assertNotIn("arc_difflogic_train", sys.modules)

    def test_sparse_ca_rejects_zero_and_salient_foreground_support(self) -> None:
        unseen = _blind((([[1]], [[2]]),), ([[9]],))
        report = FunctionalRouterSolver(
            providers=(SparseCAProvider(policies=("none",), max_rules_per_policy=1, max_programs=0),),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(unseen)
        center = next(
            item
            for item in report.initial_evaluations
            if item.hypothesis.spec.get("neighborhood") == "center"
        )
        self.assertEqual(center.hypothesis.metadata["query_support"], 0.0)
        self.assertEqual(
            center.hypothesis.spec["support_policy"]["minimum_query_support"],
            0.5,
        )
        self.assertEqual(center.rejection_reason, "insufficient_query_support")
        self.assertFalse(report.selected)

        source = [[0 for _ in range(5)] for _ in range(5)]
        target = [list(row) for row in source]
        source[2][2] = 1
        target[2][2] = 2
        query = [[0 for _ in range(30)] for _ in range(30)]
        query[15][15] = 3
        salient = _blind(((source, target),), (query,))
        report = FunctionalRouterSolver(
            providers=(SparseCAProvider(policies=("none",), max_rules_per_policy=1, max_programs=0),),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(salient)
        center = next(
            item
            for item in report.initial_evaluations
            if item.hypothesis.spec.get("neighborhood") == "center"
        )
        self.assertGreater(center.hypothesis.metadata["query_support"], 0.99)
        self.assertEqual(center.hypothesis.metadata["foreground_query_support"], 0.0)
        self.assertEqual(center.rejection_reason, "insufficient_query_support")

    def test_grid_only_bundle_outputs_are_content_addressed_and_cannot_fake_hard_rule(self) -> None:
        task = _blind((([[1]], [[1]]),), ([[2]],))

        def proposal(name: str, query: list[list[int]]) -> CandidateHypothesis:
            return CandidateHypothesis.create(
                name=name,
                source="masked_diffusion",
                source_version="fixture/v1",
                route="masked_diffusion",
                description_bits=8,
                verification_mode="grid_only",
                functional_trace=("masked_sample",),
                spec={"seed": 7},
                explicit_demo_outputs=([[1]],),
                explicit_query_outputs=(query,),
            )

        first = proposal("sample", [[2]])
        second = proposal("sample", [[3]])
        self.assertNotEqual(first.hypothesis_id, second.hypothesis_id)
        evaluated = evaluate_hypotheses((first, second), task)
        self.assertEqual(len(evaluated), 2)
        self.assertTrue(all(item.demo_exact for item in evaluated))
        self.assertTrue(all(item.rejection_reason == "missing_hard_verifier" for item in evaluated))

    def test_same_replay_artifact_id_cannot_silently_change_semantics(self) -> None:
        task = _blind((([[1]], [[1]]),), ([[2]],))
        first = _hypothesis("same-artifact", lambda grid: grid, 4)
        second = _hypothesis("same-artifact", lambda grid: ((9,),), 4)
        self.assertEqual(first.hypothesis_id, second.hypothesis_id)
        with self.assertRaisesRegex(ValueError, "inconsistent bundle semantics"):
            evaluate_hypotheses((first, second), task)
        with self.assertRaisesRegex(ValueError, "inconsistent bundle semantics"):
            evaluate_hypotheses((second, first), task)

    def test_invalid_provider_payload_is_error_isolated(self) -> None:
        task = _blind((([[1]], [[1]]),), ([[2]],))
        identity = _hypothesis("good-after-bad", lambda grid: grid, 4)
        report = FunctionalRouterSolver(
            providers=(InvalidCandidateProvider(), StaticProvider((identity,))),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(task)
        by_provider = {item.provider: item for item in report.provider_results}
        self.assertEqual(by_provider["invalid"].status, "error")
        self.assertEqual(report.status, "solved")

    def test_provider_contract_and_semantic_collision_are_isolated(self) -> None:
        task = _blind((([[1]], [[1]]),), ([[2]],))
        first = _hypothesis("colliding-artifact", lambda grid: grid, 4)
        second = _hypothesis("colliding-artifact", lambda grid: ((9,),), 4)
        payload_variant = CandidateHypothesis.create(
            name=first.name,
            source=first.source,
            source_version=first.source_version,
            route=first.route,
            description_bits=first.description_bits,
            verification_mode=first.verification_mode,
            functional_trace=first.functional_trace,
            spec=first.spec,
            metadata={"conflicting_metadata": True},
            replay=lambda grid: ((8,),),
            hard_verifier=lambda blind: True,
        )
        self.assertEqual(first.hypothesis_id, payload_variant.hypothesis_id)
        good = _hypothesis("independent-good", lambda grid: grid, 7)
        report = FunctionalRouterSolver(
            providers=(
                StaticProvider((first,), name="collision-a"),
                StaticProvider((good,), name="good"),
                StaticProvider((second,), name="collision-b"),
                StaticProvider((payload_variant,), name="collision-c"),
                SpoofedResultProvider(good),
            ),
            config=FunctionalRouterConfig(max_repair_parents=0),
        ).solve(task)
        by_provider = {item.provider: item for item in report.provider_results}
        self.assertEqual(by_provider["collision-a"].reason, "semantic_artifact_conflict")
        self.assertEqual(by_provider["collision-b"].reason, "semantic_artifact_conflict")
        self.assertEqual(by_provider["collision-c"].reason, "semantic_artifact_conflict")
        self.assertEqual(
            by_provider["collision-a"].diagnostics["collision_reasons"][first.hypothesis_id],
            "canonical_payload_conflict",
        )
        self.assertEqual(by_provider["spoofed"].status, "error")
        self.assertEqual(report.selected[0].hypothesis.name, "independent-good")
        self.assertEqual(
            {item.hypothesis.name for item in report.initial_evaluations},
            {"independent-good"},
        )
        with self.assertRaisesRegex(ValueError, "provider names must be unique"):
            FunctionalRouterSolver(
                providers=(
                    StaticProvider((good,), name="duplicate"),
                    StaticProvider((good,), name="duplicate"),
                )
            )

    def test_external_callback_is_canonicalized_filtered_and_budget_stable(self) -> None:
        task = _blind((([[1]], [[1]]),), ([[2]],))

        def proposals(reverse: bool):
            candidates = [
                CandidateHypothesis.create(
                    name=f"code-{index}",
                    source="typed_dsl",
                    source_version="untrusted/v1",
                    route="code_llm",
                    description_bits=1000 + index,
                    verification_mode="replayable",
                    functional_trace=("generated-code",),
                    spec={"artifact_sha256": f"{index + 1:064x}", "index": index},
                    metadata={"query_support": 1.0, "note": "kept"},
                    replay=lambda grid: grid,
                    hard_verifier=lambda blind: True,
                )
                for index in range(10)
            ]
            return tuple(reversed(candidates)) if reverse else tuple(candidates)

        features, decision = route_task(task)
        forward = CodeModelProvider(lambda *args: proposals(False)).propose(
            task, features, decision
        )
        reverse = CodeModelProvider(lambda *args: proposals(True)).propose(
            task, features, decision
        )
        self.assertEqual(
            [item.hypothesis_id for item in forward.candidates],
            [item.hypothesis_id for item in reverse.candidates],
        )
        self.assertEqual(len(forward.candidates), decision.budget_for("code_llm"))
        self.assertTrue(all(item.source == "code_llm" for item in forward.candidates))
        self.assertTrue(
            all(item.source_version == "afts-hybrid-external/v1" for item in forward.candidates)
        )
        self.assertTrue(
            all("query_support" not in item.metadata for item in forward.candidates)
        )
        self.assertTrue(
            all(item.metadata["external_declared_source"] == "typed_dsl" for item in forward.candidates)
        )
        mixed = CodeModelProvider(
            lambda *args: (object(), proposals(False)[0])
        ).propose(task, features, decision)
        self.assertEqual(mixed.status, "ok")
        self.assertEqual(mixed.diagnostics["invalid_candidate_count"], 1)
        self.assertEqual(len(mixed.candidates), 1)

    def test_difflogic_callback_requires_bound_self_hashed_circuit_bundle(self) -> None:
        sys.modules.pop("arc_difflogic_train", None)
        task = _blind((([[1]], [[1]]),), ([[2]],))

        def candidate(spec: dict[str, object]) -> CandidateHypothesis:
            return CandidateHypothesis.create(
                name="hard-identity",
                source="untrusted",
                source_version="fixture/v1",
                route="difflogic_hard",
                description_bits=1,
                verification_mode="hard_circuit",
                functional_trace=("hard-circuit",),
                spec=spec,
                replay=lambda grid: grid,
                hard_verifier=lambda blind: True,
            )

        features, decision = route_task(task)
        valid = DiffLogicHardProvider(lambda *args: (candidate(_difflogic_spec(task)),)).propose(
            task, features, decision
        )
        self.assertEqual(valid.status, "ok")
        self.assertGreater(valid.candidates[0].description_bits, 1)
        self.assertGreater(valid.candidates[0].metadata["hard_circuit_payload_bits"], 0)

        invalid_horizon = DiffLogicHardProvider(
            lambda *args: (candidate(_difflogic_spec(task, horizon=0)),)
        ).propose(task, features, decision)
        self.assertEqual(invalid_horizon.status, "error")
        self.assertEqual(invalid_horizon.reason, "all_candidates_invalid")

        wrong_digest = _difflogic_spec(task)
        wrong_digest["hard_circuit_sha256"] = "0" * 64
        invalid_digest = DiffLogicHardProvider(
            lambda *args: (candidate(wrong_digest),)
        ).propose(task, features, decision)
        self.assertEqual(invalid_digest.status, "error")

        invalid_operation = _difflogic_spec(task)
        invalid_operation["hard_circuit"] = {
            "input_count": 1,
            "output_count": 1,
            "gates": [{"op": "BOGUS", "inputs": [0, 0]}],
            "outputs": [1],
        }
        _rehash_difflogic_spec(invalid_operation)
        invalid_gate = DiffLogicHardProvider(
            lambda *args: (candidate(invalid_operation),)
        ).propose(task, features, decision)
        self.assertEqual(invalid_gate.status, "error")

        wrong_width = _difflogic_spec(task)
        wrong_width["hard_circuit"] = {
            "input_count": 2,
            "output_count": 1,
            "gates": [],
            "outputs": [0],
        }
        _rehash_difflogic_spec(wrong_width)
        invalid_width = DiffLogicHardProvider(
            lambda *args: (candidate(wrong_width),)
        ).propose(task, features, decision)
        self.assertEqual(invalid_width.status, "error")

        unhashed_extra = _difflogic_spec(task)
        unhashed_extra["deployment_extension"] = {"uncommitted": True}
        invalid_extra = DiffLogicHardProvider(
            lambda *args: (candidate(unhashed_extra),)
        ).propose(task, features, decision)
        self.assertEqual(invalid_extra.status, "error")
        self.assertNotIn("arc_difflogic_train", sys.modules)

    def test_d4_bgpad_is_a_distinct_demo_audited_deployment_policy(self) -> None:
        source = [[0 for _ in range(5)] for _ in range(5)]
        source[1][1] = 1
        source[3][2] = 1
        source[3][3] = 1
        target = [list(row) for row in source]
        target[1][1] = 2
        task = _blind(((source, target),), ([[1, 0, 0], [0, 0, 0], [0, 0, 0]],))
        report = FunctionalRouterSolver(
            providers=(
                SparseCAProvider(
                    policies=("d4", "d4_bgpad"),
                    max_rules_per_policy=1,
                    max_programs=0,
                ),
            ),
            config=FunctionalRouterConfig(max_selected_hypotheses=2, max_repair_parents=0),
        ).solve(task)
        by_policy = {
            item.hypothesis.spec["policy"]: item
            for item in report.initial_evaluations
            if "policy" in item.hypothesis.spec
        }
        self.assertTrue(by_policy["d4"].demo_exact)
        self.assertTrue(by_policy["d4_bgpad"].demo_exact)
        self.assertEqual(by_policy["d4"].query_outputs[0][0][0], 1)
        self.assertEqual(by_policy["d4_bgpad"].query_outputs[0][0][0], 2)


if __name__ == "__main__":
    unittest.main()
