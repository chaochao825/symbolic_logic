from __future__ import annotations

import math
import unittest

from afts_arc.candidate import CandidateRecord, CandidateStore


class CandidateTests(unittest.TestCase):
    def test_same_output_keeps_distinct_provenance_but_one_output_group(self) -> None:
        first = CandidateRecord.create(
            task_id="abc",
            test_index=0,
            source_type="dsl",
            source_version="v1",
            output=[[1, 2]],
            functional_trace=("recolor",),
        )
        second = CandidateRecord.create(
            task_id="abc",
            test_index=0,
            source_type="diffusion",
            source_version="v2",
            output=[[1, 2]],
            model_confidence=0.7,
        )
        store = CandidateStore()
        self.assertTrue(store.add(first))
        self.assertTrue(store.add(second))
        self.assertFalse(store.add(first))
        self.assertNotEqual(first.candidate_id, second.candidate_id)
        self.assertEqual(first.output_key, second.output_key)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.unique_output_count, 1)
        self.assertEqual(
            {record.source_type for record in store.records_for_output(task_id="abc", test_index=0, output=[[1, 2]])},
            {"dsl", "diffusion"},
        )

    def test_candidate_id_is_deterministic(self) -> None:
        kwargs = dict(
            task_id="abc",
            test_index=1,
            source_type="rule",
            source_version="v1",
            output=[[4]],
            generation_parameters={"b": 2, "a": 1},
        )
        self.assertEqual(
            CandidateRecord.create(**kwargs).candidate_id,
            CandidateRecord.create(**kwargs).candidate_id,
        )

    def test_rejects_invalid_confidence_or_cost(self) -> None:
        with self.assertRaises(ValueError):
            CandidateRecord.create(
                task_id="abc",
                test_index=0,
                source_type="x",
                source_version="v1",
                output=[[0]],
                model_confidence=1.1,
            )
        with self.assertRaises(ValueError):
            CandidateRecord.create(
                task_id="abc",
                test_index=0,
                source_type="x",
                source_version="v1",
                output=[[0]],
                cost_cpu_ms=-1,
            )
        with self.assertRaises(ValueError):
            CandidateRecord.create(
                task_id="abc",
                test_index=0,
                source_type="x",
                source_version="v1",
                output=[[0]],
                cost_gpu_ms=math.nan,
            )

    def test_coerces_parent_and_trace_sequences_to_tuples(self) -> None:
        candidate = CandidateRecord.create(
            task_id="abc",
            test_index=0,
            source_type="x",
            source_version="v1",
            output=[[0]],
            parent_candidate_ids=["parent"],  # type: ignore[arg-type]
            functional_trace=["step"],  # type: ignore[arg-type]
        )
        self.assertEqual(candidate.parent_candidate_ids, ("parent",))
        self.assertEqual(candidate.functional_trace, ("step",))

    def test_rejects_unordered_or_string_trace_inputs(self) -> None:
        for trace in ({"a", "b"}, "step"):
            with self.subTest(trace=trace), self.assertRaises(TypeError):
                CandidateRecord.create(
                    task_id="abc",
                    test_index=0,
                    source_type="x",
                    source_version="v1",
                    output=[[0]],
                    functional_trace=trace,  # type: ignore[arg-type]
                )

    def test_rejects_nonstandard_json_numbers(self) -> None:
        with self.assertRaises(ValueError):
            CandidateRecord.create(
                task_id="abc",
                test_index=0,
                source_type="x",
                source_version="v1",
                output=[[0]],
                generation_parameters={"temperature": math.nan},
            )

    def test_direct_constructor_rejects_duplicate_generation_parameter_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            CandidateRecord(
                candidate_id="0" * 64,
                task_id="abc",
                test_index=0,
                source_type="x",
                source_version="v1",
                output=((0,),),
                output_key="0" * 64,
                parent_candidate_ids=(),
                parse_hypothesis_id=None,
                program_hash=None,
                functional_trace=(),
                generation_parameters_json='{"x":1,"x":2}',
                model_confidence=None,
                cost_cpu_ms=0,
                cost_gpu_ms=0,
                model_calls=0,
                evidence_status="candidate",
            )

    def test_numeric_fields_are_canonicalized_for_stable_id(self) -> None:
        common = dict(
            task_id="abc",
            test_index=0,
            source_type="x",
            source_version="v1",
            output=[[0]],
        )
        integer_form = CandidateRecord.create(
            **common, model_confidence=1, cost_cpu_ms=0, cost_gpu_ms=0
        )
        float_form = CandidateRecord.create(
            **common, model_confidence=1.0, cost_cpu_ms=0.0, cost_gpu_ms=0.0
        )
        self.assertEqual(integer_form.candidate_id, float_form.candidate_id)

    def test_public_constructor_rejects_forged_derived_fields(self) -> None:
        valid = CandidateRecord.create(
            task_id="abc",
            test_index=0,
            source_type="x",
            source_version="v1",
            output=[[0]],
        )
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            CandidateRecord(
                candidate_id="0" * 64,
                task_id=valid.task_id,
                test_index=valid.test_index,
                source_type=valid.source_type,
                source_version=valid.source_version,
                output=[[0]],  # type: ignore[arg-type]
                output_key=valid.output_key,
                parent_candidate_ids=[],  # type: ignore[arg-type]
                parse_hypothesis_id=None,
                program_hash=None,
                functional_trace=[],  # type: ignore[arg-type]
                generation_parameters_json="{}",
                model_confidence=None,
                cost_cpu_ms=0,
                cost_gpu_ms=0,
                model_calls=0,
                evidence_status="candidate",
            )

    def test_factory_rejects_non_string_task_id_without_mutating_store(self) -> None:
        store = CandidateStore()
        with self.assertRaises(TypeError):
            candidate = CandidateRecord.create(
                task_id=["abc"],  # type: ignore[arg-type]
                test_index=0,
                source_type="x",
                source_version="v1",
                output=[[0]],
            )
            store.add(candidate)
        self.assertEqual(len(store), 0)
        self.assertEqual(store.unique_output_count, 0)

    def test_json_round_trip_rechecks_content_address(self) -> None:
        candidate = CandidateRecord.create(
            task_id="abc",
            test_index=0,
            source_type="dsl",
            source_version="v1",
            output=[[1, 2]],
            generation_parameters={"depth": 1},
        )
        self.assertEqual(CandidateRecord.from_json_dict(candidate.to_json_dict()), candidate)
        forged = candidate.to_json_dict()
        forged["output_key"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "output_key"):
            CandidateRecord.from_json_dict(forged)


if __name__ == "__main__":
    unittest.main()
