import csv
import hashlib
import json
import sys
import unittest
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boolean_mdl import (
    anf_coefficients,
    best_robdd,
    circuit_description_bits,
    discrete_rate_reduction,
    elias_delta_bits,
    exact_formula_library,
    formula_to_circuit,
    joint_dirichlet_code_bits,
    kt_binary_ideal_bits,
    kt_binary_prefix_bits,
    kt_independent_matrix_bits,
    mask_to_values,
    occam_error_bound,
    residual_code_bits,
    route_task_mdl,
    routed_function_description,
    task_mdl_candidates,
    threshold_library,
    values_to_mask,
)


def assignments(n_inputs: int) -> np.ndarray:
    values = np.arange(1 << n_inputs, dtype=np.uint64)[:, None]
    return ((values >> np.arange(n_inputs, dtype=np.uint64)) & 1).astype(np.uint8)


class BooleanMDLTests(unittest.TestCase):
    def test_elias_delta_known_lengths(self) -> None:
        self.assertEqual([elias_delta_bits(value) for value in (1, 2, 3, 4, 8)], [1, 4, 4, 5, 8])

    def test_kt_ideal_probabilities_normalize_for_fixed_length(self) -> None:
        total = 0.0
        for sequence in product((0, 1), repeat=5):
            total += 2.0 ** (-kt_binary_ideal_bits(sequence))
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_exact_prefix_lengths_do_not_overcount_power_of_two_probabilities(self) -> None:
        self.assertEqual(kt_binary_prefix_bits([0]), 1)
        self.assertEqual(kt_binary_prefix_bits([1]), 1)
        self.assertEqual(kt_independent_matrix_bits(np.asarray([[0]], dtype=np.uint8)), 1)
        codes = np.asarray([[0, 0]], dtype=np.uint8)
        self.assertEqual(joint_dirichlet_code_bits(codes), 2)
        empty_codes = np.empty((3, 0), dtype=np.uint8)
        self.assertEqual(kt_independent_matrix_bits(empty_codes), 0)
        self.assertEqual(joint_dirichlet_code_bits(empty_codes), 0)
        one_of_512 = np.zeros((1, 9), dtype=np.uint8)
        self.assertEqual(joint_dirichlet_code_bits(one_of_512), 9)

    def test_residual_code_satisfies_kraft_bound(self) -> None:
        truth = np.zeros(8, dtype=np.uint8)
        kraft = 0.0
        for mask in range(256):
            prediction = mask_to_values(mask, 8)
            kraft += 2.0 ** (-residual_code_bits(truth, prediction))
        self.assertLessEqual(kraft, 1.0)

    def test_independent_kt_is_basis_dependent_but_joint_code_is_not(self) -> None:
        bit = np.tile(np.asarray([0, 1], dtype=np.uint8), 64)
        redundant = np.stack((bit, bit), axis=1)
        transformed = np.stack((bit, bit ^ bit), axis=1)
        self.assertLess(kt_independent_matrix_bits(transformed), kt_independent_matrix_bits(redundant))
        self.assertEqual(joint_dirichlet_code_bits(transformed), joint_dirichlet_code_bits(redundant))

    def test_binary_codecs_reject_values_before_uint8_cast(self) -> None:
        with self.assertRaises(ValueError):
            joint_dirichlet_code_bits(np.asarray([[0, 256]]))
        with self.assertRaises(ValueError):
            kt_independent_matrix_bits(np.asarray([[0.0, np.nan]]))
        with self.assertRaises(ValueError):
            discrete_rate_reduction(np.asarray([[0], [1]]), np.asarray([0, 2]))

    def test_conditional_codec_can_lose_but_router_is_nonnegative(self) -> None:
        codes = np.zeros((128, 1), dtype=np.uint8)
        labels = np.tile(np.asarray([0, 1], dtype=np.uint8), 64)
        result = discrete_rate_reduction(codes, labels, joint=False)
        self.assertLess(result.raw_reduction_bits, 0)
        self.assertEqual(result.gain_vs_global_route_bits, 0)
        self.assertEqual(result.route_tag_bits, 1)
        self.assertEqual(result.routed_global_baseline_bits, result.global_bits + 1)
        self.assertEqual(result.routed_best_bits, result.routed_global_baseline_bits)

    def test_two_route_representation_code_satisfies_kraft(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        union_kraft = 0.0
        selected_kraft = 0.0
        for sequence in product((0, 1), repeat=4):
            result = discrete_rate_reduction(np.asarray(sequence, dtype=np.uint8)[:, None], labels, joint=True)
            union_kraft += 2.0 ** (-result.routed_global_baseline_bits)
            union_kraft += 2.0 ** (-(result.route_tag_bits + result.conditional_bits))
            selected_kraft += 2.0 ** (-result.routed_best_bits)
        self.assertLessEqual(union_kraft, 1.0)
        self.assertLessEqual(selected_kraft, 1.0)

    def test_label_cost_is_symmetric_when_labels_are_not_side_information(self) -> None:
        codes = np.tile(np.asarray([[0], [1]], dtype=np.uint8), (32, 1))
        labels = np.tile(np.asarray([0, 1], dtype=np.uint8), 32)
        side = discrete_rate_reduction(codes, labels, joint=True, labels_are_side_information=True)
        joint = discrete_rate_reduction(codes, labels, joint=True, labels_are_side_information=False)
        self.assertEqual(side.raw_reduction_bits, joint.raw_reduction_bits)
        self.assertEqual(joint.global_bits - side.global_bits, joint.label_bits)
        self.assertEqual(joint.conditional_bits - side.conditional_bits, joint.label_bits)
        self.assertEqual(side.routed_global_baseline_bits - side.routed_best_bits, side.gain_vs_global_route_bits)
        self.assertEqual(joint.routed_global_baseline_bits - joint.routed_best_bits, joint.gain_vs_global_route_bits)

    def test_exact_formula_library_covers_every_three_input_function(self) -> None:
        library = exact_formula_library(3, max_gates=4)
        self.assertEqual(len(library), 256)
        x = assignments(3)
        parity = np.bitwise_xor.reduce(x, axis=1)
        formula = library[values_to_mask(parity)]
        self.assertEqual(formula.gates, 2)
        circuit = formula_to_circuit(formula, 3)
        self.assertTrue(np.array_equal(circuit.evaluate(x), parity))
        self.assertGreater(circuit_description_bits(circuit), 0)

    def test_anf_mobius_transform_reconstructs_function(self) -> None:
        x = assignments(4)
        target = (x[:, 0] ^ (x[:, 1] & x[:, 2]) ^ x[:, 3]).astype(np.uint8)
        coefficients = anf_coefficients(target)
        reconstructed = np.zeros(len(x), dtype=np.uint8)
        for monomial in np.flatnonzero(coefficients):
            term = np.ones(len(x), dtype=np.uint8)
            for bit in range(4):
                if monomial & (1 << bit):
                    term &= x[:, bit]
            reconstructed ^= term
        self.assertTrue(np.array_equal(reconstructed, target))

    def test_robdd_and_threshold_models_are_semantically_exact(self) -> None:
        x = assignments(4)
        majority = (x.sum(axis=1) >= 2).astype(np.uint8)
        bdd, _ = best_robdd(majority)
        self.assertTrue(np.array_equal(bdd.evaluate(x), majority))
        threshold = threshold_library(4, max_abs_weight=1)[values_to_mask(majority)]
        self.assertEqual(threshold.weights, (1, 1, 1, 1))
        self.assertEqual(threshold.threshold, 2)

    def test_multi_language_code_has_kraft_slack_and_safe_gain(self) -> None:
        formulae = exact_formula_library(3, max_gates=4)
        thresholds = threshold_library(3, max_abs_weight=2)
        kraft = 0.0
        for mask in range(256):
            values = mask_to_values(mask, 8)
            _, bits, _ = routed_function_description(values, 3, formula_library=formulae, thresholds=thresholds)
            kraft += 2.0 ** (-bits)
            best, baseline, gain = route_task_mdl(task_mdl_candidates(values, 3, formula_library=formulae, thresholds=thresholds))
            self.assertGreaterEqual(gain, 0)
            self.assertLessEqual(best.total_bits, baseline)
        self.assertLessEqual(kraft, 1.0)

    def test_occam_bound_penalizes_longer_descriptions(self) -> None:
        short = occam_error_bound(0.1, 8, 1000)
        long = occam_error_bound(0.1, 128, 1000)
        self.assertLess(short, long)

    def test_committed_full_catalog_and_artifact_hashes(self) -> None:
        results = ROOT / "results"
        with (results / "discrete_theory_checks.json").open(encoding="utf-8") as handle:
            checks = json.load(handle)
        self.assertEqual(checks["formula4_coverage"], 65_536)
        self.assertTrue(checks["formula4_complete"])
        self.assertEqual(
            checks["formula4_all_histogram"],
            {"0": 6, "1": 28, "2": 196, "3": 1162, "4": 5452, "5": 17250, "6": 28404, "7": 12798, "8": 240},
        )
        self.assertEqual(checks["balanced4_functions"], 12_870)
        self.assertEqual(checks["minimum_safe_task_gain_bits"], 0)

        histogram = {}
        with (results / "exact_formula_balanced4_results.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                gates = int(row["minimum_formula_gates"])
                histogram[gates] = histogram.get(gates, 0) + 1
        self.assertEqual(histogram, {0: 4, 1: 10, 2: 46, 3: 299, 4: 1235, 5: 3424, 6: 4928, 7: 2756, 8: 168})

        with (results / "discrete_theory_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertFalse(metadata["git_at_start"]["dirty"])
        for manifest in (metadata["source_sha256"], metadata["artifact_sha256"]):
            for relative, expected in manifest.items():
                payload = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
                digest = hashlib.sha256(payload).hexdigest()
                self.assertEqual(digest, expected)


if __name__ == "__main__":
    unittest.main()
