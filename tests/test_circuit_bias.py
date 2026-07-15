import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit_bias_experiments import (  # noqa: E402
    BASIS_LIBRARY,
    BasisFormula,
    all_assignments,
    benchmark_exact_bases,
    boolean_diagnostics,
    basis_formula_dag_upper_bound_bits,
    formula_dag_metrics,
    mask_to_vector,
    named_truth_masks,
    residual_description_bits,
    synthesize_min_formula,
    vector_to_mask,
)
from run_circuit_bias import summarize_bases  # noqa: E402


class CircuitBiasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = all_assignments(3)
        self.named = named_truth_masks(3)

    def target(self, name: str) -> np.ndarray:
        return mask_to_vector(self.named[name], len(self.x))

    def test_truth_mask_round_trip(self) -> None:
        values = np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        self.assertTrue(np.array_equal(mask_to_vector(vector_to_mask(values), len(values)), values))
        with self.assertRaises(ValueError):
            vector_to_mask([0, 2, 1])

    def test_xag_is_exact_formula_oracle_for_parity3(self) -> None:
        formula, search = synthesize_min_formula(self.x, self.target("parity3"), BASIS_LIBRARY["XAG"])
        self.assertIsNotNone(formula)
        assert formula is not None
        self.assertTrue(np.array_equal(formula.evaluate(self.x), self.target("parity3")))
        self.assertEqual(formula.gate_count, 2)
        self.assertGreater(search.candidates_evaluated, 0)

    def test_mig_is_exact_formula_oracle_for_majority3(self) -> None:
        formula, _ = synthesize_min_formula(self.x, self.target("majority3"), BASIS_LIBRARY["MIG"])
        self.assertIsNotNone(formula)
        assert formula is not None
        self.assertEqual(formula.gate_count, 1)
        self.assertEqual(formula_dag_metrics(formula)["observed_hashconsed_dag_upper_bound_gates"], 1)
        complement, _ = synthesize_min_formula(self.x, 1 - self.target("majority3"), BASIS_LIBRARY["MIG"])
        self.assertIsNotNone(complement)
        assert complement is not None
        self.assertEqual(complement.gate_count, 1)

    def test_aig_reaches_parity_but_exposes_basis_cost(self) -> None:
        formula, search = synthesize_min_formula(self.x, self.target("parity3"), BASIS_LIBRARY["AIG"])
        self.assertIsNotNone(formula)
        assert formula is not None
        self.assertTrue(search.complete)
        self.assertGreater(formula.gate_count, 2)

    def test_boolean_diagnostics_separate_affine_and_monotone_bias(self) -> None:
        parity = boolean_diagnostics(self.target("parity3"), 3)
        majority = boolean_diagnostics(self.target("majority3"), 3)
        one = boolean_diagnostics(np.ones(len(self.x), dtype=np.uint8), 3)
        self.assertEqual(parity["family"], "affine")
        self.assertEqual(parity["anf_degree"], 1)
        self.assertEqual(parity["fourier_degree"], 3)
        self.assertAlmostEqual(parity["total_influence"], 3.0)
        self.assertEqual(majority["family"], "monotone_symmetric")
        self.assertEqual(majority["certificate_c0"], 2)
        self.assertEqual(majority["certificate_c1"], 2)
        self.assertEqual(one["family"], "constant")

    def test_mdl_accounts_for_model_and_residual(self) -> None:
        formula, _ = synthesize_min_formula(self.x, self.target("and3"), BASIS_LIBRARY["AIG"])
        self.assertIsNotNone(formula)
        assert formula is not None
        model_bits = basis_formula_dag_upper_bound_bits(formula, BASIS_LIBRARY["AIG"], 3, n_bases=4)
        exact_residual = residual_description_bits(self.target("and3"), formula.evaluate(self.x))
        wrong_residual = residual_description_bits(self.target("and3"), np.zeros(len(self.x), dtype=np.uint8))
        self.assertGreater(model_bits, 0)
        self.assertEqual(exact_residual, 1.0)  # the zero-error count header is still encoded
        self.assertGreater(wrong_residual, exact_residual)
        conditional_bits = basis_formula_dag_upper_bound_bits(formula, BASIS_LIBRARY["AIG"], 3)
        self.assertEqual(model_bits, conditional_bits + 2)  # four formula-basis routes are paid
        with self.assertRaises(ValueError):
            basis_formula_dag_upper_bound_bits(formula, BASIS_LIBRARY["MIG"], 3, n_bases=4)
        invalid_input = BasisFormula(0, "INPUT", "x3", 0, 0, input_index=3)
        with self.assertRaises(ValueError):
            basis_formula_dag_upper_bound_bits(invalid_input, BASIS_LIBRARY["AIG"], 3, n_bases=4)

    def test_hashcons_metrics_are_only_an_observed_dag_upper_bound(self) -> None:
        left = BasisFormula(vector_to_mask(self.x[:, 0]), "INPUT", "x0", 0, 0, input_index=0)
        right = BasisFormula(vector_to_mask(self.x[:, 1]), "INPUT", "x1", 0, 0, input_index=1)
        shared = BasisFormula(left.mask & right.mask, "AND", "AND(x0,x1)", 1, 1, args=(left, right))
        root = BasisFormula(0, "XOR", "XOR(shared,shared)", 3, 2, args=(shared, shared))
        metrics = formula_dag_metrics(root)
        self.assertEqual(metrics["formula_gate_refs"], 3)
        self.assertEqual(metrics["observed_hashconsed_dag_upper_bound_gates"], 2)
        self.assertEqual(metrics["internal_gate_fanout_max_including_output"], 2)

    def test_full_oracle_covers_all_256_functions_and_smoke_scope_is_not_full(self) -> None:
        full_rows, full_diagnostics = benchmark_exact_bases("full")
        self.assertEqual(len(full_rows), 4 * 256)
        self.assertEqual(len(full_diagnostics), 256)
        self.assertTrue(all(row["exact"] == 1 and row["search_complete"] == 1 for row in full_rows))
        selected_witness_kraft = sum(
            2.0 ** -float(row["basis_formula_dag_code_upper_bound_bits"])
            for row in full_rows
            if row["exact"] == 1
        )
        self.assertLessEqual(selected_witness_kraft, 1.0)
        smoke_rows, smoke_diagnostics = benchmark_exact_bases("smoke")
        summaries, _, _ = summarize_bases(smoke_rows, smoke_diagnostics)
        self.assertTrue(all("complete 3-input function space" not in row["scope"] for row in summaries))


if __name__ == "__main__":
    unittest.main()
