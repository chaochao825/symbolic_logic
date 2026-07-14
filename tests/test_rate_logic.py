import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logic_core import all_assignments
from rate_logic_experiments import (
    DifferentiableGateSelector,
    apply_gate,
    coding_rate,
    empirical_code_entropy,
    fit_gate_hypothesis,
    fit_rate_guided_gate,
    mcr2,
    mcr2_gradient,
    run_rate_flow,
)


class RateLogicTests(unittest.TestCase):
    def test_joint_selector_recovers_operator_and_distractor_pair(self) -> None:
        x = all_assignments(6)
        y = apply_gate("XOR", x[:, 2], x[:, 5])
        model = fit_gate_hypothesis(x, y)
        self.assertEqual((model.left, model.right, model.op), (2, 5, "XOR"))

    def test_differentiable_selector_recovers_operator_and_pair(self) -> None:
        x = all_assignments(6)
        y = apply_gate("NAND", x[:, 1], x[:, 4])
        model = DifferentiableGateSelector(6, seed=9, steps=700).fit(x, y)
        self.assertEqual(model.selected(), (1, 4, "NAND"))
        self.assertGreater(model.weights().max(), 0.95)

    def test_scalar_mcr2_is_sign_blind_for_boolean_gate_selection(self) -> None:
        x = all_assignments(6)
        y = apply_gate("OR", x[:, 0], x[:, 3])
        model, score = fit_rate_guided_gate(x, y)
        self.assertEqual(score, 0.0)
        self.assertNotEqual((model.left, model.right, model.op), (0, 3, "OR"))

    def test_coding_rate_is_measured_in_base_two(self) -> None:
        z = np.asarray([[1.0, -1.0]])
        expected = 0.5 * np.log2(1.0 + 1.0 / (2 * 0.5**2) * 2.0)
        self.assertAlmostEqual(coding_rate(z, 0.5), expected)

    def test_mcr2_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(3)
        z = rng.normal(size=(3, 8))
        labels = np.repeat([0, 1], 4)
        analytic = mcr2_gradient(z, labels, epsilon=0.7)
        numerical = np.zeros_like(z)
        delta = 1e-6
        for row in range(z.shape[0]):
            for column in range(z.shape[1]):
                plus, minus = z.copy(), z.copy()
                plus[row, column] += delta
                minus[row, column] -= delta
                numerical[row, column] = (mcr2(plus, labels, 0.7)[2] - mcr2(minus, labels, 0.7)[2]) / (2 * delta)
        self.assertTrue(np.allclose(analytic, numerical, atol=2e-6, rtol=2e-5))

    def test_rate_flow_increases_controlled_objective(self) -> None:
        rng = np.random.default_rng(7)
        z = rng.normal(size=(5, 40))
        labels = np.repeat([0, 1], 20)
        before = mcr2(z / np.linalg.norm(z, axis=0, keepdims=True), labels)[2]
        after = mcr2(run_rate_flow(z, labels, layers=8, step_size=0.15), labels)[2]
        self.assertGreater(after, before)

    def test_empirical_entropy_uses_observed_codewords(self) -> None:
        codes = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)
        self.assertAlmostEqual(empirical_code_entropy(codes), 1.0)


if __name__ == "__main__":
    unittest.main()
