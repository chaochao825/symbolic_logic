from __future__ import annotations

import math
import unittest

from afts_arc.m04a_contract import OPTIMIZER_UPDATES, VALIDATION_INTERVAL
from afts_arc.m04a_train_contract import (
    CAMPAIGN_GPU_BUDGET_NS,
    COSINE_UPDATES,
    CheckpointMetric,
    assert_campaign_budget,
    learning_rate_for_update,
    select_checkpoint,
    training_config_payload,
)


class M04aTrainContractTests(unittest.TestCase):
    def test_learning_rate_endpoints_and_cosine_decay(self) -> None:
        self.assertEqual(learning_rate_for_update(1), 1.5e-7)
        self.assertEqual(learning_rate_for_update(2_000), 3e-4)
        self.assertEqual(
            learning_rate_for_update(2_001),
            3e-4 * 0.5 * (1.0 + math.cos(math.pi / COSINE_UPDATES)),
        )
        self.assertEqual(learning_rate_for_update(OPTIMIZER_UPDATES), 0.0)
        sampled = [
            learning_rate_for_update(step)
            for step in range(2_000, OPTIMIZER_UPDATES + 1, 137)
        ]
        self.assertTrue(all(left >= right for left, right in zip(sampled, sampled[1:])))
        with self.assertRaises(ValueError):
            learning_rate_for_update(OPTIMIZER_UPDATES + 1)

    def test_training_config_pins_one_full_decay_group(self) -> None:
        payload = training_config_payload()
        self.assertRegex(str(payload["config_id"]), r"^[0-9a-f]{64}$")
        optimizer = payload["optimizer"]
        self.assertIsInstance(optimizer, dict)
        self.assertEqual(optimizer["parameter_groups"], 1)
        self.assertEqual(optimizer["decay_exclusions"], [])
        self.assertFalse(payload["grad_scaler"])
        self.assertEqual(payload["preflight_updates"], 100)
        self.assertEqual(payload["optimizer_updates"], OPTIMIZER_UPDATES)
        self.assertEqual(
            payload["validation_pass_count"],
            OPTIMIZER_UPDATES // VALIDATION_INTERVAL,
        )

    def test_checkpoint_selection_is_task_ce_then_earlier_step(self) -> None:
        rows = tuple(
            CheckpointMetric(
                optimizer_step=step,
                parent_grouped_ce=(0.25 if step in {4_000, 6_000} else 1.0 + step / 100_000),
                checkpoint_sha256=f"{index + 1:064x}",
            )
            for index, step in enumerate(
                range(VALIDATION_INTERVAL, OPTIMIZER_UPDATES + 1, VALIDATION_INTERVAL)
            )
        )
        self.assertEqual(select_checkpoint(rows).optimizer_step, 4_000)
        with self.assertRaises(ValueError):
            select_checkpoint(rows[:-1])
        with self.assertRaisesRegex(TypeError, "exact CheckpointMetric"):
            select_checkpoint(tuple(object() for _ in rows))

    def test_campaign_budget_is_strictly_greater_than_24_hours(self) -> None:
        assert_campaign_budget(CAMPAIGN_GPU_BUDGET_NS)
        with self.assertRaisesRegex(RuntimeError, "BUDGET_EXCEEDED"):
            assert_campaign_budget(CAMPAIGN_GPU_BUDGET_NS + 1)


if __name__ == "__main__":
    unittest.main()
