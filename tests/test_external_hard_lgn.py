import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "results" / "external" / "hard_lgn_v23"
sys.path.insert(0, str(ROOT / "src"))

from summarize_external_hard_lgn import summarize  # noqa: E402


class ExternalHardLgnSummaryTests(unittest.TestCase):
    def test_summary_keeps_candidate_methods_and_abc_semantics_separate(self) -> None:
        abc = [
            {
                "abc_status": "ok",
                "post_acc_delta_vs_pre_discrete": "0.0",
                "abc_and_reduction_vs_pre_nd": "0.5",
            }
        ]
        hardening = [
            {"selected": "1", "method": "redundant_task_hardened", "candidate": "argmax_best_hard"},
            {"selected": "1", "method": "redundant_task_hardened", "candidate": "gumbel_sample_7"},
            {"selected": "1", "method": "redundant_task_hardened", "candidate": "truth_table_refit"},
            {"selected": "1", "method": "more_gates_only", "candidate": "argmax_best_soft"},
        ]
        truth = [
            {"status_vs_task_selected": "WIN", "status_vs_strong_baseline": "LOSS"},
            {"status_vs_task_selected": "LOSS", "status_vs_strong_baseline": "WIN"},
        ]
        rows, selected = summarize(abc, hardening, truth)
        metrics = {row["metric"]: row for row in rows}
        self.assertEqual(len(selected), 3)
        self.assertEqual(metrics["selected_argmax_best_hard"]["value"], 1)
        self.assertEqual(metrics["selected_best_of_32_gumbel"]["value"], 1)
        self.assertEqual(metrics["selected_truth_table_refit"]["value"], 1)
        self.assertEqual(metrics["hard_accuracy_preserved_rows"]["status"], "pass")
        self.assertIn("task inputs", metrics["hard_accuracy_preserved_rows"]["interpretation"])
        self.assertIn(
            "cross-representation statistic",
            metrics["minimum_reported_post_abc_and_vs_pre_blif_node_reduction"]["interpretation"],
        )

        no_abc_rows, _ = summarize([], hardening, truth)
        no_abc = {row["metric"]: row for row in no_abc_rows}
        self.assertEqual(no_abc["successful_synthesis_rows"]["status"], "no_data")
        self.assertEqual(no_abc["hard_accuracy_preserved_rows"]["status"], "no_data")
        self.assertEqual(no_abc["minimum_reported_post_abc_and_vs_pre_blif_node_reduction"]["status"], "no_data")

    def test_committed_external_bridge_hashes_survive_line_ending_conversion(self) -> None:
        metadata = json.loads((EXTERNAL / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("CRLF-to-LF", metadata["hash_mode"])
        for name, expected in metadata["derived_normalized_sha256"].items():
            payload = (EXTERNAL / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
