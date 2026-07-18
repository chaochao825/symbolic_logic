from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from afts_arc.m04a_evidence import make_pair_cost_row
from afts_arc.m04a_pool_staging import M04APoolPairStagingSink


class M04aPoolStagingTests(unittest.TestCase):
    def test_streams_exact_chunks_and_never_claims_complete_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pair-staging"
            sink = M04APoolPairStagingSink(root, expected_pairs=(("task", 0),))
            materials = {
                "lane_traces.jsonl": b'{"trace":1}\n',
                "candidate_rows.jsonl": b'{"lane":1}\n',
                "encoder_forward_ledger.jsonl": b'{"encoder":1}\n',
                "batch_forward_ledger.jsonl": b'{"decoder":1}\n',
            }
            receipt = sink.append_pair(
                blind_task_id="task", test_index=0, materials=materials
            )
            self.assertEqual(receipt["pair_index"], 0)
            pair = make_pair_cost_row(
                blind_task_id="task",
                test_index=0,
                accepted_shapes=(),
                lane_ids=(),
                encoder_call_ids=(),
                decoder_call_ids=(),
                pair_wall_time_ns=1,
                serialization_ns=1,
            )
            result = sink.finalize(pair_cost_rows=(pair,))
            self.assertEqual(result["status"], "PAIR_MATERIALS_ONLY_NOT_POOL_COMPLETE")
            manifest = json.loads(
                (root / "pair_materials_manifest.json").read_text("utf-8")
            )
            self.assertNotEqual(manifest["status"], "COMPLETE")
            self.assertEqual((root / "lane_traces.jsonl").read_bytes(), materials["lane_traces.jsonl"])

    def test_abort_preserves_incomplete_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pair-staging"
            sink = M04APoolPairStagingSink(root, expected_pairs=(("task", 0),))
            sink.abort(failure_code="NONFINITE_LOGITS")
            marker = json.loads((root / "ABORTED.json").read_text("utf-8"))
            self.assertEqual(marker["status"], "ABORTED_NOT_POOL_COMPLETE")
            self.assertFalse((root / "pair_materials_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
