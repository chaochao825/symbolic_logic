from __future__ import annotations

from afts_arc.relational_mask import RelationalMaskProgram
from scripts.analyze_relational_mask_near_miss_quality import parent_quality


def test_parent_quality_rejects_background_dominated_false_near_miss() -> None:
    train = [
        {
            "input": [[8, 1, 8], [1, 8, 1], [8, 1, 8]],
            "output": [[8, 1, 8], [1, 4, 1], [8, 1, 8]],
        }
    ]
    wrong = RelationalMaskProgram(8, 1, 4, "cardinal4", (1,), "blank")

    quality = parent_quality(wrong, train)

    assert quality["delta_f1"] < 0.5
    assert quality["parent_improves_identity"] is False
    assert quality["quality_near_miss"] is False
