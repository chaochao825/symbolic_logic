from __future__ import annotations

import multiprocessing

import pytest

from scripts.build_arcgen_visual_trace_repair_cohort import _validated_arc1_ids
from scripts.build_arcgen_visual_trace_repair_cohort import _scan_family_worker


def _broken_generator() -> object:
    raise IndexError("external generator defect")


def test_external_generator_failure_is_typed() -> None:
    receiver, sender = multiprocessing.Pipe(duplex=False)

    _scan_family_worker("broken", _broken_generator, sender)
    result = receiver.recv()
    receiver.close()

    assert result[:4] == (None, None, None, None)
    assert result[4] == {
        "reason": "generator_failure",
        "pair_index": 0,
        "exception_type": "IndexError",
        "exception_message": "external generator defect",
    }


def test_arc1_identity_source_must_be_locked_and_complete(tmp_path) -> None:
    arcgen_root = tmp_path / "ARC-GEN"
    training = arcgen_root / "external" / "ARC-AGI" / "data" / "training"
    evaluation = arcgen_root / "external" / "ARC-AGI" / "data" / "evaluation"
    training.mkdir(parents=True)
    evaluation.mkdir(parents=True)

    with pytest.raises(ValueError, match="incomplete"):
        _validated_arc1_ids(arcgen_root, training, evaluation)

    with pytest.raises(ValueError, match="locked ARC-GEN submodule"):
        _validated_arc1_ids(arcgen_root, tmp_path, evaluation)
