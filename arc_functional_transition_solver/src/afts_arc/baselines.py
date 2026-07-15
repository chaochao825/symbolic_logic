"""Deterministic Phase-0 candidate generators used to validate the harness."""

from __future__ import annotations

from .candidate import CandidateRecord
from .grid import dihedral_variants
from .task import ARCTask


def dihedral_candidates(task: ARCTask, *, version: str = "phase0-d4-v1") -> tuple[CandidateRecord, ...]:
    """Generate unique D4 transforms of each test input.

    This is a harness baseline, not a general ARC solver. It deliberately does not
    inspect reference outputs or demonstrations when producing candidates.
    """

    candidates: list[CandidateRecord] = []
    for test_index, pair in enumerate(task.test):
        for transform_name, output in dihedral_variants(pair.input):
            candidates.append(
                CandidateRecord.create(
                    task_id=task.task_id,
                    test_index=test_index,
                    source_type="deterministic_d4",
                    source_version=version,
                    output=output,
                    functional_trace=(transform_name,),
                    generation_parameters={"transform": transform_name},
                    evidence_status="unverified_candidate",
                )
            )
    return tuple(candidates)
