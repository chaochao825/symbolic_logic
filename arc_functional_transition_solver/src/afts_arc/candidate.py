"""Immutable candidate records and exact-output deduplication."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .grid import Grid, as_grid, grid_key, grid_to_lists


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _nonempty_string(value: object, *, field: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _ordered_strings(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{field} must be an ordered tuple or list of strings")
    normalized = tuple(value)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise TypeError(f"{field} must contain non-empty strings")
    return normalized


def _strict_json_object(serialized: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(serialized, str):
        raise TypeError("generation_parameters_json must be a string")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key is not allowed: {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            serialized,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("generation_parameters_json must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError("generation parameters must be a JSON object")
    canonical = _canonical_json(parsed)
    return canonical, parsed


def _normalized_float(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or (nonnegative and normalized < 0):
        raise ValueError(f"{field} must be finite and non-negative")
    return normalized


def _candidate_payload(
    *,
    task_id: str,
    test_index: int,
    source_type: str,
    source_version: str,
    output: Grid,
    parent_candidate_ids: tuple[str, ...],
    parse_hypothesis_id: str | None,
    program_hash: str | None,
    functional_trace: tuple[str, ...],
    generation_parameters: dict[str, Any],
    model_confidence: float | None,
    cost_cpu_ms: float,
    cost_gpu_ms: float,
    model_calls: int,
    evidence_status: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "test_index": test_index,
        "source_type": source_type,
        "source_version": source_version,
        "output": grid_to_lists(output),
        "parents": list(parent_candidate_ids),
        "parse": parse_hypothesis_id,
        "program_hash": program_hash,
        "functional_trace": list(functional_trace),
        "generation_parameters": generation_parameters,
        "model_confidence": model_confidence,
        "cost_cpu_ms": cost_cpu_ms,
        "cost_gpu_ms": cost_gpu_ms,
        "model_calls": model_calls,
        "evidence_status": evidence_status,
    }


def _payload_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    task_id: str
    test_index: int
    source_type: str
    source_version: str
    output: Grid
    output_key: str
    parent_candidate_ids: tuple[str, ...]
    parse_hypothesis_id: str | None
    program_hash: str | None
    functional_trace: tuple[str, ...]
    generation_parameters_json: str
    model_confidence: float | None
    cost_cpu_ms: float
    cost_gpu_ms: float
    model_calls: int
    evidence_status: str

    def __post_init__(self) -> None:
        task_id = _nonempty_string(self.task_id, field="task_id")
        source_type = _nonempty_string(self.source_type, field="source_type")
        source_version = _nonempty_string(self.source_version, field="source_version")
        evidence_status = _nonempty_string(self.evidence_status, field="evidence_status")
        parse_id = _nonempty_string(
            self.parse_hypothesis_id, field="parse_hypothesis_id", optional=True
        )
        program_hash = _nonempty_string(
            self.program_hash, field="program_hash", optional=True
        )
        if type(self.test_index) is not int or self.test_index < 0:
            raise TypeError("test_index must be a non-negative integer")
        if type(self.model_calls) is not int or self.model_calls < 0:
            raise TypeError("model_calls must be a non-negative integer")

        output = as_grid(self.output)
        parents = _ordered_strings(self.parent_candidate_ids, field="parent_candidate_ids")
        trace = _ordered_strings(self.functional_trace, field="functional_trace")
        parameters_json, parameters = _strict_json_object(self.generation_parameters_json)
        confidence = (
            None
            if self.model_confidence is None
            else _normalized_float(self.model_confidence, field="model_confidence")
        )
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("model_confidence must be in [0, 1]")
        cpu_ms = _normalized_float(self.cost_cpu_ms, field="cost_cpu_ms", nonnegative=True)
        gpu_ms = _normalized_float(self.cost_gpu_ms, field="cost_gpu_ms", nonnegative=True)

        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "source_version", source_version)
        object.__setattr__(self, "evidence_status", evidence_status)
        object.__setattr__(self, "parse_hypothesis_id", parse_id)
        object.__setattr__(self, "program_hash", program_hash)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "parent_candidate_ids", parents)
        object.__setattr__(self, "functional_trace", trace)
        object.__setattr__(self, "generation_parameters_json", parameters_json)
        object.__setattr__(self, "model_confidence", confidence)
        object.__setattr__(self, "cost_cpu_ms", cpu_ms)
        object.__setattr__(self, "cost_gpu_ms", gpu_ms)

        expected_output_key = grid_key(output)
        payload = _candidate_payload(
            task_id=task_id,
            test_index=self.test_index,
            source_type=source_type,
            source_version=source_version,
            output=output,
            parent_candidate_ids=parents,
            parse_hypothesis_id=parse_id,
            program_hash=program_hash,
            functional_trace=trace,
            generation_parameters=parameters,
            model_confidence=confidence,
            cost_cpu_ms=cpu_ms,
            cost_gpu_ms=gpu_ms,
            model_calls=self.model_calls,
            evidence_status=evidence_status,
        )
        expected_candidate_id = _payload_id(payload)
        if self.output_key != expected_output_key:
            raise ValueError("output_key does not match output content")
        if self.candidate_id != expected_candidate_id:
            raise ValueError("candidate_id does not match canonical candidate content")

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        test_index: int,
        source_type: str,
        source_version: str,
        output: object,
        parent_candidate_ids: Sequence[str] = (),
        parse_hypothesis_id: str | None = None,
        program_hash: str | None = None,
        functional_trace: Sequence[str] = (),
        generation_parameters: dict[str, Any] | None = None,
        model_confidence: float | None = None,
        cost_cpu_ms: float = 0.0,
        cost_gpu_ms: float = 0.0,
        model_calls: int = 0,
        evidence_status: str = "candidate",
    ) -> "CandidateRecord":
        task_id = _nonempty_string(task_id, field="task_id")
        source_type = _nonempty_string(source_type, field="source_type")
        source_version = _nonempty_string(source_version, field="source_version")
        evidence_status = _nonempty_string(evidence_status, field="evidence_status")
        parse_hypothesis_id = _nonempty_string(
            parse_hypothesis_id, field="parse_hypothesis_id", optional=True
        )
        program_hash = _nonempty_string(program_hash, field="program_hash", optional=True)
        if type(test_index) is not int or test_index < 0:
            raise TypeError("test_index must be a non-negative integer")
        if type(model_calls) is not int or model_calls < 0:
            raise TypeError("model_calls must be a non-negative integer")
        normalized_output = as_grid(output)
        normalized_parents = _ordered_strings(parent_candidate_ids, field="parent_candidate_ids")
        normalized_trace = _ordered_strings(functional_trace, field="functional_trace")
        if generation_parameters is not None and not isinstance(generation_parameters, dict):
            raise TypeError("generation_parameters must be a dict or None")
        parameters_json = _canonical_json(generation_parameters or {})
        normalized_parameters_json, normalized_parameters = _strict_json_object(parameters_json)
        normalized_confidence = (
            None
            if model_confidence is None
            else _normalized_float(model_confidence, field="model_confidence")
        )
        if normalized_confidence is not None and not 0.0 <= normalized_confidence <= 1.0:
            raise ValueError("model_confidence must be in [0, 1]")
        normalized_cpu_ms = _normalized_float(
            cost_cpu_ms, field="cost_cpu_ms", nonnegative=True
        )
        normalized_gpu_ms = _normalized_float(
            cost_gpu_ms, field="cost_gpu_ms", nonnegative=True
        )
        payload = _candidate_payload(
            task_id=task_id,
            test_index=test_index,
            source_type=source_type,
            source_version=source_version,
            output=normalized_output,
            parent_candidate_ids=normalized_parents,
            parse_hypothesis_id=parse_hypothesis_id,
            program_hash=program_hash,
            functional_trace=normalized_trace,
            generation_parameters=normalized_parameters,
            model_confidence=normalized_confidence,
            cost_cpu_ms=normalized_cpu_ms,
            cost_gpu_ms=normalized_gpu_ms,
            model_calls=model_calls,
            evidence_status=evidence_status,
        )
        candidate_id = _payload_id(payload)
        return cls(
            candidate_id=candidate_id,
            task_id=task_id,
            test_index=test_index,
            source_type=source_type,
            source_version=source_version,
            output=normalized_output,
            output_key=grid_key(normalized_output),
            parent_candidate_ids=normalized_parents,
            parse_hypothesis_id=parse_hypothesis_id,
            program_hash=program_hash,
            functional_trace=normalized_trace,
            generation_parameters_json=normalized_parameters_json,
            model_confidence=normalized_confidence,
            cost_cpu_ms=normalized_cpu_ms,
            cost_gpu_ms=normalized_gpu_ms,
            model_calls=model_calls,
            evidence_status=evidence_status,
        )

    @classmethod
    def from_json_dict(cls, payload: object) -> "CandidateRecord":
        """Load the public record schema and recheck every derived identity field."""

        if not isinstance(payload, dict):
            raise TypeError("candidate record must be a JSON object")
        expected = {
            "candidate_id",
            "task_id",
            "test_index",
            "source_type",
            "source_version",
            "output",
            "output_key",
            "parent_candidate_ids",
            "parse_hypothesis_id",
            "program_hash",
            "functional_trace",
            "generation_parameters",
            "model_confidence",
            "cost_cpu_ms",
            "cost_gpu_ms",
            "model_calls",
            "evidence_status",
        }
        if set(payload) != expected:
            raise ValueError(
                f"candidate fields must be {sorted(expected)}, found {sorted(payload)}"
            )
        if not isinstance(payload["generation_parameters"], dict):
            raise TypeError("generation_parameters must be a JSON object")
        return cls(
            candidate_id=payload["candidate_id"],
            task_id=payload["task_id"],
            test_index=payload["test_index"],
            source_type=payload["source_type"],
            source_version=payload["source_version"],
            output=payload["output"],
            output_key=payload["output_key"],
            parent_candidate_ids=payload["parent_candidate_ids"],
            parse_hypothesis_id=payload["parse_hypothesis_id"],
            program_hash=payload["program_hash"],
            functional_trace=payload["functional_trace"],
            generation_parameters_json=_canonical_json(payload["generation_parameters"]),
            model_confidence=payload["model_confidence"],
            cost_cpu_ms=payload["cost_cpu_ms"],
            cost_gpu_ms=payload["cost_gpu_ms"],
            model_calls=payload["model_calls"],
            evidence_status=payload["evidence_status"],
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "test_index": self.test_index,
            "source_type": self.source_type,
            "source_version": self.source_version,
            "output": grid_to_lists(self.output),
            "output_key": self.output_key,
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "parse_hypothesis_id": self.parse_hypothesis_id,
            "program_hash": self.program_hash,
            "functional_trace": list(self.functional_trace),
            "generation_parameters": json.loads(self.generation_parameters_json),
            "model_confidence": self.model_confidence,
            "cost_cpu_ms": self.cost_cpu_ms,
            "cost_gpu_ms": self.cost_gpu_ms,
            "model_calls": self.model_calls,
            "evidence_status": self.evidence_status,
        }


class CandidateStore:
    """Keep full provenance while grouping exact duplicate outputs."""

    def __init__(self) -> None:
        self._by_id: dict[str, CandidateRecord] = {}
        self._by_output: dict[tuple[str, int, str], list[str]] = {}

    def add(self, candidate: CandidateRecord) -> bool:
        """Add a candidate and return ``True`` iff its record ID was new."""

        if not isinstance(candidate, CandidateRecord):
            raise TypeError("candidate must be a CandidateRecord")
        group_key = (candidate.task_id, candidate.test_index, candidate.output_key)
        hash(group_key)
        if candidate.candidate_id in self._by_id:
            return False
        self._by_id[candidate.candidate_id] = candidate
        self._by_output.setdefault(group_key, []).append(candidate.candidate_id)
        return True

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def unique_output_count(self) -> int:
        return len(self._by_output)

    def candidates(self) -> tuple[CandidateRecord, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))

    def output_groups(self) -> tuple[tuple[CandidateRecord, ...], ...]:
        groups: list[tuple[CandidateRecord, ...]] = []
        for group_key in sorted(self._by_output):
            ids = self._by_output[group_key]
            groups.append(tuple(self._by_id[candidate_id] for candidate_id in ids))
        return tuple(groups)

    def records_for_output(
        self, *, task_id: str, test_index: int, output: object
    ) -> tuple[CandidateRecord, ...]:
        normalized = as_grid(output)
        ids = self._by_output.get((task_id, test_index, grid_key(normalized)), [])
        return tuple(self._by_id[candidate_id] for candidate_id in ids)
