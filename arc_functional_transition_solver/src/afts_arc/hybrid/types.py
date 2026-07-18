"""Shared, oracle-free types for the functional ARC router.

The unit of selection is a whole-task hypothesis: one replayable rule (or one
explicit proposal bundle) must cover every demonstration and every query input.
Only after selection is it expanded into the legacy per-query CandidateRecord
format.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

from ..grid import Grid, as_grid, grid_key, grid_to_lists
from ..residual import DemoResidual

if TYPE_CHECKING:
    from ..blind import BlindTask
    from .router import RouteDecision, TaskFeatures


GridReplay = Callable[[Grid], object | None]
HardVerifier = Callable[["BlindTask"], bool]


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_object(value: Mapping[str, Any] | None, *, field_name: str) -> str:
    normalized = dict(value or {})
    try:
        serialized = canonical_json(normalized)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be JSON serializable") from exc
    if not isinstance(json.loads(serialized), dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return serialized


def _normalize_outputs(values: Sequence[object] | None) -> tuple[Grid, ...] | None:
    if values is None:
        return None
    return tuple(as_grid(value.tolist() if hasattr(value, "tolist") else value) for value in values)


def _identity_payload(
    *,
    name: str,
    source: str,
    source_version: str,
    route: str,
    description_bits: int,
    verification_mode: str,
    functional_trace: tuple[str, ...],
    spec: Mapping[str, Any],
    parents: tuple[str, ...],
    demo_outputs: tuple[Grid, ...] | None,
    query_outputs: tuple[Grid, ...] | None,
) -> dict[str, object]:
    return {
        "name": name,
        "source": source,
        "source_version": source_version,
        "route": route,
        "description_bits": description_bits,
        "verification_mode": verification_mode,
        "functional_trace": functional_trace,
        "spec": dict(spec),
        "parents": parents,
        "explicit_demo_output_keys": (
            None if demo_outputs is None else [grid_key(grid) for grid in demo_outputs]
        ),
        "explicit_query_output_keys": (
            None if query_outputs is None else [grid_key(grid) for grid in query_outputs]
        ),
    }


@dataclass(frozen=True, slots=True)
class CandidateHypothesis:
    """A bundle-level candidate with deterministic identity and optional replay."""

    hypothesis_id: str
    name: str
    source: str
    source_version: str
    route: str
    description_bits: int
    verification_mode: str
    functional_trace: tuple[str, ...]
    spec_json: str
    metadata_json: str
    parent_hypothesis_ids: tuple[str, ...]
    replay: GridReplay | None = field(default=None, compare=False, repr=False)
    hard_verifier: HardVerifier | None = field(default=None, compare=False, repr=False)
    explicit_demo_outputs: tuple[Grid, ...] | None = field(default=None, compare=False)
    explicit_query_outputs: tuple[Grid, ...] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        for name in ("hypothesis_id", "name", "source", "source_version", "route"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")
        if self.verification_mode not in {"grid_only", "replayable", "hard_circuit"}:
            raise ValueError("verification_mode must be grid_only, replayable, or hard_circuit")
        if type(self.description_bits) is not int or self.description_bits < 0:
            raise ValueError("description_bits must be a non-negative integer")
        if self.verification_mode != "grid_only" and self.replay is None:
            raise ValueError("replayable and hard-circuit candidates require replay")
        if self.verification_mode == "grid_only" and (
            self.explicit_demo_outputs is None or self.explicit_query_outputs is None
        ):
            raise ValueError("grid-only candidates require complete demo and query bundles")
        if not callable(self.replay) and self.replay is not None:
            raise TypeError("replay must be callable")
        if not callable(self.hard_verifier) and self.hard_verifier is not None:
            raise TypeError("hard_verifier must be callable")
        if len(self.hypothesis_id) != 24:
            raise ValueError("hypothesis_id must be a 24-character content identifier")
        decoded_objects: dict[str, dict[str, Any]] = {}
        for serialized, name in ((self.spec_json, "spec_json"), (self.metadata_json, "metadata_json")):
            try:
                decoded = json.loads(serialized)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} is not valid JSON") from exc
            if not isinstance(decoded, dict) or canonical_json(decoded) != serialized:
                raise ValueError(f"{name} must be a canonical JSON object")
            decoded_objects[name] = decoded
        demos = _normalize_outputs(self.explicit_demo_outputs)
        queries = _normalize_outputs(self.explicit_query_outputs)
        object.__setattr__(self, "explicit_demo_outputs", demos)
        object.__setattr__(self, "explicit_query_outputs", queries)
        expected_id = hashlib.sha256(
            canonical_json(
                _identity_payload(
                    name=self.name,
                    source=self.source,
                    source_version=self.source_version,
                    route=self.route,
                    description_bits=self.description_bits,
                    verification_mode=self.verification_mode,
                    functional_trace=tuple(self.functional_trace),
                    spec=decoded_objects["spec_json"],
                    parents=tuple(self.parent_hypothesis_ids),
                    demo_outputs=demos,
                    query_outputs=queries,
                )
            ).encode("ascii")
        ).hexdigest()[:24]
        if self.hypothesis_id != expected_id:
            raise ValueError("hypothesis_id does not match canonical hypothesis content")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        source: str,
        source_version: str,
        route: str,
        description_bits: int,
        verification_mode: str,
        functional_trace: Sequence[str],
        spec: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        parent_hypothesis_ids: Sequence[str] = (),
        replay: GridReplay | None = None,
        hard_verifier: HardVerifier | None = None,
        explicit_demo_outputs: Sequence[object] | None = None,
        explicit_query_outputs: Sequence[object] | None = None,
    ) -> "CandidateHypothesis":
        if not isinstance(name, str) or not name:
            raise TypeError("name must be a non-empty string")
        trace = tuple(functional_trace)
        if any(not isinstance(item, str) or not item for item in trace):
            raise TypeError("functional_trace must contain non-empty strings")
        parents = tuple(parent_hypothesis_ids)
        if any(not isinstance(item, str) or not item for item in parents):
            raise TypeError("parent_hypothesis_ids must contain non-empty strings")
        spec_json = _json_object(spec, field_name="spec")
        metadata_json = _json_object(metadata, field_name="metadata")
        demos = _normalize_outputs(explicit_demo_outputs)
        queries = _normalize_outputs(explicit_query_outputs)
        identity_payload = _identity_payload(
            name=name,
            source=source,
            source_version=source_version,
            route=route,
            description_bits=description_bits,
            verification_mode=verification_mode,
            functional_trace=trace,
            spec=json.loads(spec_json),
            parents=parents,
            demo_outputs=demos,
            query_outputs=queries,
        )
        hypothesis_id = hashlib.sha256(canonical_json(identity_payload).encode("ascii")).hexdigest()[:24]
        return cls(
            hypothesis_id=hypothesis_id,
            name=name,
            source=source,
            source_version=source_version,
            route=route,
            description_bits=description_bits,
            verification_mode=verification_mode,
            functional_trace=trace,
            spec_json=spec_json,
            metadata_json=metadata_json,
            parent_hypothesis_ids=parents,
            replay=replay,
            hard_verifier=hard_verifier,
            explicit_demo_outputs=demos,
            explicit_query_outputs=queries,
        )

    @property
    def spec(self) -> dict[str, Any]:
        return json.loads(self.spec_json)

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "source": self.source,
            "source_version": self.source_version,
            "route": self.route,
            "description_bits": self.description_bits,
            "verification_mode": self.verification_mode,
            "functional_trace": list(self.functional_trace),
            "spec": self.spec,
            "metadata": self.metadata,
            "parent_hypothesis_ids": list(self.parent_hypothesis_ids),
            "explicit_demo_output_keys": (
                None
                if self.explicit_demo_outputs is None
                else [grid_key(grid) for grid in self.explicit_demo_outputs]
            ),
            "explicit_query_output_keys": (
                None
                if self.explicit_query_outputs is None
                else [grid_key(grid) for grid in self.explicit_query_outputs]
            ),
        }


@dataclass(frozen=True, slots=True)
class ProviderResult:
    provider: str
    route: str
    status: str
    reason: str | None
    candidates: tuple[CandidateHypothesis, ...]
    diagnostics_json: str = "{}"

    def __post_init__(self) -> None:
        if self.status not in {"ok", "abstained", "error"}:
            raise ValueError("provider status must be ok, abstained, or error")
        if self.status == "ok" and not self.candidates:
            raise ValueError("an ok provider result must contain candidates")
        if self.status != "ok" and self.candidates:
            raise ValueError("abstained/error provider results cannot contain candidates")
        if any(not isinstance(item, CandidateHypothesis) for item in self.candidates):
            raise TypeError("provider candidates must be CandidateHypothesis values")
        decoded = json.loads(self.diagnostics_json)
        if not isinstance(decoded, dict) or canonical_json(decoded) != self.diagnostics_json:
            raise ValueError("diagnostics_json must be a canonical JSON object")

    @classmethod
    def ok(
        cls,
        provider: str,
        route: str,
        candidates: Sequence[CandidateHypothesis],
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "ProviderResult":
        normalized = tuple(candidates)
        if not normalized:
            return cls.abstained(provider, route, "no_candidates", diagnostics)
        return cls(provider, route, "ok", None, normalized, _json_object(diagnostics, field_name="diagnostics"))

    @classmethod
    def abstained(
        cls,
        provider: str,
        route: str,
        reason: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "ProviderResult":
        return cls(provider, route, "abstained", reason, (), _json_object(diagnostics, field_name="diagnostics"))

    @classmethod
    def error(
        cls,
        provider: str,
        route: str,
        reason: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "ProviderResult":
        return cls(provider, route, "error", reason, (), _json_object(diagnostics, field_name="diagnostics"))

    @property
    def diagnostics(self) -> dict[str, Any]:
        return json.loads(self.diagnostics_json)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "route": self.route,
            "status": self.status,
            "reason": self.reason,
            "candidate_count": len(self.candidates),
            "candidate_ids": [item.hypothesis_id for item in self.candidates],
            "diagnostics": self.diagnostics,
        }


class CandidateProvider(Protocol):
    name: str
    route: str

    def propose(
        self,
        task: "BlindTask",
        features: "TaskFeatures",
        decision: "RouteDecision",
    ) -> ProviderResult: ...


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    hypothesis: CandidateHypothesis
    demo_outputs: tuple[Grid | None, ...]
    query_outputs: tuple[Grid | None, ...]
    residuals: tuple[DemoResidual, ...]
    demo_exact: bool
    hard_verified: bool
    rejection_reason: str | None
    route_bits: int
    residual_bits: int
    support_bits: int
    total_mdl_bits: int

    @property
    def eligible(self) -> bool:
        return self.rejection_reason is None and self.demo_exact and self.hard_verified

    @property
    def agreement(self) -> float:
        cells = sum(item.comparison_cells for item in self.residuals)
        return sum(item.overlap_matches for item in self.residuals) / cells if cells else 0.0

    def query_signature(self) -> tuple[Grid, ...] | None:
        if any(item is None for item in self.query_outputs):
            return None
        return tuple(item for item in self.query_outputs if item is not None)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "hypothesis": self.hypothesis.to_json_dict(),
            "demo_exact": self.demo_exact,
            "hard_verified": self.hard_verified,
            "eligible": self.eligible,
            "rejection_reason": self.rejection_reason,
            "agreement": self.agreement,
            "mdl": {
                "route_bits": self.route_bits,
                "model_bits": self.hypothesis.description_bits,
                "residual_bits": self.residual_bits,
                "support_bits": self.support_bits,
                "total_bits": self.total_mdl_bits,
            },
            "demo_outputs": [grid_to_lists(item) if item is not None else None for item in self.demo_outputs],
            "query_outputs": [grid_to_lists(item) if item is not None else None for item in self.query_outputs],
            "residuals": [
                {
                    "pair_index": item.pair_index,
                    "exact": item.exact,
                    "shape_match": item.shape_match,
                    "mismatch_count": item.mismatch_count,
                    "agreement": item.agreement,
                    "invalid_code": item.invalid_code,
                }
                for item in self.residuals
            ],
        }


@dataclass(frozen=True, slots=True)
class RepairReceipt:
    parent_hypothesis_id: str
    strategy: str
    status: str
    reason: str | None
    candidate_ids: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "strategy": self.strategy,
            "status": self.status,
            "reason": self.reason,
            "candidate_ids": list(self.candidate_ids),
        }
