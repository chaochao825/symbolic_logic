"""Content-addressed visual-state sketches and structured control policies.

The sketch is a compact, oracle-free cache of task facts and observed execution
state.  The policy separates factual grounding from adaptive representation
diversity, mirroring the distinction between summary consistency and reasoning
variety without requiring neural training.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .types import canonical_json

if TYPE_CHECKING:
    from .control import Blackboard, ControlAction


DELIBERATION_SCHEMA_VERSION = "afts.structured-deliberation/v1"


def _content_id(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class DeliberationSketch:
    sketch_id: str
    state_id: str
    phase: str
    factual_tokens: tuple[str, ...]
    residual_tokens: tuple[str, ...]
    provider_calls: tuple[tuple[str, int], ...]
    accepted_candidate_count: int
    semantic_cluster_count: int
    demo_exact_verified_count: int

    def __post_init__(self) -> None:
        if self.phase not in {"scope", "explore", "refine", "answer"}:
            raise ValueError("unknown deliberation phase")
        if not self.state_id:
            raise ValueError("deliberation sketch requires a state ID")
        if tuple(sorted(set(self.factual_tokens))) != self.factual_tokens:
            raise ValueError("factual tokens must be unique and sorted")
        if tuple(sorted(set(self.residual_tokens))) != self.residual_tokens:
            raise ValueError("residual tokens must be unique and sorted")
        if tuple(sorted(self.provider_calls)) != self.provider_calls:
            raise ValueError("provider calls must be canonically ordered")
        for name in (
            "accepted_candidate_count",
            "semantic_cluster_count",
            "demo_exact_verified_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.demo_exact_verified_count > self.accepted_candidate_count:
            raise ValueError("verified count cannot exceed accepted candidates")
        expected = _content_id(self._payload())
        if self.sketch_id != expected:
            raise ValueError("sketch_id does not match canonical sketch content")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": DELIBERATION_SCHEMA_VERSION,
            "state_id": self.state_id,
            "phase": self.phase,
            "factual_tokens": list(self.factual_tokens),
            "residual_tokens": list(self.residual_tokens),
            "provider_calls": [list(item) for item in self.provider_calls],
            "accepted_candidate_count": self.accepted_candidate_count,
            "semantic_cluster_count": self.semantic_cluster_count,
            "demo_exact_verified_count": self.demo_exact_verified_count,
        }

    @classmethod
    def from_blackboard(cls, blackboard: "Blackboard") -> "DeliberationSketch":
        features = blackboard.features
        changed_regime = (
            "change:sparse"
            if features.mean_changed_fraction <= 0.2
            else "change:local"
            if features.mean_changed_fraction <= 0.6
            else "change:dense"
        )
        support_regime = (
            "support:high"
            if features.center_transition_support >= 0.8
            else "support:medium"
            if features.center_transition_support >= 0.5
            else "support:low"
        )
        factual = {
            "shape:change" if features.shape_change else "shape:stable",
            changed_regime,
            support_regime,
        }
        for token, enabled in (
            ("transform:d4", features.d4_consistent),
            ("evidence:object_relation", features.object_relation_evidence),
            ("transition:compressible_local", features.compressible_local_transition),
            ("hypothesis:open", features.open_hypothesis_needed),
        ):
            if enabled:
                factual.add(token)

        residual: set[str] = set()
        for signal in blackboard.residual_signals:
            for token, enabled in (
                ("residual:execution_failure", signal.execution_failure_count > 0),
                ("residual:shape_mismatch", signal.shape_mismatch_count > 0),
                ("residual:localized", signal.localized),
                ("residual:global_color", signal.unambiguous_global_color_map),
                ("residual:low_query_support", signal.low_query_support),
                ("residual:exact_unverified", signal.exact_but_unverified),
            ):
                if enabled:
                    residual.add(token)

        calls: dict[str, int] = {}
        accepted_ids: set[str] = set()
        for result in blackboard.action_results:
            if result.action.kind == "propose":
                calls[result.action.actor] = calls.get(result.action.actor, 0) + 1
            accepted_ids.update(result.accepted_candidate_ids)

        semantic_signatures = {
            (
                evaluation.demo_outputs,
                evaluation.query_outputs,
                evaluation.hard_verified,
                evaluation.rejection_reason,
            )
            for evaluation in blackboard.evaluations
        }
        exact_verified = sum(
            evaluation.demo_exact and evaluation.hard_verified
            for evaluation in blackboard.evaluations
        )
        if not blackboard.action_results:
            phase = "scope"
        elif exact_verified >= 2:
            phase = "answer"
        elif residual:
            phase = "refine"
        else:
            phase = "explore"
        values = {
            "state_id": blackboard.state_id,
            "phase": phase,
            "factual_tokens": tuple(sorted(factual)),
            "residual_tokens": tuple(sorted(residual)),
            "provider_calls": tuple(sorted(calls.items())),
            "accepted_candidate_count": len(accepted_ids),
            "semantic_cluster_count": len(semantic_signatures),
            "demo_exact_verified_count": exact_verified,
        }
        payload = {
            "schema": DELIBERATION_SCHEMA_VERSION,
            **{
                key: [list(item) for item in value]
                if key == "provider_calls"
                else list(value)
                if key in {"factual_tokens", "residual_tokens"}
                else value
                for key, value in values.items()
            },
        }
        return cls(sketch_id=_content_id(payload), **values)

    @property
    def verified_fraction(self) -> float:
        return self.demo_exact_verified_count / max(1, self.accepted_candidate_count)

    def to_json_dict(self) -> dict[str, object]:
        return {"sketch_id": self.sketch_id, **self._payload()}


@dataclass(frozen=True, slots=True)
class StructuredDeliberationPolicy:
    """Typed phase controller with independently ablatable score terms."""

    name: str = "structured_deliberation_v1"
    use_grounding: bool = True
    use_adaptive_diversity: bool = True
    use_borderline_ucb: bool = True

    @staticmethod
    def _calls(blackboard: "Blackboard") -> dict[str, int]:
        calls: dict[str, int] = {}
        for result in blackboard.action_results:
            if result.action.kind == "propose":
                calls[result.action.actor] = calls.get(result.action.actor, 0) + 1
        return calls

    @staticmethod
    def _latest_semantic_gain(blackboard: "Blackboard") -> dict[str, int]:
        evaluations = {
            item.hypothesis.hypothesis_id: item for item in blackboard.evaluations
        }
        seen: set[tuple[object, ...]] = set()
        gain: dict[str, int] = {}
        for result in blackboard.action_results:
            count = 0
            for hypothesis_id in result.accepted_candidate_ids:
                evaluation = evaluations.get(hypothesis_id)
                if evaluation is None:
                    continue
                signature = (
                    evaluation.demo_outputs,
                    evaluation.query_outputs,
                    evaluation.hard_verified,
                    evaluation.rejection_reason,
                )
                if signature not in seen:
                    seen.add(signature)
                    count += 1
            if result.action.kind == "propose":
                gain[result.action.actor] = count
        return gain

    @staticmethod
    def _grounding_rank(
        blackboard: "Blackboard", action: "ControlAction"
    ) -> int:
        features = blackboard.features
        if action.kind == "repair":
            return 0
        if action.kind != "propose":
            return 100
        if action.actor == "scene_predicate_dsl":
            if features.object_relation_evidence and not features.d4_consistent:
                return 0
            if features.all_same_shape and not features.d4_consistent:
                return 1
            return 2
        if action.actor == "typed_dsl":
            if features.shape_change or features.d4_consistent:
                return 0
            if features.object_relation_evidence:
                return 1
            return 2
        if action.route == "sparse_ca":
            if features.all_same_shape and not features.d4_consistent:
                return 0
            return 3
        return 1 + blackboard.route_decision.priority_for(action.route or "")

    def select(
        self,
        blackboard: "Blackboard",
        actions: Sequence["ControlAction"],
    ) -> "ControlAction":
        available = tuple(actions)
        if not available:
            raise ValueError("policy requires at least one compiled action")
        non_stop = tuple(item for item in available if item.kind != "stop")
        if not non_stop:
            return min(available, key=lambda item: item.action_id)

        sketch = DeliberationSketch.from_blackboard(blackboard)
        calls = self._calls(blackboard)
        semantic_gain = self._latest_semantic_gain(blackboard)
        total_calls = sum(calls.values())
        ambiguity = 1.0 - sketch.verified_fraction

        def score(action: "ControlAction") -> tuple[object, ...]:
            grounding = (
                100 * self._grounding_rank(blackboard, action)
                if self.use_grounding
                else 0
            )
            phase_penalty = 0
            if sketch.phase == "scope" and action.kind != "propose":
                phase_penalty += 500
            if sketch.phase == "refine":
                if action.kind == "repair":
                    phase_penalty -= 160
                elif action.evidence_signal_ids:
                    phase_penalty -= 40

            diversity = 0
            if self.use_adaptive_diversity and action.kind == "propose":
                adaptive = round(140 * ambiguity)
                if calls.get(action.actor, 0) == 0:
                    diversity -= adaptive
                elif semantic_gain.get(action.actor, 0) == 0:
                    diversity += adaptive

            borderline = 0
            if self.use_borderline_ucb and action.kind == "propose":
                exploration = math.sqrt(
                    math.log(total_calls + 2) / (calls.get(action.actor, 0) + 1)
                )
                borderline -= round(80 * ambiguity * exploration)

            return (
                phase_penalty + grounding + diversity + borderline,
                action.priority,
                action.actor,
                action.operator,
                action.action_id,
            )

        return min(non_stop, key=score)

