"""Immutable shared state for memory, goals, failures, and typed capability use.

The workspace is deliberately provider-agnostic.  It does not decide ARC rules
and it does not contain a learned router.  Its job is to make the information
boundary explicit: which state is shared, which memories may be retrieved, what
one capability is allowed to consume, and whether an execution really changed
the candidate frontier.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .hybrid.metareasoning import NativeBudgetLedger, NativeCostVector
from .hybrid.types import canonical_json


COGNITIVE_WORKSPACE_SCHEMA = "afts.cognitive-workspace/v1"
GOAL_SCHEMA = "afts.cognitive-goal/v1"
MEMORY_SCHEMA = "afts.cognitive-memory/v1"
FAILURE_SIGNAL_SCHEMA = "afts.cognitive-failure-signal/v1"
CAPABILITY_OPTION_SCHEMA = "afts.cognitive-capability-option/v1"
EPISODE_SCHEMA = "afts.cognitive-episode/v1"
CAPABILITY_PLAN_SCHEMA = "afts.cognitive-capability-plan/v1"

GOAL_STATUSES = ("active", "satisfied", "failed")
MEMORY_SCOPES = (
    "task_local_demo",
    "family_disjoint_validated",
    "outcome_exposed_diagnostic",
    "sealed_query_oracle",
)
RETRIEVAL_LANES = ("diagnostic", "prospective")


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _ordered_unique_strings(value: Sequence[object], *, field: str) -> tuple[str, ...]:
    normalized = tuple(value)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise TypeError(f"{field} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


def _canonical_object(value: Mapping[str, object], *, field: str) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return canonical_json(dict(value))


def _content_id(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def _strict_mapping(
    payload: object, *, expected: set[str], field: str
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{field} fields differ")
    return payload


@dataclass(frozen=True, slots=True)
class GoalFrame:
    goal_id: str
    objective: str
    success_condition: str
    parent_goal_id: str | None
    status: str

    @classmethod
    def create(
        cls,
        *,
        objective: str,
        success_condition: str,
        parent_goal_id: str | None = None,
        status: str = "active",
    ) -> "GoalFrame":
        objective = _nonempty(objective, field="goal objective")
        success_condition = _nonempty(
            success_condition, field="goal success condition"
        )
        if parent_goal_id is not None:
            parent_goal_id = _nonempty(parent_goal_id, field="parent goal ID")
        if status not in GOAL_STATUSES:
            raise ValueError("unknown goal status")
        payload = {
            "schema": GOAL_SCHEMA,
            "objective": objective,
            "success_condition": success_condition,
            "parent_goal_id": parent_goal_id,
            "status": status,
        }
        return cls(_content_id(payload), objective, success_condition, parent_goal_id, status)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": GOAL_SCHEMA,
            "goal_id": self.goal_id,
            "objective": self.objective,
            "success_condition": self.success_condition,
            "parent_goal_id": self.parent_goal_id,
            "status": self.status,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "GoalFrame":
        value = _strict_mapping(
            payload,
            expected={
                "schema",
                "goal_id",
                "objective",
                "success_condition",
                "parent_goal_id",
                "status",
            },
            field="goal",
        )
        if value["schema"] != GOAL_SCHEMA:
            raise ValueError("unsupported goal schema")
        rebuilt = cls.create(
            objective=value["objective"],
            success_condition=value["success_condition"],
            parent_goal_id=value["parent_goal_id"],
            status=value["status"],
        )
        if value["goal_id"] != rebuilt.goal_id:
            raise ValueError("goal ID differs from canonical content")
        return rebuilt


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    record_type: str
    scope: str
    payload_json: str
    source_task_id: str
    source_family_id: str | None
    validation_evidence_ids: tuple[str, ...]
    outcome_exposed: bool
    contains_query_oracle: bool

    @classmethod
    def create(
        cls,
        *,
        record_type: str,
        scope: str,
        payload: Mapping[str, object],
        source_task_id: str,
        source_family_id: str | None,
        validation_evidence_ids: Sequence[str] = (),
        outcome_exposed: bool = False,
        contains_query_oracle: bool = False,
    ) -> "MemoryRecord":
        record_type = _nonempty(record_type, field="memory record type")
        source_task_id = _nonempty(source_task_id, field="memory source task ID")
        if scope not in MEMORY_SCOPES:
            raise ValueError("unknown memory scope")
        if source_family_id is not None:
            source_family_id = _nonempty(
                source_family_id, field="memory source family ID"
            )
        evidence = _ordered_unique_strings(
            validation_evidence_ids, field="memory validation evidence"
        )
        if type(outcome_exposed) is not bool or type(contains_query_oracle) is not bool:
            raise TypeError("memory exposure flags must be boolean")
        if scope == "family_disjoint_validated":
            if source_family_id is None or not evidence:
                raise ValueError(
                    "family-disjoint memory requires a family and validation evidence"
                )
            if outcome_exposed or contains_query_oracle:
                raise ValueError("validated reusable memory cannot contain outcome data")
        elif scope == "outcome_exposed_diagnostic" and not outcome_exposed:
            raise ValueError("diagnostic memory must declare outcome exposure")
        elif scope == "sealed_query_oracle" and not contains_query_oracle:
            raise ValueError("sealed oracle memory must declare oracle content")
        elif scope == "task_local_demo" and (outcome_exposed or contains_query_oracle):
            raise ValueError("task-local demonstration memory cannot contain query outcomes")
        payload_json = _canonical_object(payload, field="memory payload")
        identity = {
            "schema": MEMORY_SCHEMA,
            "record_type": record_type,
            "scope": scope,
            "payload": json.loads(payload_json),
            "source_task_id": source_task_id,
            "source_family_id": source_family_id,
            "validation_evidence_ids": list(evidence),
            "outcome_exposed": outcome_exposed,
            "contains_query_oracle": contains_query_oracle,
        }
        return cls(
            _content_id(identity),
            record_type,
            scope,
            payload_json,
            source_task_id,
            source_family_id,
            evidence,
            outcome_exposed,
            contains_query_oracle,
        )

    @property
    def payload(self) -> dict[str, object]:
        return json.loads(self.payload_json)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": MEMORY_SCHEMA,
            "memory_id": self.memory_id,
            "record_type": self.record_type,
            "scope": self.scope,
            "payload": self.payload,
            "source_task_id": self.source_task_id,
            "source_family_id": self.source_family_id,
            "validation_evidence_ids": list(self.validation_evidence_ids),
            "outcome_exposed": self.outcome_exposed,
            "contains_query_oracle": self.contains_query_oracle,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "MemoryRecord":
        value = _strict_mapping(
            payload,
            expected={
                "schema",
                "memory_id",
                "record_type",
                "scope",
                "payload",
                "source_task_id",
                "source_family_id",
                "validation_evidence_ids",
                "outcome_exposed",
                "contains_query_oracle",
            },
            field="memory",
        )
        if value["schema"] != MEMORY_SCHEMA:
            raise ValueError("unsupported memory schema")
        evidence = value["validation_evidence_ids"]
        if not isinstance(evidence, list):
            raise TypeError("memory validation evidence must be an array")
        rebuilt = cls.create(
            record_type=value["record_type"],
            scope=value["scope"],
            payload=value["payload"],
            source_task_id=value["source_task_id"],
            source_family_id=value["source_family_id"],
            validation_evidence_ids=evidence,
            outcome_exposed=value["outcome_exposed"],
            contains_query_oracle=value["contains_query_oracle"],
        )
        if value["memory_id"] != rebuilt.memory_id:
            raise ValueError("memory ID differs from canonical content")
        return rebuilt


def retrievable_memories(
    records: Sequence[MemoryRecord],
    *,
    target_task_id: str,
    target_family_id: str | None,
    lane: str,
) -> tuple[MemoryRecord, ...]:
    """Return memories legal for one task without implicit degradation."""

    target_task_id = _nonempty(target_task_id, field="target task ID")
    if target_family_id is not None:
        target_family_id = _nonempty(target_family_id, field="target family ID")
    if lane not in RETRIEVAL_LANES:
        raise ValueError("unknown memory retrieval lane")
    selected = []
    for record in records:
        if not isinstance(record, MemoryRecord):
            raise TypeError("memory bank must contain MemoryRecord values")
        if record.scope == "sealed_query_oracle":
            continue
        if record.scope == "task_local_demo":
            if record.source_task_id == target_task_id:
                selected.append(record)
            continue
        if record.scope == "outcome_exposed_diagnostic":
            if lane == "diagnostic" and record.source_task_id == target_task_id:
                selected.append(record)
            continue
        if target_family_id is None:
            raise ValueError(
                "family ID is required to retrieve family-disjoint validated memory"
            )
        if record.source_family_id != target_family_id:
            selected.append(record)
    return tuple(sorted(selected, key=lambda record: record.memory_id))


@dataclass(frozen=True, slots=True)
class FailureSignal:
    certificate_id: str
    certificate_type: str
    current_representation: str
    target_representation: str
    affected_slots: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        certificate_type: str,
        current_representation: str,
        target_representation: str,
        affected_slots: Sequence[str],
        evidence_ids: Sequence[str],
    ) -> "FailureSignal":
        certificate_type = _nonempty(
            certificate_type, field="failure certificate type"
        )
        current_representation = _nonempty(
            current_representation, field="current representation"
        )
        target_representation = _nonempty(
            target_representation, field="target representation"
        )
        slots = tuple(sorted(_ordered_unique_strings(affected_slots, field="affected slots")))
        evidence = tuple(
            sorted(_ordered_unique_strings(evidence_ids, field="failure evidence IDs"))
        )
        if not slots or not evidence:
            raise ValueError("failure signal requires slots and evidence")
        payload = {
            "schema": FAILURE_SIGNAL_SCHEMA,
            "certificate_type": certificate_type,
            "current_representation": current_representation,
            "target_representation": target_representation,
            "affected_slots": list(slots),
            "evidence_ids": list(evidence),
        }
        return cls(
            _content_id(payload),
            certificate_type,
            current_representation,
            target_representation,
            slots,
            evidence,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": FAILURE_SIGNAL_SCHEMA,
            "certificate_id": self.certificate_id,
            "certificate_type": self.certificate_type,
            "current_representation": self.current_representation,
            "target_representation": self.target_representation,
            "affected_slots": list(self.affected_slots),
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "FailureSignal":
        value = _strict_mapping(
            payload,
            expected={
                "schema",
                "certificate_id",
                "certificate_type",
                "current_representation",
                "target_representation",
                "affected_slots",
                "evidence_ids",
            },
            field="failure signal",
        )
        if value["schema"] != FAILURE_SIGNAL_SCHEMA:
            raise ValueError("unsupported failure signal schema")
        slots = value["affected_slots"]
        evidence = value["evidence_ids"]
        if not isinstance(slots, list) or not isinstance(evidence, list):
            raise TypeError("failure slots and evidence must be arrays")
        rebuilt = cls.create(
            certificate_type=value["certificate_type"],
            current_representation=value["current_representation"],
            target_representation=value["target_representation"],
            affected_slots=slots,
            evidence_ids=evidence,
        )
        if value["certificate_id"] != rebuilt.certificate_id:
            raise ValueError("failure signal ID differs from canonical content")
        return rebuilt


@dataclass(frozen=True, slots=True)
class CapabilityOption:
    option_id: str
    actor: str
    operator: str
    input_representation: str
    output_representation: str
    accepted_certificate_types: tuple[str, ...]
    reservation: NativeCostVector
    expected_frontier_gain: int
    expected_hypothesis_reduction: int

    @classmethod
    def create(
        cls,
        *,
        actor: str,
        operator: str,
        input_representation: str,
        output_representation: str,
        accepted_certificate_types: Sequence[str],
        reservation: NativeCostVector,
        expected_frontier_gain: int,
        expected_hypothesis_reduction: int,
    ) -> "CapabilityOption":
        actor = _nonempty(actor, field="capability actor")
        operator = _nonempty(operator, field="capability operator")
        input_representation = _nonempty(
            input_representation, field="capability input representation"
        )
        output_representation = _nonempty(
            output_representation, field="capability output representation"
        )
        accepted = tuple(
            sorted(
                _ordered_unique_strings(
                    accepted_certificate_types,
                    field="accepted certificate types",
                )
            )
        )
        if not accepted:
            raise ValueError("capability option must accept at least one certificate")
        if not isinstance(reservation, NativeCostVector) or reservation.is_zero:
            raise ValueError("capability option requires a non-zero native reservation")
        for name, value in (
            ("expected_frontier_gain", expected_frontier_gain),
            ("expected_hypothesis_reduction", expected_hypothesis_reduction),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        payload = {
            "schema": CAPABILITY_OPTION_SCHEMA,
            "actor": actor,
            "operator": operator,
            "input_representation": input_representation,
            "output_representation": output_representation,
            "accepted_certificate_types": list(accepted),
            "reservation": reservation.to_json_dict(),
            "expected_frontier_gain": expected_frontier_gain,
            "expected_hypothesis_reduction": expected_hypothesis_reduction,
        }
        return cls(
            _content_id(payload),
            actor,
            operator,
            input_representation,
            output_representation,
            accepted,
            reservation,
            expected_frontier_gain,
            expected_hypothesis_reduction,
        )

    def accepts(self, failure: FailureSignal) -> bool:
        return (
            failure.certificate_type in self.accepted_certificate_types
            and failure.current_representation == self.input_representation
            and failure.target_representation == self.output_representation
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": CAPABILITY_OPTION_SCHEMA,
            "option_id": self.option_id,
            "actor": self.actor,
            "operator": self.operator,
            "input_representation": self.input_representation,
            "output_representation": self.output_representation,
            "accepted_certificate_types": list(self.accepted_certificate_types),
            "reservation": self.reservation.to_json_dict(),
            "expected_frontier_gain": self.expected_frontier_gain,
            "expected_hypothesis_reduction": self.expected_hypothesis_reduction,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "CapabilityOption":
        value = _strict_mapping(
            payload,
            expected={
                "schema",
                "option_id",
                "actor",
                "operator",
                "input_representation",
                "output_representation",
                "accepted_certificate_types",
                "reservation",
                "expected_frontier_gain",
                "expected_hypothesis_reduction",
            },
            field="capability option",
        )
        if value["schema"] != CAPABILITY_OPTION_SCHEMA:
            raise ValueError("unsupported capability option schema")
        accepted = value["accepted_certificate_types"]
        if not isinstance(accepted, list):
            raise TypeError("accepted certificate types must be an array")
        rebuilt = cls.create(
            actor=value["actor"],
            operator=value["operator"],
            input_representation=value["input_representation"],
            output_representation=value["output_representation"],
            accepted_certificate_types=accepted,
            reservation=NativeCostVector.from_json_dict(value["reservation"]),
            expected_frontier_gain=value["expected_frontier_gain"],
            expected_hypothesis_reduction=value["expected_hypothesis_reduction"],
        )
        if value["option_id"] != rebuilt.option_id:
            raise ValueError("capability option ID differs from canonical content")
        return rebuilt


@dataclass(frozen=True, slots=True)
class CapabilityPlan:
    plan_id: str
    certificate_id: str
    goal_representation: str
    option_ids: tuple[str, ...]
    reserved_cost: NativeCostVector
    predicted_frontier_gain: int
    predicted_hypothesis_reduction: int

    @classmethod
    def create(
        cls,
        *,
        certificate_id: str,
        goal_representation: str,
        option_ids: Sequence[str],
        reserved_cost: NativeCostVector,
        predicted_frontier_gain: int,
        predicted_hypothesis_reduction: int,
    ) -> "CapabilityPlan":
        certificate_id = _nonempty(certificate_id, field="plan certificate ID")
        goal_representation = _nonempty(
            goal_representation, field="plan goal representation"
        )
        options = _ordered_unique_strings(option_ids, field="plan option IDs")
        if not options:
            raise ValueError("capability plan requires at least one option")
        if not isinstance(reserved_cost, NativeCostVector) or reserved_cost.is_zero:
            raise ValueError("capability plan requires a non-zero native reservation")
        for name, value in (
            ("predicted_frontier_gain", predicted_frontier_gain),
            ("predicted_hypothesis_reduction", predicted_hypothesis_reduction),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        payload = {
            "schema": CAPABILITY_PLAN_SCHEMA,
            "certificate_id": certificate_id,
            "goal_representation": goal_representation,
            "option_ids": list(options),
            "reserved_cost": reserved_cost.to_json_dict(),
            "predicted_frontier_gain": predicted_frontier_gain,
            "predicted_hypothesis_reduction": predicted_hypothesis_reduction,
        }
        return cls(
            _content_id(payload),
            certificate_id,
            goal_representation,
            options,
            reserved_cost,
            predicted_frontier_gain,
            predicted_hypothesis_reduction,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": CAPABILITY_PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "certificate_id": self.certificate_id,
            "goal_representation": self.goal_representation,
            "option_ids": list(self.option_ids),
            "reserved_cost": self.reserved_cost.to_json_dict(),
            "predicted_frontier_gain": self.predicted_frontier_gain,
            "predicted_hypothesis_reduction": self.predicted_hypothesis_reduction,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "CapabilityPlan":
        value = _strict_mapping(
            payload,
            expected={
                "schema",
                "plan_id",
                "certificate_id",
                "goal_representation",
                "option_ids",
                "reserved_cost",
                "predicted_frontier_gain",
                "predicted_hypothesis_reduction",
            },
            field="capability plan",
        )
        if value["schema"] != CAPABILITY_PLAN_SCHEMA:
            raise ValueError("unsupported capability plan schema")
        option_ids = value["option_ids"]
        if not isinstance(option_ids, list):
            raise TypeError("plan option IDs must be an array")
        rebuilt = cls.create(
            certificate_id=value["certificate_id"],
            goal_representation=value["goal_representation"],
            option_ids=option_ids,
            reserved_cost=NativeCostVector.from_json_dict(value["reserved_cost"]),
            predicted_frontier_gain=value["predicted_frontier_gain"],
            predicted_hypothesis_reduction=value[
                "predicted_hypothesis_reduction"
            ],
        )
        if value["plan_id"] != rebuilt.plan_id:
            raise ValueError("capability plan ID differs from canonical content")
        return rebuilt


@dataclass(frozen=True, slots=True)
class WorkspaceEpisode:
    episode_id: str
    certificate_id: str
    option_id: str
    produced_candidate_ids: tuple[str, ...]
    novel_candidate_ids: tuple[str, ...]
    frontier_changed: bool
    success: bool
    charged_cost: NativeCostVector

    @classmethod
    def create(
        cls,
        *,
        certificate_id: str,
        option_id: str,
        produced_candidate_ids: Sequence[str],
        novel_candidate_ids: Sequence[str],
        success: bool,
        charged_cost: NativeCostVector,
    ) -> "WorkspaceEpisode":
        certificate_id = _nonempty(certificate_id, field="episode certificate ID")
        option_id = _nonempty(option_id, field="episode option ID")
        produced = tuple(
            sorted(_ordered_unique_strings(produced_candidate_ids, field="produced candidates"))
        )
        novel = tuple(
            sorted(_ordered_unique_strings(novel_candidate_ids, field="novel candidates"))
        )
        if not set(novel).issubset(produced):
            raise ValueError("novel candidates must be a subset of produced candidates")
        if type(success) is not bool:
            raise TypeError("episode success must be boolean")
        if not isinstance(charged_cost, NativeCostVector):
            raise TypeError("episode charged cost must be a NativeCostVector")
        payload = {
            "schema": EPISODE_SCHEMA,
            "certificate_id": certificate_id,
            "option_id": option_id,
            "produced_candidate_ids": list(produced),
            "novel_candidate_ids": list(novel),
            "frontier_changed": bool(novel),
            "success": success,
            "charged_cost": charged_cost.to_json_dict(),
        }
        return cls(
            _content_id(payload),
            certificate_id,
            option_id,
            produced,
            novel,
            bool(novel),
            success,
            charged_cost,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": EPISODE_SCHEMA,
            "episode_id": self.episode_id,
            "certificate_id": self.certificate_id,
            "option_id": self.option_id,
            "produced_candidate_ids": list(self.produced_candidate_ids),
            "novel_candidate_ids": list(self.novel_candidate_ids),
            "frontier_changed": self.frontier_changed,
            "success": self.success,
            "charged_cost": self.charged_cost.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "WorkspaceEpisode":
        value = _strict_mapping(
            payload,
            expected={
                "schema",
                "episode_id",
                "certificate_id",
                "option_id",
                "produced_candidate_ids",
                "novel_candidate_ids",
                "frontier_changed",
                "success",
                "charged_cost",
            },
            field="workspace episode",
        )
        if value["schema"] != EPISODE_SCHEMA:
            raise ValueError("unsupported workspace episode schema")
        produced = value["produced_candidate_ids"]
        novel = value["novel_candidate_ids"]
        if not isinstance(produced, list) or not isinstance(novel, list):
            raise TypeError("episode candidate IDs must be arrays")
        rebuilt = cls.create(
            certificate_id=value["certificate_id"],
            option_id=value["option_id"],
            produced_candidate_ids=produced,
            novel_candidate_ids=novel,
            success=value["success"],
            charged_cost=NativeCostVector.from_json_dict(value["charged_cost"]),
        )
        if value["episode_id"] != rebuilt.episode_id:
            raise ValueError("workspace episode ID differs from canonical content")
        if value["frontier_changed"] != rebuilt.frontier_changed:
            raise ValueError("workspace episode frontier flag differs")
        return rebuilt


@dataclass(frozen=True, slots=True)
class CognitiveWorkspaceState:
    state_id: str
    task_id: str
    family_id: str | None
    goals: tuple[GoalFrame, ...]
    memories: tuple[MemoryRecord, ...]
    failures: tuple[FailureSignal, ...]
    options: tuple[CapabilityOption, ...]
    episodes: tuple[WorkspaceEpisode, ...]
    budget: NativeBudgetLedger

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        family_id: str | None,
        goals: Sequence[GoalFrame],
        memories: Sequence[MemoryRecord],
        failures: Sequence[FailureSignal],
        options: Sequence[CapabilityOption],
        episodes: Sequence[WorkspaceEpisode],
        budget: NativeBudgetLedger,
    ) -> "CognitiveWorkspaceState":
        task_id = _nonempty(task_id, field="workspace task ID")
        if family_id is not None:
            family_id = _nonempty(family_id, field="workspace family ID")
        normalized = (
            tuple(goals),
            tuple(memories),
            tuple(failures),
            tuple(options),
            tuple(episodes),
        )
        expected_types = (
            GoalFrame,
            MemoryRecord,
            FailureSignal,
            CapabilityOption,
            WorkspaceEpisode,
        )
        for values, expected in zip(normalized, expected_types, strict=True):
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"workspace values must contain {expected.__name__}")
        for values, attribute in zip(
            normalized[1:],
            ("memory_id", "certificate_id", "option_id", "episode_id"),
            strict=True,
        ):
            identifiers = tuple(getattr(item, attribute) for item in values)
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("workspace content IDs must be unique within each store")
        if not isinstance(budget, NativeBudgetLedger):
            raise TypeError("workspace budget must be a NativeBudgetLedger")
        content = {
            "schema": COGNITIVE_WORKSPACE_SCHEMA,
            "task_id": task_id,
            "family_id": family_id,
            "goals": [item.to_json_dict() for item in normalized[0]],
            "memories": [item.to_json_dict() for item in normalized[1]],
            "failures": [item.to_json_dict() for item in normalized[2]],
            "options": [item.to_json_dict() for item in normalized[3]],
            "episodes": [item.to_json_dict() for item in normalized[4]],
            "budget": budget.to_json_dict(),
        }
        return cls(
            _content_id(content),
            task_id,
            family_id,
            normalized[0],
            normalized[1],
            normalized[2],
            normalized[3],
            normalized[4],
            budget,
        )

    @classmethod
    def empty(
        cls,
        *,
        task_id: str,
        family_id: str | None,
        budget_limit: NativeCostVector,
    ) -> "CognitiveWorkspaceState":
        return cls.create(
            task_id=task_id,
            family_id=family_id,
            goals=(),
            memories=(),
            failures=(),
            options=(),
            episodes=(),
            budget=NativeBudgetLedger(budget_limit),
        )

    @classmethod
    def from_json_dict(cls, payload: object) -> "CognitiveWorkspaceState":
        value = _strict_mapping(
            payload,
            expected={
                "schema",
                "state_id",
                "task_id",
                "family_id",
                "goals",
                "memories",
                "failures",
                "options",
                "episodes",
                "budget",
            },
            field="cognitive workspace",
        )
        if value["schema"] != COGNITIVE_WORKSPACE_SCHEMA:
            raise ValueError("unsupported cognitive workspace schema")
        arrays = {
            name: value[name]
            for name in ("goals", "memories", "failures", "options", "episodes")
        }
        if any(not isinstance(items, list) for items in arrays.values()):
            raise TypeError("workspace stores must be arrays")
        budget = value["budget"]
        if not isinstance(budget, Mapping) or set(budget) != {
            "accounting",
            "limit",
            "used",
        }:
            raise ValueError("workspace budget fields differ")
        if budget["accounting"] != "provider_native_vector":
            raise ValueError("workspace budget accounting differs")
        rebuilt = cls.create(
            task_id=value["task_id"],
            family_id=value["family_id"],
            goals=tuple(GoalFrame.from_json_dict(item) for item in arrays["goals"]),
            memories=tuple(
                MemoryRecord.from_json_dict(item) for item in arrays["memories"]
            ),
            failures=tuple(
                FailureSignal.from_json_dict(item) for item in arrays["failures"]
            ),
            options=tuple(
                CapabilityOption.from_json_dict(item) for item in arrays["options"]
            ),
            episodes=tuple(
                WorkspaceEpisode.from_json_dict(item) for item in arrays["episodes"]
            ),
            budget=NativeBudgetLedger(
                NativeCostVector.from_json_dict(budget["limit"]),
                NativeCostVector.from_json_dict(budget["used"]),
            ),
        )
        if value["state_id"] != rebuilt.state_id:
            raise ValueError("workspace state ID differs from canonical content")
        return rebuilt

    def _replace(
        self,
        *,
        goals: Sequence[GoalFrame] | None = None,
        memories: Sequence[MemoryRecord] | None = None,
        failures: Sequence[FailureSignal] | None = None,
        options: Sequence[CapabilityOption] | None = None,
        episodes: Sequence[WorkspaceEpisode] | None = None,
        budget: NativeBudgetLedger | None = None,
    ) -> "CognitiveWorkspaceState":
        return CognitiveWorkspaceState.create(
            task_id=self.task_id,
            family_id=self.family_id,
            goals=self.goals if goals is None else goals,
            memories=self.memories if memories is None else memories,
            failures=self.failures if failures is None else failures,
            options=self.options if options is None else options,
            episodes=self.episodes if episodes is None else episodes,
            budget=self.budget if budget is None else budget,
        )

    def push_goal(self, goal: GoalFrame) -> "CognitiveWorkspaceState":
        if not isinstance(goal, GoalFrame):
            raise TypeError("workspace goal must be a GoalFrame")
        expected_parent = self.goals[-1].goal_id if self.goals else None
        if goal.parent_goal_id != expected_parent or goal.status != "active":
            raise ValueError("new active goal must point to the current goal")
        return self._replace(goals=(*self.goals, goal))

    def remember(self, record: MemoryRecord) -> "CognitiveWorkspaceState":
        if not isinstance(record, MemoryRecord):
            raise TypeError("workspace memory must be a MemoryRecord")
        if record.memory_id in {item.memory_id for item in self.memories}:
            return self
        return self._replace(memories=(*self.memories, record))

    def broadcast_failure(self, failure: FailureSignal) -> "CognitiveWorkspaceState":
        if not isinstance(failure, FailureSignal):
            raise TypeError("workspace failure must be a FailureSignal")
        if failure.certificate_id in {item.certificate_id for item in self.failures}:
            return self
        return self._replace(failures=(*self.failures, failure))

    def register_option(self, option: CapabilityOption) -> "CognitiveWorkspaceState":
        if not isinstance(option, CapabilityOption):
            raise TypeError("workspace option must be a CapabilityOption")
        if option.option_id in {item.option_id for item in self.options}:
            return self
        return self._replace(options=(*self.options, option))

    def select_option(self, certificate_id: str) -> CapabilityOption:
        certificate_id = _nonempty(certificate_id, field="selected certificate ID")
        failures = {
            failure.certificate_id: failure for failure in self.failures
        }
        if certificate_id not in failures:
            raise ValueError("selected certificate is not in the workspace")
        failure = failures[certificate_id]
        applicable = tuple(
            option
            for option in self.options
            if option.accepts(failure) and self.budget.can_reserve(option.reservation)
        )
        if not applicable:
            raise ValueError("no legal budget-feasible option for the failure signal")
        # Provider-native vectors are intentionally not collapsed into an
        # undeclared scalar.  Budget filters feasibility; evidence-derived gain
        # ranks the remaining legal options deterministically.
        return min(
            applicable,
            key=lambda option: (
                -option.expected_frontier_gain,
                -option.expected_hypothesis_reduction,
                option.option_id,
            ),
        )

    def apply_option_result(
        self,
        *,
        certificate_id: str,
        option_id: str,
        produced_candidate_ids: Sequence[str],
        novel_candidate_ids: Sequence[str],
        success: bool,
    ) -> "CognitiveWorkspaceState":
        selected = self.select_option(certificate_id)
        if selected.option_id != option_id:
            raise ValueError("result option differs from deterministic selected option")
        episode = WorkspaceEpisode.create(
            certificate_id=certificate_id,
            option_id=option_id,
            produced_candidate_ids=produced_candidate_ids,
            novel_candidate_ids=novel_candidate_ids,
            success=success,
            charged_cost=selected.reservation,
        )
        return self._replace(
            episodes=(*self.episodes, episode),
            budget=self.budget.charge(selected.reservation),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": COGNITIVE_WORKSPACE_SCHEMA,
            "state_id": self.state_id,
            "task_id": self.task_id,
            "family_id": self.family_id,
            "goals": [item.to_json_dict() for item in self.goals],
            "memories": [item.to_json_dict() for item in self.memories],
            "failures": [item.to_json_dict() for item in self.failures],
            "options": [item.to_json_dict() for item in self.options],
            "episodes": [item.to_json_dict() for item in self.episodes],
            "budget": self.budget.to_json_dict(),
        }


def plan_capability_path(
    state: CognitiveWorkspaceState,
    *,
    certificate_id: str,
    goal_representation: str,
    max_depth: int = 4,
) -> CapabilityPlan:
    """Find a shortest legal representation path under the native budget.

    This is an auditable planning baseline, not a learned controller.  Native
    cost vectors determine feasibility only and are never collapsed into an
    implicit scalar.  Among equal-depth feasible paths, declared frontier and
    hypothesis effects provide deterministic ordering.
    """

    if not isinstance(state, CognitiveWorkspaceState):
        raise TypeError("capability planning requires a cognitive workspace")
    certificate_id = _nonempty(certificate_id, field="plan certificate ID")
    goal_representation = _nonempty(
        goal_representation, field="plan goal representation"
    )
    if type(max_depth) is not int or max_depth < 1:
        raise ValueError("plan max depth must be a positive integer")
    failures = {item.certificate_id: item for item in state.failures}
    if certificate_id not in failures:
        raise ValueError("plan certificate is not in the workspace")
    failure = failures[certificate_id]
    if failure.current_representation == goal_representation:
        raise ValueError("planning goal is already the current representation")

    # representation, options, reservation, gain, reduction, visited states
    frontier: tuple[
        tuple[
            str,
            tuple[CapabilityOption, ...],
            NativeCostVector,
            int,
            int,
            tuple[str, ...],
        ],
        ...,
    ] = (
        (
            failure.current_representation,
            (),
            NativeCostVector(),
            0,
            0,
            (failure.current_representation,),
        ),
    )
    for _ in range(max_depth):
        next_frontier = []
        completed = []
        for representation, path, reservation, gain, reduction, visited in frontier:
            used_option_ids = {item.option_id for item in path}
            for option in sorted(state.options, key=lambda item: item.option_id):
                if option.option_id in used_option_ids:
                    continue
                if not path:
                    if not option.accepts(failure):
                        continue
                elif option.input_representation != representation:
                    continue
                if option.output_representation in visited:
                    continue
                next_cost = reservation + option.reservation
                if not state.budget.can_reserve(next_cost):
                    continue
                candidate = (
                    option.output_representation,
                    (*path, option),
                    next_cost,
                    gain + option.expected_frontier_gain,
                    reduction + option.expected_hypothesis_reduction,
                    (*visited, option.output_representation),
                )
                if option.output_representation == goal_representation:
                    completed.append(candidate)
                else:
                    next_frontier.append(candidate)
        if completed:
            selected = min(
                completed,
                key=lambda item: (
                    -item[3],
                    -item[4],
                    tuple(option.option_id for option in item[1]),
                ),
            )
            return CapabilityPlan.create(
                certificate_id=certificate_id,
                goal_representation=goal_representation,
                option_ids=tuple(option.option_id for option in selected[1]),
                reserved_cost=selected[2],
                predicted_frontier_gain=selected[3],
                predicted_hypothesis_reduction=selected[4],
            )
        frontier = tuple(next_frontier)
        if not frontier:
            break
    raise ValueError("no legal budget-feasible capability path reaches the goal")
