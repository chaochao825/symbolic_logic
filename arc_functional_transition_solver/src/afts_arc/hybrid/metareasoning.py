"""Physical-cost accounting and information audits for metalevel ARC control.

Normalized controller units are useful for testing action-mask correctness, but
they are not a substitute for provider-native work.  This module deliberately
keeps native costs as a named vector: program expansions, scene-rule trials,
model calls, tokens, and wall time must not be silently collapsed into one
unvalidated scalar.
"""

from __future__ import annotations

import math
import hashlib
import json
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass


NATIVE_COST_SCHEMA_VERSION = "afts.native-cost/v1"
NATIVE_CONTRACT_SCHEMA_VERSION = "afts.native-cost-contract/v1"


def _flatten_costs(
    value: Mapping[str, object],
    *,
    prefix: str = "",
) -> tuple[tuple[str, float], ...]:
    flattened: list[tuple[str, float]] = []
    for raw_key, item in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise TypeError("native-cost keys must be non-empty strings")
        key = f"{prefix}.{raw_key}" if prefix else raw_key
        if isinstance(item, Mapping):
            flattened.extend(_flatten_costs(item, prefix=key))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            numeric = float(item)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError("native costs must be finite and non-negative")
            if numeric:
                flattened.append((key, numeric))
        else:
            raise TypeError("native costs must contain only numeric leaves")
    return tuple(flattened)


@dataclass(frozen=True, slots=True)
class NativeCostVector:
    """Canonical, sparse vector of provider-native physical work."""

    items: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        canonical: list[tuple[str, float]] = []
        seen: set[str] = set()
        for key, raw_value in self.items:
            if not isinstance(key, str) or not key:
                raise TypeError("native-cost keys must be non-empty strings")
            value = float(raw_value)
            if key in seen:
                raise ValueError("native-cost keys must be unique")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("native costs must be finite and non-negative")
            seen.add(key)
            if value:
                canonical.append((key, value))
        canonical.sort()
        object.__setattr__(self, "items", tuple(canonical))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
        *,
        prefix: str = "",
    ) -> "NativeCostVector":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("native cost must be a mapping")
        return cls(_flatten_costs(value, prefix=prefix))

    @classmethod
    def from_json_dict(cls, value: object) -> "NativeCostVector":
        if not isinstance(value, Mapping):
            raise TypeError("native-cost JSON must be a mapping")
        if value.get("schema") != NATIVE_COST_SCHEMA_VERSION:
            raise ValueError("unknown native-cost schema")
        raw_items = value.get("items")
        if not isinstance(raw_items, list):
            raise TypeError("native-cost JSON items must be a list")
        items: list[tuple[str, float]] = []
        for item in raw_items:
            if not isinstance(item, list) or len(item) != 2:
                raise TypeError("native-cost JSON entries must be [key, value]")
            items.append((item[0], item[1]))
        return cls(tuple(items))

    @property
    def is_zero(self) -> bool:
        return not self.items

    def to_mapping(self) -> dict[str, float]:
        return dict(self.items)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": NATIVE_COST_SCHEMA_VERSION,
            "items": [[key, value] for key, value in self.items],
        }

    def __add__(self, other: "NativeCostVector") -> "NativeCostVector":
        if not isinstance(other, NativeCostVector):
            return NotImplemented
        totals = self.to_mapping()
        for key, value in other.items:
            totals[key] = totals.get(key, 0.0) + value
        return NativeCostVector(tuple(totals.items()))

    def fits_within(self, limit: "NativeCostVector") -> bool:
        if not isinstance(limit, NativeCostVector):
            raise TypeError("native-cost limit must be a NativeCostVector")
        caps = limit.to_mapping()
        return all(value <= caps.get(key, 0.0) for key, value in self.items)

    def normalized_total(self, scales: Mapping[str, float]) -> float:
        """Return a declared scalarization; no implicit unit conversion exists."""

        total = 0.0
        for key, value in self.items:
            scale = float(scales[key])
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("native-cost scales must be finite and positive")
            total += value / scale
        return total


@dataclass(frozen=True, slots=True)
class NativeBudgetLedger:
    """Immutable token bucket over an explicitly named native-cost vector."""

    limit: NativeCostVector
    used: NativeCostVector = NativeCostVector()

    def __post_init__(self) -> None:
        if not self.used.fits_within(self.limit):
            raise ValueError("used native cost exceeds its declared limit")

    def can_reserve(self, reservation: NativeCostVector) -> bool:
        return (self.used + reservation).fits_within(self.limit)

    def charge(self, reservation: NativeCostVector) -> "NativeBudgetLedger":
        if not self.can_reserve(reservation):
            raise ValueError("native-cost reservation exceeds the remaining budget")
        return NativeBudgetLedger(self.limit, self.used + reservation)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "accounting": "provider_native_vector",
            "limit": self.limit.to_json_dict(),
            "used": self.used.to_json_dict(),
        }


@dataclass(frozen=True, slots=True)
class NativeCostReservation:
    """Pre-declared native work reserved for one typed provider option."""

    actor: str
    operator: str
    cost: NativeCostVector

    def __post_init__(self) -> None:
        if not isinstance(self.actor, str) or not self.actor:
            raise TypeError("reservation actor must be a non-empty string")
        if not isinstance(self.operator, str) or not self.operator:
            raise TypeError("reservation operator must be a non-empty string")
        if not isinstance(self.cost, NativeCostVector):
            raise TypeError("reservation cost must be a NativeCostVector")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor,
            "operator": self.operator,
            "cost": self.cost.to_json_dict(),
        }


@dataclass(frozen=True, slots=True)
class NativeCostContract:
    """Content-addressed reservations calibrated before held-out replay."""

    reservations: tuple[NativeCostReservation, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(item, NativeCostReservation) for item in self.reservations
        ):
            raise TypeError("contract requires NativeCostReservation entries")
        canonical = tuple(
            sorted(
                self.reservations,
                key=lambda item: (item.actor, item.operator),
            )
        )
        keys = tuple((item.actor, item.operator) for item in canonical)
        if len(set(keys)) != len(keys):
            raise ValueError("native-cost reservations require unique action keys")
        object.__setattr__(self, "reservations", canonical)

    @classmethod
    def from_json_dict(cls, value: object) -> "NativeCostContract":
        if not isinstance(value, Mapping):
            raise TypeError("native-cost contract JSON must be a mapping")
        if value.get("schema") != NATIVE_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unknown native-cost contract schema")
        raw = value.get("reservations")
        if not isinstance(raw, list):
            raise TypeError("contract reservations must be a list")
        reservations = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise TypeError("contract reservation must be a mapping")
            reservations.append(
                NativeCostReservation(
                    actor=item.get("actor"),
                    operator=item.get("operator"),
                    cost=NativeCostVector.from_json_dict(item.get("cost")),
                )
            )
        contract = cls(tuple(reservations))
        declared_id = value.get("contract_id")
        if declared_id is not None and declared_id != contract.contract_id:
            raise ValueError("native-cost contract ID does not match its content")
        return contract

    def reservation_for(self, actor: str, operator: str) -> NativeCostVector | None:
        return next(
            (
                item.cost
                for item in self.reservations
                if item.actor == actor and item.operator == operator
            ),
            None,
        )

    @property
    def contract_id(self) -> str:
        payload = {
            "schema": NATIVE_CONTRACT_SCHEMA_VERSION,
            "reservations": [item.to_json_dict() for item in self.reservations],
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": NATIVE_CONTRACT_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "reservations": [item.to_json_dict() for item in self.reservations],
        }


def conditional_mutual_information(
    x: Sequence[Hashable],
    y: Sequence[Hashable],
    condition: Sequence[Hashable],
) -> float:
    """Empirical plug-in estimate of I(X;Y|Z), in bits.

    The estimator is intentionally small and dependency-free.  Sparse ARC
    strata can bias it upward, so experiments must pair it with a stratified
    permutation null rather than interpreting the raw value alone.
    """

    if not (len(x) == len(y) == len(condition)):
        raise ValueError("CMI inputs must have equal length")
    if not x:
        return 0.0
    xyz = Counter(zip(x, y, condition))
    xz = Counter(zip(x, condition))
    yz = Counter(zip(y, condition))
    z = Counter(condition)
    total = float(len(x))
    information = 0.0
    for (x_value, y_value, z_value), count in xyz.items():
        numerator = count * z[z_value]
        denominator = xz[(x_value, z_value)] * yz[(y_value, z_value)]
        information += (count / total) * math.log2(numerator / denominator)
    return information
