from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from afts_arc.experiment_safety import file_sha256


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "afts_arc_relational_transducer_gate.py"
)


def _script_module() -> object:
    spec = importlib.util.spec_from_file_location("relational_transducer_gate_cli", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load relational-transducer gate CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_additional_sources_are_hashed_sorted_and_backward_compatible(
    tmp_path: Path,
) -> None:
    module = _script_module()
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    assert module._additional_source_hashes([]) == {}
    assert module._additional_source_hashes(
        [f"zeta={second}", f"alpha={first}"]
    ) == {"alpha": file_sha256(first), "zeta": file_sha256(second)}


@pytest.mark.parametrize(
    "value",
    ["missing-separator", "=missing-name", "protocol=reserved"],
)
def test_additional_sources_reject_malformed_or_reserved_values(value: str) -> None:
    module = _script_module()
    with pytest.raises(ValueError):
        module._additional_source_hashes([value])
