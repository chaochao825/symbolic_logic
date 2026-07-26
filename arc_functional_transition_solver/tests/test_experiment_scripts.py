from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from afts_arc.experiment_safety import ExperimentSafetyError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "residual_compiled_metareasoning_p1_20260726"


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "scripts" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_calibrator_only_converts_the_exact_known_legacy_cost_flag() -> None:
    calibrator = _load_script(
        "test_native_calibrator", "afts_arc_calibrate_native_budget.py"
    )
    conversions: dict[str, int] = {}
    assert calibrator._cost_mapping(
        {"work": 2, "bounded_program_trials_enabled": True},
        prefix="provider.native_cost",
        legacy_conversions=conversions,
    ) == {"provider.native_cost.work": 2.0}
    assert conversions == {"provider.native_cost.bounded_program_trials_enabled": 1}

    with pytest.raises(TypeError, match="unsupported leaf"):
        calibrator._cost_mapping(
            {"nested": {"bounded_program_trials_enabled": True}},
            prefix="provider.native_cost",
            legacy_conversions={},
        )


def test_runner_verifies_legacy_profile_id_and_reports_guard_gap(
    tmp_path: Path,
) -> None:
    runner = _load_script(
        "test_matched_budget_runner", "afts_arc_online_matched_budget.py"
    )
    profile_path = RESULT_ROOT / "native_budget_fit_offset0_max.json"
    _, _, metadata = runner._load_native_budget_profile(profile_path)
    assert metadata["fit_overlap_guard"] == "unavailable"

    tampered = json.loads(profile_path.read_text(encoding="utf-8"))
    tampered["budget_limit"]["items"][0][1] += 1
    tampered_path = tmp_path / "tampered-profile.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ExperimentSafetyError, match="profile_id"):
        runner._load_native_budget_profile(tampered_path)


def test_runner_rejects_duplicate_explicit_task_ids() -> None:
    runner = _load_script(
        "test_matched_budget_selection", "afts_arc_online_matched_budget.py"
    )
    with pytest.raises(ValueError, match="must be unique"):
        runner._select_tasks(
            (),
            seed=0,
            offset=0,
            limit=2,
            task_ids="duplicate,duplicate",
        )
