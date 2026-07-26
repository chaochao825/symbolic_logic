from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest

from afts_arc.experiment_safety import (
    ExperimentSafetyError,
    TaskFingerprint,
    atomic_write_json,
    assert_three_axis_disjoint,
    candidate_dag_id,
    canonical_json,
    canonical_sha256,
    capture_git_source_provenance,
    file_sha256,
    runtime_metadata,
    validate_native_budget_profile,
    validate_pool_manifest,
    validate_summary,
    verify_source_provenance_unchanged,
)


def _digest(character: str) -> str:
    return character * 64


def _fingerprint(task_id: str, source: str, blind: str) -> TaskFingerprint:
    return TaskFingerprint(task_id, _digest(source), _digest(blind))


def _with_id(payload: dict[str, object], field: str) -> dict[str, object]:
    body = copy.deepcopy(payload)
    body[field] = canonical_sha256(body)
    return body


def _summary() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "afts.online-matched-budget/v3",
        "training_started": False,
        "oracle_access": "posthoc_only_after_each_frozen_replay",
        "dataset": {
            "split": "training",
            "requested_task_count": 2,
            "completed_task_count": 1,
            "failed_task_count": 1,
            "task_ids": ["complete", "failed"],
        },
        "tasks": [
            {
                "task_id": "complete",
                "task_source_sha256": _digest("a"),
                "blind_content_sha256": _digest("b"),
            }
        ],
        "failures": [{"task_id": "failed", "error_type": "FixtureError"}],
    }
    return _with_id(payload, "result_id")


def _profile_v1() -> dict[str, object]:
    return _with_id(
        {
            "schema": "afts.native-budget-profile/v1",
            "oracle_fields_read": 0,
            "contract": {"schema": "fixture"},
            "budget_limit": {"work": 4},
        },
        "profile_id",
    )


def _profile_v2(fit: TaskFingerprint) -> dict[str, object]:
    head = "1" * 40
    return _with_id(
        {
            "schema": "afts.native-budget-profile/v2",
            "source_commit": head,
            "source_provenance": {
                "head": head,
                "dirty": False,
                "diff_sha256": _digest("0"),
                "tree_sha256": _digest("1"),
                "paths": ["src", "scripts", "config"],
            },
            "fit_task_count": 1,
            "fit_task_ids_sha256": canonical_sha256([fit.task_id]),
            "fit_task_fingerprints": [fit.to_json_dict()],
            "oracle_fields_read": 0,
            "contract": {"schema": "fixture"},
            "budget_limit": {"work": 4},
        },
        "profile_id",
    )


def _pool() -> dict[str, object]:
    candidate = {
        "hypothesis_id": "candidate",
        "route": "dsl_program",
        "payload": {"operator": "identity"},
    }
    native_cost = {"items": [["expansions", 2.0]]}
    batch_identity = {
        "operator": "synthesize",
        "parent_hypothesis_id": None,
        "candidates": [candidate],
        "native_cost": native_cost,
    }
    payload: dict[str, object] = {
        "schema": "afts.frozen-action-pool/v3",
        "task_id": "task",
        "task_source_sha256": _digest("c"),
        "blind_content_sha256": _digest("d"),
        "candidate_count": 1,
        "candidates": [candidate],
        "providers": [
            {
                "provider": "typed_dsl",
                "route": "dsl_program",
                "strict_live_budget_contract": False,
                "frozen_action_batches": [
                    {
                        "batch_id": canonical_sha256(batch_identity)[:24],
                        "operator": "synthesize",
                        "parent_hypothesis_id": None,
                        "candidate_ids": ["candidate"],
                        "native_cost": native_cost,
                    }
                ],
                "actions": [{"status": "ok", "diagnostics": {"elapsed": 9.0}}],
            }
        ],
        "discovery_policies": {"first": {"actions": ["dsl"]}},
        "frozen_replay_policies": {"first": {"actions": ["dsl"]}},
        "pool_closure": {"replay_candidate_count": 1},
        "oracle_used_during_discovery": False,
        "oracle_used_during_pool_closure": False,
    }
    return _with_id(payload, "pool_id")


def test_canonical_and_file_sha256_are_stable(tmp_path: Path) -> None:
    assert canonical_json({"b": 1, "a": [2]}) == '{"a":[2],"b":1}'
    assert canonical_sha256({"a": 1}) == canonical_sha256({"a": 1})
    fixture = tmp_path / "fixture.bin"
    fixture.write_bytes(b"experiment")
    assert file_sha256(fixture) == hashlib.sha256(b"experiment").hexdigest()


def test_atomic_json_write_is_immutable_and_runtime_is_explicit(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.json"
    assert atomic_write_json(target, {"value": 1})
    assert not atomic_write_json(target, {"value": 1})
    with pytest.raises(ExperimentSafetyError, match="refusing to replace"):
        atomic_write_json(target, {"value": 2})
    runtime = runtime_metadata(("definitely-not-an-installed-distribution",))
    assert runtime["python_version"]
    assert runtime["packages"] == {"definitely-not-an-installed-distribution": None}


def test_summary_integrity_enforces_counts_oracle_and_training_split() -> None:
    report = validate_summary(_summary())
    assert report.completed_task_count == 1
    assert report.task_fingerprints[0].task_id == "complete"

    for mutation, message in (
        (("oracle_access", "controller_visible"), "oracle_access"),
        (("training_started", True), "training_started"),
    ):
        changed = _summary()
        changed[mutation[0]] = mutation[1]
        changed["result_id"] = canonical_sha256(
            {key: value for key, value in changed.items() if key != "result_id"}
        )
        with pytest.raises(ExperimentSafetyError, match=message):
            validate_summary(changed)

    evaluation = _summary()
    evaluation["dataset"]["split"] = "evaluation"  # type: ignore[index]
    evaluation["result_id"] = canonical_sha256(
        {key: value for key, value in evaluation.items() if key != "result_id"}
    )
    with pytest.raises(ExperimentSafetyError, match="required 'training'"):
        validate_summary(evaluation)

    mismatched = _summary()
    mismatched["dataset"]["completed_task_count"] = 2  # type: ignore[index]
    mismatched["result_id"] = canonical_sha256(
        {key: value for key, value in mismatched.items() if key != "result_id"}
    )
    with pytest.raises(ExperimentSafetyError, match="requested_task_count"):
        validate_summary(mismatched)


def test_summary_rejects_content_id_tampering() -> None:
    summary = _summary()
    summary["oracle_access"] = "controller_visible"
    with pytest.raises(ExperimentSafetyError, match="result_id"):
        validate_summary(summary)


@pytest.mark.parametrize(
    "axis", ["task_id", "task_source_sha256", "blind_content_sha256"]
)
def test_three_axis_disjoint_rejects_each_overlap(axis: str) -> None:
    fit = _fingerprint("fit", "a", "b")
    values = {
        "task_id": "heldout",
        "task_source_sha256": _digest("c"),
        "blind_content_sha256": _digest("d"),
    }
    values[axis] = getattr(fit, axis)
    heldout = TaskFingerprint(**values)
    with pytest.raises(ExperimentSafetyError, match=axis):
        assert_three_axis_disjoint((fit,), (heldout,))


def test_native_profile_v1_is_compatible_but_cannot_guard_overlap() -> None:
    result = validate_native_budget_profile(
        _profile_v1(), heldout_fingerprints=(_fingerprint("heldout", "c", "d"),)
    )
    assert result.overlap_guard == "unavailable"
    assert result.fit_fingerprints == ()

    tampered = _profile_v1()
    tampered["budget_limit"] = {"work": 999}
    with pytest.raises(ExperimentSafetyError, match="profile_id"):
        validate_native_budget_profile(tampered)


def test_native_profile_v2_enforces_three_axis_fit_boundary() -> None:
    fit = _fingerprint("fit", "a", "b")
    safe = _fingerprint("heldout", "c", "d")
    result = validate_native_budget_profile(
        _profile_v2(fit), heldout_fingerprints=(safe,)
    )
    assert result.overlap_guard == "verified"

    source_overlap = _fingerprint("other", "a", "e")
    with pytest.raises(ExperimentSafetyError, match="task_source_sha256"):
        validate_native_budget_profile(
            _profile_v2(fit), heldout_fingerprints=(source_overlap,)
        )

    dirty = _profile_v2(fit)
    dirty["source_provenance"]["dirty"] = True  # type: ignore[index]
    dirty["profile_id"] = canonical_sha256(
        {key: value for key, value in dirty.items() if key != "profile_id"}
    )
    with pytest.raises(ExperimentSafetyError, match="produced dirty"):
        validate_native_budget_profile(dirty, heldout_fingerprints=(safe,))


def test_pool_manifest_checks_content_task_oracle_flags_and_references() -> None:
    pool = _pool()
    expected = _fingerprint("task", "c", "d")
    report = validate_pool_manifest(pool, expected_task=expected)
    assert report.pool_id == pool["pool_id"]

    tampered = copy.deepcopy(pool)
    tampered["oracle_used_during_discovery"] = True
    tampered["pool_id"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "pool_id"}
    )
    with pytest.raises(ExperimentSafetyError, match="oracle-free"):
        validate_pool_manifest(tampered, expected_task=expected)

    unknown = copy.deepcopy(pool)
    unknown["providers"][0]["frozen_action_batches"][0]["candidate_ids"] = [  # type: ignore[index]
        "missing"
    ]
    unknown["pool_id"] = canonical_sha256(
        {key: value for key, value in unknown.items() if key != "pool_id"}
    )
    with pytest.raises(ExperimentSafetyError, match="unknown candidate"):
        validate_pool_manifest(unknown, expected_task=expected)


def test_candidate_dag_id_ignores_policy_audit_but_tracks_action_content() -> None:
    first = _pool()
    second = copy.deepcopy(first)
    second["discovery_policies"] = {"new": {"actions": ["scene", "ca"]}}
    second["frozen_replay_policies"] = {"new": {"actions": []}}
    second["pool_closure"] = {"replay_candidate_count": 99}
    second["providers"][0]["actions"] = [{"status": "abstained"}]  # type: ignore[index]
    assert candidate_dag_id(first) == candidate_dag_id(second)

    changed_cost = copy.deepcopy(first)
    changed_cost["providers"][0]["frozen_action_batches"][0]["native_cost"] = {  # type: ignore[index]
        "items": [["expansions", 3.0]]
    }
    assert candidate_dag_id(first) != candidate_dag_id(changed_cost)

    extended = copy.deepcopy(first)
    extended["audit_manifest_id"] = extended["pool_id"]
    extended["candidate_dag_id"] = candidate_dag_id(extended)
    assert validate_pool_manifest(extended).candidate_dag_id == candidate_dag_id(first)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_git_source_provenance_is_scoped_and_start_end_checked(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "config", "user.name", "Fixture")
    for relative, content in (
        ("src/module.py", "VALUE = 1\n"),
        ("scripts/run.py", "print('run')\n"),
        ("config/experiment.json", "{}\n"),
        ("notes/readme.md", "outside scope\n"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "fixture")

    start = capture_git_source_provenance(tmp_path)
    assert not start.dirty

    (tmp_path / "notes/readme.md").write_text("changed only notes\n", encoding="utf-8")
    notes_only = capture_git_source_provenance(tmp_path)
    verify_source_provenance_unchanged(start, notes_only)

    (tmp_path / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    end = capture_git_source_provenance(tmp_path)
    assert end.dirty
    assert end.diff_sha256 != start.diff_sha256
    assert end.tree_sha256 != start.tree_sha256
    with pytest.raises(ExperimentSafetyError, match="source provenance changed"):
        verify_source_provenance_unchanged(start, end)
