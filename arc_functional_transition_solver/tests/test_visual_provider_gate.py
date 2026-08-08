from __future__ import annotations

import json
from pathlib import Path

import pytest

from afts_arc.visual_provider_gate import (
    EXPOSURE_REGISTRY_SCHEMA,
    VisualProviderGateError,
    aggregate_visual_provider_runs,
    audit_posterior_consensus_bridge,
    audit_posterior_error_localization,
    freeze_and_score_provider,
    freeze_provider_predictions,
    make_query_blind_payload,
    prepare_query_blind_cohort,
    rank_predictions,
)
from afts_arc.experiment_safety import canonical_sha256


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _task(output: list[list[int]]) -> dict[str, object]:
    return {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": output}],
    }


def test_query_blind_payload_is_invariant_to_gold_output() -> None:
    first = make_query_blind_payload(_task([[4]]))
    second = make_query_blind_payload(_task([[9, 9]]))
    assert first == second
    assert first["test"] == [{"input": [[3]], "output": [[3]]}]


def test_query_blind_payload_replaces_every_query_output() -> None:
    payload = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [
            {"input": [[3]], "output": [[4]]},
            {"input": [[5, 6]], "output": [[7], [8]]},
        ],
    }

    blind = make_query_blind_payload(payload)

    assert blind["test"] == [
        {"input": [[3]], "output": [[3]]},
        {"input": [[5, 6]], "output": [[5, 6]]},
    ]


def test_prepare_excludes_arc1_and_previously_exposed_ids(tmp_path: Path) -> None:
    arc2 = tmp_path / "arc2"
    arc1 = tmp_path / "arc1"
    project = tmp_path / "project"
    _write(arc2 / "00000001.json", _task([[1]]))
    _write(arc2 / "00000002.json", _task([[2]]))
    _write(arc2 / "00000003.json", _task([[3]]))
    _write(arc2 / "00000004.json", _task([[4]]))
    _write(arc1 / "00000001.json", _task([[1]]))
    _write(project / "notes" / "prior.md", {"task_id": "00000002"})
    registry_content = {
        "schema": EXPOSURE_REGISTRY_SCHEMA,
        "sources": [{"kind": "test"}],
        "task_ids": ["00000003"],
    }
    _write(
        tmp_path / "exposure.json",
        {
            "registry_id": canonical_sha256(registry_content),
            **registry_content,
        },
    )

    manifest = prepare_query_blind_cohort(
        arc2_training_dir=arc2,
        arc1_task_dirs=(arc1,),
        project_root=project,
        exposure_registry_paths=(tmp_path / "exposure.json",),
        output_dir=tmp_path / "cohort",
        seed="test",
        limit=1,
        arc1_source_commit="arc1",
        arc2_source_commit="arc2",
    )

    assert [record["task_id"] for record in manifest["tasks"]] == ["00000004"]
    assert manifest["source"]["arc1_task_dirs"] == [str(arc1.resolve())]
    blind_path = (
        tmp_path
        / "cohort"
        / "blind_data"
        / "ARC-AGI-2"
        / "data"
        / "evaluation"
        / "00000004.json"
    )
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    assert blind["test"][0]["output"] == blind["test"][0]["input"]


def test_prediction_ranking_is_frequency_then_first_seen() -> None:
    ranked = rank_predictions([[[2]], [[1]], [[2]], [[1]]])
    assert ranked == (((2,),), ((1,),))


def test_predictions_can_freeze_without_gold_or_baseline(tmp_path: Path) -> None:
    task_id = "00000003"
    manifest = {
        "schema": "afts.visual-provider-cohort/v1",
        "cohort_id": "1" * 64,
        "tasks": [{"task_id": task_id, "query_count": 1}],
    }
    predictions = tmp_path / "predictions"
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, manifest)
    _write(predictions / f"{task_id}_predictions.json", {"0": [[[3]], [[4]]]})

    frozen = freeze_provider_predictions(
        cohort_manifest=manifest_path,
        prediction_roots=(predictions,),
        provider_contract={"name": "query-blind-test"},
        output_path=tmp_path / "frozen.json",
        invalid_candidate_policy="error",
    )

    assert frozen["cohort_id"] == manifest["cohort_id"]
    assert frozen["prediction_validation"]["raw_candidate_count"] == 2
    assert frozen["prediction_validation"]["valid_candidate_count"] == 2
    assert "metrics" not in frozen


def test_predictions_freeze_before_missing_gold_is_detected(tmp_path: Path) -> None:
    arc2 = tmp_path / "arc2"
    arc1 = tmp_path / "arc1"
    project = tmp_path / "project"
    _write(arc2 / "00000003.json", _task([[3]]))
    arc1.mkdir()
    manifest = prepare_query_blind_cohort(
        arc2_training_dir=arc2,
        arc1_task_dirs=(arc1,),
        project_root=project,
        exposure_registry_paths=(),
        output_dir=tmp_path / "cohort",
        seed="test",
        limit=1,
        arc1_source_commit="arc1",
        arc2_source_commit="arc2",
    )
    predictions = tmp_path / "predictions"
    _write(predictions / "00000003_predictions.json", {"0": [[[3]]]})
    baseline = {
        "tasks": [
            {
                "task_id": "00000003",
                "policies": {
                    "fixed": {
                        "metrics": {
                            "pool_oracle_covered": False,
                            "pool_selectable_oracle_covered": False,
                            "pass_at_k": False,
                        }
                    }
                },
            }
        ]
    }
    _write(tmp_path / "baseline.json", baseline)
    output = tmp_path / "score"

    with pytest.raises(FileNotFoundError):
        freeze_and_score_provider(
            cohort_manifest=tmp_path / "cohort" / "manifest.json",
            gold_training_dir=tmp_path / "missing-gold",
            prediction_roots=(predictions,),
            baseline_summary=tmp_path / "baseline.json",
            object_provider_summary=None,
            provider_contract={"name": "test"},
            output_dir=output,
            pilot_unique_gate=1,
            invalid_candidate_policy="error",
        )

    frozen = json.loads((output / "frozen_predictions.json").read_text(encoding="utf-8"))
    assert frozen["cohort_id"] == manifest["cohort_id"]
    assert "metrics" not in frozen


def test_score_supports_multiple_queries(tmp_path: Path) -> None:
    task_id = "00000003"
    task = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [
            {"input": [[3]], "output": [[4]]},
            {"input": [[5]], "output": [[6]]},
        ],
    }
    arc2 = tmp_path / "arc2"
    arc1 = tmp_path / "arc1"
    _write(arc2 / f"{task_id}.json", task)
    arc1.mkdir()
    prepare_query_blind_cohort(
        arc2_training_dir=arc2,
        arc1_task_dirs=(arc1,),
        project_root=tmp_path / "project",
        exposure_registry_paths=(),
        output_dir=tmp_path / "cohort",
        seed="test",
        limit=1,
        arc1_source_commit="arc1",
        arc2_source_commit="arc2",
    )
    predictions = tmp_path / "predictions"
    _write(
        predictions / f"{task_id}_predictions.json",
        {"0": [[[4]], [[9]]], "1": [[[0]], [[6]]]},
    )
    _write(
        tmp_path / "baseline.json",
        {
            "tasks": [
                {
                    "task_id": task_id,
                    "policies": {
                        "fixed": {
                            "metrics": {
                                "pool_oracle_covered": False,
                                "pool_selectable_oracle_covered": False,
                                "pass_at_k": False,
                            }
                        }
                    },
                }
            ]
        },
    )

    summary = freeze_and_score_provider(
        cohort_manifest=tmp_path / "cohort" / "manifest.json",
        gold_training_dir=arc2,
        prediction_roots=(predictions,),
        baseline_summary=tmp_path / "baseline.json",
        object_provider_summary=None,
        provider_contract={"name": "test"},
        output_dir=tmp_path / "score",
        pilot_unique_gate=1,
        invalid_candidate_policy="error",
    )

    assert summary["metrics"]["visual_raw_coverage"] == 1
    assert summary["metrics"]["visual_selectable_coverage"] == 1
    assert summary["metrics"]["visual_pass_at_1"] == 0
    assert summary["metrics"]["visual_pass_at_2"] == 1
    assert summary["metrics"]["visual_unique_selectable_coverage"] == 1
    first_query, second_query = summary["tasks"][0]["queries"]
    assert first_query["gold_rank_in_unique_candidates"] == 1
    assert first_query["gold_sample_support_count"] == 1
    assert first_query["gold_sample_support_fraction"] == 0.5
    assert first_query["gold_shape"] == [1, 1]
    assert first_query["modal_sample_shape_matches_gold"] is True
    assert first_query["best_same_shape_mismatch_count"] == 0
    assert second_query["gold_rank_in_unique_candidates"] == 2
    assert second_query["top_1_same_shape_mismatch_count"] == 1


def test_invalid_candidates_are_rejected_before_gold_scoring(tmp_path: Path) -> None:
    task_id = "00000003"
    arc2 = tmp_path / "arc2"
    arc1 = tmp_path / "arc1"
    _write(arc2 / f"{task_id}.json", _task([[4]]))
    arc1.mkdir()
    prepare_query_blind_cohort(
        arc2_training_dir=arc2,
        arc1_task_dirs=(arc1,),
        project_root=tmp_path / "project",
        exposure_registry_paths=(),
        output_dir=tmp_path / "cohort",
        seed="test",
        limit=1,
        arc1_source_commit="arc1",
        arc2_source_commit="arc2",
    )
    predictions = tmp_path / "predictions"
    _write(
        predictions / f"{task_id}_predictions.json",
        {"0": [[[11]], [], [[4]], [[4]]]},
    )
    _write(
        tmp_path / "baseline.json",
        {
            "tasks": [
                {
                    "task_id": task_id,
                    "policies": {
                        "fixed": {
                            "metrics": {
                                "pool_oracle_covered": False,
                                "pool_selectable_oracle_covered": False,
                                "pass_at_k": False,
                            }
                        }
                    },
                }
            ]
        },
    )

    summary = freeze_and_score_provider(
        cohort_manifest=tmp_path / "cohort" / "manifest.json",
        gold_training_dir=arc2,
        prediction_roots=(predictions,),
        baseline_summary=tmp_path / "baseline.json",
        object_provider_summary=None,
        provider_contract={"name": "test"},
        output_dir=tmp_path / "score",
        pilot_unique_gate=1,
        invalid_candidate_policy="reject",
    )

    frozen = json.loads(
        (tmp_path / "score" / "frozen_predictions.json").read_text(encoding="utf-8")
    )
    validation = frozen["prediction_validation"]
    assert validation["raw_candidate_count"] == 4
    assert validation["valid_candidate_count"] == 2
    assert validation["rejected_candidate_count"] == 2
    assert summary["metrics"]["visual_pass_at_1"] == 1
    query = summary["tasks"][0]["queries"][0]
    assert query["raw_sample_count"] == 4
    assert query["valid_sample_count"] == 2
    assert query["rejected_sample_count"] == 2


def test_strict_invalid_candidate_failure_precedes_gold_access(tmp_path: Path) -> None:
    task_id = "00000003"
    arc2 = tmp_path / "arc2"
    arc1 = tmp_path / "arc1"
    _write(arc2 / f"{task_id}.json", _task([[4]]))
    arc1.mkdir()
    prepare_query_blind_cohort(
        arc2_training_dir=arc2,
        arc1_task_dirs=(arc1,),
        project_root=tmp_path / "project",
        exposure_registry_paths=(),
        output_dir=tmp_path / "cohort",
        seed="test",
        limit=1,
        arc1_source_commit="arc1",
        arc2_source_commit="arc2",
    )
    predictions = tmp_path / "predictions"
    _write(predictions / f"{task_id}_predictions.json", {"0": [[[11]]]})
    _write(
        tmp_path / "baseline.json",
        {
            "tasks": [
                {
                    "task_id": task_id,
                    "policies": {
                        "fixed": {
                            "metrics": {
                                "pool_oracle_covered": False,
                                "pool_selectable_oracle_covered": False,
                                "pass_at_k": False,
                            }
                        }
                    },
                }
            ]
        },
    )

    with pytest.raises(
        VisualProviderGateError,
        match="invalid prediction for 00000003 query 0 sample 0",
    ):
        freeze_and_score_provider(
            cohort_manifest=tmp_path / "cohort" / "manifest.json",
            gold_training_dir=tmp_path / "missing-gold",
            prediction_roots=(predictions,),
            baseline_summary=tmp_path / "baseline.json",
            object_provider_summary=None,
            provider_contract={"name": "test"},
            output_dir=tmp_path / "score",
            pilot_unique_gate=1,
            invalid_candidate_policy="error",
        )

    assert not (tmp_path / "score").exists()


def test_posterior_disagreement_localizes_top_1_error(tmp_path: Path) -> None:
    task_id = "00000003"
    arc2 = tmp_path / "arc2"
    arc1 = tmp_path / "arc1"
    _write(arc2 / f"{task_id}.json", _task([[1, 1], [1, 2]]))
    arc1.mkdir()
    prepare_query_blind_cohort(
        arc2_training_dir=arc2,
        arc1_task_dirs=(arc1,),
        project_root=tmp_path / "project",
        exposure_registry_paths=(),
        output_dir=tmp_path / "cohort",
        seed="test",
        limit=1,
        arc1_source_commit="arc1",
        arc2_source_commit="arc2",
    )
    predictions = tmp_path / "predictions"
    _write(
        predictions / f"{task_id}_predictions.json",
        {
            "0": [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
                [[1, 1], [1, 2]],
            ]
        },
    )
    _write(
        tmp_path / "baseline.json",
        {
            "tasks": [
                {
                    "task_id": task_id,
                    "policies": {
                        "fixed": {
                            "metrics": {
                                "pool_oracle_covered": False,
                                "pool_selectable_oracle_covered": False,
                                "pass_at_k": False,
                            }
                        }
                    },
                }
            ]
        },
    )
    freeze_and_score_provider(
        cohort_manifest=tmp_path / "cohort" / "manifest.json",
        gold_training_dir=arc2,
        prediction_roots=(predictions,),
        baseline_summary=tmp_path / "baseline.json",
        object_provider_summary=None,
        provider_contract={"name": "test"},
        output_dir=tmp_path / "score",
        pilot_unique_gate=1,
        invalid_candidate_policy="error",
    )

    summary = audit_posterior_error_localization(
        cohort_manifest=tmp_path / "cohort" / "manifest.json",
        gold_training_dir=arc2,
        prediction_roots=(predictions,),
        frozen_predictions=tmp_path / "score" / "frozen_predictions.json",
        output_dir=tmp_path / "posterior",
        invalid_candidate_policy="error",
        mask_fractions=(0.1,),
        disagreement_thresholds=(0.1,),
    )

    query = summary["queries"][0]
    localization = query["localization"]
    assert localization["error_count"] == 1
    assert localization["error_localization_auc"] == 1.0
    mask = localization["fraction_masks"][0]
    assert mask["selected_count"] == 1
    assert mask["precision"] == 1.0
    assert mask["recall"] == 1.0
    assert summary["decision"]["strong_localization_signal"] is True


def test_posterior_consensus_adds_a_novel_exact_candidate(tmp_path: Path) -> None:
    task_id = "00000003"
    arc2 = tmp_path / "arc2"
    arc1 = tmp_path / "arc1"
    _write(arc2 / f"{task_id}.json", _task([[1, 1], [1, 1]]))
    arc1.mkdir()
    prepare_query_blind_cohort(
        arc2_training_dir=arc2,
        arc1_task_dirs=(arc1,),
        project_root=tmp_path / "project",
        exposure_registry_paths=(),
        output_dir=tmp_path / "cohort",
        seed="test",
        limit=1,
        arc1_source_commit="arc1",
        arc2_source_commit="arc2",
    )
    predictions = tmp_path / "predictions"
    _write(
        predictions / f"{task_id}_predictions.json",
        {
            "0": [
                [[2, 1], [1, 1]],
                [[2, 1], [1, 1]],
                [[1, 2], [1, 1]],
                [[1, 1], [2, 1]],
                [[1, 1], [1, 2]],
            ]
        },
    )
    _write(
        tmp_path / "baseline.json",
        {
            "tasks": [
                {
                    "task_id": task_id,
                    "policies": {
                        "fixed": {
                            "metrics": {
                                "pool_oracle_covered": False,
                                "pool_selectable_oracle_covered": False,
                                "pass_at_k": False,
                            }
                        }
                    },
                }
            ]
        },
    )
    freeze_and_score_provider(
        cohort_manifest=tmp_path / "cohort" / "manifest.json",
        gold_training_dir=arc2,
        prediction_roots=(predictions,),
        baseline_summary=tmp_path / "baseline.json",
        object_provider_summary=None,
        provider_contract={"name": "test"},
        output_dir=tmp_path / "score",
        pilot_unique_gate=1,
        invalid_candidate_policy="error",
    )

    summary = audit_posterior_consensus_bridge(
        cohort_manifest=tmp_path / "cohort" / "manifest.json",
        gold_training_dir=arc2,
        prediction_roots=(predictions,),
        frozen_predictions=tmp_path / "score" / "frozen_predictions.json",
        visual_score_summary=tmp_path / "score" / "summary.json",
        output_dir=tmp_path / "bridge",
        invalid_candidate_policy="error",
    )

    metrics = summary["metrics"]
    assert metrics["novel_frontier_query_count"] == 1
    assert metrics["consensus_novel_exact_query_count"] == 1
    assert metrics["unique_raw_recovery"] == 1
    assert metrics["unique_selectable_novel_recovery"] == 1
    assert summary["decision"]["advance_to_matched_cost_bridge_gate"] is True


def test_aggregate_combines_disjoint_content_verified_cohorts(tmp_path: Path) -> None:
    def contract(raw: int) -> dict[str, object]:
        return {
            "controller_frozen": True,
            "source": {"checkpoint": "same"},
            "test_time_configuration": {"seed": 42},
            "candidate_contract": {
                "invalid_candidate_policy": "reject",
                "invalid_candidate_rule": "fixed",
                "ranking": "fixed",
                "raw_candidate_count": raw,
                "valid_candidate_count": raw,
                "rejected_candidate_count": 0,
            },
            "query_blind_protocol": {
                "gold_output_present_in_provider_root": False,
                "provider_visible_test_output": "input sentinel",
                "released_loader_repair": "fixed",
            },
            "compute": {"gpu_seconds": 10},
        }

    def score(cohort: str, task_id: str, selectable: int) -> dict[str, object]:
        metrics = {
            "task_count": 1,
            "base_raw_coverage": 0,
            "base_selectable_coverage": 0,
            "visual_raw_coverage": 1,
            "visual_selectable_coverage": selectable,
            "visual_pass_at_1": selectable,
            "visual_pass_at_2": selectable,
            "visual_unique_raw_coverage": 1,
            "visual_unique_selectable_coverage": selectable,
            "base_plus_visual_raw_coverage": 1,
            "base_plus_visual_selectable_coverage": selectable,
        }
        body = {
            "schema": "afts.visual-provider-gate/v2",
            "cohort_id": cohort,
            "frozen_prediction_id": f"freeze-{cohort}",
            "provider_contract": contract(3),
            "metrics": metrics,
            "decision": {},
            "tasks": [
                {
                    "task_id": task_id,
                    "queries": [
                        {
                            "gold_rank_in_unique_candidates": 3,
                            "raw_oracle_covered": True,
                            "best_same_shape_mismatch_rate": 0.0,
                        }
                    ],
                }
            ],
        }
        return {"result_id": canonical_sha256(body), **body}

    def posterior(cohort: str, task_id: str) -> dict[str, object]:
        body = {
            "schema": "afts.visual-posterior-localization/v1",
            "cohort_id": cohort,
            "frozen_prediction_id": f"freeze-{cohort}",
            "aggregate": {"task_count": 1},
            "decision": {},
            "queries": [
                {
                    "task_id": task_id,
                    "query_index": 0,
                    "top_1_exact": False,
                    "localization": {
                        "error_localization_auc": 0.9,
                        "fraction_masks": [
                            {
                                "fraction": 0.1,
                                "precision": 0.5,
                                "recall": 0.6,
                                "recall_lift_over_random": 6.0,
                            }
                        ],
                    },
                }
            ],
            "replay_id": f"replay-{cohort}",
        }
        return {"result_id": canonical_sha256(body), **body}

    score_paths: list[Path] = []
    posterior_paths: list[Path] = []
    for index, selectable in enumerate((0, 1)):
        score_path = tmp_path / f"score-{index}.json"
        posterior_path = tmp_path / f"posterior-{index}.json"
        _write(score_path, score(f"cohort-{index}", f"0000000{index}", selectable))
        _write(
            posterior_path,
            posterior(f"cohort-{index}", f"0000000{index}"),
        )
        score_paths.append(score_path)
        posterior_paths.append(posterior_path)

    summary = aggregate_visual_provider_runs(
        score_summaries=score_paths,
        posterior_summaries=posterior_paths,
        output_dir=tmp_path / "aggregate",
        confirmation_score_index=1,
        bridge_summary=None,
    )

    metrics = summary["solver_track"]["metrics"]
    assert metrics["task_count"] == 2
    assert metrics["visual_raw_coverage"] == 2
    assert metrics["visual_selectable_coverage"] == 1
    assert summary["mechanism_track"]["posterior_localization"][
        "median_error_localization_auc"
    ] == 0.9
    assert summary["decision"][
        "confirmation_nonzero_unique_selectable_replicated"
    ] is True
