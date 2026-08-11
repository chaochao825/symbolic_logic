"""Close a query-boundary-aware receipt for a reference-scale NVARC run."""

from __future__ import annotations

import math
from collections.abc import Mapping

from afts_arc.experiment_safety import canonical_sha256


NVARC_RUN_RECEIPT_SCHEMA = "afts.nvarc-trm-run-receipt/v2"
NVARC_EXPECTED_METRIC_NAMES = {
    "ARC/pass@1",
    "ARC/pass@10",
    "ARC/pass@100",
    "ARC/pass@1000",
    "ARC/pass@2",
    "ARC/pass@5",
    "all/accuracy",
    "all/exact_accuracy",
    "all/lm_loss",
    "all/q_halt_accuracy",
    "all/q_halt_loss",
    "all/steps",
}
NVARC_REFERENCE_CONFIG: dict[str, object] = {
    "architecture": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
    "beta1": 0.9,
    "beta2": 0.95,
    "checkpoint_every_eval": True,
    "ema": True,
    "ema_rate": 0.999,
    "epochs": 2000,
    "eval_interval": 2000,
    "forward_dtype": "bfloat16",
    "freeze_weights": False,
    "global_batch_size": 128,
    "h_cycles": 4,
    "halt_max_steps": 10,
    "hidden_size": 512,
    "l_cycles": 4,
    "l_layers": 2,
    "learning_rate": 0.0001,
    "learning_rate_min_ratio": 1.0,
    "learning_rate_warmup_steps": 200,
    "loss_head": "losses@ACTLossHead",
    "loss_type": "stablemax_cross_entropy",
    "min_eval_interval": 0,
    "num_heads": 8,
    "puzzle_embedding_length": 16,
    "puzzle_embedding_learning_rate": 0.01,
    "puzzle_embedding_weight_decay": 0.1,
    "seed": 0,
    "weight_decay": 0.1,
}


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _exact_fields(
    value: Mapping[str, object], *, expected: set[str], field: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise ValueError(f"{field} fields differ: missing={missing}, unknown={unknown}")


def build_nvarc_run_receipt(
    *,
    cohort_id: str,
    configuration: Mapping[str, object],
    completion: Mapping[str, object],
    expected_completion: Mapping[str, object],
    artifacts: Mapping[str, object],
    source: Mapping[str, object],
    runtime: Mapping[str, object],
    upstream_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Validate the frozen run contract without claiming query-blind evaluation."""

    if not cohort_id:
        raise ValueError("cohort_id must not be empty")
    if dict(configuration) != NVARC_REFERENCE_CONFIG:
        raise ValueError(
            "NVARC configuration differs from the frozen reference contract"
        )
    completion = _object(completion, field="completion")
    _exact_fields(
        completion,
        expected={
            "attempts_per_query",
            "dropped_final_batch_size",
            "eval_batch_count",
            "exact_replayed_training_steps",
            "exit_code",
            "group_order_size",
            "groups_consumed",
            "metadata_estimated_training_steps",
            "observed_training_steps",
            "query_count",
            "task_count",
        },
        field="completion",
    )
    expected_completion = _object(
        expected_completion, field="expected_completion"
    )
    _exact_fields(
        expected_completion,
        expected=set(completion),
        field="expected_completion",
    )
    if dict(completion) != expected_completion:
        raise ValueError("NVARC run did not complete the frozen workload")
    artifacts = _object(artifacts, field="artifacts")
    source = _object(source, field="source")
    runtime = _object(runtime, field="runtime")
    upstream_metrics = _object(upstream_metrics, field="upstream_metrics")
    if not artifacts or not source or not runtime or not upstream_metrics:
        raise ValueError("run receipt sections must not be empty")
    if not all(
        isinstance(digest, str) and len(digest) == 64 for digest in artifacts.values()
    ):
        raise ValueError("artifact identities must be SHA-256 digests")
    if set(upstream_metrics) != NVARC_EXPECTED_METRIC_NAMES:
        raise ValueError("upstream metric set differs from the frozen evaluator")
    for name, value in upstream_metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"upstream metric {name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"upstream metric {name} is not finite")
    if float(upstream_metrics["all/steps"]) != 10.0:
        raise ValueError("upstream evaluator did not execute ten inference steps")
    content: dict[str, object] = {
        "artifacts": dict(sorted(artifacts.items())),
        "cohort_id": cohort_id,
        "completion": dict(completion),
        "configuration": dict(configuration),
        "controller_training_started": False,
        "query_label_boundary": {
            "candidate_model_updates_use_query_gold": False,
            "candidate_ordering_uses_query_gold": False,
            "project_solution_read_before_submission_freeze": False,
            "public_evaluation_read": False,
            "upstream_evaluator_reads_query_gold_for_metrics": True,
        },
        "runtime": dict(runtime),
        "schema": NVARC_RUN_RECEIPT_SCHEMA,
        "scientific_scope": "static reference-scale candidate anchor",
        "source": dict(source),
        "upstream_metrics": dict(sorted(upstream_metrics.items())),
    }
    return {"receipt_id": canonical_sha256(content), **content}
