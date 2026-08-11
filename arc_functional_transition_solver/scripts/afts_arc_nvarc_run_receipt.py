"""Close an immutable receipt for a completed reference-scale NVARC run."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402
from afts_arc.nvarc_run import (  # noqa: E402
    NVARC_REFERENCE_CONFIG,
    build_nvarc_run_receipt,
)


_NP_METRIC = re.compile(r"'([^']+)': np\.float32\(([-+0-9.eE]+)\)")
_ARC_METRIC = re.compile(r"'(ARC/pass@[^']+)': ([-+0-9.eE]+)")
_FINAL_STEP = re.compile(r"}\s+(\d+)\s*$")


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _load_json_object(path: Path) -> Mapping[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), field=str(path))


def _load_yaml_object(path: Path) -> Mapping[str, object]:
    return _object(yaml.safe_load(path.read_text(encoding="utf-8")), field=str(path))


def _configuration(config: Mapping[str, object]) -> dict[str, object]:
    arch = _object(config["arch"], field="all_config.arch")
    loss = _object(arch["loss"], field="all_config.arch.loss")
    return {
        "architecture": arch["name"],
        "beta1": config["beta1"],
        "beta2": config["beta2"],
        "checkpoint_every_eval": config["checkpoint_every_eval"],
        "ema": config["ema"],
        "ema_rate": config["ema_rate"],
        "epochs": config["epochs"],
        "eval_interval": config["eval_interval"],
        "forward_dtype": arch["forward_dtype"],
        "freeze_weights": config["freeze_weights"],
        "global_batch_size": config["global_batch_size"],
        "h_cycles": arch["H_cycles"],
        "halt_max_steps": arch["halt_max_steps"],
        "hidden_size": arch["hidden_size"],
        "l_cycles": arch["L_cycles"],
        "l_layers": arch["L_layers"],
        "learning_rate": config["lr"],
        "learning_rate_min_ratio": config["lr_min_ratio"],
        "learning_rate_warmup_steps": config["lr_warmup_steps"],
        "loss_head": loss["name"],
        "loss_type": loss["loss_type"],
        "min_eval_interval": config["min_eval_interval"],
        "num_heads": arch["num_heads"],
        "puzzle_embedding_length": arch["puzzle_emb_len"],
        "puzzle_embedding_learning_rate": config["puzzle_emb_lr"],
        "puzzle_embedding_weight_decay": config["puzzle_emb_weight_decay"],
        "seed": config["seed"],
        "weight_decay": config["weight_decay"],
    }


def _submission_counts(
    *, challenges: Mapping[str, object], submission: Mapping[str, object]
) -> tuple[int, int, int]:
    if set(challenges) != set(submission):
        raise ValueError("challenge and NVARC submission task sets differ")
    attempt_fields = {f"attempt_{rank}" for rank in range(1, 11)}
    query_count = 0
    for task_id in sorted(challenges):
        task = _object(challenges[task_id], field=f"challenge[{task_id}]")
        queries = _sequence(task["test"], field=f"challenge[{task_id}].test")
        submitted = _sequence(submission[task_id], field=f"submission[{task_id}]")
        if len(queries) != len(submitted):
            raise ValueError(f"submission query count differs for {task_id}")
        for query_index, raw_attempts in enumerate(submitted):
            attempts = _object(
                raw_attempts, field=f"submission[{task_id}][{query_index}]"
            )
            if set(attempts) != attempt_fields:
                raise ValueError(
                    "submission must contain exactly attempt_1..attempt_10"
                )
        query_count += len(queries)
    return len(challenges), query_count, 10


def _metrics_and_steps(stdout: str) -> tuple[dict[str, float], int, int]:
    metric_lines = [line for line in stdout.splitlines() if line.startswith("{'all':")]
    if len(metric_lines) != 1:
        raise ValueError("NVARC stdout must contain exactly one final metric record")
    metric_line = metric_lines[0]
    metrics = {
        f"all/{name}": float(value) for name, value in _NP_METRIC.findall(metric_line)
    }
    metrics.update(
        {name: float(value) for name, value in _ARC_METRIC.findall(metric_line)}
    )
    if not metrics:
        raise ValueError("NVARC final metric record contains no parseable metrics")
    final_step_match = _FINAL_STEP.search(metric_line)
    if final_step_match is None:
        raise ValueError("NVARC final metric record has no training-step count")
    eval_batch_count = sum(
        line.startswith("Processing batch ") for line in stdout.splitlines()
    )
    return metrics, int(final_step_match.group(1)), eval_batch_count


def _artifact_hashes(
    *,
    run_dir: Path,
    data_dir: Path,
    challenges: Path,
    cohort_manifest: Path,
    initial_checkpoint: Path,
    runner: Path,
    schedule_audit: Path,
    final_step: int,
) -> dict[str, str]:
    paths = {
        "all_config": run_dir / "checkpoint" / "all_config.yaml",
        "challenges": challenges,
        "cohort_manifest": cohort_manifest,
        "dataset_recursive_manifest": run_dir / "dataset_recursive.sha256",
        "executed_source_manifest": run_dir / "executed_source.sha256",
        "initial_checkpoint": initial_checkpoint,
        "output_checkpoint": run_dir / "checkpoint" / f"step_{final_step}",
        "run_stderr": run_dir / "run.stderr.log",
        "run_stdout": run_dir / "run.stdout.log",
        "runner": runner,
        "schedule_audit": schedule_audit,
        "submission": run_dir
        / "checkpoint"
        / f"evaluator_ARC_step_{final_step}"
        / "submission.json",
    }
    for path in data_dir.rglob("*"):
        if path.is_file():
            paths[f"dataset:{path.relative_to(data_dir).as_posix()}"] = path
    if not paths:
        raise ValueError("run receipt has no artifacts")
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--cohort-manifest", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--schedule-audit", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-model-source-commit", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    run_dir = Path(arguments.run_dir).resolve()
    data_dir = Path(arguments.data_dir).resolve()
    challenges_path = Path(arguments.challenges).resolve()
    cohort_manifest_path = Path(arguments.cohort_manifest).resolve()
    initial_checkpoint = Path(arguments.initial_checkpoint).resolve()
    runner = Path(arguments.runner).resolve()
    schedule_audit_path = Path(arguments.schedule_audit).resolve()
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    exit_code = int((run_dir / "exit_code.txt").read_text(encoding="utf-8").strip())
    source_commit = (run_dir / "source_commit.txt").read_text(encoding="utf-8").strip()
    model_source_commit = (
        (run_dir / "model_source_commit.txt").read_text(encoding="utf-8").strip()
    )
    if source_commit != arguments.expected_source_commit:
        raise ValueError("NVARC source commit differs from the frozen contract")
    if model_source_commit != arguments.expected_model_source_commit:
        raise ValueError("TRM source commit differs from the frozen contract")
    config_path = run_dir / "checkpoint" / "all_config.yaml"
    schedule_audit = _load_json_object(schedule_audit_path)
    if schedule_audit["schema"] != "afts.nvarc-dataset-boundary-audit/v1":
        raise ValueError("unsupported NVARC schedule-audit schema")
    if schedule_audit["status"] != "clean":
        raise ValueError("NVARC dataset boundary audit did not pass")
    schedule = _object(schedule_audit["schedule"], field="schedule_audit.schedule")
    counts = _object(schedule_audit["counts"], field="schedule_audit.counts")
    final_step = int(schedule["full_batch_count"])
    submission_path = (
        run_dir
        / "checkpoint"
        / f"evaluator_ARC_step_{final_step}"
        / "submission.json"
    )
    challenges = _load_json_object(challenges_path)
    submission = _load_json_object(submission_path)
    task_count, query_count, attempts_per_query = _submission_counts(
        challenges=challenges, submission=submission
    )
    stdout = (run_dir / "run.stdout.log").read_text(encoding="utf-8")
    metrics, observed_steps, eval_batch_count = _metrics_and_steps(stdout)
    artifacts = _artifact_hashes(
        run_dir=run_dir,
        data_dir=data_dir,
        challenges=challenges_path,
        cohort_manifest=cohort_manifest_path,
        initial_checkpoint=initial_checkpoint,
        runner=runner,
        schedule_audit=schedule_audit_path,
        final_step=final_step,
    )
    source = {
        "model_source_commit": model_source_commit,
        "model_source_tracked_diff_sha256": file_sha256(
            run_dir / "model_source_tracked.diff"
        ),
        "nvarc_source_commit": source_commit,
        "nvarc_source_tracked_diff_sha256": file_sha256(
            run_dir / "source_tracked.diff"
        ),
    }
    runtime = {
        "cuda_visible_devices": (run_dir / "cuda_visible_devices.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "finished_at": (run_dir / "finished_at.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "pip_freeze_sha256": file_sha256(run_dir / "pip_freeze.txt"),
        "python": (run_dir / "python_version.txt").read_text(encoding="utf-8").strip(),
        "started_at": (run_dir / "started_at.txt").read_text(encoding="utf-8").strip(),
    }
    receipt = build_nvarc_run_receipt(
        cohort_id=arguments.cohort_id,
        configuration=_configuration(_load_yaml_object(config_path)),
        completion={
            "attempts_per_query": attempts_per_query,
            "dropped_final_batch_size": schedule["dropped_final_batch_size"],
            "eval_batch_count": eval_batch_count,
            "exact_replayed_training_steps": schedule["full_batch_count"],
            "exit_code": exit_code,
            "group_order_size": schedule["group_order_size"],
            "groups_consumed": schedule["groups_consumed"],
            "metadata_estimated_training_steps": schedule[
                "metadata_estimated_training_steps"
            ],
            "observed_training_steps": observed_steps,
            "query_count": query_count,
            "task_count": task_count,
        },
        expected_completion={
            "attempts_per_query": 10,
            "dropped_final_batch_size": schedule["dropped_final_batch_size"],
            "eval_batch_count": math.ceil(
                int(counts["test_input_count"])
                / int(NVARC_REFERENCE_CONFIG["global_batch_size"])
            ),
            "exact_replayed_training_steps": final_step,
            "exit_code": 0,
            "group_order_size": schedule["group_order_size"],
            "groups_consumed": schedule["groups_consumed"],
            "metadata_estimated_training_steps": schedule[
                "metadata_estimated_training_steps"
            ],
            "observed_training_steps": final_step,
            "query_count": query_count,
            "task_count": task_count,
        },
        artifacts=artifacts,
        source=source,
        runtime=runtime,
        upstream_metrics=metrics,
    )
    atomic_write_json(output, receipt)
    print(json.dumps({"output": str(output), "receipt_id": receipt["receipt_id"]}))


if __name__ == "__main__":
    main()
