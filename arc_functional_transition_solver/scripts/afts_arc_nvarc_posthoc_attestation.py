"""Close a transparent post-run attestation for an exposed development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402
from afts_arc.nvarc_posthoc import build_nvarc_posthoc_attestation  # noqa: E402


_FINAL_STEP = re.compile(r"}\s+(\d+)\s*$")


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _load_object(path: Path) -> Mapping[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), field=str(path))


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _submission_counts(
    challenges: Mapping[str, object], submission: Mapping[str, object]
) -> tuple[int, int]:
    if set(challenges) != set(submission):
        raise ValueError("challenge and submission task sets differ")
    queries = 0
    for task_id in sorted(challenges):
        task = _object(challenges[task_id], field=f"challenge[{task_id}]")
        expected = _sequence(task["test"], field="challenge test")
        observed = _sequence(submission[task_id], field="submission task")
        if len(expected) != len(observed):
            raise ValueError("challenge and submission query counts differ")
        queries += len(expected)
    return len(challenges), queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--schedule-audit", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--nvarc-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--expected-nvarc-commit", required=True)
    parser.add_argument("--expected-model-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    run_dir = arguments.run_dir.resolve()
    challenges_path = arguments.challenges.resolve()
    schedule_path = arguments.schedule_audit.resolve()
    runner = arguments.runner.resolve()
    nvarc_root = arguments.nvarc_root.resolve()
    model_root = arguments.model_root.resolve()
    schedule_audit = _load_object(schedule_path)
    if schedule_audit["status"] != "clean":
        raise ValueError("dataset boundary audit did not pass")
    schedule = _object(schedule_audit["schedule"], field="schedule")
    challenges = _load_object(challenges_path)
    final_step = int(schedule["full_batch_count"])
    submission_path = (
        run_dir
        / "checkpoint"
        / f"evaluator_ARC_step_{final_step}"
        / "submission.json"
    )
    submission = _load_object(submission_path)
    task_count, query_count = _submission_counts(challenges, submission)
    stdout = (run_dir / "run.stdout.log").read_text(encoding="utf-8")
    final_lines = [line for line in stdout.splitlines() if line.startswith("{'all':")]
    if len(final_lines) != 1:
        raise ValueError("run stdout must contain exactly one final metric line")
    match = _FINAL_STEP.search(final_lines[0])
    if match is None:
        raise ValueError("final metric line has no observed training step")
    observed_step = int(match.group(1))
    eval_batch_count = sum(
        line.startswith("Processing batch ") for line in stdout.splitlines()
    )

    recorded_nvarc_commit = (
        run_dir / "source_commit.txt"
    ).read_text(encoding="utf-8").strip()
    current_nvarc_commit = _git(nvarc_root, "rev-parse", "HEAD").strip()
    current_model_commit = _git(model_root, "rev-parse", "HEAD").strip()
    if current_nvarc_commit != arguments.expected_nvarc_commit:
        raise ValueError("current NVARC source commit differs from expectation")
    if current_model_commit != arguments.expected_model_commit:
        raise ValueError("current model source commit differs from expectation")
    recorded_nvarc_status = (run_dir / "source_status.txt").read_text(
        encoding="utf-8"
    )
    current_nvarc_status = _git(nvarc_root, "status", "--short")
    model_status = _git(model_root, "status", "--short")
    nvarc_diff = _git(nvarc_root, "diff", "--binary", "--no-ext-diff")
    model_diff = _git(model_root, "diff", "--binary", "--no-ext-diff")
    loss_source = model_root / "models" / "losses.py"
    model_source = model_root / "models" / "recursive_reasoning" / "trm.py"
    artifacts = {
        "all_config": file_sha256(run_dir / "checkpoint" / "all_config.yaml"),
        "challenges": file_sha256(challenges_path),
        "dataset_schedule_audit": file_sha256(schedule_path),
        "initial_checkpoint_manifest": file_sha256(run_dir / "checkpoint.sha256"),
        "output_checkpoint": file_sha256(
            run_dir / "checkpoint" / f"step_{final_step}"
        ),
        "run_stderr": file_sha256(run_dir / "run.stderr.log"),
        "run_stdout": file_sha256(run_dir / "run.stdout.log"),
        "runner": file_sha256(runner),
        "submission": file_sha256(submission_path),
    }
    source = {
        "eval_source_sha256": file_sha256(nvarc_root / "eval-arc-k-10.py"),
        "evaluator_source_sha256": file_sha256(model_root / "evaluators" / "arc.py"),
        "model_source_commit": current_model_commit,
        "model_source_status_sha256": hashlib.sha256(
            model_status.encode("utf-8")
        ).hexdigest(),
        "model_source_tracked_diff_sha256": hashlib.sha256(
            model_diff.encode("utf-8")
        ).hexdigest(),
        "nvarc_source_commit": current_nvarc_commit,
        "nvarc_source_status_sha256": hashlib.sha256(
            current_nvarc_status.encode("utf-8")
        ).hexdigest(),
        "nvarc_source_tracked_diff_sha256": hashlib.sha256(
            nvarc_diff.encode("utf-8")
        ).hexdigest(),
    }
    attestation = build_nvarc_posthoc_attestation(
        cohort_id=arguments.cohort_id,
        completion={
            "eval_batch_count": eval_batch_count,
            "exit_code": int(
                (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip()
            ),
            "expected_training_steps": final_step,
            "observed_training_steps": observed_step,
            "query_count": query_count,
            "task_count": task_count,
        },
        artifacts=artifacts,
        source=source,
        integrity_checks={
            "checkpoint_loss_source_matches_current": file_sha256(
                run_dir / "checkpoint" / "losses.py"
            )
            == file_sha256(loss_source),
            "checkpoint_model_source_matches_current": file_sha256(
                run_dir / "checkpoint" / "trm.py"
            )
            == file_sha256(model_source),
            "nvarc_commit_matches_run_start": recorded_nvarc_commit
            == current_nvarc_commit,
            "nvarc_status_matches_run_start": recorded_nvarc_status
            == current_nvarc_status,
        },
        limitations=(
            "the executed runner did not capture model-source git metadata at run start",
            "evaluator source hashes were captured after completion rather than by the runner",
            "development query gold had been exposed before this candidate freeze",
        ),
    )
    atomic_write_json(output, attestation)
    print(
        json.dumps(
            {"attestation_id": attestation["attestation_id"], "output": str(output)}
        )
    )


if __name__ == "__main__":
    main()
