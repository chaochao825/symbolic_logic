"""Prepare and score a query-blind static visual-provider experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    resolved = str(source_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


_bootstrap()

from afts_arc.visual_provider_gate import (  # noqa: E402
    aggregate_visual_provider_runs,
    audit_posterior_consensus_bridge,
    audit_posterior_error_localization,
    freeze_and_score_provider,
    freeze_provider_predictions,
    prepare_query_blind_cohort,
)
from afts_arc.visual_structure_bridge import (  # noqa: E402
    freeze_visual_structure_bridge,
    score_visual_structure_bridge,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("arc2_training_dir", type=Path)
    prepare.add_argument("project_root", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--arc1-task-dir", action="append", type=Path, required=True)
    prepare.add_argument(
        "--exposure-registry", action="append", type=Path, required=True
    )
    prepare.add_argument("--seed", required=True)
    prepare.add_argument("--limit", type=int, required=True)
    prepare.add_argument("--arc1-source-commit", required=True)
    prepare.add_argument("--arc2-source-commit", required=True)

    score = commands.add_parser("score")
    score.add_argument("cohort_manifest", type=Path)
    score.add_argument("gold_training_dir", type=Path)
    score.add_argument("baseline_summary", type=Path)
    score.add_argument("provider_contract", type=Path)
    score.add_argument("output_dir", type=Path)
    score.add_argument("--prediction-root", action="append", type=Path, required=True)
    score.add_argument("--object-provider-summary", type=Path)
    score.add_argument("--pilot-unique-gate", type=int, default=1)
    score.add_argument(
        "--invalid-candidate-policy",
        choices=("error", "reject"),
        required=True,
    )

    freeze = commands.add_parser("freeze")
    freeze.add_argument("cohort_manifest", type=Path)
    freeze.add_argument("provider_contract", type=Path)
    freeze.add_argument("output_path", type=Path)
    freeze.add_argument("--prediction-root", action="append", type=Path, required=True)
    freeze.add_argument(
        "--invalid-candidate-policy",
        choices=("error", "reject"),
        required=True,
    )

    posterior = commands.add_parser("posterior-audit")
    posterior.add_argument("cohort_manifest", type=Path)
    posterior.add_argument("gold_training_dir", type=Path)
    posterior.add_argument("frozen_predictions", type=Path)
    posterior.add_argument("output_dir", type=Path)
    posterior.add_argument("--prediction-root", action="append", type=Path, required=True)
    posterior.add_argument(
        "--invalid-candidate-policy",
        choices=("error", "reject"),
        required=True,
    )
    posterior.add_argument(
        "--mask-fraction", action="append", type=float, required=True
    )
    posterior.add_argument(
        "--disagreement-threshold", action="append", type=float, required=True
    )

    bridge = commands.add_parser("posterior-bridge")
    bridge.add_argument("cohort_manifest", type=Path)
    bridge.add_argument("gold_training_dir", type=Path)
    bridge.add_argument("frozen_predictions", type=Path)
    bridge.add_argument("visual_score_summary", type=Path)
    bridge.add_argument("output_dir", type=Path)
    bridge.add_argument("--prediction-root", action="append", type=Path, required=True)
    bridge.add_argument(
        "--invalid-candidate-policy",
        choices=("error", "reject"),
        required=True,
    )

    structure_freeze = commands.add_parser("structure-freeze")
    structure_freeze.add_argument("cohort_manifest", type=Path)
    structure_freeze.add_argument("blind_task_dir", type=Path)
    structure_freeze.add_argument("frozen_predictions", type=Path)
    structure_freeze.add_argument("output_path", type=Path)
    structure_freeze.add_argument(
        "--prediction-root", action="append", type=Path, required=True
    )
    structure_freeze.add_argument(
        "--invalid-candidate-policy",
        choices=("error", "reject"),
        required=True,
    )

    structure_score = commands.add_parser("structure-score")
    structure_score.add_argument("cohort_manifest", type=Path)
    structure_score.add_argument("gold_task_dir", type=Path)
    structure_score.add_argument("candidate_artifact", type=Path)
    structure_score.add_argument("visual_score_summary", type=Path)
    structure_score.add_argument("output_path", type=Path)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("output_dir", type=Path)
    aggregate.add_argument("--score-summary", action="append", type=Path, required=True)
    aggregate.add_argument(
        "--posterior-summary", action="append", type=Path, required=True
    )
    aggregate.add_argument("--confirmation-score-index", type=int, required=True)
    aggregate.add_argument("--bridge-summary", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_query_blind_cohort(
            arc2_training_dir=args.arc2_training_dir,
            arc1_task_dirs=tuple(args.arc1_task_dir),
            project_root=args.project_root,
            exposure_registry_paths=tuple(args.exposure_registry),
            output_dir=args.output_dir,
            seed=args.seed,
            limit=args.limit,
            arc1_source_commit=args.arc1_source_commit,
            arc2_source_commit=args.arc2_source_commit,
        )
    elif args.command == "score":
        provider_contract = json.loads(args.provider_contract.read_text(encoding="utf-8"))
        if not isinstance(provider_contract, dict):
            raise RuntimeError("provider contract must be a JSON object")
        result = freeze_and_score_provider(
            cohort_manifest=args.cohort_manifest,
            gold_training_dir=args.gold_training_dir,
            prediction_roots=tuple(args.prediction_root),
            baseline_summary=args.baseline_summary,
            object_provider_summary=args.object_provider_summary,
            provider_contract=provider_contract,
            output_dir=args.output_dir,
            pilot_unique_gate=args.pilot_unique_gate,
            invalid_candidate_policy=args.invalid_candidate_policy,
        )
    elif args.command == "freeze":
        provider_contract = json.loads(args.provider_contract.read_text(encoding="utf-8"))
        if not isinstance(provider_contract, dict):
            raise RuntimeError("provider contract must be a JSON object")
        result = freeze_provider_predictions(
            cohort_manifest=args.cohort_manifest,
            prediction_roots=tuple(args.prediction_root),
            provider_contract=provider_contract,
            output_path=args.output_path,
            invalid_candidate_policy=args.invalid_candidate_policy,
        )
    elif args.command == "posterior-audit":
        result = audit_posterior_error_localization(
            cohort_manifest=args.cohort_manifest,
            gold_training_dir=args.gold_training_dir,
            prediction_roots=tuple(args.prediction_root),
            frozen_predictions=args.frozen_predictions,
            output_dir=args.output_dir,
            invalid_candidate_policy=args.invalid_candidate_policy,
            mask_fractions=tuple(args.mask_fraction),
            disagreement_thresholds=tuple(args.disagreement_threshold),
        )
    elif args.command == "posterior-bridge":
        result = audit_posterior_consensus_bridge(
            cohort_manifest=args.cohort_manifest,
            gold_training_dir=args.gold_training_dir,
            prediction_roots=tuple(args.prediction_root),
            frozen_predictions=args.frozen_predictions,
            visual_score_summary=args.visual_score_summary,
            output_dir=args.output_dir,
            invalid_candidate_policy=args.invalid_candidate_policy,
        )
    elif args.command == "structure-freeze":
        result = freeze_visual_structure_bridge(
            cohort_manifest=args.cohort_manifest,
            blind_task_dir=args.blind_task_dir,
            prediction_roots=tuple(args.prediction_root),
            frozen_predictions=args.frozen_predictions,
            output_path=args.output_path,
            invalid_candidate_policy=args.invalid_candidate_policy,
        )
    elif args.command == "structure-score":
        result = score_visual_structure_bridge(
            cohort_manifest=args.cohort_manifest,
            gold_task_dir=args.gold_task_dir,
            candidate_artifact=args.candidate_artifact,
            visual_score_summary=args.visual_score_summary,
            output_path=args.output_path,
        )
    else:
        result = aggregate_visual_provider_runs(
            score_summaries=tuple(args.score_summary),
            posterior_summaries=tuple(args.posterior_summary),
            output_dir=args.output_dir,
            confirmation_score_index=args.confirmation_score_index,
            bridge_summary=args.bridge_summary,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
