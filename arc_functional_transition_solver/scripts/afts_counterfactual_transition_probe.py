"""Run a query-blind v4 search-policy probe for one named challenge task."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.counterfactual_transition import (  # noqa: E402
    COUNTERFACTUAL_STRATEGIES,
    synthesize_counterfactual_transition_arms,
)
from afts_arc.object_graph_rewrite_gate import (  # noqa: E402
    blind_task_from_challenge,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=COUNTERFACTUAL_STRATEGIES,
        default=COUNTERFACTUAL_STRATEGIES,
    )
    parser.add_argument("--max-first-stage-trials", type=int, default=2_000)
    parser.add_argument("--max-parents", type=int, default=4)
    parser.add_argument("--max-transition-trials", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=32)
    args = parser.parse_args()

    challenges = json.loads(args.challenges.read_text(encoding="utf-8"))
    if not isinstance(challenges, dict):
        raise TypeError("challenge file must contain a JSON object")
    if args.task_id not in challenges:
        raise ValueError("probe task ID is absent from the challenge file")
    raw_task = challenges[args.task_id]
    if not isinstance(raw_task, dict):
        raise TypeError("probe challenge must be a JSON object")

    start = time.monotonic()
    result = synthesize_counterfactual_transition_arms(
        blind_task_from_challenge(raw_task),
        strategies=tuple(args.strategies),
        max_first_stage_trials=args.max_first_stage_trials,
        max_parents=args.max_parents,
        max_transition_trials=args.max_transition_trials,
        max_candidates=args.max_candidates,
    )
    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "elapsed_seconds": time.monotonic() - start,
                "query_gold_read": False,
                "parent_count": result.parent_count,
                "certificate_nodes": [
                    certificate.node_id for certificate in result.certificates
                ],
                "arms": [
                    {
                        "strategy": arm.strategy,
                        "program_trials": arm.program_trials,
                        "padding_trials": arm.padding_trials,
                        "candidate_count": len(arm.candidates),
                        "improving_trial_count": arm.improving_trial_count,
                    }
                    for arm in result.arms
                ],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
