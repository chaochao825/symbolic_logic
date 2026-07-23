"""Ordinary (non-evidence) CLI for the hybrid functional ARC solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .blind import BlindTask
from .hybrid import (
    BudgetVector,
    CodeModelProvider,
    DiffLogicHardProvider,
    DslProgramProvider,
    FunctionalRouterConfig,
    FunctionalRouterSolver,
    MaskedDiffusionProvider,
    OnlineControlConfig,
    OnlineFunctionalRouterSolver,
    SparseCAProvider,
)
from .search import SearchConfig
from .task import load_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Route an ARC task through DSL, sparse CA/D4/bgpad, optional external "
            "sources, demo-exact/MDL/hard verification, and residual repair."
        )
    )
    parser.add_argument("task", help="official-format ARC JSON task")
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument("--max-depth", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-instruction-options", type=int, default=64)
    parser.add_argument("--max-exact-programs", type=int, default=128)
    parser.add_argument("--max-selected", type=int, default=2)
    parser.add_argument("--max-repair-parents", type=int, default=8)
    parser.add_argument("--minimum-repair-agreement", type=float, default=0.5)
    parser.add_argument(
        "--controller",
        choices=("online", "static"),
        default="online",
        help="online recompiles legal actions after every residual; static is the v1 run-all baseline",
    )
    parser.add_argument("--compute-units", type=int, default=96)
    parser.add_argument("--controller-steps", type=int, default=16)
    parser.add_argument("--provider-calls", type=int, default=8)
    parser.add_argument("--repair-attempts", type=int, default=8)
    parser.add_argument("--candidate-slots", type=int, default=80)
    parser.add_argument("--provider-batch-size", type=int, default=4)
    parser.add_argument("--localized-residual-fraction", type=float, default=0.35)
    parser.add_argument(
        "--no-ca",
        action="store_true",
        help="disable the optional repository-root NumPy CA backend",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    repository_root = Path(__file__).resolve().parents[3]
    root_source = repository_root / "src"
    if (root_source / "arc_ca.py").is_file() and str(root_source) not in sys.path:
        sys.path.insert(0, str(root_source))
    args = build_parser().parse_args(argv)
    source_task = load_task(args.task)
    blind = BlindTask.from_task(source_task)
    providers = [
        DslProgramProvider(
            SearchConfig(
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                max_instruction_options=args.max_instruction_options,
                max_exact_programs=args.max_exact_programs,
            )
        )
    ]
    if not args.no_ca:
        providers.append(SparseCAProvider())
    providers.extend(
        (
            CodeModelProvider(),
            DiffLogicHardProvider(),
            MaskedDiffusionProvider(),
        )
    )
    if args.controller == "online":
        solver = OnlineFunctionalRouterSolver(
            providers=providers,
            config=OnlineControlConfig(
                budget_limit=BudgetVector(
                    compute_units=args.compute_units,
                    controller_steps=args.controller_steps,
                    provider_calls=args.provider_calls,
                    repair_attempts=args.repair_attempts,
                    candidate_slots=args.candidate_slots,
                ),
                provider_batch_size=args.provider_batch_size,
                max_selected_hypotheses=args.max_selected,
                minimum_repair_agreement=args.minimum_repair_agreement,
                localized_residual_fraction=args.localized_residual_fraction,
            ),
        )
    else:
        solver = FunctionalRouterSolver(
            providers=providers,
            config=FunctionalRouterConfig(
                max_selected_hypotheses=args.max_selected,
                max_repair_parents=args.max_repair_parents,
                minimum_repair_agreement=args.minimum_repair_agreement,
            ),
        )
    report = solver.solve(blind).to_json_dict(task_id=source_task.task_id)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
