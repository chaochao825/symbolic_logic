"""Hybrid functional-transition solver for ARC."""

from .control import (
    ActionResult,
    Blackboard,
    BudgetLedger,
    BudgetVector,
    ControlAction,
    FixedSchedulePolicy,
    FrozenCandidatePoolProvider,
    ResidualActionCompiler,
    ResidualCompilerConfig,
    ResidualFirstPolicy,
    ResidualSignal,
)
from .online import OnlineControlConfig, OnlineFunctionalRouterSolver, OnlineSolveReport
from .control_evaluation import (
    AggregateControlMetrics,
    TaskControlMetrics,
    aggregate_control_metrics,
    evaluate_online_report_with_oracle,
)
from .orchestrator import FunctionalRouterConfig, FunctionalRouterSolver, SolveReport
from .providers import (
    CodeModelProvider,
    DiffLogicHardProvider,
    DslProgramProvider,
    MaskedDiffusionProvider,
    OptionalCallbackProvider,
    SparseCAProvider,
)
from .repair import generate_repairs, global_color_map_repair, local_transition_repair
from .router import (
    RouteAssignment,
    RouteDecision,
    TaskFeatures,
    extract_task_features,
    route_task,
)
from .types import (
    CandidateEvaluation,
    CandidateHypothesis,
    CandidateProvider,
    ProviderResult,
    RepairReceipt,
)
from .verification import evaluate_hypotheses, evaluate_hypothesis, rank_verified

__all__ = [
    "ActionResult",
    "AggregateControlMetrics",
    "Blackboard",
    "BudgetLedger",
    "BudgetVector",
    "CandidateEvaluation",
    "CandidateHypothesis",
    "CandidateProvider",
    "CodeModelProvider",
    "DiffLogicHardProvider",
    "DslProgramProvider",
    "ControlAction",
    "FixedSchedulePolicy",
    "FunctionalRouterConfig",
    "FunctionalRouterSolver",
    "FrozenCandidatePoolProvider",
    "MaskedDiffusionProvider",
    "OnlineControlConfig",
    "OnlineFunctionalRouterSolver",
    "OnlineSolveReport",
    "OptionalCallbackProvider",
    "ProviderResult",
    "RepairReceipt",
    "RouteAssignment",
    "RouteDecision",
    "ResidualActionCompiler",
    "ResidualCompilerConfig",
    "ResidualFirstPolicy",
    "ResidualSignal",
    "SolveReport",
    "SparseCAProvider",
    "TaskFeatures",
    "TaskControlMetrics",
    "aggregate_control_metrics",
    "evaluate_hypotheses",
    "evaluate_hypothesis",
    "evaluate_online_report_with_oracle",
    "extract_task_features",
    "generate_repairs",
    "global_color_map_repair",
    "local_transition_repair",
    "rank_verified",
    "route_task",
]
