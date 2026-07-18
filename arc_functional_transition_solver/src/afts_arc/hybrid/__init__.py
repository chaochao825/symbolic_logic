"""Hybrid functional-transition solver for ARC."""

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
from .router import RouteAssignment, RouteDecision, TaskFeatures, extract_task_features, route_task
from .types import (
    CandidateEvaluation,
    CandidateHypothesis,
    CandidateProvider,
    ProviderResult,
    RepairReceipt,
)
from .verification import evaluate_hypotheses, evaluate_hypothesis, rank_verified

__all__ = [
    "CandidateEvaluation",
    "CandidateHypothesis",
    "CandidateProvider",
    "CodeModelProvider",
    "DiffLogicHardProvider",
    "DslProgramProvider",
    "FunctionalRouterConfig",
    "FunctionalRouterSolver",
    "MaskedDiffusionProvider",
    "OptionalCallbackProvider",
    "ProviderResult",
    "RepairReceipt",
    "RouteAssignment",
    "RouteDecision",
    "SolveReport",
    "SparseCAProvider",
    "TaskFeatures",
    "evaluate_hypotheses",
    "evaluate_hypothesis",
    "extract_task_features",
    "generate_repairs",
    "global_color_map_repair",
    "local_transition_repair",
    "rank_verified",
    "route_task",
]
