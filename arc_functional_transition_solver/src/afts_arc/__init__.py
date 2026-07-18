"""Core evidence and evaluation types for the ARC functional-transition solver."""

from ._source_bootstrap import BOOTSTRAP_RUNTIME_SOURCE_FINGERPRINT
from .candidate import CandidateRecord, CandidateStore
from .grid import Grid, GridValidationError, as_grid, grid_key
from .scoring import DatasetScore, TaskScore, score_dataset, score_task
from .task import ARCPair, ARCTask, load_task

__all__ = [
    "ARCPair",
    "ARCTask",
    "BOOTSTRAP_RUNTIME_SOURCE_FINGERPRINT",
    "CandidateRecord",
    "CandidateStore",
    "DatasetScore",
    "Grid",
    "GridValidationError",
    "TaskScore",
    "as_grid",
    "grid_key",
    "load_task",
    "score_dataset",
    "score_task",
]
