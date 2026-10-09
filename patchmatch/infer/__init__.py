"""Layer 5: the evaluation I/O contract and the solver."""

from .input_reader import EvalInput, Query, load_eval_input
from .output_writer import GroundingRecord, RetrievalRecord, write_predictions
from .solver import Solver, SolverConfig, solve_directory

__all__ = [
    "EvalInput",
    "Query",
    "load_eval_input",
    "GroundingRecord",
    "RetrievalRecord",
    "write_predictions",
    "Solver",
    "SolverConfig",
    "solve_directory",
]
