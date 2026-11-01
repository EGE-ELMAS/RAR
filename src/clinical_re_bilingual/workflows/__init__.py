"""Executable experiment workflows."""

from .inference import predict_dataset, run_inference_workflow
from .rar import RARWorkflowResult, run_rar_workflow


def run_baseline_workflow(**kwargs):
    """Load the supervised training stack only when a baseline run is requested."""
    from .baselines import run_baseline_workflow as run

    return run(**kwargs)


__all__ = [
    "RARWorkflowResult",
    "predict_dataset",
    "run_baseline_workflow",
    "run_inference_workflow",
    "run_rar_workflow",
]
