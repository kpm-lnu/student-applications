"""Library components shared by the adaptive-sampling experiments.

Submodules (lazily imported via ``__getattr__``):
  * :mod:`.problems`  - parametric forward PDE/ODE/analytic models
    with scalar QoIs (`ReactionDiffusionProblem`, AMICon Heat/AdvDiff
    in FD and FEM flavours, Branin).
  * :mod:`.surrogates` - `RBFSurrogate` and `KrigingSurrogate` with a
    shared `fit / predict / latent` interface; kernel registry.
  * :mod:`.sampling`  - space-filling (Random/LHS/Halton) and adaptive
    greedy samplers (P-, f-, beta-, MEPE, EIGF).
  * :mod:`.metrics`   - error metrics (NRMSE, R^2, MAE, max errors).
  * :mod:`.experiment` - equal-budget benchmark driver
    (`run_comparison`, `samples_to_tolerance`, `summarise`).
  * :mod:`.fem`       - P1 finite elements on 1D uniform meshes.
  * :mod:`.plotting`  - figure generators for the report.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "1.0.0"

_SYMBOL_MODULES = {
    "Metrics": ".metrics",
    "compute_metrics": ".metrics",
    "KERNELS": ".surrogates",
    "RBFSurrogate": ".surrogates",
    "KrigingSurrogate": ".surrogates",
    "get_kernel": ".surrogates",
    "pairwise_distances": ".surrogates",
    "GreedyContext": ".sampling",
    "GreedySampler": ".sampling",
    "BetaGreedySampler": ".sampling",
    "Sampler": ".sampling",
    "SpaceFillingSampler": ".sampling",
    "SAMPLER_REGISTRY": ".sampling",
    "get_sampler": ".sampling",
    "list_samplers": ".sampling",
    "register_sampler": ".sampling",
    "HeatProblem": ".problems",
    "HeatAMIConProblem": ".problems",
    "AdvDiffAMIConProblem": ".problems",
    "HeatAMICon_FD": ".problems",
    "AdvDiffAMICon_FD": ".problems",
    "BraninProblem": ".problems",
    "Problem": ".problems",
    "ReactionDiffusionProblem": ".problems",
    "ExperimentConfig": ".experiment",
    "make_test_set": ".experiment",
    "run_comparison": ".experiment",
    "samples_to_tolerance": ".experiment",
    "summarise": ".experiment",
}

_SUBMODULES = {"fem", "plotting"}

__all__ = [
    *sorted(_SYMBOL_MODULES),
    *sorted(_SUBMODULES),
    "__version__",
]


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _SYMBOL_MODULES:
        module = import_module(_SYMBOL_MODULES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
