"""The local approximations around a fit: Laplace, the sandwich and the
Laplace mixture, and their results (guide: Local Approximations)."""

from commol.local.methods import (
    GibbsTarget,
    Laplace,
    LaplaceMixture,
    LikelihoodTarget,
    Sandwich,
)
from commol.local.results import (
    DeltaBand,
    ImportanceResult,
    LaplaceResult,
    MixtureComponent,
    SandwichResult,
)


__all__ = [
    "GibbsTarget",
    "Laplace",
    "LaplaceMixture",
    "LikelihoodTarget",
    "Sandwich",
    # Results
    "DeltaBand",
    "ImportanceResult",
    "LaplaceResult",
    "MixtureComponent",
    "SandwichResult",
]
