"""The local approximations around a fit: Laplace, the sandwich and the
Laplace mixture (guide: Local Approximations)."""

from commol.local.methods import (
    GibbsTarget,
    Laplace,
    LaplaceMixture,
    LikelihoodTarget,
    Sandwich,
)

__all__ = [
    "GibbsTarget",
    "Laplace",
    "LaplaceMixture",
    "LikelihoodTarget",
    "Sandwich",
]
