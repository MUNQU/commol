"""The local approximations around a fit: Laplace, the sandwich and the
Laplace mixture (guide: Local Approximations)."""

from commol.local.methods import (
    GibbsTarget,
    Laplace,
    LaplaceMixture,
    LikelihoodTarget,
    Sandwich,
)
from commol.local import runners as _runners  # noqa: F401  (registers the runners)

__all__ = [
    "GibbsTarget",
    "Laplace",
    "LaplaceMixture",
    "LikelihoodTarget",
    "Sandwich",
]
