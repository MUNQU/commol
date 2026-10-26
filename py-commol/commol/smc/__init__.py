"""The sequential Monte Carlo methods: the Gibbs and Bayesian posteriors and
ABC (guide: Sequential Monte Carlo)."""

from commol.smc.methods import (
    ABCRejection,
    ABCSMC,
    BayesianPosterior,
    BootstrapCoverageEta,
    GibbsPosterior,
    InformationMatchingEta,
    LossDistance,
    SeriesDistance,
)
from commol.smc import runners as _runners  # noqa: F401  (registers the runners)

__all__ = [
    "ABCRejection",
    "ABCSMC",
    "BayesianPosterior",
    "BootstrapCoverageEta",
    "GibbsPosterior",
    "InformationMatchingEta",
    "LossDistance",
    "SeriesDistance",
]
