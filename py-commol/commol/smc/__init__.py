"""The sequential Monte Carlo methods: the Gibbs and Bayesian posteriors and
ABC, and their results (guide: Sequential Monte Carlo)."""

from commol.smc.methods import (
    ABCSMC,
    ABCRejection,
    BayesianPosterior,
    BootstrapCoverageEta,
    GibbsPosterior,
    InformationMatchingEta,
    LossDistance,
    SeriesDistance,
)
from commol.smc.results import (
    GibbsResult,
    SMCDiagnostics,
    SMCResult,
)


__all__ = [
    "ABCRejection",
    "ABCSMC",
    "BayesianPosterior",
    "BootstrapCoverageEta",
    "GibbsPosterior",
    "InformationMatchingEta",
    "LossDistance",
    "SeriesDistance",
    # Results
    "GibbsResult",
    "SMCDiagnostics",
    "SMCResult",
]
