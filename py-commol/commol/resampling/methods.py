"""The resampling methods: the Bayesian, residual and wild block bootstraps,
the block jackknife and the parametric bootstrap, their settings and their
results.

Each fits the model again to reweighted or resampled data, starting from the
fit, with no prior. Blocks of consecutive positions, across every series at
once, keep the dependence of the data inside each block. Every method but the
parametric bootstrap needs no noise model; the parametric bootstrap draws the
data from the observation models.
"""

from typing import Literal

from pydantic import Field

from commol._calibration.method import UncertaintyMethod
from commol._data.ranges import RangeChecked
from commol.resampling.results import (
    BootstrapResult,
    JackknifeResult,
)
from commol.resampling.runners import run_block_jackknife, run_bootstrap


class BayesianBootstrap(UncertaintyMethod[BootstrapResult], RangeChecked):
    """
    The Bayesian (weighted) bootstrap by blocks.

    Each replicate draws one Dirichlet(1, …, 1) weight per block, scaled to
    mean 1, multiplies the weight of every observation of the block by it,
    and fits. With a loss it is the loss-likelihood bootstrap of Lyddon,
    Holmes and Walker (2019); with one position per block, the Bayesian
    bootstrap of Rubin (1981) and the weighted likelihood bootstrap of Newton
    and Raftery (1994). The observed values never change.

    Attributes
    ----------
    kind : "bayesian"
        The bootstrap, which the engine reads.
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block.
    starts : int
        Starts of the fit and of each replicate (default 1): the fit, then
        rows of a Latin hypercube over the bounds.
    """

    _runner = staticmethod(run_bootstrap)

    kind: Literal["bayesian"] = Field(default="bayesian", description="The bootstrap.")
    replicates: int = Field(description="Number of replicates.")
    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    starts: int = Field(default=1, description="Starts per fit.")


class ParametricBootstrap(UncertaintyMethod[BootstrapResult], RangeChecked):
    """
    The parametric bootstrap of the problem's observation models.

    Each replicate draws every observed value from the model of its series
    at the fit, around the fitted prediction and with the fitted noise
    parameters, and fits the new data. It needs a problem with
    ``observation_models``; the observations of a replicate are independent
    given the fit, so it assumes the models hold.

    Attributes
    ----------
    kind : "parametric"
        The bootstrap, which the engine reads.
    replicates : int
        Number of replicates, at least 1.
    starts : int
        Starts of the fit and of each replicate (default 1): the fit, then
        rows of a Latin hypercube over the bounds.
    """

    _runner = staticmethod(run_bootstrap)

    kind: Literal["parametric"] = Field(
        default="parametric", description="The bootstrap."
    )
    replicates: int = Field(description="Number of replicates.")
    starts: int = Field(default=1, description="Starts per fit.")


class ResidualBootstrap(UncertaintyMethod[BootstrapResult], RangeChecked):
    """
    The residual block bootstrap.

    Each replicate joins blocks of residuals of the fit, drawn with
    replacement across every series at once, adds them to the fitted values
    on `residual_scale`, and fits the synthetic data.

    Attributes
    ----------
    kind : "residual"
        The bootstrap, which the engine reads.
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block; with ``scheme="stationary"`` the mean length.
    residual_scale : {"additive", "multiplicative"}
        ``"additive"``: ``r = y − ŷ`` and ``ŷ + r*``. ``"multiplicative"``:
        ``r = log(y / ŷ)`` and ``ŷ · exp(r*)``; it needs positive observed and
        fitted values and keeps synthetic values positive.
    scheme : {"moving", "stationary"}
        ``"moving"``: blocks of `block_length` consecutive positions (Künsch
        1989). ``"stationary"``: blocks of geometric length with mean
        `block_length`, wrapped around the end (Politis and Romano 1994).
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    _runner = staticmethod(run_bootstrap)

    kind: Literal["residual"] = Field(default="residual", description="The bootstrap.")
    replicates: int = Field(description="Number of replicates.")
    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    residual_scale: Literal["additive", "multiplicative"] = Field(
        description="The scale of the residuals."
    )
    scheme: Literal["moving", "stationary"] = Field(
        description="How blocks of residuals are drawn."
    )
    starts: int = Field(default=1, description="Starts per fit.")


class WildBootstrap(UncertaintyMethod[BootstrapResult], RangeChecked):
    """
    The wild block bootstrap.

    Each replicate multiplies the additive residuals of each block by one
    draw of `multipliers` and fits ``ŷ + r · v``. Every residual stays at its
    own position, so it holds under noise whose size changes along the
    series.

    Attributes
    ----------
    kind : "wild"
        The bootstrap, which the engine reads.
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block.
    multipliers : {"rademacher", "mammen"}
        ``"rademacher"``: ±1 with probability 1/2. ``"mammen"``: two values
        with mean 0, variance 1 and third moment 1 (Mammen 1993), which keeps
        the skewness of the residuals.
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    _runner = staticmethod(run_bootstrap)

    kind: Literal["wild"] = Field(default="wild", description="The bootstrap.")
    replicates: int = Field(description="Number of replicates.")
    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    multipliers: Literal["rademacher", "mammen"] = Field(
        description="The distribution of the multipliers."
    )
    starts: int = Field(default=1, description="Starts per fit.")


class BlockJackknife(UncertaintyMethod[JackknifeResult], RangeChecked):
    """
    The block jackknife (Künsch 1989): one fit per block left out.

    It gives a standard error and a bias of each parameter, and the influence
    of each block, with as many fits as blocks. A block is left out by giving
    its observations weight 0.

    Attributes
    ----------
    block_length : int | "auto"
        Positions per block; it must leave at least two blocks.
    starts : int
        Starts of the fit and of each leave-out fit (default 1).
    """

    _runner = staticmethod(run_block_jackknife)

    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    starts: int = Field(default=1, description="Starts per fit.")
