"""What a sensitivity analysis varies, its methods and their results.

An `InputSpace` names the values that vary, as calibration parameters with
their types and bounds, and how: independent distributions, one per input, or
a weighted sample whose inputs may depend on each other. Each method reduces
each run to the quantities of interest of `commol.quantities.quantities`
in Rust, so a design of thousands of runs never returns to Python.
"""

from typing import TYPE_CHECKING

from pydantic import Field

from commol._calibration.method import SensitivityMethod
from commol._data.ranges import RangeChecked
from commol.quantities.quantities import _Quantity
from commol.sensitivity.runners import (
    run_chaos,
    run_morris,
    run_prcc,
    run_propagation,
    run_sobol,
)

if TYPE_CHECKING:
    pass

from commol.sensitivity.results import (
    ChaosResult,
    MorrisResult,
    PRCCResult,
    PropagationResult,
    SobolResult,
)


class Propagation(SensitivityMethod[PropagationResult], RangeChecked):
    """
    Monte Carlo propagation: run the draws of the inputs and reduce each run
    to the quantities.

    With independent inputs the draws are a Latin hypercube of their
    distributions; with inputs from a sample, the sample is the design, with
    its weights.

    Attributes
    ----------
    draws : int | None
        The number of points of the Latin hypercube; required with
        independent inputs, and left unset with inputs from a sample.
    quantities : list[quantity]
        What each run is reduced to; none gives only the sample, for a
        `Predictive`.
    """

    _runner = staticmethod(run_propagation)

    draws: int | None = Field(
        default=None, description="Points of the Latin hypercube."
    )
    quantities: list[_Quantity] = Field(
        default_factory=list, description="What each run is reduced to."
    )


class PRCC(SensitivityMethod[PRCCResult], RangeChecked):
    """
    Latin hypercube sampling with partial rank correlation coefficients
    (Marino et al. 2008): how steadily each quantity rises or falls with each
    input.

    With independent inputs the draws are a Latin hypercube of their
    distributions; with inputs from a sample, the sample is the design, and
    the ranks, the correlations and the test read its weights. A
    `Trajectory` gives the coefficients over the axis.

    Attributes
    ----------
    draws : int | None
        The number of points of the Latin hypercube; required with
        independent inputs, and left unset with inputs from a sample.
    quantities : list[quantity]
        What each run is reduced to.
    """

    _runner = staticmethod(run_prcc)

    draws: int | None = Field(
        default=None, description="Points of the Latin hypercube."
    )
    quantities: list[_Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )


class Morris(SensitivityMethod[MorrisResult], RangeChecked):
    """
    Morris screening (Morris 1991), with the ``μ*`` of Campolongo, Cariboni
    and Saltelli (2007): which inputs matter, in few runs.

    Each trajectory starts at a random level of every input and moves the
    inputs one at a time, by half the levels; the levels are the centres of
    strata of equal probability of each distribution. It needs independent
    inputs: a step that moves one input breaks the dependence of a sample.

    Attributes
    ----------
    trajectories : int
        The number of trajectories; each runs one more point than there are
        inputs.
    levels : int
        The number of levels of each input; even, at least 2.
    quantities : list[quantity]
        What each run is reduced to.
    """

    _runner = staticmethod(run_morris)

    trajectories: int = Field(description="The number of trajectories.")
    levels: int = Field(description="The number of levels; even.")
    quantities: list[_Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )


class Sobol(SensitivityMethod[SobolResult], RangeChecked):
    """
    Sobol indices on the design of Saltelli (2010): first-order indices by
    the estimator of Saltelli (2010) and total indices by the estimator of
    Jansen (1999), with intervals from a bootstrap of the rows.

    The base matrices hold independent Monte Carlo draws, so the bootstrap
    intervals are valid. It needs independent inputs: with dependent inputs
    the indices lose their meaning and no longer add up; PRCC and propagation
    take a sample.

    Attributes
    ----------
    draws : int
        The rows of each base matrix; the analysis runs
        ``draws · (inputs + 2)`` points.
    replicates : int
        The bootstrap replicates of the intervals, at least 2.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.
    quantities : list[quantity]
        What each run is reduced to.
    """

    _runner = staticmethod(run_sobol)

    draws: int = Field(description="Rows of each base matrix.")
    replicates: int = Field(description="Bootstrap replicates.")
    confidence_level: float = Field(description="Level of the intervals.")
    quantities: list[_Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )


class PolynomialChaos(SensitivityMethod[ChaosResult], RangeChecked):
    """
    A polynomial chaos expansion (Sudret 2008): a polynomial surrogate of
    each quantity, and the Sobol indices of its coefficients, in far fewer
    runs than `Sobol` when the quantity is smooth.

    Every input goes to ``u = F(x)``, uniform on ``[0, 1]`` through its
    distribution, so one basis of orthonormal Legendre polynomials fits every
    distribution, truncated ones included. The coefficients come from least
    squares on a Latin hypercube; the leave-one-out error says how well the
    expansion predicts the model. It needs independent inputs.

    The basis keeps every term whose degrees ``α`` have
    ``Σ α_i^q ≤ degree^q`` and that holds at most ``max_interaction``
    inputs. The defaults keep the total-degree basis, of
    ``(k + degree)! / (k! degree!)`` terms in ``k`` inputs; a truncation
    keeps the terms of low interaction order, so a basis of many inputs
    needs far fewer terms and runs. The number of terms is known before any
    run.

    Attributes
    ----------
    degree : int
        The degree of the basis: the largest degree of an input in a term,
        and the total degree of a term when ``q_norm`` is 1.
    q_norm : float
        The q of the hyperbolic truncation, in ``(0, 1]``: below 1, a term
        of several inputs keeps a lower total degree than a term of one.
    max_interaction : int | None
        The largest number of inputs of a term; None for no limit.
    draws : int
        The points of the Latin hypercube; more than the terms, and two or
        three times as many is the usual choice.
    quantities : list[quantity]
        What each run is reduced to.
    """

    _runner = staticmethod(run_chaos)

    degree: int = Field(description="The degree of the basis.")
    q_norm: float = Field(default=1.0, description="The q of the q-norm of the basis.")
    max_interaction: int | None = Field(
        default=None, description="The largest number of inputs of a term."
    )
    draws: int = Field(description="Points of the Latin hypercube.")
    quantities: list[_Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
