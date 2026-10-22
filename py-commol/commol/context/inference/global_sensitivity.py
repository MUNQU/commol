"""What a sensitivity analysis varies, its methods and their results.

An `InputSpace` names the values that vary, as calibration parameters with
their types and bounds, and how: independent distributions, one per input, or
a weighted sample whose inputs may depend on each other. Each method reduces
each run to the quantities of interest of `commol.context.inference.quantities`
in Rust, so a design of thousands of runs never returns to Python.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, model_validator

from commol.commol_rs import _commol_rs as commol_rs
from commol.context._model import engine_state
from commol.context.calibration import CalibrationParameter
from commol.context.constants import CalibrationParameterType
from commol.context.inference.distributions import Distribution, Uniform
from commol.context.inference.levels import ConfidenceLevel
from commol.context.inference.method import SensitivityMethod
from commol.context.inference.quantities import Quantity
from commol.context.inference.sample import ParameterSample

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from commol.context.visualization import PlotConfig


class InputSpace(BaseModel):
    """
    What a sensitivity analysis varies.

    Each input is a `CalibrationParameter`, a model parameter or an initial
    condition with its bounds, as a calibration sets it. The inputs follow
    independent distributions, one per input, truncated to its bounds, or
    are the draws of a sample, with their weights; the inputs of a sample may
    depend on each other, so only the methods that allow it take one. Build
    it with `independent`, `uniform_on_bounds` or `from_sample`.

    Attributes
    ----------
    parameters : list[CalibrationParameter]
        The inputs, in order. A factor or a noise parameter is not an input:
        both belong to observations, and an analysis has none.
    distributions : dict[str, Distribution] | None
        The distribution of each input, for independent inputs.
    sample : ParameterSample | None
        The draws of the inputs, for inputs from a sample.
    initial_condition_remainder : str | None
        The bin or compartment that takes what the initial conditions leave
        of the sum the model states, as in a calibration problem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameters: list[CalibrationParameter] = Field(
        min_length=1, description="The inputs, in order."
    )
    distributions: dict[str, Distribution] | None = Field(
        default=None, description="The distribution of each independent input."
    )
    sample: ParameterSample | None = Field(
        default=None, description="The draws of inputs from a sample."
    )
    initial_condition_remainder: str | None = Field(
        default=None,
        description="Bin or compartment that takes what the initial conditions leave.",
    )

    @model_validator(mode="after")
    def check_inputs(self) -> Self:
        """Check the inputs against the distributions or the sample."""
        ids = [parameter.id for parameter in self.parameters]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"Inputs repeat the ids {repeated}.")
        observed = {
            CalibrationParameterType.FACTOR: (
                "a factor, which multiplies an observation"
            ),
            CalibrationParameterType.NOISE: (
                "a noise parameter, which belongs to an observation model"
            ),
        }
        for parameter in self.parameters:
            what = observed.get(CalibrationParameterType(parameter.parameter_type))
            if what is not None:
                raise ValueError(
                    f"'{parameter.id}' is {what}; a sensitivity analysis has no "
                    "observations."
                )
        if (self.distributions is None) == (self.sample is None):
            raise ValueError("Give distributions or a sample, not both.")
        if self.distributions is not None:
            missing = [i for i in ids if i not in self.distributions]
            extra = sorted(set(self.distributions) - set(ids))
            if missing:
                raise ValueError(
                    f"The inputs {missing} have no distribution: give one per "
                    "input, or InputSpace.uniform_on_bounds(parameters)."
                )
            if extra:
                raise ValueError(f"The distributions of {extra} name no input.")
        if self.sample is not None:
            bounds = [(p.min_bound, p.max_bound) for p in self.parameters]
            if self.sample.names != ids or list(self.sample.bounds) != bounds:
                raise ValueError(
                    f"The sample has the inputs {self.sample.names} with the bounds "
                    f"{list(self.sample.bounds)}; the parameters are {ids} with the "
                    f"bounds {bounds}, in this order."
                )
        return self

    @classmethod
    def independent(
        cls,
        parameters: Sequence[CalibrationParameter],
        distributions: Mapping[str, Distribution],
        *,
        initial_condition_remainder: str | None = None,
    ) -> Self:
        """
        Independent inputs, each with its distribution.

        Parameters
        ----------
        parameters : Sequence[CalibrationParameter]
            The inputs, in order.
        distributions : Mapping[str, Distribution]
            The distribution of each input, truncated to its bounds.
        initial_condition_remainder : str | None, optional
            The bin or compartment that takes what the initial conditions
            leave.

        Returns
        -------
        InputSpace
            The input space.

        Raises
        ------
        ValueError
            If an input has no distribution or is a factor, a distribution
            names no input, or an id repeats.
        """
        return cls(
            parameters=list(parameters),
            distributions=dict(distributions),
            initial_condition_remainder=initial_condition_remainder,
        )

    @classmethod
    def uniform_on_bounds(
        cls,
        parameters: Sequence[CalibrationParameter],
        *,
        initial_condition_remainder: str | None = None,
    ) -> Self:
        """
        Independent inputs, each uniform on its bounds; they must be finite.

        Parameters
        ----------
        parameters : Sequence[CalibrationParameter]
            The inputs, in order.
        initial_condition_remainder : str | None, optional
            The bin or compartment that takes what the initial conditions
            leave.

        Returns
        -------
        InputSpace
            The input space.
        """
        return cls.independent(
            parameters,
            {parameter.id: Uniform() for parameter in parameters},
            initial_condition_remainder=initial_condition_remainder,
        )

    @classmethod
    def from_sample(
        cls,
        sample: ParameterSample,
        parameters: Sequence[CalibrationParameter],
        *,
        initial_condition_remainder: str | None = None,
    ) -> Self:
        """
        The draws of a sample, with their weights, such as a posterior.

        Its inputs may depend on each other, so only PRCC and propagation
        take it. The parameters give the type of each value, which a sample
        does not hold.

        Parameters
        ----------
        sample : ParameterSample
            The draws.
        parameters : Sequence[CalibrationParameter]
            The inputs, in the order and with the bounds of the sample.
        initial_condition_remainder : str | None, optional
            The bin or compartment that takes what the initial conditions
            leave.

        Returns
        -------
        InputSpace
            The input space.

        Raises
        ------
        ValueError
            If the sample's names or bounds are not the parameters'.
        """
        return cls(
            parameters=list(parameters),
            sample=sample,
            initial_condition_remainder=initial_condition_remainder,
        )

    @property
    def names(self) -> list[str]:
        """The ids of the inputs, in order."""
        return [parameter.id for parameter in self.parameters]

    def rust_distributions(self) -> list[tuple[str, str, list[float]]] | None:
        """The distributions as the Rust runners read them, or None."""
        if self.distributions is None:
            return None
        return [self.distributions[name].rust_prior(name) for name in self.names]


class QuantitySummary(BaseModel):
    """
    The weighted distribution of one quantity across the runs of a
    propagation.

    Attributes
    ----------
    label : str
        The quantity.
    confidence_level : float
        The level of the central interval.
    mean : float
        The weighted mean.
    median : float
        The weighted median.
    lower : float
        The quantile ``(1 − c)/2``.
    upper : float
        The quantile ``(1 + c)/2``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    confidence_level: float
    mean: float
    median: float
    lower: float
    upper: float


class PropagationResult(BaseModel):
    """
    The draws of the inputs and what each run gives.

    Its `sample` is a `ParameterSample`, so `Predictive(analysis,
    result.sample, positions)` gives the bands, plots and reports of the
    propagation, as for a calibration.

    Attributes
    ----------
    sample : ParameterSample
        The draws of the inputs: a Latin hypercube of their distributions, or
        the sample of the input space itself.
    labels : list[str]
        The label of each quantity.
    values : list[list[float]]
        The quantities of each run, ``[run][quantity]``: one run per draw, or
        with a stochastic engine one per replicate of each draw.
    weights : list[float]
        The weight of each run; with a stochastic engine each replicate has
        an equal share of its draw's weight. They sum to 1.
    draw_seeds : list[int]
        The seed of each draw of `sample`, in its order. With a stochastic
        engine, replicate ``i`` of draw ``j`` runs with the seed replicate
        ``i`` of ``Simulation.replicates(count, seed=draw_seeds[j])`` takes, so
        those replicates of the model at that draw give the runs of the draw.

    A quantity that reads a NaN is NaN in `values`, and `summary` raises on
    it: `where` keeps the runs a condition chooses, which conditions the
    result on that condition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: ParameterSample
    labels: list[str]
    values: list[list[float]]
    weights: list[float]
    draw_seeds: list[int]

    @property
    def effective_size(self) -> float:
        """
        Kish's effective size of the weights of the runs, ``(Σw)²/Σw²``: the
        number of equally weighted runs that carry as much information.
        """
        weights = np.asarray(self.weights, dtype=float)
        return float(weights.sum() ** 2 / (weights**2).sum())

    def where(self, mask: ArrayLike) -> "PropagationResult":
        """
        The runs where `mask` is true, with their weights renormalised to sum
        to 1.

        A filter conditions the result: a summary of the runs it keeps is the
        distribution of each quantity given the condition of the mask, such
        as ``np.isfinite(values[:, i])`` or a compartment above a threshold,
        not the distribution over every run. `effective_size` says how many
        runs the kept weights are worth. `sample` and `draw_seeds` stay those
        of the draws of the inputs.

        Parameters
        ----------
        mask : ArrayLike
            One boolean per run, in the order of `values`.

        Returns
        -------
        PropagationResult
            The kept runs, their values and their renormalised weights.

        Raises
        ------
        ValueError
            If `mask` has not one boolean per run, keeps no run, or keeps
            only runs of weight 0.
        """
        keep = np.asarray(mask)
        if keep.dtype != np.bool_ or keep.shape != (len(self.values),):
            raise ValueError(
                f"The mask has one boolean per run: {len(self.values)} runs, got "
                f"an array of {keep.dtype} with the shape {keep.shape}"
            )
        weights = np.asarray(self.weights, dtype=float)[keep]
        if weights.sum() <= 0.0:
            raise ValueError(
                "The mask keeps no run of positive weight: a summary needs at least one"
            )
        return self.model_copy(
            update={
                "values": [row for row, kept in zip(self.values, keep) if kept],
                "weights": (weights / weights.sum()).tolist(),
            }
        )

    def summary(self, confidence_level: float) -> list[QuantitySummary]:
        """
        The weighted distribution of each quantity across the runs.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        list[QuantitySummary]
            One summary per quantity, in the order of `labels`.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1, or a
            quantity is NaN in a run: `where` keeps the runs a condition
            chooses.
        """
        levels = [0.5, *commol_rs.inference.interval_levels(confidence_level)]
        summaries = []
        for index, label in enumerate(self.labels):
            column = [row[index] for row in self.values]
            try:
                median, lower, upper = commol_rs.inference.weighted_quantiles(
                    column, self.weights, levels
                )
            except ValueError as error:
                raise ValueError(
                    f"Quantity '{label}', in the runs of values counted from 0: "
                    f"{error}. where(mask) keeps the runs a condition chooses."
                ) from error
            summaries.append(
                QuantitySummary(
                    label=label,
                    confidence_level=confidence_level,
                    mean=commol_rs.inference.weighted_mean(column, self.weights),
                    median=median,
                    lower=lower,
                    upper=upper,
                )
            )
        return summaries


class Propagation(SensitivityMethod[PropagationResult]):
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
    quantities : list[Quantity]
        What each run is reduced to; none gives only the sample, for a
        `Predictive`.
    """

    draws: int | None = Field(
        default=None, ge=1, description="Points of the Latin hypercube."
    )
    quantities: list[Quantity] = Field(
        default_factory=list, description="What each run is reduced to."
    )


class PRCCResult(BaseModel):
    """
    The partial rank correlation of each input with each quantity.

    A coefficient near 1 or −1 says the quantity rises or falls steadily
    with the input once the other inputs are accounted for; near 0, that it
    does not move steadily with it. The p-value tests the coefficient against
    0 with Kish's effective size of the draws.

    Attributes
    ----------
    sample : ParameterSample
        The draws the coefficients come from.
    inputs : list[str]
        The inputs, in order.
    labels : list[str]
        The label of each quantity.
    series : list[str]
        The series of each quantity: its label, or the name of its
        `Trajectory`.
    positions : list[float | None]
        The position of each quantity of a `Trajectory`, None for the others.
    coefficients : list[list[float]]
        The PRCC of each input for each quantity, ``[quantity][input]``; NaN
        for a quantity with one value in every draw.
    p_values : list[list[float]]
        The two-sided p-value of each coefficient, ``[quantity][input]``.
    effective_size : float
        Kish's effective size of the draws: their number with equal weights.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: ParameterSample
    inputs: list[str]
    labels: list[str]
    series: list[str]
    positions: list[float | None]
    coefficients: list[list[float]]
    p_values: list[list[float]]
    effective_size: float

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        x_label: str = "Step",
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw the PRCC of each input: one panel per series.

        A series of a `Trajectory` draws one line per input over its
        positions; any other quantity draws one bar per input.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        x_label : str
            The label of the x axis of a series over positions.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the name of its series. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a series.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _prcc

        return _prcc(
            self, output_file=output_file, config=config, x_label=x_label, axes=axes
        )


class PRCC(SensitivityMethod[PRCCResult]):
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
    quantities : list[Quantity]
        What each run is reduced to.
    """

    draws: int | None = Field(
        default=None, ge=1, description="Points of the Latin hypercube."
    )
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )


class MorrisResult(BaseModel):
    """
    The Morris indices of each input for each quantity.

    The elementary effects are on the scale of probabilities: a step moves
    an input by half the range of `F(x)`, so the indices of inputs of
    different units compare. Rank the inputs by `mu_star`; a `sigma` near or
    above `mu_star` says the effect of the input depends on the other inputs
    or is not linear.

    Attributes
    ----------
    inputs : list[str]
        The inputs, in order.
    labels : list[str]
        The label of each quantity.
    mu : list[list[float]]
        The mean elementary effect, ``[quantity][input]``.
    mu_star : list[list[float]]
        The mean absolute elementary effect, ``[quantity][input]``.
    sigma : list[list[float]]
        The standard deviation of the elementary effects,
        ``[quantity][input]``; NaN with one trajectory.
    runs : int
        The number of points run: one more than the inputs per trajectory.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: list[str]
    labels: list[str]
    mu: list[list[float]]
    mu_star: list[list[float]]
    sigma: list[list[float]]
    runs: int

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw ``σ`` against ``μ*`` of each input: one panel per quantity,
        one labelled point per input.

        An input far to the right matters; one high up acts through
        interactions or a non-linear effect.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the label of its quantity. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a quantity.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _morris

        return _morris(self, output_file=output_file, config=config, axes=axes)


class Morris(SensitivityMethod[MorrisResult]):
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
    quantities : list[Quantity]
        What each run is reduced to.
    """

    trajectories: int = Field(ge=1, description="The number of trajectories.")
    levels: int = Field(ge=2, description="The number of levels; even.")
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )


class SobolResult(BaseModel):
    """
    The Sobol indices of each input for each quantity, with their bootstrap
    intervals.

    The first-order index is the share of the variance of the quantity the
    input explains alone; the total index adds every interaction the input
    takes part in. A total index near 0 says the input can be fixed; a total
    index well above the first-order one says the input acts through
    interactions.

    Attributes
    ----------
    inputs : list[str]
        The inputs, in order.
    labels : list[str]
        The label of each quantity.
    confidence_level : float
        The level of the intervals.
    first, total : list[list[float]]
        The first-order and total indices, ``[quantity][input]``; NaN for a
        quantity of no variance.
    first_lower, first_upper : list[list[float]]
        The ends of the interval of each first-order index; NaN when a
        bootstrap replicate of the rows has no variance, which more
        `draws` avoid.
    total_lower, total_upper : list[list[float]]
        The ends of the interval of each total index.
    runs : int
        The number of points run: ``draws · (inputs + 2)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: list[str]
    labels: list[str]
    confidence_level: float
    first: list[list[float]]
    total: list[list[float]]
    first_lower: list[list[float]]
    first_upper: list[list[float]]
    total_lower: list[list[float]]
    total_upper: list[list[float]]
    runs: int

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw the first-order and total Sobol indices: one panel per
        quantity, one pair of bars per input, with their intervals.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the label of its quantity. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a quantity.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _sobol

        return _sobol(self, output_file=output_file, config=config, axes=axes)


class Sobol(SensitivityMethod[SobolResult]):
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
    quantities : list[Quantity]
        What each run is reduced to.
    """

    draws: int = Field(ge=2, description="Rows of each base matrix.")
    replicates: int = Field(ge=2, description="Bootstrap replicates.")
    confidence_level: ConfidenceLevel = Field(description="Level of the intervals.")
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )


class ChaosResult(BaseModel):
    """
    A polynomial chaos expansion of each quantity, its accuracy and the
    Sobol indices of its coefficients.

    The expansion is a surrogate of the model: `predict` gives the
    quantities at any values of the inputs inside their bounds, at the cost
    of a polynomial. Read
    `loo_error` first: the indices are those of the expansion, and they are
    those of the model only as far as the expansion predicts it.

    Attributes
    ----------
    inputs : list[str]
        The inputs, in order.
    input_space : InputSpace
        The inputs and their distributions, which `predict` maps values
        through.
    labels : list[str]
        The label of each quantity.
    degree : int
        The degree of the basis: the bound of the q-norm of every term.
    q_norm : float
        The q of the q-norm of the basis; 1 for the total-degree basis.
    max_interaction : int | None
        The largest number of inputs of a term; None for no limit.
    indices : list[list[int]]
        The degree of each input in each term, ``[term][input]``; term 0 is
        the constant.
    coefficients : list[list[float]]
        The coefficient of each term for each quantity, ``[quantity][term]``,
        on orthonormal Legendre polynomials of ``u = F(x)``.
    loo_error : list[float]
        The leave-one-out error of each quantity, relative to its variance;
        NaN for a quantity of no variance.
    mean, variance : list[float]
        The mean and the variance of each quantity under the expansion.
    first, total : list[list[float]]
        The first-order and total Sobol indices, ``[quantity][input]``; NaN
        for a quantity of no variance.
    runs : int
        The number of points run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: list[str]
    input_space: InputSpace
    labels: list[str]
    degree: int
    q_norm: float
    max_interaction: int | None
    indices: list[list[int]]
    coefficients: list[list[float]]
    loo_error: list[float]
    mean: list[float]
    variance: list[float]
    first: list[list[float]]
    total: list[list[float]]
    runs: int

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw the first-order and total Sobol indices of the expansion: one
        panel per quantity, one pair of bars per input.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the label of its quantity. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a quantity.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.api.plotting.figures import _sobol

        return _sobol(self, output_file=output_file, config=config, axes=axes)

    def predict(self, values: Sequence[Sequence[float]]) -> list[list[float]]:
        """
        The quantities the expansion gives at each set of input values.

        Parameters
        ----------
        values : Sequence[Sequence[float]]
            One value per input, in the order of `inputs`, for each point,
            each inside the bounds of its input.

        Returns
        -------
        list[list[float]]
            The value of each quantity at each point, ``[point][quantity]``.

        Raises
        ------
        ValueError
            If a point has not one value per input or a value outside the
            bounds of its input, the truncation is not valid, or the
            coefficients do not fit the basis.
        """
        return commol_rs.inference.chaos_evaluate(
            engine_state.rust_parameters(self.input_space.parameters),
            self.input_space.rust_distributions() or [],
            self.degree,
            self.q_norm,
            self.max_interaction,
            self.coefficients,
            [list(point) for point in values],
        )


class PolynomialChaos(SensitivityMethod[ChaosResult]):
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
    quantities : list[Quantity]
        What each run is reduced to.
    """

    degree: int = Field(ge=1, description="The degree of the basis.")
    q_norm: float = Field(
        default=1.0, gt=0.0, le=1.0, description="The q of the q-norm of the basis."
    )
    max_interaction: int | None = Field(
        default=None, ge=1, description="The largest number of inputs of a term."
    )
    draws: int = Field(ge=2, description="Points of the Latin hypercube.")
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
