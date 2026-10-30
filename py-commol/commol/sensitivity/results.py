"""The results of the methods of `commol.sensitivity`."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import ConfigDict, Field

from commol._data.engine_json import EngineChecked, engine_json
from commol._data.result_model import ResultModel
from commol._problem.problem import CalibrationParameter
from commol.commol_rs import _commol_rs as commol_rs
from commol.distributions.distributions import Uniform, _Distribution
from commol.uncertainty.sample import ParameterSample

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from commol.plotting.config import PlotConfig


class InputSpace(EngineChecked):
    """
    What a sensitivity analysis varies.

    Each input is a `CalibrationParameter`, a model parameter or an initial
    condition with its bounds, as a calibration sets it. The inputs follow
    independent distributions, one per input, truncated to its bounds, or
    are the draws of a sample, with their weights; the inputs of a sample may
    depend on each other, so only the methods that allow it take one. Build
    it with `independent`, `uniform_on_bounds` or `from_sample`: it has
    distributions or a sample, exactly one, with no default, which
    `SensitivityAnalysis` checks with the inputs of the sample.

    Attributes
    ----------
    parameters : list[CalibrationParameter]
        The inputs, in order. A factor or a noise parameter is not an input:
        both belong to observations, and an analysis has none.
    distributions : dict[str, distribution] | None
        The distribution of each input, for independent inputs.
    sample : ParameterSample | None
        The draws of the inputs, for inputs from a sample.
    initial_condition_remainder : str | None
        The bin or compartment that takes what the initial conditions leave
        of the sum the model states, as in a calibration problem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    _engine_handles: ClassVar[tuple[str, ...]] = ("sample",)

    parameters: list[CalibrationParameter] = Field(description="The inputs, in order.")
    distributions: dict[str, _Distribution] | None = Field(
        default=None, description="The distribution of each independent input."
    )
    sample: ParameterSample | None = Field(
        default=None, description="The draws of inputs from a sample."
    )
    initial_condition_remainder: str | None = Field(
        default=None,
        description="Bin or compartment that takes what the initial conditions leave.",
    )

    @classmethod
    def independent(
        cls,
        parameters: Sequence[CalibrationParameter],
        distributions: Mapping[str, _Distribution],
        *,
        initial_condition_remainder: str | None = None,
    ) -> Self:
        """
        Independent inputs, each with its distribution.

        Parameters
        ----------
        parameters : Sequence[CalibrationParameter]
            The inputs, in order.
        distributions : Mapping[str, distribution]
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

        Notes
        -----
        `SensitivityAnalysis` checks that the sample's names and bounds are
        the parameters'.
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


class QuantitySummary(ResultModel):
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


class PropagationResult(ResultModel):
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
        return commol_rs.inference.effective_size(self.weights)

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
        kept = [weight for weight, kept in zip(self.weights, keep) if kept]
        return self.model_copy(
            update={
                "values": [row for row, kept in zip(self.values, keep) if kept],
                "weights": commol_rs.inference.normalized_weights(kept),
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
        # The level is checked before the values, also when there are none.
        commol_rs.inference.interval_levels(confidence_level)
        summaries = []
        for index, label in enumerate(self.labels):
            column = [row[index] for row in self.values]
            try:
                summary = commol_rs.inference.weighted_summary(
                    column, self.weights, confidence_level
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
                    mean=summary.mean,
                    median=summary.median,
                    lower=summary.lower,
                    upper=summary.upper,
                )
            )
        return summaries


class PRCCResult(ResultModel):
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
        from commol.sensitivity.plots import prcc_figure

        return prcc_figure(
            self, output_file=output_file, config=config, x_label=x_label, axes=axes
        )


class MorrisResult(ResultModel):
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
        from commol.sensitivity.plots import morris_figure

        return morris_figure(self, output_file=output_file, config=config, axes=axes)


class SobolResult(ResultModel):
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
        from commol.sensitivity.plots import sobol_figure

        return sobol_figure(self, output_file=output_file, config=config, axes=axes)


class ChaosResult(ResultModel):
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
        from commol.sensitivity.plots import sobol_figure

        return sobol_figure(self, output_file=output_file, config=config, axes=axes)

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
            engine_json(self.input_space),
            self.degree,
            self.q_norm,
            self.max_interaction,
            self.coefficients,
            [list(point) for point in values],
        )
