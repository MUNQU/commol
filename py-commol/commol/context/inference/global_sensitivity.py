"""What a sensitivity analysis varies and what it reads from each run.

An `InputSpace` names the values that vary, as calibration parameters with
their types and bounds, and how: independent distributions, one per input, or
a weighted sample whose inputs may depend on each other. A quantity of
interest reduces each run to a scalar in Rust, through the observation
operator of the calibration, so a design of thousands of runs never returns
to Python.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from commol.commol_rs import _commol_rs as commol_rs
from commol.context._model import engine_state
from commol.context.calibration import CalibrationParameter
from commol.context.constants import CalibrationParameterType
from commol.context.inference.distributions import Distribution, Uniform
from commol.context.inference.method import SensitivityMethod
from commol.context.inference.sample import ParameterSample

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from commol.context.visualization import PlotConfig

type RustQuantity = tuple[str, list[str], list[float]]
"""A quantity as Rust reads it: its kind, its outputs and its positions."""


class QuantityBase(BaseModel, ABC):
    """
    A scalar read from each run: one output, or the sum of several, as an
    observation reads them.

    Attributes
    ----------
    outputs : str | list[str]
        The output read, or the outputs summed.
    label : str | None
        The name of the quantity in results and plots; by default it says
        what the quantity reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rust_kind: ClassVar[str]

    outputs: str | list[str] = Field(description="The output, or outputs summed.")
    label: str | None = Field(default=None, description="The name of the quantity.")

    @field_validator("outputs")
    @classmethod
    def require_outputs(cls, value: str | list[str]) -> str | list[str]:
        """An empty list sums nothing."""
        if isinstance(value, list) and not value:
            raise ValueError("outputs must not be empty")
        return value

    @property
    def output_list(self) -> list[str]:
        """The outputs summed, as a list."""
        return [self.outputs] if isinstance(self.outputs, str) else list(self.outputs)

    @property
    def output_name(self) -> str:
        """The outputs as a label reads them."""
        return " + ".join(self.output_list)

    @abstractmethod
    def expand(self) -> list[tuple[str, RustQuantity]]:
        """The label and the Rust form of each scalar this quantity gives."""

    def series(self) -> list[tuple[str, float | None]]:
        """
        The series and the position of each scalar this quantity gives: its
        label and no position, or for a `Trajectory` its name and each
        position.
        """
        return [(label, None) for label, _ in self.expand()]


class _OneScalar(QuantityBase):
    """A quantity that gives one scalar per run."""

    @abstractmethod
    def default_label(self) -> str:
        """The label when none is given."""

    @abstractmethod
    def rust_quantity(self) -> RustQuantity:
        """The quantity as Rust reads it."""

    def expand(self) -> list[tuple[str, RustQuantity]]:
        """The label and the Rust form of the scalar."""
        return [(self.label or self.default_label(), self.rust_quantity())]


Position = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
"""A position on the axis of the model."""


def _position_text(at: float) -> str:
    """`at` as the shortest text that reads back as it: `repr`, with no `.0`."""
    return repr(at).removesuffix(".0")


class ValueAt(_OneScalar):
    """
    The value at a position.

    Attributes
    ----------
    at : float
        The position.
    """

    rust_kind: ClassVar[str] = "value_at"
    kind: Literal["value_at"] = "value_at"
    at: Position = Field(description="The position.")

    def default_label(self) -> str:
        """`outputs` at `at`."""
        return f"{self.output_name} at {_position_text(self.at)}"

    def rust_quantity(self) -> RustQuantity:
        """The kind, the outputs and the position."""
        return (self.rust_kind, self.output_list, [self.at])


class Gain(_OneScalar):
    """
    The value at `end` minus the value at `start`: what entered a cumulative
    output over ``(start, end]``, as an observation with the window
    ``end - start`` reads it.

    Attributes
    ----------
    start : float
        The start of the window.
    end : float
        The end of the window, after `start`.
    """

    rust_kind: ClassVar[str] = "gain"
    kind: Literal["gain"] = "gain"
    start: Position = Field(description="The start of the window.")
    end: Position = Field(description="The end of the window.")

    @model_validator(mode="after")
    def end_after_start(self) -> Self:
        """A window ends after it starts."""
        if self.end <= self.start:
            raise ValueError(
                f"A gain ends after it starts, got start {self.start} and end "
                f"{self.end}"
            )
        return self

    def default_label(self) -> str:
        """The gain of `outputs` over the window."""
        start, end = _position_text(self.start), _position_text(self.end)
        return f"{self.output_name} gain over ({start}, {end}]"

    def rust_quantity(self) -> RustQuantity:
        """The kind, the outputs, the start and the end."""
        return (self.rust_kind, self.output_list, [self.start, self.end])


class _AtPositions(QuantityBase):
    """A quantity read at several positions."""

    positions: list[Position] = Field(
        min_length=1, description="The positions, ascending."
    )

    @field_validator("positions")
    @classmethod
    def ascending(cls, value: list[float]) -> list[float]:
        """The positions ascend, each once."""
        if any(later <= earlier for earlier, later in zip(value, value[1:])):
            raise ValueError("positions must ascend, each once")
        return value


class _Peak(_AtPositions, _OneScalar):
    """A reduction of the values at several positions."""

    def rust_quantity(self) -> RustQuantity:
        """The kind, the outputs and the positions."""
        return (self.rust_kind, self.output_list, list(self.positions))


class Peak(_Peak):
    """
    The largest value at the positions.

    Attributes
    ----------
    positions : list[float]
        The positions compared, ascending.
    """

    rust_kind: ClassVar[str] = "peak"
    kind: Literal["peak"] = "peak"

    def default_label(self) -> str:
        """The peak of `outputs`."""
        return f"peak of {self.output_name}"


class PeakPosition(_Peak):
    """
    The first of the positions where the value is largest.

    Attributes
    ----------
    positions : list[float]
        The positions compared, ascending.
    """

    rust_kind: ClassVar[str] = "peak_position"
    kind: Literal["peak_position"] = "peak_position"

    def default_label(self) -> str:
        """The position of the peak of `outputs`."""
        return f"position of the peak of {self.output_name}"


class Trajectory(_AtPositions):
    """
    The value at each position: one `ValueAt` per position, for an analysis
    over the axis.

    Attributes
    ----------
    positions : list[float]
        The positions, ascending.
    label : str | None
        The name of the series; each scalar is named after it and its
        position.
    """

    rust_kind: ClassVar[str] = "value_at"
    kind: Literal["trajectory"] = "trajectory"

    def series(self) -> list[tuple[str, float | None]]:
        """Its name and each position."""
        name = self.label or self.output_name
        return [(name, at) for at in self.positions]

    def expand(self) -> list[tuple[str, RustQuantity]]:
        """One `ValueAt` per position."""
        name = self.label or self.output_name
        return [
            (
                f"{name} at {_position_text(at)}",
                (self.rust_kind, self.output_list, [at]),
            )
            for at in self.positions
        ]


type Quantity = Annotated[
    ValueAt | Gain | Peak | PeakPosition | Trajectory, Field(discriminator="kind")
]
"""One of the quantities of interest."""


def expand_quantities(
    quantities: Sequence[QuantityBase],
) -> tuple[list[str], list[RustQuantity]]:
    """
    The label and the Rust form of every scalar of `quantities`, in order.

    Parameters
    ----------
    quantities : Sequence[Quantity]
        The quantities; a `Trajectory` gives one scalar per position.

    Returns
    -------
    tuple[list[str], list[RustQuantity]]
        The labels and the quantities as Rust reads them.

    Raises
    ------
    ValueError
        If there is no quantity, or two scalars have one label.
    """
    if not quantities:
        raise ValueError("Give at least one quantity of interest.")
    expanded = [pair for quantity in quantities for pair in quantity.expand()]
    labels = [label for label, _ in expanded]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise ValueError(
            f"Two quantities have the labels {repeated}: give each a label of its own."
        )
    return labels, [rust for _, rust in expanded]


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
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: ParameterSample
    labels: list[str]
    values: list[list[float]]
    weights: list[float]

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
            If the confidence level is not strictly between 0 and 1.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"The confidence level is strictly between 0 and 1, got "
                f"{confidence_level}"
            )
        levels = [0.5, (1 - confidence_level) / 2, (1 + confidence_level) / 2]
        summaries = []
        for index, label in enumerate(self.labels):
            column = [row[index] for row in self.values]
            median, lower, upper = commol_rs.inference.weighted_quantiles(
                column, self.weights, levels
            )
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
        `base_draws` avoid.
    total_lower, total_upper : list[list[float]]
        The ends of the interval of each total index.
    runs : int
        The number of points run: ``base_draws · (inputs + 2)``.
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
    base_draws : int
        The rows of each base matrix; the analysis runs
        ``base_draws · (inputs + 2)`` points.
    bootstrap : int
        The bootstrap replicates of the intervals, at least 2.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.
    quantities : list[Quantity]
        What each run is reduced to.
    """

    base_draws: int = Field(ge=2, description="Rows of each base matrix.")
    bootstrap: int = Field(ge=2, description="Bootstrap replicates.")
    confidence_level: float = Field(
        gt=0.0, lt=1.0, description="Level of the intervals."
    )
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
        The total degree of the basis.
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
            bounds of its input, or the coefficients do not fit the degree.
        """
        return commol_rs.inference.chaos_evaluate(
            engine_state.rust_parameters(self.input_space.parameters),
            self.input_space.rust_distributions() or [],
            self.degree,
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

    Attributes
    ----------
    degree : int
        The total degree of the basis: in ``k`` inputs it has
        ``(k + degree)! / (k! degree!)`` terms.
    draws : int
        The points of the Latin hypercube; more than the terms, and two or
        three times as many is the usual choice.
    quantities : list[Quantity]
        What each run is reduced to.
    """

    degree: int = Field(ge=1, description="The total degree of the basis.")
    draws: int = Field(ge=2, description="Points of the Latin hypercube.")
    quantities: list[Quantity] = Field(
        min_length=1, description="What each run is reduced to."
    )
