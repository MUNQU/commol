"""Quantities of interest: the scalars a method reads from each run.

A quantity reads a series of a run and reduces it. The series is the level
of some outputs (`Level`, their sum at each position), or their increments
over windows of one width (`Increments`). The reduction is the value at a
position (`ValueAt`), the largest value or its first position over given
positions (`Peak`, `PeakPosition`), or the value at each position
(`Trajectory`). `Gain` is one window given by its start and its end.
`Ratio` divides two quantities of one value each.

Every quantity follows the same rules:

- it reads only the positions it names, with no interpolation;
- outputs are summed at each row before any reduction;
- `PeakPosition` gives the first of the positions where the value is
  largest;
- a NaN in the values a quantity reads makes it NaN;
- with a stochastic engine, a quantity is computed for each replicate;
- the ratio of a run whose denominator is 0 is NaN.

The quantities cross to Rust as the JSON of their classes, and Rust reduces
each run, through the observation operator of the calibration.
`quantity_schedule` gives the positions a run must report at, so the
results of ``Simulation.run_at(quantity_schedule(quantities))`` give the
quantities of that run (`SimulationResults.quantities`).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import (
    JsonValue,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from commol.commol_rs import _commol_rs as commol_rs
from commol._data.engine_json import engine_json
from commol._data.ranges import RangeChecked


def _position_text(at: float) -> str:
    """`at` as the shortest text that reads back as it: `repr`, with no `.0`."""
    return repr(at).removesuffix(".0")


def _output_list(value: JsonValue) -> JsonValue:
    """One output as a list of one."""
    return [value] if isinstance(value, str) else value


class _Outputs(BaseModel):
    """Outputs summed at each row; one name is a list of one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outputs: list[str] = Field(min_length=1, description="The outputs summed.")

    @field_validator("outputs", mode="before")
    @classmethod
    def _one_output(cls, value: JsonValue) -> JsonValue:
        """One name is a list of one."""
        return _output_list(value)

    @property
    def output_name(self) -> str:
        """The outputs as a label reads them."""
        return " + ".join(self.outputs)


class Level(_Outputs):
    """
    The level of some outputs: their sum at each position.

    A quantity given a name or a list of names reads this series.

    Attributes
    ----------
    outputs : list[str]
        The outputs summed; one name is a list of one.
    """

    kind: Literal["level"] = "level"

    def __init__(
        self, outputs: str | list[str], *, kind: Literal["level"] = "level"
    ) -> None:
        BaseModel.__init__(self, outputs=outputs, kind=kind)

    @property
    def text(self) -> str:
        """The series as a label reads it: the outputs."""
        return self.output_name

    def check_ends(self, positions: Sequence[float]) -> None:
        """A level reads any position."""


class Increments(_Outputs, RangeChecked):
    """
    The increments of some outputs: at each end, the gain of their sum over
    ``(end - window, end]``, as an observation with the window `window`
    reads it.

    The positions of a quantity of increments are window ends, each at
    least `window` within the rounding of the window (`window_start` of
    `crates/commol-calibration/src/observation.rs`), so no window starts
    before 0.

    Attributes
    ----------
    outputs : list[str]
        The outputs summed; one name is a list of one.
    window : float
        The width of each window, above 0.
    """

    kind: Literal["increments"] = "increments"
    window: float = Field(description="The width.")

    def __init__(
        self,
        outputs: str | list[str],
        window: float,
        *,
        kind: Literal["increments"] = "increments",
    ) -> None:
        BaseModel.__init__(self, outputs=outputs, window=window, kind=kind)

    @property
    def text(self) -> str:
        """The series as a label reads it: the gain over windows of a width."""
        return f"{self.output_name} gain over windows of {_position_text(self.window)}"

    def check_ends(self, positions: Sequence[float]) -> None:
        """
        No window starts before 0, with the rule of the engine
        (`check_window_end` of `crates/commol-calibration/src/observation.rs`).

        Raises
        ------
        ValueError
            With the message of the engine, if a window starts before 0.
        """
        for at in positions:
            commol_rs.calibration.check_window_end(at, self.window)


type _Series = Annotated[Level | Increments, Field(discriminator="kind")]
"""What a quantity reads at each position: a `Level` or `Increments`."""


def _series(value: JsonValue) -> JsonValue | Level:
    """A name or a list of names as the `Level` of those outputs."""
    if isinstance(value, str | list):
        return Level.model_validate({"outputs": value})
    return value


class QuantityBase(BaseModel, ABC):
    """
    A scalar, or one scalar per position, read from each run.

    Attributes
    ----------
    label : str | None
        The name of the quantity in results and plots; by default it says
        what the quantity reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str | None = Field(default=None, description="The name of the quantity.")

    @abstractmethod
    def labels(self) -> list[str]:
        """The label of each scalar this quantity gives."""

    def axes(self) -> list[tuple[str, float | None]]:
        """
        The series and the position of each scalar this quantity gives: its
        label and no position, or for a `Trajectory` its name and each
        position.
        """
        return [(label, None) for label in self.labels()]


class _OfSeries(QuantityBase):
    """A quantity that reads a series."""

    series: _Series = Field(description="The series read.")

    @field_validator("series", mode="before")
    @classmethod
    def _level_of_outputs(cls, value: JsonValue) -> JsonValue | Level:
        """A name or a list of names is the level of those outputs."""
        return _series(value)


class _OneScalar(QuantityBase):
    """A quantity that gives one scalar per run."""

    @abstractmethod
    def default_label(self) -> str:
        """The label when none is given."""

    def labels(self) -> list[str]:
        """The label of the scalar."""
        return [self.label or self.default_label()]


class ValueAt(_OfSeries, _OneScalar, RangeChecked):
    """
    The value of a series at a position: the level of the outputs, or their
    gain over the window that ends there.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    at : float
        The position; with `Increments`, the end of the window.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["value_at"] = "value_at"
    at: float = Field(description="The position.")

    @model_validator(mode="after")
    def _window_after_zero(self) -> Self:
        """The window of increments ends at least its width after 0."""
        self.series.check_ends([self.at])
        return self

    def default_label(self) -> str:
        """The series at `at`."""
        return f"{self.series.text} at {_position_text(self.at)}"


class Gain(_Outputs, _OneScalar, RangeChecked):
    """
    The value at `end` minus the value at `start`: what entered a cumulative
    output over ``(start, end]``, as an observation with the window
    ``end - start`` that starts at exactly `start` reads it.

    Attributes
    ----------
    outputs : list[str]
        The outputs summed; one name is a list of one.
    start : float
        The start of the window.
    end : float
        The end of the window, after `start`.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["gain"] = "gain"
    start: float = Field(description="The start of the window.")
    end: float = Field(description="The end of the window.")

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
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


class _AtPositions(_OfSeries):
    """A quantity that reads a series at several positions."""

    positions: list[float] = Field(
        min_length=1, description="The positions, ascending."
    )

    @field_validator("positions")
    @classmethod
    def _ascending(cls, value: list[float]) -> list[float]:
        """The positions ascend, each once."""
        if any(later <= earlier for earlier, later in zip(value, value[1:])):
            raise ValueError("positions must ascend, each once")
        return value

    @model_validator(mode="after")
    def _windows_after_zero(self) -> Self:
        """The window of increments ends at least its width after 0."""
        self.series.check_ends(self.positions)
        return self


class Peak(_AtPositions, _OneScalar, RangeChecked):
    """
    The largest value of a series at the positions.

    The series decides what the peak is: the peak of `Increments(acc,
    window=7)` at the ends 7, 14, ... is the largest gain of a window of 7,
    while the peak of the level of an accumulator, which never falls, is
    its value at the last position. A NaN at a position makes the peak NaN.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    positions : list[float]
        The positions compared, ascending; with `Increments`, window ends.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["peak"] = "peak"

    def default_label(self) -> str:
        """The peak of the series."""
        return f"peak of {self.series.text}"


class PeakPosition(_AtPositions, _OneScalar, RangeChecked):
    """
    The first of the positions where the value of a series is largest; NaN
    when the value at a position is NaN.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    positions : list[float]
        The positions compared, ascending; with `Increments`, window ends.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["peak_position"] = "peak_position"

    def default_label(self) -> str:
        """The position of the peak of the series."""
        return f"position of the peak of {self.series.text}"


class Trajectory(_AtPositions, RangeChecked):
    """
    The value of a series at each position: one scalar per position, for an
    analysis over the axis.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    positions : list[float]
        The positions, ascending; with `Increments`, window ends.
    label : str | None
        The name of the series; each scalar is named after it and its
        position.
    """

    kind: Literal["trajectory"] = "trajectory"

    def _name(self) -> str:
        """The name of the series."""
        return self.label or self.series.text

    def labels(self) -> list[str]:
        """The name and each position."""
        return [f"{self._name()} at {_position_text(at)}" for at in self.positions]

    def axes(self) -> list[tuple[str, float | None]]:
        """Its name and each position."""
        return [(self._name(), at) for at in self.positions]


type _Operand = Annotated[
    ValueAt | Gain | Peak | PeakPosition | Trajectory, Field(discriminator="kind")
]
"""A quantity that is not a ratio; a `Ratio` takes the ones of one value."""


class Ratio(_OneScalar):
    """
    The ratio of two quantities of one value each, numerator N over
    denominator D.

    The ratio of a run whose denominator is 0 is NaN, also when the
    numerator is not 0, and follows the NaN rule of every method. A method
    that reduces the points of a design (`PRCC`, `Morris`, `Sobol`,
    `PolynomialChaos`) reads one value per point; with a stochastic engine,
    `replicate_rule` says how the replicates of a point combine. A `Propagation`,
    the `Scenarios` and the quantities of one run keep the ratio of each run.

    Attributes
    ----------
    numerator : ValueAt | Gain | Peak
        The quantity divided.
    denominator : ValueAt | Gain | Peak
        The quantity it is divided by.
    replicate_rule : {"within", "within_positive", "pooled"}
        ``"within"``: the mean of the ratio of each replicate, E[N/D]; a
        replicate with D = 0 leaves the point with no value.
        ``"within_positive"``: the mean of the ratio of the replicates with
        D > 0, E[N/D | D > 0]; a point with no such replicate has no value.
        ``"pooled"``: the ratio of the means, E[N]/E[D]; a point with
        E[D] = 0 has no value. With a deterministic engine, the three give
        N/D, and a point with D = 0 has no value.
    label : str | None
        The name of the quantity in results and plots. If None, a name that
        says what the quantity reads.
    """

    kind: Literal["ratio"] = "ratio"
    numerator: _Operand = Field(description="The quantity divided.")
    denominator: _Operand = Field(description="The quantity it is divided by.")
    replicate_rule: Literal["within", "within_positive", "pooled"] = Field(
        description="How the replicates of a stochastic run combine."
    )

    @field_validator("numerator", "denominator")
    @classmethod
    def _one_value(cls, value: QuantityBase) -> QuantityBase:
        """An operand gives one value per run."""
        if isinstance(value, PeakPosition | Trajectory):
            raise ValueError(
                f"A Ratio divides quantities of one value each (ValueAt, Gain, "
                f"Peak), got the {type(value).__name__} '{value.labels()[0]}'"
            )
        return value

    def default_label(self) -> str:
        """The numerator over the denominator."""
        numerator = self.numerator.labels()[0]
        denominator = self.denominator.labels()[0]
        return f"({numerator}) / ({denominator})"


type _Quantity = Annotated[
    ValueAt | Gain | Peak | PeakPosition | Trajectory | Ratio,
    Field(discriminator="kind"),
]
"""One of the quantities of interest."""


def quantity_labels(quantities: Sequence[QuantityBase]) -> list[str]:
    """
    The label of every scalar of `quantities`, in order.

    Parameters
    ----------
    quantities : Sequence[quantity]
        The quantities; a `Trajectory` gives one scalar per position.

    Returns
    -------
    list[str]
        The labels.

    Raises
    ------
    ValueError
        If there is no quantity, or two scalars have one label.
    """
    if not quantities:
        raise ValueError("Give at least one quantity of interest.")
    labels = [label for quantity in quantities for label in quantity.labels()]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise ValueError(
            f"Two quantities have the labels {repeated}: give each a label of its own."
        )
    return labels


def quantity_schedule(quantities: Sequence[QuantityBase]) -> list[float]:
    """
    The positions a run must report at for `quantities` to read it.

    They are every position a quantity reads and the start of every window,
    sorted, each once. A `SimulationResults` of
    ``Simulation.run_at(quantity_schedule(quantities))`` holds them all, so
    `SimulationResults.quantities` reads it; a `Propagation` runs at the same
    positions.

    Parameters
    ----------
    quantities : Sequence[quantity]
        The quantities.

    Returns
    -------
    list[float]
        The positions, ascending.

    Raises
    ------
    ValueError
        If there is no quantity.
    """
    if not quantities:
        raise ValueError("Give at least one quantity of interest.")
    return commol_rs.inference.quantity_schedule(engine_json(list(quantities)))
