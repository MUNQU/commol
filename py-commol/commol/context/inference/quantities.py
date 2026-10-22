"""Quantities of interest: the scalars a method reads from each run.

A quantity reads a series of a run and reduces it. The series is the level
of some outputs (`Level`, their sum at each position), or their increments
over windows of one width (`Increments`). The reduction is the value at a
position (`ValueAt`), the largest value or its first position over given
positions (`Peak`, `PeakPosition`), or the value at each position
(`Trajectory`). `Gain` is one window given by its start and its end.

Every quantity follows the same rules:

- it reads only the positions it names, with no interpolation;
- outputs are summed at each row before any reduction;
- `PeakPosition` gives the first of the positions where the value is
  largest;
- a NaN in the values a quantity reads makes it NaN;
- with a stochastic engine, a quantity is computed for each replicate.

The quantities cross to Rust as the JSON of their classes, and Rust reduces
each run, through the observation operator of the calibration.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Position = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
"""A position on the axis of the model."""


def _position_text(at: float) -> str:
    """`at` as the shortest text that reads back as it: `repr`, with no `.0`."""
    return repr(at).removesuffix(".0")


def _output_list(value: object) -> object:
    """One output as a list of one."""
    return [value] if isinstance(value, str) else value


class _Outputs(BaseModel):
    """Outputs summed at each row; one name is a list of one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outputs: list[str] = Field(min_length=1, description="The outputs summed.")

    @field_validator("outputs", mode="before")
    @classmethod
    def one_output(cls, value: object) -> object:
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

    def __init__(self, outputs: str | list[str], **data: Any) -> None:
        super().__init__(outputs=outputs, **data)

    @property
    def text(self) -> str:
        """The series as a label reads it: the outputs."""
        return self.output_name

    def check_ends(self, positions: Sequence[float]) -> None:
        """A level reads any position."""


class Increments(_Outputs):
    """
    The increments of some outputs: at each end, the gain of their sum over
    ``(end - window, end]``, as an observation with the window `window`
    reads it.

    The positions of a quantity of increments are window ends, each at
    least `window`, so no window starts before 0.

    Attributes
    ----------
    outputs : list[str]
        The outputs summed; one name is a list of one.
    window : float
        The width of each window, above 0.
    """

    kind: Literal["increments"] = "increments"
    window: float = Field(gt=0.0, allow_inf_nan=False, description="The width.")

    def __init__(self, outputs: str | list[str], window: float, **data: Any) -> None:
        super().__init__(outputs=outputs, **{**data, "window": window})

    @property
    def text(self) -> str:
        """The series as a label reads it: the gain over windows of a width."""
        return f"{self.output_name} gain over windows of {_position_text(self.window)}"

    def check_ends(self, positions: Sequence[float]) -> None:
        """
        Each end is at least the window, so no window starts before 0.

        Raises
        ------
        ValueError
            If an end is less than the window.
        """
        early = [at for at in positions if at < self.window]
        if early:
            raise ValueError(
                f"Increments over windows of {_position_text(self.window)} start "
                f"at 0 at the earliest, so each end is at least the window; got "
                f"the end {_position_text(early[0])}"
            )


type Series = Annotated[Level | Increments, Field(discriminator="kind")]
"""What a quantity reads at each position: a `Level` or `Increments`."""


def _series(value: object) -> object:
    """A name or a list of names as the `Level` of those outputs."""
    if isinstance(value, str | list):
        return Level(outputs=value)
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

    def rust_json(self) -> str:
        """The quantity as Rust reads it: its JSON, without the label."""
        return self.model_dump_json(exclude={"label"})


class _OfSeries(QuantityBase):
    """A quantity that reads a series."""

    series: Series = Field(description="The series read.")

    @field_validator("series", mode="before")
    @classmethod
    def level_of_outputs(cls, value: object) -> object:
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


class ValueAt(_OfSeries, _OneScalar):
    """
    The value of a series at a position: the level of the outputs, or their
    gain over the window that ends there.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    at : float
        The position; with `Increments`, the end of the window.
    """

    kind: Literal["value_at"] = "value_at"
    at: Position = Field(description="The position.")

    @model_validator(mode="after")
    def window_after_zero(self) -> Self:
        """The window of increments ends at least its width after 0."""
        self.series.check_ends([self.at])
        return self

    def default_label(self) -> str:
        """The series at `at`."""
        return f"{self.series.text} at {_position_text(self.at)}"


class Gain(_Outputs, _OneScalar):
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
    """

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


class _AtPositions(_OfSeries):
    """A quantity that reads a series at several positions."""

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

    @model_validator(mode="after")
    def windows_after_zero(self) -> Self:
        """The window of increments ends at least its width after 0."""
        self.series.check_ends(self.positions)
        return self


class Peak(_AtPositions, _OneScalar):
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
    """

    kind: Literal["peak"] = "peak"

    def default_label(self) -> str:
        """The peak of the series."""
        return f"peak of {self.series.text}"


class PeakPosition(_AtPositions, _OneScalar):
    """
    The first of the positions where the value of a series is largest; NaN
    when the value at a position is NaN.

    Attributes
    ----------
    series : Level | Increments
        The series read; a name or a list of names is their `Level`.
    positions : list[float]
        The positions compared, ascending; with `Increments`, window ends.
    """

    kind: Literal["peak_position"] = "peak_position"

    def default_label(self) -> str:
        """The position of the peak of the series."""
        return f"position of the peak of {self.series.text}"


class Trajectory(_AtPositions):
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


type Quantity = Annotated[
    ValueAt | Gain | Peak | PeakPosition | Trajectory, Field(discriminator="kind")
]
"""One of the quantities of interest."""


def expand_quantities(quantities: Sequence[QuantityBase]) -> tuple[list[str], str]:
    """
    The label of every scalar of `quantities`, in order, and the quantities
    as Rust reads them.

    Parameters
    ----------
    quantities : Sequence[Quantity]
        The quantities; a `Trajectory` gives one scalar per position.

    Returns
    -------
    tuple[list[str], str]
        The labels, and the JSON list of the quantities.

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
    return labels, rust_quantities(quantities)


def rust_quantities(quantities: Sequence[QuantityBase]) -> str:
    """The JSON list of `quantities` as Rust reads them, without labels."""
    return "[" + ",".join(quantity.rust_json() for quantity in quantities) + "]"
