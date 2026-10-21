"""Values in units, converted to steps of a model axis, and rates in steps."""

import math
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from commol.commol_rs import _commol_rs as commol_rs

from commol.context import _windows as windows
from commol.context._model.units import DEFAULT_STEP_UNIT, check_step_unit
from commol.utils.equations import define_units, ureg

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import EngineTypeProtocol

WHOLE_STEP_TOLERANCE = 1e-9
"""A number of steps this close to an integer is that integer."""


class AxisUnits(BaseModel):
    """
    The unit of one step on a model axis, and the engine that reads its rates.

    The axis of a model is its independent variable, of any quantity: time
    (``"day"``), a length (``"meter"``) or ``"step"``, the default increment.
    `steps` converts a value in a unit of the same dimension to steps.
    `rate_from_mean_span` and `rate_from_probability` ask the engine what a
    rate means (`rate_meaning`) and give the rate it reads. `window_start`,
    `window_end`, `window_index` and `window_ends` give the grid of windows of
    one width, a span in a unit: window ``w`` covers the positions
    ``(w * width, (w + 1) * width]``.

    `Model.axis_units` builds the one of a model: its ``step_unit`` and its
    engine.

    Attributes
    ----------
    step_unit : str
        Unit of one step, of one dimension with exponent 1. A name that
        `pint` does not define is a unit of its own dimension.
    engine : str | None
        Name of the engine that reads the rates, such as
        ``"DifferenceEquations"``. None allows only `steps`.

    Examples
    --------
    >>> hours = AxisUnits(step_unit="hour")
    >>> hours.steps(3, "day")
    72.0
    >>> daily = AxisUnits(step_unit="day", engine="DifferentialEquations")
    >>> daily.rate_from_mean_span(2, "week")
    0.07142857142857142
    >>> hours.window_end(2, 1, "week")
    504.0
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_unit: str = Field(
        default=DEFAULT_STEP_UNIT, description="Unit of one step on the axis."
    )
    engine: str | None = Field(
        default=None, description="Name of the engine that reads the rates."
    )

    @model_validator(mode="after")
    def check_units_and_engine(self) -> Self:
        """Check the step unit, and ask the engine's list for the engine."""
        define_units(self.step_unit)
        check_step_unit(self.step_unit)
        if self.engine is not None:
            commol_rs.engine.EngineType(self.engine)
        return self

    def steps(self, value: float, unit: str) -> float:
        """
        Number of steps that a value in `unit` spans on the axis.

        A result within ``1e-9`` of an integer is that integer, so
        ``steps(4.1, "hour")`` with a step of ``minute`` is ``246.0``, not
        ``245.99999999999997``. Whether a position must be a whole number of
        steps is the engine's check.

        Parameters
        ----------
        value : float
            The value, in `unit`.
        unit : str
            A unit of the dimension of the step unit, such as ``"day"`` for a
            step of ``"hour"``.

        Returns
        -------
        float
            The value in steps.

        Raises
        ------
        ValueError
            If `unit` has another dimension than the step unit; the message
            names both units, and says to declare a step unit when the step
            unit is ``step``.
        """
        define_units(unit)
        try:
            quantity = ureg.Quantity(float(value), ureg.parse_units(unit))
            result = float(quantity.to(ureg.parse_units(self.step_unit)).magnitude)
        except Exception as error:
            message = (
                f"Cannot convert {value} {unit} to steps of '{self.step_unit}': "
                f"'{unit}' has dimension "
                f"{ureg.parse_units(unit).dimensionality} and '{self.step_unit}' "
                f"has dimension {ureg.parse_units(self.step_unit).dimensionality}."
            )
            if self.step_unit == DEFAULT_STEP_UNIT:
                message += (
                    " Declare the unit of a step (step_unit in ModelBuilder) "
                    "to convert values in that unit."
                )
            raise ValueError(message) from error
        nearest = round(result)
        if math.isclose(result, nearest, rel_tol=0.0, abs_tol=WHOLE_STEP_TOLERANCE):
            return float(nearest)
        return result

    def rate_from_mean_span(self, span: float, unit: str) -> float:
        """
        Rate of one outflow whose residence has a mean `span` on the axis.

        The engine says what a rate means (`rate_meaning`). For a per-step
        rate (``DifferenceEquations``) it is ``1 - exp(-1/n)``, with `span`
        equal to ``n`` steps: the bin keeps ``exp(-k/n)`` of its start at each
        step ``k``, as the continuous process does, and its mean residence is
        ``1/rate`` steps, a little more than ``n``. For a continuous rate
        (``DifferentialEquations``) it is ``1/n``.

        Parameters
        ----------
        span : float
            Mean span of the residence, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.

        Returns
        -------
        float
            The rate, per step.

        Raises
        ------
        ValueError
            If there is no engine, `unit` has another dimension than the step
            unit, or `span` is not positive.
        """
        return self._engine_type().rate_from_mean_span(self.steps(span, unit))

    def rate_from_probability(
        self, probability: float, period: float, unit: str
    ) -> float:
        """
        Rate of one outflow that moves `probability` of the bin over a period.

        The engine says what a rate means (`rate_meaning`). For a per-step
        rate (``DifferenceEquations``) it is ``1 - (1 - p)^(1/n)``, with the
        period equal to ``n`` steps; for a continuous rate
        (``DifferentialEquations``) it is ``-ln(1 - p)/n``. Either way the
        bin keeps ``1 - p`` of its start after the period.

        Parameters
        ----------
        probability : float
            Fraction of the bin that leaves over the period, between 0 and 1.
            A continuous rate needs it below 1.
        period : float
            Length of the period, in `unit`.
        unit : str
            Unit of `period`, of the dimension of the step unit.

        Returns
        -------
        float
            The rate, per step.

        Raises
        ------
        ValueError
            If there is no engine, `unit` has another dimension than the step
            unit, `period` is not positive, or `probability` is outside the
            range the engine accepts.
        """
        return self._engine_type().rate_from_probability(
            probability, self.steps(period, unit)
        )

    def window_start(self, index: int, span: float, unit: str) -> float:
        """
        Position at which window `index` opens.

        Parameters
        ----------
        index : int
            Zero-based window index.
        span : float
            Width of one window, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.

        Returns
        -------
        float
            ``index`` times the width in steps.

        Raises
        ------
        ValueError
            If the width is not a positive whole number of steps, `unit` has
            another dimension than the step unit, or `index` is negative.
        """
        return windows.window_start(index, self._window_steps(span, unit))

    def window_end(self, index: int, span: float, unit: str) -> float:
        """
        Position at which window `index` closes, equal to the start of the next.

        Parameters
        ----------
        index : int
            Zero-based window index.
        span : float
            Width of one window, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.

        Returns
        -------
        float
            ``index + 1`` times the width in steps.

        Raises
        ------
        ValueError
            If the width is not a positive whole number of steps, `unit` has
            another dimension than the step unit, or `index` is negative.
        """
        return windows.window_end(index, self._window_steps(span, unit))

    def window_index(self, at: float, span: float, unit: str) -> int:
        """
        Window that holds a position.

        The exact inverse of `window_end`, and the window that holds any
        position in between: ``max(ceil(at / width) - 1, 0)``, with the width
        in steps. Position 0 belongs to window 0. A report places each
        observation with this rule.

        Parameters
        ----------
        at : float
            A position on the axis, in steps.
        span : float
            Width of one window, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.

        Returns
        -------
        int
            Zero-based window index.

        Raises
        ------
        ValueError
            If the width is not a positive whole number of steps, `unit` has
            another dimension than the step unit, or `at` is negative.
        """
        return windows.window_index(at, self._window_steps(span, unit))

    def window_ends(self, span: float, unit: str, end: float) -> list[float]:
        """
        Positions that close each complete window up to a position.

        Parameters
        ----------
        span : float
            Width of one window, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.
        end : float
            Last position covered, in steps, such as the last position of a
            run.

        Returns
        -------
        list[float]
            The width in steps, two widths, ... up to `end`. A trailing
            partial window is not included.

        Raises
        ------
        ValueError
            If the width is not a positive whole number of steps, or `unit`
            has another dimension than the step unit.
        """
        return windows.window_ends(self._window_steps(span, unit), end)

    @property
    def rate_meaning(self) -> str:
        """
        What a rate means to the engine: ``"per_step"`` or ``"continuous"``.

        Raises
        ------
        ValueError
            If there is no engine.
        """
        return self._engine_type().rate_meaning

    def _window_steps(self, span: float, unit: str) -> float:
        """Width of a window in steps; it must be a positive whole number."""
        width = self.steps(span, unit)
        if not (width.is_integer() and width > 0):
            raise ValueError(
                f"A window must be a positive whole number of steps: {span} {unit} "
                f"is {width:g} steps of '{self.step_unit}'."
            )
        return width

    def _engine_type(self) -> "EngineTypeProtocol":
        if self.engine is None:
            raise ValueError(
                "A rate needs the engine that reads it: give AxisUnits an "
                "engine, or use Model.axis_units()."
            )
        return commol_rs.engine.EngineType(self.engine)
