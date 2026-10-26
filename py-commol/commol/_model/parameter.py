"""Model parameters: constants, formulas, series and uncalibrated values."""

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from commol.commol_rs import _commol_rs as commol_rs


class SeriesValue(BaseModel):
    """
    An empirical series for a parameter value.

    Attributes
    ----------
    data : list[tuple[float, float]]
        (position, value) points on the axis. Need not be pre-sorted; the
        engine sorts them.
    mode : str
        How the series is read between its points: ``"pulse"`` (the value on
        the step of a point, 0 on every other step; integer positions only),
        ``"step_function"`` (the value of the last point at or before ``t``),
        or ``"linear"`` (linear interpolation in ``t``). The engine decides
        which positions its model type accepts: a DifferenceEquations or
        StochasticSteps model only integer steps. The model rejects a
        position its engine cannot run when it is built or its parameters
        change.
    """

    model_config = ConfigDict(extra="forbid")

    data: list[tuple[float, float]]
    mode: str = Field(default="pulse", pattern="^(pulse|step_function|linear)$")

    @field_validator("data")
    @classmethod
    def _validate_data(
        cls, value: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Check the points with the rule of the engine (`check_series` of
        `crates/commol-model/src/values.rs`): at least one point, each
        position finite, not negative and once, each value finite.
        """
        commol_rs.model.check_series(value)
        return value


class Parameter(BaseModel):
    """
    Defines a global model parameter.

    Attributes
    ----------
    id : str
        The identifier of the parameter.
    value : float | str | SeriesValue | None
        Value of the parameter. Can be:
        - float: A numerical constant value
        - str: A mathematical formula that can reference other parameters,
               special variables (total, total_<category>, step, t, pi, e), or contain
               mathematical expressions
        - SeriesValue: An empirical series evaluated via binary search
        - None: Indicates that the parameter needs to be calibrated before use
    description : str | None
        A human-readable description of the parameter.
    unit : str | None
        The unit of the parameter (e.g., "1/day", "dimensionless", "item").
        If None, the parameter has no unit specified.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default=..., description="Identifier of the parameter.")
    value: float | str | SeriesValue | None = Field(
        default=...,
        description=(
            "Value of the parameter. Can be a float (constant), "
            "str (formula), SeriesValue (empirical series), "
            "or None (requires calibration)."
        ),
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the parameter."
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the parameter (e.g., '1/day', 'dimensionless', 'item').",
    )

    @field_validator("value")
    @classmethod
    def _validate_value(
        cls, value: float | str | SeriesValue | None
    ) -> float | str | SeriesValue | None:
        """Validate the parameter value."""
        if value is None or isinstance(value, SeriesValue):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        return value.strip()

    @model_validator(mode="after")
    def _validate_formula(self) -> Self:
        """Check a formula value as a rate is checked, with the parser of the
        engine."""
        if isinstance(self.value, str):
            try:
                commol_rs.core.MathExpression(self.value).check()
            except ValueError as error:
                raise ValueError(f"Parameter '{self.id}': {error}")
        return self

    def is_calibrated(self) -> bool:
        """
        Check if the parameter has a value (is calibrated).

        Returns
        -------
        bool
            True if the parameter has a value, False if it needs calibration.
        """
        return self.value is not None

    def with_value(
        self, value: float | str | list[tuple[float, float]] | SeriesValue | None
    ) -> "Parameter":
        """
        This parameter with another value.

        Parameters
        ----------
        value : float | str | list[tuple[float, float]] | SeriesValue | None
            The new value, as `Parameter.value` takes it. A list of
            (position, value) points replaces the points of a series and keeps
            its mode; a `SeriesValue` sets both.

        Returns
        -------
        Parameter
            A new parameter with the same id, description and unit.

        Raises
        ------
        ValueError
            If ``value`` is a list and this parameter is not a series, or
            ``value`` is not a valid value; the message names the parameter.
        """
        try:
            if isinstance(value, list):
                if not isinstance(self.value, SeriesValue):
                    raise ValueError(
                        f"Parameter '{self.id}' is not a series, so a list of "
                        "points has no mode: pass a SeriesValue(data=..., mode=...)"
                    )
                value = SeriesValue(data=value, mode=self.value.mode)
            return Parameter(
                id=self.id, value=value, description=self.description, unit=self.unit
            )
        except ValidationError as error:
            reasons = "; ".join(detail["msg"] for detail in error.errors())
            raise ValueError(
                f"Parameter '{self.id}' cannot take {value!r}: {reasons}"
            ) from None
