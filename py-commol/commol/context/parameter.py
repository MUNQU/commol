from pydantic import BaseModel, Field, field_validator


class TimeSeriesValue(BaseModel):
    """
    An empirical time series for a parameter value.

    Attributes
    ----------
    data : list[tuple[int, float]]
        (step, value) pairs. Need not be pre-sorted; the engine sorts them.
    mode : str
        Interpolation mode: ``"pulse"`` (non-zero only at listed steps),
        ``"step_function"`` (zero-order hold), or ``"linear"`` (interpolation).
    """

    data: list[tuple[int, float]]
    mode: str = Field(default="pulse", pattern="^(pulse|step_function|linear)$")


class Parameter(BaseModel):
    """
    Defines a global model parameter.

    Attributes
    ----------
    id : str
        The identifier of the parameter.
    value : float | str | TimeSeriesValue | None
        Value of the parameter. Can be:
        - float: A numerical constant value
        - str: A mathematical formula that can reference other parameters,
               special variables (N, N_category, step/t, pi, e), or contain
               mathematical expressions
        - TimeSeriesValue: An empirical time series evaluated via binary search
        - None: Indicates that the parameter needs to be calibrated before use
    description : str | None
        A human-readable description of the parameter.
    unit : str | None
        The unit of the parameter (e.g., "1/day", "dimensionless", "person").
        If None, the parameter has no unit specified.
    """

    id: str = Field(default=..., description="Identifier of the parameter.")
    value: float | str | TimeSeriesValue | None = Field(
        default=...,
        description=(
            "Value of the parameter. Can be a float (constant), "
            "str (formula), TimeSeriesValue (empirical series), "
            "or None (requires calibration)."
        ),
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the parameter."
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the parameter (e.g., '1/day', 'dimensionless', 'person').",
    )

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, value: float | str | TimeSeriesValue | None
    ) -> float | str | TimeSeriesValue | None:
        """Validate the parameter value."""
        if value is None or isinstance(value, TimeSeriesValue):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if not value.strip():
            raise ValueError("Formula cannot be empty")
        return value.strip()

    def is_calibrated(self) -> bool:
        """
        Check if the parameter has a value (is calibrated).

        Returns
        -------
        bool
            True if the parameter has a value, False if it needs calibration.
        """
        return self.value is not None
