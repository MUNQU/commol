from pydantic import BaseModel, Field, field_validator


class Parameter(BaseModel):
    """
    Defines a global model parameter.

    Attributes
    ----------
    id : str
        The identifier of the parameter.
    value : float | str | None
        Value of the parameter. Can be:
        - float: A numerical constant value
        - str: A mathematical formula that can reference other parameters,
               special variables (N, N_category, step/t, pi, e), or contain
               mathematical expressions
        - None: Indicates that the parameter needs to be calibrated before use
    description : str | None
        A human-readable description of the parameter.
    unit : str | None
        The unit of the parameter (e.g., "1/day", "dimensionless", "person").
        If None, the parameter has no unit specified.
    """

    id: str = Field(default=..., description="Identifier of the parameter.")
    value: float | str | None = Field(
        default=...,
        description=(
            "Value of the parameter. Can be a float (constant), "
            "str (formula), or None (requires calibration)."
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
    def validate_value(cls, value: float | str | None) -> float | str | None:
        """Validate the parameter value."""
        if value is None:
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
