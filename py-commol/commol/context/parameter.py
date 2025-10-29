from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """
    Defines a global model parameter.

    Attributes
    ----------
    id : str
        The identifier of the parameter.
    value : float | None
        Numerical value of the parameter. Can be None to indicate that the
        parameter needs to be calibrated before use.
    description : str | None
        A human-readable description of the parameter.
    unit : str | None
        The unit of the parameter (e.g., "1/day", "dimensionless", "person").
        If None, the parameter has no unit specified.
    """

    id: str = Field(default=..., description="Identifier of the parameter.")
    value: float | None = Field(
        default=...,
        description=(
            "Numerical value of the parameter. "
            "Can be None to indicate calibration is required."
        ),
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the parameter."
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the parameter (e.g., '1/day', 'dimensionless', 'person').",
    )

    def is_calibrated(self) -> bool:
        """
        Check if the parameter has a value (is calibrated).

        Returns
        -------
        bool
            True if the parameter has a value, False if it needs calibration.
        """
        return self.value is not None
