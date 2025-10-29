from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """
    Defines a global model parameter.

    Attributes
    ----------
    id : str
        The identifier of the parameter.
    value : float
        Numerical value of the parameter.
    description : str | None
        A human-readable description of the parameter.
    unit : str | None
        The unit of the parameter (e.g., "1/day", "dimensionless", "person").
        If None, the parameter has no unit specified.
    """

    id: str = Field(..., description="Identifier of the parameter.")
    value: float = Field(..., description="Numerical value of the parameter.")
    description: str | None = Field(
        default=None, description="Human-readable description of the parameter."
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the parameter (e.g., '1/day', 'dimensionless', 'person').",
    )
