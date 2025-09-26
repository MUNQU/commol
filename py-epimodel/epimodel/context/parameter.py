from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """
    Defines a global model parameter.

    Attributes
    ----------
    id : str
        The identifier of the parameter.
    description : str | None
        A human-readable description of the parameter.
    value : float
        Numerical value of the parameter.
    """
    id: str = Field(..., description="Identifier of the parameter.")
    description: str | None = Field(
        None, 
        description="Human-readable description of the parameter."
    )
    value: float = Field(
        ..., 
        description="Numerical value of the parameter."
    )
