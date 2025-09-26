from pydantic import BaseModel, Field

from .condition import Condition


class Transition(BaseModel):
    """
    Defines a rule for system evolution.

    Attributes
    ----------
    id : str
        Id of the transition.
    source : list[str]
        The origin compartments.
    target : list[str]
        The destination compartments.
    rate : str | None
        The mathematical formula for the flow between compartments (ODE).
    condition : Condition | None
        Logical restrictions for the transition.
    """
    id: str = Field(..., description="Id of the transition.")
    source: list[str] = Field(..., description="Origin compartments.")
    target: list[str] = Field(..., description="Destination compartments.")
    
    # --- Fields specific to ODE ---

    rate: str | None = Field(
        ..., 
        description="Mathematical formula for the flow between compartments (ODE)."
    )
    
    # --- Common field ---

    condition: Condition | None = Field(
        None, 
        description="Logical restrictions for the transition."
    )
