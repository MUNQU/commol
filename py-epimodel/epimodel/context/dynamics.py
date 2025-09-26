from typing import Literal

from pydantic import BaseModel

from .transition import Transition
from epimodel.constants import ModelTypes


class Dynamics(BaseModel):
    """
    Defines how the system evolves.

    Attributes
    ----------
    typology : Literal["DifferenceEquations"]
        The type of model.
    transitions : List[Transition]
        A list of rules for state changes.
    """
    typology: Literal[ModelTypes.DIFFERENCE_EQUATIONS]
    transitions: list[Transition]
