from typing import override

from pydantic import BaseModel, Field


class DiseaseState(BaseModel):
    """
    Defines a single disease state of a person regarding the disease.

    Attributes
    ----------
    id : str
        Identifier of the disease state.
    name : str
        A descriptive, human-readable name for the disease state.
    """
    id: str = Field(
        ...,
        description="Identifier of the disease state."
    )
    name: str = Field(
        ...,
        description="Descriptive, human-readable name for the disease state."
    )

    @override 
    def __hash__(self):
            return hash(self.id)

    @override 
    def __eq__(self, other: object):
        return isinstance(other, DiseaseState) and self.id == other.id
