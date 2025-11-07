from typing import override

from pydantic import BaseModel, Field


class Bin(BaseModel):
    """
    Defines a single bin (base category) in the compartmental model.

    A bin represents a fundamental category before stratification. The combination
    of a bin with all stratifications produces the actual compartments.

    Attributes
    ----------
    id : str
        Identifier of the bin.
    name : str
        A descriptive, human-readable name for the bin.
    unit : str | None
        The unit of measurement for this bin.
        If None, the bin will use the model-level bin_unit when specified.
    """

    id: str = Field(..., description="Identifier of the bin.")
    name: str = Field(..., description="Descriptive, human-readable name for the bin.")
    unit: str | None = Field(
        None,
        description="Unit of measurement for this bin.",
    )

    @override
    def __hash__(self) -> int:
        return hash(self.id)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Bin) and self.id == other.id
