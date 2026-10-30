"""Stratifications that split compartments into categories."""

from typing import override

from pydantic import BaseModel, ConfigDict, Field

from commol._model.stratification_condition import StratificationCondition


class Stratification(BaseModel):
    """
    Defines a categorical subdivision of the compartments.

    Attributes
    ----------
    id : str
        Identifier of the stratification.
    categories : list[str]
        The categories, at least one, each once; the engine checks them when
        the model is built.
    description : str | None
        A human-readable description of the stratification.
    conditions : list[StratificationCondition] | None
        When set, this stratification only expands compartments whose
        already-applied stratification categories satisfy ALL conditions.
        Compartments that do not satisfy the conditions are kept without
        appending this stratification's categories.

        May only reference stratifications declared before this one.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default=..., description="Identifier of the stratification.")
    categories: list[str] = Field(
        default=...,
        description="List of the different stratification groups identifiers.",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the stratification."
    )
    conditions: list[StratificationCondition] | None = Field(
        default=None,
        description=(
            "Conditions on previously-declared stratifications that must be "
            "satisfied for this stratification to expand a compartment."
        ),
    )

    @override
    def __hash__(self) -> int:
        return hash(self.id)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Stratification) and self.id == other.id
