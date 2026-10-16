"""Who the neighbors of a unit are: the terms that carry a process across
structures.

A rate may read ``neighbors_<bin>``, the amount of a bin among the neighbors
of a unit, and ``neighbors``, the total of its neighbors. A rate that reads
them and no compartment is proportional: it is a rate per unit of its source.
The structure says who the neighbors are; the engine resolves the terms
(``commol_rs.model.transition_flows`` shows each flow as it runs).
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

try:
    from commol.commol_rs import _commol_rs as commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e


NEIGHBORS_VARIABLE: str = commol_rs.model.NEIGHBORS
"""The rate variable that reads the total of the neighbors of a unit."""

NEIGHBORS_PREFIX: str = commol_rs.model.NEIGHBORS_PREFIX
"""Prefix of the rate variables ``neighbors_<bin>``."""


class Complete(BaseModel):
    """
    Every unit is a neighbor of every other.

    ``neighbors_<bin>`` is the bin total and ``neighbors`` is ``total``, so
    ``k1 * neighbors_B / neighbors`` from ``A`` moves ``k1 * A * B / total``.

    Attributes
    ----------
    type : Literal["complete"]
        The tag of the neighbors in JSON.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["complete"] = "complete"


class Groups(BaseModel):
    """
    Units are neighbors by the categories of one stratification.

    In a compartment of category ``g``, ``neighbors_<bin>`` is
    ``Σ_g' w[g][g'] × (amount of the bin in g')`` and ``neighbors`` is
    ``Σ_g' w[g][g'] × (amount of every bin in g')``.

    Attributes
    ----------
    type : Literal["groups"]
        The tag of the neighbors in JSON.
    stratification : str
        Id of the stratification whose categories group the units.
    weights : dict[str, dict[str, float]] | None
        ``weights[g][g']``, the weight of category ``g'`` for a unit in ``g``:
        one finite number of at least 0 for every pair of categories. None
        is the identity: a unit sees only its own category.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["groups"] = "groups"
    stratification: str
    weights: dict[str, dict[str, float]] | None = None

    def __init__(
        self,
        stratification: str,
        weights: Mapping[str, Mapping[str, float]] | None = None,
        **data: object,
    ) -> None:
        """
        Group the neighbors of a unit by the categories of a stratification.

        Parameters
        ----------
        stratification : str
            Id of the stratification whose categories group the units.
        weights : Mapping[str, Mapping[str, float]] | None, default=None
            ``weights[g][g']``, the weight of category ``g'`` for a unit in
            ``g``. None is the identity.
        **data : object
            The other fields, as JSON holds them.
        """
        super().__init__(
            stratification=stratification,
            weights=None
            if weights is None
            else {own: dict(row) for own, row in weights.items()},
            **data,
        )


type Neighbors = Annotated[Complete | Groups, Field(discriminator="type")]
"""Who the neighbors of a unit are: `Complete` or `Groups`."""
