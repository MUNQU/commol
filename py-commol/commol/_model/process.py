"""The process of a model: what changes and how, written once.

A process holds the bins, the accumulators, the stratifications, the
matrices between the categories of a stratification and the transitions.
It is the same for every structure: the structure of a model
(`ModelStructure`) says how the process is laid out, such as amounts per
compartment (`Aggregate`). The engine checks the rules of the process when
the model is built (`Aggregate.check_model`): unique ids, the categories of
each stratification, and what each transition moves.
"""

from pydantic import BaseModel, ConfigDict, Field

from commol._model.bin import Accumulator, Bin
from commol._model.matrix import CategoryMatrix
from commol._model.stratification import Stratification
from commol._model.transition import Transition


class Process(BaseModel):
    """
    What changes in a model and how, written once for every structure.

    Attributes
    ----------
    bins : list[Bin]
        The states of the process. Stratifications expand each one into
        compartments.
    accumulators : list[Accumulator]
        Counters of the flow of the transitions that name them.
    stratifications : list[Stratification]
        Categorical subdivisions of the bins, in expansion order.
    matrices : list[CategoryMatrix]
        Matrices between the categories of a stratification, which rates
        read through ``matrix_sum``.
    transitions : list[Transition]
        Flows between bins, or between the categories of one stratification.
    """

    model_config = ConfigDict(extra="forbid")

    bins: list[Bin] = Field(default_factory=list)
    accumulators: list[Accumulator] = Field(default_factory=list)
    stratifications: list[Stratification] = Field(default_factory=list)
    matrices: list[CategoryMatrix] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """
        Whether the process holds nothing.

        Returns
        -------
        bool
            True when it has no bin, accumulator, stratification, matrix or
            transition.
        """
        return not (
            self.bins
            or self.accumulators
            or self.stratifications
            or self.matrices
            or self.transitions
        )

    def bin(self, bin_id: str) -> Bin | None:
        """
        The bin with an id.

        Parameters
        ----------
        bin_id : str
            Id of a bin.

        Returns
        -------
        Bin | None
            The bin, or None when the process has no bin with that id.
        """
        return next((item for item in self.bins if item.id == bin_id), None)
