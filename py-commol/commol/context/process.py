"""The process of a model: what changes and how, written once.

A process holds the bins, the accumulators, the stratifications and the
transitions. It is the same for every structure: the structure of a model
(`ModelStructure`) says how the process is laid out, such as amounts per
compartment (`Aggregate`). This module checks what the process holds by
itself; the structure checks it against the rest of the model.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from commol.context.bin import Accumulator, Bin
from commol.context.stratification import Stratification
from commol.context.transition import Transition


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
    transitions : list[Transition]
        Flows between bins, or between the categories of one stratification.
    """

    model_config = ConfigDict(extra="forbid")

    bins: list[Bin] = Field(default_factory=list)
    accumulators: list[Accumulator] = Field(default_factory=list)
    stratifications: list[Stratification] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> Self:
        """Validate that bin, stratification and transition ids are unique."""
        for label, ids in (
            ("bin", [item.id for item in self.bins]),
            ("stratification", [item.id for item in self.stratifications]),
            ("transition", [item.id for item in self.transitions]),
        ):
            if len(ids) != len(set(ids)):
                duplicates = [item for item in set(ids) if ids.count(item) > 1]
                raise ValueError(f"Duplicate {label} IDs found: {duplicates}")
        return self

    @model_validator(mode="after")
    def _validate_stratified_rates(self) -> Self:
        """Validate that stratified rates name stratifications and categories."""
        strat_map = {strat.id: strat for strat in self.stratifications}

        for transition in self.transitions:
            for idx, stratified_rate in enumerate(transition.stratified_rates or []):
                for condition in stratified_rate.conditions:
                    if condition.stratification not in strat_map:
                        raise ValueError(
                            f"In transition '{transition.id}', stratified rate "
                            f"{idx}: Stratification "
                            f"'{condition.stratification}' not found. "
                            f"Available: {list(strat_map.keys())}"
                        )

                    # A condition without a category only redirects the target.
                    strat = strat_map[condition.stratification]
                    if (
                        condition.category is not None
                        and condition.category not in strat.categories
                    ):
                        raise ValueError(
                            f"In transition '{transition.id}', stratified rate "
                            f"{idx}: Category '{condition.category}' not found "
                            f"in stratification '{condition.stratification}'. "
                            f"Available: {strat.categories}"
                        )
        return self

    @model_validator(mode="after")
    def _validate_transition_ids(self) -> Self:
        """
        Check that each transition moves between bins, or between the
        categories of one stratification.
        """
        bin_ids = {bin_item.id for bin_item in self.bins}
        category_to_stratification = {
            cat: strat.id for strat in self.stratifications for cat in strat.categories
        }
        categories_ids = set(category_to_stratification)
        bin_and_categories_ids = bin_ids | categories_ids

        for transition in self.transitions:
            transition_ids = set(transition.source) | set(transition.target)

            if not transition_ids.issubset(bin_and_categories_ids):
                invalid_ids = transition_ids - bin_and_categories_ids
                raise ValueError(
                    f"Transition '{transition.id}' contains invalid ids: "
                    f"{invalid_ids}. Ids must be defined in Bin ids "
                    f"or Stratification Categories."
                )

            is_bin_flow = transition_ids.issubset(bin_ids)
            is_stratification_flow = transition_ids.issubset(categories_ids)

            if not is_bin_flow and not is_stratification_flow:
                raise ValueError(
                    f"Transition '{transition.id}' mixes id types. "
                    f"Found Bin ids ({transition_ids & bin_ids}) and "
                    f"Stratification Categories ids "
                    f"({transition_ids & categories_ids}). "
                    "Transitions must be purely Bin flow or purely "
                    f"Stratification flow."
                )

            if is_stratification_flow:
                parent_stratification_ids = {
                    category_to_stratification[cat_id] for cat_id in transition_ids
                }
                if len(parent_stratification_ids) > 1:
                    raise ValueError(
                        f"Transition '{transition.id}' is a Stratification flow "
                        f"but involves categories from multiple stratifications: "
                        f"{', '.join(sorted(parent_stratification_ids))}. A single "
                        f"transition must only move between categories belonging "
                        f"to the same parent stratification."
                    )
        return self

    def is_empty(self) -> bool:
        """
        Whether the process holds nothing.

        Returns
        -------
        bool
            True when it has no bin, accumulator, stratification or
            transition.
        """
        return not (
            self.bins or self.accumulators or self.stratifications or self.transitions
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
