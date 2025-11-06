import math
from typing import Self

from pydantic import BaseModel, model_validator, field_validator

from commol.context.bin import Bin
from commol.context.dynamics import Transition
from commol.context.initial_conditions import InitialConditions
from commol.context.stratification import Stratification


class Population(BaseModel):
    """
    Defines the compartments, stratifications, and initial conditions of the population.

    Attributes
    ----------
    disease_states : list[Bin]
        A list of compartments or states that make up the model.
    stratifications : list[Stratification]
        A list of categorical subdivisions of the population.
    initial_conditions: Initialization
        Initial state of the subpopulations and stratifications.
    """

    bins: list[Bin]
    stratifications: list[Stratification]
    transitions: list[Transition]
    initial_conditions: InitialConditions

    @field_validator("bins")
    @classmethod
    def validate_bins_not_empty(cls, v: list[Bin]) -> list[Bin]:
        if not v:
            raise ValueError("At least one bin must be defined.")
        return v

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        """
        Validates that bin and stratification IDs are unique.
        """
        bin_ids = [ds.id for ds in self.bins]
        if len(bin_ids) != len(set(bin_ids)):
            duplicates = [item for item in set(bin_ids) if bin_ids.count(item) > 1]
            raise ValueError(f"Duplicate bin IDs found: {duplicates}")

        stratification_ids = [s.id for s in self.stratifications]
        if len(stratification_ids) != len(set(stratification_ids)):
            duplicates = [
                item
                for item in set(stratification_ids)
                if stratification_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate stratification IDs found: {duplicates}")

        return self

    @model_validator(mode="after")
    def validate_bin_initial_conditions(self) -> Self:
        """
        Validates initial conditions against the defined model bins.
        """
        initial_conditions = self.initial_conditions

        bins_map = {bin_item.id: bin_item for bin_item in self.bins}

        bin_fractions_dict = {
            bf.bin: bf.fraction for bf in initial_conditions.bin_fractions
        }

        actual_bins = set(bin_fractions_dict.keys())
        expected_bins = set(bins_map.keys())

        if actual_bins != expected_bins:
            missing = expected_bins - actual_bins
            extra = actual_bins - expected_bins
            raise ValueError(
                (
                    f"Initial bin fractions keys must exactly match "
                    f"bin ids. Missing ids: {missing}, Extra ids: {extra}."
                )
            )

        # Only validate sum if all fractions are calibrated (not None)
        # The InitialConditions validator handles the None case
        if all(frac is not None for frac in bin_fractions_dict.values()):
            # Filter out None values before summing (type checker requirement)
            calibrated_fractions = [
                f for f in bin_fractions_dict.values() if f is not None
            ]
            bins_sum_fractions = sum(calibrated_fractions)
            if not math.isclose(bins_sum_fractions, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"Bin fractions must sum to 1.0, but got {bins_sum_fractions:.7f}."
                )

        return self

    @model_validator(mode="after")
    def validate_stratified_rates(self) -> Self:
        """
        Validates that stratified rates reference existing stratifications and
        categories.
        """
        strat_map = {strat.id: strat for strat in self.stratifications}

        for transition in self.transitions:
            if transition.stratified_rates:
                for idx, stratified_rate in enumerate(transition.stratified_rates):
                    for condition in stratified_rate.conditions:
                        # Validate stratification exists
                        if condition.stratification not in strat_map:
                            raise ValueError(
                                (
                                    f"In transition '{transition.id}', stratified rate "
                                    f"{idx}: Stratification "
                                    f"'{condition.stratification}' not found. "
                                    f"Available: {list(strat_map.keys())}"
                                )
                            )

                        # Validate category exists
                        strat = strat_map[condition.stratification]
                        if condition.category not in strat.categories:
                            raise ValueError(
                                (
                                    f"In transition '{transition.id}', stratified rate "
                                    f"{idx}: Category '{condition.category}' not found "
                                    f"in stratification '{condition.stratification}'. "
                                    f"Available: {strat.categories}"
                                )
                            )

        return self

    @model_validator(mode="after")
    def validate_stratification_initial_conditions(self) -> Self:
        """
        Validates initial conditions against the defined model Stratification.
        """
        initial_conditions = self.initial_conditions

        strat_map = {strat.id: strat for strat in self.stratifications}

        actual_strat = {
            sf.stratification for sf in initial_conditions.stratification_fractions
        }
        expected_strat = set(strat_map.keys())

        if actual_strat != expected_strat:
            missing = expected_strat - actual_strat
            extra = actual_strat - expected_strat
            raise ValueError(
                (
                    f"Initial stratification fractions keys must exactly match "
                    f"stratification ids. Missing ids: {missing}, Extra ids: {extra}."
                )
            )

        for strat_fractions in initial_conditions.stratification_fractions:
            strat_id = strat_fractions.stratification
            strat_instance = strat_map[strat_id]

            fractions_dict = {
                sf.category: sf.fraction for sf in strat_fractions.fractions
            }

            categories_expected = set(strat_instance.categories)
            categories_actual = set(fractions_dict.keys())

            if categories_actual != categories_expected:
                missing = categories_expected - categories_actual
                extra = categories_actual - categories_expected
                raise ValueError(
                    (
                        f"Categories for stratification '{strat_id}' must exactly "
                        f"match defined categories in instance '{strat_instance.id}'. "
                        f"Missing categories: {missing}, Extra categories: {extra}."
                    )
                )

            strat_sum_fractions = sum(fractions_dict.values())
            if not math.isclose(strat_sum_fractions, 1.0, abs_tol=1e-6):
                raise ValueError(
                    (
                        f"Stratification fractions for '{strat_id}' must sum to 1.0, "
                        f"but got {strat_sum_fractions:.7}."
                    )
                )

        return self
