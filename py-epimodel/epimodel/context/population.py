import math
from typing import Self

from pydantic import BaseModel, model_validator, field_validator

from epimodel.context.disease_state import DiseaseState
from epimodel.context.dynamics import Transition
from epimodel.context.initial_conditions import InitialConditions
from epimodel.context.stratification import Stratification


class Population(BaseModel):
    """
    Defines the compartments, stratifications, and initial conditions of the population.

    Attributes
    ----------
    disease_states : list[DiseaseState]
        A list of compartments or states that make up the model.
    stratifications : list[Stratification]
        A list of categorical subdivisions of the population.
    initial_conditions: Initialization
        Initial state of the subpopulations and stratifications.
    """

    disease_states: list[DiseaseState]
    stratifications: list[Stratification]
    transitions: list[Transition]
    initial_conditions: InitialConditions

    @field_validator("disease_states")
    @classmethod
    def validate_disease_states_not_empty(
        cls, v: list[DiseaseState]
    ) -> list[DiseaseState]:
        if not v:
            raise ValueError("At least one disease state must be defined.")
        return v

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        """
        Validates that disease state and stratification IDs are unique.
        """
        disease_state_ids = [ds.id for ds in self.disease_states]
        if len(disease_state_ids) != len(set(disease_state_ids)):
            duplicates = [
                item
                for item in set(disease_state_ids)
                if disease_state_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate disease state IDs found: {duplicates}")

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
    def validate_disease_state_initial_conditions(self) -> Self:
        """
        Validates initial conditions against the defined model Subpopulation.
        """
        initial_conditions = self.initial_conditions

        disease_states_map = {state.id: state for state in self.disease_states}

        disease_state_fractions_dict = {
            dsf.disease_state: dsf.fraction
            for dsf in initial_conditions.disease_state_fractions
        }

        actual_state = set(disease_state_fractions_dict.keys())
        expected_state = set(disease_states_map.keys())

        if actual_state != expected_state:
            missing = expected_state - actual_state
            extra = actual_state - expected_state
            raise ValueError(
                (
                    f"Initial disease state fractions keys must exactly match "
                    f"disease state ids. Missing ids: {missing}, Extra ids: {extra}."
                )
            )

        states_sum_fractions = sum(disease_state_fractions_dict.values())
        if not math.isclose(states_sum_fractions, 1.0, abs_tol=1e-6):
            raise ValueError(
                (
                    f"Disease state fractions must sum to 1.0, "
                    f"but got {states_sum_fractions:.7f}."
                )
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
