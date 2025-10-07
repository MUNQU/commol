import math
from typing import Self

from pydantic import BaseModel, model_validator

from .disease_state import DiseaseState
from .initial_conditions import InitialConditions
from .stratification import Stratification
from .transition import Transition


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

    @model_validator(mode="after")
    def validate_disease_state_initial_conditions(self) -> Self:
        """
        Validates initial conditions against the defined model Subpopulation.
        """
        initial_conditions = self.initial_conditions

        disease_states_map = {state.id: state for state in self.disease_states}

        actual_state = set(initial_conditions.disease_state_fraction.keys())
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

        states_sum_fractions = sum(initial_conditions.disease_state_fraction.values())
        if not math.isclose(states_sum_fractions, 1.0, abs_tol=1e-6):
            raise ValueError(
                (
                    f"Disease state fractions must sum to 1.0, "
                    f"but got {states_sum_fractions:.7f}."
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
