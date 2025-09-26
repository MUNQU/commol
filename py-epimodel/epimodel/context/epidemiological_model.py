from typing import Self

from pydantic import BaseModel, Field, model_validator

from .dynamics import Dynamics
from .parameter import Parameter
from .population import Population


class EpidemiologicalModel(BaseModel):
    """
    Root class of epidemiological model.

    Attributes
    ----------
    name : str
        A unique name that identifies the model.
    description : str | None
        A human-readable description of the model's purpose and function.
    version : str | None
        The version number of the model.
    population : Population
        Population details, subpopulations, stratifications and initial conditions.
    parameters : list[Parameter]
        A list of global model parameters.
    dynamics : Dynamics
        The rules that govern system evolution.
    """
    
    name: str = Field(..., description="Name which identifies the model.")
    description: str | None = Field(
        None, 
        description="Human-readable description of the model's purpose and function."
    )
    version: str | None = Field(None, description="Version number of the model.")

    population: Population
    parameters: list[Parameter]
    dynamics: Dynamics
    
    @model_validator(mode="after")
    def validate_transition_ids(self) -> Self:
        """
        Validates that transition ids (source/target) are consistent in type 
        and match the defined DiseaseState IDs or Stratification Categories 
        in the Population instance.
        """
        
        disease_state_ids = {state.id for state in self.population.disease_states}
        categories_ids = {
            cat for strat in self.population.stratifications for cat in strat.categories
        }
        disease_state_and_categories_ids = disease_state_ids.union(categories_ids)

        for transition in self.dynamics.transitions:
            source = set(transition.source)
            target = set(transition.target)
            transition_ids = source.union(target)
            
            if not transition_ids.issubset(disease_state_and_categories_ids):
                invalid_ids = transition_ids - disease_state_and_categories_ids
                raise ValueError((
                    f"Transition '{transition.id}' contains invalid ids: "
                    f"{invalid_ids}. Ids must be defined in DiseaseState ids "
                    f"or Stratification Categories."
                ))
                
            is_disease_state_flow = transition_ids.issubset(disease_state_ids)
            is_stratification_flow = transition_ids.issubset(categories_ids)

            if (not is_disease_state_flow) and (not is_stratification_flow):
                disease_state_elements = transition_ids.intersection(disease_state_ids)
                categories_elements = transition_ids.intersection(categories_ids)
                raise ValueError((
                    f"Transition '{transition.id}' mixes id types. "
                    f"Found DiseaseState ids ({disease_state_elements}) and "
                    f"Stratification Categories ids ({categories_elements}). "
                    "Transitions must be purely Disease State flow or purely "
                    f"Stratification flow."
                ))
            
            if is_stratification_flow:
                category_to_stratification_map = {
                    cat: strat.id 
                    for strat in self.population.stratifications 
                    for cat in strat.categories
                }
                parent_stratification_ids = {
                    category_to_stratification_map[cat_id]
                    for cat_id in transition_ids
                }
                if len(parent_stratification_ids) > 1:
                    mixed_strats = ", ".join(parent_stratification_ids)
                    raise ValueError((
                        f"Transition '{transition.id}' is a Stratification flow but "
                        f"involves categories from multiple stratifications: "
                        f"{mixed_strats}. A single transition must only move between "
                        "categories belonging to the same parent stratification."
                    ))

        return self
