from pydantic import BaseModel, Field


DistributionFractions = dict[str, float]


class InitialConditions(BaseModel):
    """
    Initial conditions for a simulation.
    
    Attributes
    ----------
    population_size : int
        Population size.
    disease_state_fraction : DistributionFractions
        Fractions for disease states. Keys are disease state ids and values are their 
        initial fractional size.
    stratification_fractions : dict[str, DistributionFractions], optional
        Fractions for stratifications. Keys are stratification ids. Values are 
        dictionaries whose keys are the stratification categories and whose values are 
        their initial fractional size.
    """
    population_size: int = Field(..., description="Population size.")  
    disease_state_fraction: DistributionFractions = Field(
        ..., 
        description=(
            "Fractions for disease states. Keys are disease state ids and values are "
            "their initial fractional size."
        )
    )
    stratification_fractions: dict[str, DistributionFractions] = Field(
        default_factory=dict, 
        description=(
            "Fractions for stratifications. Keys are stratification ids. Values are "
            "dictionaries whose keys are the stratification categories and whose "
            "values are their initial fractional size."
        )
    )
