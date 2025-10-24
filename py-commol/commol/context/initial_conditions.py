from pydantic import BaseModel, Field


class BinFraction(BaseModel):
    """
    Fraction for a single bin.

    Attributes
    ----------
    bin : str
        The bin id.
    fraction : float
        The fractional size of this bin.
    """

    bin: str = Field(..., description="The bin id.")
    fraction: float = Field(..., description="The fractional size of this bin.")


class StratificationFraction(BaseModel):
    """
    Fractions for a single stratification category.

    Attributes
    ----------
    category : str
        The stratification category name.
    fraction : float
        The fractional size of this category.
    """

    category: str = Field(..., description="The stratification category name.")
    fraction: float = Field(..., description="The fractional size of this category.")


class StratificationFractions(BaseModel):
    """
    Fractions for a stratification.

    Attributes
    ----------
    stratification : str
        The stratification id.
    fractions : list[StratificationFraction]
        List of category fractions for this stratification.
    """

    stratification: str = Field(..., description="The stratification id.")
    fractions: list[StratificationFraction] = Field(
        ..., description="List of category fractions for this stratification."
    )


class InitialConditions(BaseModel):
    """
    Initial conditions for a simulation.

    Attributes
    ----------
    population_size : int
        Population size.
    bin_fractions : list[BinFraction]
        List of bin fractions. Each item contains a bin id and
        its initial fractional size.
    stratification_fractions : list[StratificationFractions], optional
        List of stratification fractions. Each item contains a stratification id and
        its category fractions.
    """

    population_size: int = Field(..., description="Population size.")
    bin_fractions: list[BinFraction] = Field(
        ...,
        description=(
            "List of bin fractions. Each item contains a bin id "
            "and its initial fractional size."
        ),
    )
    stratification_fractions: list[StratificationFractions] = Field(
        default_factory=list,
        description=(
            "List of stratification fractions. Each item contains a stratification id "
            "and its category fractions."
        ),
    )
