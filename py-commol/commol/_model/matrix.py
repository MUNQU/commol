"""A matrix between the categories of a stratification.

A rate reads it through ``matrix_sum(M, e)``: in the rate of a compartment of
category ``i``, ``Σ_j M[i, j] · e_j``, with each bin of ``e`` read as its
amount in category ``j`` and ``total`` as the total of ``j``. It also gives
the rates and the formula parameters the variables ``<M>_<i>_<j>``, its
entries, and ``<M>_radius``, its spectral radius. The engine checks the
matrix against the process and names these variables
(``commol_rs.model.matrix_variables``).
"""

from pydantic import BaseModel, ConfigDict, Field


class CategoryMatrix(BaseModel):
    """
    A matrix between the categories of a stratification.

    ``ModelBuilder.add_matrix`` adds one to the process.

    Attributes
    ----------
    id : str
        The id that ``matrix_sum`` reads, and the start of the names of its
        variables.
    stratification : str
        The id of the stratification whose categories are its rows and its
        columns.
    values : dict[str, dict[str, float]]
        ``values[i][j]``, the value of row ``i`` and column ``j``: one finite
        number for each pair of categories of the stratification.
    unit : str | None
        The unit of its values, which the unit check reads.
    description : str | None
        A human-readable description.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default=..., description="Identifier of the matrix.")
    stratification: str = Field(
        default=..., description="Stratification of its rows and columns."
    )
    values: dict[str, dict[str, float]] = Field(
        default=..., description="The value of each row and column, by category."
    )
    unit: str | None = Field(default=None, description="Unit of its values.")
    description: str | None = Field(
        default=None, description="Human-readable description of the matrix."
    )
