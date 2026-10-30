"""Conditions on stratification categories."""

from pydantic import BaseModel, ConfigDict, Field


class StratificationCondition(BaseModel):
    """
    Specifies a category within a stratification for rate matching.

    The condition holds for a compartment whose category in the
    stratification is ``category``, or one of ``categories``. A condition
    gives one of them, or neither when it only sets ``to``.

    When ``to`` is set, the matched category in the source compartment is
    replaced with the ``to`` category in the target compartment name. This
    enables cross-category transitions within the same bin (e.g., from ``g1``
    to ``g2``).

    Attributes
    ----------
    stratification : str
        The ID of the stratification (e.g., "group", "type")
    category : str | None
        The category within that stratification (e.g., "g1", "t1")
    categories : list[str] | None
        A set of categories within that stratification: the compartment has
        one of them (e.g., ``["g2", "g3"]``). It counts as one condition when
        the most specific stratified rate wins, and it takes no ``to``.
    to : str | None
        Target category override. When set, the target compartment will use
        this category instead of the source compartment's category for this
        stratification.
    """

    model_config = ConfigDict(extra="forbid")

    stratification: str = Field(default=..., description="ID of the stratification")
    category: str | None = Field(
        default=None,
        description=(
            "Category within the stratification. When None, the condition acts as a "
            "target-only override: it does not filter source compartments but still "
            "contributes a 'to' category to the computed target compartment name."
        ),
    )
    categories: list[str] | None = Field(
        default=None,
        description=(
            "Categories within the stratification, one of which the compartment "
            "has. Give it instead of 'category'."
        ),
    )
    to: str | None = Field(
        default=None,
        description=(
            "Target category override. When set, the target compartment uses "
            "this category instead of the source's for this stratification."
        ),
    )

    def names(self) -> list[str]:
        """
        The categories the condition filters by.

        Returns
        -------
        list[str]
            ``[category]``, the set ``categories``, or empty when the
            condition filters nothing.
        """
        if self.category is not None:
            return [self.category]
        return list(self.categories or [])
