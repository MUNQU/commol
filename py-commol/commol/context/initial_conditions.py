"""Initial amounts: the amount of each bin, category splits and overrides.

The engine holds the rule of an initial amount (`check_bin_amount` and
`check_compartment_amount` of `crates/commol-model/src/values.rs`): each
amount is checked with it when the object is built or an amount changes.
"""

from collections.abc import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from commol.commol_rs import _commol_rs as commol_rs

# Separator of a qualified category id, as in ``group:g1``.
CATEGORY_SEPARATOR = ":"

# Fields of a structure that states fractions of one total, and the field
# that states each of them now.
_FRACTION_FIELDS = {
    "total": "bin_amounts",
    "bin_fractions": "bin_amounts",
    "compartment_fractions": "compartment_amounts",
}


class InitialConditions(BaseModel):
    """
    How much starts where: the amount of each bin and the split of it.

    Every amount is in the unit of its bin, so bins of different units, such
    as a count and a concentration, start without a total over them.
    `Aggregate` holds these fields and their methods; the checks against the
    bins and the stratifications of the process are in
    `Aggregate.check_model`.

    Attributes
    ----------
    bin_amounts : dict[str, float | None]
        The amount of each bin at the start, by bin id, in the unit of the
        bin; finite and not negative. None leaves an amount for calibration.
    stratification_fractions : dict[str, dict[str, float]]
        The fraction of each category, by stratification id and category: the
        split of the amount of every bin the stratification applies to; each
        from 0 to 1, and a sum of 1 within 1e-6. A conditional stratification
        splits only the group its conditions select.
    compartment_amounts : dict[str, float]
        Amounts of single expanded compartments, by compartment name, applied
        after the expansion.
    """

    model_config = ConfigDict(extra="forbid")

    bin_amounts: dict[str, float | None] = Field(
        ...,
        description=(
            "The amount of each bin, by bin id, in the unit of the bin. None "
            "leaves an amount for calibration."
        ),
    )
    stratification_fractions: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="The fraction of each category, by stratification id.",
    )
    compartment_amounts: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Amounts of single expanded compartments, applied after the expansion."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def reject_fractions_of_a_total(cls, data: JsonValue) -> JsonValue:
        """Name the field of amounts that replaces a field of fractions.

        A total over bins of different units has no meaning, so the initial
        state is stated as amounts, and a structure with ``total``,
        ``bin_fractions`` or ``compartment_fractions`` is rejected with the
        field to use.
        """
        if isinstance(data, Mapping):
            stated = [field for field in _FRACTION_FIELDS if field in data]
            if stated:
                replacements = sorted({_FRACTION_FIELDS[field] for field in stated})
                raise ValueError(
                    f"An aggregate structure states the amount of each bin in "
                    f"the unit of the bin, not fractions of a total: replace "
                    f"{', '.join(stated)} with {' and '.join(replacements)}, "
                    f'such as bin_amounts={{"A": 990, "B": 10, "C": 0}}.'
                )
        return data

    @field_validator("bin_amounts")
    @classmethod
    def validate_bin_amounts(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        """Check every bin amount with the rule of the engine."""
        commol_rs.model.check_bin_amounts(list(value.items()))
        return value

    @field_validator("stratification_fractions")
    @classmethod
    def validate_stratification_fractions(
        cls, value: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Check the fractions of every stratification with the rule of the
        engine."""
        _check_fractions(value)
        return value

    @field_validator("compartment_amounts")
    @classmethod
    def validate_compartment_amounts(cls, value: dict[str, float]) -> dict[str, float]:
        """Check every compartment amount with the rule of the engine."""
        commol_rs.model.check_compartment_amounts(list(value.items()))
        return value

    def get_uncalibrated_bins(self) -> list[str]:
        """
        Get list of bin IDs that have uncalibrated amounts (value = None).

        Returns
        -------
        list[str]
            List of bin IDs with uncalibrated initial conditions.
        """
        return [bin_id for bin_id, amount in self.bin_amounts.items() if amount is None]

    def get_categories_with_fractions(self) -> set[str]:
        """
        Get the categories that have a declared initial fraction.

        Returns
        -------
        set[str]
            Category names collected from ``stratification_fractions``. A
            category declared on a stratification but absent from the initial
            conditions is not included.
        """
        return {
            category
            for categories in self.stratification_fractions.values()
            for category in categories
        }

    def resolve_category(self, category: str) -> tuple[str, str]:
        """
        Resolve a category id to its stratification and its bare name.

        A qualified id (``group:g1``) names the stratification. A bare name
        must be declared by exactly one stratification.

        Parameters
        ----------
        category : str
            A bare or qualified category id.

        Returns
        -------
        tuple[str, str]
            The stratification id and the bare category name.

        Raises
        ------
        ValueError
            If the category is not found or a bare name is ambiguous.
        """
        stratification_id, separator, name = category.rpartition(CATEGORY_SEPARATOR)
        owners = [
            stratification
            for stratification, categories in self.stratification_fractions.items()
            if (not separator or stratification == stratification_id)
            and name in categories
        ]
        if len(owners) > 1:
            qualified = " or ".join(
                f"'{owner}{CATEGORY_SEPARATOR}{name}'" for owner in owners
            )
            raise ValueError(
                f"Category '{category}' is ambiguous: it is declared by "
                f"stratifications {owners}. Qualify it as {qualified}."
            )
        if not owners:
            raise ValueError(
                f"Category '{category}' not found in initial conditions. "
                f"Available categories: {sorted(self.get_categories_with_fractions())}"
            )
        return owners[0], name

    def category_id(self, stratification: str, category: str) -> str:
        """
        Shortest id that names a category without ambiguity.

        Parameters
        ----------
        stratification : str
            Id of the stratification that declares the category.
        category : str
            Bare category name.

        Returns
        -------
        str
            The bare name, or ``stratification:category`` when another
            stratification declares a category of the same name.
        """
        declaring = sum(
            category in categories
            for categories in self.stratification_fractions.values()
        )
        if declaring > 1:
            return f"{stratification}{CATEGORY_SEPARATOR}{category}"
        return category

    def get_category_fraction(self, category: str) -> float:
        """
        Get the fraction of a category within the group it subdivides.

        Parameters
        ----------
        category : str
            A category id, bare or qualified (``group:g1``).

        Returns
        -------
        float
            The declared fraction, relative to the group the category's
            stratification applies to rather than to the whole model.

        Raises
        ------
        ValueError
            If the category is not found or a bare name is ambiguous.
        """
        stratification, name = self.resolve_category(category)
        return self.stratification_fractions[stratification][name]

    def _updated_fractions(
        self, stratification: str, named: Mapping[str, float]
    ) -> dict[str, float]:
        """The fractions of a single stratification after an update that
        names ``named``; the one category left unnamed takes the rest."""
        categories = self.stratification_fractions[stratification]
        unnamed = [category for category in categories if category not in named]
        if len(unnamed) > 1:
            raise ValueError(
                f"Cannot update stratification '{stratification}': "
                f"categories {sorted(unnamed)} were left unnamed. At most one "
                f"category may be omitted, so that it can take the remaining "
                f"fraction."
            )
        remainder = max(0.0, 1.0 - sum(named.values()))
        return {category: named.get(category, remainder) for category in categories}

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update stratification category fractions by category name.

        Categories belonging to several stratifications may be given in one
        call. A category name declared by more than one stratification must be
        qualified with its stratification, as in ``group:g1``. Within each
        stratification, at most one category may be omitted; the omitted
        category receives the rest of 1, or 0 when the given values pass 1.
        The fractions of every stratification are checked with the rule of the
        engine before any of them changes.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping category names, bare or qualified, to their new
            fraction values.

        Raises
        ------
        ValueError
            If a category is not found or is ambiguous, or a stratification
            has more than one category omitted; with the message of the
            engine, if a fraction is not from 0 to 1 or the fractions of a
            stratification do not sum to 1 within 1e-6.
        """
        by_stratification: dict[str, dict[str, float]] = {}
        for category, value in fractions.items():
            stratification, name = self.resolve_category(category)
            by_stratification.setdefault(stratification, {})[name] = value

        updated = {
            stratification: self._updated_fractions(stratification, named)
            for stratification, named in by_stratification.items()
        }
        _check_fractions(updated)
        for stratification, categories in updated.items():
            self.stratification_fractions[stratification].update(categories)

    def _upsert_compartment_amounts(self, amounts: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_amounts`` has its
        amount replaced; a new one is appended. Compartment names depend on
        the expansion of the process, so
        :meth:`~commol.context.aggregate.Aggregate.update_compartment_amounts`
        validates them against the engine, not here.

        Raises
        ------
        ValueError
            With the message of the engine, if an amount is not finite or is
            negative.
        """
        commol_rs.model.check_compartment_amounts(list(amounts.items()))
        self.compartment_amounts.update(amounts)

    def update_bin_amounts(self, amounts: Mapping[str, float | None]) -> None:
        """
        Update bin amounts by bin ID.

        Every amount is checked before any bin changes, so a rejected call
        leaves the initial conditions unchanged. The bins of one unit keep no
        sum: to move an amount between bins, update every bin involved.

        Parameters
        ----------
        amounts : Mapping[str, float | None]
            Dictionary mapping bin IDs to their new amount, in the unit of the
            bin. None marks a bin that needs calibration.

        Raises
        ------
        ValueError
            If a bin ID is not found in the initial conditions, or, with the
            message of the engine, an amount is not finite or is negative.
        """
        unknown = sorted(set(amounts) - set(self.bin_amounts))
        if unknown:
            raise ValueError(
                f"Bin(s) {unknown} not found in initial conditions. "
                f"Available bins: {list(self.bin_amounts)}"
            )
        commol_rs.model.check_bin_amounts(list(amounts.items()))
        self.bin_amounts.update(amounts)


def _check_fractions(fractions: Mapping[str, Mapping[str, float]]) -> None:
    """Check the fractions of each stratification with the rule of the
    engine."""
    commol_rs.model.check_fractions(
        [
            (stratification, list(categories.items()))
            for stratification, categories in fractions.items()
        ]
    )
