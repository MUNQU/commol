"""Initial amounts: the amount of each bin, category splits and overrides.

The engine holds the rule of an initial amount (`check_bin_amount` and
`check_compartment_amount` of `crates/commol-model/src/rules.rs`): each
amount is checked with it when the object is built or an amount changes. It
also resolves a category id, bare or qualified (`resolve_category` of
`crates/commol-model/src/initial_conditions.rs`).
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

from commol._data.engine_json import engine_json
from commol.commol_rs import _commol_rs as commol_rs

# Separator of a qualified category id, as in ``group:g1``.
CATEGORY_SEPARATOR = commol_rs.model.CATEGORY_SEPARATOR

# The key of a split by category that names the earlier stratification it
# differs by, as in ``{"by": "group", "g1": {...}, "g2": {...}}``.
FRACTIONS_BY = commol_rs.model.FRACTIONS_BY

# The fractions of one stratification: one split, or a split by category,
# as `StratificationFractions` of `crates/commol-core/src/types/aggregate.rs`
# reads them.
type _Fractions = dict[str, float] | dict[str, str | dict[str, float]]

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
    stratification_fractions : dict[str, dict]
        The split of each stratification, by stratification id: a
        ``dict[str, float]``, the fraction of each category, such as
        ``{"t1": 0.4, "t2": 0.6}``, the split of the amount of every bin the
        stratification applies to; or a
        ``dict[str, str | dict[str, float]]``, a split per category of a
        stratification declared before it, such as
        ``{"by": "group", "g1": {"t1": 0.25, "t2": 0.75}, "g2": {"t1": 0.5,
        "t2": 0.5}}``. Each fraction is from 0 to 1, and each split sums to 1
        within 1e-6. A conditional stratification splits only the group its
        conditions select.
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
    stratification_fractions: dict[str, _Fractions] = Field(
        default_factory=dict,
        description=(
            "The fraction of each category, by stratification id, or a split "
            "per category of an earlier stratification under 'by'."
        ),
    )
    compartment_amounts: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Amounts of single expanded compartments, applied after the expansion."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_fractions_of_a_total(cls, data: JsonValue) -> JsonValue:
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
    def _validate_bin_amounts(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        """Check every bin amount with the rule of the engine."""
        commol_rs.model.check_bin_amounts(list(value.items()))
        return value

    @field_validator("stratification_fractions")
    @classmethod
    def _validate_stratification_fractions(
        cls, value: dict[str, _Fractions]
    ) -> dict[str, _Fractions]:
        """Check the fractions of every stratification with the rule of the
        engine."""
        _check_fractions(value)
        return value

    @field_validator("compartment_amounts")
    @classmethod
    def _validate_compartment_amounts(cls, value: dict[str, float]) -> dict[str, float]:
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
            Category names collected from ``stratification_fractions``, the
            categories of each split of a split by category included. A
            category declared on a stratification but absent from the initial
            conditions is not included.
        """
        return {
            category
            for fractions in self.stratification_fractions.values()
            for category in split_categories(fractions)
        }

    def resolve_category(self, category: str) -> tuple[str, str]:
        """
        Resolve a category id to its stratification and its bare name.

        A qualified id (``group:g1``) names the stratification. A bare name
        must be declared by exactly one stratification. The engine resolves
        it (``commol_rs.model.resolve_category``) among the categories of the
        fractions, as a calibrated initial condition resolves.

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
        return commol_rs.model.resolve_category(
            category,
            [
                (stratification, split_categories(fractions))
                for stratification, fractions in self.stratification_fractions.items()
            ],
        )

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
            category in split_categories(fractions)
            for fractions in self.stratification_fractions.values()
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
            If the category is not found, a bare name is ambiguous, or the
            fractions of its stratification differ by the category of an
            earlier one.
        """
        stratification, name = self.resolve_category(category)
        fractions = self.stratification_fractions[stratification]
        by = split_by(fractions)
        if by is not None:
            raise ValueError(
                f"The fractions of stratification '{stratification}' differ by "
                f"the category of '{by}': read them in stratification_fractions."
            )
        return category_splits(fractions)[0][1][name]

    def _updated_fractions(
        self, stratification: str, named: Mapping[str, float]
    ) -> _Fractions:
        """The fractions of a single stratification after an update that
        names ``named``: in each split, the named categories take their
        values and the one category left unnamed takes the rest."""
        fractions = self.stratification_fractions[stratification]
        splits = {
            of: _updated_split(stratification, split, named)
            for of, split in category_splits(fractions)
        }
        by = split_by(fractions)
        if by is None:
            return splits[None]
        return {FRACTIONS_BY: by, **{str(of): split for of, split in splits.items()}}

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update stratification category fractions by category name.

        Categories belonging to several stratifications may be given in one
        call. A category name declared by more than one stratification must be
        qualified with its stratification, as in ``group:g1``. Within each
        stratification, at most one category may be omitted; the omitted
        category receives the rest of 1, or 0 when the given values pass 1.
        A stratification whose fractions differ by the category of an earlier
        one takes the values in every split, as a calibrated category does.
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
            engine, if a fraction is not from 0 to 1 or a split does not sum
            to 1 within 1e-6.
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
        self.stratification_fractions.update(updated)

    def _upsert_compartment_amounts(self, amounts: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_amounts`` has its
        amount replaced; a new one is appended. Compartment names depend on
        the expansion of the process, so
        :meth:`~commol._model.aggregate.Aggregate.update_compartment_amounts`
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


def split_by(fractions: _Fractions) -> str | None:
    """The stratification a split by category differs by, or None for one
    split: the field ``by`` when it holds a text, as the deserializer of
    `StratificationFractions` in `crates/commol-core/src/types/aggregate.rs`
    reads it, so a category may be named ``by`` in one split."""
    by = fractions.get(FRACTIONS_BY)
    return by if isinstance(by, str) else None


def category_splits(
    fractions: _Fractions,
) -> list[tuple[str | None, dict[str, float]]]:
    """Each split of a stratification, with the category of the stratification
    it differs by: one split with None for one split."""
    if split_by(fractions) is None:
        return [
            (
                None,
                {
                    category: fraction
                    for category, fraction in fractions.items()
                    if isinstance(fraction, int | float)
                },
            )
        ]
    return [
        (of, dict(split)) for of, split in fractions.items() if isinstance(split, dict)
    ]


def _updated_split(
    stratification: str, split: Mapping[str, float], named: Mapping[str, float]
) -> dict[str, float]:
    """A split after an update that names ``named``; the one category left
    unnamed takes the rest."""
    unnamed = [category for category in split if category not in named]
    if len(unnamed) > 1:
        raise ValueError(
            f"Cannot update stratification '{stratification}': "
            f"categories {sorted(unnamed)} were left unnamed. At most one "
            f"category may be omitted, so that it can take the remaining "
            f"fraction."
        )
    remainder = max(0.0, 1.0 - sum(named.values()))
    return {category: named.get(category, remainder) for category in split}


def split_categories(fractions: _Fractions) -> list[str]:
    """The categories the splits of a stratification name, in their order."""
    return list(
        dict.fromkeys(
            category for _, split in category_splits(fractions) for category in split
        )
    )


def _check_fractions(fractions: Mapping[str, _Fractions]) -> None:
    """Check the fractions of each stratification with the rule of the
    engine."""
    commol_rs.model.check_fractions(engine_json(fractions))
