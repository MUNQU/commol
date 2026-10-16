"""Initial amounts: the total, and bin, category and compartment fractions."""

import math
from collections.abc import Iterable, Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Separator of a qualified category id, as in ``group:g1``.
CATEGORY_SEPARATOR = ":"


class InitialConditions(BaseModel):
    """
    How much starts where: the total and the fractions of it.

    `Aggregate` holds these fields and their methods; the checks against the
    bins and the stratifications of the process are in
    `Aggregate.check_model`.

    Attributes
    ----------
    total : float
        The sum of every compartment at the start, finite and greater than 0.
        It can be a count, a mass or a concentration.
    bin_fractions : dict[str, float | None]
        The fraction of the total in each bin, by bin id. None leaves a
        fraction for calibration.
    stratification_fractions : dict[str, dict[str, float]]
        The fraction of each category, by stratification id and category.
        A conditional stratification splits only the group its conditions
        select.
    compartment_fractions : dict[str, float]
        Fractions of the total in single expanded compartments, by compartment
        name, applied after the expansion.
    """

    model_config = ConfigDict(extra="forbid")

    total: float = Field(
        ...,
        description="The sum of every compartment at the start, finite and > 0.",
    )
    bin_fractions: dict[str, float | None] = Field(
        ...,
        description=(
            "The fraction of the total in each bin, by bin id. None leaves a "
            "fraction for calibration."
        ),
    )
    stratification_fractions: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="The fraction of each category, by stratification id.",
    )
    compartment_fractions: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Fractions of the total in single expanded compartments, applied "
            "after the expansion."
        ),
    )

    @field_validator("total")
    @classmethod
    def validate_total(cls, value: float) -> float:
        """Enforce a finite total greater than 0.

        Every fraction of a zero total is zero, and the report divides by the
        total, so a zero, negative or non-finite total is rejected as a domain
        error.
        """
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"total must be finite and greater than 0, got {value}.")
        return value

    @field_validator("compartment_fractions")
    @classmethod
    def validate_compartment_fraction_range(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        """Enforce that every compartment fraction lies within ``[0.0, 1.0]``."""
        for compartment, fraction in value.items():
            _check_fraction(f"compartment '{compartment}'", fraction)
        return value

    @staticmethod
    def _validate_bin_fraction_total(candidate: Mapping[str, float | None]) -> None:
        """
        Enforce the bin-fraction conservation rule on a complete mapping.

        Uncalibrated (``None``) fractions floor the calibrated total strictly
        below 1.0 so they can be filled later. When every bin is calibrated,
        the fractions must sum to 1.0, because they are shares of a single
        total. Construction and updates apply the same rule, so an update
        never produces a definition that cannot be loaded again.

        Parameters
        ----------
        candidate : Mapping[str, float | None]
            Fraction of every bin.

        Raises
        ------
        ValueError
            If the fractions break the rule.
        """
        calibrated = [value for value in candidate.values() if value is not None]
        if not calibrated:
            return
        total = sum(calibrated)
        uncalibrated = sum(1 for value in candidate.values() if value is None)

        if uncalibrated > 0:
            if total >= 1.0:
                raise ValueError(
                    f"Calibrated bin fractions sum to {total:.4f}, but must be "
                    f"LESS than 1.0 to leave room for {uncalibrated} "
                    f"uncalibrated fraction(s). Calibrated fractions: "
                    f"{[item for item in candidate.items() if item[1] is not None]}"
                )
        elif not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(
                f"Bin fractions must sum to 1.0, got {total:.4f}. "
                f"Fractions: {list(candidate.values())}"
            )

    @model_validator(mode="after")
    def validate_calibrated_fractions_sum_to_one(self) -> Self:
        """
        Validate that bin fractions sum appropriately.

        Rules:
        - If some fractions are None (uncalibrated): calibrated ones must sum to LESS
            than 1.0
        - If all fractions are calibrated (no None): must sum to exactly 1.0
        - If all fractions are None: skip validation (will be set before simulation)
        """
        self._validate_bin_fraction_total(self.bin_fractions)
        return self

    def get_uncalibrated_bins(self) -> list[str]:
        """
        Get list of bin IDs that have uncalibrated fractions (value = None).

        Returns
        -------
        list[str]
            List of bin IDs with uncalibrated initial conditions.
        """
        return [
            bin_id
            for bin_id, fraction in self.bin_fractions.items()
            if fraction is None
        ]

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

    def _fraction_subgroup_total(self, categories: Iterable[str] = ()) -> float:
        """
        Get the initial total of a subgroup from the fractions alone.

        Category fractions are relative to the group their stratification
        applies to, so the fractions of a nested chain of categories multiply.
        Passing no categories gives the total. Per-compartment overrides are
        not included; `Aggregate.subgroup_total` includes them.
        """
        total = self.total
        for category in categories:
            total *= self.get_category_fraction(category)
        return total

    def _apply_stratification_update(
        self, stratification: str, named: Mapping[str, float]
    ) -> None:
        """Write the fractions of a single stratification."""
        categories = self.stratification_fractions[stratification]
        unnamed = [category for category in categories if category not in named]
        total = sum(named.values())

        if len(unnamed) > 1:
            raise ValueError(
                f"Cannot update stratification '{stratification}': "
                f"categories {sorted(unnamed)} were left unnamed. At most one "
                f"category may be omitted, so that it can take the remaining "
                f"fraction."
            )
        if not unnamed and not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Fractions for stratification '{stratification}' "
                f"must sum to 1.0, got {total:.6f}. Values: {dict(named)}"
            )
        if unnamed and total > 1.0 + 1e-6:
            raise ValueError(
                f"Fractions for stratification '{stratification}' "
                f"sum to {total:.6f}, which would leave a negative fraction for "
                f"'{unnamed[0]}'. Values: {dict(named)}"
            )

        remainder = max(0.0, 1.0 - total)
        for category in categories:
            categories[category] = named.get(category, remainder)

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update stratification category fractions by category name.

        Categories belonging to several stratifications may be given in one
        call. A category name declared by more than one stratification must be
        qualified with its stratification, as in ``group:g1``. Within each
        stratification, at most one category may be omitted; the omitted
        category receives the remaining fraction so that the stratification
        sums to 1.0. When every category is given, the values
        must already sum to 1.0.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping category names, bare or qualified, to their new
            fraction values. Each value must lie between 0.0 and 1.0.

        Raises
        ------
        ValueError
            If a category is not found or is ambiguous, a value lies outside
            [0.0, 1.0], a stratification has more than one category omitted,
            fully specified values do not sum to 1.0, or the given values
            exceed 1.0.
        """
        for category, value in fractions.items():
            _check_fraction(f"category '{category}'", value)

        by_stratification: dict[str, dict[str, float]] = {}
        for category, value in fractions.items():
            stratification, name = self.resolve_category(category)
            by_stratification.setdefault(stratification, {})[name] = value

        for stratification, named in by_stratification.items():
            self._apply_stratification_update(stratification, named)

    def _upsert_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_fractions`` has its
        fraction replaced; a new one is appended. Compartment names and the
        sum of the amounts depend on the expansion of the process, so
        :meth:`~commol.context.aggregate.Aggregate.update_compartment_fractions`
        validates them against the engine, not here.

        Raises
        ------
        ValueError
            If a value lies outside ``[0.0, 1.0]``.
        """
        for compartment, value in fractions.items():
            _check_fraction(f"compartment '{compartment}'", value)
        self.compartment_fractions.update(fractions)

    def update_bin_fractions(self, fractions: Mapping[str, float | None]) -> None:
        """
        Update bin fractions by bin ID.

        The candidate fractions are validated before any bin is mutated, so a
        rejected call leaves the initial conditions unchanged. The conservation
        rule of :meth:`validate_calibrated_fractions_sum_to_one` is re-applied
        to the resulting mapping: the bin fractions must sum to 1.0, while
        uncalibrated (``None``) bins reserve the remainder. To move an amount
        between bins, update every bin involved in one call.

        Parameters
        ----------
        fractions : Mapping[str, float | None]
            Dictionary mapping bin IDs to their new fraction values.
            None values indicate bins that need calibration.

        Raises
        ------
        ValueError
            If a bin ID is not found in the initial conditions, or the
            resulting fractions violate the conservation rule.
        """
        unknown = sorted(set(fractions) - set(self.bin_fractions))
        if unknown:
            raise ValueError(
                f"Bin(s) {unknown} not found in initial conditions. "
                f"Available bins: {list(self.bin_fractions)}"
            )

        candidate = {**self.bin_fractions, **fractions}
        self._validate_bin_fraction_total(candidate)
        self.bin_fractions.update(fractions)


def _check_fraction(name: str, value: float) -> None:
    """Reject a fraction outside ``[0.0, 1.0]``, naming what it is of."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Fraction for {name} must be between 0.0 and 1.0, got {value}."
        )
