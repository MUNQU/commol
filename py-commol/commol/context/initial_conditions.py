"""Initial conditions: population size, bin, category and compartment fractions."""

import math
from collections.abc import Iterable, Mapping
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

# Separator of a qualified category id, as in ``sex:female``.
CATEGORY_SEPARATOR = ":"


class BinFraction(BaseModel):
    """
    Fraction for a single bin.

    Attributes
    ----------
    bin : str
        The bin id.
    fraction : float | None
        The fractional size of this bin. Set to None if this initial condition
        needs to be calibrated.
    """

    bin: str = Field(default=..., description="The bin id.")
    fraction: float | None = Field(
        default=...,
        description=(
            "The fractional size of this bin. "
            "Set to None if this initial condition needs to be calibrated."
        ),
    )

    def is_calibrated(self) -> bool:
        """Check if this bin fraction has a calibrated value (not None)."""
        return self.fraction is not None


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

    category: str = Field(default=..., description="The stratification category name.")
    fraction: float = Field(
        default=..., description="The fractional size of this category."
    )


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
        default=..., description="List of category fractions for this stratification."
    )


class CompartmentFraction(BaseModel):
    """
    Fraction for a single expanded compartment.

    Attributes
    ----------
    compartment : str
        The expanded compartment name, for example ``IH_y60`` or ``S_oe60_v_c``.
    fraction : float
        The fraction of the total population in this compartment,
        between 0.0 and 1.0.
    """

    compartment: str = Field(default=..., description="The expanded compartment name.")
    fraction: float = Field(
        default=...,
        description="The fraction of the total population in this compartment.",
    )

    @field_validator("fraction")
    @classmethod
    def validate_fraction_range(cls, value: float) -> float:
        """Enforce that the fraction lies within ``[0.0, 1.0]``."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Fraction must be between 0.0 and 1.0, got {value}.")
        return value


class InitialConditions(BaseModel):
    """
    Initial conditions for a simulation.

    Attributes
    ----------
    population_size : int
        Population size.
    bin_fractions : list[BinFraction]
        List of bin fractions. Each item contains a bin id and
        its initial fractional size. Fractions can be None if they need calibration.
    stratification_fractions : list[StratificationFractions], optional
        List of stratification fractions. Each item contains a stratification id and
        its category fractions.
    compartment_fractions : list[CompartmentFraction], optional
        List of per-compartment overrides applied after the uniform expansion. Each
        item contains an expanded compartment name and the fraction of the total
        population it holds.
    """

    population_size: int = Field(..., description="Population size.")
    bin_fractions: list[BinFraction] = Field(
        default=...,
        description=(
            "List of bin fractions. Each item contains a bin id and its initial "
            "fractional size. Fractions can be None if they need calibration."
        ),
    )
    stratification_fractions: list[StratificationFractions] = Field(
        default_factory=list,
        description=(
            "List of stratification fractions. Each item contains a stratification id "
            "and its category fractions."
        ),
    )
    compartment_fractions: list[CompartmentFraction] = Field(
        default_factory=list,
        description=(
            "List of per-compartment overrides applied after the uniform expansion. "
            "Each item contains an expanded compartment name and the fraction of the "
            "total population it holds."
        ),
    )

    @field_validator("population_size")
    @classmethod
    def validate_population_size(cls, value: int) -> int:
        """Enforce a strictly positive population size.

        A zero population size would make every fraction-to-head-count
        conversion a division by zero, so it is rejected as a domain error
        rather than surfacing later as ``ZeroDivisionError``.
        """
        if value <= 0:
            raise ValueError(f"population_size must be greater than 0, got {value}.")
        return value

    @model_validator(mode="after")
    def validate_compartment_fractions_unique(self) -> Self:
        """Validate that each compartment appears at most once."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for compartment_fraction in self.compartment_fractions:
            name = compartment_fraction.compartment
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate compartment fractions found: {sorted(duplicates)}."
            )
        return self

    @staticmethod
    def _validate_bin_fraction_total(candidate: Mapping[str, float | None]) -> None:
        """
        Enforce the bin-fraction conservation rule on a complete mapping.

        Uncalibrated (``None``) fractions floor the calibrated total strictly
        below 1.0 so they can be filled later. When every bin is calibrated,
        the fractions must sum to 1.0, because they are shares of a single
        population. Construction and updates apply the same rule, so an update
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
        self._validate_bin_fraction_total(
            {bf.bin: bf.fraction for bf in self.bin_fractions}
        )
        return self

    def get_uncalibrated_bins(self) -> list[str]:
        """
        Get list of bin IDs that have uncalibrated fractions (value = None).

        Returns
        -------
        list[str]
            List of bin IDs with uncalibrated initial conditions.
        """
        return [bf.bin for bf in self.bin_fractions if bf.fraction is None]

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
            fraction.category
            for stratification in self.stratification_fractions
            for fraction in stratification.fractions
        }

    def resolve_category(self, category: str) -> tuple[str, str]:
        """
        Resolve a category id to its stratification and its bare name.

        A qualified id (``sex:female``) names the stratification. A bare name
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
        owner, name = self._owning_stratification(category)
        return owner.stratification, name

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
            any(fraction.category == category for fraction in item.fractions)
            for item in self.stratification_fractions
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
            A category id, bare or qualified (``sex:female``).

        Returns
        -------
        float
            The declared fraction, relative to the group the category's
            stratification applies to rather than to the whole population.

        Raises
        ------
        ValueError
            If the category is not found or a bare name is ambiguous.
        """
        owner, name = self._owning_stratification(category)
        return next(
            fraction.fraction
            for fraction in owner.fractions
            if fraction.category == name
        )

    def subgroup_population(self, categories: Iterable[str] = ()) -> float:
        """
        Get the initial head count of a subgroup.

        Category fractions are relative to the group their stratification
        applies to, so the fractions of a nested chain of categories multiply.
        Passing no categories gives the whole population.

        Parameters
        ----------
        categories : Iterable[str], optional
            Category ids, bare or qualified, forming a chain from the whole
            population down to the subgroup, outermost first.

        Returns
        -------
        float
            Number of individuals in the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found or a bare name is ambiguous.
        """
        population = float(self.population_size)
        for category in categories:
            population *= self.get_category_fraction(category)
        return population

    def _owning_stratification(
        self, category: str
    ) -> tuple[StratificationFractions, str]:
        """
        Return the stratification that declares a category, and its bare name.

        A qualified id (``sex:female``) names the stratification. A bare name
        must be declared by exactly one stratification.
        """
        stratification_id, separator, name = category.rpartition(CATEGORY_SEPARATOR)
        owners = [
            stratification
            for stratification in self.stratification_fractions
            if (not separator or stratification.stratification == stratification_id)
            and any(fraction.category == name for fraction in stratification.fractions)
        ]
        if len(owners) > 1:
            qualified = " or ".join(
                f"'{owner.stratification}{CATEGORY_SEPARATOR}{name}'"
                for owner in owners
            )
            raise ValueError(
                f"Category '{category}' is ambiguous: it is declared by "
                f"stratifications {[owner.stratification for owner in owners]}. "
                f"Qualify it as {qualified}."
            )
        if not owners:
            raise ValueError(
                f"Category '{category}' not found in initial conditions. "
                f"Available categories: {sorted(self.get_categories_with_fractions())}"
            )
        return owners[0], name

    def _apply_stratification_update(
        self,
        stratification: StratificationFractions,
        named: Mapping[str, float],
    ) -> None:
        """Write the fractions of a single stratification."""
        unnamed = [
            fraction.category
            for fraction in stratification.fractions
            if fraction.category not in named
        ]
        total = sum(named.values())

        if len(unnamed) > 1:
            raise ValueError(
                f"Cannot update stratification '{stratification.stratification}': "
                f"categories {sorted(unnamed)} were left unnamed. At most one "
                f"category may be omitted, so that it can take the remaining "
                f"fraction."
            )
        if not unnamed and not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Fractions for stratification '{stratification.stratification}' "
                f"must sum to 1.0, got {total:.6f}. Values: {dict(named)}"
            )
        if unnamed and total > 1.0 + 1e-6:
            raise ValueError(
                f"Fractions for stratification '{stratification.stratification}' "
                f"sum to {total:.6f}, which would leave a negative fraction for "
                f"'{unnamed[0]}'. Values: {dict(named)}"
            )

        remainder = max(0.0, 1.0 - total)
        for fraction in stratification.fractions:
            fraction.fraction = named.get(fraction.category, remainder)

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update stratification category fractions by category name.

        Categories belonging to several stratifications may be given in one
        call. A category name declared by more than one stratification must be
        qualified with its stratification, as in ``sex:female``. Within each
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
            [0.0, 1.0], a stratification has more than one category omitted, fully
            specified
            values do not sum to 1.0, or the given values exceed 1.0.
        """
        for category, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Fraction for category '{category}' must be between 0.0 and "
                    f"1.0, got {value}."
                )

        by_stratification: dict[str, dict[str, float]] = {}
        for category, value in fractions.items():
            owner, name = self._owning_stratification(category)
            by_stratification.setdefault(owner.stratification, {})[name] = value

        for stratification in self.stratification_fractions:
            named = by_stratification.get(stratification.stratification)
            if named is not None:
                self._apply_stratification_update(stratification, named)

    def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_fractions`` has its
        fraction replaced; a new one is appended. Compartment names and the
        total population depend on the model expansion, so they are validated
        by :meth:`~commol.context.model.Model.update_compartment_fractions`
        against the engine, not here.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new
            fraction of the total population. Each value must lie between
            0.0 and 1.0.

        Raises
        ------
        ValueError
            If a value lies outside ``[0.0, 1.0]``.
        """
        updated = {
            compartment: CompartmentFraction(compartment=compartment, fraction=value)
            for compartment, value in fractions.items()
        }
        existing = {override.compartment for override in self.compartment_fractions}
        self.compartment_fractions = [
            updated.get(override.compartment, override)
            for override in self.compartment_fractions
        ] + [
            override
            for compartment, override in updated.items()
            if compartment not in existing
        ]

    def update_bin_fractions(self, fractions: Mapping[str, float | None]) -> None:
        """
        Update bin fractions by bin ID.

        The candidate fractions are validated before any bin is mutated, so a
        rejected call leaves the initial conditions unchanged. The conservation
        rule of :meth:`validate_calibrated_fractions_sum_to_one` is re-applied
        to the resulting mapping: the bin fractions must sum to 1.0, while
        uncalibrated (``None``) bins reserve the remainder. To move population
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
        known = {bf.bin for bf in self.bin_fractions}
        unknown = sorted(set(fractions) - known)
        if unknown:
            raise ValueError(
                f"Bin(s) {unknown} not found in initial conditions. "
                f"Available bins: {[bf.bin for bf in self.bin_fractions]}"
            )

        candidate = {bf.bin: bf.fraction for bf in self.bin_fractions}
        candidate.update(fractions)
        self._validate_bin_fraction_total(candidate)

        for bin_id, fraction in fractions.items():
            for bf in self.bin_fractions:
                if bf.bin == bin_id:
                    bf.fraction = fraction
                    break
