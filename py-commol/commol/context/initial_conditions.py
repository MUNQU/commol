import math
from collections.abc import Iterable, Mapping
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


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

    def _validate_bin_fraction_total(
        self,
        candidate: Mapping[str, float | None],
        *,
        complete: bool,
    ) -> None:
        """
        Enforce the bin-fraction conservation rule on a candidate mapping.

        The rule mirrors :meth:`validate_calibrated_fractions_sum_to_one`:
        uncalibrated (``None``) fractions floor the calibrated total strictly
        below 1.0 so they can be filled later. When every bin is calibrated,
        a complete specification must sum to 1.0; a partial update may leave the
        total below 1.0 but never above it, because bin fractions are shares of
        a single population.

        Parameters
        ----------
        candidate : Mapping[str, float | None]
            Bin id to fraction after the candidate update.
        complete : bool
            Whether the caller specified every bin, so the sum-to-1 rule
            applies in full.

        Raises
        ------
        ValueError
            If the candidate bins would make the total exceed the population or
            a complete specification does not sum to 1.0.
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
        elif complete:
            if not math.isclose(total, 1.0, abs_tol=1e-4):
                raise ValueError(
                    f"Bin fractions must sum to 1.0, got {total:.4f}. "
                    f"Fractions: {list(candidate.values())}"
                )
        elif total > 1.0 + 1e-4:
            raise ValueError(
                f"Bin fractions must not sum to more than 1.0, got {total:.4f}. "
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
            {bf.bin: bf.fraction for bf in self.bin_fractions}, complete=True
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

    def get_category_fraction(self, category: str) -> float:
        """
        Get the fraction of a category within the group it subdivides.

        Parameters
        ----------
        category : str
            A stratification category name.

        Returns
        -------
        float
            The declared fraction, relative to the group the category's
            stratification applies to rather than to the whole population.

        Raises
        ------
        ValueError
            If the category is not found.
        """
        for stratification in self.stratification_fractions:
            for fraction in stratification.fractions:
                if fraction.category == category:
                    return fraction.fraction
        raise ValueError(
            f"Category '{category}' not found in initial conditions. "
            f"Available categories: {sorted(self.get_categories_with_fractions())}"
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
            Category names forming a chain from the whole population down to
            the subgroup, outermost first.

        Returns
        -------
        float
            Number of individuals in the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found.
        """
        population = float(self.population_size)
        for category in categories:
            population *= self.get_category_fraction(category)
        return population

    def _owning_stratification(self, category: str) -> StratificationFractions:
        """Return the stratification that declares a category."""
        for stratification in self.stratification_fractions:
            if any(
                fraction.category == category for fraction in stratification.fractions
            ):
                return stratification
        raise ValueError(
            f"Category '{category}' not found in initial conditions. "
            f"Available categories: {sorted(self.get_categories_with_fractions())}"
        )

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
        call. Within each stratification, at most one category may be omitted;
        the omitted category receives the remaining fraction so that the
        stratification sums to 1.0. When every category is given, the values
        must already sum to 1.0.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping category names to their new fraction values.
            Each value must lie between 0.0 and 1.0.

        Raises
        ------
        ValueError
            If a category is not found, a value lies outside [0.0, 1.0], a
            stratification has more than one category omitted, fully specified
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
            owner = self._owning_stratification(category).stratification
            by_stratification.setdefault(owner, {})[category] = value

        for stratification in self.stratification_fractions:
            named = by_stratification.get(stratification.stratification)
            if named is not None:
                self._apply_stratification_update(stratification, named)

    def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_fractions`` has its
        fraction replaced; a new one is appended. The candidate overrides are
        validated before they are stored, so a rejected call leaves the
        initial conditions unchanged. The check compares the overrides plus the
        bin fractions they do not cover against the total population; it does
        not expand compartments.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new
            fraction of the total population. Each value must lie between
            0.0 and 1.0.

        Raises
        ------
        ValueError
            If a value lies outside ``[0.0, 1.0]``, or the overrides make the
            initial population exceed ``population_size``.
        """
        self._upsert_compartment_fractions(fractions, validate_total=True)

    def _upsert_compartment_fractions(
        self, fractions: Mapping[str, float], *, validate_total: bool
    ) -> None:
        """
        Write overrides, optionally skipping the over-allocation guard.

        Calibration write-back persists scored engine states that may be
        non-conserving (the Rust ``set_initial_condition`` has no total check),
        so it calls this with ``validate_total=False``. The fraction range is
        always enforced.
        """
        for compartment, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Fraction for compartment '{compartment}' must be between "
                    f"0.0 and 1.0, got {value}."
                )

        candidate = {
            override.compartment: override.fraction
            for override in self.compartment_fractions
        }
        candidate.update(fractions)
        if validate_total:
            self._validate_compartment_fraction_total(candidate)

        for compartment, value in fractions.items():
            for existing in self.compartment_fractions:
                if existing.compartment == compartment:
                    existing.fraction = value
                    break
            else:
                self.compartment_fractions.append(
                    CompartmentFraction(compartment=compartment, fraction=value)
                )

    def _validate_compartment_fraction_total(
        self, overrides: Mapping[str, float]
    ) -> None:
        """
        Enforce that overrides plus uncovered bin fractions stay within 1.0.

        A bin counts as covered when an override names it or one of its
        expanded compartments. Uncalibrated (``None``) bin fractions are
        skipped: a violation that only appears once they receive their
        calibrated fractions is not detected here.

        Parameters
        ----------
        overrides : Mapping[str, float]
            Candidate override fractions keyed by compartment name.

        Raises
        ------
        ValueError
            If the overrides make the initial population exceed
            ``population_size``.
        """
        covered_bins = {
            bin_fraction.bin
            for bin_fraction in self.bin_fractions
            if any(
                compartment == bin_fraction.bin
                or compartment.startswith(f"{bin_fraction.bin}_")
                for compartment in overrides
            )
        }
        uncovered = sum(
            bin_fraction.fraction
            for bin_fraction in self.bin_fractions
            if bin_fraction.fraction is not None
            and bin_fraction.bin not in covered_bins
        )
        total = sum(overrides.values())
        if total + uncovered > 1.0 + 1e-6:
            raise ValueError(
                "Per-compartment overrides would make the initial population "
                "exceed population_size."
            )

    def update_bin_fractions(self, fractions: Mapping[str, float | None]) -> None:
        """
        Update bin fractions by bin ID.

        The candidate fractions are validated before any bin is mutated, so a
        rejected call leaves the initial conditions unchanged. The conservation
        rule of :meth:`validate_calibrated_fractions_sum_to_one` is re-applied:
        a complete specification must sum to 1.0 and a partial one must not
        exceed it, while uncalibrated (``None``) bins reserve the remainder.

        Calibration write-back never reaches this method with a non-conserving
        update: a bin value that would break the declared mapping is diverted
        to per-compartment overrides by
        :meth:`~commol.context.model.Model._route_calibration_values`.

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
        self._validate_bin_fraction_total(candidate, complete=set(fractions) >= known)

        for bin_id, fraction in fractions.items():
            for bf in self.bin_fractions:
                if bf.bin == bin_id:
                    bf.fraction = fraction
                    break
