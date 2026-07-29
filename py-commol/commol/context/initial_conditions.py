import math
from collections.abc import Iterable, Mapping
from typing import Self

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def validate_calibrated_fractions_sum_to_one(self) -> Self:
        """
        Validate that bin fractions sum appropriately.

        Rules:
        - If all fractions are calibrated (no None): must sum to exactly 1.0
        - If some fractions are None (uncalibrated): calibrated ones must sum to LESS
            than 1.0
        - If all fractions are None: skip validation (will be set before simulation)
        """
        calibrated_fractions = [
            bf.fraction for bf in self.bin_fractions if bf.fraction is not None
        ]

        # If all fractions are None (all need calibration), skip validation
        if not calibrated_fractions:
            return self

        # If some fractions are calibrated, check if they sum correctly
        # Note: We can't validate the sum if some are None, so we only warn
        total = sum(calibrated_fractions)
        uncalibrated_count = sum(1 for bf in self.bin_fractions if bf.fraction is None)

        if uncalibrated_count > 0:
            # Some fractions are None: calibrated ones MUST be LESS than 1.0
            if total >= 1.0:
                raise ValueError(
                    (
                        f"Calibrated bin fractions sum to {total:.4f}, but must be "
                        f"LESS than 1.0 to leave room for {uncalibrated_count} "
                        f"uncalibrated fraction(s). Calibrated fractions: "
                        f"{
                            [
                                (bf.bin, bf.fraction)
                                for bf in self.bin_fractions
                                if bf.fraction is not None
                            ]
                        }"
                    )
                )
        else:
            # All fractions are calibrated, must sum to 1.0
            if not math.isclose(total, 1.0, abs_tol=1e-4):
                raise ValueError(
                    (
                        f"Bin fractions must sum to 1.0, got {total:.4f}. "
                        f"Fractions: {[bf.fraction for bf in self.bin_fractions]}"
                    )
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

    def update_bin_fractions(self, fractions: Mapping[str, float | None]) -> None:
        """
        Update bin fractions by bin ID.

        Parameters
        ----------
        fractions : Mapping[str, float | None]
            Dictionary mapping bin IDs to their new fraction values.
            None values indicate bins that need calibration.

        Raises
        ------
        ValueError
            If a bin ID is not found in the initial conditions.
        """
        for bin_id, fraction in fractions.items():
            found = False
            for bf in self.bin_fractions:
                if bf.bin == bin_id:
                    bf.fraction = fraction
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Bin '{bin_id}' not found in initial conditions. "
                    f"Available bins: {[bf.bin for bf in self.bin_fractions]}"
                )
