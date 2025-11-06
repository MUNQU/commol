import math
from typing import Mapping, Self

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
