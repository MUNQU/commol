"""The confidence level of an interval: one type, checked in Rust."""

from typing import Annotated

from pydantic import AfterValidator

from commol.commol_rs import _commol_rs as commol_rs


def check_confidence_level(confidence_level: float) -> float:
    """
    Check a confidence level with the one check of commol.

    Parameters
    ----------
    confidence_level : float
        The level of a central interval, such as 0.95.

    Returns
    -------
    float
        The confidence level.

    Raises
    ------
    ValueError
        If ``confidence_level`` is not strictly between 0 and 1.
    """
    commol_rs.inference.interval_levels(confidence_level)
    return confidence_level


type ConfidenceLevel = Annotated[float, AfterValidator(check_confidence_level)]
"""The level of a central interval, strictly between 0 and 1."""
