"""The levels and shares of the settings: one type each, checked in Rust."""

from typing import Annotated

from pydantic import AfterValidator, ValidationInfo

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


def check_share(value: float, info: ValidationInfo) -> float:
    """
    Check a setting that is a share with the one check of commol.

    Parameters
    ----------
    value : float
        The share, such as the part of the particles a step keeps.
    info : ValidationInfo
        The field being validated; the message names it.

    Returns
    -------
    float
        The share.

    Raises
    ------
    ValueError
        If ``value`` is not strictly between 0 and 1.
    """
    commol_rs.inference.check_open_unit(info.field_name or "share", value)
    return value


type Share = Annotated[float, AfterValidator(check_share)]
"""A setting strictly between 0 and 1, such as a share or a probability."""
