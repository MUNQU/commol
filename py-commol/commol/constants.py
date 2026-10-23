"""Constants shared across the package, read from the engine."""

from commol.commol_rs import _commol_rs as commol_rs

TOTAL_VARIABLE: str = commol_rs.model.TOTAL
"""The sum of every compartment, accumulators excluded, in every expression."""

RESERVED_VARIABLE_NAMES: tuple[str, ...] = tuple(commol_rs.model.ENGINE_VARIABLES)
"""Variables the engine provides in every expression."""

SUBGROUP_TOTAL_PREFIX: str = commol_rs.model.SUBGROUP_TOTAL_PREFIX
"""Prefix of the subgroup totals ``total_<cat>…`` the engine provides."""
