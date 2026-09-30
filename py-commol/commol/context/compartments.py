"""Expansion of bins and stratifications into expanded compartment names.

This mirrors, in pure Python, the Rust ``generate_compartments`` logic in
``crates/commol-difference/src/builder.rs``: bins are expanded over every
stratification in declaration order, and a conditional stratification only
expands the compartments whose already-applied categories satisfy all of its
conditions. Compartments that do not satisfy the conditions are kept without
appending that stratification's categories.
"""

from collections.abc import Iterable

from commol.context.bin import Bin
from commol.context.stratification import Stratification


def expand_over_stratifications(
    source_ids: Iterable[str],
    stratifications: Iterable[Stratification],
) -> list[tuple[str, dict[str, str]]]:
    """
    Expand source ids across the stratifications that apply to each of them.

    Each element is a 2-tuple ``(source_id, applied_categories)`` where
    ``applied_categories`` maps each stratification id that was actually
    applied to its chosen category.

    Parameters
    ----------
    source_ids : Iterable[str]
        Base ids to expand, typically bin ids.
    stratifications : Iterable[Stratification]
        Stratifications in declaration order. A conditional stratification is
        applied only to ids whose already-applied categories satisfy its
        conditions.

    Returns
    -------
    list[tuple[str, dict[str, str]]]
        Expanded ``(name, applied_categories)`` pairs in declaration order.
    """
    partials: list[tuple[str, dict[str, str]]] = [
        (source_id, {}) for source_id in source_ids
    ]

    for stratification in stratifications:
        new_partials: list[tuple[str, dict[str, str]]] = []
        for name, applied in partials:
            conditions_met = stratification.conditions is None or all(
                applied.get(condition.stratification) == condition.category
                for condition in stratification.conditions
            )
            if conditions_met:
                for category in stratification.categories:
                    new_partials.append(
                        (f"{name}_{category}", {**applied, stratification.id: category})
                    )
            else:
                new_partials.append((name, applied))
        partials = new_partials

    return partials


def generate_compartment_names(
    bins: Iterable[Bin],
    stratifications: Iterable[Stratification],
) -> list[str]:
    """
    Generate all expanded compartment names of a model definition.

    Parameters
    ----------
    bins : Iterable[Bin]
        Bins of the population, in declaration order.
    stratifications : Iterable[Stratification]
        Stratifications in declaration order, including their conditions.

    Returns
    -------
    list[str]
        Expanded compartment names, in the same order as the engine expands
        them.
    """
    return [
        name
        for name, _ in expand_over_stratifications(
            (bin_item.id for bin_item in bins), stratifications
        )
    ]
