"""Expansion of bins and stratifications into expanded compartment names.

Bins are expanded over every stratification in declaration order. A conditional
stratification expands only the compartments whose already-applied categories
satisfy all of its conditions. A condition without a category is satisfied by
every compartment.
"""

from collections.abc import Iterable, Mapping

from commol.context.bin import Bin
from commol.context.stratification import Stratification


def stratification_conditions_met(
    stratification: Stratification,
    applied: Mapping[str, str],
) -> bool:
    """
    Check whether a stratification's conditions hold for a compartment.

    A condition without a category is satisfied by every compartment. A
    condition with a category is satisfied when the compartment already carries
    that category for the conditioning stratification.

    Parameters
    ----------
    stratification : Stratification
        Stratification whose conditions are checked.
    applied : Mapping[str, str]
        Categories already applied to the compartment, keyed by
        stratification id.

    Returns
    -------
    bool
        True when the stratification may expand the compartment.
    """
    conditions = stratification.conditions
    if conditions is None:
        return True
    return all(
        condition.category is None
        or applied.get(condition.stratification) == condition.category
        for condition in conditions
    )


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
            if stratification_conditions_met(stratification, applied):
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


def uniform_compartment_fractions(
    bins: Iterable[Bin],
    stratifications: Iterable[Stratification],
    bin_fractions: Mapping[str, float | None],
    stratification_fractions: Iterable[tuple[str, Mapping[str, float]]],
) -> dict[str, float]:
    """
    Compute the uniform expansion of bin fractions, before overrides.

    Each bin starts from its declared fraction, treating an uncalibrated
    ``None`` as zero, and is split by every stratification that applies to it.
    Only the first entry declared for a stratification id contributes its
    category fractions.

    Parameters
    ----------
    bins : Iterable[Bin]
        Bins of the population, in declaration order.
    stratifications : Iterable[Stratification]
        Stratifications in declaration order, including their conditions.
    bin_fractions : Mapping[str, float | None]
        Fraction per bin id, ``None`` for an uncalibrated bin.
    stratification_fractions : Iterable[tuple[str, Mapping[str, float]]]
        ``(stratification id, category fractions)`` pairs in declaration order.

    Returns
    -------
    dict[str, float]
        Fraction of the total population per expanded compartment.
    """
    stratification_maps: dict[str, Mapping[str, float]] = {}
    for stratification_id, categories in stratification_fractions:
        stratification_maps.setdefault(stratification_id, categories)

    partials: list[tuple[str, dict[str, str], float]] = [
        (bin_item.id, {}, bin_fractions.get(bin_item.id) or 0.0) for bin_item in bins
    ]

    for stratification in stratifications:
        fraction_map = stratification_maps.get(stratification.id, {})
        new_partials: list[tuple[str, dict[str, str], float]] = []
        for name, applied, value in partials:
            if stratification_conditions_met(stratification, applied):
                for category in stratification.categories:
                    new_partials.append(
                        (
                            f"{name}_{category}",
                            {**applied, stratification.id: category},
                            value * fraction_map.get(category, 0.0),
                        )
                    )
            else:
                new_partials.append((name, applied, value))
        partials = new_partials

    return {name: value for name, _, value in partials}
