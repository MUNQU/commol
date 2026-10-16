"""The ``initial_conditions`` section of a report of an aggregate model."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict

from commol.context._model.engine_state import InitialCompartment
from commol.context.initial_conditions import CATEGORY_SEPARATOR
from commol.context.report_entry import Interval, entry_interval

if TYPE_CHECKING:
    from commol.context.aggregate import Aggregate
    from commol.context.probabilistic_calibration import ParameterSetStatistics
    from commol.context.process import Process

WHOLE_GROUP = "total"


class InitialConditionEntry(TypedDict):
    """Reported value of one initial condition."""

    group: str
    fraction: float | None
    value: float | None
    calibrated: bool
    interval: NotRequired[Interval]


def _group_label(group: tuple[str, ...]) -> str:
    """Name of the subgroup a fraction is taken of."""
    return "+".join(group) if group else WHOLE_GROUP


def _entry(
    entry_id: str,
    group: str,
    fraction: float | None,
    value: float | None,
    calibrated_ids: set[str],
    statistics: "Mapping[str, ParameterSetStatistics] | None",
    interval_scale: float = 1.0,
) -> InitialConditionEntry:
    """One reported initial condition, with its interval when there is one."""
    entry: InitialConditionEntry = {
        "group": group,
        "fraction": fraction,
        "value": value,
        "calibrated": entry_id in calibrated_ids,
    }
    interval = entry_interval(entry_id, statistics, interval_scale)
    if interval is not None:
        entry["interval"] = interval
    return entry


def _category_ids(
    structure: "Aggregate", stratification: str, category: str
) -> list[str]:
    """
    Ids that name a stratification category, preferred first.

    The qualified id (``group:g1``) always names it. The bare name does too
    when no other stratification declares a category of that name.
    """
    qualified = f"{stratification}{CATEGORY_SEPARATOR}{category}"
    preferred = structure.category_id(stratification, category)
    return [qualified] if preferred == qualified else [preferred, qualified]


def _override_is_calibrated(
    process: "Process",
    structure: "Aggregate",
    compartment: InitialCompartment,
    calibrated_ids: set[str],
) -> bool:
    """
    Whether a compartment override row was written back from calibration.

    It was when a calibrated id names the compartment, its bin, or a category
    of a stratification applied to it: a category value splits the category
    and its complement, so both sides are written back.
    """
    if compartment.name in calibrated_ids or compartment.bin in calibrated_ids:
        return True
    categories_by_stratification = {
        stratification.id: stratification.categories
        for stratification in process.stratifications
    }
    return any(
        not calibrated_ids.isdisjoint(
            _category_ids(structure, stratification, category)
        )
        for stratification in compartment.categories
        for category in categories_by_stratification[stratification]
    )


def initial_conditions_report(
    process: "Process",
    structure: "Aggregate",
    calibrated_ids: set[str],
    parameter_statistics: "Mapping[str, ParameterSetStatistics] | None",
) -> dict[str, InitialConditionEntry]:
    """
    Every bin fraction, stratification category and compartment override.

    The ``initial_conditions`` section of :func:`commol.applied_parameters_report`
    for an aggregate model.

    Every row reports its effective amount the engine builds from the
    uniform expansion plus the declared compartment overrides, and its fraction
    as that amount over the declared ``total``. A single
    denominator is used for bin, category and override rows, so the fractions
    line up with one another and with the ``total`` row. An override row is reported
    ``calibrated`` when its compartment carries a calibrated parameter id, or
    when a calibrated bin or category parameter id covers it. A declared
    override that no calibrated id covers reports ``calibrated`` false.

    A bin whose declared fraction is ``None`` keeps the ``None`` fraction and
    amount that mark it as needing calibration, even when overrides touch
    some of its compartments; the override rows still carry their own numbers.
    """
    total = structure.total
    state = structure.initial_state(process)
    by_name = {compartment.name: compartment for compartment in state}
    override_names = set(structure.compartment_fractions)

    report: dict[str, InitialConditionEntry] = {
        "total": {
            "group": WHOLE_GROUP,
            "fraction": 1.0,
            "value": total,
            "calibrated": False,
        }
    }
    for bin_id, declared in structure.bin_fractions.items():
        compartments = [c for c in state if c.bin == bin_id]
        fraction: float | None
        value: float | None
        if declared is None:
            # A bin needing calibration keeps its marker even when overrides
            # pin some of its compartments; the override rows carry the numbers.
            fraction = None
            value = None
        elif override_names.isdisjoint(c.name for c in compartments):
            # No override touches this row, so report the declared fraction
            # verbatim rather than an expansion sum that drifts in the last
            # float bits.
            fraction = declared
            value = fraction * total
        else:
            value = sum(c.amount for c in compartments)
            fraction = value / total
        report[bin_id] = _entry(
            bin_id,
            WHOLE_GROUP,
            fraction,
            value,
            calibrated_ids,
            parameter_statistics,
        )
    for stratification, categories in structure.stratification_fractions.items():
        group = structure.get_conditioning_categories(process, stratification)
        # A conditional stratification's category is calibrated as a fraction
        # of its conditioning subgroup, but reported as a fraction of the
        # total. The group's share of the total is the same factor that
        # converts one unit into the other, so it rescales the interval.
        interval_scale = (
            structure.subgroup_total(process, group) / total if group else 1.0
        )
        for category in categories:
            row_value = sum(
                c.amount for c in state if c.categories.get(stratification) == category
            )
            # A category name two stratifications declare is reported under
            # its qualified id, so the rows do not overwrite each other.
            ids = _category_ids(structure, stratification, category)
            entry_id = next((i for i in ids if i in calibrated_ids), ids[0])
            report[ids[0]] = _entry(
                entry_id,
                _group_label(group),
                row_value / total,
                row_value,
                calibrated_ids,
                parameter_statistics,
                interval_scale,
            )
    for compartment, override in structure.compartment_fractions.items():
        calibrated = _override_is_calibrated(
            process, structure, by_name[compartment], calibrated_ids
        )
        report[compartment] = _entry(
            compartment,
            WHOLE_GROUP,
            override,
            override * total,
            {compartment} if calibrated else set(),
            parameter_statistics,
        )
    return report
