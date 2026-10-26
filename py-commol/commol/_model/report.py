"""The ``initial_amounts`` and ``category_splits`` sections of a report."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict

from commol._model.engine_state import InitialCompartment
from commol._model.initial_conditions import CATEGORY_SEPARATOR
from commol._model.report_entry import Interval, entry_interval

if TYPE_CHECKING:
    from commol._model.aggregate import Aggregate
    from commol._model.process import Process
    from commol.uncertainty.sample import ParameterSummary


class InitialAmountEntry(TypedDict):
    """Reported initial amount of one bin or one overridden compartment."""

    amount: float | None
    unit: str | None
    calibrated: bool
    interval: NotRequired[Interval]


class CategorySplitEntry(TypedDict):
    """Reported split of one category within the group it splits."""

    group: list[str]
    fraction: float
    calibrated: bool
    interval: NotRequired[Interval]


def _interval(
    entry: InitialAmountEntry | CategorySplitEntry,
    entry_id: str,
    statistics: "Mapping[str, ParameterSummary] | None",
) -> None:
    """Add the interval of an entry, when the sample has one."""
    interval = entry_interval(entry_id, statistics)
    if interval is not None:
        entry["interval"] = interval


def _amount_entry(
    entry_id: str,
    amount: float | None,
    unit: str | None,
    calibrated: bool,
    statistics: "Mapping[str, ParameterSummary] | None",
) -> InitialAmountEntry:
    """One reported amount, with its interval when there is one."""
    entry: InitialAmountEntry = {
        "amount": amount,
        "unit": unit,
        "calibrated": calibrated,
    }
    _interval(entry, entry_id, statistics)
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


def _unit(process: "Process", bin_id: str) -> str | None:
    """The unit of a bin."""
    bin_item = process.bin(bin_id)
    return None if bin_item is None else bin_item.unit


def initial_amounts_report(
    process: "Process",
    structure: "Aggregate",
    calibrated_ids: set[str],
    parameter_statistics: "Mapping[str, ParameterSummary] | None",
) -> dict[str, InitialAmountEntry]:
    """
    The amount of every bin and of every overridden compartment.

    The ``initial_amounts`` section of `Model.applied_parameters`
    for an aggregate model. Each row gives its amount in the unit of its bin,
    and no row is a share of a total, so bins of different units, such as
    a count and a concentration, are reported side by side.

    A bin row reports the amount the engine builds: its declared amount, or
    the sum of its compartments when an override touches one of them. A bin
    whose declared amount is ``None`` keeps the ``None`` that marks it as
    needing calibration; the override rows still carry their own numbers. An
    override row is reported ``calibrated`` when its compartment carries a
    calibrated parameter id, or when a calibrated bin or category parameter id
    covers it.
    """
    state = structure.initial_state(process)
    by_name = {compartment.name: compartment for compartment in state}
    override_names = set(structure.compartment_amounts)

    report: dict[str, InitialAmountEntry] = {}
    for bin_id, declared in structure.bin_amounts.items():
        compartments = [c for c in state if c.bin == bin_id]
        amount = declared
        if declared is not None and not override_names.isdisjoint(
            c.name for c in compartments
        ):
            amount = sum(c.amount for c in compartments)
        report[bin_id] = _amount_entry(
            bin_id,
            amount,
            _unit(process, bin_id),
            bin_id in calibrated_ids,
            parameter_statistics,
        )
    for name, amount in structure.compartment_amounts.items():
        compartment = by_name[name]
        report[name] = _amount_entry(
            name,
            amount,
            _unit(process, compartment.bin),
            _override_is_calibrated(process, structure, compartment, calibrated_ids),
            parameter_statistics,
        )
    return report


def category_splits_report(
    process: "Process",
    structure: "Aggregate",
    calibrated_ids: set[str],
    parameter_statistics: "Mapping[str, ParameterSummary] | None",
) -> dict[str, CategorySplitEntry]:
    """
    The split of every category within the group its stratification splits.

    The ``category_splits`` section of `Model.applied_parameters`
    for an aggregate model. A category has no amount of its own, because it
    can split bins of different units; its row gives the declared fraction
    and ``group``, the categories that select the subgroup a conditional
    stratification splits (empty when it splits every bin). A category name
    two stratifications declare is reported under its qualified id, so the
    rows do not overwrite each other.
    """
    report: dict[str, CategorySplitEntry] = {}
    for stratification, categories in structure.stratification_fractions.items():
        group = list(structure.get_conditioning_categories(process, stratification))
        for category, fraction in categories.items():
            ids = _category_ids(structure, stratification, category)
            entry_id = next((i for i in ids if i in calibrated_ids), ids[0])
            entry: CategorySplitEntry = {
                "group": group,
                "fraction": fraction,
                "calibrated": entry_id in calibrated_ids,
            }
            _interval(entry, entry_id, parameter_statistics)
            report[ids[0]] = entry
    return report
