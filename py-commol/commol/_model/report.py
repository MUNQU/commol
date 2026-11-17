"""The ``initial_amounts`` and ``category_splits`` sections of a report."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict

from commol._model.aggregate import split_groups
from commol._model.calibration_write_back import covered_bins
from commol._model.engine_state import build_engine, initial_condition_coverage
from commol._model.initial_conditions import split_by
from commol._model.report_entry import EntrySpread, entry_spread

if TYPE_CHECKING:
    from commol._model.aggregate import Aggregate
    from commol._model.model import Model
    from commol._model.process import Process
    from commol._model.report_entry import EntrySummary
    from commol._problem.problem import CalibrationProblem


class InitialAmountEntry(TypedDict):
    """Reported initial amount of one bin or one overridden compartment."""

    amount: float | None
    unit: str | None
    calibrated: bool
    spread: NotRequired[EntrySpread]


class CategorySplitEntry(TypedDict):
    """Reported split of one category within the group it splits."""

    group: list[str]
    fraction: float
    calibrated: bool
    spread: NotRequired[EntrySpread]


class CalibratedInitialConditions(NamedTuple):
    """
    What the calibrated initial conditions of a problem set, as the engine
    resolves them on the model (`initial_condition_coverage`).

    Attributes
    ----------
    ids : frozenset[str]
        The id of each calibrated initial-condition value.
    sets : frozenset[str]
        The compartments whose amount a value or the remainder sets.
    writes : frozenset[str]
        The compartments a value or the remainder writes: those they set, and
        both sides of each pair of a category split.
    splits : dict[tuple[str, str], str]
        The id of each value the engine resolves as a category split, by its
        stratification and its category.
    """

    ids: frozenset[str]
    sets: frozenset[str]
    writes: frozenset[str]
    splits: dict[tuple[str, str], str]


def calibrated_initial_conditions[S: "Aggregate"](
    model: "Model[S]",
    problem: "CalibrationProblem",
    values: Mapping[str, float],
) -> CalibratedInitialConditions:
    """
    What the initial conditions of `problem` that `values` holds set.

    The engine resolves the values and the remainder on `model`: the kind of
    each value, and the compartments it sets and writes. A value it resolves
    as a category split names its category through the engine too
    (`Aggregate.resolve_category`), as the write-back does.

    Parameters
    ----------
    model : Model[Aggregate]
        The model that holds the applied values.
    problem : CalibrationProblem
        The problem the values came from, for the type of each value and the
        remainder.
    values : Mapping[str, float]
        Calibrated value per parameter id.

    Returns
    -------
    CalibratedInitialConditions
        Empty when `values` holds no initial condition of `problem`.

    Raises
    ------
    ValueError
        If the engine cannot resolve an initial-condition id or the remainder.
    """
    parameters = [
        parameter
        for parameter in problem.parameters
        if parameter.parameter_type == "initial_condition" and parameter.id in values
    ]
    if not parameters:
        return CalibratedInitialConditions(frozenset(), frozenset(), frozenset(), {})
    coverage, remainder = initial_condition_coverage(
        build_engine(model), parameters, problem.initial_condition_remainder
    )
    sets = set(remainder)
    writes = set(remainder)
    splits: dict[tuple[str, str], str] = {}
    for parameter, target in zip(parameters, coverage, strict=True):
        sets.update(target.compartments)
        writes.update(target.written)
        if target.kind == "category":
            splits[model.structure.resolve_category(parameter.id)] = parameter.id
    return CalibratedInitialConditions(
        frozenset(parameter.id for parameter in parameters),
        frozenset(sets),
        frozenset(writes),
        splits,
    )


def _spread(
    entry: InitialAmountEntry | CategorySplitEntry,
    value_id: str | None,
    statistics: "Mapping[str, EntrySummary] | None",
) -> None:
    """Add the spread of the calibrated value that names an entry, when the
    sample has one."""
    spread = None if value_id is None else entry_spread(value_id, statistics)
    if spread is not None:
        entry["spread"] = spread


def _amount_entry(
    entry_id: str,
    amount: float | None,
    unit: str | None,
    calibrated: bool,
    initial_conditions: CalibratedInitialConditions,
    statistics: "Mapping[str, EntrySummary] | None",
) -> InitialAmountEntry:
    """One reported amount, with the spread of the value of its id when
    there is one."""
    entry: InitialAmountEntry = {
        "amount": amount,
        "unit": unit,
        "calibrated": calibrated,
    }
    named = entry_id if entry_id in initial_conditions.ids else None
    _spread(entry, named, statistics)
    return entry


def _unit(process: "Process", bin_id: str) -> str | None:
    """The unit of a bin."""
    bin_item = process.bin(bin_id)
    return None if bin_item is None else bin_item.unit


def initial_amounts_report(
    process: "Process",
    structure: "Aggregate",
    calibrated: CalibratedInitialConditions,
    parameter_statistics: "Mapping[str, EntrySummary] | None",
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
    needing calibration; the override rows still carry their own numbers. A
    bin row is ``calibrated`` when the calibrated values and the remainder
    set every compartment of the bin, as the write-back writes its amount
    (`covered_bins`), and an override row when they write its compartment,
    a split of a category included (`calibrated`).
    """
    state = structure.initial_state(process)
    by_name = {compartment.name: compartment for compartment in state}
    override_names = set(structure.compartment_amounts)
    set_bins = covered_bins(state, calibrated.sets)

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
            bin_id in set_bins,
            calibrated,
            parameter_statistics,
        )
    for name, amount in structure.compartment_amounts.items():
        report[name] = _amount_entry(
            name,
            amount,
            _unit(process, by_name[name].bin),
            name in calibrated.writes,
            calibrated,
            parameter_statistics,
        )
    return report


def category_splits_report(
    process: "Process",
    structure: "Aggregate",
    calibrated: CalibratedInitialConditions,
    parameter_statistics: "Mapping[str, EntrySummary] | None",
) -> dict[str, CategorySplitEntry]:
    """
    The split of every category within the group its stratification splits.

    The ``category_splits`` section of `Model.applied_parameters`
    for an aggregate model. A category has no amount of its own, because it
    can split bins of different units; its row gives the declared fraction
    and ``group``, the categories that select the subgroup a conditional
    stratification splits (empty when it splits every bin; the categories of
    one stratification are a set, of which the subgroup has one). A category name
    two stratifications declare is reported under its qualified id, so the
    rows do not overwrite each other. A stratification split by the category
    of an earlier one gives a row per split, under ``<category> of
    <category of by>`` (``t1 of g1``), with that category in ``group``.

    Every row of a stratification is ``calibrated`` when a calibrated value
    is a split of one of its categories (`calibrated`): the value sets its
    category and the complement, in every split. Only the rows of the
    category the value names gain its spread.
    """
    split_stratifications = {stratification for stratification, _ in calibrated.splits}
    report: dict[str, CategorySplitEntry] = {}
    for stratification in structure.stratification_fractions:
        splits = split_groups(structure, process, stratification)
        by = split_by(structure.stratification_fractions[stratification])
        for group, split in splits:
            for category, fraction in split.items():
                entry: CategorySplitEntry = {
                    "group": list(group),
                    "fraction": fraction,
                    "calibrated": stratification in split_stratifications,
                }
                _spread(
                    entry,
                    calibrated.splits.get((stratification, category)),
                    parameter_statistics,
                )
                key = structure.category_id(stratification, category)
                report[key if by is None else f"{key} of {group[-1]}"] = entry
    return report
