"""The aggregate structure: amounts per compartment.

An `Aggregate` lays the process of a model out as amounts: the bins expand
over the stratifications into compartments, each compartment holds an amount,
and the structure says how much starts where (the amount of each bin, in the
unit of the bin, and the split of it among the categories) and who the
neighbors of a unit are (`Complete`, `Groups`). The engine
(``commol_rs.model``) expands, plans and initializes the model; this module
validates the structure and asks the engine.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field

from commol._model import engine_state
from commol._model.engine_state import InitialCompartment
from commol._model.initial_conditions import (
    InitialConditions,
    category_splits,
    split_by,
)
from commol._model.neighbors import _Neighbors
from commol._model.process import Process
from commol._model.structure import ModelStructure, ReportSections, SeriesLabel

if TYPE_CHECKING:
    from commol.description.blocks import Section
    from commol._model.model import Model
    from commol._problem.problem import CalibrationParameter, CalibrationProblem
    from commol.commol_rs._commol_rs import EngineProtocol
    from commol._model.report_entry import EntrySummary


class Aggregate(ModelStructure, InitialConditions):
    """
    Amounts per compartment: how much starts where, and who the neighbors of
    a unit are.

    The bins of the process expand over its stratifications into
    compartments. ``bin_amounts`` gives the amount of each bin at the start,
    in the unit of the bin, ``stratification_fractions`` splits it among the
    categories, and ``compartment_amounts`` overrides single compartments.
    Bins of different units, such as a count and a concentration, need no
    total over them.

    Attributes
    ----------
    kind : Literal["aggregate"]
        The model kind.
    variant : str | None
        The variant of the structure (see `ModelStructure`), None for
        `Aggregate` itself.
    bin_amounts : dict[str, float | None]
        The amount of each bin at the start, by bin id, in the unit of the
        bin, such as ``{"A": 990, "B": 10, "C": 1e5}``; finite and not
        negative. None leaves an amount for calibration.
    stratification_fractions : dict[str, dict]
        The fraction of each category, by stratification id, a
        ``dict[str, float]`` such as ``{"group": {"g1": 0.3, "g2": 0.7}}``:
        the split of the amount of every bin it applies to. A stratification
        may instead split each category of a stratification declared before
        it by its own fractions, a ``dict[str, str | dict[str, float]]`` such
        as ``{"type": {"by": "group", "g1": {"t1": 0.25, "t2": 0.75},
        "g2": {"t1": 0.5, "t2": 0.5}}}``: the compartment ``A_g1_t1`` then
        starts with the amount of ``A`` times 0.3 times 0.25. Every
        compartment of ``type`` needs a category of ``group``. A conditional
        stratification splits only the group its conditions select.
    compartment_amounts : dict[str, float]
        Amounts of single expanded compartments, such as ``{"A_g1": 0.0}``,
        applied after the expansion. The names are checked against the engine
        when the model is built.
    neighbors : Complete | Groups | None
        Who the neighbors of a unit are, for the rate variables
        ``neighbors_<bin>`` and ``neighbors``. Required, with no default, when
        a rate reads them.
    """

    kind: Literal["aggregate"] = "aggregate"
    neighbors: _Neighbors | None = Field(
        default=None,
        description="Who the neighbors of a unit are; required when a rate reads them.",
    )

    # ------------------------------------------------------------------
    # The interface of a model kind
    # ------------------------------------------------------------------

    def check_model(self, model: "Model[Self]") -> None:
        """
        Check the structure against the process and the parameters.

        The process needs a bin and a transition, the amounts must name every
        bin, and the engine must check the fractions against the
        stratifications (every stratification and category named, a split by
        category by a stratification declared before it), plan every
        transition, accept every id (no parameter id is an engine variable, a
        bin, a compartment or a subgroup total), resolve the neighbors, find
        every variable a rate or a formula parameter reads, and build the
        initial state.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Raises
        ------
        ValueError
            If one of these checks fails.
        """
        process = model.process
        if not process.bins:
            raise ValueError("At least one bin must be defined.")
        if not process.transitions:
            raise ValueError("At least one transition must be defined.")
        self._check_bin_amounts(process)
        engine_state.transition_flows(process, self)
        engine_state.check_model(model)
        if self.compartment_amounts:
            engine_state.initial_state(process, self)

    def outputs_by_source(self, model: "Model[Self]") -> dict[str, list[str]]:
        """
        Map each bin and accumulator id to the output names it expands into.

        A bin or accumulator produces one output per combination of the
        stratification categories that apply to it. Without stratifications it
        produces a single output named after the id itself.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Returns
        -------
        dict[str, list[str]]
            Bin and accumulator ids, each mapped to its output names in
            declaration order.
        """
        process = model.process
        outputs: dict[str, list[str]] = {
            source_id: []
            for source_id in (
                *(bin_item.id for bin_item in process.bins),
                *(accumulator.id for accumulator in process.accumulators),
            )
        }
        compartments, accumulators = engine_state.expanded_outputs(process)
        for output in (*compartments, *accumulators):
            outputs[output.source].append(output.name)
        return outputs

    def series_label(self, model: "Model[Self]", series_id: str) -> SeriesLabel:
        """
        The name and unit of a bin, for a plot of its series.

        Parameters
        ----------
        model : Model
            The model that holds this structure.
        series_id : str
            A bin id, or any other series id.

        Returns
        -------
        SeriesLabel
            The bin's name and unit, or the id and no unit when no bin has it.
        """
        bin_item = model.process.bin(series_id)
        if bin_item is None:
            return SeriesLabel(series_id, None)
        return SeriesLabel(bin_item.name or series_id, bin_item.unit)

    def uncalibrated_values(self) -> list[str]:
        """
        Ids of the bins whose initial amount needs calibration.

        Returns
        -------
        list[str]
            Bin ids with a None amount.
        """
        return self.get_uncalibrated_bins()

    def check_calibration_values(
        self,
        model: "Model[Self]",
        engine: "EngineProtocol",
        parameters: Sequence["CalibrationParameter"],
        remainder: str | None,
    ) -> None:
        """
        Check that the remainder has the unit of every amount it gives.

        The remainder takes what the amounts set leave of the sum it keeps, so
        a bin, prefix or compartment value must be in the unit of the
        remainder; a category split has no unit.

        Parameters
        ----------
        model : Model
            The model the values are set on.
        engine : EngineProtocol
            The engine built from it.
        parameters : Sequence[CalibrationParameter]
            The values set, with their types and bounds.
        remainder : str | None
            The bin or compartment that takes what the initial conditions
            leave, if any.

        Raises
        ------
        ValueError
            If an amount set is in another unit than the remainder.
        """
        if remainder is None:
            return
        initial_conditions = [
            param for param in parameters if param.parameter_type == "initial_condition"
        ]
        covered, kept = engine_state.initial_condition_coverage(
            engine, initial_conditions, remainder
        )
        bin_of = {c.name: c.bin for c in self.initial_state(model.process)}

        def unit(compartment: str) -> str | None:
            bin_item = model.process.bin(bin_of[compartment])
            return None if bin_item is None else bin_item.unit

        remainder_unit = unit(kept[0])
        for param, compartments in zip(initial_conditions, covered, strict=True):
            for compartment in compartments:
                if unit(compartment) != remainder_unit:
                    raise ValueError(
                        f"The initial-condition remainder '{remainder}' is in "
                        f"{_unit_name(remainder_unit)}, but the calibrated "
                        f"initial condition '{param.id}' sets '{compartment}', "
                        f"in {_unit_name(unit(compartment))}. The remainder "
                        f"keeps a sum of amounts of one unit: name a remainder "
                        f"of the unit of '{param.id}'."
                    )

    def with_calibrated_values(
        self,
        model: "Model[Self]",
        problem: "CalibrationProblem",
        values: Mapping[str, float],
    ) -> Self:
        """
        The structure with calibrated initial conditions written back.

        The written structure rebuilds the initial state the engine
        scored. Bin values update the amounts they declare, in the unit of
        the bin, and category values the splits. Every compartment those do
        not reproduce, such as an exact compartment value or a value that
        overlaps a per-compartment override, is written as a per-compartment
        override.

        Parameters
        ----------
        model : Model
            The model the problem was built on. It is not changed.
        problem : CalibrationProblem
            The problem, for its parameters and remainder.
        values : Mapping[str, float]
            Calibrated value per parameter id.

        Returns
        -------
        Aggregate
            A copy that holds the calibrated initial conditions.

        Raises
        ------
        ValueError
            If the engine cannot resolve an initial-condition id or the
            remainder, or the values leave the remainder below zero.
        """
        from commol._model.calibration_write_back import calibrated_structure

        return calibrated_structure(model, problem, values)

    def report_sections(
        self,
        model: "Model[Self]",
        calibrated_ids: set[str],
        parameter_statistics: "Mapping[str, EntrySummary] | None",
    ) -> ReportSections:
        """
        The ``initial_amounts`` and ``category_splits`` sections of a report.

        See `Model.applied_parameters`.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, EntrySummary] | None
            Statistics of each id from a sample, or None without one.

        Returns
        -------
        ReportSections
            ``{"initial_amounts": ..., "category_splits": ...}``.
        """
        from commol._model.report import category_splits_report, initial_amounts_report

        return {
            "initial_amounts": initial_amounts_report(
                model.process, self, calibrated_ids, parameter_statistics
            ),
            "category_splits": category_splits_report(
                model.process, self, calibrated_ids, parameter_statistics
            ),
        }

    def describe(self, model: "Model[Self]") -> "list[Section]":
        """
        The entities, dynamics, initialization and design concepts.

        ``entities``: the bins, accumulators, stratifications and expanded
        compartments. ``dynamics``: the state graphs, the compact and expanded
        forms of the equations the engine simulates, the impulses and the
        neighbors. ``initialization``: the bin amounts, the category splits,
        the overrides and the initial state the engine builds. ``design_concepts``: the
        interaction of the units, when the structure says who their neighbors
        are.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Returns
        -------
        list[Section]
            The sections of an aggregate model.
        """
        from commol._model.aggregate_description import describe_aggregate

        return describe_aggregate(model)

    def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
        """Check the units of the model; see `Model.check_unit_consistency`."""
        from commol._model.units import UnitChecker

        UnitChecker(model).check_unit_consistency(verbose)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def update_compartment_amounts(
        self, process: Process, amounts: Mapping[str, float]
    ) -> None:
        """
        Upsert per-compartment initial condition overrides.

        A compartment already declared has its amount replaced; a new one is
        appended. The engine validates the result against the expansion of
        the process before it is stored, so a rejected call leaves the
        structure unchanged.

        Parameters
        ----------
        process : Process
            The process of the model, such as ``model.process``.
        amounts : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new
            amount, in the unit of their bin.

        Raises
        ------
        ValueError
            If a name is not an expanded compartment of the model, or an
            amount is not finite or is negative.
        """
        candidate = self.model_copy(deep=True)
        candidate._upsert_compartment_amounts(amounts)
        engine_state.initial_state(process, candidate)
        self.compartment_amounts = candidate.compartment_amounts

    def initial_state(self, process: Process) -> list[InitialCompartment]:
        """
        Expanded compartments and the amount each one starts with.

        The engine builds this state from the bin amounts, the stratification
        fractions and the per-compartment overrides; it is the state a
        simulation starts from.

        Parameters
        ----------
        process : Process
            The process of the model, such as ``model.process``.

        Returns
        -------
        list[InitialCompartment]
            One entry per expanded compartment, in engine order.

        Raises
        ------
        ValueError
            If an amount is not valid or an override names no compartment of
            the structure.
        """
        return engine_state.initial_state(process, self)

    def get_conditioning_categories(
        self, process: Process, stratification_id: str
    ) -> tuple[str, ...]:
        """
        Get the categories a stratification is conditioned on.

        An unconditional stratification subdivides the whole model and has
        no conditioning categories. A conditional one subdivides only the
        subgroup its conditions select, and its fractions are relative to that
        subgroup rather than to the whole model.

        Parameters
        ----------
        process : Process
            The process of the model, such as ``model.process``.
        stratification_id : str
            Id of a stratification of the process.

        Returns
        -------
        tuple[str, ...]
            Ids of the categories selecting the subgroup this stratification
            subdivides, empty for an unconditional stratification. The
            categories of one stratification are a set: the subgroup has one
            of them (`subgroup_total`). A category name another
            stratification also declares is qualified, as in ``group:g1``.

        Raises
        ------
        ValueError
            If the stratification is not found.
        """
        for stratification in process.stratifications:
            if stratification.id == stratification_id:
                return tuple(
                    self.category_id(condition.stratification, category)
                    for condition in (stratification.conditions or [])
                    for category in condition.names()
                )
        raise ValueError(
            f"Stratification '{stratification_id}' not found. Available: "
            f"{sorted(s.id for s in process.stratifications)}"
        )

    def subgroup_total(self, process: Process, categories: Iterable[str] = ()) -> float:
        """
        Get the initial sum of the compartments of a subgroup.

        The sum is taken of the state the engine builds, overrides included,
        over every compartment that has, in each stratification the categories
        name, one of the categories of that stratification: ``["g2", "g3",
        "s1"]`` sums the compartments of ``g2`` or ``g3`` that have ``s1``.
        Passing no categories gives the sum of every compartment. It adds every bin it
        covers, so over bins of different units it has no unit.

        Parameters
        ----------
        process : Process
            The process of the model, such as ``model.process``.
        categories : Iterable[str], optional
            Category ids, bare or qualified (``group:g1``).

        Returns
        -------
        float
            The sum of the compartments of the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found or a bare name is ambiguous.
        """
        sets: dict[str, set[str]] = {}
        for category in categories:
            stratification, name = self.resolve_category(category)
            sets.setdefault(stratification, set()).add(name)
        return sum(
            compartment.amount
            for compartment in self.initial_state(process)
            if all(
                compartment.categories.get(stratification) in names
                for stratification, names in sets.items()
            )
        )

    # ------------------------------------------------------------------
    # Checks against the model
    # ------------------------------------------------------------------

    def _check_bin_amounts(self, process: Process) -> None:
        """Check that the bin amounts name every bin of the process."""
        actual_bins = set(self.bin_amounts)
        expected_bins = {bin_item.id for bin_item in process.bins}
        if actual_bins != expected_bins:
            raise ValueError(
                f"Initial bin amounts keys must exactly match bin ids. "
                f"Missing ids: {sorted(expected_bins - actual_bins)}, "
                f"Extra ids: {sorted(actual_bins - expected_bins)}."
            )


def split_groups(
    structure: Aggregate, process: Process, stratification: str
) -> list[tuple[tuple[str, ...], dict[str, float]]]:
    """
    Each split of a stratification, with the categories that select the
    group it splits.

    The group of one split is the group the conditions of the stratification
    select (`Aggregate.get_conditioning_categories`). The group of each split
    of a split by category adds the category of the stratification it differs
    by, last, so ``("g1",)`` for the split of ``g1``; that category takes the
    place of the categories a condition names in that stratification.

    Parameters
    ----------
    structure : Aggregate
        The structure that holds the fractions.
    process : Process
        The process of the model, such as ``model.process``.
    stratification : str
        Id of a stratification of the process.

    Returns
    -------
    list[tuple[tuple[str, ...], dict[str, float]]]
        The category ids of the group and the fraction of each category, one
        entry per split.
    """
    conditions = structure.get_conditioning_categories(process, stratification)
    fractions = structure.stratification_fractions[stratification]
    by = split_by(fractions)
    if by is None:
        return [(conditions, split) for _, split in category_splits(fractions)]
    others = tuple(c for c in conditions if structure.resolve_category(c)[0] != by)
    groups = []
    for of, split in category_splits(fractions):
        assert of is not None, "each split of a split by category has its category"
        groups.append(((*others, structure.category_id(by, of)), split))
    return groups


def _unit_name(unit: str | None) -> str:
    """A unit as a message names it."""
    return "no unit" if unit is None else f"'{unit}'"
