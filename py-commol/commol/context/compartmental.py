"""The structure of a compartmental model.

Bins expand over stratifications into compartments, transitions move
amounts between them, and initial conditions say how they start. The
engine (``commol_rs.model``) expands, plans and initializes the structure;
this module validates it and asks the engine.
"""

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, field_validator, model_validator

from commol.constants import RESERVED_VARIABLE_NAMES, SUBGROUP_TOTAL_PREFIX
from commol.context._model import engine_state
from commol.context._model.engine_state import InitialCompartment
from commol.context.bin import Accumulator, Bin
from commol.context.initial_conditions import InitialConditions
from commol.context.stratification import Stratification
from commol.context.structure import ModelStructure, ReportSections, SeriesLabel
from commol.context.transition import Transition
from commol.utils.security import get_expression_variables

if TYPE_CHECKING:
    from commol.context.calibration import CalibrationProblem
    from commol.context.model import Model
    from commol.context.probabilistic_calibration import ParameterSetStatistics


class CompartmentalStructure(ModelStructure):
    """
    Bins, stratifications, transitions and initial conditions of a model.

    Attributes
    ----------
    kind : Literal["compartmental"]
        The model kind.
    bins : list[Bin]
        The states of the model. Stratifications expand each one into
        compartments.
    accumulators : list[Accumulator]
        Counters of the flow of the transitions that name them.
    stratifications : list[Stratification]
        Categorical subdivisions of the bins, in expansion order.
    transitions : list[Transition]
        Flows between bins, or between the categories of one stratification.
    initial_conditions : InitialConditions
        The total and the fractions the compartments start with.
    """

    kind: Literal["compartmental"] = "compartmental"

    bins: list[Bin]
    accumulators: list[Accumulator] = Field(default_factory=list)
    stratifications: list[Stratification]
    transitions: list[Transition]
    initial_conditions: InitialConditions

    @field_validator("bins")
    @classmethod
    def validate_bins_not_empty(cls, v: list[Bin]) -> list[Bin]:
        """Require at least one bin."""
        if not v:
            raise ValueError("At least one bin must be defined.")
        return v

    @field_validator("transitions")
    @classmethod
    def validate_transitions_not_empty(cls, v: list[Transition]) -> list[Transition]:
        """Require at least one transition."""
        if not v:
            raise ValueError("At least one transition must be defined.")
        return v

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        """Validate that bin, stratification and transition ids are unique."""
        for label, ids in (
            ("bin", [item.id for item in self.bins]),
            ("stratification", [item.id for item in self.stratifications]),
            ("transition", [item.id for item in self.transitions]),
        ):
            if len(ids) != len(set(ids)):
                duplicates = [item for item in set(ids) if ids.count(item) > 1]
                raise ValueError(f"Duplicate {label} IDs found: {duplicates}")
        return self

    @model_validator(mode="after")
    def validate_bin_initial_conditions(self) -> Self:
        """Validate the bin fractions against the bins."""
        bin_fractions = {
            bf.bin: bf.fraction for bf in self.initial_conditions.bin_fractions
        }
        actual_bins = set(bin_fractions)
        expected_bins = {bin_item.id for bin_item in self.bins}
        if actual_bins != expected_bins:
            raise ValueError(
                f"Initial bin fractions keys must exactly match bin ids. "
                f"Missing ids: {expected_bins - actual_bins}, "
                f"Extra ids: {actual_bins - expected_bins}."
            )

        # The sum is checked once every fraction has a value; InitialConditions
        # checks the uncalibrated case.
        calibrated = [f for f in bin_fractions.values() if f is not None]
        if len(calibrated) == len(bin_fractions):
            total = sum(calibrated)
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(f"Bin fractions must sum to 1.0, but got {total:.7f}.")
        return self

    @model_validator(mode="after")
    def validate_stratified_rates(self) -> Self:
        """Validate that stratified rates name stratifications and categories."""
        strat_map = {strat.id: strat for strat in self.stratifications}

        for transition in self.transitions:
            for idx, stratified_rate in enumerate(transition.stratified_rates or []):
                for condition in stratified_rate.conditions:
                    if condition.stratification not in strat_map:
                        raise ValueError(
                            f"In transition '{transition.id}', stratified rate "
                            f"{idx}: Stratification "
                            f"'{condition.stratification}' not found. "
                            f"Available: {list(strat_map.keys())}"
                        )

                    # A condition without a category only redirects the target.
                    strat = strat_map[condition.stratification]
                    if (
                        condition.category is not None
                        and condition.category not in strat.categories
                    ):
                        raise ValueError(
                            f"In transition '{transition.id}', stratified rate "
                            f"{idx}: Category '{condition.category}' not found "
                            f"in stratification '{condition.stratification}'. "
                            f"Available: {strat.categories}"
                        )
        return self

    @model_validator(mode="after")
    def validate_stratification_initial_conditions(self) -> Self:
        """Validate the stratification fractions against the stratifications."""
        strat_map = {strat.id: strat for strat in self.stratifications}
        actual_strat = {
            sf.stratification for sf in self.initial_conditions.stratification_fractions
        }
        expected_strat = set(strat_map)
        if actual_strat != expected_strat:
            raise ValueError(
                f"Initial stratification fractions keys must exactly match "
                f"stratification ids. Missing ids: {expected_strat - actual_strat}, "
                f"Extra ids: {actual_strat - expected_strat}."
            )

        for strat_fractions in self.initial_conditions.stratification_fractions:
            strat_id = strat_fractions.stratification
            strat_instance = strat_map[strat_id]
            fractions = {sf.category: sf.fraction for sf in strat_fractions.fractions}

            expected = set(strat_instance.categories)
            actual = set(fractions)
            if actual != expected:
                raise ValueError(
                    f"Categories for stratification '{strat_id}' must exactly "
                    f"match defined categories in instance '{strat_instance.id}'. "
                    f"Missing categories: {expected - actual}, "
                    f"Extra categories: {actual - expected}."
                )

            total = sum(fractions.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"Stratification fractions for '{strat_id}' must sum to 1.0, "
                    f"but got {total:.7}."
                )
        return self

    # ------------------------------------------------------------------
    # The interface of a model kind
    # ------------------------------------------------------------------

    def check_model(self, model: "Model[Self]") -> None:
        """
        Check the structure against the parameters of its model.

        Parameter ids must not be engine variables or bin ids, every variable
        of a rate must be defined, transitions must move between bins or
        between the categories of one stratification, and the engine must
        plan every transition and build the initial state.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Raises
        ------
        ValueError
            If one of these checks fails.
        """
        self._check_parameter_names(model)
        engine_state.transition_flows(self)
        engine_state.check_names(model)
        if self.initial_conditions.compartment_fractions:
            engine_state.initial_state(self)
        self._check_formula_variables(model)
        self._check_transition_ids()

    def outputs_by_source(self) -> dict[str, list[str]]:
        """
        Map each bin and accumulator id to the output names it expands into.

        A bin or accumulator produces one output per combination of the
        stratification categories that apply to it. Without stratifications it
        produces a single output named after the id itself.

        Returns
        -------
        dict[str, list[str]]
            Bin and accumulator ids, each mapped to its output names in
            declaration order.
        """
        outputs: dict[str, list[str]] = {
            source_id: []
            for source_id in (
                *(bin_item.id for bin_item in self.bins),
                *(accumulator.id for accumulator in self.accumulators),
            )
        }
        compartments, accumulators = engine_state.expanded_outputs(self)
        for output in (*compartments, *accumulators):
            outputs[output.source].append(output.name)
        return outputs

    def series_label(self, series_id: str) -> SeriesLabel:
        """
        The name and unit of a bin, for a plot of its series.

        Parameters
        ----------
        series_id : str
            A bin id, or any other series id.

        Returns
        -------
        SeriesLabel
            The bin's name and unit, or the id and no unit when no bin has it.
        """
        bin_item = next((b for b in self.bins if b.id == series_id), None)
        if bin_item is None:
            return SeriesLabel(series_id, None)
        return SeriesLabel(bin_item.name or series_id, bin_item.unit)

    def uncalibrated_values(self) -> list[str]:
        """
        Ids of the bins whose initial fraction needs calibration.

        Returns
        -------
        list[str]
            Bin ids with a None fraction.
        """
        return self.initial_conditions.get_uncalibrated_bins()

    def with_calibrated_values(
        self,
        model: "Model[Self]",
        problem: "CalibrationProblem",
        values: Mapping[str, float],
    ) -> Self:
        """
        The structure with calibrated initial conditions written back.

        The written structure rebuilds the initial state the engine
        scored. Bin and category values update the fractions they declare.
        Every compartment those fractions do not reproduce, such as an exact
        compartment value or a value that overlaps a per-compartment override,
        is written as a per-compartment override.

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
        CompartmentalStructure
            A copy that holds the calibrated initial conditions.

        Raises
        ------
        ValueError
            If the engine cannot resolve an initial-condition id or the
            remainder, or the initial amounts exceed the total.
        """
        from commol.context._model.calibration_write_back import (
            calibrated_initial_conditions,
        )

        initial_conditions = calibrated_initial_conditions(model, problem, values)
        return self.model_copy(update={"initial_conditions": initial_conditions})

    def report_sections(
        self,
        model: "Model[Self]",  # noqa: ARG002
        calibrated_ids: set[str],
        parameter_statistics: "Mapping[str, ParameterSetStatistics] | None",
    ) -> ReportSections:
        """
        The ``initial_conditions`` section of a report of applied values.

        See :func:`commol.applied_parameters_report`.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, ParameterSetStatistics] | None
            Ensemble spread per id, or None without an ensemble.

        Returns
        -------
        ReportSections
            ``{"initial_conditions": ...}``.
        """
        from commol.context._model.report import initial_conditions_report

        return {
            "initial_conditions": initial_conditions_report(
                self, calibrated_ids, parameter_statistics
            )
        }

    def print_equations(
        self, model: "Model[Self]", output_file: str | None, format: str
    ) -> None:
        """Print the equations of the model; see `Model.print_equations`."""
        from commol.context._model.equations import EquationPrinter

        EquationPrinter(model).print_equations(output_file, format)

    def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
        """Check the units of the model; see `Model.check_unit_consistency`."""
        from commol.context._model.units import UnitChecker

        UnitChecker(model).check_unit_consistency(verbose)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def update_initial_conditions(
        self, bin_fractions: Mapping[str, float | None]
    ) -> None:
        """
        Update initial condition fractions for specified bins.

        Parameters
        ----------
        bin_fractions : Mapping[str, float | None]
            Dictionary mapping bin IDs to their new fraction values.

        Raises
        ------
        ValueError
            If a bin ID in the dictionary doesn't exist in the model.
        """
        self.initial_conditions.update_bin_fractions(bin_fractions)

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update initial stratification fractions for specified categories.

        Within each stratification, at most one category may be omitted; the
        omitted category receives the remaining fraction.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping category names to their new fraction values.

        Raises
        ------
        ValueError
            If a category doesn't exist in the model, a value lies outside
            [0.0, 1.0], a stratification has more than one category omitted, or
            the resulting fractions do not sum to 1.0.
        """
        self.initial_conditions.update_stratification_fractions(fractions)

    def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment initial condition overrides.

        A compartment already declared has its fraction replaced; a new one is
        appended. The engine validates the result before it is stored, so a
        rejected call leaves the structure unchanged.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new fraction
            of the total.

        Raises
        ------
        ValueError
            If a name is not an expanded compartment of the model, a value lies
            outside ``[0.0, 1.0]``, or the resulting overrides make the initial
            amounts exceed the total.
        """
        candidate = self.initial_conditions.model_copy(deep=True)
        candidate.update_compartment_fractions(fractions)
        engine_state.initial_state(
            self.model_copy(update={"initial_conditions": candidate})
        )
        self.initial_conditions = candidate

    def initial_state(self) -> list[InitialCompartment]:
        """
        Expanded compartments and the amount each one starts with.

        The engine builds this state from the bin fractions, the stratification
        fractions and the per-compartment overrides; it is the state a
        simulation starts from.

        Returns
        -------
        list[InitialCompartment]
            One entry per expanded compartment, in engine order.

        Raises
        ------
        ValueError
            If the per-compartment overrides are not valid for the structure.
        """
        return engine_state.initial_state(self)

    def get_conditioning_categories(self, stratification_id: str) -> tuple[str, ...]:
        """
        Get the categories a stratification is conditioned on.

        An unconditional stratification subdivides the whole model and has
        no conditioning categories. A conditional one subdivides only the
        subgroup its conditions select, and its fractions are relative to that
        subgroup rather than to the whole model.

        Parameters
        ----------
        stratification_id : str
            Id of a stratification of the model.

        Returns
        -------
        tuple[str, ...]
            Ids of the categories selecting the subgroup this stratification
            subdivides, empty for an unconditional stratification. A category
            name another stratification also declares is qualified, as in
            ``sex:female``.

        Raises
        ------
        ValueError
            If the stratification is not found.
        """
        for stratification in self.stratifications:
            if stratification.id == stratification_id:
                return tuple(
                    self.initial_conditions.category_id(
                        condition.stratification, condition.category
                    )
                    for condition in (stratification.conditions or [])
                    if condition.category is not None
                )
        raise ValueError(
            f"Stratification '{stratification_id}' not found. Available: "
            f"{sorted(s.id for s in self.stratifications)}"
        )

    def subgroup_total(self, categories: Iterable[str] = ()) -> float:
        """
        Get the initial total of a subgroup.

        See
        :meth:`~commol.context.initial_conditions.InitialConditions.subgroup_total`.

        Per-compartment overrides are folded into the sum for every compartment
        the category chain selects, so the result matches the state the engine
        builds. Without overrides the declared fractions are multiplied.

        Parameters
        ----------
        categories : Iterable[str], optional
            Category ids, bare or qualified (``sex:female``), forming a chain
            from the whole model down to the subgroup, outermost first.

        Returns
        -------
        float
            The sum of the compartments of the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found or a bare name is ambiguous.
        """
        selected = tuple(categories)
        initial_conditions = self.initial_conditions
        if not initial_conditions.compartment_fractions:
            return initial_conditions.subgroup_total(selected)
        resolved = [initial_conditions.resolve_category(c) for c in selected]
        return sum(
            compartment.amount
            for compartment in self.initial_state()
            if all(
                compartment.categories.get(stratification) == name
                for stratification, name in resolved
            )
        )

    # ------------------------------------------------------------------
    # Checks against the model
    # ------------------------------------------------------------------

    def _check_parameter_names(self, model: "Model[Self]") -> None:
        """
        Check that parameter ids are not engine variables or bin ids.

        Expanded compartment names and subgroup totals are checked by the
        engine (``engine_state.check_names``).
        """
        bin_ids = {bin_item.id for bin_item in self.bins}
        parameter_ids = {p.id for p in model.parameters}

        special = sorted(parameter_ids & set(RESERVED_VARIABLE_NAMES))
        if special:
            raise ValueError(
                f"Parameter IDs conflict with engine variables: {special}. "
                f"{', '.join(RESERVED_VARIABLE_NAMES)} are reserved. "
                f"Please rename the conflicting parameters."
            )

        conflicting_names = bin_ids & parameter_ids
        if conflicting_names:
            raise ValueError(
                f"Parameter IDs conflict with reserved compartment names: "
                f"{sorted(conflicting_names)}. "
                f"Bin IDs ({sorted(bin_ids)}) are reserved variables. "
                f"Please rename the conflicting parameters."
            )

    def _check_formula_variables(self, model: "Model[Self]") -> None:
        """Check that every variable of a rate expression is defined."""
        parameter_ids = {param.id for param in model.parameters}
        valid_identifiers = parameter_ids | self._structure_identifiers()

        for transition in self.transitions:
            rates = [("rate", transition.rate)] if transition.rate else []
            rates += [
                ("stratified_rate", stratified.rate)
                for stratified in transition.stratified_rates or []
            ]
            for context, rate in rates:
                self._check_rate_variables(
                    rate, transition.id, context, valid_identifiers, parameter_ids
                )

    def _structure_identifiers(self) -> set[str]:
        """Identifiers the structure and the engine define for rates."""
        category_ids = {
            cat for strat in self.stratifications for cat in strat.categories
        }
        return (
            {bin_item.id for bin_item in self.bins}
            | category_ids
            | set(RESERVED_VARIABLE_NAMES)
            | self._subgroup_total_names()
            | self._full_compartment_names()
        )

    def _subgroup_total_names(self) -> set[str]:
        """Names of the subgroup totals the engine provides for rates."""
        return set(engine_state.subgroup_total_names(self))

    def _full_compartment_names(self) -> set[str]:
        """Returns all full stratified compartment names."""
        if not self.stratifications:
            return set()
        compartments, _ = engine_state.expanded_outputs(self)
        return {output.name for output in compartments}

    def _check_rate_variables(
        self,
        rate: str,
        transition_id: str,
        context: str,
        valid_identifiers: set[str],
        parameter_ids: set[str],
    ) -> None:
        """Check the variables of one rate expression."""
        variables = get_expression_variables(rate)
        undefined_vars = [var for var in variables if var not in valid_identifiers]
        if not undefined_vars:
            return
        bin_ids = {bin_item.id for bin_item in self.bins}
        message = (
            f"Undefined variables in transition '{transition_id}' "
            f"{context} '{rate}': {', '.join(undefined_vars)}. "
            f"Available parameters: "
            f"{', '.join(sorted(parameter_ids)) if parameter_ids else 'none'}. "
            f"Available bins: "
            f"{', '.join(sorted(bin_ids)) if bin_ids else 'none'}."
        )
        if any(var.startswith(SUBGROUP_TOTAL_PREFIX) for var in undefined_vars):
            names = sorted(
                name
                for name in self._subgroup_total_names()
                if name.startswith(SUBGROUP_TOTAL_PREFIX)
            )
            message += (
                " Subgroup total names follow the declaration order of the "
                "stratifications. Available subgroup totals: "
                f"{', '.join(names) if names else 'none'}."
            )
        raise ValueError(message)

    def _check_transition_ids(self) -> None:
        """
        Check that each transition moves between bins, or between the
        categories of one stratification.
        """
        bin_ids = {bin_item.id for bin_item in self.bins}
        category_to_stratification = {
            cat: strat.id for strat in self.stratifications for cat in strat.categories
        }
        categories_ids = set(category_to_stratification)
        bin_and_categories_ids = bin_ids | categories_ids

        for transition in self.transitions:
            transition_ids = set(transition.source) | set(transition.target)

            if not transition_ids.issubset(bin_and_categories_ids):
                invalid_ids = transition_ids - bin_and_categories_ids
                raise ValueError(
                    f"Transition '{transition.id}' contains invalid ids: "
                    f"{invalid_ids}. Ids must be defined in Bin ids "
                    f"or Stratification Categories."
                )

            is_bin_flow = transition_ids.issubset(bin_ids)
            is_stratification_flow = transition_ids.issubset(categories_ids)

            if not is_bin_flow and not is_stratification_flow:
                raise ValueError(
                    f"Transition '{transition.id}' mixes id types. "
                    f"Found Bin ids ({transition_ids & bin_ids}) and "
                    f"Stratification Categories ids "
                    f"({transition_ids & categories_ids}). "
                    "Transitions must be purely Bin flow or purely "
                    f"Stratification flow."
                )

            if is_stratification_flow:
                parent_stratification_ids = {
                    category_to_stratification[cat_id] for cat_id in transition_ids
                }
                if len(parent_stratification_ids) > 1:
                    raise ValueError(
                        f"Transition '{transition.id}' is a Stratification flow "
                        f"but involves categories from multiple stratifications: "
                        f"{', '.join(parent_stratification_ids)}. A single "
                        f"transition must only move between categories belonging "
                        f"to the same parent stratification."
                    )
