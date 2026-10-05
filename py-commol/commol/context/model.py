from collections.abc import Iterable, Mapping
from itertools import combinations, product
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from commol.constants import PrintEquationsOutputFormat
from commol.context._model.serialization import render_json
from commol.context.calibration import (
    CalibrationProblem,
    CalibrationResult,
    calibrated_values,
)
from commol.context.compartments import (
    expand_over_stratifications,
    generate_compartment_names,
    uniform_compartment_fractions,
)
from commol.context.constants import CalibrationParameterType
from commol.context.dynamics import Dynamics, Transition
from commol.context.parameter import Parameter
from commol.context.population import Population
from commol.context.probabilistic_calibration import ProbabilisticCalibrationResult
from commol.utils.security import get_expression_variables


class Model(BaseModel):
    """
    Root class of compartment model.

    Attributes
    ----------
    name : str
        A unique name that identifies the model.
    description : str | None
        A human-readable description of the model's purpose and function.
    version : str | None
        The version number of the model.
    population : Population
        Population details, subpopulations, stratifications and initial conditions.
    parameters : list[Parameter]
        A list of global model parameters.
    dynamics : Dynamics
        The rules that govern system evolution.
    """

    name: str = Field(..., description="Name which identifies the model.")
    description: str | None = Field(
        None,
        description="Human-readable description of the model's purpose and function.",
    )
    version: str | None = Field(None, description="Version number of the model.")

    population: Population
    parameters: list[Parameter]
    dynamics: Dynamics

    # Compartment pairs depend only on the (immutable after construction)
    # stratifications, so memoizing them per model instance keeps repeated
    # report lookups linear in the number of categories.
    _category_pairs_cache: dict[str, list[tuple[str, str]]] = PrivateAttr(
        default_factory=dict
    )

    @classmethod
    def from_json(cls, file_path: str | Path) -> Self:
        """
        Loads a model from a JSON file.

        The method reads the specified JSON file, parses its content, and validates
        it against the Model schema.

        Parameters
        ----------
        file_path : str | Path
            The path to the JSON file.

        Returns
        -------
        Model
            A validated Model instance.

        Raises
        ------
        FileNotFoundError
            If the file at `file_path` does not exist.
        pydantic.ValidationError
            If the JSON content does not conform to the Model schema.
        """
        with open(file_path, "r") as f:
            json_data = f.read()

        return cls.model_validate_json(json_data)

    def to_json(self, file_path: str | Path, indent: int = 2) -> None:
        """
        Save the model to a JSON file.

        Every field is written, including those left unset, so the file always
        reloads through :meth:`from_json` to an equal model.

        Long numeric arrays, such as a time-series parameter, are written on a
        single line.

        Parameters
        ----------
        file_path : str | Path
            Path of the file to write.
        indent : int, optional
            Indentation width for nested structures.
        """
        payload = self.model_dump(mode="json")
        Path(file_path).write_text(
            render_json(payload, indent=indent), encoding="utf-8"
        )

    @model_validator(mode="after")
    def validate_unique_parameter_ids(self) -> Self:
        """
        Validates that parameter IDs are unique.
        """
        parameter_ids = [p.id for p in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            duplicates = [
                item for item in set(parameter_ids) if parameter_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
        return self

    @model_validator(mode="after")
    def validate_parameter_names_not_reserved(self) -> Self:
        """
        Validates that parameter IDs do not conflict with reserved variable names.

        Bin IDs are reserved variables that cannot be used as parameter names.
        When stratifications are present, base compartment names represent the
        sum of all stratified versions (e.g., S = S_young + S_old).

        Parameters cannot use these reserved names to avoid conflicts during
        rate expression evaluation.
        """
        bin_ids = {bin_item.id for bin_item in self.population.bins}
        parameter_ids = {p.id for p in self.parameters}

        conflicting_names = bin_ids & parameter_ids
        if conflicting_names:
            raise ValueError(
                f"Parameter IDs conflict with reserved compartment names: "
                f"{sorted(conflicting_names)}. "
                f"Bin IDs ({sorted(bin_ids)}) are reserved variables. "
                f"Please rename the conflicting parameters."
            )
        return self

    def update_parameters(self, parameter_values: Mapping[str, float | None]) -> None:
        """
        Update parameter values in the model.

        Parameters
        ----------
        parameter_values : Mapping[str, float | None]
            Dictionary mapping parameter IDs to their new values.

        Raises
        ------
        ValueError
            If a parameter ID in the dictionary doesn't exist in the model.
        """
        param_dict = {param.id: param for param in self.parameters}

        for param_id, value in parameter_values.items():
            if param_id not in param_dict:
                raise ValueError(
                    (
                        f"Parameter '{param_id}' not found in model. "
                        f"Available parameters: {', '.join(param_dict.keys())}"
                    )
                )
            param_dict[param_id].value = value

    def get_uncalibrated_parameters(self) -> list[str]:
        """
        Get a list of parameter IDs that have None values (need calibration).

        Returns
        -------
        list[str]
            List of parameter IDs that require calibration.
        """
        return [param.id for param in self.parameters if param.value is None]

    def get_uncalibrated_initial_conditions(self) -> list[str]:
        """
        Get a list of bin IDs that have None fractions (need calibration).

        Returns
        -------
        list[str]
            List of bin IDs with uncalibrated initial conditions.
        """
        return self.population.initial_conditions.get_uncalibrated_bins()

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
        self.population.initial_conditions.update_bin_fractions(bin_fractions)

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
        self.population.initial_conditions.update_stratification_fractions(fractions)

    def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment initial condition overrides.

        A compartment already declared in the model definition has its
        fraction replaced; a new one is appended.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new fraction
            of the total population.

        Raises
        ------
        ValueError
            If a name is not an expanded compartment of the model, a value lies
            outside ``[0.0, 1.0]``, or the resulting overrides make the initial
            population exceed ``population_size``.
        """
        available = set(
            generate_compartment_names(
                self.population.bins, self.population.stratifications
            )
        )
        unknown = sorted(set(fractions) - available)
        if unknown:
            raise ValueError(
                f"Unknown compartment(s) in initial conditions: {unknown}. "
                f"Available compartments: {sorted(available)}"
            )
        self._validate_compartment_fraction_total(fractions)
        self.population.initial_conditions.update_compartment_fractions(fractions)

    def _validate_compartment_fraction_total(
        self, new_fractions: Mapping[str, float]
    ) -> None:
        """
        Reject an upsert whose post-upsert compartment total exceeds the population.

        Every expanded compartment contributes its override fraction, or the
        expansion fraction it keeps when it carries no override. The check runs
        on the candidate overrides, so a rejected upsert leaves the model
        unchanged.

        Parameters
        ----------
        new_fractions : Mapping[str, float]
            Compartment overrides to validate against the current ones.

        Raises
        ------
        ValueError
            If the candidate overrides make the initial population exceed
            ``population_size``.
        """
        expanded = [name for name, _ in self._expanded_compartments()]
        expanded_set = set(expanded)
        base = self._base_compartment_fractions()
        candidate = {
            override.compartment: override.fraction
            for override in self.population.initial_conditions.compartment_fractions
        }
        candidate.update(new_fractions)
        total = sum(candidate.get(name, base.get(name, 0.0)) for name in expanded)
        total += sum(
            value for name, value in candidate.items() if name not in expanded_set
        )
        if total > 1.0 + 1e-6:
            raise ValueError(
                "Per-compartment overrides would make the initial population "
                "exceed population_size."
            )

    def apply_calibration_parameters(
        self,
        result: CalibrationResult
        | ProbabilisticCalibrationResult
        | Mapping[str, float],
        problem: CalibrationProblem,
    ) -> None:
        """
        Write calibrated values back onto the model.

        Each value is routed by the ``parameter_type`` declared for it in
        ``problem``: ``parameter`` values update model parameters,
        ``initial_condition`` values update bin fractions, stratification
        category fractions, or expanded-compartment fractions, and ``scale``
        values are ignored because they apply to observations rather than to
        the model.

        Parameters
        ----------
        result : CalibrationResult | ProbabilisticCalibrationResult | Mapping
            The calibration outcome, or a mapping of parameter id to value.
            For a probabilistic result the point parameters of the selected
            ensemble are applied.
        problem : CalibrationProblem
            The problem the result came from, used to resolve parameter types.

        Raises
        ------
        ValueError
            If an id is not declared in ``problem``, or if an initial condition
            id is neither a bin, a stratification category, nor an expanded
            compartment of the model.
        """
        values = calibrated_values(result)
        types = {param.id: param.parameter_type for param in problem.parameters}
        unknown = sorted(set(values) - set(types))
        if unknown:
            raise ValueError(
                f"Cannot apply values with no declared parameter type: {unknown}. "
                f"Declared calibration parameters: {sorted(types)}"
            )

        compartments = set(
            generate_compartment_names(
                self.population.bins, self.population.stratifications
            )
        )
        parameters, bin_fractions, stratification_fractions, compartment_fractions = (
            self._route_calibration_values(values, types, compartments)
        )

        # Applying the routed values mutates several definition blocks in turn.
        # Snapshot them so a validation failure late in the sequence (for
        # example an over-allocating compartment override) rolls the model back
        # to the state it had before the call rather than leaving a partial
        # write-back.
        parameters_snapshot = {param.id: param.value for param in self.parameters}
        initial_conditions_snapshot = self.population.initial_conditions.model_copy(
            deep=True
        )
        try:
            if parameters:
                self.update_parameters(parameters)
            if bin_fractions:
                # The routing diverts any bin value whose update would break
                # the declared mapping, so the public conservation guard is
                # safe and is re-applied here as the single source of truth.
                self.update_initial_conditions(bin_fractions)
            if stratification_fractions:
                self.update_stratification_fractions(stratification_fractions)
            if compartment_fractions:
                # Calibration write-back is as permissive as the engine: the
                # scored engine state may be non-conserving (the Rust
                # ``set_initial_condition`` has no total check), so persisting
                # it must not be rejected by the definition-level total guard.
                self.population.initial_conditions._upsert_compartment_fractions(
                    compartment_fractions, validate_total=False
                )
        except Exception:
            for param in self.parameters:
                if param.id in parameters_snapshot:
                    param.value = parameters_snapshot[param.id]
            self.population.initial_conditions = initial_conditions_snapshot
            raise

    def _base_compartment_fractions(self) -> dict[str, float]:
        """Uniform expansion fractions of the declared bin and category values."""
        initial = self.population.initial_conditions
        return uniform_compartment_fractions(
            self.population.bins,
            self.population.stratifications,
            {bf.bin: bf.fraction for bf in initial.bin_fractions},
            (
                (
                    stratification.stratification,
                    {
                        fraction.category: fraction.fraction
                        for fraction in stratification.fractions
                    },
                )
                for stratification in initial.stratification_fractions
            ),
        )

    def _initial_population_fractions(self) -> dict[str, float]:
        """Expansion fractions with the current compartment overrides applied."""
        fractions = self._base_compartment_fractions()
        for override in self.population.initial_conditions.compartment_fractions:
            fractions[override.compartment] = override.fraction
        return fractions

    def _expanded_compartments(self) -> list[tuple[str, dict[str, str]]]:
        """Expanded compartment names with the categories applied to each."""
        return expand_over_stratifications(
            (bin_item.id for bin_item in self.population.bins),
            self.population.stratifications,
        )

    def _covered_compartments(self, param_id: str) -> list[str]:
        """Compartments whose initial condition the parameter id updates."""
        prefix = f"{param_id}_"
        return [
            name
            for name, _ in self._expanded_compartments()
            if name == param_id or name.startswith(prefix)
        ]

    def _declaring_stratifications(self, category: str) -> int:
        """Number of stratifications that declare a fraction for a category."""
        declared = self.population.initial_conditions.stratification_fractions
        return sum(
            1
            for stratification in declared
            if any(
                fraction.category == category for fraction in stratification.fractions
            )
        )

    def _category_pairs(self, category: str) -> list[tuple[str, str]]:
        """Pairs of compartments that differ only in one category token."""
        cached = self._category_pairs_cache.get(category)
        if cached is not None:
            return cached
        expanded = self._expanded_compartments()
        tokens_by_name = {name: name.split("_") for name, _ in expanded}
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for name, tokens in tokens_by_name.items():
            for index, token in enumerate(tokens):
                if index == 0 or token != category:
                    continue
                complements = [
                    candidate
                    for candidate, candidate_tokens in tokens_by_name.items()
                    if len(candidate_tokens) == len(tokens)
                    and candidate_tokens[index] != category
                    and all(
                        position == index or candidate_tokens[position] == other
                        for position, other in enumerate(tokens)
                    )
                ]
                if len(complements) != 1:
                    continue
                pair: tuple[str, str] = (name, complements[0])
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        # Match the engine's replay order: it sorts its complementary pairs by
        # compartment index (calibration_problem.rs). Order matters when a
        # category is shared by two stratifications and pairs overlap.
        position = {name: index for index, (name, _) in enumerate(expanded)}
        pairs.sort(key=lambda pair: (position[pair[0]], position[pair[1]]))
        self._category_pairs_cache[category] = pairs
        return pairs

    @staticmethod
    def _aggregate_weights(
        covered: list[str],
        base_fractions: Mapping[str, float],
    ) -> dict[str, float]:
        """Share of a bin-level value each covered compartment receives."""
        total = sum(base_fractions.get(name, 0.0) for name in covered)
        if total > 0.0:
            return {name: base_fractions.get(name, 0.0) / total for name in covered}
        return dict.fromkeys(covered, 1.0 / len(covered))

    def _route_calibration_values(
        self,
        values: Mapping[str, float],
        types: Mapping[str, str],
        compartments: set[str],
    ) -> tuple[
        dict[str, float | None],
        dict[str, float | None],
        dict[str, float],
        dict[str, float],
    ]:
        """
        Route calibrated values to the definition fields they update.

        Initial-condition values are replayed in parameter order over a single
        working population that starts from the declared expansion plus the
        current compartment overrides, so a bin-level value scales its covered
        compartments by the weights the engine precomputes from that base and a
        category value splits each complementary pair by the live pair total.
        Engine parity is the primary invariant: each parameter reads the
        working population once and writes its targets, exactly as
        ``calibration_problem.rs`` snapshots the engine population once per
        parameter.

        A bin or category value whose covered compartments intersect an
        existing override, or a compartment affected by an earlier value in
        this replay, is persisted as per-compartment overrides holding the
        replayed values. A bin value whose update would make the complete
        declared bin-fraction mapping fail its conservation rule is diverted
        the same way, so the definition never holds a non-conserving bin
        fraction. A stratification-category value is also persisted when
        its name is declared by more than one stratification, because a single
        stratification fraction cannot express the cross-stratification split.
        Every other value updates the bin or stratification fraction it
        declares. An exact compartment value is always persisted as an override.

        Parameters
        ----------
        values : Mapping[str, float]
            Calibrated value per parameter id.
        types : Mapping[str, str]
            Declared ``parameter_type`` per parameter id, in problem order.
        compartments : set[str]
            Expanded compartment names of the model.

        Returns
        -------
        tuple
            Parameter values, bin fractions, stratification fractions, and
            compartment fractions, in that order.

        Raises
        ------
        ValueError
            If an initial condition id is neither a bin, a stratification
            category, nor an expanded compartment of the model.
        """
        bin_ids = {bin_item.id for bin_item in self.population.bins}
        categories = self.population.initial_conditions.get_categories_with_fractions()
        initial_conditions = self.population.initial_conditions
        population_size = float(initial_conditions.population_size)
        if population_size <= 0.0:
            raise ValueError(
                "population_size must be greater than 0 before calibration values "
                f"can be written back, got {initial_conditions.population_size}."
            )
        base_fractions = self._initial_population_fractions()
        working = {
            name: base_fractions.get(name, 0.0) * population_size
            for name, _ in self._expanded_compartments()
        }
        affected: set[str] = {
            fraction.compartment
            for fraction in initial_conditions.compartment_fractions
        }
        diverted: set[str] = set()

        parameters: dict[str, float | None] = {}
        bin_fractions: dict[str, float | None] = {}
        stratification_fractions: dict[str, float] = {}

        for param_id, parameter_type in types.items():
            if param_id not in values:
                continue
            value = values[param_id]
            if parameter_type == CalibrationParameterType.PARAMETER:
                parameters[param_id] = value
            elif parameter_type == CalibrationParameterType.SCALE:
                continue
            else:
                route = self._replay_initial_condition(
                    param_id,
                    value,
                    bin_ids,
                    categories,
                    compartments,
                    base_fractions,
                    working,
                    affected,
                    diverted,
                    bin_fractions,
                    population_size,
                )
                if route is not None:
                    field, key = route
                    if field == "bin":
                        bin_fractions[key] = value
                    else:
                        stratification_fractions[key] = value

        compartment_fractions = {
            name: working[name] / population_size for name in sorted(diverted)
        }

        return (
            parameters,
            bin_fractions,
            stratification_fractions,
            compartment_fractions,
        )

    def _bin_update_keeps_the_mapping_valid(
        self,
        bin_id: str,
        value: float,
        routed_bins: Mapping[str, float | None],
    ) -> bool:
        """
        Whether setting a bin fraction keeps the declared mapping valid.

        Calibration write-back must not persist a bin fraction the definition
        cannot hold: a value whose update makes the declared bin-fraction
        mapping fail :meth:`_validate_bin_fraction_total` is diverted to
        per-compartment overrides instead. The candidate applies the bin
        values routed earlier in the same replay, so a combination that is
        only valid as a whole (every bin named) is not split apart.

        Parameters
        ----------
        bin_id : str
            Id of the bin being updated.
        value : float
            Candidate fraction for that bin.
        routed_bins : Mapping[str, float | None]
            Bin fractions already routed in this replay.

        Returns
        -------
        bool
            True when the candidate mapping passes the conservation rule.
        """
        initial_conditions = self.population.initial_conditions
        candidate = {
            fraction.bin: fraction.fraction
            for fraction in initial_conditions.bin_fractions
        }
        candidate.update(routed_bins)
        candidate[bin_id] = value
        known = {fraction.bin for fraction in initial_conditions.bin_fractions}
        complete = set(routed_bins) | {bin_id} >= known
        try:
            initial_conditions._validate_bin_fraction_total(
                candidate, complete=complete
            )
        except ValueError:
            return False
        return True

    def _replay_initial_condition(
        self,
        param_id: str,
        value: float,
        bin_ids: set[str],
        categories: set[str],
        compartments: set[str],
        base_fractions: Mapping[str, float],
        working: dict[str, float],
        affected: set[str],
        diverted: set[str],
        routed_bins: Mapping[str, float | None],
        population_size: float,
    ) -> tuple[str, str] | None:
        """
        Apply one initial-condition value to the working population.

        Parameters
        ----------
        param_id : str
            Id of the calibrated initial condition.
        value : float
            Calibrated fraction.
        bin_ids : set[str]
            Bin ids of the model.
        categories : set[str]
            Category names that declare an initial fraction.
        compartments : set[str]
            Expanded compartment names of the model.
        base_fractions : Mapping[str, float]
            Starting population fractions including the current overrides.
        working : dict[str, float]
            Working head counts including the current overrides, updated in
            place.
        affected : set[str]
            Compartments that carry an existing override or were covered by an
            earlier value in this replay.
        diverted : set[str]
            Compartments persisted as overrides in this replay.
        routed_bins : Mapping[str, float | None]
            Bin fractions already routed in this replay, used to test the
            conservation of a later bin update against the whole mapping.
        population_size : float
            Total population the fractions are taken of.

        Returns
        -------
        tuple[str, str] | None
            Definition field to update and its key, or None when the value is
            persisted as compartment overrides.

        Raises
        ------
        ValueError
            If the id is neither a bin, a stratification category, nor an
            expanded compartment of the model.
        """
        if param_id in bin_ids:
            covered = self._covered_compartments(param_id)
            weights = self._aggregate_weights(covered, base_fractions)
            for name, weight in weights.items():
                working[name] = value * weight * population_size
            route: tuple[str, str] | None = None
            # Divert when the covered compartments intersect an existing
            # override or an earlier value, and when updating the declared bin
            # fraction would make the complete mapping fail its conservation
            # rule. The replayed head counts are pinned as overrides either
            # way, so the definition stays valid and rebuilds the scored state.
            if set(covered) & affected or not self._bin_update_keeps_the_mapping_valid(
                param_id, value, routed_bins
            ):
                diverted.update(covered)
            else:
                route = ("bin", param_id)
            affected.update(covered)
            return route
        if param_id in categories:
            pairs = self._category_pairs(param_id)
            covered = {name for pair in pairs for name in pair}
            # The engine takes the population snapshot once per parameter and
            # then iterates its pairs (calibration_problem.rs). Reading the live
            # working totals inside the loop made overlapping pairs (a category
            # shared by two stratifications) compound their edits, diverging
            # from the engine replay. A shared category is always diverted,
            # because no single stratification fraction can express the
            # cross-stratification split, and a repeated apply re-scores from
            # the reloaded population exactly as a fresh engine problem would.
            shared = self._declaring_stratifications(param_id) > 1
            pair_totals = [
                working[category_name] + working[complement_name]
                for category_name, complement_name in pairs
            ]
            for (category_name, complement_name), pair_total in zip(pairs, pair_totals):
                working[category_name] = value * pair_total
                working[complement_name] = (1.0 - value) * pair_total
            route = None
            # A category declared by more than one stratification cannot be
            # represented by a single stratification fraction, so the replayed
            # head counts are persisted as per-compartment overrides instead.
            if covered & affected or shared:
                diverted.update(covered)
            else:
                route = ("stratification", param_id)
            affected.update(covered)
            return route
        if param_id in compartments:
            working[param_id] = value * population_size
            diverted.add(param_id)
            affected.add(param_id)
            return None
        raise ValueError(
            f"Initial condition '{param_id}' is neither a bin, a "
            f"stratification category, nor an expanded compartment. "
            f"Available bins: {sorted(bin_ids)}. Available categories: "
            f"{sorted(categories)}. Available compartments: "
            f"{sorted(compartments)}"
        )

    @model_validator(mode="after")
    def validate_formula_variables(self) -> Self:
        """
        Validate that all variables in rate expressions are defined.
        This is done by gathering all valid identifiers and checking each
        transition's rate expressions against them.
        """
        valid_identifiers = self._get_valid_identifiers()

        for transition in self.dynamics.transitions:
            self._validate_transition_rates(transition, valid_identifiers)
        return self

    def _get_valid_identifiers(self) -> set[str]:
        """Gathers all valid identifiers for use in rate expressions."""
        special_vars = {"N", "step", "pi", "e", "t"}
        param_ids = {param.id for param in self.parameters}
        bin_ids = {bin_item.id for bin_item in self.population.bins}

        strat_category_ids: set[str] = {
            cat for strat in self.population.stratifications for cat in strat.categories
        }

        subpopulation_n_vars = self._get_subpopulation_n_vars()
        full_compartment_names = self._get_full_compartment_names()
        bin_subpopulation_vars = self._get_bin_subpopulation_vars()

        return (
            param_ids
            | bin_ids
            | strat_category_ids
            | special_vars
            | subpopulation_n_vars
            | full_compartment_names
            | bin_subpopulation_vars
        )

    def _get_subpopulation_n_vars(self) -> set[str]:
        """Generates all possible N_{category...} variable names."""
        if not self.population.stratifications:
            return set()

        subpopulation_n_vars: set[str] = set()
        category_groups = [s.categories for s in self.population.stratifications]

        full_category_combos = product(*category_groups)

        for combo_tuple in full_category_combos:
            for i in range(1, len(combo_tuple) + 1):
                for subset in combinations(combo_tuple, i):
                    var_name = f"N_{'_'.join(subset)}"
                    subpopulation_n_vars.add(var_name)

        return subpopulation_n_vars

    def get_conditioning_categories(self, stratification_id: str) -> tuple[str, ...]:
        """
        Get the categories a stratification is conditioned on.

        An unconditional stratification subdivides the whole population and has
        no conditioning categories. A conditional one subdivides only the
        subgroup its conditions select, and its fractions are relative to that
        subgroup rather than to the whole population.

        Parameters
        ----------
        stratification_id : str
            Id of a stratification of the model.

        Returns
        -------
        tuple[str, ...]
            Categories selecting the subgroup this stratification subdivides,
            empty for an unconditional stratification.

        Raises
        ------
        ValueError
            If the stratification is not found.
        """
        for stratification in self.population.stratifications:
            if stratification.id == stratification_id:
                return tuple(
                    condition.category
                    for condition in (stratification.conditions or [])
                    if condition.category is not None
                )
        raise ValueError(
            f"Stratification '{stratification_id}' not found in model "
            f"'{self.name}'. Available: "
            f"{sorted(s.id for s in self.population.stratifications)}"
        )

    def subgroup_population(self, categories: Iterable[str] = ()) -> float:
        """
        Get the initial head count of a subgroup.

        See
        :meth:`~commol.context.initial_conditions.InitialConditions.subgroup_population`.

        Per-compartment overrides are folded into the sum for every compartment
        the category chain selects, not just when it selects one, so the head
        count matches the population the engine builds. Without overrides the
        declared fractions are multiplied exactly as before.

        Parameters
        ----------
        categories : Iterable[str], optional
            Category names forming a chain from the whole population down to
            the subgroup, outermost first.

        Returns
        -------
        float
            Number of individuals in the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found.
        """
        selected = tuple(categories)
        initial_conditions = self.population.initial_conditions
        # Validate the chain first so an unknown category still raises.
        for category in selected:
            initial_conditions.get_category_fraction(category)
        if not initial_conditions.compartment_fractions:
            return initial_conditions.subgroup_population(selected)
        population_size = float(initial_conditions.population_size)
        effective = self._initial_population_fractions()
        total = sum(
            effective.get(name, 0.0)
            for name, applied in self._expanded_compartments()
            if all(category in applied.values() for category in selected)
        )
        return total * population_size

    def get_outputs_by_source(self) -> dict[str, list[str]]:
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
        from commol.context._model.helpers import ModelCompartmentHelper

        helper = ModelCompartmentHelper(self)
        outputs: dict[str, list[str]] = {
            source_id: []
            for source_id in (
                *(bin_item.id for bin_item in self.population.bins),
                *(accumulator.id for accumulator in self.population.accumulators),
            )
        }
        expanded = (
            *helper.generate_compartments(),
            *helper.generate_accumulator_outputs(),
        )
        for source_id, applied in expanded:
            name = "_".join(
                [source_id]
                + [
                    applied[stratification.id]
                    for stratification in self.population.stratifications
                    if stratification.id in applied
                ]
            )
            outputs[source_id].append(name)
        return outputs

    def _get_full_compartment_names(self) -> set[str]:
        """Returns all full stratified compartment names."""
        if not self.population.stratifications:
            return set()

        from commol.context._model.helpers import ModelCompartmentHelper

        helper = ModelCompartmentHelper(self)
        compartments = helper.generate_compartments()
        return {
            helper.compartment_to_string(comp, PrintEquationsOutputFormat.TEXT)
            for comp in compartments
        }

    def _get_bin_subpopulation_vars(self) -> set[str]:
        """Generates partial bin-stratification sum variable names.

        For bins and stratifications, generates the combinations. These represent
        partial sums over one or more stratification dimensions. Requires 2+
        stratifications (with 1, partial sums duplicate existing names).
        """
        if len(self.population.stratifications) < 2:
            return set()

        bin_strat_vars: set[str] = set()
        bin_ids = [bin_item.id for bin_item in self.population.bins]
        category_groups = [s.categories for s in self.population.stratifications]

        for combo_tuple in product(*category_groups):
            num_cats = len(combo_tuple)
            full_mask = (1 << num_cats) - 1
            for subset_mask in range(1, full_mask):
                subset = [
                    cat for k, cat in enumerate(combo_tuple) if (subset_mask >> k) & 1
                ]
                suffix = "_".join(subset)
                for bin_id in bin_ids:
                    bin_strat_vars.add(f"{bin_id}_{suffix}")

        return bin_strat_vars

    def _validate_transition_rates(
        self, transition: Transition, valid_identifiers: set[str]
    ) -> None:
        """Validates the rate expressions for a single transition."""
        if transition.rate:
            self._validate_rate_expression(
                transition.rate, transition.id, "rate", valid_identifiers
            )

        if transition.stratified_rates:
            for sr in transition.stratified_rates:
                self._validate_rate_expression(
                    sr.rate, transition.id, "stratified_rate", valid_identifiers
                )

    def _validate_rate_expression(
        self, rate: str, transition_id: str, context: str, valid_identifiers: set[str]
    ) -> None:
        """Validates variables in a single rate expression."""
        variables = get_expression_variables(rate)
        undefined_vars = [var for var in variables if var not in valid_identifiers]
        if undefined_vars:
            param_ids = {param.id for param in self.parameters}
            bin_ids = {bin_item.id for bin_item in self.population.bins}
            raise ValueError(
                (
                    f"Undefined variables in transition '{transition_id}' "
                    f"{context} '{rate}': {', '.join(undefined_vars)}. "
                    f"Available parameters: "
                    f"{', '.join(sorted(param_ids)) if param_ids else 'none'}. "
                    f"Available bins: "
                    f"{', '.join(sorted(bin_ids)) if bin_ids else 'none'}."
                )
            )

    @model_validator(mode="after")
    def validate_transition_ids(self) -> Self:
        """
        Validates that transition ids (source/target) are consistent in type
        and match the defined Bin IDs or Stratification Categories
        in the Population instance.
        """

        bin_ids = {bin_item.id for bin_item in self.population.bins}
        categories_ids = {
            cat for strat in self.population.stratifications for cat in strat.categories
        }
        bin_and_categories_ids = bin_ids.union(categories_ids)

        for transition in self.dynamics.transitions:
            source = set(transition.source)
            target = set(transition.target)
            transition_ids = source.union(target)

            if not transition_ids.issubset(bin_and_categories_ids):
                invalid_ids = transition_ids - bin_and_categories_ids
                raise ValueError(
                    (
                        f"Transition '{transition.id}' contains invalid ids: "
                        f"{invalid_ids}. Ids must be defined in Bin ids "
                        f"or Stratification Categories."
                    )
                )

            is_bin_flow = transition_ids.issubset(bin_ids)
            is_stratification_flow = transition_ids.issubset(categories_ids)

            if (not is_bin_flow) and (not is_stratification_flow):
                bin_elements = transition_ids.intersection(bin_ids)
                categories_elements = transition_ids.intersection(categories_ids)
                raise ValueError(
                    (
                        f"Transition '{transition.id}' mixes id types. "
                        f"Found Bin ids ({bin_elements}) and "
                        f"Stratification Categories ids ({categories_elements}). "
                        "Transitions must be purely Bin flow or purely "
                        f"Stratification flow."
                    )
                )

            if is_stratification_flow:
                category_to_stratification_map = {
                    cat: strat.id
                    for strat in self.population.stratifications
                    for cat in strat.categories
                }
                parent_stratification_ids = {
                    category_to_stratification_map[cat_id] for cat_id in transition_ids
                }
                if len(parent_stratification_ids) > 1:
                    mixed_strats = ", ".join(parent_stratification_ids)
                    raise ValueError(
                        (
                            f"Transition '{transition.id}' is a Stratification flow "
                            f"but involves categories from multiple stratifications: "
                            f"{mixed_strats}. A single transition must only move "
                            f"between categories belonging to the same parent "
                            f"stratification."
                        )
                    )

        return self

    # ------------------------------------------------------------------
    # Delegation to extracted modules
    # ------------------------------------------------------------------

    def print_equations(
        self,
        output_file: str | None = None,
        format: str = PrintEquationsOutputFormat.TEXT,
    ) -> None:
        """
        Prints the equations of the model in mathematical form.

        Displays model metadata and the system of equations in both
        compact (mathematical notation) and expanded (individual equations) forms.

        For DifferentialEquations models, displays equations as dX/dt = ...
        For DifferenceEquations models,
        displays equations as [X(t+Dt) - X(t)] / Dt = ...

        Parameters
        ----------
        output_file : str | None
            If provided, writes the equations to this file path instead of printing
            to console. If None, prints to console.
        format : str, default="text"
            Output format for equations. Must be one of:
            - "text": Plain text format (default)
            - "latex": LaTeX mathematical notation format

        Raises
        ------
        ValueError
            If format is not "text" or "latex"

        Examples
        --------
        >>> model.print_equations()  # Print to console in text format
        >>> model.print_equations(output_file="equations.txt")  # Save text format
        >>> model.print_equations(format="latex")  # Print LaTeX to console
        >>> model.print_equations(
        ...     output_file="equations.txt", format="latex"
        ... )  # Save LaTeX
        """
        from commol.context._model.equations import EquationPrinter

        EquationPrinter(self).print_equations(output_file, format)

    def check_unit_consistency(self, verbose: bool = False) -> None:
        """
        Check unit consistency of all equations in the model.

        This method validates that all transition rates have consistent units.
        It only performs the check if ALL parameters have units specified.
        If any parameter lacks a unit, the check is skipped.

        For difference equation models, all rates should have units that result in
        population change rates (e.g., "person/day" or "1/day" when multiplied by
        population).

        Parameters
        ----------
        verbose : bool, default=False
            If True, prints a success message when all units are consistent.

        Raises
        ------
        UnitConsistencyError
            If unit inconsistencies are found in any equation.
        ValueError
            If the model type doesn't support unit checking.

        Notes
        -----
        - Bin variables are assumed to have units of "person"
        - Predefined variables (N, N_young, etc.) have units of "person"
        - Time step variables (t, step) are dimensionless
        - Mathematical constants (pi, e) are dimensionless
        """
        from commol.context._model.units import UnitChecker

        UnitChecker(self).check_unit_consistency(verbose)
