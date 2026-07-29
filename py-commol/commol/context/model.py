from collections.abc import Mapping
from itertools import combinations, product
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from commol.constants import PrintEquationsOutputFormat
from commol.context._model.serialization import render_json
from commol.context.calibration import CalibrationProblem, CalibrationResult
from commol.context.constants import CalibrationParameterType
from commol.context.dynamics import Dynamics, Transition
from commol.context.parameter import Parameter
from commol.context.population import Population
from commol.context.probabilistic_calibration import ProbabilisticCalibrationResult
from commol.utils.security import get_expression_variables


def _calibrated_values(
    result: CalibrationResult | ProbabilisticCalibrationResult | Mapping[str, float],
) -> Mapping[str, float]:
    """
    Return the parameter values carried by a calibration outcome.

    For a probabilistic result these are the point parameters of the selected
    ensemble.
    """
    if isinstance(result, CalibrationResult):
        return result.best_parameters
    if isinstance(result, ProbabilisticCalibrationResult):
        return result.selected_ensemble.point_parameters
    return result


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
        ``initial_condition`` values update bin fractions or stratification
        category fractions, and ``scale`` values are ignored because they apply
        to observations rather than to the model.

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
            id refers to an expanded compartment, which the model definition
            cannot represent.
        """
        values = _calibrated_values(result)
        types = {param.id: param.parameter_type for param in problem.parameters}
        unknown = sorted(set(values) - set(types))
        if unknown:
            raise ValueError(
                f"Cannot apply values with no declared parameter type: {unknown}. "
                f"Declared calibration parameters: {sorted(types)}"
            )

        bin_ids = {bin_item.id for bin_item in self.population.bins}
        categories = self.population.initial_conditions.get_categories_with_fractions()

        parameters: dict[str, float | None] = {}
        bin_fractions: dict[str, float | None] = {}
        stratification_fractions: dict[str, float] = {}

        for param_id, value in values.items():
            parameter_type = types[param_id]
            if parameter_type == CalibrationParameterType.PARAMETER:
                parameters[param_id] = value
            elif parameter_type == CalibrationParameterType.SCALE:
                continue
            elif param_id in bin_ids:
                bin_fractions[param_id] = value
            elif param_id in categories:
                stratification_fractions[param_id] = value
            else:
                raise ValueError(
                    f"Initial condition '{param_id}' is neither a bin nor a "
                    f"stratification category. Per-compartment initial "
                    f"conditions can be calibrated but cannot be written back, "
                    f"because the model defines initial conditions as bin and "
                    f"stratification fractions only. Available bins: "
                    f"{sorted(bin_ids)}. Available categories: "
                    f"{sorted(categories)}"
                )

        if parameters:
            self.update_parameters(parameters)
        if bin_fractions:
            self.update_initial_conditions(bin_fractions)
        if stratification_fractions:
            self.update_stratification_fractions(stratification_fractions)

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
