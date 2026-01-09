import re
from collections.abc import Mapping
from itertools import combinations, product
from pathlib import Path
from typing import Self

import pint
from pydantic import BaseModel, Field, model_validator

from commol.constants import ModelTypes, PrintEquationsOutputFormat
from commol.context.dynamics import Dynamics, Transition
from commol.context.parameter import Parameter
from commol.context.population import Population
from commol.context.stratification import Stratification
from commol.utils.equations import (
    UnitConsistencyError,
    check_equation_units,
    get_predefined_variable_units,
    ureg,
)
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

        return (
            param_ids
            | bin_ids
            | strat_category_ids
            | special_vars
            | subpopulation_n_vars
        )

    def _get_subpopulation_n_vars(self) -> set[str]:
        """Generates all possible N_{category...} variable names."""
        if not self.population.stratifications:
            return set()

        subpopulation_n_vars: set[str] = set()
        category_groups = [s.categories for s in self.population.stratifications]

        # All possible combinations of categories across different stratifications
        full_category_combos = product(*category_groups)

        for combo_tuple in full_category_combos:
            # For each combo, find all non-empty subsets
            for i in range(1, len(combo_tuple) + 1):
                for subset in combinations(combo_tuple, i):
                    var_name = f"N_{'_'.join(subset)}"
                    subpopulation_n_vars.add(var_name)

        return subpopulation_n_vars

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
        # Validate format parameter
        if format not in PrintEquationsOutputFormat:
            raise ValueError(
                f"Invalid format: {format}. "
                f"Must be one of {list(PrintEquationsOutputFormat)}"
            )

        lines = self._generate_model_header()

        # Always generate both compact and expanded forms
        lines.extend(self._generate_compact_form(format=format))
        lines.append("")
        lines.extend(self._generate_expanded_form(format=format))

        output = "\n".join(lines)
        self._write_output(output, output_file)

    def _generate_model_header(self) -> list[str]:
        """Generate the header lines with model metadata."""
        lines: list[str] = []
        lines.append("=" * 40)
        lines.append("MODEL INFORMATION")
        lines.append("=" * 40)
        lines.append(f"Model: {self.name}")
        lines.append(f"Model Type: {self.dynamics.typology}")
        lines.append(f"Number of Bins: {len(self.population.bins)}")
        lines.append(
            f"Number of Stratifications: {len(self.population.stratifications)}"
        )
        lines.append(f"Number of Parameters: {len(self.parameters)}")
        lines.append(f"Number of Transitions: {len(self.dynamics.transitions)}")

        # List bins
        bin_ids = [bin_item.id for bin_item in self.population.bins]
        lines.append(f"Bins: {', '.join(bin_ids)}")

        # List stratifications
        if self.population.stratifications:
            lines.append("Stratifications:")
            for strat in self.population.stratifications:
                categories = ", ".join(strat.categories)
                lines.append(f"  - {strat.id}: [{categories}]")

        lines.append("")
        return lines

    def _collect_bin_and_category_ids(self) -> set[str]:
        """Collect all IDs from bins and stratification categories."""
        all_ids = {bin_item.id for bin_item in self.population.bins}
        for strat in self.population.stratifications:
            all_ids.update(strat.categories)
        return all_ids

    def _build_flow_equations(
        self, bin_and_category_ids: set[str]
    ) -> dict[str, dict[str, list[str]]]:
        """Build a mapping of bins and categories to their inflows and outflows."""
        equations: dict[str, dict[str, list[str]]] = {
            id_: {"inflows": [], "outflows": []} for id_ in bin_and_category_ids
        }
        for transition in self.dynamics.transitions:
            rate = transition.rate if transition.rate else ""
            source_counts = {
                state: transition.source.count(state)
                for state in set(transition.source)
            }
            target_counts = {
                state: transition.target.count(state)
                for state in set(transition.target)
            }
            all_states = set(transition.source) | set(transition.target)
            for state in all_states:
                net_change = target_counts.get(state, 0) - source_counts.get(state, 0)
                if net_change > 0:
                    equations[state]["inflows"].append(rate)
                elif net_change < 0:
                    equations[state]["outflows"].append(rate)

        return equations

    def _format_bin_equation(self, flows: dict[str, list[str]], format: str) -> str:
        """
        Format the equation for a single bin or category from its flows.

        Parameters
        ----------
        flows : dict[str, list[str]]
            Dictionary with 'inflows' and 'outflows' lists
        format : str
            Output format: "text" or "latex"
        """
        terms: list[str] = []

        for inflow in flows["inflows"]:
            if inflow:  # Only add if not empty
                # Format rate for LaTeX if needed
                if format == PrintEquationsOutputFormat.LATEX:
                    formatted_inflow = self._latex_rate_expression(inflow)
                else:
                    formatted_inflow = inflow
                terms.append(f"+ ({formatted_inflow})")

        for outflow in flows["outflows"]:
            if outflow:  # Only add if not empty
                # Format rate for LaTeX if needed
                if format == PrintEquationsOutputFormat.LATEX:
                    formatted_outflow = self._latex_rate_expression(outflow)
                else:
                    formatted_outflow = outflow
                terms.append(f"- ({formatted_outflow})")

        if not terms:
            return "0"

        result = " ".join(terms)
        # Remove leading + sign and space if present
        if result.startswith("+ "):
            result = result[2:]
        return result

    def _generate_compact_form(self, format: str) -> list[str]:
        """
        Generate compact mathematical notation form for stratified models.

        Parameters
        ----------
        format : str
            Output format: "text" or "latex"
        """
        lines: list[str] = []
        lines.append("=" * 40)
        lines.append("COMPACT FORM")
        lines.append("=" * 40)
        lines.append("")

        bin_ids = [bin_item.id for bin_item in self.population.bins]
        bin_transitions, stratification_transitions = (
            self._separate_transitions_by_type()
        )

        compartments = self._generate_compartments()

        lines.extend(
            self._format_bin_transitions_compact_stratified(
                bin_transitions, compartments, format
            )
        )
        lines.extend(
            self._format_stratification_transitions_compact_stratified(
                stratification_transitions, bin_ids, format
            )
        )
        lines.extend(self._format_total_system_size(bin_ids))

        return lines

    def _generate_compartments(self) -> list[tuple[str, ...]]:
        """
        Generate all compartment combinations from bins and stratifications.
        """
        bin_ids = [state.id for state in self.population.bins]

        if not self.population.stratifications:
            return [(state,) for state in bin_ids]

        strat_categories = [
            strat.categories for strat in self.population.stratifications
        ]

        compartments: list[tuple[str, ...]] = []
        for bin_id in bin_ids:
            for strat_combo in product(*strat_categories):
                compartments.append((bin_id,) + strat_combo)

        return compartments

    def _compartment_to_string(self, compartment: tuple[str, ...], format: str) -> str:
        """
        Convert compartment tuple to string.

        Parameters
        ----------
        compartment : tuple[str, ...]
            Compartment as tuple (e.g., ('S', 'young', 'urban'))
        format : str, default="text"
            Output format: "text" or "latex"

        Returns
        -------
        str
            Formatted compartment string

        Examples
        --------
        Text: ('S', 'young', 'urban') -> 'S_young_urban'
        LaTeX: ('S', 'young', 'urban') -> 'S_{young,urban}'
        """
        if format == PrintEquationsOutputFormat.LATEX:
            # Use _latex_variable to format with subscripts
            compartment_str = "_".join(compartment)
            return self._latex_variable(compartment_str)
        else:
            # Text format (original)
            return "_".join(compartment)

    def _get_rate_for_compartment(
        self, transition: Transition, compartment: tuple[str, ...]
    ) -> str | None:
        """Get the appropriate rate for a compartment, considering stratified rates."""
        if not transition.stratified_rates or len(compartment) == 1:
            return transition.rate

        compartment_strat_map: dict[str, str] = {}
        for i, strat in enumerate(self.population.stratifications):
            compartment_strat_map[strat.id] = compartment[i + 1]

        for strat_rate in transition.stratified_rates:
            matches = True
            for condition in strat_rate.conditions:
                if (
                    compartment_strat_map.get(condition.stratification)
                    != condition.category
                ):
                    matches = False
                    break
            if matches:
                return strat_rate.rate

        # No stratified rate matched, use fallback
        return transition.rate

    def _separate_transitions_by_type(
        self,
    ) -> tuple[list[Transition], list[Transition]]:
        """Separate transitions into bin and stratification types."""
        bin_ids = [bin_item.id for bin_item in self.population.bins]
        bin_id_set = set(bin_ids)

        bin_transitions: list[Transition] = []
        stratification_transitions: list[Transition] = []

        for transition in self.dynamics.transitions:
            transition_ids = set(transition.source) | set(transition.target)
            if transition_ids.issubset(bin_id_set):
                bin_transitions.append(transition)
            else:
                stratification_transitions.append(transition)

        return bin_transitions, stratification_transitions

    def _group_transitions_by_stratification(
        self, transitions: list[Transition]
    ) -> dict[str, list[Transition]]:
        """Group stratification transitions by their stratification ID."""
        strat_by_id: dict[str, list[Transition]] = {}
        for strat in self.population.stratifications:
            strat_by_id[strat.id] = []
            for transition in transitions:
                transition_states = set(transition.source) | set(transition.target)
                if transition_states.issubset(set(strat.categories)):
                    strat_by_id[strat.id].append(transition)
        return strat_by_id

    def _format_total_system_size(self, bin_ids: list[str]) -> list[str]:
        """Format the total system size information."""
        lines: list[str] = []

        num_disease_states = len(bin_ids)
        if not self.population.stratifications:
            total_equations = num_disease_states
            lines.append(
                (
                    f"Total System: {total_equations} coupled equations "
                    f"({num_disease_states} bins)"
                )
            )
            return lines

        num_strat_combinations = 1
        strat_details: list[str] = []
        for strat in self.population.stratifications:
            num_cat = len(strat.categories)
            num_strat_combinations *= num_cat
            strat_details.append(f"{num_cat} {strat.id}")

        total_equations = num_disease_states * num_strat_combinations

        lines.append(
            (
                f"Total System: {total_equations} coupled equations "
                f"({num_disease_states} bins × {' × '.join(strat_details)})"
            )
        )

        return lines

    def _format_bin_transitions_compact_stratified(
        self,
        bin_transitions: list[Transition],
        compartments: list[tuple[str, ...]],
        format: str,
    ) -> list[str]:
        """
        Format bin transitions showing specific compartments and rates.

        Parameters
        ----------
        bin_transitions : list[Transition]
            List of bin transitions
        compartments : list[tuple[str, ...]]
            List of compartments
        format : str
            Output format: "text" or "latex"
        """
        lines: list[str] = []

        if not bin_transitions:
            return lines

        lines.append("Bin Transitions:")

        # Check if we have complete units for annotation
        show_units = self._has_all_units()
        variable_units = self._build_variable_units() if show_units else None

        for transition in bin_transitions:
            source_str = (
                ", ".join(sorted(set(transition.source)))
                if transition.source
                else "none"
            )
            target_str = (
                ", ".join(sorted(set(transition.target)))
                if transition.target
                else "none"
            )
            lines.append(
                f"{transition.id.capitalize()} ({source_str} -> {target_str}):"
            )

            # Handle influx vs normal transitions
            if not transition.source and transition.target:
                lines.extend(
                    self._format_influx_transition_lines(
                        transition, compartments, variable_units, show_units, format
                    )
                )
            else:
                lines.extend(
                    self._format_normal_transition_lines(
                        transition, compartments, variable_units, show_units, format
                    )
                )

            lines.append("")

        return lines

    def _format_influx_transition_lines(
        self,
        transition: Transition,
        compartments: list[tuple[str, ...]],
        variable_units: dict[str, str] | None,
        show_units: bool,
        format: str,
    ) -> list[str]:
        """Format lines for influx transitions (empty source)."""
        lines: list[str] = []
        for compartment in compartments:
            bin_id = compartment[0]
            if bin_id in transition.target:
                target_compartment_str = self._compartment_to_string(
                    compartment, format
                )
                rate = self._get_rate_for_compartment(transition, compartment)
                rate_with_unit = self._format_rate_with_unit(
                    rate, variable_units, show_units, format
                )

                if format == PrintEquationsOutputFormat.LATEX:
                    arrow = self._latex_transition_arrow("none", target_compartment_str)
                    lines.append(f"  ${arrow}: {rate_with_unit}$")
                else:
                    lines.append(
                        f"  none -> {target_compartment_str}: {rate_with_unit}"
                    )
        return lines

    def _format_normal_transition_lines(
        self,
        transition: Transition,
        compartments: list[tuple[str, ...]],
        variable_units: dict[str, str] | None,
        show_units: bool,
        format: str,
    ) -> list[str]:
        """Format lines for normal transitions (source to target)."""
        lines: list[str] = []
        for compartment in compartments:
            bin_id = compartment[0]
            if bin_id in transition.source:
                source_compartment_str = self._compartment_to_string(
                    compartment, format
                )
                target_compartment_str = self._get_target_compartment_str(
                    compartment, bin_id, transition.target, format
                )

                rate = self._get_rate_for_compartment(transition, compartment)
                rate_with_unit = self._format_rate_with_unit(
                    rate, variable_units, show_units, format
                )

                if format == PrintEquationsOutputFormat.LATEX:
                    arrow = self._latex_transition_arrow(
                        source_compartment_str, target_compartment_str
                    )
                    lines.append(f"  ${arrow}: {rate_with_unit}$")
                else:
                    lines.append(
                        f"  {source_compartment_str} -> "
                        f"{target_compartment_str}: {rate_with_unit}"
                    )
        return lines

    def _get_target_compartment_str(
        self,
        compartment: tuple[str, ...],
        bin_id: str,
        target_bins: list[str],
        format: str,
    ) -> str:
        """Get the target compartment string for a transition."""
        if not target_bins:
            return "none"

        target_bin = target_bins[0]
        if format == PrintEquationsOutputFormat.LATEX:
            target_compartment = list(compartment)
            target_compartment[0] = target_bin
            return self._compartment_to_string(tuple(target_compartment), format)
        else:
            source_compartment_str = self._compartment_to_string(compartment, format)
            return source_compartment_str.replace(bin_id, target_bin, 1)

    def _build_stratified_for_each_line(
        self, bin_ids: list[str], other_strats: list["Stratification"]
    ) -> str:
        if other_strats:
            other_strats_strs = [
                f"each {s.id} in {{{', '.join(s.categories)}}}" for s in other_strats
            ]
            return (
                f"For each bin X in {{{', '.join(bin_ids)}}} "
                f"and {', '.join(other_strats_strs)}:"
            )
        return f"For each bin X in {{{', '.join(bin_ids)}}}:"

    def _build_stratified_transition_line(
        self,
        trans: Transition,
        strat_idx: int,
        combo: tuple[str, ...],
        variable_units: dict[str, str] | None = None,
        show_units: bool = True,
        format: str = PrintEquationsOutputFormat.TEXT,
    ) -> str:
        """
        Build a transition line for stratification transitions.

        Parameters
        ----------
        trans : Transition
            The transition to format.
        strat_idx : int
            Index of the stratification being transitioned.
        combo : tuple[str, ...]
            Combination of categories for other stratifications.
        variable_units : dict[str, str] | None
            Pre-computed variable units for efficiency.
        show_units : bool, default=True
            If True, annotate variables and show final unit.
            If False, return the plain rate expression.

        Returns
        -------
        str
            Formatted transition line with unit annotation.
        """
        src_cat = trans.source[0]
        tgt_cat = trans.target[0]

        source_parts = [""] * len(self.population.stratifications)
        target_parts = [""] * len(self.population.stratifications)
        source_parts[strat_idx] = src_cat
        target_parts[strat_idx] = tgt_cat

        combo_idx = 0
        for i in range(len(self.population.stratifications)):
            if i != strat_idx:
                source_parts[i] = combo[combo_idx]
                target_parts[i] = combo[combo_idx]
                combo_idx += 1

        # Format compartment strings based on output format
        if format == PrintEquationsOutputFormat.LATEX:
            source_comp_parts = "_".join(["X"] + source_parts)
            target_comp_parts = "_".join(["X"] + target_parts)
            source_comp = self._latex_variable(source_comp_parts)
            target_comp = self._latex_variable(target_comp_parts)
        else:
            source_comp = f"X_{'_'.join(source_parts)}"
            target_comp = f"X_{'_'.join(target_parts)}"

        sample_compartment = ("X",) + tuple(source_parts)
        rate = self._get_rate_for_compartment(trans, sample_compartment)

        # Format the rate expression
        if format == PrintEquationsOutputFormat.LATEX:
            # LaTeX format
            effective_rate = rate if rate else trans.rate
            if effective_rate:
                rate_expr = (
                    f"{self._latex_rate_expression(effective_rate)} "
                    f"\\cdot {source_comp}"
                )
            else:
                rate_expr = f"\\text{{None}} \\cdot {source_comp}"
            arrow = self._latex_transition_arrow(source_comp, target_comp)
            return f"  ${arrow}: {rate_expr}$"
        elif show_units and rate and self.population.bins:
            # Text format with units
            first_bin_id = self.population.bins[0].id
            concrete_compartment = (first_bin_id,) + tuple(source_parts)
            concrete_comp_str = self._compartment_to_string(
                concrete_compartment, format
            )
            full_rate_expr = f"{rate} * {concrete_comp_str}"

            # Annotate the concrete expression with units
            annotated_expr = self._annotate_rate_variables(
                full_rate_expr, variable_units
            )

            # Replace the concrete compartment back with X (generic bin placeholder)
            bin_unit = self.population.bins[0].unit
            annotated_expr = annotated_expr.replace(
                f"{concrete_comp_str}({bin_unit})", f"{source_comp}({bin_unit})"
            )

            # Get final unit
            unit = self._get_rate_unit(full_rate_expr, variable_units)
            unit_str = f" [{unit}]" if unit else ""

            return f"  {source_comp} -> {target_comp}: {annotated_expr}{unit_str}"
        else:
            # Text format without annotations
            rate_expr = (
                f"{rate} * {source_comp}" if rate else f"{trans.rate} * {source_comp}"
            )
            return f"  {source_comp} -> {target_comp}: {rate_expr}"

    def _format_stratification_transitions_compact_stratified(
        self,
        stratification_transitions: list[Transition],
        bin_ids: list[str],
        format: str,
    ) -> list[str]:
        """
        Format stratification transitions showing movements between categories.

        Parameters
        ----------
        stratification_transitions : list[Transition]
            List of stratification transitions
        bin_ids : list[str]
            List of bin IDs
        format : str
            Output format: "text" or "latex"
        """
        lines: list[str] = []
        strat_by_id = self._group_transitions_by_stratification(
            stratification_transitions
        )

        # Check if we have complete units for annotation
        show_units = self._has_all_units()

        # Cache variable units for efficiency when formatting multiple rates
        variable_units = self._build_variable_units() if show_units else None

        for strat_idx, strat in enumerate(self.population.stratifications):
            if not strat_by_id.get(strat.id):
                continue

            transition = strat_by_id[strat.id][0]
            source_cat = transition.source[0] if transition.source else "none"
            target_cat = transition.target[0] if transition.target else "none"

            if not source_cat or not target_cat:
                continue

            lines.append(
                (
                    f"{strat.id.capitalize()} Stratification Transitions "
                    f"({source_cat} -> {target_cat}):"
                )
            )

            other_strats = [
                s
                for i, s in enumerate(self.population.stratifications)
                if i != strat_idx
            ]

            lines.append(self._build_stratified_for_each_line(bin_ids, other_strats))

            for trans in strat_by_id[strat.id]:
                other_cat_combos = (
                    list(product(*[s.categories for s in other_strats]))
                    if other_strats
                    else [()]
                )

                for combo in other_cat_combos:
                    lines.append(
                        self._build_stratified_transition_line(
                            trans, strat_idx, combo, variable_units, show_units, format
                        )
                    )

            lines.append("")

        return lines

    def _get_equation_lhs(self, variable_name: str, format: str) -> str:
        """
        Get the left-hand side of an equation based on model type.

        Parameters
        ----------
        variable_name : str
            The name of the variable (already LaTeX-formatted if format is "latex")
        format : str, default="text"
            Output format: "text" or "latex"

        Returns
        -------
        str
            The formatted LHS
        """
        if format == PrintEquationsOutputFormat.LATEX:
            # variable_name is already LaTeX-formatted (e.g., S_{young})
            # Don't call _latex_variable again
            if self.dynamics.typology == ModelTypes.DIFFERENTIAL_EQUATIONS:
                return f"\\frac{{d{variable_name}}}{{dt}}"
            else:  # DIFFERENCE_EQUATIONS:
                return (
                    f"\\frac{{{variable_name}(t+\\Delta t) - "
                    f"{variable_name}(t)}}{{\\Delta t}}"
                )
        else:
            # Text format (original)
            if self.dynamics.typology == ModelTypes.DIFFERENTIAL_EQUATIONS:
                return f"d{variable_name}/dt"
            else:  # DIFFERENCE_EQUATIONS
                return f"[{variable_name}(t+Dt) - {variable_name}(t)] / Dt"

    def _generate_expanded_form(self, format: str) -> list[str]:
        """
        Generate expanded form with individual equations for each compartment.

        Parameters
        ----------
        format : str
            Output format: "text" or "latex"
        """
        lines: list[str] = []

        lines.append("=" * 40)
        lines.append("EXPANDED FORM")
        lines.append("=" * 40)

        has_stratifications = len(self.population.stratifications) > 0

        if has_stratifications:
            compartments = self._generate_compartments()
            bin_transitions, stratification_transitions = (
                self._separate_transitions_by_type()
            )

            for compartment in compartments:
                compartment_str = self._compartment_to_string(compartment, format)
                equation = self._build_compartment_equation(
                    compartment, bin_transitions, stratification_transitions, format
                )
                lhs = self._get_equation_lhs(compartment_str, format)
                if format == PrintEquationsOutputFormat.LATEX:
                    lines.append(f"\\[{lhs} = {equation}\\]")
                else:
                    lines.append(f"{lhs} = {equation}")
        else:
            bin_and_category_ids = self._collect_bin_and_category_ids()
            equations = self._build_flow_equations(bin_and_category_ids)
            bin_ids = [bin_item.id for bin_item in self.population.bins]

            for bin_id in bin_ids:
                equation = self._format_bin_equation(equations[bin_id], format)
                lhs = self._get_equation_lhs(bin_id, format)
                if format == PrintEquationsOutputFormat.LATEX:
                    lines.append(f"\\[{lhs} = {equation}\\]")
                else:
                    lines.append(f"{lhs} = {equation}")

        return lines

    def _build_compartment_equation(
        self,
        compartment: tuple[str, ...],
        bin_transitions: list[Transition],
        stratification_transitions: list[Transition],
        format: str,
    ) -> str:
        """
        Build the complete equation for a specific compartment.

        Parameters
        ----------
        compartment : tuple[str, ...]
            The compartment tuple
        bin_transitions : list[Transition]
            List of bin transitions
        stratification_transitions : list[Transition]
            List of stratification transitions
        format : str
            Output format: "text" or "latex"
        """
        terms: list[str] = []

        # Add bin transition terms
        terms.extend(
            self._get_bin_transition_terms(compartment, bin_transitions, format)
        )

        # Add stratification transition terms
        for transition in stratification_transitions:
            flow_term = self._get_stratification_flow_for_compartment(
                compartment, transition, format
            )
            if flow_term:
                terms.append(flow_term)

        if not terms:
            return "0"

        equation = " ".join(terms)
        if equation.startswith("+ "):
            return equation[2:]
        if equation.startswith("+"):
            return equation[1:]
        return equation

    def _get_bin_transition_terms(
        self,
        compartment: tuple[str, ...],
        bin_transitions: list[Transition],
        format: str,
    ) -> list[str]:
        """Extract terms from bin transitions for a compartment."""
        terms: list[str] = []
        bin_id = compartment[0]

        for transition in bin_transitions:
            source_count = transition.source.count(bin_id)
            target_count = transition.target.count(bin_id)
            net_change = target_count - source_count

            if net_change != 0:
                rate = self._get_rate_for_compartment(transition, compartment)
                if rate:
                    # Format rate for LaTeX if needed
                    if format == PrintEquationsOutputFormat.LATEX:
                        formatted_rate = self._latex_rate_expression(rate)
                    else:
                        formatted_rate = rate

                    if net_change > 0:
                        terms.append(f"+ ({formatted_rate})")
                    else:
                        terms.append(f"- ({formatted_rate})")

        return terms

    def _get_stratification_flow_for_compartment(
        self, compartment: tuple[str, ...], transition: Transition, format: str
    ) -> str | None:
        """
        Calculate stratification flow term for a compartment.

        Parameters
        ----------
        compartment : tuple[str, ...]
            The compartment tuple
        transition : Transition
            The transition
        format : str
            Output format: "text" or "latex"
        """
        if len(compartment) == 1:
            return None

        transition_states = set(transition.source) | set(transition.target)
        target_strat_idx = None

        for i, strat in enumerate(self.population.stratifications):
            if transition_states.issubset(set(strat.categories)):
                target_strat_idx = i
                break

        if target_strat_idx is None:
            return None

        compartment_category = compartment[target_strat_idx + 1]
        source_categories = transition.source
        target_categories = transition.target

        source_count = source_categories.count(compartment_category)
        target_count = target_categories.count(compartment_category)
        net_change = target_count - source_count

        if net_change == 0:
            return None

        rate = self._get_rate_for_compartment(transition, compartment)
        if not rate:
            return None

        # Format rate and compartment strings
        if format == PrintEquationsOutputFormat.LATEX:
            formatted_rate = self._latex_rate_expression(rate)
            mult_op = " \\cdot "
        else:
            formatted_rate = rate
            mult_op = " * "

        if net_change < 0:
            compartment_str = self._compartment_to_string(compartment, format)
            return f"- ({formatted_rate}{mult_op}{compartment_str})"
        else:
            source_category = source_categories[0] if source_categories else None
            if source_category:
                source_compartment = list(compartment)
                source_compartment[target_strat_idx + 1] = source_category
                source_compartment_str = self._compartment_to_string(
                    tuple(source_compartment), format
                )
                return f"+ ({formatted_rate}{mult_op}{source_compartment_str})"

        return None

    def _write_output(self, output: str, output_file: str | None) -> None:
        """Write output to file or console."""
        if output_file:
            with open(output_file, "w") as f:
                _ = f.write(output)
        else:
            print(output)

    def _latex_variable(self, var_name: str) -> str:
        """
        Format variable name for LaTeX math mode.

        Converts variable names to LaTeX format with proper subscripts.
        Variables with underscores get subscripts with comma-separated values.
        All output is in math mode (not text mode).

        Parameters
        ----------
        var_name : str
            Variable name to format

        Returns
        -------
        str
            LaTeX-formatted variable name

        Examples
        --------
        - S -> S
        - beta -> \\beta (if common Greek letter name)
        - S_young_urban -> S_{young,urban}
        - N_young -> N_{young}
        """
        # Greek letters commonly used in epidemiology
        greek_letters = {
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
            "nu",
            "xi",
            "omicron",
            "pi",
            "rho",
            "sigma",
            "tau",
            "upsilon",
            "phi",
            "chi",
            "psi",
            "omega",
        }

        # Check if the variable is a Greek letter
        if var_name in greek_letters:
            return f"\\{var_name}"

        # Handle variables with underscores (subscripts)
        if "_" in var_name:
            parts = var_name.split("_")
            base = parts[0]
            subscripts = parts[1:]

            # Format base (check if it's a Greek letter)
            if base in greek_letters:
                base = f"\\{base}"

            # Join subscripts with commas
            subscript_str = ",".join(subscripts)
            return f"{base}_{{{subscript_str}}}"

        return var_name

    def _latex_transition_arrow(self, source: str, target: str) -> str:
        """
        Format transition arrow for LaTeX.

        Parameters
        ----------
        source : str
            Source compartment (or "none" for influx) - should already be
            LaTeX-formatted
        target : str
            Target compartment (or "none" for outflux) - should already be
            LaTeX-formatted

        Returns
        -------
        str
            LaTeX-formatted transition arrow

        Examples
        --------
        - none -> S_{young}: \\varnothing \\to S_{young}
        - S_{young} -> I_{young}: S_{young} \\to I_{young}
        - S -> none: S \\to \\varnothing
        """
        formatted_source = "\\varnothing" if source == "none" else source
        formatted_target = "\\varnothing" if target == "none" else target
        return f"{formatted_source} \\to {formatted_target}"

    def _latex_rate_expression(self, rate: str) -> str:
        """
        Convert rate expression to LaTeX math mode.

        Transforms rate expressions by:
        - Formatting all variables as math variables (bins, parameters)
        - Converting * to \\cdot
        - Converting / to \\frac{numerator}{denominator}
        - Wrapping units in \\text{...} when annotated

        Parameters
        ----------
        rate : str
            The rate expression to convert

        Returns
        -------
        str
            LaTeX-formatted rate expression

        Examples
        --------
        Input:  "beta * S * I / N"
        Output: "\\frac{\\beta \\cdot S \\cdot I}{N}"

        Input:  "beta(1/day) * S(person)"
        Output: "\\beta\\ (\\frac{1}{\\text{day}}) \\cdot S\\ (\\text{person})"
        """
        if not rate:
            return rate

        # First, replace * with \cdot
        latex_rate = rate.replace(" * ", " \\cdot ")

        # Get all variables in the expression
        variables = get_expression_variables(rate)

        # Sort variables by length (descending) to avoid partial replacements
        sorted_vars = sorted(variables, key=lambda x: len(x), reverse=True)

        # Replace each variable with its LaTeX formatted version
        for var in sorted_vars:
            latex_var = self._latex_variable(var)
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(var) + r"\b"
            # Escape backslashes in replacement string for re.sub
            latex_rate = re.sub(pattern, latex_var.replace("\\", "\\\\"), latex_rate)

        # Handle units: wrap them in \text{...}
        # Pattern: (unit) or [unit] where unit contains letters
        latex_rate = re.sub(r"\(([^)]*[a-zA-Z][^)]*)\)", r"(\\text{\1})", latex_rate)
        latex_rate = re.sub(r"\[([^\]]*[a-zA-Z][^\]]*)\]", r"[\\text{\1}]", latex_rate)

        # Convert division to fractions
        # This is complex because we need to handle operator precedence
        # For now, convert simple divisions like "a / b" to "\frac{a}{b}"
        # More complex cases with parentheses need careful handling
        latex_rate = self._convert_division_to_frac(latex_rate)

        return latex_rate

    def _convert_division_to_frac(self, expr: str) -> str:
        """
        Convert division operations to LaTeX fractions.

        Handles operator precedence and nested parentheses correctly.
        For example: "a \\cdot b / c" becomes "\\frac{a \\cdot b}{c}"
                    "(a + b) / (c \\cdot d)" becomes "\\frac{a + b}{c \\cdot d}"

        Parameters
        ----------
        expr : str
            Expression with division operators

        Returns
        -------
        str
            Expression with divisions converted to \\frac
        """
        if " / " not in expr:
            return expr

        # Find division operators not inside parentheses
        divisions = []
        paren_depth = 0
        i = 0

        while i < len(expr):
            if expr[i] == "(":
                paren_depth += 1
            elif expr[i] == ")":
                paren_depth -= 1
            elif paren_depth == 0 and i + 3 <= len(expr) and expr[i : i + 3] == " / ":
                divisions.append(i)
                i += 2  # Skip the " / " for next iteration
            i += 1

        if not divisions:
            # No divisions at top level, return as is
            return expr

        # Process divisions from left to right
        # Split expression at division points
        parts = []
        start = 0
        for div_pos in divisions:
            parts.append(expr[start:div_pos].strip())
            start = div_pos + 3  # Skip " / "
        parts.append(expr[start:].strip())

        # Build fractions from left to right
        # a / b / c becomes \frac{\frac{a}{b}}{c}
        result = self._strip_outer_parens(parts[0])
        for i in range(1, len(parts)):
            denominator = self._strip_outer_parens(parts[i])
            result = f"\\frac{{{result}}}{{{denominator}}}"

        return result

    def _strip_outer_parens(self, expr: str) -> str:
        """
        Remove outer parentheses if they wrap the entire expression.

        Parameters
        ----------
        expr : str
            Expression to process

        Returns
        -------
        str
            Expression with outer parentheses removed if applicable
        """
        expr = expr.strip()
        if not expr.startswith("(") or not expr.endswith(")"):
            return expr

        # Check if parentheses actually wrap the whole expression
        # Need to ensure they're matching outer parentheses
        depth = 0
        for i, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

            # If depth reaches 0 before the end, these aren't wrapping parentheses
            if depth == 0 and i < len(expr) - 1:
                return expr

        # If we get here, the parentheses wrap the whole expression
        return expr[1:-1]

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
        self._validate_unit_check_preconditions()

        # Build variable units mapping
        variable_units = self._build_variable_units()

        # Check each transition and collect errors
        errors = self._collect_unit_errors(variable_units)

        if errors:
            error_message = "Unit consistency check failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise UnitConsistencyError(error_message)

        if verbose:
            print("Unit consistency check passed successfully.")

    def _register_custom_units(self) -> None:
        """Register custom units in the pint registry."""
        # Collect all units that need to be registered
        units_to_register = set()

        # Add bin units
        for bin_item in self.population.bins:
            if bin_item.unit:
                units_to_register.add(bin_item.unit)

        # Add parameter units
        for param in self.parameters:
            if param.unit:
                # Extract individual unit names from compound units like "1/semester"
                # Remove operators and numbers to get unit names
                unit_str = param.unit
                # Split by common operators and filter out numbers/empty strings
                unit_parts = re.split(r"[*/\s\(\)]+", unit_str)
                for part in unit_parts:
                    part = part.strip()
                    if part and not part.replace(".", "").replace("-", "").isdigit():
                        units_to_register.add(part)

        # Known time unit aliases that should be defined relative to existing time units
        # rather than as new base dimensions
        known_time_aliases = {
            # Periods
            "decade": "10 * year",
            "century": "100 * year",
            "millennium": "1000 * year",
            "fortnight": "14 * day",
            "biweek": "14 * day",
            "semester": "6 * month",
            "trimester": "3 * month",
            "quarter": "3 * month",
            "bimester": "2 * month",
            # Common abbreviations
            "wk": "week",
            "mo": "month",
            "mon": "month",
            "yr": "year",
            "hr": "hour",
            "min": "minute",
            "sec": "second",
            # Plural/singular variants
            "secs": "second",
            "mins": "minute",
            "hrs": "hour",
            "wks": "week",
            "mons": "month",
            "yrs": "year",
        }

        # Register each unit if not already defined
        for unit_name in units_to_register:
            try:
                # Try to use the unit to see if it exists
                ureg(unit_name)
            except pint.UndefinedUnitError:
                # Check if it's a known time alias
                if unit_name in known_time_aliases:
                    ureg.define(f"{unit_name} = {known_time_aliases[unit_name]}")
                else:
                    # If it doesn't exist, define it as a new base unit
                    # Use the unit name as-is and create a unique dimension for it
                    dimension_name = f"{unit_name}_dimension"
                    ureg.define(f"{unit_name} = [{dimension_name}]")

    def _validate_unit_check_preconditions(self) -> None:
        """
        Validate preconditions for unit consistency checking.

        Raises
        ------
        UnitConsistencyError
            If constant parameters are missing units.
        ValueError
            If the model type doesn't support unit checking.
        """
        # Register any custom bin units before validation
        self._register_custom_units()
        # Check if all non-formula parameters have units
        non_formula_params_missing_units = [
            p
            for p in self.parameters
            if p.unit is None and not isinstance(p.value, str)
        ]

        if non_formula_params_missing_units:
            param_names = ", ".join([p.id for p in non_formula_params_missing_units])
            raise UnitConsistencyError(
                (
                    f"Cannot perform unit consistency check. The following constant "
                    f"parameters are missing units: {param_names}. "
                    f"Please specify units for all parameters, or use formulas "
                    f"to allow automatic unit inference."
                )
            )

        if self.dynamics.typology != ModelTypes.DIFFERENCE_EQUATIONS:
            raise ValueError(
                (
                    f"Unit checking is only supported for DifferenceEquations models. "
                    f"Current model type: {self.dynamics.typology}"
                )
            )

    def _collect_unit_errors(self, variable_units: dict[str, str]) -> list[str]:
        """
        Collect unit consistency errors from all transitions.

        Parameters
        ----------
        variable_units : dict[str, str]
            Mapping of variable names to their units.

        Returns
        -------
        list[str]
            List of error messages for inconsistent units.
        """
        errors: list[str] = []

        for transition in self.dynamics.transitions:
            # Check main rate
            if transition.rate:
                is_consistent, error_msg = self._check_transition_rate_units(
                    transition.rate,
                    transition.id,
                    variable_units,
                )
                if not is_consistent and error_msg:
                    errors.append(error_msg)

            # Check stratified rates
            if transition.stratified_rates:
                for idx, strat_rate in enumerate(transition.stratified_rates):
                    is_consistent, error_msg = self._check_transition_rate_units(
                        strat_rate.rate,
                        f"{transition.id} (stratified rate {idx + 1})",
                        variable_units,
                    )
                    if not is_consistent and error_msg:
                        errors.append(error_msg)

        return errors

    def _build_variable_units(self) -> dict[str, str]:
        """Build a mapping of all variables to their units."""
        variable_units: dict[str, str] = {}

        # Add base units for parameters, bins, and special variables
        self._add_base_variable_units(variable_units)

        # Infer units for formula parameters
        self._infer_formula_parameter_units(variable_units)

        return variable_units

    def _add_base_variable_units(self, variable_units: dict[str, str]) -> None:
        """Add units for parameters, bins, categories, and special variables."""
        self._add_parameter_units(variable_units)
        self._add_bin_units(variable_units)
        self._add_stratification_category_units(variable_units)
        self._add_compartment_units(variable_units)
        self._add_predefined_variable_units(variable_units)
        self._add_special_variable_units(variable_units)

    def _add_parameter_units(self, variable_units: dict[str, str]) -> None:
        """Add units for parameters with explicit units."""
        for param in self.parameters:
            if param.unit:
                variable_units[param.id] = param.unit

    def _add_bin_units(self, variable_units: dict[str, str]) -> None:
        """Add units for bins."""
        for state in self.population.bins:
            if state.unit:
                variable_units[state.id] = state.unit

    def _add_stratification_category_units(
        self, variable_units: dict[str, str]
    ) -> None:
        """Add units for stratification categories (inherit from bins)."""
        if not self.population.bins or not self.population.bins[0].unit:
            return

        bin_unit = self.population.bins[0].unit
        for strat in self.population.stratifications:
            for category in strat.categories:
                variable_units[category] = bin_unit

    def _add_compartment_units(self, variable_units: dict[str, str]) -> None:
        """Add units for compartment combinations (e.g., S_young, I_old)."""
        if not self.population.stratifications:
            return

        compartments = self._generate_compartments()
        for compartment in compartments:
            compartment_str = self._compartment_to_string(
                compartment, format=PrintEquationsOutputFormat.TEXT
            )
            bin_id = compartment[0]
            bin_obj = next((b for b in self.population.bins if b.id == bin_id), None)
            if bin_obj and bin_obj.unit:
                variable_units[compartment_str] = bin_obj.unit

    def _add_predefined_variable_units(self, variable_units: dict[str, str]) -> None:
        """Add units for predefined variables (N, N_young, etc.)."""
        bin_unit = self.population.bins[0].unit if self.population.bins else None
        predefined_units = get_predefined_variable_units(
            self.population.stratifications, bin_unit
        )
        variable_units.update(predefined_units)

    def _add_special_variable_units(self, variable_units: dict[str, str]) -> None:
        """Add dimensionless special variables (t, step, pi, e)."""
        variable_units["step"] = "dimensionless"
        variable_units["t"] = "dimensionless"
        variable_units["pi"] = "dimensionless"
        variable_units["e"] = "dimensionless"

    def _infer_formula_parameter_units(self, variable_units: dict[str, str]) -> None:
        """Infer units for formula parameters through iterative resolution."""
        max_iterations = 10
        formula_params_without_units: list[Parameter] = []

        for _ in range(max_iterations):
            inferred_any = self._try_infer_formula_units(
                variable_units, formula_params_without_units
            )
            if not inferred_any:
                break

        # Validate that all formula parameters have units
        self._validate_formula_parameter_units(
            variable_units, formula_params_without_units
        )

    def _try_infer_formula_units(
        self,
        variable_units: dict[str, str],
        failed_params: list[Parameter],
    ) -> bool:
        """
        Try to infer units for one iteration. Returns True if any units were inferred.
        """
        inferred_any = False

        for param in self.parameters:
            if param.id in variable_units or not isinstance(param.value, str):
                continue

            try:
                if self._infer_single_formula_unit(param, variable_units):
                    inferred_any = True
            except Exception:
                if param not in failed_params:
                    failed_params.append(param)

        return inferred_any

    def _infer_single_formula_unit(
        self, param: Parameter, variable_units: dict[str, str]
    ) -> bool:
        """Infer unit for a single formula parameter. Returns True if successful."""
        if not isinstance(param.value, str):
            return False

        formula_vars = get_expression_variables(param.value)

        # Check if all variables have units
        formula_var_units: dict[str, str] = {}
        for var in formula_vars:
            if var in variable_units:
                formula_var_units[var] = variable_units[var]
            else:
                return False  # Not all variables have units yet

        # Infer the unit
        if formula_var_units:
            from commol.utils.equations import parse_equation_unit

            inferred_unit = parse_equation_unit(param.value, formula_var_units)
            variable_units[param.id] = str(inferred_unit.units)
        else:
            # Formula has no variables (e.g., "2 * 3")
            variable_units[param.id] = "dimensionless"

        return True

    def _validate_formula_parameter_units(
        self,
        variable_units: dict[str, str],
        failed_params: list[Parameter],
    ) -> None:
        """Validate that all formula parameters have units, raise errors if not."""
        for param in failed_params:
            if param.id not in variable_units:
                if not isinstance(param.value, str):
                    raise UnitConsistencyError(
                        (
                            f"Cannot infer unit for parameter '{param.id}'. "
                            f"Parameter value is not a formula string. "
                            f"Please provide an explicit unit for this parameter."
                        )
                    )

                formula_vars = get_expression_variables(param.value)
                missing_vars = [v for v in formula_vars if v not in variable_units]

                if missing_vars:
                    raise UnitConsistencyError(
                        (
                            f"Cannot infer unit for formula parameter '{param.id}'. "
                            f"Formula '{param.value}' references variables without "
                            f"units: {', '.join(missing_vars)}. "
                            f"Please specify units for all referenced parameters or "
                            f"provide an explicit unit for '{param.id}'."
                        )
                    )
                else:
                    raise UnitConsistencyError(
                        (
                            f"Cannot infer unit for formula parameter '{param.id}'. "
                            f"Formula '{param.value}' could not be parsed. "
                            f"Please provide an explicit unit for this parameter."
                        )
                    )

    def _infer_time_unit(self) -> str | None:
        """
        Infer the time unit used in the model from parameter units.

        Returns
        -------
        str | None
            The inferred time unit (e.g., "day", "week", "month"), or None if no
            time unit can be inferred.

        Notes
        -----
        This method looks for parameters with units like "1/time_unit" or
        "quantity/time_unit" to infer the time unit used throughout the model.
        """
        from commol.utils.equations import ureg

        for param in self.parameters:
            if param.unit is None:
                continue

            try:
                unit_obj = ureg(param.unit)

                # Check if the unit has a time dimension in the denominator
                # (e.g., "1/day", "person/week", "individual/month")
                if "[time]" in str(unit_obj.dimensionality):
                    # Extract the time component
                    # The dimensionality will be like {[time]: -1, ...}
                    time_dimension = unit_obj.dimensionality.get("[time]", 0)
                    if isinstance(time_dimension, (int, float)) and time_dimension < 0:
                        # Get the time unit by analyzing the unit string
                        unit_str = str(unit_obj.units)

                        # Common time units to check
                        time_units = [
                            "second",
                            "minute",
                            "hour",
                            "day",
                            "week",
                            "fortnight",
                            "month",
                            "year",
                            "semester",
                            "wk",
                            "mon",
                            "yr",
                            "s",
                            "min",
                            "h",
                            "d",
                        ]

                        for time_unit in time_units:
                            if time_unit in unit_str:
                                return time_unit

            except Exception:
                continue

        return None

    def _check_transition_rate_units(
        self,
        rate: str,
        transition_id: str,
        variable_units: dict[str, str],
    ) -> tuple[bool, str | None]:
        """
        Check units for a single transition rate.

        For transitions in difference equations, rates represent the absolute change
        in population per time step, so they should have units of "bin_unit/time_unit"
        (e.g., "person/day", "individual/week").
        """
        # Get variables used in the rate expression
        variables = get_expression_variables(rate)

        # Build variable units for this specific rate
        rate_variable_units: dict[str, str] = {}
        for var in variables:
            if var in variable_units:
                rate_variable_units[var] = variable_units[var]
            else:
                # Variable not found
                # This should have been caught by earlier validation
                return (
                    False,
                    (
                        f"Transition '{transition_id}': Variable '{var}' in rate "
                        f"'{rate}' has no defined unit"
                    ),
                )

        # Determine the expected unit based on the transition's bins
        # Extract the bin unit from the transition source/target bins
        bin_unit = self._get_transition_bin_unit(transition_id)
        if not bin_unit:
            # Fallback to first bin's unit
            bin_unit = (
                self.population.bins[0].unit if self.population.bins else "person"
            )

        time_unit = self._infer_time_unit() or "day"

        expected_unit = f"{bin_unit}/{time_unit}"

        # Check unit consistency
        is_consistent, error_msg = check_equation_units(
            rate, rate_variable_units, expected_unit
        )

        if not is_consistent:
            return (
                False,
                f"Transition '{transition_id}': {error_msg}",
            )

        return (True, None)

    def _get_transition_bin_unit(self, transition_id: str) -> str | None:
        """
        Get the bin unit for a specific transition by looking at its source/target bins.

        Parameters
        ----------
        transition_id : str
            The ID of the transition to check.

        Returns
        -------
        str | None
            The unit of the bins involved in this transition, or None if not found.
        """
        # Find the transition
        for transition in self.dynamics.transitions:
            if transition.id == transition_id:
                # Get bins from source or target
                bin_ids = transition.source + transition.target

                # Find the first bin that has a unit defined
                for bin_id in bin_ids:
                    for bin_obj in self.population.bins:
                        if bin_obj.id == bin_id and bin_obj.unit:
                            return bin_obj.unit
                break

        return None

    def _get_rate_unit(
        self, rate: str, variable_units: dict[str, str] | None = None
    ) -> str | None:
        """
        Calculate the unit for a transition rate expression.

        Parameters
        ----------
        rate : str
            The rate expression to analyze.
        variable_units : dict[str, str] | None
            Pre-computed variable units mapping. If None, will be computed.
            Providing this parameter improves performance when calling this
            method multiple times.

        Returns
        -------
        str | None
            The unit of the rate expression, or None if units cannot be determined.
        """
        try:
            # Register custom units before parsing
            self._register_custom_units()

            # Use provided variable units or build them
            if variable_units is None:
                variable_units = self._build_variable_units()

            # Get variables used in the rate expression
            variables = get_expression_variables(rate)

            # Build variable units for this specific rate
            rate_variable_units: dict[str, str] = {}
            for var in variables:
                if var in variable_units:
                    rate_variable_units[var] = variable_units[var]
                else:
                    # Variable not found, cannot determine unit
                    return None

            # Parse the equation to get its unit
            from commol.utils.equations import parse_equation_unit

            equation_unit = parse_equation_unit(rate, rate_variable_units)
            return str(equation_unit.units)

        except Exception:
            # If any error occurs, return None
            return None

    def _annotate_rate_variables(
        self, rate: str, variable_units: dict[str, str] | None = None
    ) -> str:
        """
        Annotate variables in a rate expression with their units in parentheses.

        Parameters
        ----------
        rate : str
            The rate expression to annotate.
        variable_units : dict[str, str] | None
            Pre-computed variable units mapping for efficiency.

        Returns
        -------
        str
            The rate expression with variables annotated with their units,
            e.g., "beta(1/day) * S(person) * I(person) / N(person)".
            If units cannot be determined, returns the original rate.
        """
        if not rate:
            return rate

        try:
            # Use provided variable units or build them
            if variable_units is None:
                variable_units = self._build_variable_units()

            # Get variables used in the rate expression
            variables = get_expression_variables(rate)

            # Build a mapping of variable -> annotated version
            annotated_rate = rate
            # Sort by length descending to avoid partial replacements
            # (e.g., replace "beta_young" before "beta")
            sorted_vars = sorted(variables, key=lambda x: len(x), reverse=True)
            for var in sorted_vars:
                if var in variable_units:
                    unit = variable_units[var]
                    # Replace the variable with annotated version
                    # Use word boundaries to avoid partial matches
                    import re

                    pattern = r"\b" + re.escape(var) + r"\b"
                    replacement = f"{var}({unit})"
                    annotated_rate = re.sub(pattern, replacement, annotated_rate)

            return annotated_rate

        except Exception:
            # If any error occurs, return original rate
            return rate

    def _has_all_units(self) -> bool:
        """
        Check if all units are defined. Raises error if partial units.

        Returns
        -------
        bool
            True if all units defined, False if no units defined.

        Raises
        ------
        ValueError
            If units are partially defined.
        """
        has_any_bin_unit = any(b.unit for b in self.population.bins)
        has_any_param_unit = any(
            p.unit for p in self.parameters if not isinstance(p.value, str)
        )

        if not has_any_bin_unit and not has_any_param_unit:
            return False

        # If any units exist, all must be defined
        if not all(b.unit for b in self.population.bins):
            raise ValueError("Some bins have units but not all")
        if any(p.unit is None for p in self.parameters if not isinstance(p.value, str)):
            raise ValueError("Some parameters have units but not all")

        return True

    def _format_rate_with_unit(
        self,
        rate: str | None,
        variable_units: dict[str, str] | None = None,
        show_units: bool = True,
        format: str = PrintEquationsOutputFormat.TEXT,
    ) -> str:
        """
        Format a rate expression with variable units and final unit annotation.

        Parameters
        ----------
        rate : str | None
            The rate expression to format.
        variable_units : dict[str, str] | None
            Pre-computed variable units mapping for efficiency.
        show_units : bool, default=True
            If True, annotate variables and show final unit.
            If False, return the plain rate expression.
        format : str, default=PrintEquationsOutputFormat.TEXT
            Output format: "text" or "latex"

        Returns
        -------
        str
            The formatted rate string with variables annotated and final unit.
            Returns "None" if rate is None.
        """
        if not rate:
            return (
                "None" if format == PrintEquationsOutputFormat.TEXT else "\\text{None}"
            )

        if format == PrintEquationsOutputFormat.LATEX:
            # For LaTeX format
            if not show_units:
                # No units, just convert to LaTeX
                return self._latex_rate_expression(rate)

            # Annotate variables with their units first
            annotated_rate = self._annotate_rate_variables(rate, variable_units)

            # Add final unit
            unit = self._get_rate_unit(rate, variable_units)
            unit_suffix = f" [{unit}]" if unit else ""

            # Convert annotated rate to LaTeX
            latex_rate = self._latex_rate_expression(annotated_rate)

            # Wrap final unit in \text{} if present
            if unit_suffix:
                latex_unit_suffix = f" [\\text{{{unit}}}]"
                return f"{latex_rate}{latex_unit_suffix}"
            return latex_rate

        if not show_units:
            return rate

        # Annotate variables with their units (text format)
        annotated_rate = self._annotate_rate_variables(rate, variable_units)

        # Add final unit
        unit = self._get_rate_unit(rate, variable_units)
        unit_suffix = f" [{unit}]" if unit else ""

        return f"{annotated_rate}{unit_suffix}"
