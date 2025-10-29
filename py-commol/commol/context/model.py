from itertools import product, combinations
from typing import Self

from pydantic import BaseModel, Field, model_validator

from commol.constants import ModelTypes
from commol.context.dynamics import Dynamics, Transition
from commol.context.parameter import Parameter
from commol.context.population import Population
from commol.context.stratification import Stratification
from commol.utils.equations import (
    UnitConsistencyError,
    check_all_parameters_have_units,
    check_equation_units,
    get_predefined_variable_units,
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

    def print_equations(self, output_file: str | None = None) -> None:
        """
        Prints the difference equations of the model in mathematical form.

        Displays model metadata and the system of difference equations in both
        compact (mathematical notation) and expanded (individual equations) forms.

        Parameters
        ----------
        output_file : str | None
            If provided, writes the equations to this file path instead of printing
            to console. If None, prints to console.

        Raises
        ------
        ValueError
            If the model is not a DifferenceEquations model.
        """

        if self.dynamics.typology != ModelTypes.DIFFERENCE_EQUATIONS:
            raise ValueError(
                (
                    f"print_equations only supports DifferenceEquations models. "
                    f"Current model type: {self.dynamics.typology}"
                )
            )

        lines = self._generate_model_header()

        # Always generate both compact and expanded forms
        lines.extend(self._generate_compact_form())
        lines.append("")
        lines.extend(self._generate_expanded_form())

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

    def _format_bin_equation(self, flows: dict[str, list[str]]) -> str:
        """Format the equation for a single bin or category from its flows."""
        terms: list[str] = []

        for inflow in flows["inflows"]:
            if inflow:  # Only add if not empty
                terms.append(f"+ ({inflow})")

        for outflow in flows["outflows"]:
            if outflow:  # Only add if not empty
                terms.append(f"- ({outflow})")

        if not terms:
            return "0"

        result = " ".join(terms)
        # Remove leading + sign and space if present
        if result.startswith("+ "):
            result = result[2:]
        return result

    def _generate_compact_form(self) -> list[str]:
        """Generate compact mathematical notation form for stratified models."""
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
                bin_transitions, compartments
            )
        )
        lines.extend(
            self._format_stratification_transitions_compact_stratified(
                stratification_transitions, bin_ids
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

    def _compartment_to_string(self, compartment: tuple[str, ...]) -> str:
        """Convert compartment tuple to string like 'S_young_urban'."""
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

    def _format_stratification_transitions_compact(
        self, bin_ids: list[str], stratification_transitions: list[Transition]
    ) -> list[str]:
        """Format stratification transitions in compact form."""
        lines: list[str] = []
        strat_by_id = self._group_transitions_by_stratification(
            stratification_transitions
        )

        for strat in self.population.stratifications:
            if strat_by_id[strat.id]:
                lines.append(f"Stratification Transitions ({strat.id}):")
                disease_states_str = ", ".join(bin_ids)
                lines.append(f"For each bin X in {{{disease_states_str}}}:")

                for category in strat.categories:
                    equation = self._build_category_equation(
                        category, strat_by_id[strat.id]
                    )
                    if equation:
                        lines.append(f"  dX_{category}/dt: {equation}")

                lines.append("")

        return lines

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

    def _build_category_equation(
        self, category: str, transitions: list[Transition]
    ) -> str:
        """Build equation for a stratification category."""
        inflows: list[str] = []
        outflows: list[str] = []

        for transition in transitions:
            if not transition.rate:
                continue

            source_count = transition.source.count(category)
            target_count = transition.target.count(category)
            net_change = target_count - source_count

            if net_change > 0:
                inflows.append(f"+ ({transition.rate} * X)")
            elif net_change < 0:
                outflows.append(f"- ({transition.rate} * X)")

        terms = inflows + outflows
        if not terms:
            return ""

        result = " ".join(terms)
        # Remove leading + sign if present
        if result.startswith("+"):
            result = result[1:]
        return result

    def _format_bin_transitions_compact(
        self, bin_ids: list[str], bin_transitions: list[Transition]
    ) -> list[str]:
        """Format bin transitions in compact form."""
        lines: list[str] = []

        if not bin_transitions:
            return lines

        lines.append("Bin Transitions:")

        if self.population.stratifications:
            all_categories = [
                cat
                for strat in self.population.stratifications
                for cat in strat.categories
            ]
            categories_str = ", ".join(all_categories)
            lines.append(f"For each stratification s in {{{categories_str}}}:")

        for bin_id in bin_ids:
            equation = self._build_bin_equation_from_transitions(
                bin_id, bin_transitions
            )
            if equation:
                suffix = "_s" if self.population.stratifications else ""
                lines.append(f"  d{bin_id}{suffix}/dt: {equation}")

        lines.append("")
        return lines

    def _build_bin_equation_from_transitions(
        self, bin_id: str, transitions: list[Transition]
    ) -> str:
        """Build equation for a bin."""
        inflows: list[str] = []
        outflows: list[str] = []

        for transition in transitions:
            if not transition.rate:
                continue

            source_count = transition.source.count(bin_id)
            target_count = transition.target.count(bin_id)
            net_change = target_count - source_count

            if net_change > 0:
                inflows.append(f"+ ({transition.rate})")
            elif net_change < 0:
                outflows.append(f"- ({transition.rate})")

        terms = inflows + outflows
        if not terms:
            return ""

        result = " ".join(terms)
        # Remove leading + sign if present
        if result.startswith("+"):
            result = result[1:]
        return result

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
        self, bin_transitions: list[Transition], compartments: list[tuple[str, ...]]
    ) -> list[str]:
        """Format bin transitions showing specific compartments and rates."""
        lines: list[str] = []

        if not bin_transitions:
            return lines

        lines.append("Bin Transitions:")

        for transition in bin_transitions:
            source_bins = transition.source
            target_bins = transition.target

            source_str = ", ".join(sorted(set(source_bins))) if source_bins else "none"
            target_str = ", ".join(sorted(set(target_bins))) if target_bins else "none"
            lines.append(
                f"{transition.id.capitalize()} ({source_str} -> {target_str}):"
            )

            for compartment in compartments:
                bin_id = compartment[0]

                if bin_id in source_bins:
                    source_compartment_str = self._compartment_to_string(compartment)

                    if target_bins:
                        target_bin = target_bins[0]
                        target_compartment_str = source_compartment_str.replace(
                            bin_id, target_bin, 1
                        )
                    else:
                        target_compartment_str = "none"

                    rate = self._get_rate_for_compartment(transition, compartment)

                    lines.append(
                        (
                            f"  {source_compartment_str} -> "
                            f"{target_compartment_str}: {rate}"
                        )
                    )

            lines.append("")

        return lines

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
    ) -> str:
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

        source_comp = f"X_{'_'.join(source_parts)}"
        target_comp = f"X_{'_'.join(target_parts)}"

        sample_compartment = ("X",) + tuple(source_parts)
        rate = self._get_rate_for_compartment(trans, sample_compartment)
        rate_expr = (
            f"{rate} * {source_comp}" if rate else f"{trans.rate} * {source_comp}"
        )

        return f"  {source_comp} -> {target_comp}: {rate_expr}"

    def _format_stratification_transitions_compact_stratified(
        self,
        stratification_transitions: list[Transition],
        bin_ids: list[str],
    ) -> list[str]:
        """Format stratification transitions showing movements between categories."""
        lines: list[str] = []
        strat_by_id = self._group_transitions_by_stratification(
            stratification_transitions
        )

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
                        self._build_stratified_transition_line(trans, strat_idx, combo)
                    )

            lines.append("")

        return lines

    def _generate_expanded_form(self) -> list[str]:
        """Generate expanded form with individual equations for each compartment."""
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
                compartment_str = self._compartment_to_string(compartment)
                equation = self._build_compartment_equation(
                    compartment, bin_transitions, stratification_transitions
                )
                lines.append(f"d{compartment_str}/dt = {equation}")
        else:
            bin_and_category_ids = self._collect_bin_and_category_ids()
            equations = self._build_flow_equations(bin_and_category_ids)
            bin_ids = [bin_item.id for bin_item in self.population.bins]

            for bin_id in bin_ids:
                equation = self._format_bin_equation(equations[bin_id])
                lines.append(f"d{bin_id}/dt = {equation}")

        return lines

    def _build_compartment_equation(
        self,
        compartment: tuple[str, ...],
        bin_transitions: list[Transition],
        stratification_transitions: list[Transition],
    ) -> str:
        """Build the complete equation for a specific compartment."""
        terms: list[str] = []
        bin_id = compartment[0]

        for transition in bin_transitions:
            source_count = transition.source.count(bin_id)
            target_count = transition.target.count(bin_id)
            net_change = target_count - source_count

            if net_change != 0:
                rate = self._get_rate_for_compartment(transition, compartment)
                if rate:
                    if net_change > 0:
                        terms.append(f"+ ({rate})")
                    else:
                        terms.append(f"- ({rate})")

        for transition in stratification_transitions:
            flow_term = self._get_stratification_flow_for_compartment(
                compartment, transition
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

    def _get_stratification_flow_for_compartment(
        self, compartment: tuple[str, ...], transition: Transition
    ) -> str | None:
        """Calculate stratification flow term for a compartment."""
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

        if net_change < 0:
            compartment_str = self._compartment_to_string(compartment)
            return f"- ({rate} * {compartment_str})"
        else:
            source_category = source_categories[0] if source_categories else None
            if source_category:
                source_compartment = list(compartment)
                source_compartment[target_strat_idx + 1] = source_category
                source_compartment_str = self._compartment_to_string(
                    tuple(source_compartment)
                )
                return f"+ ({rate} * {source_compartment_str})"

        return None

    def _write_output(self, output: str, output_file: str | None) -> None:
        """Write output to file or console."""
        if output_file:
            with open(output_file, "w") as f:
                _ = f.write(output)
        else:
            print(output)

    def check_unit_consistency(self) -> None:
        """
        Check unit consistency of all equations in the model.

        This method validates that all transition rates have consistent units.
        It only performs the check if ALL parameters have units specified.
        If any parameter lacks a unit, the check is skipped.

        For difference equation models, all rates should have units that result in
        population change rates (e.g., "person/day" or "1/day" when multiplied by
        population).

        Raises
        ------
        UnitConsistencyError
            If unit inconsistencies are found in any equation.
        ValueError
            If the model type doesn't support unit checking.

        Examples
        --------
        >>> model.check_unit_consistency()  # Raises exception if inconsistent

        Notes
        -----
        - Bin variables are assumed to have units of "person"
        - Predefined variables (N, N_young, etc.) have units of "person"
        - Time step variables (t, step) are dimensionless
        - Mathematical constants (pi, e) are dimensionless
        """
        # Check if all parameters have units
        if not check_all_parameters_have_units(self.parameters):
            # Skip check if not all parameters have units
            return

        if self.dynamics.typology != ModelTypes.DIFFERENCE_EQUATIONS:
            raise ValueError(
                (
                    f"Unit checking is only supported for DifferenceEquations models. "
                    f"Current model type: {self.dynamics.typology}"
                )
            )

        # Build variable units mapping
        variable_units = self._build_variable_units()

        # Check each transition
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

        if errors:
            error_message = "Unit consistency check failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise UnitConsistencyError(error_message)

    def _build_variable_units(self) -> dict[str, str]:
        """Build a mapping of all variables to their units."""
        variable_units: dict[str, str] = {}

        # Add parameter units
        for param in self.parameters:
            if param.unit:
                variable_units[param.id] = param.unit

        # Add bin units (all are "person")
        for state in self.population.bins:
            variable_units[state.id] = "person"

        # Add stratification category units (all are "person" for population counts)
        for strat in self.population.stratifications:
            for category in strat.categories:
                variable_units[category] = "person"

        # Add predefined variable units (N, N_young, etc.)
        predefined_units = get_predefined_variable_units(
            self.population.stratifications
        )
        variable_units.update(predefined_units)

        # Add dimensionless special variables
        variable_units["step"] = "dimensionless"
        variable_units["t"] = "dimensionless"
        variable_units["pi"] = "dimensionless"
        variable_units["e"] = "dimensionless"

        return variable_units

    def _check_transition_rate_units(
        self,
        rate: str,
        transition_id: str,
        variable_units: dict[str, str],
    ) -> tuple[bool, str | None]:
        """
        Check units for a single transition rate.

        For transitions in difference equations, rates represent the absolute change
        in population per time step, so they should have units of "person/day"
        (or person/timestep).
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

        # For difference equations, rates should result in person per time unit
        # This represents the absolute change in population per time step
        expected_unit = "person/day"

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
