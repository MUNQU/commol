from typing import Self

from pydantic import BaseModel, Field, model_validator

from epimodel.constants import ModelTypes
from epimodel.context.dynamics import Dynamics, Transition
from epimodel.context.parameter import Parameter
from epimodel.context.population import Population


class Model(BaseModel):
    """
    Root class of epidemiological model.

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
    def validate_transition_ids(self) -> Self:
        """
        Validates that transition ids (source/target) are consistent in type
        and match the defined DiseaseState IDs or Stratification Categories
        in the Population instance.
        """

        disease_state_ids = {state.id for state in self.population.disease_states}
        categories_ids = {
            cat for strat in self.population.stratifications for cat in strat.categories
        }
        disease_state_and_categories_ids = disease_state_ids.union(categories_ids)

        for transition in self.dynamics.transitions:
            source = set(transition.source)
            target = set(transition.target)
            transition_ids = source.union(target)

            if not transition_ids.issubset(disease_state_and_categories_ids):
                invalid_ids = transition_ids - disease_state_and_categories_ids
                raise ValueError(
                    (
                        f"Transition '{transition.id}' contains invalid ids: "
                        f"{invalid_ids}. Ids must be defined in DiseaseState ids "
                        f"or Stratification Categories."
                    )
                )

            is_disease_state_flow = transition_ids.issubset(disease_state_ids)
            is_stratification_flow = transition_ids.issubset(categories_ids)

            if (not is_disease_state_flow) and (not is_stratification_flow):
                disease_state_elements = transition_ids.intersection(disease_state_ids)
                categories_elements = transition_ids.intersection(categories_ids)
                raise ValueError(
                    (
                        f"Transition '{transition.id}' mixes id types. "
                        f"Found DiseaseState ids ({disease_state_elements}) and "
                        f"Stratification Categories ids ({categories_elements}). "
                        "Transitions must be purely Disease State flow or purely "
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

        # Check if model has stratifications
        has_stratifications = len(self.population.stratifications) > 0

        if has_stratifications:
            # Enhanced output for stratified models
            lines.extend(self._generate_compact_form())
            lines.append("")
            lines.extend(self._generate_expanded_form())
        else:
            # Simple output for non-stratified models
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
        lines.append(f"Number of Disease States: {len(self.population.disease_states)}")
        lines.append(
            f"Number of Stratifications: {len(self.population.stratifications)}"
        )
        lines.append(f"Number of Parameters: {len(self.parameters)}")
        lines.append(f"Number of Transitions: {len(self.dynamics.transitions)}")

        # List disease states
        disease_state_ids = [state.id for state in self.population.disease_states]
        lines.append(f"Disease States: {', '.join(disease_state_ids)}")

        # List stratifications
        if self.population.stratifications:
            lines.append("Stratifications:")
            for strat in self.population.stratifications:
                categories = ", ".join(strat.categories)
                lines.append(f"  - {strat.id}: [{categories}]")

        lines.append("")
        return lines

    def _collect_state_ids(self) -> set[str]:
        """Collect all state IDs from disease states and stratifications."""
        state_ids = {state.id for state in self.population.disease_states}
        for strat in self.population.stratifications:
            state_ids.update(strat.categories)
        return state_ids

    def _build_flow_equations(
        self, state_ids: set[str]
    ) -> dict[str, dict[str, list[str]]]:
        """Build a mapping of states to their inflows and outflows."""
        equations: dict[str, dict[str, list[str]]] = {
            state_id: {"inflows": [], "outflows": []} for state_id in state_ids
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

    def _format_state_equation(self, flows: dict[str, list[str]]) -> str:
        """Format the equation for a single state from its flows."""
        terms: list[str] = []

        for inflow in flows["inflows"]:
            terms.append(f"+ ({inflow})" if inflow else "+ ()")

        for outflow in flows["outflows"]:
            terms.append(f"- ({outflow})" if outflow else "- ()")

        if not terms:
            return ""

        result = " ".join(terms)
        # Remove leading + sign if present
        if result.startswith("+"):
            result = result[1:]
        return result

    def _generate_compact_form(self) -> list[str]:
        """Generate compact mathematical notation form for stratified models."""
        lines: list[str] = []
        lines.append("=" * 40)
        lines.append("COMPACT FORM")
        lines.append("=" * 40)
        lines.append("")

        disease_state_ids = [state.id for state in self.population.disease_states]
        disease_transitions, stratification_transitions = (
            self._separate_transitions_by_type()
        )

        lines.extend(
            self._format_stratification_transitions_compact(
                disease_state_ids, stratification_transitions
            )
        )
        lines.extend(
            self._format_disease_transitions_compact(
                disease_state_ids, disease_transitions
            )
        )
        lines.extend(self._format_total_system_size(disease_state_ids))

        return lines

    def _separate_transitions_by_type(
        self,
    ) -> tuple[list[Transition], list[Transition]]:
        """Separate transitions into disease and stratification types."""
        disease_state_ids = [state.id for state in self.population.disease_states]
        disease_state_set = set(disease_state_ids)

        disease_transitions: list[Transition] = []
        stratification_transitions: list[Transition] = []

        for transition in self.dynamics.transitions:
            transition_states = set(transition.source) | set(transition.target)
            if transition_states.issubset(disease_state_set):
                disease_transitions.append(transition)
            else:
                stratification_transitions.append(transition)

        return disease_transitions, stratification_transitions

    def _format_stratification_transitions_compact(
        self, disease_state_ids: list[str], stratification_transitions: list[Transition]
    ) -> list[str]:
        """Format stratification transitions in compact form."""
        lines: list[str] = []
        strat_by_id = self._group_transitions_by_stratification(
            stratification_transitions
        )

        for strat in self.population.stratifications:
            if strat_by_id[strat.id]:
                lines.append(f"Stratification Transitions ({strat.id}):")
                disease_states_str = ", ".join(disease_state_ids)
                lines.append(f"For each disease state X in {{{disease_states_str}}}:")

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

    def _format_disease_transitions_compact(
        self, disease_state_ids: list[str], disease_transitions: list[Transition]
    ) -> list[str]:
        """Format disease state transitions in compact form."""
        lines: list[str] = []

        if not disease_transitions:
            return lines

        lines.append("Disease State Transitions:")

        if self.population.stratifications:
            all_categories = [
                cat
                for strat in self.population.stratifications
                for cat in strat.categories
            ]
            categories_str = ", ".join(all_categories)
            lines.append(f"For each stratification s in {{{categories_str}}}:")

        for disease_state in disease_state_ids:
            equation = self._build_disease_state_equation(
                disease_state, disease_transitions
            )
            if equation:
                suffix = "_s" if self.population.stratifications else ""
                lines.append(f"  d{disease_state}{suffix}/dt: {equation}")

        lines.append("")
        return lines

    def _build_disease_state_equation(
        self, disease_state: str, transitions: list[Transition]
    ) -> str:
        """Build equation for a disease state."""
        inflows: list[str] = []
        outflows: list[str] = []

        for transition in transitions:
            if not transition.rate:
                continue

            source_count = transition.source.count(disease_state)
            target_count = transition.target.count(disease_state)
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

    def _format_total_system_size(self, disease_state_ids: list[str]) -> list[str]:
        """Format the total system size information."""
        lines: list[str] = []

        num_disease_states = len(disease_state_ids)
        num_strat_combinations = 1
        for strat in self.population.stratifications:
            num_strat_combinations *= len(strat.categories)
        total_equations = num_disease_states * num_strat_combinations

        lines.append(
            (
                f"Total System: {total_equations} coupled equations "
                f"({num_disease_states} disease states × {num_strat_combinations} "
                f"stratification)"
            )
        )

        return lines

    def _generate_expanded_form(self) -> list[str]:
        """Generate expanded form with individual equations."""
        lines: list[str] = []

        has_stratifications = len(self.population.stratifications) > 0

        if has_stratifications:
            lines.append("=" * 40)
            lines.append("EXPANDED FORM")
            lines.append("=" * 40)
        else:
            lines.append("=" * 40)
            lines.append("EQUATIONS")
            lines.append("=" * 40)

        state_ids = self._collect_state_ids()
        equations = self._build_flow_equations(state_ids)

        # Order: disease states first (in order),
        # then stratification categories (in order)
        disease_state_ids = [state.id for state in self.population.disease_states]
        stratification_category_ids = [
            cat for strat in self.population.stratifications for cat in strat.categories
        ]
        ordered_state_ids = disease_state_ids + stratification_category_ids

        for state_id in ordered_state_ids:
            equation = self._format_state_equation(equations[state_id])
            lines.append(f"d{state_id}/dt = {equation}")

        return lines

    def _write_output(self, output: str, output_file: str | None) -> None:
        """Write output to file or console."""
        if output_file:
            with open(output_file, "w") as f:
                _ = f.write(output)
        else:
            print(output)
