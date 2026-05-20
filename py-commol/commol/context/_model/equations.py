import re

from commol.constants import ModelTypes, PrintEquationsOutputFormat
from commol.context._model.helpers import (
    ModelCompartmentHelper,
    replace_bin_in_rate,
)
from commol.context.dynamics import Transition
from commol.utils.security import get_expression_variables


class EquationPrinter:
    """Formats and prints model equations in text or LaTeX."""

    def __init__(self, model) -> None:
        self._model = model
        self._helper = ModelCompartmentHelper(model)

    def print_equations(
        self,
        output_file: str | None = None,
        format: str = PrintEquationsOutputFormat.TEXT,
    ) -> None:
        """Print the equations of the model in mathematical form."""
        if format not in PrintEquationsOutputFormat:
            raise ValueError(
                f"Invalid format: {format}. "
                f"Must be one of {list(PrintEquationsOutputFormat)}"
            )

        lines = self._generate_model_header()
        lines.extend(self._generate_compact_form(format=format))
        lines.append("")
        lines.extend(self._generate_expanded_form(format=format))

        output = "\n".join(lines)
        _write_output(output, output_file)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _generate_model_header(self) -> list[str]:
        """Generate the header lines with model metadata."""
        lines: list[str] = []
        lines.append("=" * 40)
        lines.append("MODEL INFORMATION")
        lines.append("=" * 40)
        lines.append(f"Model: {self._model.name}")
        lines.append(f"Model Type: {self._model.dynamics.typology}")
        lines.append(f"Number of Bins: {len(self._model.population.bins)}")
        lines.append(
            f"Number of Stratifications: {len(self._model.population.stratifications)}"
        )
        lines.append(f"Number of Parameters: {len(self._model.parameters)}")
        lines.append(f"Number of Transitions: {len(self._model.dynamics.transitions)}")

        bin_ids = [bin_item.id for bin_item in self._model.population.bins]
        lines.append(f"Bins: {', '.join(bin_ids)}")

        if self._model.population.stratifications:
            lines.append("Stratifications:")
            for strat in self._model.population.stratifications:
                categories = ", ".join(strat.categories)
                lines.append(f"  - {strat.id}: [{categories}]")

        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Non-stratified helpers
    # ------------------------------------------------------------------

    def _collect_bin_and_category_ids(self) -> set[str]:
        all_ids = {bin_item.id for bin_item in self._model.population.bins}
        for strat in self._model.population.stratifications:
            all_ids.update(strat.categories)
        return all_ids

    def _build_flow_equations(
        self, bin_and_category_ids: set[str]
    ) -> dict[str, dict[str, list[str]]]:
        equations: dict[str, dict[str, list[str]]] = {
            id_: {"inflows": [], "outflows": []} for id_ in bin_and_category_ids
        }
        for transition in self._model.dynamics.transitions:
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
        terms: list[str] = []

        for inflow in flows["inflows"]:
            if inflow:
                if format == PrintEquationsOutputFormat.LATEX:
                    formatted_inflow = latex_rate_expression(inflow)
                else:
                    formatted_inflow = inflow
                terms.append(f"+ ({formatted_inflow})")

        for outflow in flows["outflows"]:
            if outflow:
                if format == PrintEquationsOutputFormat.LATEX:
                    formatted_outflow = latex_rate_expression(outflow)
                else:
                    formatted_outflow = outflow
                terms.append(f"- ({formatted_outflow})")

        if not terms:
            return "0"

        result = " ".join(terms)
        if result.startswith("+ "):
            result = result[2:]
        return result

    # ------------------------------------------------------------------
    # Compact form
    # ------------------------------------------------------------------

    def _generate_compact_form(self, format: str) -> list[str]:
        lines: list[str] = []
        lines.append("=" * 40)
        lines.append("COMPACT FORM")
        lines.append("=" * 40)
        lines.append("")

        bin_ids = [bin_item.id for bin_item in self._model.population.bins]
        bin_transitions, stratification_transitions, cross_category_transitions = (
            self._helper.separate_transitions_by_type()
        )

        compartments = self._helper.generate_compartments()

        lines.extend(
            self._format_bin_transitions_compact_stratified(
                bin_transitions, compartments, format
            )
        )
        lines.extend(
            self._format_cross_category_transitions_compact(
                cross_category_transitions, compartments, format
            )
        )
        lines.extend(
            self._format_stratification_transitions_compact_stratified(
                stratification_transitions, bin_ids, format
            )
        )
        lines.extend(self._format_total_system_size(bin_ids, len(compartments)))

        return lines

    def _format_cross_category_transitions_compact(
        self,
        cross_category_transitions: list[Transition],
        compartments: list[tuple[str, dict[str, str]]],
        format: str,
    ) -> list[str]:
        lines: list[str] = []
        if not cross_category_transitions:
            return lines

        show_units = self._helper.has_all_units()
        variable_units = self._helper.build_variable_units() if show_units else None

        for transition in cross_category_transitions:
            source_str = (
                ", ".join(sorted(set(transition.source)))
                if transition.source
                else "none"
            )
            target_str = (
                ", ".join(sorted(set(transition.target)))
                if transition.target
                else source_str
            )
            lines.append(f"{transition.id} ({source_str} -> {target_str}):")

            for compartment in compartments:
                bin_id = compartment[0]
                if bin_id not in transition.source:
                    continue
                flows = self._get_cross_category_flows(transition, compartment)
                for rate_str, target_comp in flows:
                    source_compartment_str = self._helper.compartment_to_string(
                        compartment, format
                    )
                    target_compartment_str = self._helper.compartment_to_string(
                        target_comp, format
                    )
                    rate_with_unit = self._helper.format_rate_with_unit(
                        rate_str, variable_units, show_units, format
                    )

                    if format == PrintEquationsOutputFormat.LATEX:
                        arrow = _latex_transition_arrow(
                            source_compartment_str, target_compartment_str
                        )
                        lines.append(f"  ${arrow}: {rate_with_unit}$")
                    else:
                        lines.append(
                            f"  {source_compartment_str} -> "
                            f"{target_compartment_str}: {rate_with_unit}"
                        )

            lines.append("")

        return lines

    def _format_bin_transitions_compact_stratified(
        self,
        bin_transitions: list[Transition],
        compartments: list[tuple[str, dict[str, str]]],
        format: str,
    ) -> list[str]:
        lines: list[str] = []
        if not bin_transitions:
            return lines

        lines.append("Bin Transitions:")

        show_units = self._helper.has_all_units()
        variable_units = self._helper.build_variable_units() if show_units else None

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
            lines.append(f"{transition.id} ({source_str} -> {target_str}):")

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
        compartments: list[tuple[str, dict[str, str]]],
        variable_units: dict[str, str] | None,
        show_units: bool,
        format: str,
    ) -> list[str]:
        lines: list[str] = []
        for compartment in compartments:
            bin_id = compartment[0]
            if bin_id in transition.target:
                target_compartment_str = self._helper.compartment_to_string(
                    compartment, format
                )
                rate = self._get_rate_for_compartment(transition, compartment)
                rate_with_unit = self._helper.format_rate_with_unit(
                    rate, variable_units, show_units, format
                )

                if format == PrintEquationsOutputFormat.LATEX:
                    arrow = _latex_transition_arrow("none", target_compartment_str)
                    lines.append(f"  ${arrow}: {rate_with_unit}$")
                else:
                    lines.append(
                        f"  none -> {target_compartment_str}: {rate_with_unit}"
                    )
        return lines

    def _format_normal_transition_lines(
        self,
        transition: Transition,
        compartments: list[tuple[str, dict[str, str]]],
        variable_units: dict[str, str] | None,
        show_units: bool,
        format: str,
    ) -> list[str]:
        lines: list[str] = []
        for compartment in compartments:
            bin_id = compartment[0]
            if bin_id in transition.source:
                source_compartment_str = self._helper.compartment_to_string(
                    compartment, format
                )
                target_compartment_str = self._get_target_compartment_str(
                    compartment, bin_id, transition.target, format
                )

                rate = self._get_rate_for_compartment(transition, compartment)
                rate_with_unit = self._helper.format_rate_with_unit(
                    rate, variable_units, show_units, format
                )

                if format == PrintEquationsOutputFormat.LATEX:
                    arrow = _latex_transition_arrow(
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
        compartment: tuple[str, dict[str, str]],
        bin_id: str,
        target_bins: list[str],
        format: str,
    ) -> str:
        if not target_bins:
            return "none"

        target_bin = target_bins[0]
        if format == PrintEquationsOutputFormat.LATEX:
            _, applied = compartment
            target_compartment: tuple[str, dict[str, str]] = (target_bin, applied)
            return self._helper.compartment_to_string(target_compartment, format)
        else:
            source_compartment_str = self._helper.compartment_to_string(
                compartment, format
            )
            return source_compartment_str.replace(bin_id, target_bin, 1)

    def _format_total_system_size(
        self, bin_ids: list[str], num_compartments: int | None = None
    ) -> list[str]:
        lines: list[str] = []

        num_bins = len(bin_ids)
        if not self._model.population.stratifications:
            total_equations = num_bins
            lines.append(
                f"Total System: {total_equations} coupled equations ({num_bins} bins)"
            )
            return lines

        num_strat_combinations = 1
        strat_details: list[str] = []
        for strat in self._model.population.stratifications:
            num_cat = len(strat.categories)
            num_strat_combinations *= num_cat
            strat_details.append(f"{num_cat} {strat.id}")

        total_equations = (
            num_compartments
            if num_compartments is not None
            else num_bins * num_strat_combinations
        )

        lines.append(
            f"Total System: {total_equations} coupled equations "
            f"({num_bins} bins × {' × '.join(strat_details)})"
        )

        return lines

    def _build_stratified_for_each_line(
        self, bin_ids: list[str], other_strats: list
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
        src_cat = trans.source[0]
        tgt_cat = trans.target[0]

        source_parts = [""] * len(self._model.population.stratifications)
        target_parts = [""] * len(self._model.population.stratifications)
        source_parts[strat_idx] = src_cat
        target_parts[strat_idx] = tgt_cat

        combo_idx = 0
        for i in range(len(self._model.population.stratifications)):
            if i != strat_idx:
                source_parts[i] = combo[combo_idx]
                target_parts[i] = combo[combo_idx]
                combo_idx += 1

        if format == PrintEquationsOutputFormat.LATEX:
            source_comp_parts = "_".join(["X"] + source_parts)
            target_comp_parts = "_".join(["X"] + target_parts)
            source_comp = latex_variable(source_comp_parts)
            target_comp = latex_variable(target_comp_parts)
        else:
            source_comp = f"X_{'_'.join(source_parts)}"
            target_comp = f"X_{'_'.join(target_parts)}"

        sample_applied = {
            s.id: source_parts[i]
            for i, s in enumerate(self._model.population.stratifications)
        }
        sample_compartment: tuple[str, dict[str, str]] = ("X", sample_applied)
        rate = self._get_rate_for_compartment(trans, sample_compartment)

        if format == PrintEquationsOutputFormat.LATEX:
            effective_rate = rate if rate else trans.rate
            if effective_rate:
                rate_expr = (
                    f"{latex_rate_expression(effective_rate)} \\cdot {source_comp}"
                )
            else:
                rate_expr = f"\\text{{None}} \\cdot {source_comp}"
            arrow = _latex_transition_arrow(source_comp, target_comp)
            return f"  ${arrow}: {rate_expr}$"
        elif show_units and rate and self._model.population.bins:
            first_bin_id = self._model.population.bins[0].id
            concrete_compartment: tuple[str, dict[str, str]] = (
                first_bin_id,
                sample_applied,
            )
            concrete_comp_str = self._helper.compartment_to_string(
                concrete_compartment, format
            )
            full_rate_expr = f"{rate} * {concrete_comp_str}"

            annotated_expr = self._helper.annotate_rate_variables(
                full_rate_expr, variable_units
            )

            bin_unit = self._model.population.bins[0].unit
            annotated_expr = annotated_expr.replace(
                f"{concrete_comp_str}({bin_unit})", f"{source_comp}({bin_unit})"
            )

            unit = self._helper.get_rate_unit(full_rate_expr, variable_units)
            unit_str = f" [{unit}]" if unit else ""

            return f"  {source_comp} -> {target_comp}: {annotated_expr}{unit_str}"
        else:
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
        lines: list[str] = []
        strat_by_id = self._helper.group_transitions_by_stratification(
            stratification_transitions
        )

        show_units = self._helper.has_all_units()
        variable_units = self._helper.build_variable_units() if show_units else None

        for strat_idx, strat in enumerate(self._model.population.stratifications):
            if not strat_by_id.get(strat.id):
                continue

            transition = strat_by_id[strat.id][0]
            source_cat = transition.source[0] if transition.source else "none"
            target_cat = transition.target[0] if transition.target else "none"

            if not source_cat or not target_cat:
                continue

            lines.append(
                f"{strat.id.capitalize()} Stratification Transitions "
                f"({source_cat} -> {target_cat}):"
            )

            other_strats = [
                s
                for i, s in enumerate(self._model.population.stratifications)
                if i != strat_idx
            ]

            lines.append(self._build_stratified_for_each_line(bin_ids, other_strats))

            from itertools import product

            for trans in strat_by_id[strat.id]:
                other_cat_combos = (
                    list(product(*[s.categories for s in other_strats]))
                    if other_strats
                    else [()]
                )

                for combo in other_cat_combos:
                    lines.append(
                        self._build_stratified_transition_line(
                            trans,
                            strat_idx,
                            combo,
                            variable_units,
                            show_units,
                            format,
                        )
                    )

            lines.append("")

        return lines

    # ------------------------------------------------------------------
    # Expanded form
    # ------------------------------------------------------------------

    def _generate_expanded_form(self, format: str) -> list[str]:
        lines: list[str] = []

        lines.append("=" * 40)
        lines.append("EXPANDED FORM")
        lines.append("=" * 40)

        has_stratifications = len(self._model.population.stratifications) > 0

        if has_stratifications:
            compartments = self._helper.generate_compartments()
            (
                bin_transitions,
                stratification_transitions,
                cross_category_transitions,
            ) = self._helper.separate_transitions_by_type()

            for compartment in compartments:
                equation = self._build_compartment_equation(
                    compartment,
                    bin_transitions,
                    stratification_transitions,
                    cross_category_transitions,
                    format,
                )
                if equation is None:
                    continue
                compartment_str = self._helper.compartment_to_string(
                    compartment, format
                )
                lhs = _get_equation_lhs(
                    compartment_str, format, self._model.dynamics.typology
                )
                if format == PrintEquationsOutputFormat.LATEX:
                    lines.append(f"\\[{lhs} = {equation}\\]")
                else:
                    lines.append(f"{lhs} = {equation}")
        else:
            bin_and_category_ids = self._collect_bin_and_category_ids()
            equations = self._build_flow_equations(bin_and_category_ids)
            bin_ids = [bin_item.id for bin_item in self._model.population.bins]

            for bin_id in bin_ids:
                equation = self._format_bin_equation(equations[bin_id], format)
                lhs = _get_equation_lhs(bin_id, format, self._model.dynamics.typology)
                if format == PrintEquationsOutputFormat.LATEX:
                    lines.append(f"\\[{lhs} = {equation}\\]")
                else:
                    lines.append(f"{lhs} = {equation}")

        return lines

    def _build_compartment_equation(
        self,
        compartment: tuple[str, dict[str, str]],
        bin_transitions: list[Transition],
        stratification_transitions: list[Transition],
        cross_category_transitions: list[Transition],
        format: str,
    ) -> str | None:
        raw_terms: list[tuple[int, str]] = []

        raw_terms.extend(
            self._get_bin_transition_raw_terms(compartment, bin_transitions)
        )
        raw_terms.extend(
            self._get_stratification_raw_terms(compartment, stratification_transitions)
        )
        raw_terms.extend(
            self._get_cross_category_raw_terms(compartment, cross_category_transitions)
        )

        raw_terms = simplify_complementary_terms(raw_terms)

        terms = self._format_raw_terms(raw_terms, format)

        if not terms:
            return None

        equation = " ".join(terms)
        if equation.startswith("+ "):
            return equation[2:]
        if equation.startswith("+"):
            return equation[1:]
        return equation

    def _get_bin_transition_raw_terms(
        self,
        compartment: tuple[str, dict[str, str]],
        bin_transitions: list[Transition],
    ) -> list[tuple[int, str]]:
        raw_terms: list[tuple[int, str]] = []
        bin_id = compartment[0]

        for transition in bin_transitions:
            source_count = transition.source.count(bin_id)
            target_count = transition.target.count(bin_id)
            net_change = target_count - source_count

            if net_change != 0:
                rate = self._get_rate_for_compartment(transition, compartment)
                if rate:
                    sign = 1 if net_change > 0 else -1
                    raw_terms.append((sign, rate))

        return raw_terms

    def _get_stratification_raw_terms(
        self,
        compartment: tuple[str, dict[str, str]],
        stratification_transitions: list[Transition],
    ) -> list[tuple[int, str]]:
        raw_terms: list[tuple[int, str]] = []
        for transition in stratification_transitions:
            result = self._get_stratification_raw_term(compartment, transition)
            if result is not None:
                raw_terms.append(result)
        return raw_terms

    def _get_stratification_raw_term(
        self,
        compartment: tuple[str, dict[str, str]],
        transition: Transition,
    ) -> tuple[int, str] | None:
        _, applied = compartment
        if not applied:
            return None

        transition_states = set(transition.source) | set(transition.target)
        target_strat = None
        for strat in self._model.population.stratifications:
            if transition_states.issubset(set(strat.categories)):
                target_strat = strat
                break

        if target_strat is None:
            return None

        compartment_category = applied.get(target_strat.id)
        if compartment_category is None:
            return None

        source_count = transition.source.count(compartment_category)
        target_count = transition.target.count(compartment_category)
        net_change = target_count - source_count
        if net_change == 0:
            return None

        rate = self._get_rate_for_compartment(transition, compartment)
        if not rate:
            return None

        compartment_str = self._helper.compartment_to_string(
            compartment, PrintEquationsOutputFormat.TEXT
        )
        if net_change < 0:
            full_rate = f"{rate} * {compartment_str}"
            return (-1, full_rate)

        source_categories = transition.source
        source_category = source_categories[0] if source_categories else None
        if source_category:
            source_applied = {**applied, target_strat.id: source_category}
            source_compartment: tuple[str, dict[str, str]] = (
                compartment[0],
                source_applied,
            )
            source_str = self._helper.compartment_to_string(
                source_compartment, PrintEquationsOutputFormat.TEXT
            )
            full_rate = f"{rate} * {source_str}"
            return (1, full_rate)

        return None

    def _get_cross_category_raw_terms(
        self,
        compartment: tuple[str, dict[str, str]],
        cross_category_transitions: list[Transition],
    ) -> list[tuple[int, str]]:
        raw_terms: list[tuple[int, str]] = []
        bin_id = compartment[0]
        compartment_str = self._helper.compartment_to_string(
            compartment, PrintEquationsOutputFormat.TEXT
        )

        for transition in cross_category_transitions:
            if bin_id not in transition.source:
                continue

            # Outflows
            for rate_str, target_comp in self._get_cross_category_flows(
                transition, compartment
            ):
                target_str = self._helper.compartment_to_string(
                    target_comp, PrintEquationsOutputFormat.TEXT
                )
                if target_str != compartment_str:
                    raw_terms.append((-1, rate_str))

            # Inflows
            for source_comp in self._helper.generate_compartments():
                if source_comp[0] != bin_id or source_comp == compartment:
                    continue
                for rate_str, target_comp in self._get_cross_category_flows(
                    transition, source_comp
                ):
                    target_str = self._helper.compartment_to_string(
                        target_comp, PrintEquationsOutputFormat.TEXT
                    )
                    if target_str == compartment_str:
                        raw_terms.append((1, rate_str))

        return raw_terms

    def _format_raw_terms(
        self,
        raw_terms: list[tuple[int, str]],
        format: str,
    ) -> list[str]:
        terms: list[str] = []
        for sign, rate in raw_terms:
            formatted_rate = self._format_rate_expr(rate, format)
            prefix = "+" if sign > 0 else "-"
            terms.append(f"{prefix} ({formatted_rate})")
        return terms

    # ------------------------------------------------------------------
    # Rate resolution
    # ------------------------------------------------------------------

    def _compute_target_with_overrides(
        self,
        source_applied: dict[str, str],
        target_bin: str,
        conditions: list,
    ) -> tuple[str, dict[str, str]]:
        override_map = {c.stratification: c.to for c in conditions if c.to is not None}
        target_specifiers = {
            c.stratification: c.category
            for c in conditions
            if c.stratification not in source_applied and c.to is None
        }

        target_applied: dict[str, str] = {}
        for strat in self._model.population.stratifications:
            if strat.id in override_map:
                effective_cat = override_map[strat.id]
            elif strat.id in target_specifiers:
                effective_cat = target_specifiers[strat.id]
            elif strat.id in source_applied:
                effective_cat = source_applied[strat.id]
            else:
                effective_cat = None

            applies = strat.conditions is None or all(
                target_applied.get(c.stratification) == c.category
                for c in strat.conditions
            )

            if applies and effective_cat is not None:
                target_applied[strat.id] = effective_cat

        return (target_bin, target_applied)

    def _get_cross_category_flows(
        self,
        transition: Transition,
        source_compartment: tuple[str, dict[str, str]],
    ) -> list[tuple[str, tuple[str, dict[str, str]]]]:
        _, source_applied = source_compartment
        if not transition.stratified_rates:
            return []

        target_bin = (
            transition.target[0] if transition.target else source_compartment[0]
        )
        flows: list[tuple[str, tuple[str, dict[str, str]]]] = []

        for sr in transition.stratified_rates:
            source_matches = all(
                source_applied.get(c.stratification) == c.category
                for c in sr.conditions
                if c.stratification in source_applied
            )
            if source_matches:
                target = self._compute_target_with_overrides(
                    source_applied, target_bin, sr.conditions
                )
                flows.append((sr.rate, target))

        return flows

    def _match_stratified_rate(
        self,
        transition: Transition,
        compartment: tuple[str, dict[str, str]],
    ) -> str | None:
        _, applied = compartment
        for strat_rate in transition.stratified_rates or []:
            if all(
                applied.get(c.stratification) == c.category
                for c in strat_rate.conditions
            ):
                return strat_rate.rate
        return None

    def _apply_per_compartment_substitution(
        self,
        rate: str,
        transition: Transition,
        compartment: tuple[str, dict[str, str]],
    ) -> str:
        _, applied = compartment
        strat_parts = [
            applied[s.id]
            for s in self._model.population.stratifications
            if s.id in applied
        ]
        strat_suffix = ("_" + "_".join(strat_parts)) if strat_parts else ""
        for bin_name in transition.source[:1] + transition.target[:1]:
            full_name = bin_name + strat_suffix
            rate = replace_bin_in_rate(rate, bin_name, full_name)
        return rate

    def _get_rate_for_compartment(
        self,
        transition: Transition,
        compartment: tuple[str, dict[str, str]],
    ) -> str | None:
        _, applied = compartment
        if not transition.stratified_rates or not applied:
            rate = transition.rate
        else:
            rate = self._match_stratified_rate(transition, compartment)
            if rate is None:
                rate = transition.rate

        if rate and transition.per_compartment and applied:
            rate = self._apply_per_compartment_substitution(
                rate, transition, compartment
            )

        return rate

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_rate_expr(self, rate: str, format: str) -> str:
        if format == PrintEquationsOutputFormat.LATEX:
            return latex_rate_expression(rate)
        return rate


# ======================================================================
# Module-level functions (no model dependency)
# ======================================================================


def _write_output(output: str, output_file: str | None) -> None:
    """Write output to file or console."""
    if output_file:
        with open(output_file, "w") as f:
            _ = f.write(output)
    else:
        print(output)


def _get_equation_lhs(variable_name: str, format: str, typology: str) -> str:
    """Get the left-hand side of an equation based on model type."""
    if format == PrintEquationsOutputFormat.LATEX:
        if typology == ModelTypes.DIFFERENTIAL_EQUATIONS:
            return f"\\frac{{d{variable_name}}}{{dt}}"
        else:
            return (
                f"\\frac{{{variable_name}(t+\\Delta t) - "
                f"{variable_name}(t)}}{{\\Delta t}}"
            )
    else:
        if typology == ModelTypes.DIFFERENTIAL_EQUATIONS:
            return f"d{variable_name}/dt"
        else:
            return f"[{variable_name}(t+Dt) - {variable_name}(t)] / Dt"


def _latex_transition_arrow(source: str, target: str) -> str:
    """Format transition arrow for LaTeX."""
    formatted_source = "\\varnothing" if source == "none" else source
    formatted_target = "\\varnothing" if target == "none" else target
    return f"{formatted_source} \\to {formatted_target}"


# Greek letters commonly used in mathematical modeling
_GREEK_LETTERS = {
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


def latex_variable(var_name: str) -> str:
    """Format variable name for LaTeX math mode."""
    if var_name in _GREEK_LETTERS:
        return f"\\{var_name}"

    if "_" in var_name:
        parts = var_name.split("_")
        base = parts[0]
        subscripts = parts[1:]

        if base in _GREEK_LETTERS:
            base = f"\\{base}"

        subscript_str = ",".join(subscripts)
        return f"{base}_{{{subscript_str}}}"

    return var_name


def latex_rate_expression(rate: str) -> str:
    """Convert rate expression to LaTeX math mode."""
    if not rate:
        return rate

    latex_rate = rate.replace(" * ", " \\cdot ")

    variables = get_expression_variables(rate)
    sorted_vars = sorted(variables, key=lambda x: len(x), reverse=True)

    for var in sorted_vars:
        latex_var = latex_variable(var)
        pattern = r"\b" + re.escape(var) + r"\b"
        latex_rate = re.sub(pattern, latex_var.replace("\\", "\\\\"), latex_rate)

    # Wrap units in \text{} — exclude content with LaTeX commands or math operators
    latex_rate = re.sub(
        r"\(([^)\\{}+\-]*[a-zA-Z][^)\\{}+\-]*)\)", r"(\\text{\1})", latex_rate
    )
    latex_rate = re.sub(
        r"\[([^\]\\{}+\-]*[a-zA-Z][^\]\\{}+\-]*)\]", r"[\\text{\1}]", latex_rate
    )

    latex_rate = _convert_division_to_frac(latex_rate)

    return latex_rate


def _convert_division_to_frac(expr: str) -> str:
    """Convert division operations to LaTeX fractions."""
    if " / " not in expr:
        return expr

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
            i += 2
        i += 1

    if not divisions:
        return expr

    parts = []
    start = 0
    for div_pos in divisions:
        parts.append(expr[start:div_pos].strip())
        start = div_pos + 3
    parts.append(expr[start:].strip())

    result = _strip_outer_parens(parts[0])
    for i in range(1, len(parts)):
        denominator = _strip_outer_parens(parts[i])
        result = f"\\frac{{{result}}}{{{denominator}}}"

    return result


def _strip_outer_parens(expr: str) -> str:
    """Remove outer parentheses if they wrap the entire expression."""
    expr = expr.strip()
    if not expr.startswith("(") or not expr.endswith(")"):
        return expr

    depth = 0
    for i, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if depth == 0 and i < len(expr) - 1:
            return expr

    return expr[1:-1]


# ------------------------------------------------------------------
# Complementary term simplification
# ------------------------------------------------------------------


def tokenize_rate(rate: str) -> list[str]:
    """Tokenize a rate expression by ``*`` respecting parentheses."""
    tokens: list[str] = []
    depth = 0
    current_start = 0
    i = 0
    while i < len(rate):
        if rate[i] == "(":
            depth += 1
        elif rate[i] == ")":
            depth -= 1
        elif depth == 0 and i + 3 <= len(rate) and rate[i : i + 3] == " * ":
            tokens.append(rate[current_start:i].strip())
            i += 3
            current_start = i
            continue
        i += 1
    tokens.append(rate[current_start:].strip())
    return tokens


def are_complementary_factors(varying: list[str]) -> bool:
    """Check if a list of varying tokens are complementary percentages."""
    if len(varying) < 2:
        return False

    complement_idx = None
    for i, token in enumerate(varying):
        if token.startswith("(1 - ") and token.endswith(")"):
            complement_idx = i
            break

    if complement_idx is None:
        return False

    complement_token = varying[complement_idx]
    other_tokens = [t for i, t in enumerate(varying) if i != complement_idx]

    expected = "(1 - " + " - ".join(sorted(other_tokens)) + ")"

    inner = complement_token[1:-1]
    parts = [p.strip() for p in inner.split(" - ")]
    if not parts or parts[0] != "1":
        return False
    actual_sorted = "(1 - " + " - ".join(sorted(parts[1:])) + ")"

    return actual_sorted == expected


def simplify_complementary_terms(
    raw_terms: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Simplify groups of terms with complementary percentage factors."""
    if len(raw_terms) < 2:
        return raw_terms

    result: list[tuple[int, str]] = []
    used: set[int] = set()

    for sign_val in (-1, 1):
        indices = [
            i for i, (s, _) in enumerate(raw_terms) if s == sign_val and i not in used
        ]
        if len(indices) < 2:
            continue

        groups = find_common_factor_groups([raw_terms[i][1] for i in indices])

        for group_indices, common_tokens in groups:
            actual_indices = [indices[gi] for gi in group_indices]
            varying = []
            for gi in group_indices:
                rate_tokens = tokenize_rate(raw_terms[indices[gi]][1])
                diff = [t for t in rate_tokens if t not in common_tokens]
                if len(diff) == 1:
                    varying.append(diff[0])
                else:
                    varying = []
                    break

            if varying and are_complementary_factors(varying):
                simplified = " * ".join(common_tokens)
                result.append((sign_val, simplified))
                used.update(actual_indices)

    for i, term in enumerate(raw_terms):
        if i not in used:
            result.append(term)

    return _restore_term_order(raw_terms, result, used)


def find_common_factor_groups(
    rates: list[str],
) -> list[tuple[list[int], list[str]]]:
    """Find groups of rates that share common token factors."""
    tokenized = [tokenize_rate(r) for r in rates]
    groups: list[tuple[list[int], list[str]]] = []
    used_in_group: set[int] = set()

    for i in range(len(rates)):
        if i in used_in_group:
            continue

        group = _build_candidate_group(i, tokenized, used_in_group)

        if len(group) >= 2:
            common = _validate_group(group, tokenized)
            if common:
                groups.append((group, common))
                used_in_group.update(group)

    return groups


def _build_candidate_group(
    i: int,
    tokenized: list[list[str]],
    used: set[int],
) -> list[int]:
    group = [i]
    tokens_i = tokenized[i]

    for j in range(i + 1, len(tokenized)):
        if j in used:
            continue
        tokens_j = tokenized[j]
        common = [t for t in tokens_i if t in tokens_j]
        diff_i = [t for t in tokens_i if t not in common]
        diff_j = [t for t in tokens_j if t not in common]
        if len(diff_i) == 1 and len(diff_j) == 1 and common:
            group.append(j)

    return group


def _validate_group(
    group: list[int],
    tokenized: list[list[str]],
) -> list[str] | None:
    common = [t for t in tokenized[group[0]] if t in tokenized[group[1]]]
    for idx in group:
        diff = [t for t in tokenized[idx] if t not in common]
        if len(diff) != 1:
            return None
    return common


def _restore_term_order(
    original: list[tuple[int, str]],
    result: list[tuple[int, str]],
    used: set[int],
) -> list[tuple[int, str]]:
    """Restore term ordering after simplification."""
    if not used:
        return result

    original_set = set(original)
    simplified = [t for t in result if t not in original_set]
    non_simplified = [t for t in result if t in original_set]

    ordered: list[tuple[int, str]] = []
    simplified_iter = iter(simplified)
    non_simplified_iter = iter(non_simplified)
    simplified_placed = False

    for i, _term in enumerate(original):
        if i in used:
            if not simplified_placed:
                s = next(simplified_iter, None)
                if s is not None:
                    ordered.append(s)
                simplified_placed = True
        else:
            ns = next(non_simplified_iter, None)
            if ns is not None:
                ordered.append(ns)

    for s in simplified_iter:
        ordered.append(s)
    for ns in non_simplified_iter:
        ordered.append(ns)

    return ordered
