import re

import pint

from commol.constants import PrintEquationsOutputFormat
from commol.context.dynamics import Transition
from commol.utils.equations import (
    UnitConsistencyError,
    get_predefined_variable_units,
    ureg,
)
from commol.utils.security import get_expression_variables


class ModelCompartmentHelper:
    """Utilities shared by EquationPrinter and UnitChecker."""

    def __init__(self, model) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Compartment generation
    # ------------------------------------------------------------------

    def expand_over_stratifications(
        self, source_ids: list[str]
    ) -> list[tuple[str, dict[str, str]]]:
        """Expand ids across the stratifications that apply to each of them.

        Each element is a 2-tuple ``(source_id, applied_categories)`` where
        ``applied_categories`` maps each stratification ID that was actually
        applied to its chosen category. A conditional stratification is applied
        only to ids whose already-applied categories satisfy its conditions.
        """
        partials: list[tuple[str, dict[str, str]]] = [
            (source_id, {}) for source_id in source_ids
        ]

        for strat in self._model.population.stratifications:
            new_partials: list[tuple[str, dict[str, str]]] = []
            for name, applied in partials:
                conditions_met = strat.conditions is None or all(
                    applied.get(c.stratification) == c.category
                    for c in strat.conditions
                )
                if conditions_met:
                    for cat in strat.categories:
                        new_partials.append((name, {**applied, strat.id: cat}))
                else:
                    new_partials.append((name, applied))
            partials = new_partials

        return partials

    def generate_compartments(self) -> list[tuple[str, dict[str, str]]]:
        """Generate all compartment combinations from bins and stratifications.

        Each element is a 2-tuple ``(bin_id, applied_categories)`` where
        ``applied_categories`` maps each stratification ID that was actually
        applied to this compartment to its chosen category.
        """
        return self.expand_over_stratifications(
            [state.id for state in self._model.population.bins]
        )

    def generate_accumulator_outputs(self) -> list[tuple[str, dict[str, str]]]:
        """Generate all accumulator output combinations.

        Each element is a 2-tuple ``(accumulator_id, applied_categories)``,
        following the same expansion as compartments.
        """
        return self.expand_over_stratifications(
            [accumulator.id for accumulator in self._model.population.accumulators]
        )

    def compartment_to_string(
        self, compartment: tuple[str, dict[str, str]], format: str
    ) -> str:
        """Convert compartment tuple to display string."""
        bin_id, applied = compartment
        parts = [bin_id] + [
            applied[s.id]
            for s in self._model.population.stratifications
            if s.id in applied
        ]
        name = "_".join(parts)

        if format == PrintEquationsOutputFormat.LATEX:
            from commol.context._model.equations import latex_variable

            return latex_variable(name)
        return name

    # ------------------------------------------------------------------
    # Transition classification
    # ------------------------------------------------------------------

    def separate_transitions_by_type(
        self,
    ) -> tuple[list[Transition], list[Transition], list[Transition]]:
        """Separate transitions into bin, stratification, and cross-category."""
        bin_id_set = {bin_item.id for bin_item in self._model.population.bins}

        bin_transitions: list[Transition] = []
        stratification_transitions: list[Transition] = []
        cross_category_transitions: list[Transition] = []

        for transition in self._model.dynamics.transitions:
            if transition_has_category_overrides(transition):
                cross_category_transitions.append(transition)
            else:
                transition_ids = set(transition.source) | set(transition.target)
                if transition_ids.issubset(bin_id_set):
                    bin_transitions.append(transition)
                else:
                    stratification_transitions.append(transition)

        return bin_transitions, stratification_transitions, cross_category_transitions

    def group_transitions_by_stratification(
        self, transitions: list[Transition]
    ) -> dict[str, list[Transition]]:
        """Group stratification transitions by their stratification ID."""
        strat_by_id: dict[str, list[Transition]] = {}
        for strat in self._model.population.stratifications:
            strat_by_id[strat.id] = []
            for transition in transitions:
                transition_states = set(transition.source) | set(transition.target)
                if transition_states.issubset(set(strat.categories)):
                    strat_by_id[strat.id].append(transition)
        return strat_by_id

    # ------------------------------------------------------------------
    # Unit helpers
    # ------------------------------------------------------------------

    def has_all_units(self) -> bool:
        """Check if all units are defined. Raises error if partial units."""
        has_any_bin_unit = any(b.unit for b in self._model.population.bins)
        has_any_param_unit = any(
            p.unit for p in self._model.parameters if not isinstance(p.value, str)
        )

        if not has_any_bin_unit and not has_any_param_unit:
            return False

        if not all(b.unit for b in self._model.population.bins):
            raise ValueError("Some bins have units but not all")
        if any(
            p.unit is None
            for p in self._model.parameters
            if not isinstance(p.value, str)
        ):
            raise ValueError("Some parameters have units but not all")

        return True

    def build_variable_units(self) -> dict[str, str]:
        """Build a mapping of all variables to their units."""
        variable_units: dict[str, str] = {}
        self._add_base_variable_units(variable_units)
        self._infer_formula_parameter_units(variable_units)
        return variable_units

    def _add_base_variable_units(self, variable_units: dict[str, str]) -> None:
        self._add_parameter_units(variable_units)
        self._add_bin_units(variable_units)
        self._add_stratification_category_units(variable_units)
        self._add_compartment_units(variable_units)
        self._add_predefined_variable_units(variable_units)
        self._add_special_variable_units(variable_units)

    def _add_parameter_units(self, variable_units: dict[str, str]) -> None:
        for param in self._model.parameters:
            if param.unit:
                variable_units[param.id] = param.unit

    def _add_bin_units(self, variable_units: dict[str, str]) -> None:
        for state in self._model.population.bins:
            if state.unit:
                variable_units[state.id] = state.unit

    def _add_stratification_category_units(
        self, variable_units: dict[str, str]
    ) -> None:
        if not self._model.population.bins or not self._model.population.bins[0].unit:
            return
        bin_unit = self._model.population.bins[0].unit
        for strat in self._model.population.stratifications:
            for category in strat.categories:
                variable_units[category] = bin_unit

    def _add_compartment_units(self, variable_units: dict[str, str]) -> None:
        if not self._model.population.stratifications:
            return
        compartments = self.generate_compartments()
        for compartment in compartments:
            compartment_str = self.compartment_to_string(
                compartment, format=PrintEquationsOutputFormat.TEXT
            )
            bin_id = compartment[0]
            bin_obj = next(
                (b for b in self._model.population.bins if b.id == bin_id), None
            )
            if bin_obj and bin_obj.unit:
                variable_units[compartment_str] = bin_obj.unit

    def _add_predefined_variable_units(self, variable_units: dict[str, str]) -> None:
        bin_unit = (
            self._model.population.bins[0].unit if self._model.population.bins else None
        )
        predefined_units = get_predefined_variable_units(
            self._model.population.stratifications, bin_unit
        )
        variable_units.update(predefined_units)

    @staticmethod
    def _add_special_variable_units(variable_units: dict[str, str]) -> None:
        variable_units["step"] = "dimensionless"
        variable_units["t"] = "dimensionless"
        variable_units["pi"] = "dimensionless"
        variable_units["e"] = "dimensionless"

    def _infer_formula_parameter_units(self, variable_units: dict[str, str]) -> None:
        max_iterations = 10
        formula_params_without_units: list = []

        for _ in range(max_iterations):
            inferred_any = self._try_infer_formula_units(
                variable_units, formula_params_without_units
            )
            if not inferred_any:
                break

        _validate_formula_parameter_units(variable_units, formula_params_without_units)

    def _try_infer_formula_units(
        self,
        variable_units: dict[str, str],
        failed_params: list,
    ) -> bool:
        inferred_any = False
        for param in self._model.parameters:
            if param.id in variable_units or not isinstance(param.value, str):
                continue
            try:
                if _infer_single_formula_unit(param, variable_units):
                    inferred_any = True
            except Exception:
                if param not in failed_params:
                    failed_params.append(param)
        return inferred_any

    def get_rate_unit(
        self, rate: str, variable_units: dict[str, str] | None = None
    ) -> str | None:
        """Calculate the unit for a transition rate expression."""
        try:
            self.register_custom_units()
            if variable_units is None:
                variable_units = self.build_variable_units()
            variables = get_expression_variables(rate)
            rate_variable_units: dict[str, str] = {}
            for var in variables:
                if var in variable_units:
                    rate_variable_units[var] = variable_units[var]
                else:
                    return None
            from commol.utils.equations import parse_equation_unit

            equation_unit = parse_equation_unit(rate, rate_variable_units)
            return str(equation_unit.units)
        except Exception:
            return None

    def annotate_rate_variables(
        self, rate: str, variable_units: dict[str, str] | None = None
    ) -> str:
        """Annotate variables in a rate expression with their units."""
        if not rate:
            return rate
        try:
            if variable_units is None:
                variable_units = self.build_variable_units()
            variables = get_expression_variables(rate)
            annotated_rate = rate
            sorted_vars = sorted(variables, key=lambda x: len(x), reverse=True)
            for var in sorted_vars:
                if var in variable_units:
                    unit = variable_units[var]
                    pattern = r"\b" + re.escape(var) + r"\b"
                    replacement = f"{var}({unit})"
                    annotated_rate = re.sub(pattern, replacement, annotated_rate)
            return annotated_rate
        except Exception:
            return rate

    def format_rate_with_unit(
        self,
        rate: str | None,
        variable_units: dict[str, str] | None = None,
        show_units: bool = True,
        format: str = PrintEquationsOutputFormat.TEXT,
    ) -> str:
        """Format a rate expression with variable units and final unit annotation."""
        if not rate:
            return (
                "None" if format == PrintEquationsOutputFormat.TEXT else "\\text{None}"
            )

        if format == PrintEquationsOutputFormat.LATEX:
            from commol.context._model.equations import latex_rate_expression

            if not show_units:
                return latex_rate_expression(rate)
            annotated_rate = self.annotate_rate_variables(rate, variable_units)
            unit = self.get_rate_unit(rate, variable_units)
            latex_rate = latex_rate_expression(annotated_rate)
            if unit:
                return f"{latex_rate} [\\text{{{unit}}}]"
            return latex_rate

        if not show_units:
            return rate

        annotated_rate = self.annotate_rate_variables(rate, variable_units)
        unit = self.get_rate_unit(rate, variable_units)
        unit_suffix = f" [{unit}]" if unit else ""
        return f"{annotated_rate}{unit_suffix}"

    def register_custom_units(self) -> None:
        """Register custom units in the pint registry."""
        units_to_register: set[str] = set()
        for bin_item in self._model.population.bins:
            if bin_item.unit:
                units_to_register.add(bin_item.unit)
        for param in self._model.parameters:
            if param.unit:
                unit_parts = re.split(r"[*/\s\(\)]+", param.unit)
                for part in unit_parts:
                    part = part.strip()
                    if part and not part.replace(".", "").replace("-", "").isdigit():
                        units_to_register.add(part)

        known_time_aliases = {
            "decade": "10 * year",
            "century": "100 * year",
            "millennium": "1000 * year",
            "fortnight": "14 * day",
            "biweek": "14 * day",
            "semester": "6 * month",
            "trimester": "3 * month",
            "quarter": "3 * month",
            "bimester": "2 * month",
            "wk": "week",
            "mo": "month",
            "mon": "month",
            "yr": "year",
            "hr": "hour",
            "min": "minute",
            "sec": "second",
            "secs": "second",
            "mins": "minute",
            "hrs": "hour",
            "wks": "week",
            "mons": "month",
            "yrs": "year",
        }

        for unit_name in units_to_register:
            try:
                ureg(unit_name)
            except pint.UndefinedUnitError:
                if unit_name in known_time_aliases:
                    ureg.define(f"{unit_name} = {known_time_aliases[unit_name]}")
                else:
                    dimension_name = f"{unit_name}_dimension"
                    ureg.define(f"{unit_name} = [{dimension_name}]")


# ------------------------------------------------------------------
# Module-level helpers (no model dependency)
# ------------------------------------------------------------------


def replace_bin_in_rate(rate: str, bin_name: str, replacement: str) -> str:
    """Replace a base bin name in a rate expression with word-boundary matching."""
    return re.sub(rf"\b{re.escape(bin_name)}\b", replacement, rate)


def transition_has_category_overrides(transition: Transition) -> bool:
    """Check if any stratified rate has ``to`` overrides."""
    if not transition.stratified_rates:
        return False
    return any(
        c.to is not None for sr in transition.stratified_rates for c in sr.conditions
    )


def _infer_single_formula_unit(param, variable_units: dict[str, str]) -> bool:
    """Infer unit for a single formula parameter. Returns True if successful."""
    if not isinstance(param.value, str):
        return False
    formula_vars = get_expression_variables(param.value)
    formula_var_units: dict[str, str] = {}
    for var in formula_vars:
        if var in variable_units:
            formula_var_units[var] = variable_units[var]
        else:
            return False
    if formula_var_units:
        from commol.utils.equations import parse_equation_unit

        inferred_unit = parse_equation_unit(param.value, formula_var_units)
        variable_units[param.id] = str(inferred_unit.units)
    else:
        variable_units[param.id] = "dimensionless"
    return True


def _validate_formula_parameter_units(
    variable_units: dict[str, str],
    failed_params: list,
) -> None:
    """Validate that all formula parameters have units, raise errors if not."""
    for param in failed_params:
        if param.id not in variable_units:
            if not isinstance(param.value, str):
                raise UnitConsistencyError(
                    f"Cannot infer unit for parameter '{param.id}'. "
                    f"Parameter value is not a formula string. "
                    f"Please provide an explicit unit for this parameter."
                )
            formula_vars = get_expression_variables(param.value)
            missing_vars = [v for v in formula_vars if v not in variable_units]
            if missing_vars:
                raise UnitConsistencyError(
                    f"Cannot infer unit for formula parameter '{param.id}'. "
                    f"Formula '{param.value}' references variables without "
                    f"units: {', '.join(missing_vars)}. "
                    f"Please specify units for all referenced parameters or "
                    f"provide an explicit unit for '{param.id}'."
                )
            else:
                raise UnitConsistencyError(
                    f"Cannot infer unit for formula parameter '{param.id}'. "
                    f"Formula '{param.value}' could not be parsed. "
                    f"Please provide an explicit unit for this parameter."
                )
