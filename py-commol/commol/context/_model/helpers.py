import re

from commol.constants import PrintEquationsOutputFormat
from commol.context._model import engine_state
from commol.context.neighbors import NEIGHBORS_PREFIX, NEIGHBORS_VARIABLE
from commol.utils.equations import (
    DEFAULT_STEP_UNIT,
    UnitConsistencyError,
    define_units,
    get_predefined_variable_units,
)
from commol.utils.security import get_expression_variables


class ModelCompartmentHelper:
    """Utilities shared by EquationPrinter and UnitChecker."""

    def __init__(self, model) -> None:
        self._model = model
        self._expanded: (
            tuple[list[tuple[str, dict[str, str]]], list[tuple[str, dict[str, str]]]]
            | None
        ) = None

    # ------------------------------------------------------------------
    # Compartment generation
    # ------------------------------------------------------------------

    def _expanded_outputs(
        self,
    ) -> tuple[list[tuple[str, dict[str, str]]], list[tuple[str, dict[str, str]]]]:
        """Compartments and accumulator outputs, as the engine expands them."""
        if self._expanded is None:
            compartments, accumulators = engine_state.expanded_outputs(
                self._model.process
            )
            self._expanded = (
                [(output.source, output.categories) for output in compartments],
                [(output.source, output.categories) for output in accumulators],
            )
        return self._expanded

    def generate_compartments(self) -> list[tuple[str, dict[str, str]]]:
        """Generate all compartment combinations from bins and stratifications.

        Each element is a 2-tuple ``(bin_id, applied_categories)`` where
        ``applied_categories`` maps each stratification ID that was actually
        applied to this compartment to its chosen category. The engine expands
        them.
        """
        return self._expanded_outputs()[0]

    def generate_accumulator_outputs(self) -> list[tuple[str, dict[str, str]]]:
        """Generate all accumulator output combinations.

        Each element is a 2-tuple ``(accumulator_id, applied_categories)``,
        following the same expansion as compartments.
        """
        return self._expanded_outputs()[1]

    def compartment_to_string(
        self, compartment: tuple[str, dict[str, str]], format: str
    ) -> str:
        """Convert compartment tuple to display string."""
        bin_id, applied = compartment
        parts = [bin_id] + [
            applied[s.id]
            for s in self._model.process.stratifications
            if s.id in applied
        ]
        name = "_".join(parts)

        if format == PrintEquationsOutputFormat.LATEX:
            from commol.context._model.equations import latex_variable

            return latex_variable(name)
        return name

    # ------------------------------------------------------------------
    # Unit helpers
    # ------------------------------------------------------------------

    def has_all_units(self) -> bool:
        """Check if all units are defined. Raises error if partial units."""
        has_any_bin_unit = any(b.unit for b in self._model.process.bins)
        has_any_param_unit = any(
            p.unit for p in self._model.parameters if not isinstance(p.value, str)
        )

        if not has_any_bin_unit and not has_any_param_unit:
            return False

        if not all(b.unit for b in self._model.process.bins):
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
        for state in self._model.process.bins:
            if state.unit:
                variable_units[state.id] = state.unit

    def _add_stratification_category_units(
        self, variable_units: dict[str, str]
    ) -> None:
        if not self._model.process.bins or not self._model.process.bins[0].unit:
            return
        bin_unit = self._model.process.bins[0].unit
        for strat in self._model.process.stratifications:
            for category in strat.categories:
                variable_units[category] = bin_unit

    def _add_compartment_units(self, variable_units: dict[str, str]) -> None:
        if not self._model.process.stratifications:
            return
        compartments = self.generate_compartments()
        for compartment in compartments:
            compartment_str = self.compartment_to_string(
                compartment, format=PrintEquationsOutputFormat.TEXT
            )
            bin_id = compartment[0]
            bin_obj = next(
                (b for b in self._model.process.bins if b.id == bin_id), None
            )
            if bin_obj and bin_obj.unit:
                variable_units[compartment_str] = bin_obj.unit

    def _add_predefined_variable_units(self, variable_units: dict[str, str]) -> None:
        bin_unit = (
            self._model.process.bins[0].unit if self._model.process.bins else None
        )
        predefined_units = get_predefined_variable_units(
            engine_state.subgroup_total_names(self._model.process), bin_unit
        )
        variable_units.update(predefined_units)
        if bin_unit is not None:
            # The amounts among the neighbors of a unit are amounts of bins.
            variable_units[NEIGHBORS_VARIABLE] = bin_unit
            for bin_item in self._model.process.bins:
                variable_units[f"{NEIGHBORS_PREFIX}{bin_item.id}"] = (
                    bin_item.unit or bin_unit
                )

    def _add_special_variable_units(self, variable_units: dict[str, str]) -> None:
        # `t` is a position on the axis; `step` counts increments.
        variable_units["step"] = "dimensionless"
        variable_units["t"] = self._model.step_unit or DEFAULT_STEP_UNIT
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
        """Define the units of the bins, parameters and step that pint lacks."""
        units = [b.unit for b in self._model.process.bins if b.unit]
        units += [p.unit for p in self._model.parameters if p.unit]
        if self._model.step_unit:
            units.append(self._model.step_unit)
        for unit in units:
            define_units(unit)


# ------------------------------------------------------------------
# Module-level helpers (no model dependency)
# ------------------------------------------------------------------


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
