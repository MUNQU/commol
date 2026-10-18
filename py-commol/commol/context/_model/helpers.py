from commol.constants import SUBGROUP_TOTAL_PREFIX, TOTAL_VARIABLE
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
    """Utilities shared by the description of the dynamics and UnitChecker."""

    def __init__(self, model) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Unit helpers
    # ------------------------------------------------------------------

    def has_all_units(self) -> bool:
        """True when every bin and every constant parameter has a unit."""
        return all(b.unit for b in self._model.process.bins) and all(
            p.unit is not None
            for p in self._model.parameters
            if not isinstance(p.value, str)
        )

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

    def bin_units(self) -> dict[str, str]:
        """The unit of each bin; the unit check requires every bin to have one."""
        return {b.id: b.unit for b in self._model.process.bins if b.unit}

    def _common_bin_unit(self) -> str | None:
        """The unit every bin shares, or None when the bins have several."""
        units = set(self.bin_units().values())
        return units.pop() if len(units) == 1 else None

    def mixed_unit_totals(self) -> set[str]:
        """
        The variables that add amounts of every bin: ``total``, the subgroup
        totals ``total_<categories>``, a category and ``neighbors``. With bins
        of different units they have none.
        """
        if self._common_bin_unit() is not None:
            return set()
        process = self._model.process
        return {
            TOTAL_VARIABLE,
            NEIGHBORS_VARIABLE,
            *(
                name
                for name in engine_state.subgroup_total_names(process)
                if name.startswith(SUBGROUP_TOTAL_PREFIX)
            ),
            *(category for s in process.stratifications for category in s.categories),
        }

    def _add_stratification_category_units(
        self, variable_units: dict[str, str]
    ) -> None:
        bin_unit = self._common_bin_unit()
        if bin_unit is None:
            return
        for strat in self._model.process.stratifications:
            for category in strat.categories:
                variable_units[category] = bin_unit

    def _add_compartment_units(self, variable_units: dict[str, str]) -> None:
        bin_units = self.bin_units()
        for compartment in engine_state.initial_state(
            self._model.process, self._model.structure
        ):
            variable_units[compartment.name] = bin_units[compartment.bin]

    def _add_predefined_variable_units(self, variable_units: dict[str, str]) -> None:
        bin_units = self.bin_units()
        bin_unit = self._common_bin_unit()
        # The amounts among the neighbors of a unit are amounts of bins.
        for bin_id, unit in bin_units.items():
            variable_units[f"{NEIGHBORS_PREFIX}{bin_id}"] = unit
        if bin_unit is None:
            return
        variable_units.update(
            get_predefined_variable_units(
                engine_state.subgroup_total_names(self._model.process), bin_unit
            )
        )
        variable_units[NEIGHBORS_VARIABLE] = bin_unit

    def _add_special_variable_units(self, variable_units: dict[str, str]) -> None:
        # `t` is a position on the axis; `step` counts increments.
        variable_units["step"] = "dimensionless"
        variable_units["t"] = self._model.step_unit or DEFAULT_STEP_UNIT
        variable_units["pi"] = "dimensionless"
        variable_units["e"] = "dimensionless"

    def _infer_formula_parameter_units(self, variable_units: dict[str, str]) -> None:
        """
        Infer the unit of each formula parameter from the units it reads.

        A formula whose variables have units but no unit of its own, such as
        ``exp(k)`` with ``k`` in ``1/day``, raises.

        Raises
        ------
        UnitConsistencyError
            If the unit of a formula cannot be inferred from its variables.
        """
        failures: dict[str, Exception] = {}
        # Each round infers at least one formula, or stops.
        while self._try_infer_formula_units(variable_units, failures):
            pass
        for param in self._model.parameters:
            if param.id in failures and param.id not in variable_units:
                raise UnitConsistencyError(
                    f"Cannot infer unit for formula parameter '{param.id}' = "
                    f"'{param.value}': {failures[param.id]}. Please provide an "
                    f"explicit unit for this parameter."
                )

    def _try_infer_formula_units(
        self, variable_units: dict[str, str], failures: dict[str, Exception]
    ) -> bool:
        """Infer the formulas whose variables have units; True if any was."""
        inferred_any = False
        for param in self._model.parameters:
            if param.id in variable_units or not isinstance(param.value, str):
                continue
            try:
                if _infer_single_formula_unit(param, variable_units):
                    inferred_any = True
            except Exception as error:
                failures[param.id] = error
        return inferred_any

    def get_rate_unit(self, rate: str, variable_units: dict[str, str]) -> str | None:
        """Calculate the unit for a transition rate expression."""
        try:
            self.register_custom_units()
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

    def annotate_rate_variables(self, rate: str, variable_units: dict[str, str]) -> str:
        """Annotate variables in a rate expression with their units."""
        from commol.context._model.equations import replace_variables

        try:
            return replace_variables(
                rate,
                rate,
                lambda var: (
                    f"{var}({variable_units[var]})" if var in variable_units else var
                ),
            )
        except Exception:
            return rate

    def format_rate_with_unit(
        self,
        rate: str,
        variable_units: dict[str, str] | None,
        latex: bool = False,
    ) -> str:
        """
        Format a rate expression, with the unit of each variable and of the
        rate when the model has units (`variable_units` is not None).
        """
        if latex:
            from commol.context._model.equations import latex_rate_expression

            latex_rate = latex_rate_expression(rate, variable_units)
            if variable_units is None:
                return latex_rate
            unit = self.get_rate_unit(rate, variable_units)
            return f"{latex_rate} [\\text{{{unit}}}]" if unit else latex_rate

        if variable_units is None:
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
