from commol.constants import ModelTypes
from commol.context._model.helpers import ModelCompartmentHelper
from commol.utils.equations import (
    UnitConsistencyError,
    check_equation_units,
    ureg,
)
from commol.utils.security import get_expression_variables


class UnitChecker:
    """Validates unit consistency of model equations."""

    def __init__(self, model) -> None:
        self._model = model
        self._helper = ModelCompartmentHelper(model)

    def check_unit_consistency(self, verbose: bool = False) -> None:
        """Check unit consistency of all equations in the model."""
        self._validate_unit_check_preconditions()

        variable_units = self._helper.build_variable_units()

        errors = self._collect_unit_errors(variable_units)

        if errors:
            error_message = "Unit consistency check failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise UnitConsistencyError(error_message)

        if verbose:
            print("Unit consistency check passed successfully.")

    def _validate_unit_check_preconditions(self) -> None:
        """Validate preconditions for unit consistency checking."""
        self._helper.register_custom_units()
        non_formula_params_missing_units = [
            p
            for p in self._model.parameters
            if p.unit is None and not isinstance(p.value, str)
        ]

        if non_formula_params_missing_units:
            param_names = ", ".join([p.id for p in non_formula_params_missing_units])
            raise UnitConsistencyError(
                f"Cannot perform unit consistency check. The following constant "
                f"parameters are missing units: {param_names}. "
                f"Please specify units for all parameters, or use formulas "
                f"to allow automatic unit inference."
            )

        if self._model.dynamics.typology != ModelTypes.DIFFERENCE_EQUATIONS:
            raise ValueError(
                f"Unit checking is only supported for DifferenceEquations models. "
                f"Current model type: {self._model.dynamics.typology}"
            )

    def _collect_unit_errors(self, variable_units: dict[str, str]) -> list[str]:
        """Collect unit consistency errors from all transitions."""
        errors: list[str] = []

        for transition in self._model.dynamics.transitions:
            if transition.rate:
                is_consistent, error_msg = self._check_transition_rate_units(
                    transition.rate,
                    transition.id,
                    variable_units,
                )
                if not is_consistent and error_msg:
                    errors.append(error_msg)

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

    def _infer_time_unit(self) -> str | None:
        """Infer the time unit used in the model from parameter units."""
        for param in self._model.parameters:
            if param.unit is None:
                continue

            try:
                unit_obj = ureg(param.unit)

                if "[time]" in str(unit_obj.dimensionality):
                    time_dimension = unit_obj.dimensionality.get("[time]", 0)
                    if isinstance(time_dimension, (int, float)) and time_dimension < 0:
                        unit_str = str(unit_obj.units)

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
        """Check units for a single transition rate."""
        variables = get_expression_variables(rate)

        rate_variable_units: dict[str, str] = {}
        for var in variables:
            if var in variable_units:
                rate_variable_units[var] = variable_units[var]
            else:
                return (
                    False,
                    (
                        f"Transition '{transition_id}': Variable '{var}' in rate "
                        f"'{rate}' has no defined unit"
                    ),
                )

        bin_unit = self._get_transition_bin_unit(transition_id)
        if not bin_unit:
            bin_unit = (
                self._model.population.bins[0].unit
                if self._model.population.bins
                else "person"
            )

        time_unit = self._infer_time_unit() or "day"

        expected_unit = f"{bin_unit}/{time_unit}"

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
        """Get the bin unit for a specific transition."""
        for transition in self._model.dynamics.transitions:
            if transition.id == transition_id:
                bin_ids = transition.source + transition.target

                for bin_id in bin_ids:
                    for bin_obj in self._model.population.bins:
                        if bin_obj.id == bin_id and bin_obj.unit:
                            return bin_obj.unit
                break

        return None
