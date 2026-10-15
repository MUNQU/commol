from commol.context._model.helpers import ModelCompartmentHelper
from commol.utils.equations import (
    DEFAULT_STEP_UNIT,
    UnitConsistencyError,
    check_equation_units,
    ureg,
)
from commol.utils.security import get_expression_variables


def check_step_unit(unit: str) -> str:
    """
    Check that a declared step unit is one unit of one dimension.

    The axis of a model is its independent variable, of any kind: time
    (``day``), a length (``meter``) or any other quantity. So any single
    dimension is accepted, with exponent 1: ``1/day`` and ``m**2`` are
    not lengths of a step.

    Parameters
    ----------
    unit : str
        The declared step unit.

    Returns
    -------
    str
        The unit, unchanged.

    Raises
    ------
    UnitConsistencyError
        If the unit is not defined, or does not have exactly one dimension
        with exponent 1.
    """
    try:
        dimensionality = ureg.parse_units(unit).dimensionality
    except Exception as error:
        raise UnitConsistencyError(
            f"The step unit '{unit}' is not a unit: {error}"
        ) from error
    exponents = list(dict(dimensionality).values())
    if exponents != [1]:
        raise UnitConsistencyError(
            f"The step unit '{unit}' must have one dimension with exponent 1 "
            f"(such as 'day' or 'meter'), got '{dimensionality}'."
        )
    return unit


class UnitChecker:
    """
    Validates unit consistency of model equations.

    The engine converts no value between units. So the check uses one step
    unit, the declared ``step_unit`` of the model or else ``step``, and
    requires every flow to be exactly ``<bin unit>/<step unit>`` and every
    impulse to be ``<bin unit>``. What the value of a flow means (an amount
    per step or an instantaneous rate) is the engine's; the unit is the same.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._helper = ModelCompartmentHelper(model)

    def check_unit_consistency(self, verbose: bool = False) -> None:
        """Check unit consistency of all equations in the model."""
        self._validate_unit_check_preconditions()
        self._step_unit()

        variable_units = self._helper.build_variable_units()

        errors = self._collect_unit_errors(variable_units)

        if errors:
            error_message = "Unit consistency check failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            if self._model.step_unit is None:
                error_message += (
                    f"\nThe model declares no step unit, so it is "
                    f"'{DEFAULT_STEP_UNIT}'. Declare step_unit in ModelBuilder "
                    f"(such as step_unit='day') when the rates use another unit."
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

        bins_without_unit = [b.id for b in self._model.structure.bins if not b.unit]
        if bins_without_unit:
            raise UnitConsistencyError(
                f"Cannot perform unit consistency check. The following bins have "
                f"no unit: {', '.join(bins_without_unit)}. Set bin_unit in "
                f"ModelBuilder, or unit in add_bin."
            )

    def _collect_unit_errors(self, variable_units: dict[str, str]) -> list[str]:
        """Collect unit consistency errors from all transitions."""
        errors: list[str] = []

        for transition in self._model.structure.transitions:
            if transition.rate:
                is_consistent, error_msg = self._check_transition_rate_units(
                    transition.rate,
                    transition.id,
                    variable_units,
                    transition.is_impulse,
                )
                if not is_consistent and error_msg:
                    errors.append(error_msg)

            if transition.stratified_rates:
                for idx, strat_rate in enumerate(transition.stratified_rates):
                    is_consistent, error_msg = self._check_transition_rate_units(
                        strat_rate.rate,
                        f"{transition.id} (stratified rate {idx + 1})",
                        variable_units,
                        transition.is_impulse,
                    )
                    if not is_consistent and error_msg:
                        errors.append(error_msg)

        return errors

    def _step_unit(self) -> str:
        """
        The step unit: the declared ``step_unit`` of the model, else ``step``.

        Raises
        ------
        UnitConsistencyError
            If the declared step unit is not one unit of one dimension.
        """
        if self._model.step_unit is not None:
            return check_step_unit(self._model.step_unit)
        return DEFAULT_STEP_UNIT

    def _check_transition_rate_units(
        self,
        rate: str,
        transition_id: str,
        variable_units: dict[str, str],
        is_impulse: bool = False,
    ) -> tuple[bool, str | None]:
        """
        Check units for a single transition rate.

        A flow moves ``<bin unit>`` per step; an impulse moves ``<bin unit>``
        at once.
        """
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
            bin_unit = self._model.structure.bins[0].unit

        expected_unit = (
            str(bin_unit) if is_impulse else f"{bin_unit}/{self._step_unit()}"
        )

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
        for transition in self._model.structure.transitions:
            if transition.id == transition_id:
                bin_ids = transition.source + transition.target

                for bin_id in bin_ids:
                    for bin_obj in self._model.structure.bins:
                        if bin_obj.id == bin_id and bin_obj.unit:
                            return bin_obj.unit
                break

        return None
