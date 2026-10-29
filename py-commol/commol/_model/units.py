from typing import TYPE_CHECKING

from commol._model import engine_state
from commol._model.formula_tree import Node, formula_text, variables
from commol._model.helpers import ModelCompartmentHelper
from commol._model.unit_checks import (
    DEFAULT_STEP_UNIT,
    UnitConsistencyError,
    check_formula_units,
    ureg,
)

if TYPE_CHECKING:
    from commol._model.model import Model


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

    def __init__(self, model: "Model") -> None:
        self._model = model
        self._helper = ModelCompartmentHelper(model)
        self._mixed_unit_totals = self._helper.mixed_unit_totals()

    def check_unit_consistency(self, verbose: bool) -> None:
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

        bins_without_unit = [b.id for b in self._model.process.bins if not b.unit]
        if bins_without_unit:
            raise UnitConsistencyError(
                f"Cannot perform unit consistency check. The following bins have "
                f"no unit: {', '.join(bins_without_unit)}. Set bin_unit in "
                f"ModelBuilder, or unit in add_bin."
            )

        matrices_without_unit = [
            m.id for m in self._model.process.matrices if not m.unit
        ]
        if matrices_without_unit:
            raise UnitConsistencyError(
                f"Cannot perform unit consistency check. The following matrices "
                f"have no unit: {', '.join(matrices_without_unit)}. Set unit in "
                f"add_matrix."
            )

    def _collect_unit_errors(self, variable_units: dict[str, str]) -> list[str]:
        """
        Collect unit consistency errors from every flow the engine plans.

        The engine plans the flows (``engine_state.transition_flows``), so a
        proportional rate is checked as its flow, the rate times its source,
        and a read of the neighbors of a unit as the sums it is. A flow moves
        the unit of the bin of its source compartment, or of its target
        compartment when it has no source.
        """
        errors = self._scale_unit_errors(variable_units)
        process = self._model.process
        impulses = {t.id for t in process.transitions if t.is_impulse}
        bin_units = {bin_item.id: bin_item.unit for bin_item in process.bins}
        compartment_units = {
            compartment.name: bin_units[compartment.bin]
            for compartment in engine_state.initial_state(
                process, self._model.structure
            )
        }
        checked: set[tuple[str, Node, str]] = set()
        flows = engine_state.transition_flows(process, self._model.structure)
        for flow in flows:
            compartment = flow.source if flow.source is not None else flow.target
            assert compartment is not None, "a flow has a source or a target"
            bin_unit = str(compartment_units[compartment])
            if (flow.transition, flow.term, bin_unit) in checked:
                continue
            checked.add((flow.transition, flow.term, bin_unit))
            error = self._flow_unit_error(
                flow.term,
                flow.transition,
                variable_units,
                bin_unit,
                flow.transition in impulses,
            )
            if error is not None:
                errors.append(error)

        return errors

    def _scale_unit_errors(self, variable_units: dict[str, str]) -> list[str]:
        """
        A scale of a matrix has no unit, so the entries keep the unit of the
        matrix whatever its value.
        """
        errors: list[str] = []
        for matrix in self._model.process.matrices:
            for side, category, parameter in matrix.scales():
                unit = variable_units[parameter]
                if not ureg.parse_units(unit).dimensionless:
                    errors.append(
                        f"Matrix '{matrix.id}' scales the {side} of '{category}' by "
                        f"'{parameter}' in '{unit}'; a scale is dimensionless, so the "
                        f"entries keep the unit of the matrix"
                    )
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

    def _flow_unit_error(
        self,
        rate: Node,
        transition_id: str,
        variable_units: dict[str, str],
        bin_unit: str,
        is_impulse: bool,
    ) -> str | None:
        """
        The error of the units of one flow of a transition, or None.

        A flow moves ``<bin unit>`` per step; an impulse moves ``<bin unit>``
        at once.
        """
        text = formula_text(rate)
        for var in variables(rate):
            if var in self._mixed_unit_totals:
                units = ", ".join(sorted(set(self._helper.bin_units().values())))
                return (
                    f"Transition '{transition_id}': Variable '{var}' in rate "
                    f"'{text}' adds bins of different units ({units}); read the "
                    "bins or the compartments it sums instead"
                )
            if var not in variable_units:
                return (
                    f"Transition '{transition_id}': Variable '{var}' in rate "
                    f"'{text}' has no defined unit"
                )

        expected_unit = bin_unit if is_impulse else f"{bin_unit}/{self._step_unit()}"

        is_consistent, error_msg = check_formula_units(
            rate, variable_units, expected_unit
        )
        if not is_consistent:
            return f"Transition '{transition_id}': {error_msg}"
        return None
