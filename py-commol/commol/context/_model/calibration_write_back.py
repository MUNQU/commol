"""Write calibrated initial conditions back onto a model definition."""

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from commol.context._model.engine_state import (
    calibrated_initial_state,
    initial_state,
)
from commol.context.constants import CalibrationParameterType
from commol.context.initial_conditions import CATEGORY_SEPARATOR, InitialConditions

if TYPE_CHECKING:
    from commol.context.calibration import CalibrationProblem
    from commol.context.compartmental import CompartmentalStructure
    from commol.context.model import Model

# Relative tolerance under which a rebuilt amount matches the scored one.
_MATCH_TOLERANCE = 1e-9


def with_initial_conditions(
    structure: "CompartmentalStructure", initial_conditions: InitialConditions
) -> "CompartmentalStructure":
    """Shallow copy of a structure that holds other initial conditions."""
    return structure.model_copy(update={"initial_conditions": initial_conditions})


def calibrated_initial_conditions[S: "CompartmentalStructure"](
    model: "Model[S]",
    problem: "CalibrationProblem",
    values: Mapping[str, float],
) -> InitialConditions:
    """
    Initial conditions that rebuild the state the engine scored.

    The engine computes the calibrated initial state. The declared
    fractions are first updated so the definition stays readable: each
    calibrated bin, and the remainder when it is a bin, takes the share of the
    total the engine gave it, and each calibrated category takes the
    split the engine applied. Every compartment whose rebuilt amount still
    differs from the scored one is then pinned as a per-compartment override.
    The result therefore rebuilds the scored state exactly, whatever the
    combination of bin, category and exact-compartment values.

    Parameters
    ----------
    model : Model[CompartmentalStructure]
        The model the calibration problem was built on. It is not modified.
    problem : CalibrationProblem
        The calibration problem, for its parameters and remainder.
    values : Mapping[str, float]
        Calibrated value per parameter id.

    Returns
    -------
    InitialConditions
        A new initial-conditions block.

    Raises
    ------
    ValueError
        If the engine cannot resolve an initial-condition id or the remainder,
        or the initial amounts exceed the total.
    """
    structure = model.structure
    current = structure.initial_conditions
    parameters = [
        param
        for param in problem.parameters
        if param.parameter_type == CalibrationParameterType.INITIAL_CONDITION
        and param.id in values
    ]
    if not parameters:
        return current.model_copy(deep=True)

    scored = calibrated_initial_state(
        model,
        parameters,
        [values[param.id] for param in parameters],
        problem.initial_condition_remainder,
    )

    candidate = _declared_fraction_update(
        structure, scored.values, scored.state, scored.covered
    )
    try:
        rebuilt = initial_state(with_initial_conditions(structure, candidate))
    except ValueError:
        # The declared updates combined with the existing overrides do not
        # build; pin the scored state on the current definition instead.
        candidate = current.model_copy(deep=True)
        rebuilt = initial_state(with_initial_conditions(structure, candidate))

    total = current.total
    target = scored.state
    pins = {
        compartment.name: min(1.0, max(0.0, target[compartment.name] / total))
        for compartment in rebuilt
        if not math.isclose(
            compartment.amount,
            target[compartment.name],
            rel_tol=_MATCH_TOLERANCE,
            abs_tol=_MATCH_TOLERANCE * total,
        )
    }
    if pins:
        candidate.update_compartment_fractions(pins)
    return candidate


def _declared_fraction_update(
    structure: "CompartmentalStructure",
    applied: Mapping[str, float],
    scored: Mapping[str, float],
    covered: Sequence[str],
) -> InitialConditions:
    """
    Copy of the initial conditions with bin and category values written back.

    Every bin a calibrated value or the remainder covers takes the share of the
    total the engine gave it; every calibrated category takes the split
    the engine applied. A group of updates that the definition rejects is
    skipped; the pins that follow cover the compartments it would have set.
    """
    candidate = structure.initial_conditions.model_copy(deep=True)
    total = candidate.total
    bin_ids = {bin_item.id for bin_item in structure.bins}
    categories = candidate.get_categories_with_fractions()
    state = initial_state(structure)

    bin_totals: dict[str, float] = {}
    for compartment in state:
        bin_totals[compartment.bin] = (
            bin_totals.get(compartment.bin, 0.0) + scored[compartment.name]
        )

    covered_names = set(covered)
    touched_bins = {
        compartment.bin for compartment in state if compartment.name in covered_names
    }
    bin_fractions: dict[str, float | None] = {
        bin_id: bin_totals[bin_id] / total for bin_id in sorted(touched_bins)
    }
    category_fractions = {
        param_id: value
        for param_id, value in applied.items()
        if param_id not in bin_ids
        and (CATEGORY_SEPARATOR in param_id or param_id in categories)
    }

    if bin_fractions:
        try:
            candidate.update_bin_fractions(bin_fractions)
        except ValueError:
            pass
    if category_fractions:
        try:
            candidate.update_stratification_fractions(category_fractions)
        except ValueError:
            pass
    return candidate
