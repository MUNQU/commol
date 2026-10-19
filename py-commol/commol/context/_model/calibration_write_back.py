"""Write calibrated initial conditions back onto a model definition."""

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from commol.context._model.engine_state import (
    InitialCompartment,
    calibrated_initial_state,
    initial_state,
)
from commol.context.constants import CalibrationParameterType
from commol.context.initial_conditions import CATEGORY_SEPARATOR

if TYPE_CHECKING:
    from commol.context.aggregate import Aggregate
    from commol.context.calibration import CalibrationProblem
    from commol.context.model import Model

# Relative tolerance under which a rebuilt amount matches the scored one.
_MATCH_TOLERANCE = 1e-9


def calibrated_structure[S: "Aggregate"](
    model: "Model[S]",
    problem: "CalibrationProblem",
    values: Mapping[str, float],
) -> S:
    """
    An aggregate structure that rebuilds the state the engine scored.

    The engine computes the calibrated initial state. The declared amounts
    and splits are first updated so the definition stays readable: each bin
    whose every compartment a calibrated value or the remainder sets takes
    the amount the engine gave it, in the unit of the bin, and each
    calibrated category takes the split the engine applied. Every compartment
    whose rebuilt amount still differs from the scored one is then pinned as
    a per-compartment override. The result therefore rebuilds the scored state
    exactly, whatever the combination of bin, category and exact-compartment
    values.

    Parameters
    ----------
    model : Model[Aggregate]
        The model the calibration problem was built on. It is not modified.
    problem : CalibrationProblem
        The calibration problem, for its parameters and remainder.
    values : Mapping[str, float]
        Calibrated value per parameter id.

    Returns
    -------
    Aggregate
        A copy of the structure of the model with the calibrated initial
        conditions.

    Raises
    ------
    ValueError
        If the engine cannot resolve an initial-condition id or the remainder,
        or the values leave the remainder below zero.
    """
    structure = model.structure
    process = model.process
    parameters = [
        param
        for param in problem.parameters
        if param.parameter_type == CalibrationParameterType.INITIAL_CONDITION
        and param.id in values
    ]
    if not parameters:
        return structure.model_copy(deep=True)

    scored = calibrated_initial_state(
        model,
        parameters,
        [values[param.id] for param in parameters],
        problem.initial_condition_remainder,
    )

    candidate = _declared_update(model, scored.values, scored.state, scored.covered)
    rebuilt = initial_state(process, candidate)
    bin_amounts = _bin_amounts(rebuilt, scored.state)
    target = scored.state
    pins = {
        compartment.name: max(0.0, target[compartment.name])
        for compartment in rebuilt
        if not math.isclose(
            compartment.amount,
            target[compartment.name],
            rel_tol=_MATCH_TOLERANCE,
            abs_tol=_MATCH_TOLERANCE * max(1.0, bin_amounts[compartment.bin]),
        )
    }
    if pins:
        candidate._upsert_compartment_amounts(pins)
    return candidate


def _declared_update[S: "Aggregate"](
    model: "Model[S]",
    applied: Mapping[str, float],
    scored: Mapping[str, float],
    covered: Sequence[str],
) -> S:
    """
    Copy of the structure with bin and category values written back.

    Every bin whose every compartment a calibrated value or the remainder
    sets takes the amount the engine gave it; a bin of which a value sets only
    some compartments keeps its amount, and the pins that follow cover those
    compartments. Every calibrated category takes the split the engine
    applied. A group of category updates that the definition rejects is
    skipped; the pins that follow cover the compartments it would have set.
    """
    candidate = model.structure.model_copy(deep=True)
    bin_ids = {bin_item.id for bin_item in model.process.bins}
    categories = candidate.get_categories_with_fractions()
    state = initial_state(model.process, model.structure)

    covered_names = set(covered)
    partly_covered = {c.bin for c in state if c.name not in covered_names}
    touched_bins = {c.bin for c in state} - partly_covered
    bin_amounts = {
        bin_id: amount
        for bin_id, amount in _bin_amounts(state, scored).items()
        if bin_id in touched_bins
    }
    category_fractions = {
        param_id: value
        for param_id, value in applied.items()
        if param_id not in bin_ids
        and (CATEGORY_SEPARATOR in param_id or param_id in categories)
    }

    candidate.update_bin_amounts(bin_amounts)
    if category_fractions:
        try:
            candidate.update_stratification_fractions(category_fractions)
        except ValueError:
            pass
    return candidate


def _bin_amounts(
    compartments: Sequence[InitialCompartment], scored: Mapping[str, float]
) -> dict[str, float]:
    """The scored amount of each bin: the sum of its compartments."""
    amounts: dict[str, float] = {}
    for compartment in compartments:
        amounts[compartment.bin] = (
            amounts.get(compartment.bin, 0.0) + scored[compartment.name]
        )
    return amounts
