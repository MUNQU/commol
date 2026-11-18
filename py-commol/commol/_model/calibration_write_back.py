"""Write calibrated initial conditions back onto a model definition."""

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

from commol._model.engine_state import (
    InitialCompartment,
    calibrated_initial_state,
    initial_state,
    write_back,
)
from commol._model.initial_conditions import ScoredAmount

if TYPE_CHECKING:
    from commol._model.aggregate import Aggregate
    from commol._model.model import Model
    from commol._problem.problem import CalibrationProblem


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
    calibrated category takes the split the engine applied. The scored
    amounts of an earlier write-back are dropped. Every compartment whose
    rebuilt amount still differs from the scored one in any bit, sign
    included, as ``-0.0`` from ``0.0``, is then written: a rounding to whole
    units or a rate that steps on a start value (``if``, a comparison,
    ``floor``, ``1 / x``) can turn on its last bit. The engine sorts each one
    (`engine_state.write_back`): a scored amount when its expansion lies
    within the last bits of the scored amount, it has no override and its
    bin has an amount, which the build drops once an edit changes the
    compartment; otherwise a per-compartment override, such as an
    exact-compartment value or a compartment the user overrides. The result
    therefore rebuilds the scored state bit for bit on every engine,
    whatever the combination of bin, category and exact-compartment values.

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
        if param.parameter_type == "initial_condition" and param.id in values
    ]
    if not parameters:
        return structure.model_copy(deep=True)

    scored = calibrated_initial_state(
        model,
        parameters,
        [values[param.id] for param in parameters],
        problem.initial_condition_remainder,
    )

    candidate = _declared_update(
        model,
        {param_id: scored.values[param_id] for param_id in scored.categories},
        scored.state,
        scored.covered,
    )
    candidate.scored_amounts = {}
    written = write_back(process, candidate, scored.state)
    if written.compartment_amounts:
        candidate._upsert_compartment_amounts(written.compartment_amounts)
    candidate.scored_amounts = {
        name: ScoredAmount(built=built, scored=amount)
        for name, (built, amount) in written.scored_amounts.items()
    }
    return candidate


def _declared_update[S: "Aggregate"](
    model: "Model[S]",
    category_fractions: Mapping[str, float],
    scored: Mapping[str, float],
    covered: Sequence[str],
) -> S:
    """
    Copy of the structure with bin and category values written back.

    ``category_fractions`` holds the applied value of each id that the engine
    resolves as the split of a category.

    The overrides of the compartments that a value or the remainder sets are
    dropped, so the overrides that follow are only those of this write-back,
    not those of an earlier one. Every bin whose every compartment a
    calibrated value or the remainder sets takes the amount the engine gave
    it; a bin of which a value sets only some compartments keeps its amount,
    and the overrides that follow cover those compartments. Every calibrated
    category takes the split the engine applied. A group of category updates
    that the definition rejects is skipped; the overrides that follow cover
    the compartments it would have set.
    """
    covered_names = set(covered)
    candidate = model.structure.model_copy(deep=True)
    candidate.compartment_amounts = {
        name: amount
        for name, amount in candidate.compartment_amounts.items()
        if name not in covered_names
    }
    state = initial_state(model.process, model.structure)

    touched_bins = covered_bins(state, covered)
    bin_amounts = {
        bin_id: amount
        for bin_id, amount in _bin_amounts(state, scored).items()
        if bin_id in touched_bins
    }

    candidate.update_bin_amounts(bin_amounts)
    if category_fractions:
        try:
            candidate.update_stratification_fractions(category_fractions)
        except ValueError:
            pass
    return candidate


def covered_bins(
    compartments: Sequence[InitialCompartment], covered: Collection[str]
) -> set[str]:
    """
    The bins whose every compartment is in `covered`.

    With the compartments that the calibrated values and the remainder set,
    these are the bins whose amount the write-back writes; a value that sets
    only part of a bin leaves its amount and overrides its compartments.

    Parameters
    ----------
    compartments : Sequence[InitialCompartment]
        Every compartment of the model, with its bin.
    covered : Collection[str]
        The names of the covered compartments.

    Returns
    -------
    set[str]
        The ids of the bins that `covered` covers whole.
    """
    names = set(covered)
    partly_covered = {c.bin for c in compartments if c.name not in names}
    return {c.bin for c in compartments} - partly_covered


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
