"""The engine of a model, and what it says about an aggregate model.

The engine is the single source of truth for how bins expand over
stratifications, how fractions become amounts, how per-compartment
overrides apply, and how calibrated values reach the initial state.
This module only asks the engine; it does not reproduce any of those rules.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

from commol.commol_rs import _commol_rs as commol_rs

from commol.context._model.formula_tree import Node, Variable, product, read_tree
from commol.context.engine_json import engine_json

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import EngineProtocol
    from commol.context.calibration import CalibrationParameter
    from commol.context.aggregate import Aggregate
    from commol.context.model import Model
    from commol.context.process import Process


class InitialCompartment(NamedTuple):
    """
    One expanded compartment and the amount it starts with.

    Attributes
    ----------
    name : str
        Expanded compartment name, for example ``A_g1``.
    bin : str
        Bin the compartment expands from.
    categories : dict[str, str]
        Category applied to the compartment, keyed by stratification id. A
        conditional stratification whose conditions the compartment does not
        meet is absent.
    amount : float
        Initial amount, including per-compartment overrides.
    """

    name: str
    bin: str
    categories: dict[str, str]
    amount: float


class ExpandedOutput(NamedTuple):
    """
    A bin or accumulator expanded over the stratifications.

    Attributes
    ----------
    name : str
        Expanded name, for example ``A_g1``.
    source : str
        Bin or accumulator the output expands from.
    categories : dict[str, str]
        Category applied to the output, keyed by stratification id.
    """

    name: str
    source: str
    categories: dict[str, str]


def build_engine(model: "Model | str") -> "EngineProtocol":
    """
    Build the engine that a model names.

    Parameters
    ----------
    model : Model | str
        The model definition, or its JSON.

    Returns
    -------
    EngineProtocol
        The engine of ``model.engine.type``, with its settings.

    Raises
    ------
    RuntimeError
        If the engine cannot compile the model.
    """
    model_json = model if isinstance(model, str) else engine_json(model)
    return commol_rs.engine.Engine(model_json)


def initial_state(
    process: "Process", structure: "Aggregate"
) -> list[InitialCompartment]:
    """
    Expand a process into its compartments and their initial amounts.

    Parameters
    ----------
    process : Process
        The process of the model.
    structure : Aggregate
        The aggregate structure, which says how much starts where.

    Returns
    -------
    list[InitialCompartment]
        One entry per expanded compartment, in engine order.

    Raises
    ------
    ValueError
        If a bin amount or a per-compartment override is not finite or is
        negative, or an override names a compartment that is not part of the
        expansion.
    """
    return [
        InitialCompartment(name, bin_id, dict(categories), amount)
        for name, bin_id, categories, amount in commol_rs.model.initial_state(
            engine_json(process), engine_json(structure)
        )
    ]


def expanded_outputs(
    process: "Process",
) -> tuple[list[ExpandedOutput], list[ExpandedOutput]]:
    """
    Expand the bins and accumulators of a process over its stratifications.

    Parameters
    ----------
    process : Process
        The process of the model.

    Returns
    -------
    tuple[list[ExpandedOutput], list[ExpandedOutput]]
        The expanded compartments and accumulator outputs, in engine order.
    """
    compartments, accumulators = commol_rs.model.expanded_outputs(engine_json(process))
    return (
        [
            ExpandedOutput(name, source, dict(cats))
            for name, source, cats in compartments
        ],
        [
            ExpandedOutput(name, source, dict(cats))
            for name, source, cats in accumulators
        ],
    )


class TransitionFlow(NamedTuple):
    """
    One flow of an amount between two compartments, as the engine runs it.

    Attributes
    ----------
    transition : str
        Id of the transition the flow comes from.
    source : str | None
        Compartment the flow leaves, None for a source-less transition.
    target : str | None
        Compartment the flow enters, None for a target-less transition.
    rate : str
        Rate expression the engine evaluates.
    proportional_to : str | None
        Compartment a proportional rate is multiplied by, or None for an absolute
        flow. The pieces of a transition with several sources (``A + B -> C``)
        share the flow of the first source.
    flow : int
        Index of the flow the piece belongs to. The pieces of one flow move
        together: in a stochastic engine they are one event.
    """

    transition: str
    source: str | None
    target: str | None
    rate: str
    proportional_to: str | None
    flow: int

    @property
    def term(self) -> Node:
        """The tree of the flow: the tree of the rate, times its source if
        proportional."""
        rate = formula_tree(self.rate)
        if self.proportional_to is None:
            return rate
        return product(rate, Variable(self.proportional_to))


def transition_flows(
    process: "Process", structure: "Aggregate"
) -> list[TransitionFlow]:
    """
    Every transition flow of a process, exactly as the engine simulates it.

    Parameters
    ----------
    process : Process
        The process of the model.
    structure : Aggregate
        The aggregate structure, whose neighbors resolve ``neighbors_<bin>``
        and ``neighbors``, whatever the engine of the model.

    Returns
    -------
    list[TransitionFlow]
        One entry per flow piece, in transition order.

    Raises
    ------
    ValueError
        If a transition mixes bins and categories, a category transition has
        no source or no target, or a rate reads the neighbors and the
        structure has none.
    """
    return [
        TransitionFlow(*flow)
        for flow in commol_rs.model.transition_flows(
            engine_json(process), engine_json(structure)
        )
    ]


def check_model_values(
    model: "Model",
    engine: "EngineProtocol",
    parameters: Sequence["CalibrationParameter"],
    remainder: str | None = None,
) -> None:
    """
    Check the parameters and the values of the model kind that a calibration sets.

    The engine sets each ``parameter`` value as the calibration objective
    does, so its message names a parameter the model does not have or one
    that a formula or a series sets. ``factor`` and ``noise`` values belong
    to no model kind. Every other type must be one the kind of the model
    accepts (`ModelStructure.calibration_value_types`), and the engine must
    resolve each value and the remainder as the calibration objective does,
    so its message names the problem: an unknown id, an ambiguous category,
    or a category of a stratification that is not binary.

    Parameters
    ----------
    model : Model
        The model the calibration runs on.
    engine : EngineProtocol
        The engine built from it.
    parameters : Sequence[CalibrationParameter]
        Calibration parameters, in problem order.
    remainder : str | None
        The id that takes what the values leave of the sum the kind keeps,
        such as a compartment or bin of an aggregate model.

    Raises
    ------
    ValueError
        If the engine does not set a parameter, the kind does not accept a
        type of value, or a value or the remainder cannot be resolved; the
        message of the engine names the parameter or the initial condition.
    """
    accepted = model.structure.calibration_value_types()
    generic = {
        "parameter",
        "factor",
        "noise",
    }
    for param in parameters:
        if param.parameter_type not in generic and param.parameter_type not in accepted:
            raise ValueError(
                f"Calibration parameter '{param.id}' is a "
                f"{param.parameter_type} value, which a {model.structure.kind} "
                f"model does not accept. It accepts: "
                f"{sorted(generic | accepted)}"
            )
    commol_rs.calibration.check_model_values(
        engine,
        engine_json(list(parameters)),
        initial_condition_remainder=remainder,
    )


def initial_condition_coverage(
    engine: "EngineProtocol",
    parameters: Sequence["CalibrationParameter"],
    remainder: str | None,
) -> tuple[list[list[str]], list[str]]:
    """
    The compartments each initial-condition value sets, and the remainder's.

    Parameters
    ----------
    engine : EngineProtocol
        The engine built from an aggregate model.
    parameters : Sequence[CalibrationParameter]
        Initial-condition calibration parameters, in problem order.
    remainder : str | None
        The id that takes what the values leave, if any.

    Returns
    -------
    tuple[list[list[str]], list[str]]
        For each parameter, the compartments it sets (none for a category
        split), and the compartments of the remainder (none without one).
    """
    values, remainder_indices = commol_rs.calibration.initial_condition_coverage(
        engine,
        engine_json(list(parameters)),
        initial_condition_remainder=remainder,
    )
    names = engine.compartments
    return (
        [[names[index] for index in indices] for indices in values],
        [names[index] for index in remainder_indices],
    )


class CalibratedInitialState(NamedTuple):
    """
    Initial state the engine scores for a set of calibration values.

    Attributes
    ----------
    values : dict[str, float]
        Value the engine applied per parameter id, clamped to its bounds.
    state : dict[str, float]
        Initial amount per expanded compartment, in engine order.
    covered : list[str]
        Compartments whose amount a calibrated bin or compartment value, or
        the remainder, sets.
    """

    values: dict[str, float]
    state: dict[str, float]
    covered: list[str]


def calibrated_initial_state[S: "Aggregate"](
    model: "Model[S]",
    parameters: Sequence["CalibrationParameter"],
    values: Sequence[float],
    remainder: str | None = None,
) -> CalibratedInitialState:
    """
    Initial state the engine scores for a set of calibration values.

    The engine applies the values as the calibration objective does: clamped
    to their bounds, written as bins, then category splits, then exact
    compartments, with the remainder taking what the others leave of the
    amounts the model states.

    Parameters
    ----------
    model : Model[Aggregate]
        The model definition the calibration problem was built on.
    parameters : Sequence[CalibrationParameter]
        Calibration parameters, in problem order.
    values : Sequence[float]
        Value of each parameter, parallel to ``parameters``.
    remainder : str | None
        Compartment or bin that takes what the other compartments leave.

    Returns
    -------
    CalibratedInitialState
        The values the engine applied and the state it built.

    Raises
    ------
    ValueError
        If an initial-condition id or the remainder cannot be resolved, or the
        values leave the remainder below zero.
    """
    engine = build_engine(model)
    applied, state, covered = commol_rs.calibration.calibrated_initial_state(
        engine,
        engine_json(list(parameters)),
        list(values),
        initial_condition_remainder=remainder,
    )
    return CalibratedInitialState(
        values=dict(zip((param.id for param in parameters), applied, strict=True)),
        state=dict(zip(engine.compartments, state, strict=True)),
        covered=[engine.compartments[index] for index in covered],
    )


def check_model(model: "Model") -> None:
    """
    Check an aggregate model in the engine.

    The engine checks the variant and the neighbors of the structure, the
    variables the rates read, and the names the model uses against the names
    the engine provides.

    Parameters
    ----------
    model : Model
        An aggregate model.

    Raises
    ------
    ValueError
        If the variant or the neighbors do not fit the process, a parameter
        id is an expanded compartment, a subgroup total or a name of the
        neighbors, a rate or formula parameter is not in the language of
        formulas, uses a ``total_…`` name that is not a subgroup total of the
        model or reads a variable the model does not define, or a rate reads
        neighbors the model cannot resolve.
    """
    commol_rs.model.check_model(engine_json(model))


def subgroup_total_names(process: "Process") -> list[str]:
    """
    Names of the subgroup totals a rate of a process may read.

    They are the ``total_<categories>`` names, the bin totals and the partial
    bin totals, as the engine provides them.

    Parameters
    ----------
    process : Process
        The process of the model.

    Returns
    -------
    list[str]
        The names, sorted. A process without stratifications has none.
    """
    return commol_rs.model.subgroup_total_names(engine_json(process))


@lru_cache(maxsize=1024)
def formula_tree(formula: str) -> Node:
    """
    The tree of a formula, as the parser of the engine reads it.

    Parameters
    ----------
    formula : str
        A formula of the language, such as a rate.

    Returns
    -------
    Node
        The root of its tree (``formula_tree.py``), the one the engine
        compiles. A tree is immutable, so one formula has one tree.

    Raises
    ------
    ValueError
        If the formula is not in the language of formulas.
    """
    return read_tree(commol_rs.core.MathExpression(formula).tree())
