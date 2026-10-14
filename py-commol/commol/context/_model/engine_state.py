"""Initial state of a model, as the Rust engine builds it.

The engine is the single source of truth for how bins expand over
stratifications, how fractions become head counts, how per-compartment
overrides apply, and how calibrated values reach the initial population.
This module only asks the engine; it does not reproduce any of those rules.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

try:
    from commol.commol_rs import _commol_rs as commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e

from commol.context.constants import CalibrationParameterType

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import (
        CalibrationConstraintProtocol,
        CalibrationParameterProtocol,
        CalibrationParameterTypeProtocol,
        EngineProtocol,
        ObservedDataPointProtocol,
        RustModelProtocol,
    )
    from commol.context.calibration import (
        CalibrationConstraint,
        CalibrationParameter,
        ObservedDataPoint,
    )
    from commol.context.model import Model


class InitialCompartment(NamedTuple):
    """
    One expanded compartment and the population it starts with.

    Attributes
    ----------
    name : str
        Expanded compartment name, for example ``IH_y60``.
    bin : str
        Bin the compartment expands from.
    categories : dict[str, str]
        Category applied to the compartment, keyed by stratification id. A
        conditional stratification whose conditions the compartment does not
        meet is absent.
    population : float
        Initial head count, including per-compartment overrides.
    """

    name: str
    bin: str
    categories: dict[str, str]
    population: float


class ExpandedOutput(NamedTuple):
    """
    A bin or accumulator expanded over the stratifications.

    Attributes
    ----------
    name : str
        Expanded name, for example ``IH_y60``.
    source : str
        Bin or accumulator the output expands from.
    categories : dict[str, str]
        Category applied to the output, keyed by stratification id.
    """

    name: str
    source: str
    categories: dict[str, str]


def _rust_model(model: "Model") -> "RustModelProtocol":
    return commol_rs.core.Model.from_json(model.model_dump_json())


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
    model_json = model if isinstance(model, str) else model.model_dump_json()
    return commol_rs.engine.Engine(commol_rs.core.Model.from_json(model_json))


def initial_state(model: "Model") -> list[InitialCompartment]:
    """
    Expand a model into its compartments and their initial populations.

    Parameters
    ----------
    model : Model
        The model definition.

    Returns
    -------
    list[InitialCompartment]
        One entry per expanded compartment, in engine order.

    Raises
    ------
    ValueError
        If a per-compartment override names an unknown or repeated compartment,
        holds a fraction outside ``[0.0, 1.0]``, or makes the initial population
        exceed ``population_size``.
    """
    return [
        InitialCompartment(name, bin_id, dict(categories), population)
        for name, bin_id, categories, population in commol_rs.model.initial_state(
            _rust_model(model)
        )
    ]


def expanded_outputs(
    model: "Model",
) -> tuple[list[ExpandedOutput], list[ExpandedOutput]]:
    """
    Expand the bins and accumulators of a model over its stratifications.

    Parameters
    ----------
    model : Model
        The model definition.

    Returns
    -------
    tuple[list[ExpandedOutput], list[ExpandedOutput]]
        The expanded compartments and accumulator outputs, in engine order.
    """
    compartments, accumulators = commol_rs.model.expanded_outputs(_rust_model(model))
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
    One flow of population between two compartments, as the engine runs it.

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
    per_capita_of : str | None
        Compartment a per-capita rate is multiplied by, or None for an absolute
        flow. The pieces of a transition with several sources (``A + B -> C``)
        share the flow of the first source.
    """

    transition: str
    source: str | None
    target: str | None
    rate: str
    per_capita_of: str | None

    @property
    def term(self) -> str:
        """The flow as an expression: the rate, times its base if per capita."""
        if self.per_capita_of is None:
            return self.rate
        return f"{self.rate} * {self.per_capita_of}"


def transition_flows(model: "Model") -> list[TransitionFlow]:
    """
    Every transition flow of a model, exactly as the engine simulates it.

    Parameters
    ----------
    model : Model
        The model definition, whatever its engine.

    Returns
    -------
    list[TransitionFlow]
        One entry per flow piece, in transition order.

    Raises
    ------
    ValueError
        If a transition mixes bins and categories, or a category transition
        has no source or no target.
    """
    return [
        TransitionFlow(*flow)
        for flow in commol_rs.model.transition_flows(_rust_model(model))
    ]


def _rust_parameter_type(parameter_type: str) -> "CalibrationParameterTypeProtocol":
    rust_types = commol_rs.calibration.CalibrationParameterType
    if parameter_type == CalibrationParameterType.PARAMETER:
        return rust_types.Parameter
    if parameter_type == CalibrationParameterType.INITIAL_CONDITION:
        return rust_types.InitialCondition
    if parameter_type == CalibrationParameterType.FACTOR:
        return rust_types.Factor
    raise ValueError(f"Unknown parameter type: {parameter_type}")


def _rust_parameters(
    parameters: Sequence["CalibrationParameter"],
) -> list["CalibrationParameterProtocol"]:
    return [
        commol_rs.calibration.CalibrationParameter(
            id=param.id,
            parameter_type=_rust_parameter_type(param.parameter_type),
            min_bound=param.min_bound,
            max_bound=param.max_bound,
            initial_guess=param.initial_guess,
        )
        for param in parameters
    ]


def rust_observed_data(
    points: Sequence["ObservedDataPoint"],
) -> list["ObservedDataPointProtocol"]:
    """The observations of a calibration problem, as the engine takes them."""
    return [
        commol_rs.calibration.ObservedDataPoint(
            at=point.at,
            label=point.label,
            outputs=point.observed_outputs,
            value=point.value,
            weight=point.weight,
            factor_id=point.factor_id,
            window=point.window,
        )
        for point in points
    ]


def rust_constraints(
    constraints: Sequence["CalibrationConstraint"],
) -> list["CalibrationConstraintProtocol"]:
    """The constraints of a calibration problem, as the engine takes them."""
    return [
        commol_rs.calibration.CalibrationConstraint(
            id=constraint.id,
            expression=constraint.expression,
            description=constraint.description,
            weight=constraint.weight,
            at=constraint.at,
        )
        for constraint in constraints
    ]


def check_positions(
    engine: "EngineProtocol",
    observed_data: Sequence["ObservedDataPoint"],
    constraints: Sequence["CalibrationConstraint"] = (),
) -> None:
    """
    Check every position of a calibration problem with the engine.

    The engine decides which positions its model type accepts, and its
    message names the observation or constraint that holds a rejected one.

    Parameters
    ----------
    engine : EngineProtocol
        The engine the calibration runs on.
    observed_data : Sequence[ObservedDataPoint]
        Observations, at their positions and windows.
    constraints : Sequence[CalibrationConstraint]
        Constraints, at their positions.

    Raises
    ------
    ValueError
        If the engine does not report at a position.
    """
    commol_rs.calibration.check_positions(
        engine, rust_observed_data(observed_data), rust_constraints(constraints)
    )


def check_initial_conditions(
    engine: "EngineProtocol",
    parameters: Sequence["CalibrationParameter"],
    remainder: str | None = None,
) -> None:
    """
    Check that every initial-condition id and the remainder resolve.

    The engine resolves the ids as the calibration objective does, so its
    message names the problem: an unknown id, an ambiguous category, or a
    category of a stratification that is not binary.

    Parameters
    ----------
    engine : EngineProtocol
        The engine the calibration runs on.
    parameters : Sequence[CalibrationParameter]
        Calibration parameters, in problem order.
    remainder : str | None
        Compartment or bin that absorbs the difference to the population size.

    Raises
    ------
    ValueError
        If an initial-condition id or the remainder cannot be resolved.
    """
    try:
        commol_rs.calibration.check_initial_conditions(
            engine,
            _rust_parameters(parameters),
            initial_condition_remainder=remainder,
        )
    except ValueError as error:
        raise ValueError(f"Calibration initial condition: {error}") from error


class CalibratedInitialPopulation(NamedTuple):
    """
    Initial state the engine scores for a set of calibration values.

    Attributes
    ----------
    values : dict[str, float]
        Value the engine applied per parameter id, clamped to its bounds.
    population : dict[str, float]
        Initial head count per expanded compartment, in engine order.
    covered : list[str]
        Compartments whose head count a calibrated bin or compartment value, or
        the remainder, sets.
    """

    values: dict[str, float]
    population: dict[str, float]
    covered: list[str]


def calibrated_initial_population(
    model: "Model",
    parameters: Sequence["CalibrationParameter"],
    values: Sequence[float],
    remainder: str | None = None,
) -> CalibratedInitialPopulation:
    """
    Initial state the engine scores for a set of calibration values.

    The engine applies the values as the calibration objective does: clamped
    to their bounds, written as bins, then category splits, then exact
    compartments, with the remainder absorbing the difference to the
    population size.

    Parameters
    ----------
    model : Model
        The model definition the calibration problem was built on.
    parameters : Sequence[CalibrationParameter]
        Calibration parameters, in problem order.
    values : Sequence[float]
        Value of each parameter, parallel to ``parameters``.
    remainder : str | None
        Compartment or bin that absorbs the difference to the population size.

    Returns
    -------
    CalibratedInitialPopulation
        The values the engine applied and the population it built.

    Raises
    ------
    ValueError
        If an initial-condition id or the remainder cannot be resolved, or the
        initial population exceeds the population size.
    """
    engine = build_engine(model)
    applied, population, covered = commol_rs.calibration.calibrated_initial_population(
        engine,
        _rust_parameters(parameters),
        list(values),
        model.population.initial_conditions.population_size,
        initial_condition_remainder=remainder,
    )
    return CalibratedInitialPopulation(
        values=dict(zip((param.id for param in parameters), applied, strict=True)),
        population=dict(zip(engine.compartments, population, strict=True)),
        covered=[engine.compartments[index] for index in covered],
    )


def check_names(model: "Model") -> None:
    """
    Check the names a model uses against the names the engine provides.

    Parameters
    ----------
    model : Model
        The model definition.

    Raises
    ------
    ValueError
        If a parameter id is an expanded compartment or a subpopulation total,
        or a rate or formula parameter uses an ``N_…`` name that is not a
        subpopulation of the model.
    """
    commol_rs.model.check_names(_rust_model(model))
