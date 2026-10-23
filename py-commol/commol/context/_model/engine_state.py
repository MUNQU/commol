"""The engine of a model, and what it says about an aggregate model.

The engine is the single source of truth for how bins expand over
stratifications, how fractions become amounts, how per-compartment
overrides apply, and how calibrated values reach the initial state.
This module only asks the engine; it does not reproduce any of those rules.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from commol.commol_rs import _commol_rs as commol_rs

from commol.context.constants import CalibrationParameterType

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import (
        CalibrationConstraintProtocol,
        CalibrationParameterProtocol,
        CalibrationParameterTypeProtocol,
        EngineProtocol,
        LossConfigProtocol,
        ObservationModelProtocol,
        ObservedDataPointProtocol,
        RustAggregateProtocol,
        RustModelProtocol,
        RustProcessProtocol,
    )
    from commol.context.calibration import (
        CalibrationConstraint,
        CalibrationParameter,
        CalibrationProblem,
        ObservedDataPoint,
    )
    from commol.context.aggregate import Aggregate
    from commol.context.model import Model
    from commol.context.observation_models import ObservationModel
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


def _rust_model(model: "Model") -> "RustModelProtocol":
    return commol_rs.core.Model.from_json(model.model_dump_json())


def _rust_process(process: "Process") -> "RustProcessProtocol":
    return commol_rs.core.Process.from_json(process.model_dump_json())


def _rust_structure(structure: "Aggregate") -> "RustAggregateProtocol":
    return commol_rs.core.Aggregate.from_json(structure.model_dump_json())


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
            _rust_process(process), _rust_structure(structure)
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
    compartments, accumulators = commol_rs.model.expanded_outputs(
        _rust_process(process)
    )
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
    def term(self) -> str:
        """The flow as an expression: the rate, times its source if proportional."""
        if self.proportional_to is None:
            return self.rate
        return f"{self.rate} * {self.proportional_to}"


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
            _rust_process(process), _rust_structure(structure)
        )
    ]


def _rust_parameter_type(parameter_type: str) -> "CalibrationParameterTypeProtocol":
    rust_types = commol_rs.calibration.CalibrationParameterType
    return {
        CalibrationParameterType.PARAMETER: rust_types.Parameter,
        CalibrationParameterType.INITIAL_CONDITION: rust_types.InitialCondition,
        CalibrationParameterType.FACTOR: rust_types.Factor,
        CalibrationParameterType.NOISE: rust_types.Noise,
    }[CalibrationParameterType(parameter_type)]


def rust_parameters(
    parameters: Sequence["CalibrationParameter"],
) -> list["CalibrationParameterProtocol"]:
    """The parameters of a calibration problem, as the engine takes them."""
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
            trials=point.trials,
        )
        for point in points
    ]


def rust_observation_models(
    models: Mapping[str, "ObservationModel"],
) -> list[tuple[str, "ObservationModelProtocol"]]:
    """The observation models of a calibration problem, as the engine takes them."""
    from commol.context.observation_models import (
        BinomialObservations,
        LogNormalObservations,
        NegativeBinomialObservations,
        NormalObservations,
        PoissonObservations,
        StudentTObservations,
    )

    rust = commol_rs.calibration.ObservationModel

    def convert(model: "ObservationModel") -> "ObservationModelProtocol":
        match model:
            case NormalObservations():
                return rust.normal(model.sd)
            case LogNormalObservations():
                return rust.log_normal(model.sigma, model.center == "median")
            case StudentTObservations():
                return rust.student_t(model.df, model.scale)
            case PoissonObservations():
                return rust.poisson()
            case NegativeBinomialObservations():
                return rust.negative_binomial(model.overdispersion)
            case BinomialObservations():
                return rust.binomial(model.trials)

    return [(label, convert(model)) for label, model in models.items()]


def rust_constraints(
    constraints: Sequence["CalibrationConstraint"],
) -> list["CalibrationConstraintProtocol"]:
    """The constraints of a calibration problem, as the engine takes them."""
    return [
        commol_rs.calibration.CalibrationConstraint(
            id=constraint.id,
            expression=constraint.expression,
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


def check_model_values(
    model: "Model",
    engine: "EngineProtocol",
    parameters: Sequence["CalibrationParameter"],
    remainder: str | None = None,
) -> None:
    """
    Check the values of the model kind that a calibration sets.

    ``parameter``, ``factor`` and ``noise`` values are not checked here: they
    belong to no model kind. Every other type
    must be one the kind of the model accepts
    (`ModelStructure.calibration_value_types`), and the engine must resolve
    each value and the remainder as the calibration objective does, so its
    message names the problem: an unknown id, an ambiguous category, or a
    category of a stratification that is not binary.

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
        If the kind does not accept a type of value, or a value or the
        remainder cannot be resolved.
    """
    accepted = model.structure.calibration_value_types()
    generic = {
        CalibrationParameterType.PARAMETER,
        CalibrationParameterType.FACTOR,
        CalibrationParameterType.NOISE,
    }
    for param in parameters:
        if param.parameter_type not in generic and param.parameter_type not in accepted:
            raise ValueError(
                f"Calibration parameter '{param.id}' is a "
                f"{param.parameter_type} value, which a {model.structure.kind} "
                f"model does not accept. It accepts: "
                f"{sorted(generic | accepted)}"
            )
    try:
        commol_rs.calibration.check_model_values(
            engine,
            rust_parameters(parameters),
            initial_condition_remainder=remainder,
        )
    except ValueError as error:
        raise ValueError(f"Calibration initial condition: {error}") from error


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
        rust_parameters(parameters),
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
        rust_parameters(parameters),
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
        neighbors, a rate or formula parameter uses a ``total_…`` name that
        is not a subgroup total of the model, or a rate reads neighbors the
        model cannot resolve.
    """
    commol_rs.model.check_model(_rust_model(model))


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
    return commol_rs.model.subgroup_total_names(_rust_process(process))


REPLICATE_SUMMARIES = ('"mean"', '"median"', "Quantile(level)", '"crps"')
"""The options of ``CalibrationProblem.replicate_summary``, as a message lists them."""


def rust_loss_config(
    problem: "CalibrationProblem", loss_function: str, engine_type: str
) -> "LossConfigProtocol":
    """
    The Rust loss of a problem, with the replicates of a stochastic engine.

    Parameters
    ----------
    problem : CalibrationProblem
        The calibration problem.
    loss_function : str
        The problem's loss.
    engine_type : str
        The engine of the model. A stochastic engine runs
        ``problem.replicates`` replicates at each evaluation; a deterministic
        one does not use them.

    Returns
    -------
    LossConfigProtocol
        The loss, its normalization and its replicates.

    Raises
    ------
    ValueError
        If the engine is stochastic and the problem has no ``replicates``, no
        ``replicate_summary`` (the message lists the options) or no ``seed``.
    """
    from commol.context.calibration import Quantile
    from commol.context.constants import LossFunction

    builders = {
        LossFunction.SSE: commol_rs.calibration.LossConfig.sse,
        LossFunction.RMSE: commol_rs.calibration.LossConfig.rmse,
        LossFunction.MAE: commol_rs.calibration.LossConfig.mae,
    }
    builder = builders[LossFunction(loss_function)]
    # Per-series normalization is orthogonal to the chosen metric.
    loss_config = builder().normalized(problem.normalize_observations)
    if not commol_rs.engine.EngineType(engine_type).is_stochastic:
        return loss_config
    if problem.replicate_summary is None:
        raise ValueError(
            f"The {engine_type} engine is stochastic, so the calibration compares "
            "the observations with its replicates: set replicate_summary to one of "
            f"{', '.join(REPLICATE_SUMMARIES)}. There is no default: what to fit "
            "is a modelling choice."
        )
    if problem.replicates is None:
        raise ValueError(
            f"The {engine_type} engine is stochastic: set replicates, how many "
            "replicates each evaluation runs."
        )
    if problem.seed is None:
        raise ValueError(
            f"The {engine_type} engine is stochastic: set seed, so the replicates "
            "and the calibration are reproducible."
        )
    summary = problem.replicate_summary
    if isinstance(summary, Quantile):
        return loss_config.with_replicates(
            problem.replicates, problem.seed, "quantile", summary.level
        )
    return loss_config.with_replicates(problem.replicates, problem.seed, summary)
