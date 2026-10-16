from typing import ClassVar, Protocol

class BinProtocol(Protocol):
    id: str
    name: str

class StratificationProtocol(Protocol):
    id: str
    categories: list[str]

class StratificationConditionProtocol(Protocol):
    stratification: str
    category: str

class StratifiedRateProtocol(Protocol):
    conditions: list[StratificationConditionProtocol]
    rate: str

class TransitionProtocol(Protocol):
    id: str
    source: list[str]
    target: list[str]
    rate: RateMathExpressionProtocol | None
    stratified_rates: list[StratifiedRateProtocol] | None

class ParameterProtocol(Protocol):
    id: str
    value: float
    description: str | None

class StratificationFractionProtocol(Protocol):
    category: str
    fraction: float

class StratificationFractionsProtocol(Protocol):
    stratification: str
    fractions: list[StratificationFractionProtocol]

class InitialConditionsProtocol(Protocol):
    def __init__(
        self,
        total: float,
        bin_fractions: list[tuple[str, float | None]],
        stratification_fractions: dict[str, dict[str, float]] | None = None,
        compartment_fractions: list[tuple[str, float]] | None = None,
    ) -> None: ...
    total: float
    bin_fractions: list[tuple[str, float | None]]
    stratification_fractions: list[tuple[str, list[tuple[str, float]]]]
    compartment_fractions: list[tuple[str, float]]

class RustProcessProtocol(Protocol):
    """The process of a model: bins, accumulators, stratifications and transitions."""

    def __init__(
        self,
        bins: list[BinProtocol],
        stratifications: list[StratificationProtocol],
        transitions: list[TransitionProtocol] | None = None,
        accumulators: list[object] | None = None,
    ) -> None: ...
    @staticmethod
    def from_json(json_str: str) -> RustProcessProtocol: ...

class RustAggregateProtocol(Protocol):
    """The aggregate structure: amounts per compartment."""

    def __init__(self, initial_conditions: InitialConditionsProtocol) -> None: ...
    @staticmethod
    def from_json(json_str: str) -> RustAggregateProtocol: ...

class RustModelProtocol(Protocol):
    """A model of any kind: its header, process, structure and engine."""

    def __init__(
        self,
        name: str,
        process: RustProcessProtocol,
        structure: RustAggregateProtocol,
        parameters: list[ParameterProtocol],
        engine: str,
        description: str | None = None,
        version: str | None = None,
    ) -> None: ...
    @staticmethod
    def from_json(json_str: str) -> RustModelProtocol: ...

class EngineProtocol(Protocol):
    """The engine a model names, built from the model."""

    def __init__(self, model: RustModelProtocol) -> None: ...
    @property
    def engine_type(self) -> str: ...
    def run(self, num_steps: int) -> tuple[list[float], list[list[float]]]: ...
    def run_at(self, positions: list[float]) -> list[list[float]]: ...
    def check_position(self, position: float) -> None: ...
    def step(self) -> None: ...
    def reset(self) -> None: ...
    def set_seed(self, seed: int) -> None: ...
    def run_replicates(
        self,
        seed: int,
        count: int,
        num_steps: int | None = None,
        positions: list[float] | None = None,
    ) -> tuple[list[float], list[int], list[list[list[float]]]]: ...
    def set_parameter(self, parameter_id: str, value: float) -> None: ...
    def get_parameters(self) -> dict[str, float]: ...
    def current_position(self) -> float: ...
    def set_initial_condition(self, compartment_index: int, value: float) -> None: ...
    @property
    def state(self) -> list[float]: ...
    @property
    def compartments(self) -> list[str]: ...
    @property
    def output_names(self) -> list[str]: ...

class EngineTypeProtocol(Protocol):
    """One entry of the engine's list of engines."""

    def __init__(self, name: str) -> None: ...
    @staticmethod
    def names() -> list[str]: ...
    @property
    def name(self) -> str: ...
    @property
    def kinds(self) -> list[str]: ...
    @property
    def is_stochastic(self) -> bool: ...
    @property
    def rate_meaning(self) -> str: ...
    def rate_from_mean_span(self, span: float) -> float: ...
    def rate_from_probability(self, probability: float, period: float) -> float: ...
    @property
    def dynamics_form(self) -> str: ...
    @property
    def schedule(self) -> str: ...
    def equation_lhs(self, variable: str, latex: bool) -> str | None: ...
    def check_settings(self, settings_json: str | None = None) -> None: ...
    def check_kind(self, kind: str) -> None: ...

class ModelKindProtocol(Protocol):
    """One kind of model of the engine's list."""

    def __init__(self, name: str) -> None: ...
    @staticmethod
    def names() -> list[str]: ...
    @property
    def name(self) -> str: ...
    def value_types(self) -> list[str]: ...
    def engines(self) -> list[str]: ...

class InfeasibleStateError(ValueError):
    """A run left the values its model allows, such as an amount below zero."""

class EngineModule(Protocol):
    Engine: type[EngineProtocol]
    InfeasibleStateError: type[InfeasibleStateError]
    EngineType: type[EngineTypeProtocol]
    ModelKind: type[ModelKindProtocol]

class ModelModule(Protocol):
    NEIGHBORS: str
    NEIGHBORS_PREFIX: str
    def initial_state(
        self, process: RustProcessProtocol, structure: RustAggregateProtocol
    ) -> list[tuple[str, str, list[tuple[str, str]], float]]: ...
    def transition_flows(
        self, process: RustProcessProtocol, structure: RustAggregateProtocol
    ) -> list[tuple[str, str | None, str | None, str, str | None, int]]: ...
    def check_model(self, model: RustModelProtocol) -> None: ...
    def subgroup_total_names(self, process: RustProcessProtocol) -> list[str]: ...
    def expanded_outputs(
        self, process: RustProcessProtocol
    ) -> tuple[
        list[tuple[str, str, list[tuple[str, str]]]],
        list[tuple[str, str, list[tuple[str, str]]]],
    ]: ...

class ObservedDataPointProtocol(Protocol):
    def __init__(
        self,
        at: float,
        label: str,
        outputs: list[str],
        value: float,
        weight: float | None = None,
        factor_id: str | None = None,
        window: float | None = None,
    ) -> None: ...
    @property
    def at(self) -> float: ...

class CalibrationParameterTypeProtocol(Protocol):
    Parameter: ClassVar["CalibrationParameterTypeProtocol"]
    InitialCondition: ClassVar["CalibrationParameterTypeProtocol"]
    Factor: ClassVar["CalibrationParameterTypeProtocol"]

class CalibrationParameterProtocol(Protocol):
    def __init__(
        self,
        id: str,
        parameter_type: CalibrationParameterTypeProtocol,
        min_bound: float,
        max_bound: float,
        initial_guess: float | None = None,
    ) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def parameter_type(self) -> CalibrationParameterTypeProtocol: ...
    @property
    def min_bound(self) -> float: ...
    @property
    def max_bound(self) -> float: ...

class CalibrationConstraintProtocol(Protocol):
    def __init__(
        self,
        id: str,
        expression: str,
        description: str | None = None,
        weight: float = 1.0,
        at: list[float] | None = None,
    ) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def expression(self) -> str: ...
    @property
    def description(self) -> str | None: ...
    @property
    def weight(self) -> float: ...
    @property
    def at(self) -> list[float] | None: ...

class LossConfigProtocol(Protocol):
    @staticmethod
    def sse() -> LossConfigProtocol: ...
    @staticmethod
    def rmse() -> LossConfigProtocol: ...
    @staticmethod
    def mae() -> LossConfigProtocol: ...
    def normalized(self, enabled: bool = True) -> LossConfigProtocol: ...
    def with_replicates(
        self, count: int, seed: int, summary: str, quantile: float | None = None
    ) -> LossConfigProtocol: ...
    def aggregate(self, residuals: list[float], weights: list[float]) -> float: ...
    @property
    def normalize_observations(self) -> bool: ...

class NelderMeadConfigProtocol(Protocol):
    def __init__(
        self,
        max_iterations: int = 1000,
        sd_tolerance: float = 1e-6,
        initial_simplex_size: float = 0.1,
        alpha: float | None = None,
        gamma: float | None = None,
        rho: float | None = None,
        sigma: float | None = None,
        verbose: bool = False,
        header_interval: int = 100,
    ) -> None: ...

class PSOInertiaConstantProtocol(Protocol):
    def __init__(self, factor: float) -> None: ...
    @property
    def factor(self) -> float: ...

class PSOInertiaChaoticProtocol(Protocol):
    def __init__(self, w_min: float, w_max: float) -> None: ...
    @property
    def w_min(self) -> float: ...
    @property
    def w_max(self) -> float: ...

class PSOAccelerationConstantProtocol(Protocol):
    def __init__(self, cognitive: float, social: float) -> None: ...
    @property
    def cognitive(self) -> float: ...
    @property
    def social(self) -> float: ...

class PSOAccelerationTimeVaryingProtocol(Protocol):
    def __init__(
        self, c1_initial: float, c1_final: float, c2_initial: float, c2_final: float
    ) -> None: ...
    @property
    def c1_initial(self) -> float: ...
    @property
    def c1_final(self) -> float: ...
    @property
    def c2_initial(self) -> float: ...
    @property
    def c2_final(self) -> float: ...

class PSOMutationProtocol(Protocol):
    def __init__(
        self, strategy: str, scale: float, probability: float, application: str
    ) -> None: ...
    @property
    def strategy(self) -> str: ...
    @property
    def scale(self) -> float: ...
    @property
    def probability(self) -> float: ...
    @property
    def application(self) -> str: ...

class PSOVelocityProtocol(Protocol):
    def __init__(
        self, clamp_factor: float | None = None, mutation_threshold: float | None = None
    ) -> None: ...
    @property
    def clamp_factor(self) -> float | None: ...
    @property
    def mutation_threshold(self) -> float | None: ...

class ParticleSwarmConfigProtocol(Protocol):
    def __init__(
        self,
        num_particles: int = 20,
        max_iterations: int = 1000,
        verbose: bool = False,
        inertia: PSOInertiaConstantProtocol | PSOInertiaChaoticProtocol | None = None,
        acceleration: PSOAccelerationConstantProtocol
        | PSOAccelerationTimeVaryingProtocol
        | None = None,
        mutation: PSOMutationProtocol | None = None,
        velocity: PSOVelocityProtocol | None = None,
        initialization: str = "uniform",
        seed: int | None = None,
    ) -> None: ...

class OptimizationConfigProtocol(Protocol):
    @staticmethod
    def nelder_mead(
        config: NelderMeadConfigProtocol | None = None,
    ) -> OptimizationConfigProtocol: ...
    @staticmethod
    def particle_swarm(
        config: ParticleSwarmConfigProtocol | None = None,
    ) -> OptimizationConfigProtocol: ...

class CalibrationResultProtocol(Protocol):
    @property
    def best_parameters(self) -> list[tuple[str, float]]: ...
    @property
    def final_loss(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def termination_reason(self) -> str: ...

class CalibrationEvaluationProtocol(Protocol):
    @property
    def parameters(self) -> list[float]: ...
    @property
    def loss(self) -> float: ...

class CalibrationResultWithHistoryProtocol(Protocol):
    @property
    def result(self) -> CalibrationResultProtocol: ...
    @property
    def evaluations(self) -> list[CalibrationEvaluationProtocol]: ...

class StartOutcomeProtocol(Protocol):
    @property
    def start(self) -> list[float]: ...
    @property
    def result(self) -> CalibrationResultProtocol | None: ...
    @property
    def error(self) -> str | None: ...

class CalibrationModule(Protocol):
    ObservedDataPoint: type[ObservedDataPointProtocol]
    CalibrationParameterType: type[CalibrationParameterTypeProtocol]
    CalibrationParameter: type[CalibrationParameterProtocol]
    CalibrationConstraint: type[CalibrationConstraintProtocol]
    LossConfig: type[LossConfigProtocol]
    NelderMeadConfig: type[NelderMeadConfigProtocol]
    PSOInertiaConstant: type[PSOInertiaConstantProtocol]
    PSOInertiaChaotic: type[PSOInertiaChaoticProtocol]
    PSOAccelerationConstant: type[PSOAccelerationConstantProtocol]
    PSOAccelerationTimeVarying: type[PSOAccelerationTimeVaryingProtocol]
    PSOMutation: type[PSOMutationProtocol]
    PSOVelocity: type[PSOVelocityProtocol]
    ParticleSwarmConfig: type[ParticleSwarmConfigProtocol]
    OptimizationConfig: type[OptimizationConfigProtocol]
    CalibrationResult: type[CalibrationResultProtocol]
    CalibrationEvaluation: type[CalibrationEvaluationProtocol]
    CalibrationResultWithHistory: type[CalibrationResultWithHistoryProtocol]
    StartOutcome: type[StartOutcomeProtocol]
    def check_positions(
        self,
        engine: EngineProtocol,
        observed_data: list[ObservedDataPointProtocol],
        constraints: list[CalibrationConstraintProtocol],
    ) -> None: ...
    def check_model_values(
        self,
        engine: EngineProtocol,
        parameters: list[CalibrationParameterProtocol],
        initial_condition_remainder: str | None = None,
    ) -> None: ...
    def calibrated_initial_state(
        self,
        engine: EngineProtocol,
        parameters: list[CalibrationParameterProtocol],
        values: list[float],
        initial_condition_remainder: str | None = None,
    ) -> tuple[list[float], list[float], list[int]]: ...

class CalibrationProblemProtocol(Protocol):
    def __init__(
        self,
        engine: EngineProtocol,
        observed_data: list[ObservedDataPointProtocol],
        parameters: list[CalibrationParameterProtocol],
        constraints: list[CalibrationConstraintProtocol],
        loss_config: LossConfigProtocol,
        initial_condition_remainder: str | None = None,
    ) -> None: ...
    @property
    def parameter_names(self) -> list[str]: ...
    @property
    def observation_count(self) -> int: ...
    @property
    def replicate_count(self) -> int: ...
    def optimize(
        self, config: OptimizationConfigProtocol, starts: int, seed: int
    ) -> list[StartOutcomeProtocol]: ...
    def optimize_with_history(
        self, config: OptimizationConfigProtocol
    ) -> CalibrationResultWithHistoryProtocol: ...
    def predict_observations(
        self, parameter_sets: list[list[float]]
    ) -> list[list[list[float]]]: ...
    def predict_positions(
        self, parameter_sets: list[list[float]], positions: list[float]
    ) -> list[list[list[list[float]]]]: ...
    def predictive(
        self, sample: ParameterSampleProtocol, positions: list[float]
    ) -> PredictiveProtocol: ...
    def observation_draws(
        self,
        sample: ParameterSampleProtocol,
        observations: list[ObservedDataPointProtocol],
    ) -> ObservationDrawsProtocol: ...

class ParameterSampleProtocol(Protocol):
    def __init__(
        self,
        names: list[str],
        bounds: list[tuple[float, float]],
        values: list[list[float]],
        log_weights: list[float],
        groups: list[int] | None = None,
    ) -> None: ...
    @property
    def weights(self) -> list[float]: ...
    @property
    def log_weights(self) -> list[float]: ...
    def mean(self) -> list[float]: ...
    def covariance(self) -> list[list[float]]: ...
    def correlation(self) -> list[list[float]]: ...
    def quantiles(self, levels: list[float]) -> list[list[float]]: ...
    def interval(self, confidence_level: float) -> tuple[list[float], list[float]]: ...
    def effective_size(self) -> float: ...
    def resample(self, count: int, seed: int) -> list[int]: ...

type _Band2 = tuple[
    list[list[float]], list[list[float]], list[list[float]], list[list[float]]
]
type _Band1 = tuple[list[float], list[float], list[float], list[float]]

class PredictiveProtocol(Protocol):
    @property
    def positions(self) -> list[float]: ...
    @property
    def trajectories(self) -> list[list[list[float]]]: ...
    @property
    def weights(self) -> list[float]: ...
    def band(self, confidence_level: float) -> _Band2: ...
    def series_band(
        self, outputs: list[int], cumulative: bool, confidence_level: float
    ) -> _Band1: ...

class ObservationDrawsProtocol(Protocol):
    @property
    def values(self) -> list[list[float]]: ...
    @property
    def weights(self) -> list[float]: ...
    def quantiles(self, levels: list[float]) -> list[list[float]]: ...
    def band(self, confidence_level: float) -> _Band1: ...
    def coverage(
        self, observed: list[float], confidence_level: float
    ) -> list[float]: ...
    def interval_score(
        self, observed: list[float], confidence_level: float
    ) -> list[float]: ...
    def weighted_interval_score(
        self, observed: list[float], confidence_levels: list[float]
    ) -> list[float]: ...
    def crps(self, observed: list[float]) -> list[float]: ...
    def pit(self, observed: list[float], seed: int) -> list[float]: ...

class InferenceModule(Protocol):
    CalibrationProblem: type[CalibrationProblemProtocol]
    ParameterSample: type[ParameterSampleProtocol]
    Predictive: type[PredictiveProtocol]
    ObservationDraws: type[ObservationDrawsProtocol]
    def weighted_quantiles(
        self, values: list[float], weights: list[float], levels: list[float]
    ) -> list[float]: ...
    def weighted_mean(self, values: list[float], weights: list[float]) -> float: ...
    def by_series(
        self, labels: list[str], values: list[float]
    ) -> list[tuple[str, float]]: ...

class MathExpressionProtocol(Protocol):
    def __init__(self, expression: str) -> None: ...
    def validate(self) -> None: ...

class RateMathExpressionProtocol(Protocol):
    @staticmethod
    def from_string_py(s: str) -> RateMathExpressionProtocol: ...
    @staticmethod
    def parameter(name: str) -> RateMathExpressionProtocol: ...
    @staticmethod
    def formula(formula: str) -> RateMathExpressionProtocol: ...
    @staticmethod
    def constant(value: float) -> RateMathExpressionProtocol: ...
    def py_get_variables(self) -> list[str]: ...

class CoreModule(Protocol):
    Model: type[RustModelProtocol]
    Process: type[RustProcessProtocol]
    Aggregate: type[RustAggregateProtocol]
    Bin: type[BinProtocol]
    Stratification: type[StratificationProtocol]
    StratificationCondition: type[StratificationConditionProtocol]
    StratifiedRate: type[StratifiedRateProtocol]
    Transition: type[TransitionProtocol]
    Parameter: type[ParameterProtocol]
    InitialConditions: type[InitialConditionsProtocol]
    StratificationFraction: type[StratificationFractionProtocol]
    StratificationFractions: type[StratificationFractionsProtocol]
    MathExpression: type[MathExpressionProtocol]
    RateMathExpression: type[RateMathExpressionProtocol]

class RustCommolModule(Protocol):
    core: CoreModule
    model: ModelModule
    engine: EngineModule
    calibration: CalibrationModule
    inference: InferenceModule

core: CoreModule
model: ModelModule
engine: EngineModule
calibration: CalibrationModule
inference: InferenceModule
rust_commol: RustCommolModule
