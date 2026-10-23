from contextvars import ContextVar
from typing import ClassVar, Literal, Protocol

class RustProcessProtocol(Protocol):
    """The process of a model: bins, accumulators, stratifications and transitions."""

    @staticmethod
    def from_json(json_str: str) -> RustProcessProtocol: ...

class RustAggregateProtocol(Protocol):
    """The aggregate structure: amounts per compartment."""

    @staticmethod
    def from_json(json_str: str) -> RustAggregateProtocol: ...

class RustModelProtocol(Protocol):
    """A model of any kind: its header, process, structure and engine."""

    @staticmethod
    def from_json(json_str: str) -> RustModelProtocol: ...

class EngineProtocol(Protocol):
    """The engine a model names, built from the model."""

    def __init__(self, model: RustModelProtocol) -> None: ...
    @property
    def engine_type(self) -> str: ...
    def run(self, num_steps: int) -> tuple[list[float], list[list[float]]]: ...
    def run_at(self, positions: list[float]) -> list[list[float]]: ...
    def step(self) -> None: ...
    def reset(self) -> None: ...
    def set_seed(self, seed: int | None = None) -> int: ...
    def run_replicates(
        self, seed: int | None, count: int, grid: int | list[float]
    ) -> tuple[int, list[float], list[int], list[list[list[float]]]]: ...
    def current_position(self) -> float: ...
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
    def value_types(self) -> list[str]: ...
    def engines(self) -> list[str]: ...

class InfeasibleStateError(ValueError):
    """A run left the values its model allows, such as an amount below zero."""

class EngineModule(Protocol):
    ODE_DEFAULT_RTOL: float
    ODE_DEFAULT_ATOL: float
    STOCHASTIC_EVENTS_DEFAULT_EPSILON: float
    STOCHASTIC_EVENTS_DEFAULT_TOLERANCE: float
    Engine: type[EngineProtocol]
    InfeasibleStateError: type[InfeasibleStateError]
    EngineType: type[EngineTypeProtocol]
    ModelKind: type[ModelKindProtocol]

class ModelModule(Protocol):
    NEIGHBORS: str
    NEIGHBORS_PREFIX: str
    ENGINE_VARIABLES: list[str]
    TOTAL: str
    SUBGROUP_TOTAL_PREFIX: str
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
        trials: int | None = None,
    ) -> None: ...

class CalibrationParameterTypeProtocol(Protocol):
    Parameter: ClassVar["CalibrationParameterTypeProtocol"]
    InitialCondition: ClassVar["CalibrationParameterTypeProtocol"]
    Factor: ClassVar["CalibrationParameterTypeProtocol"]
    Noise: ClassVar["CalibrationParameterTypeProtocol"]

class ObservationModelProtocol(Protocol):
    @staticmethod
    def normal(sd: float | str) -> ObservationModelProtocol: ...
    @staticmethod
    def log_normal(sigma: float | str, median: bool) -> ObservationModelProtocol: ...
    @staticmethod
    def student_t(df: float | str, scale: float | str) -> ObservationModelProtocol: ...
    @staticmethod
    def poisson() -> ObservationModelProtocol: ...
    @staticmethod
    def negative_binomial(overdispersion: float | str) -> ObservationModelProtocol: ...
    @staticmethod
    def binomial(trials: int | None = None) -> ObservationModelProtocol: ...

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
        weight: float = 1.0,
        at: list[float] | None = None,
    ) -> None: ...

class LossConfigProtocol(Protocol):
    @staticmethod
    def sse() -> LossConfigProtocol: ...
    @staticmethod
    def rmse() -> LossConfigProtocol: ...
    @staticmethod
    def mae() -> LossConfigProtocol: ...
    def normalized(self, enabled: bool = True) -> LossConfigProtocol: ...
    def with_replicates(
        self,
        count: int | None = None,
        summary: str | None = None,
        quantile: float | None = None,
    ) -> LossConfigProtocol: ...
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
        particles: int = 20,
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
        improvement_stop: tuple[int, float, float] | None = None,
        radius_stop: float | None = None,
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
    PSO_DEFAULT_INERTIA_FACTOR: float
    PSO_DEFAULT_ACCELERATION: float
    ObservedDataPoint: type[ObservedDataPointProtocol]
    CalibrationParameterType: type[CalibrationParameterTypeProtocol]
    CalibrationParameter: type[CalibrationParameterProtocol]
    CalibrationConstraint: type[CalibrationConstraintProtocol]
    LossConfig: type[LossConfigProtocol]
    ObservationModel: type[ObservationModelProtocol]
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
    def index_at(self, positions: list[float], at: list[float]) -> list[int]: ...
    def window_start(self, at: float, window: float) -> float: ...
    def window_increments(
        self,
        series: list[float],
        positions: list[float],
        window: float,
        ends: list[float],
    ) -> list[float]: ...
    def check_model_values(
        self,
        engine: EngineProtocol,
        parameters: list[CalibrationParameterProtocol],
        initial_condition_remainder: str | None = None,
    ) -> None: ...
    def initial_condition_coverage(
        self,
        engine: EngineProtocol,
        parameters: list[CalibrationParameterProtocol],
        initial_condition_remainder: str | None = None,
    ) -> tuple[list[list[int]], list[int]]: ...
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
        loss_config: LossConfigProtocol | None = None,
        initial_condition_remainder: str | None = None,
        observation_models: list[tuple[str, ObservationModelProtocol]] | None = None,
        seed: int | None = None,
    ) -> None: ...
    @property
    def seed(self) -> int: ...
    def optimize(
        self, config: OptimizationConfigProtocol, starts: int, seed: int
    ) -> list[StartOutcomeProtocol]: ...
    def optimize_with_history(
        self, config: OptimizationConfigProtocol
    ) -> CalibrationResultWithHistoryProtocol: ...
    def predict_observations(
        self, parameter_sets: list[list[float]]
    ) -> list[list[list[float]]]: ...
    def predictive(
        self, sample: ParameterSampleProtocol, positions: list[float], seed: int
    ) -> PredictiveProtocol: ...
    def observation_draws(
        self,
        sample: ParameterSampleProtocol,
        observations: list[ObservedDataPointProtocol],
        seed: int,
        residual_noise: tuple[list[float], int, str] | None = None,
        distribution_noise: bool = False,
    ) -> ObservationDrawsProtocol: ...
    def log_score(
        self,
        sample: ParameterSampleProtocol,
        observations: list[ObservedDataPointProtocol],
        seed: int,
    ) -> list[float]: ...
    def pointwise_log_likelihood(self, values: list[float]) -> list[float]: ...
    def parametric_bootstrap(
        self,
        config: OptimizationConfigProtocol,
        fit: list[float],
        replicates: int,
        starts: int,
        seed: int,
    ) -> BootstrapRunProtocol: ...
    def bayesian_bootstrap(
        self,
        config: OptimizationConfigProtocol,
        fit: list[float],
        replicates: int,
        block_length: int | None,
        starts: int,
        seed: int,
    ) -> BootstrapRunProtocol: ...
    def residual_bootstrap(
        self,
        config: OptimizationConfigProtocol,
        fit: list[float],
        replicates: int,
        block_length: int | None,
        starts: int,
        seed: int,
        residual_scale: str,
        scheme: str,
    ) -> BootstrapRunProtocol: ...
    def wild_bootstrap(
        self,
        config: OptimizationConfigProtocol,
        fit: list[float],
        replicates: int,
        block_length: int | None,
        starts: int,
        seed: int,
        multipliers: str,
    ) -> BootstrapRunProtocol: ...
    def jackknife(
        self,
        config: OptimizationConfigProtocol,
        fit: list[float],
        block_length: int | None,
        starts: int,
        seed: int,
    ) -> JackknifeRunProtocol: ...

class BootstrapRunProtocol(Protocol):
    @property
    def values(self) -> list[list[float]]: ...
    @property
    def groups(self) -> list[int]: ...
    @property
    def failed(self) -> list[str]: ...
    @property
    def not_converged(self) -> int: ...
    @property
    def negative_values(self) -> int: ...
    @property
    def block_length(self) -> int | None: ...
    @property
    def block_length_estimates(self) -> list[tuple[str, float]] | None: ...

class BootstrapSettingsProtocol(Protocol):
    def __init__(
        self,
        method: str,
        replicates: int,
        starts: int,
        block_length: int | None = None,
        residual_scale: str | None = None,
        scheme: str | None = None,
        multipliers: str | None = None,
    ) -> None: ...

class ProfileProtocol(Protocol):
    @property
    def center(self) -> float: ...
    @property
    def threshold(self) -> float: ...
    @property
    def values(self) -> list[float]: ...
    @property
    def statistics(self) -> list[float]: ...
    @property
    def parameters(self) -> list[list[float]]: ...
    @property
    def multipliers(self) -> list[float | None]: ...
    @property
    def converged(self) -> list[bool]: ...
    @property
    def ends(self) -> tuple[str, str]: ...
    @property
    def interval(self) -> tuple[float | None, float | None]: ...
    @property
    def evaluations(self) -> int: ...
    @property
    def rises(self) -> list[float] | None: ...
    @property
    def verdict(self) -> str | None: ...
    @property
    def lowest(self) -> float: ...
    @property
    def structure(self) -> ProfileStructureProtocol | None: ...
    @property
    def box_crossings(self) -> tuple[int | None, int | None]: ...
    @property
    def refit_error(self) -> float | None: ...
    @property
    def inaccurate_refit(self) -> bool: ...
    @property
    def below_fit(self) -> bool: ...
    @property
    def box_exits(
        self,
    ) -> tuple[tuple[float, int] | None, tuple[float, int] | None]: ...

class ProfileStructureProtocol(Protocol):
    @property
    def verdict(self) -> str | None: ...
    @property
    def undetermined(self) -> str | None: ...
    @property
    def on_bound(self) -> int | None: ...
    @property
    def largest_rise(self) -> float | None: ...
    @property
    def compensating(self) -> list[int]: ...
    @property
    def joint_rise(self) -> float | None: ...

class ProfileRunProtocol(Protocol):
    @property
    def profiles(self) -> list[ProfileProtocol]: ...
    @property
    def fit_loss(self) -> float: ...
    @property
    def tolerance(self) -> float: ...
    @property
    def optimizer_tolerance(self) -> float: ...
    @property
    def noise_floor(self) -> float: ...
    @property
    def evaluation_floor(self) -> float | None: ...
    @property
    def replicates(self) -> list[int] | None: ...
    @property
    def failed(self) -> list[str]: ...
    @property
    def not_converged(self) -> int: ...

class JackknifeRunProtocol(Protocol):
    @property
    def spans(self) -> list[tuple[float, float]]: ...
    @property
    def estimate(self) -> list[float]: ...
    @property
    def bias(self) -> list[float]: ...
    @property
    def standard_error(self) -> list[float]: ...
    @property
    def pseudo_values(self) -> list[list[float]]: ...
    @property
    def influence(self) -> list[list[float]]: ...
    @property
    def failed(self) -> list[str]: ...
    @property
    def not_converged(self) -> int: ...
    @property
    def block_length(self) -> int: ...
    @property
    def block_length_estimates(self) -> list[tuple[str, float]] | None: ...

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
    def trajectories(self) -> list[list[list[float]]]: ...
    @property
    def weights(self) -> list[float]: ...
    @property
    def draw_seeds(self) -> list[int]: ...
    def band(self, confidence_level: float) -> _Band2: ...
    def series_band(
        self, outputs: list[int], cumulative: bool, confidence_level: float
    ) -> _Band1: ...

class ObservationDrawsProtocol(Protocol):
    @property
    def values(self) -> list[list[float]]: ...
    @property
    def weights(self) -> list[float]: ...
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
    def pit(self, observed: list[float]) -> list[float]: ...

class PointRankProtocol(Protocol):
    @property
    def values(self) -> list[float]: ...
    @property
    def singular_values(self) -> list[float]: ...
    @property
    def threshold(self) -> float: ...
    @property
    def rank(self) -> int: ...

class StructuralAnalysisProtocol(Protocol):
    @property
    def points(self) -> list[PointRankProtocol]: ...
    @property
    def skipped(self) -> list[str]: ...
    @property
    def generic_rank(self) -> int: ...
    @property
    def projections(self) -> list[float]: ...
    @property
    def verdicts(
        self,
    ) -> list[Literal["identifiable", "in_combination", "non_identifiable"]]: ...
    @property
    def combinations(self) -> list[list[tuple[int, float]]]: ...

class CollinearityVerdictProtocol(Protocol):
    @property
    def verdict(self) -> Literal["determined", "poorly_determined"]: ...
    @property
    def insensitive(self) -> bool: ...
    @property
    def collinear_with(self) -> list[int]: ...

class CollinearityAnalysisProtocol(Protocol):
    @property
    def sensitivity(self) -> list[float]: ...
    @property
    def pair_gamma(self) -> list[list[float]]: ...
    @property
    def best_by_size(self) -> list[tuple[list[int], float, float]]: ...
    @property
    def admissible(self) -> list[tuple[list[int], float, float]]: ...
    @property
    def verdicts(self) -> list[CollinearityVerdictProtocol]: ...
    @property
    def points(self) -> int: ...
    @property
    def skipped(self) -> list[str]: ...

class SampleVerdictProtocol(Protocol):
    @property
    def verdict(self) -> Literal["determined", "poorly_determined"]: ...
    @property
    def correlated_with(self) -> list[tuple[int, float]]: ...
    @property
    def modes(self) -> int: ...

class SampleAnalysisProtocol(Protocol):
    @property
    def contraction(self) -> list[float]: ...
    @property
    def correlation(self) -> list[list[float]]: ...
    @property
    def correlated_pairs(self) -> list[tuple[int, int, float]]: ...
    @property
    def modes(self) -> list[list[float]]: ...
    @property
    def verdicts(self) -> list[SampleVerdictProtocol]: ...

class StartsAnalysisProtocol(Protocol):
    @property
    def near_best(self) -> list[int]: ...
    @property
    def spread(self) -> list[float]: ...
    @property
    def flagged(self) -> list[bool]: ...

class DistributionProtocol(Protocol):
    def __init__(
        self, kind: str, parameters: list[float], lower: float, upper: float
    ) -> None: ...
    def log_density(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def quantile(self, p: float) -> float: ...
    def sample(self, count: int, seed: int) -> list[float]: ...

class SMCRunProtocol(Protocol):
    @property
    def values(self) -> list[list[float]]: ...
    @property
    def log_weights(self) -> list[float]: ...
    @property
    def kept(self) -> list[float | None]: ...
    @property
    def levels(self) -> list[float]: ...
    @property
    def effective_sizes(self) -> list[float]: ...
    @property
    def resampled(self) -> list[bool]: ...
    @property
    def acceptance_rates(self) -> list[float]: ...
    @property
    def move_counts(self) -> list[int]: ...
    @property
    def log_evidence(self) -> float: ...
    @property
    def evaluations(self) -> int: ...
    @property
    def stop_reason(self) -> str | None: ...

class EtaCoverageProtocol(Protocol):
    @property
    def eta(self) -> float: ...
    @property
    def coverage(self) -> float: ...
    @property
    def failed(self) -> list[str]: ...

_Priors = list[tuple[str, str, list[float]]] | None

class LocalRunProtocol(Protocol):
    values: list[list[float]]
    center: list[float]
    covariance: list[list[float]]
    degrees_of_freedom: float | None
    notes: list[str]

class MixtureRunProtocol(Protocol):
    values: list[list[float]]
    log_weights: list[float]
    centers: list[list[float]]
    weights: list[float]
    covariances: list[list[list[float]]]
    log_evidences: list[float]
    k_hat: float
    effective_size: float
    reliable: bool
    notes: list[str]

class ConformalRunProtocol(Protocol):
    @property
    def labels(self) -> list[str]: ...
    @property
    def positions(self) -> list[float]: ...
    @property
    def prediction(self) -> list[float]: ...
    @property
    def lower(self) -> list[float]: ...
    @property
    def upper(self) -> list[float]: ...
    @property
    def scores(self) -> list[tuple[str, float]]: ...
    @property
    def failed(self) -> list[str]: ...
    @property
    def not_converged(self) -> int: ...
    @property
    def bound(self) -> float | None: ...
    @property
    def blocks(self) -> int | None: ...
    @property
    def block_length(self) -> int | None: ...
    @property
    def block_length_estimates(self) -> list[tuple[str, float]] | None: ...

class SobolRunProtocol(Protocol):
    @property
    def first(self) -> list[list[float]]: ...
    @property
    def total(self) -> list[list[float]]: ...
    @property
    def first_lower(self) -> list[list[float]]: ...
    @property
    def first_upper(self) -> list[list[float]]: ...
    @property
    def total_lower(self) -> list[list[float]]: ...
    @property
    def total_upper(self) -> list[list[float]]: ...
    @property
    def runs(self) -> int: ...

class ChaosRunProtocol(Protocol):
    @property
    def indices(self) -> list[list[int]]: ...
    @property
    def coefficients(self) -> list[list[float]]: ...
    @property
    def loo_error(self) -> list[float]: ...
    @property
    def mean(self) -> list[float]: ...
    @property
    def variance(self) -> list[float]: ...
    @property
    def first(self) -> list[list[float]]: ...
    @property
    def total(self) -> list[list[float]]: ...
    @property
    def runs(self) -> int: ...

type _Draws = tuple[list[list[float]], list[float]]

class ForwardAnalysisProtocol(Protocol):
    def __init__(
        self,
        engine: EngineProtocol,
        parameters: list[CalibrationParameterProtocol],
        distributions: _Priors = None,
        sample: ParameterSampleProtocol | None = None,
        initial_condition_remainder: str | None = None,
        replicates: int | None = None,
        seed: int | None = None,
    ) -> None: ...
    @property
    def seed(self) -> int: ...
    def propagate(
        self, draws: int | None, quantities: str
    ) -> tuple[_Draws, list[list[float]], list[float], list[int]]: ...
    def prcc(
        self, draws: int | None, quantities: str
    ) -> tuple[_Draws, list[list[float]], list[list[float]], float]: ...
    def morris(
        self, trajectories: int, levels: int, quantities: str
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]], int]: ...
    def sobol(
        self,
        draws: int,
        replicates: int,
        confidence_level: float,
        quantities: str,
    ) -> SobolRunProtocol: ...
    def chaos(
        self,
        degree: int,
        q_norm: float,
        max_interaction: int | None,
        draws: int,
        quantities: str,
    ) -> ChaosRunProtocol: ...
    def predictive(
        self, sample: ParameterSampleProtocol, positions: list[float]
    ) -> PredictiveProtocol: ...
    def observation_draws(
        self,
        sample: ParameterSampleProtocol,
        observations: list[ObservedDataPointProtocol],
    ) -> ObservationDrawsProtocol: ...

class InferenceModule(Protocol):
    CalibrationProblem: type[CalibrationProblemProtocol]
    ParameterSample: type[ParameterSampleProtocol]
    Predictive: type[PredictiveProtocol]
    ObservationDraws: type[ObservationDrawsProtocol]
    BootstrapRun: type[BootstrapRunProtocol]
    JackknifeRun: type[JackknifeRunProtocol]
    Profile: type[ProfileProtocol]
    ProfileRun: type[ProfileRunProtocol]
    ProfileStructure: type[ProfileStructureProtocol]
    BootstrapSettings: type[BootstrapSettingsProtocol]
    def trace_profiles(
        self,
        problem: CalibrationProblemProtocol,
        config: OptimizationConfigProtocol,
        fit: list[float],
        parameters: list[int],
        predictions: list[ObservedDataPointProtocol],
        threshold: str,
        confidence_level: float,
        max_steps: int,
        seed: int,
        bootstrap: BootstrapSettingsProtocol | None = None,
    ) -> ProfileRunProtocol: ...
    Distribution: type[DistributionProtocol]
    SMCRun: type[SMCRunProtocol]
    EtaCoverage: type[EtaCoverageProtocol]
    ConformalRun: type[ConformalRunProtocol]
    ForwardAnalysis: type[ForwardAnalysisProtocol]
    SobolRun: type[SobolRunProtocol]
    ChaosRun: type[ChaosRunProtocol]
    def chaos_evaluate(
        self,
        parameters: list[CalibrationParameterProtocol],
        distributions: list[tuple[str, str, list[float]]],
        degree: int,
        q_norm: float,
        max_interaction: int | None,
        coefficients: list[list[float]],
        points: list[list[float]],
    ) -> list[list[float]]: ...
    def quantile_error(
        self, values: list[float], level: float
    ) -> tuple[float, float]: ...
    def weighted_quantiles(
        self, values: list[float], weights: list[float], levels: list[float]
    ) -> list[float]: ...
    def interval_levels(self, confidence_level: float) -> tuple[float, float]: ...
    def check_open_unit(self, name: str, value: float) -> None: ...
    def check_levels(self, levels: list[float]) -> None: ...
    def weighted_mean(self, values: list[float], weights: list[float]) -> float: ...
    def quantities_as_read(self, json: str) -> str: ...
    def quantity_schedule(self, json: str) -> list[float]: ...
    def quantities_of_runs(
        self,
        json: str,
        names: list[str],
        output_names: list[str],
        positions: list[float],
        runs: list[list[list[float]]],
    ) -> list[list[float]]: ...
    def by_series(
        self, labels: list[str], values: list[float]
    ) -> list[tuple[str, float]]: ...
    def structural_rank(
        self,
        problem: CalibrationProblemProtocol,
        points: int,
        dense: bool,
        seed: int,
    ) -> StructuralAnalysisProtocol: ...
    def collinearity(
        self,
        problem: CalibrationProblemProtocol,
        points: list[list[float]],
        weights: list[float],
        max_subset_size: int,
        threshold: float,
    ) -> CollinearityAnalysisProtocol: ...
    def sample_identifiability(
        self,
        sample: ParameterSampleProtocol,
        correlation_threshold: float,
        priors: _Priors = None,
    ) -> SampleAnalysisProtocol: ...
    def gibbs_posterior(
        self,
        problem: CalibrationProblemProtocol,
        priors: _Priors,
        eta: float,
        particles: int,
        resample_threshold: float,
        target_cess: float,
        stay_probability: float,
        seed: int,
    ) -> SMCRunProtocol: ...
    def bayesian_posterior(
        self,
        problem: CalibrationProblemProtocol,
        priors: _Priors,
        particles: int,
        resample_threshold: float,
        target_cess: float,
        stay_probability: float,
        seed: int,
    ) -> SMCRunProtocol: ...
    def laplace(
        self,
        problem: CalibrationProblemProtocol,
        fit: list[float],
        curvature: str,
        draws: int,
        seed: int,
    ) -> LocalRunProtocol: ...
    def sandwich(
        self,
        problem: CalibrationProblemProtocol,
        fit: list[float],
        curvature: str,
        block_length: int | None,
        draws: int,
        seed: int,
    ) -> tuple[LocalRunProtocol, int, int, list[tuple[str, float]] | None]: ...
    def laplace_mixture(
        self,
        problem: CalibrationProblemProtocol,
        priors: _Priors,
        eta: float | None,
        ends: list[list[float]],
        draws: int,
        curvature: str,
        seed: int,
    ) -> MixtureRunProtocol: ...
    def delta_band(
        self,
        problem: CalibrationProblemProtocol,
        fit: list[float],
        covariance: list[list[float]],
        degrees_of_freedom: float | None,
        positions: list[float],
        confidence_level: float,
    ) -> tuple[list[str], list[list[float]], list[list[float]], list[list[float]]]: ...
    def information_matching_eta(
        self,
        problem: CalibrationProblemProtocol,
        fit: list[float],
        block_length: int | None,
    ) -> float: ...
    def bootstrap_coverage_eta(
        self,
        problem: CalibrationProblemProtocol,
        priors: _Priors,
        fit: list[float],
        replicates: int,
        block_length: int | None,
        confidence_level: float,
        particles: int,
        resample_threshold: float,
        target_cess: float,
        stay_probability: float,
        seed: int,
    ) -> EtaCoverageProtocol: ...
    def abc_smc(
        self,
        problem: CalibrationProblemProtocol,
        priors: _Priors,
        metric: str | None,
        particles: int,
        target_cess: float,
        min_acceptance: float,
        stay_probability: float,
        seed: int,
    ) -> tuple[SMCRunProtocol, bool]: ...
    def abc_rejection(
        self,
        problem: CalibrationProblemProtocol,
        priors: _Priors,
        metric: str | None,
        draws: int,
        quantile: float,
        seed: int,
    ) -> tuple[SMCRunProtocol, bool]: ...
    def starts_identifiability(
        self,
        ends: list[list[float]],
        losses: list[float],
        converged: list[bool],
        bounds: list[tuple[float, float]],
        parameter_tolerance: float,
        loss_tolerance: float,
    ) -> StartsAnalysisProtocol: ...
    def check_conformal_settings(self, score: str | None, by: str | None) -> None: ...
    def split_conformal(
        self,
        problem: CalibrationProblemProtocol,
        fit: list[float] | None,
        sample: ParameterSampleProtocol | None,
        calibration: list[ObservedDataPointProtocol],
        positions: list[float],
        score: str,
        by: str,
        confidence_level: float,
    ) -> ConformalRunProtocol: ...
    def rolling_conformal(
        self,
        problem: CalibrationProblemProtocol,
        config: OptimizationConfigProtocol,
        fit: list[float],
        origins: list[float],
        positions: list[float],
        score: str,
        by: str,
        confidence_level: float,
        starts: int,
        seed: int,
    ) -> ConformalRunProtocol: ...
    def jackknife_plus(
        self,
        problem: CalibrationProblemProtocol,
        config: OptimizationConfigProtocol,
        fit: list[float],
        block_length: int | None,
        positions: list[float],
        score: str,
        by: str,
        confidence_level: float,
        starts: int,
        seed: int,
    ) -> ConformalRunProtocol: ...
    def enbpi(
        self,
        problem: CalibrationProblemProtocol,
        config: OptimizationConfigProtocol,
        fit: list[float],
        replicates: int,
        block_length: int | None,
        positions: list[float],
        score: str,
        by: str,
        confidence_level: float,
        starts: int,
        seed: int,
    ) -> ConformalRunProtocol: ...
    def adaptive_conformal(
        self,
        problem: CalibrationProblemProtocol,
        fit: list[float] | None,
        sample: ParameterSampleProtocol | None,
        stream: list[ObservedDataPointProtocol],
        step: float,
        score: str,
        by: str,
        confidence_level: float,
    ) -> ConformalRunProtocol: ...

class MathExpressionProtocol(Protocol):
    def __init__(self, expression: str) -> None: ...
    def check(self) -> int: ...
    @property
    def variables(self) -> list[str]: ...
    @property
    def inputs(self) -> list[str]: ...

class ParallelismProtocol(Protocol):
    def __init__(self, threads: int, levels: str) -> None: ...
    @property
    def threads(self) -> int: ...
    @property
    def levels(self) -> str: ...

class ParallelismModule(Protocol):
    Parallelism: type[ParallelismProtocol]
    def current(self) -> ParallelismProtocol: ...
    def setting(self) -> ContextVar[ParallelismProtocol | None]: ...

class CoreModule(Protocol):
    Model: type[RustModelProtocol]
    Process: type[RustProcessProtocol]
    Aggregate: type[RustAggregateProtocol]
    MathExpression: type[MathExpressionProtocol]
    FUNCTIONS: list[str]

class RustCommolModule(Protocol):
    core: CoreModule
    model: ModelModule
    engine: EngineModule
    calibration: CalibrationModule
    inference: InferenceModule
    parallelism: ParallelismModule

core: CoreModule
model: ModelModule
engine: EngineModule
calibration: CalibrationModule
inference: InferenceModule
parallelism: ParallelismModule
rust_commol: RustCommolModule
__version__: str
__commit__: str | None
