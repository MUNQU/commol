from typing import Protocol

from commol.constants import LogicOperators, ModelTypes, VariablePrefixes

class RuleProtocol(Protocol):
    variable: str
    operator: LogicOperators
    value: str | int | float | bool

class ConditionProtocol(Protocol):
    logic: LogicOperators
    rules: list[RuleProtocol]

class DiseaseStateProtocol(Protocol):
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
    condition: ConditionProtocol | None

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
    population_size: int
    disease_state_fraction: dict[str, float]
    stratification_fractions: list[StratificationFractionsProtocol]

class PopulationProtocol(Protocol):
    disease_states: list[DiseaseStateProtocol]
    stratifications: list[StratificationProtocol]
    transitions: list[TransitionProtocol]
    initial_conditions: InitialConditionsProtocol

class DynamicsProtocol(Protocol):
    typology: ModelTypes
    transitions: list[TransitionProtocol]

class RustModelProtocol(Protocol):
    name: str
    description: str | None
    version: str | None
    population: PopulationProtocol
    parameters: list[ParameterProtocol]
    dynamics: DynamicsProtocol
    @staticmethod
    def from_json(json_string: str) -> RustModelProtocol: ...

class DifferenceEquationsProtocol(Protocol):
    def __init__(self, model: RustModelProtocol) -> None: ...
    def run(self, num_steps: int) -> list[list[float]]: ...
    def step(self) -> None: ...
    @property
    def population(self) -> list[float]: ...
    @property
    def compartments(self) -> list[str]: ...

class DifferenceModule(Protocol):
    DifferenceEquations: type[DifferenceEquationsProtocol]

class ObservedDataPointProtocol(Protocol):
    def __init__(
        self,
        step: int,
        compartment: str,
        value: float,
        weight: float | None = None,
        scale_id: str | None = None,
    ) -> None: ...
    @property
    def time_step(self) -> int: ...
    @property
    def compartment(self) -> str: ...
    @property
    def value(self) -> float: ...
    @property
    def scale_id(self) -> str | None: ...

class CalibrationParameterTypeProtocol(Protocol):
    Parameter: "CalibrationParameterTypeProtocol"
    InitialCondition: "CalibrationParameterTypeProtocol"
    Scale: "CalibrationParameterTypeProtocol"

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
        time_steps: list[int] | None = None,
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
    def time_steps(self) -> list[int] | None: ...

class LossConfigProtocol(Protocol):
    @staticmethod
    def sse() -> LossConfigProtocol: ...
    @staticmethod
    def rmse() -> LossConfigProtocol: ...
    @staticmethod
    def mae() -> LossConfigProtocol: ...
    @staticmethod
    def weighted_sse() -> LossConfigProtocol: ...

class NelderMeadConfigProtocol(Protocol):
    def __init__(
        self,
        max_iterations: int = 1000,
        sd_tolerance: float = 1e-6,
        simplex_perturbation: float = 1.1,
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
    def best_parameters(self) -> dict[str, float]: ...
    @property
    def best_parameters_list(self) -> list[float]: ...
    @property
    def parameter_names(self) -> list[str]: ...
    @property
    def final_loss(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def termination_reason(self) -> str: ...
    def to_dict(self) -> dict[str, object]: ...

class CalibrationEvaluationProtocol(Protocol):
    def __init__(
        self,
        parameters: list[float],
        loss: float,
        predictions: list[list[float]],
    ) -> None: ...
    @property
    def parameters(self) -> list[float]: ...
    @property
    def loss(self) -> float: ...
    @property
    def predictions(self) -> list[list[float]]: ...

class CalibrationResultWithHistoryProtocol(Protocol):
    @property
    def best_parameters(self) -> dict[str, float]: ...
    @property
    def best_parameters_list(self) -> list[float]: ...
    @property
    def parameter_names(self) -> list[str]: ...
    @property
    def final_loss(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def termination_reason(self) -> str: ...
    @property
    def evaluations(self) -> list[CalibrationEvaluationProtocol]: ...

class ParetoSolutionProtocol(Protocol):
    @property
    def ensemble_size(self) -> int: ...
    @property
    def ci_width(self) -> float: ...
    @property
    def coverage(self) -> float: ...
    @property
    def size_penalty(self) -> float: ...
    @property
    def selected_indices(self) -> list[int]: ...

class EnsembleSelectionResultProtocol(Protocol):
    @property
    def selected_ensemble(self) -> list[int]: ...
    @property
    def pareto_front(self) -> list[ParetoSolutionProtocol]: ...
    @property
    def selected_pareto_index(self) -> int: ...

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
    ParetoSolution: type[ParetoSolutionProtocol]
    EnsembleSelectionResult: type[EnsembleSelectionResultProtocol]

    def calibrate(
        self,
        engine: DifferenceEquationsProtocol,
        observed_data: list[ObservedDataPointProtocol],
        parameters: list[CalibrationParameterProtocol],
        constraints: list[CalibrationConstraintProtocol],
        loss_config: LossConfigProtocol,
        optimization_config: OptimizationConfigProtocol,
        initial_population_size: int,
    ) -> CalibrationResultProtocol: ...
    def calibrate_with_history(
        self,
        engine: DifferenceEquationsProtocol,
        observed_data: list[ObservedDataPointProtocol],
        parameters: list[CalibrationParameterProtocol],
        constraints: list[CalibrationConstraintProtocol],
        loss_config: LossConfigProtocol,
        optimization_config: OptimizationConfigProtocol,
        initial_population_size: int,
    ) -> CalibrationResultWithHistoryProtocol: ...
    def run_multiple_calibrations(
        self,
        engine: DifferenceEquationsProtocol,
        observed_data: list[ObservedDataPointProtocol],
        parameters: list[CalibrationParameterProtocol],
        constraints: list[CalibrationConstraintProtocol],
        loss_config: LossConfigProtocol,
        optimization_config: OptimizationConfigProtocol,
        initial_population_size: int,
        n_runs: int,
        seed: int,
    ) -> list[CalibrationResultWithHistoryProtocol]: ...
    def select_optimal_ensemble(
        self,
        candidates: list[CalibrationEvaluationProtocol],
        observed_data_tuples: list[tuple[int, int, float]],
        population_size: int,
        generations: int,
        confidence_level: float,
        seed: int,
        pareto_preference: float,
        ensemble_size_mode: str,
        ensemble_size: int | None = None,
        ensemble_size_min: int | None = None,
        ensemble_size_max: int | None = None,
        ci_margin_factor: float = 0.1,
        ci_sample_sizes: list[int] | None = None,
        nsga_crossover_probability: float = 0.9,
    ) -> EnsembleSelectionResultProtocol: ...
    def deduplicate_evaluations(
        self,
        evaluations: list[CalibrationEvaluationProtocol],
        tolerance: float,
    ) -> list[CalibrationEvaluationProtocol]: ...
    def generate_predictions_parallel(
        self,
        engine: DifferenceEquationsProtocol,
        parameter_sets: list[list[float]],
        parameter_names: list[str],
        time_steps: int,
    ) -> list[list[list[float]]]: ...
    def select_cluster_representatives(
        self,
        evaluations: list[CalibrationEvaluationProtocol],
        cluster_labels: list[int],
        max_representatives: int,
        elite_fraction: float,
        strategy: str,
        selection_method: str = "crowding_distance",
        quality_temperature: float = 1.0,
        seed: int = 42,
        k_neighbors_min: int = 5,
        k_neighbors_max: int = 10,
        sparsity_weight: float = 2.0,
        stratum_fit_weight: float = 10.0,
    ) -> list[int]: ...

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
    Population: type[PopulationProtocol]
    DiseaseState: type[DiseaseStateProtocol]
    Stratification: type[StratificationProtocol]
    StratificationCondition: type[StratificationConditionProtocol]
    StratifiedRate: type[StratifiedRateProtocol]
    Transition: type[TransitionProtocol]
    Parameter: type[ParameterProtocol]
    InitialConditions: type[InitialConditionsProtocol]
    StratificationFraction: type[StratificationFractionProtocol]
    StratificationFractions: type[StratificationFractionsProtocol]
    Condition: type[ConditionProtocol]
    Rule: type[RuleProtocol]
    LogicOperator: type[LogicOperators]
    ModelTypes: type[ModelTypes]
    VariablePrefixes: type[VariablePrefixes]
    Dynamics: type[DynamicsProtocol]
    MathExpression: type[MathExpressionProtocol]
    RateMathExpression: type[RateMathExpressionProtocol]

class RustEpiModelModule(Protocol):
    core: CoreModule
    difference: DifferenceModule
    calibration: CalibrationModule

core: CoreModule
difference: DifferenceModule
calibration: CalibrationModule
rust_epimodel: RustEpiModelModule
