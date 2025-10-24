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
    ) -> None: ...
    @property
    def time_step(self) -> int: ...
    @property
    def compartment(self) -> str: ...
    @property
    def value(self) -> float: ...

class CalibrationParameterProtocol(Protocol):
    def __init__(
        self,
        id: str,
        min_bound: float,
        max_bound: float,
        initial_guess: float | None = None,
    ) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def min_bound(self) -> float: ...
    @property
    def max_bound(self) -> float: ...

class LossConfigProtocol(Protocol):
    @staticmethod
    def sum_squared_error() -> LossConfigProtocol: ...
    @staticmethod
    def root_mean_squared_error() -> LossConfigProtocol: ...
    @staticmethod
    def mean_absolute_error() -> LossConfigProtocol: ...
    @staticmethod
    def weighted_sse() -> LossConfigProtocol: ...

class NelderMeadConfigProtocol(Protocol):
    def __init__(
        self,
        max_iterations: int = 1000,
        sd_tolerance: float = 1e-6,
        alpha: float | None = None,
        gamma: float | None = None,
        rho: float | None = None,
        sigma: float | None = None,
        verbose: bool = False,
        header_interval: int = 100,
    ) -> None: ...

class ParticleSwarmConfigProtocol(Protocol):
    def __init__(
        self,
        num_particles: int = 20,
        max_iterations: int = 1000,
        target_cost: float | None = None,
        inertia_factor: float | None = None,
        cognitive_factor: float | None = None,
        social_factor: float | None = None,
        verbose: bool = False,
        header_interval: int = 100,
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

class CalibrationModule(Protocol):
    ObservedDataPoint: type[ObservedDataPointProtocol]
    CalibrationParameter: type[CalibrationParameterProtocol]
    LossConfig: type[LossConfigProtocol]
    NelderMeadConfig: type[NelderMeadConfigProtocol]
    ParticleSwarmConfig: type[ParticleSwarmConfigProtocol]
    OptimizationConfig: type[OptimizationConfigProtocol]
    CalibrationResult: type[CalibrationResultProtocol]

    def calibrate(
        self,
        engine: DifferenceEquationsProtocol,
        observed_data: list[ObservedDataPointProtocol],
        parameters: list[CalibrationParameterProtocol],
        loss_config: LossConfigProtocol,
        optimization_config: OptimizationConfigProtocol,
    ) -> CalibrationResultProtocol: ...

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
