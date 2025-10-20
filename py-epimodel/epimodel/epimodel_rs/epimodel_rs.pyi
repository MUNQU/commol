from typing import Protocol

from epimodel.constants import LogicOperators, ModelTypes, VariablePrefixes

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

core: CoreModule
difference: DifferenceModule
rust_epimodel: RustEpiModelModule
