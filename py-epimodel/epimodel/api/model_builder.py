import copy
from typing import Literal, Self, TypedDict

from epimodel.context.condition import Condition, Rule
from epimodel.context.disease_state import DiseaseState
from epimodel.context.dynamics import Dynamics
from epimodel.context.initial_conditions import InitialConditions
from epimodel.context.model import Model
from epimodel.context.parameter import Parameter
from epimodel.context.population import Population
from epimodel.context.stratification import Stratification
from epimodel.context.transition import Transition
from epimodel.constants import ModelTypes, LogicOperators


class RuleDict(TypedDict):
    """Type definition for rule dictionary used in create_condition method."""
    variable: str
    operator: Literal[
        LogicOperators.EQ, 
        LogicOperators.NEQ, 
        LogicOperators.GT,
        LogicOperators.GET,
        LogicOperators.LT,
        LogicOperators.LET,
    ]
    value: str | int | float | bool


class ModelBuilder:
    """
    A programmatic interface for building epidemiological models.
    
    This class provides a fluent API for constructing Model instances by progressively
    adding disease states, stratifications, transitions, parameters, and 
    initial conditions. It includes validation methods to ensure model consistency 
    before building.
    
    Attributes
    ----------
    _name : str
        The model name.
    _description : str | None
        The model description.
    _version : str | None
        The model version.
    _disease_states : list[DiseaseState]
        List of disease states in the model.
    _stratifications : list[Stratification]
        List of population stratifications.
    _transitions : list[Transition]
        List of transitions between states.
    _parameters : list[Parameter]
        List of model parameters.
    _initial_conditions : InitialConditions | None
        Initial population conditions.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str| None = None, 
        version: str| None = None,
    ):
        """
        Initialize the ModelBuilder.
        
        Parameters
        ----------
        name : str
            The unique name that identifies the model.
        description : str | None, default=None
            A human-readable description of the model's purpose and function.
        version : str | None, default=None
            The version number of the model.
        """
        self._name: str = name
        self._description: str| None = description
        self._version: str| None = version
        
        self._disease_states: list[DiseaseState] = []
        self._stratifications: list[Stratification] = []
        self._transitions: list[Transition] = []
        self._parameters: list[Parameter] = []
        self._initial_conditions: InitialConditions | None = None
    
    def add_disease_state(self, id: str, name: str) -> Self:
        """
        Add a disease state to the model.
        
        Parameters
        ----------
        id : str
            Unique identifier for the disease state.
        name : str
            Human-readable name for the disease state.
            
        Returns
        -------
        ModelBuilder
            Self for method chaining.
            
        Raises
        ------
        ValueError
            If a disease state with the same id already exists.
        """
        if any(state.id == id for state in self._disease_states):
            raise ValueError(f"Disease state with id '{id}' already exists")
        
        self._disease_states.append(DiseaseState(id=id, name=name))
        return self
    
    def add_stratification(self, id: str, categories: list[str]) -> Self:
        """
        Add a population stratification to the model.
        
        Parameters
        ----------
        id : str
            Unique identifier for the stratification.
        categories : list[str]
            list of category identifiers within this stratification.
            
        Returns
        -------
        ModelBuilder
            Self for method chaining.
            
        Raises
        ------
        ValueError
            If a stratification with the same id already exists or if categories are 
            empty.
        """
        if any(strat.id == id for strat in self._stratifications):
            raise ValueError(f"Stratification with id '{id}' already exists")
        
        if not categories:
            raise ValueError("Categories cannot be empty")
        
        self._stratifications.append(Stratification(id=id, categories=categories))
        return self
    
    def add_parameter(
        self, 
        id: str, 
        value: float, 
        description: str | None = None,
    ) -> Self:
        """
        Add a global parameter to the model.
        
        Parameters
        ----------
        id : str
            Unique identifier for the parameter.
        value : float
            Numerical value of the parameter.
        description : str | None, default=None
            Human-readable description of the parameter.
            
        Returns
        -------
        ModelBuilder
            Self for method chaining.
            
        Raises
        ------
        ValueError
            If a parameter with the same id already exists.
        """
        if any(param.id == id for param in self._parameters):
            raise ValueError(f"Parameter with id '{id}' already exists")
        
        self._parameters.append(
            Parameter(id=id, value=value, description=description)
        )
        return self
    
    def add_transition(
        self,
        id: str,
        source: list[str],
        target: list[str],
        rate: str | None = None,
        condition: Condition| None = None
    ) -> Self:
        """
        Add a transition between states to the model.
        
        Parameters
        ----------
        id : str
            Unique identifier for the transition.
        source : list[str]
            List of source state/category identifiers.
        target : list[str]
            List of target state/category identifiers.
        rate : float | str | None, default=None
            Mathematical formula for the transition rate.
        condition : Condition| None, default=None
            Logical conditions that must be met for the transition.
            
        Returns
        -------
        ModelBuilder
            Self for method chaining.
            
        Raises
        ------
        ValueError
            If a transition with the same id already exists.
        """
        if any(trans.id == id for trans in self._transitions):
            raise ValueError(f"Transition with id '{id}' already exists")
        
        self._transitions.append(Transition(
            id=id,
            source=source,
            target=target,
            rate=rate,
            condition=condition
        ))
        return self
    
    def create_condition(
        self,
        logic: Literal[LogicOperators.AND, LogicOperators.OR],
        rules: list[RuleDict]
    ) -> Condition:
        """
        Create a condition object for use in transitions.
        
        Parameters
        ----------
        logic : Literal["and", "or"]
            How to combine the rules.
        rules : List[RuleDict]
            List of rule dictionaries with 'variable', 'operator', and 'value' keys.
            Each dictionary must have:
            - 'variable': str (format '<prefix>:<variable_id>')
            - 'operator': Literal["eq", "neq", "gt", "get", "lt", "let"]
            - 'value': str | int | float | bool

        Returns
        -------
        Condition
            The created condition object.

        Examples
        --------
        >>> condition = builder.create_condition("and", [
        ...     {"variable": "states:I", "operator": "gt", "value": 0},
        ...     {"variable": "strati:age", "operator": "eq", "value": "adult"}
        ... ])
        """
        rule_objects: list[Rule] = []
        for rule_dict in rules:
            rule_objects.append(Rule(
                variable=rule_dict["variable"],
                operator=rule_dict["operator"],
                value=rule_dict["value"]
            ))
        
        return Condition(logic=logic, rules=rule_objects)
    
    def set_initial_conditions(
        self,
        population_size: int,
        disease_state_fractions: dict[str, float],
        stratification_fractions: dict[str, dict[str, float]] | None = None
    ) -> Self:
        """
        Set the initial conditions for the model.
        
        Parameters
        ----------
        population_size : int
            Total population size.
        disease_state_fractions : dict[str, float]
            Fractions of population in each disease state. Keys are state ids, 
            values are fractions.
        stratification_fractions : dict[str, dict[str, float]] | None, default=None
            Fractions for each stratification category. Outer keys are stratification 
            ids, inner keys are category ids, values are fractions.
            
        Returns
        -------
        ModelBuilder
            Self for method chaining.
            
        Raises
        ------
        ValueError
            If initial conditions have already been set.
        """
        if self._initial_conditions is not None:
            raise ValueError("Initial conditions have already been set")

        self._initial_conditions = InitialConditions(
            population_size=population_size,
            disease_state_fraction=disease_state_fractions,
            stratification_fractions=stratification_fractions or {}
        )
        return self
    
    def validate_completeness(self) -> None:
        """
        Validate that all required components have been added to the model.
        
        Raises
        ------
        ValueError
            If any required components are missing.
        """
        if not self._disease_states:
            raise ValueError("At least one disease state must be defined")
        
        if not self._transitions:
            raise ValueError("At least one transition must be defined")
        
        if self._initial_conditions is None:
            raise ValueError("Initial conditions must be set")
    
    def get_summary(self) -> dict[str, str | int | list[str] | None]:
        """
        Get a summary of the current model builder state.
        
        Returns
        -------
        dict[str, dict[str, str | int | list[str] | None]]
            Dictionary containing summary information about the model being built.
        """
        return {
            "name": self._name,
            "description": self._description,
            "version": self._version,
            "disease_states_count": len(self._disease_states),
            "disease_state_ids": [state.id for state in self._disease_states],
            "stratifications_count": len(self._stratifications),
            "stratification_ids": [strat.id for strat in self._stratifications],
            "transitions_count": len(self._transitions),
            "transition_ids": [trans.id for trans in self._transitions],
            "parameters_count": len(self._parameters),
            "parameter_ids": [param.id for param in self._parameters],
            "has_initial_conditions": self._initial_conditions is not None
        }
    
    def clone(self) -> Self:
        """
        Create a deep copy of this ModelBuilder.
        
        Returns
        -------
        ModelBuilder
            A new ModelBuilder instance with the same configuration.
        """

        new_builder = type(self)(self._name, self._description, self._version)

        new_builder._disease_states = copy.deepcopy(self._disease_states)
        new_builder._stratifications = copy.deepcopy(self._stratifications)
        new_builder._transitions = copy.deepcopy(self._transitions)
        new_builder._parameters = copy.deepcopy(self._parameters)
        new_builder._initial_conditions = copy.deepcopy(self._initial_conditions)

        return new_builder
    
    def reset(self) -> Self:
        """
        Reset the builder to empty state while keeping name, description, and version.
        
        Returns
        -------
        ModelBuilder
            Self for method chaining.
        """
        self._disease_states.clear()
        self._stratifications.clear()
        self._transitions.clear()
        self._parameters.clear()
        self._initial_conditions = None
        return self
    
    def build(
        self, 
        typology: Literal[ModelTypes.DIFFERENCE_EQUATIONS],
        validate: bool = True
    ) -> Model:
        """
        Build and return the final Model instance.
        
        Parameters
        ----------
        typology : Literal["DifferenceEquations"]
            Type of the model.
        validate : bool, default=True
            Whether to perform validation before building the model.
            
        Returns
        -------
        Model
            The constructed epidemiological model.
            
        Raises
        ------
        ValueError
            If validation fails or required components are missing.
        """
        if validate:
            self.validate_completeness()
            
        assert self._initial_conditions is not None, (
            "Internal error: Initial conditions should have been validated."
        )
        
        population = Population(
            disease_states=self._disease_states,
            stratifications=self._stratifications,
            transitions=self._transitions,
            initial_conditions=self._initial_conditions
        )
        
        dynamics = Dynamics(
            typology=typology,
            transitions=self._transitions
        )
        
        return Model(
            name=self._name,
            description=self._description,
            version=self._version,
            population=population,
            parameters=self._parameters,
            dynamics=dynamics
        )
