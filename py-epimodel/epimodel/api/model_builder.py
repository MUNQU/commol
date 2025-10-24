import copy
import logging
from typing import Literal, Self, TypedDict

from epimodel.context.disease_state import DiseaseState
from epimodel.context.dynamics import (
    Condition,
    Dynamics,
    Rule,
    StratificationCondition,
    StratifiedRate,
    Transition,
)
from epimodel.context.initial_conditions import (
    DiseaseStateFraction,
    InitialConditions,
    StratificationFraction,
    StratificationFractions,
)
from epimodel.context.model import Model
from epimodel.context.parameter import Parameter
from epimodel.context.population import Population
from epimodel.context.stratification import Stratification
from epimodel.constants import ModelTypes, LogicOperators


logger = logging.getLogger(__name__)


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


class DiseaseStateFractionDict(TypedDict):
    """Type definition for a single disease state fraction."""

    disease_state: str
    fraction: float


class StratificationFractionDict(TypedDict):
    """Type definition for a single stratification category fraction."""

    category: str
    fraction: float


class StratificationFractionsDict(TypedDict):
    """Type definition for stratification fractions dictionary."""

    stratification: str
    fractions: list[StratificationFractionDict]


class StratificationConditionDict(TypedDict):
    """Type definition for a stratification condition in a stratified rate."""

    stratification: str
    category: str


class StratifiedRateDict(TypedDict):
    """Type definition for a stratified rate."""

    conditions: list[StratificationConditionDict]
    rate: str | float


class ModelBuilder:
    """
    A programmatic interface for building compartment models.

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
        description: str | None = None,
        version: str | None = None,
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
        self._description: str | None = description
        self._version: str | None = version

        self._disease_states: list[DiseaseState] = []
        self._stratifications: list[Stratification] = []
        self._transitions: list[Transition] = []
        self._parameters: list[Parameter] = []
        self._initial_conditions: InitialConditions | None = None

        logging.info(
            (
                f"Initialized ModelBuilder: name='{self._name}', "
                f"version='{self._version or 'N/A'}'"
            )
        )

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
        """
        self._disease_states.append(DiseaseState(id=id, name=name))
        logging.info(f"Added disease state: id='{id}', name='{name}'")
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
        """
        self._stratifications.append(Stratification(id=id, categories=categories))
        logging.info(f"Added stratification: id='{id}', categories={categories}")
        return self

    def add_parameter(
        self,
        id: str,
        value: float,
        description: str | None = None,
        unit: str | None = None,
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
        unit : str | None, default=None
            Unit of the parameter (e.g., "1/day", "dimensionless", "person").
            Used for unit consistency checking in equations.

        Returns
        -------
        ModelBuilder
            Self for method chaining.
        """
        self._parameters.append(
            Parameter(id=id, value=value, description=description, unit=unit)
        )
        logging.info(f"Added parameter: id='{id}', value={value}, unit='{unit}'")
        return self

    def add_transition(
        self,
        id: str,
        source: list[str],
        target: list[str],
        rate: str | float | None = None,
        stratified_rates: list[StratifiedRateDict] | None = None,
        condition: Condition | None = None,
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
        rate : str | float | None, default=None
            Default mathematical formula, parameter reference, or constant value for
            the transition rate. Used when no stratified rate matches.
            Can be:
            - A parameter reference (e.g., "beta")
            - A constant value (e.g., "0.5" or 0.5)
            - A mathematical formula (e.g., "beta * S * I / N")

            Special variables available in formulas:
            - N: Total population (automatically calculated)
            - step or t: Current simulation step (both are equivalent)
            - pi, e: Mathematical constants

        stratified_rates : list[dict] | None, default=None
            List of stratification-specific rates. Each dict must contain:
            - "conditions": List of dicts with "stratification" and "category" keys
            - "rate": Rate expression string

        condition : Condition| None, default=None
            Logical conditions that must be met for the transition.

        Returns
        -------
        ModelBuilder
            Self for method chaining.
        """
        # Convert rate to string if numeric
        if isinstance(rate, int) or isinstance(rate, float):
            rate = str(rate)

        # Convert stratified rates dicts to Pydantic objects
        stratified_rates_objects: list[StratifiedRate] | None = None
        if stratified_rates:
            stratified_rates_objects = []
            for rate_dict in stratified_rates:
                conditions = [
                    StratificationCondition(**cond) for cond in rate_dict["conditions"]
                ]
                stratified_rates_objects.append(
                    StratifiedRate(conditions=conditions, rate=str(rate_dict["rate"]))
                )

        self._transitions.append(
            Transition(
                id=id,
                source=source,
                target=target,
                rate=rate,
                stratified_rates=stratified_rates_objects,
                condition=condition,
            )
        )
        logging.info(
            (
                f"Added transition: id='{id}', source={source}, target={target}, "
                f"rate='{rate}', stratified_rates={
                    len(stratified_rates_objects) if stratified_rates_objects else 0
                }"
            )
        )
        return self

    def create_condition(
        self,
        logic: Literal[LogicOperators.AND, LogicOperators.OR],
        rules: list[RuleDict],
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
        >>> condition = builder.create_condition(
        ...     "and",
        ...     [
        ...         {"variable": "states:I", "operator": "gt", "value": 0},
        ...         {"variable": "strati:age", "operator": "eq", "value": "adult"},
        ...     ],
        ... )
        """
        rule_objects: list[Rule] = []
        for rule_dict in rules:
            rule_objects.append(
                Rule(
                    variable=rule_dict["variable"],
                    operator=rule_dict["operator"],
                    value=rule_dict["value"],
                )
            )

        return Condition(logic=logic, rules=rule_objects)

    def set_initial_conditions(
        self,
        population_size: int,
        disease_state_fractions: list[DiseaseStateFractionDict],
        stratification_fractions: list[StratificationFractionsDict] | None = None,
    ) -> Self:
        """
        Set the initial conditions for the model.

        Parameters
        ----------
        population_size : int
            Total population size.
        disease_state_fractions : list[DiseaseStateFractionDict]
            List of disease state fractions. Each item is a dictionary with:
            - "disease_state": str (disease state id)
            - "fraction": float (fractional size)

            Example:
            [
                {"disease_state": "S", "fraction": 0.99},
                {"disease_state": "I", "fraction": 0.01},
                {"disease_state": "R", "fraction": 0.0}
            ]
        stratification_fractions : list[StratificationFractionsDict] | None,
            default=None
            List of stratification fractions. Each item is a dictionary with:
            - "stratification": str (stratification id)
            - "fractions": list of dicts, each with "category" and "fraction"

            Example:
            [
                {
                    "stratification": "age_group",
                    "fractions": [
                        {"category": "young", "fraction": 0.3},
                        {"category": "adult", "fraction": 0.5},
                        {"category": "elderly", "fraction": 0.2}
                    ]
                }
            ]

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

        disease_state_fractions_list: list[DiseaseStateFraction] = []
        for ds_dict in disease_state_fractions:
            disease_state_fractions_list.append(
                DiseaseStateFraction(
                    disease_state=ds_dict["disease_state"],
                    fraction=ds_dict["fraction"],
                )
            )

        strat_fractions_list: list[StratificationFractions] = []
        if stratification_fractions:
            for strat_dict in stratification_fractions:
                fractions_list: list[StratificationFraction] = []
                for frac_dict in strat_dict["fractions"]:
                    fractions_list.append(
                        StratificationFraction(
                            category=frac_dict["category"],
                            fraction=frac_dict["fraction"],
                        )
                    )
                strat_fractions_list.append(
                    StratificationFractions(
                        stratification=strat_dict["stratification"],
                        fractions=fractions_list,
                    )
                )

        self._initial_conditions = InitialConditions(
            population_size=population_size,
            disease_state_fractions=disease_state_fractions_list,
            stratification_fractions=strat_fractions_list,
        )
        state_ids = [ds["disease_state"] for ds in disease_state_fractions]
        logging.info(
            (
                f"Set initial conditions: population_size={population_size}, "
                f"states={state_ids}"
            )
        )
        return self

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
            "has_initial_conditions": self._initial_conditions is not None,
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

    def build(self, typology: Literal[ModelTypes.DIFFERENCE_EQUATIONS]) -> Model:
        """
        Build and return the final Model instance.

        Parameters
        ----------
        typology : Literal["DifferenceEquations"]
            Type of the model.

        Returns
        -------
        Model
            The constructed compartment model.

        Raises
        ------
        ValueError
            If validation fails or required components are missing.
        """
        if self._initial_conditions is None:
            raise ValueError("Initial conditions must be set")

        population = Population(
            disease_states=self._disease_states,
            stratifications=self._stratifications,
            transitions=self._transitions,
            initial_conditions=self._initial_conditions,
        )

        dynamics = Dynamics(typology=typology, transitions=self._transitions)

        model = Model(
            name=self._name,
            description=self._description,
            version=self._version,
            population=population,
            parameters=self._parameters,
            dynamics=dynamics,
        )

        logging.info(
            f"Model '{self._name}' successfully built with typology '{typology}'."
        )

        return model
