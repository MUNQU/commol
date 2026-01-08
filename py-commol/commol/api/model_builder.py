import copy
import logging
from typing import Literal, Self, TypedDict, cast

from commol.constants import LogicOperators, ModelTypes
from commol.context.bin import Bin
from commol.context.dynamics import (
    Condition,
    Dynamics,
    Rule,
    StratificationCondition,
    StratifiedRate,
    Transition,
)
from commol.context.initial_conditions import (
    BinFraction,
    InitialConditions,
    StratificationFraction,
    StratificationFractions,
)
from commol.context.model import Model
from commol.context.parameter import Parameter
from commol.context.population import Population
from commol.context.stratification import Stratification

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


class BinFractionDict(TypedDict):
    """Type definition for a single bin fraction."""

    bin: str
    fraction: float | None


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
    adding bins, stratifications, transitions, parameters, and
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
    _disease_states : list[Bin]
        List of bins in the model.
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
        bin_unit: str | None = None,
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
        bin_unit : str | None, default=None
            The default unit for all bins.
            Individual bins can override this with their own unit parameter.
            Units are optional but required for model.print_equations() and
            model.check_unit_consistency().
        """
        self._name: str = name
        self._description: str | None = description
        self._version: str | None = version
        self._bin_unit: str | None = bin_unit

        self._bins: list[Bin] = []
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

    def add_bin(self, id: str, name: str, unit: str | None = None) -> Self:
        """
        Add a bin to the model.

        Parameters
        ----------
        id : str
            Unique identifier for the bin.
        name : str
            Human-readable name for the bin.
        unit : str | None, default=None
            Unit of measurement for this bin. If None, uses the model-level bin_unit.

        Returns
        -------
        ModelBuilder
            Self for method chaining.
        """
        # Use bin-specific unit, or fall back to model-level bin_unit
        final_unit = unit if unit is not None else self._bin_unit
        self._bins.append(Bin(id=id, name=name, unit=final_unit))
        logging.info(f"Added bin: id='{id}', name='{name}', unit='{final_unit}'")
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
        value: float | str | None,
        description: str | None = None,
        unit: str | None = None,
    ) -> Self:
        """
        Add a global parameter to the model.

        Parameters
        ----------
        id : str
            Unique identifier for the parameter.
        value : float | str | None
            Value of the parameter. Can be:
            - float: A numerical constant value
            - str: A mathematical formula that can reference other parameters,
                   special variables (N, N_category, step/t, pi, e), or contain
                   mathematical expressions (e.g., "beta * 2", "N_young / N")
            - None: Indicates that the parameter needs to be calibrated before use

            Special variables available in formulas:
            - N: Total population (automatically calculated)
            - N_{category}: Population in specific category (e.g., N_young, N_old)
            - N_{cat1}_{cat2}: Population in category combinations
            - step or t: Current simulation step
            - pi, e: Mathematical constants

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

        This method supports two distinct behaviors:

        When you specify multiple sources without the $compartment placeholder,
        a SINGLE transition is created that affects all sources simultaneously.
        The rate is evaluated ONCE per time step, and that value is applied to
        all source compartments at once.

        Example:
            .add_transition(
                id="interaction",
                source=["S", "I"],
                target=["I", "I"],
                rate="beta * S * I"
            )

        This creates ONE transition where:
        - The rate "beta * S * I" is calculated once
        - That rate removes from both S and I simultaneously
        - That rate adds to I twice (once per target entry)
        - Resulting equations:
          dS/dt = ... - (beta*S*I)
          dI/dt = ... - (beta*S*I) + 2*(beta*S*I) = ... + (beta*S*I)

        When you use the $compartment placeholder in the rate formula with multiple
        sources, the system automatically expands this into multiple independent
        transitions - one for each source compartment. Each transition has its own
        rate calculation where $compartment is replaced with the actual compartment
        name.

        Example:
            .add_transition(
                id="death",
                source=["S", "L", "I", "R"],
                target=[],
                rate="d * $compartment"
            )

        This automatically expands to FOUR separate transitions:
        - Transition 1: S -> [] with rate "d * S"
        - Transition 2: L -> [] with rate "d * L"
        - Transition 3: I -> [] with rate "d * I"
        - Transition 4: R -> [] with rate "d * R"

        Each transition's rate is evaluated independently, giving per-compartment
        dynamics. This is ideal for processes like per-capita death rates, where the
        rate should be proportional to each compartment's population.

        **$compartment Usage Rules**:
        - Only valid when len(source) > 1
        - Target must be empty [] or have exactly one element
        - Can be used with stratified_rates - the placeholder will be expanded in
          both the base rate and all stratified rate expressions
        - The placeholder $compartment will be replaced with each source compartment
          name

        **$compartment Restrictions**:
        - Using $compartment with a single source will raise an error (use the
          compartment name directly instead)
        - Using $compartment with multiple targets will raise an error (ambiguous
          which target corresponds to which source)

        Parameters
        ----------
        id : str
            Unique identifier for the transition. When using $compartment expansion,
            the system will append "__<compartment_name>" to create unique IDs for
            each expanded transition (e.g., "death__S", "death__L").
        source : list[str]
            List of source state/category identifiers. When using $compartment in
            the rate, each source will generate a separate transition.
        target : list[str]
            List of target state/category identifiers. When using $compartment,
            target must be empty [] or have exactly one element.
        rate : str | float | None, default=None
            Default mathematical formula, parameter reference, or constant value for
            the transition rate. Used when no stratified rate matches.
            Can be:
            - A parameter reference (e.g., "beta")
            - A constant value (e.g., "0.5" or 0.5)
            - A mathematical formula (e.g., "beta * S * I / N")
            - A formula with $compartment placeholder (e.g., "d * $compartment")

            Special variables available in formulas:
            - N: Total population (automatically calculated)
            - step or t: Current simulation step (both are equivalent)
            - pi, e: Mathematical constants
            - $compartment: Expands to each source compartment name (only with
              multiple sources)

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

        Raises
        ------
        ValueError
            - If $compartment is used with only one source compartment
            - If $compartment is used with multiple targets
            - If rate is None when using $compartment

        Examples
        --------
        Standard single transition affecting multiple sources:
        >>> builder.add_transition(
        ...     id="infection",
        ...     source=["S", "I"],
        ...     target=["I", "I"],
        ...     rate="beta * S * I / N",
        ... )

        Expanded transitions using $compartment (creates 4 separate transitions):
        >>> builder.add_transition(
        ...     id="death",
        ...     source=["S", "L", "I", "R"],
        ...     target=[],
        ...     rate="d * $compartment",
        ... )

        Per-compartment flow with target:
        >>> builder.add_transition(
        ...     id="treatment",
        ...     source=["I_mild", "I_severe"],
        ...     target=["R"],
        ...     rate="treatment_rate * $compartment",
        ... )
        """
        # Convert rate to string if numeric
        if isinstance(rate, int) or isinstance(rate, float):
            rate = str(rate)

        # Check for $compartment placeholder and handle expansion
        if rate and "$compartment" in rate:
            self._validate_compartment_placeholder(id, source, target)
            self._expand_compartment_transition(
                id, source, target, rate, stratified_rates, condition
            )
            return self

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

    def _validate_compartment_placeholder(
        self,
        id: str,
        source: list[str],
        target: list[str],
    ) -> None:
        """
        Validate that $compartment placeholder is used correctly.

        Parameters
        ----------
        id : str
            Transition identifier (for error messages)
        source : list[str]
            Source compartments
        target : list[str]
            Target compartments
        rate : str
            Rate formula
        stratified_rates : list[StratifiedRateDict] | None
            Stratified rates

        Raises
        ------
        ValueError
            If $compartment usage violates any rules
        """
        # Check that we have either multiple sources OR multiple targets, but not both
        has_multiple_sources = len(source) > 1
        has_multiple_targets = len(target) > 1

        if not has_multiple_sources and not has_multiple_targets:
            raise ValueError(
                (
                    f"Transition '{id}': $compartment placeholder requires either "
                    f"multiple source compartments or multiple target compartments "
                    f"(found {len(source)} source(s) and {len(target)} target(s)). "
                    f"If you have a single source/target, use the compartment name "
                    f"directly in the rate formula instead of $compartment."
                )
            )

        if has_multiple_sources and has_multiple_targets:
            raise ValueError(
                (
                    f"Transition '{id}': $compartment placeholder cannot be used "
                    f"with both multiple sources and multiple targets "
                    f"(found {len(source)} sources and {len(target)} targets). "
                    f"This would create ambiguous mappings. Use either multiple "
                    f"sources with single/empty target, or single/empty source with "
                    f"multiple targets."
                )
            )

    def _expand_compartment_transition(
        self,
        id: str,
        source: list[str],
        target: list[str],
        rate: str,
        stratified_rates: list[StratifiedRateDict] | None,
        condition: Condition | None,
    ) -> None:
        """
        Expand a transition with $compartment placeholder into multiple transitions.

        Parameters
        ----------
        id : str
            Base transition identifier
        source : list[str]
            Source compartments
            (will create one transition per source if multiple sources)
        target : list[str]
            Target compartments
            (will create one transition per target if multiple targets)
        rate : str
            Rate formula containing $compartment
        stratified_rates : list[StratifiedRateDict] | None
            Stratified rates (can contain $compartment which will be expanded)
        condition : Condition | None
            Transition condition
        """
        # Determine if we're expanding over sources or targets
        has_multiple_sources = len(source) > 1
        has_multiple_targets = len(target) > 1

        if has_multiple_sources:
            compartments = source
            expand_type = "source"
        else:
            compartments = target
            expand_type = "target"

        logging.info(
            (
                f"Expanding transition '{id}' with $compartment placeholder: "
                f"{len(compartments)} {expand_type} compartments will create "
                f"{len(compartments)} separate transitions"
            )
        )

        for compartment in compartments:
            # Replace $compartment with actual compartment name in base rate
            expanded_rate = rate.replace("$compartment", compartment)

            # Expand stratified rates if present
            expanded_stratified_rates: list[StratifiedRate] | None = None
            if stratified_rates:
                expanded_stratified_rates = []
                for rate_dict in stratified_rates:
                    # Replace $compartment in stratified rate expression
                    strat_rate_value = rate_dict["rate"]
                    # Convert to string if it's a number
                    if isinstance(strat_rate_value, (int, float)):
                        strat_rate_str = str(strat_rate_value)
                    else:
                        strat_rate_str = strat_rate_value

                    expanded_strat_rate = strat_rate_str.replace(
                        "$compartment", compartment
                    )

                    # Convert conditions to Pydantic objects
                    conditions = [
                        StratificationCondition(**cond)
                        for cond in rate_dict["conditions"]
                    ]

                    expanded_stratified_rates.append(
                        StratifiedRate(conditions=conditions, rate=expanded_strat_rate)
                    )

            # Generate unique ID for this expanded transition
            expanded_id = f"{id}__{compartment}"

            # Create the expanded transition with appropriate source/target
            if has_multiple_sources:
                expanded_source = [compartment]
                expanded_target = target
            elif has_multiple_targets:
                expanded_source = source
                expanded_target = [compartment]
            else:
                raise ValueError(
                    "The transition must have multiple sources or targets."
                )

            self._transitions.append(
                Transition(
                    id=expanded_id,
                    source=expanded_source,
                    target=expanded_target,
                    rate=expanded_rate,
                    stratified_rates=expanded_stratified_rates,
                    condition=condition,
                )
            )

            strat_info = (
                f", {len(expanded_stratified_rates)} stratified rates"
                if expanded_stratified_rates
                else ""
            )
            logging.info(
                (
                    f"  Created expanded transition: id='{expanded_id}', "
                    f"source=['{compartment}'], target={target}, "
                    f"rate='{expanded_rate}'{strat_info}"
                )
            )

        logging.info(
            f"Successfully expanded transition '{id}' into {len(source)} transitions"
        )

    def create_condition(
        self,
        logic: Literal["and", "or"],
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

        return Condition(
            logic=cast(
                Literal[LogicOperators.AND, LogicOperators.OR], LogicOperators(logic)
            ),
            rules=rule_objects,
        )

    def set_initial_conditions(
        self,
        population_size: int,
        bin_fractions: list[BinFractionDict],
        stratification_fractions: list[StratificationFractionsDict] | None = None,
    ) -> Self:
        """
        Set the initial conditions for the model.

        Parameters
        ----------
        population_size : int
            Total population size.
        bin_fractions : list[BinFractionDict]
            List of bin fractions. Each item is a dictionary with:
            - "bin": str (bin id)
            - "fraction": float (fractional size)

            Example:
            [
                {"bin": "S", "fraction": 0.99},
                {"bin": "I", "fraction": 0.01},
                {"bin": "R", "fraction": 0.0}
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

        bin_fractions_list: list[BinFraction] = []
        for bf_dict in bin_fractions:
            bin_fractions_list.append(
                BinFraction(
                    bin=bf_dict["bin"],
                    fraction=bf_dict["fraction"],
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
            bin_fractions=bin_fractions_list,
            stratification_fractions=strat_fractions_list,
        )
        bin_ids = [bf["bin"] for bf in bin_fractions]
        logging.info(
            (
                f"Set initial conditions: population_size={population_size}, "
                f"bins={bin_ids}"
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
            "disease_states_count": len(self._bins),
            "bin_ids": [state.id for state in self._bins],
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

        new_builder._bins = copy.deepcopy(self._bins)
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
        self._bins.clear()
        self._stratifications.clear()
        self._transitions.clear()
        self._parameters.clear()
        self._initial_conditions = None
        return self

    def _validate_typology(self, typology: str) -> ModelTypes:
        """
        Validate and convert typology string to ModelTypes enum.

        Parameters
        ----------
        typology : str
            The model typology as a string.

        Returns
        -------
        ModelTypes
            The validated ModelTypes enum value.

        Raises
        ------
        ValueError
            If the typology string is not a valid ModelTypes value.
        """
        try:
            return ModelTypes(typology)
        except ValueError:
            valid_values = [t.value for t in ModelTypes]
            raise ValueError(
                f"Invalid typology: '{typology}'. Must be one of {valid_values}"
            ) from None

    def build(self, typology: str) -> Model:
        """
        Build and return the final Model instance.

        Parameters
        ----------
        typology : str
            Type of the model. Must be one of the valid ModelTypes values:
            "DifferenceEquations".

        Returns
        -------
        Model
            The constructed compartment model.

        Raises
        ------
        ValueError
            If validation fails, required components are missing,
            or typology is invalid.
        """
        if self._initial_conditions is None:
            raise ValueError("Initial conditions must be set")

        # Validate and convert typology string to enum
        validated_typology = self._validate_typology(typology)

        population = Population(
            bins=self._bins,
            stratifications=self._stratifications,
            transitions=self._transitions,
            initial_conditions=self._initial_conditions,
        )

        dynamics = Dynamics(
            typology=validated_typology,
            transitions=self._transitions,
        )

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
