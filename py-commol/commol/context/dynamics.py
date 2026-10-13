"""Transitions, stratified rates and the dynamics of a model."""

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from commol.commol_rs import _commol_rs as commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e
from commol.constants import ModelTypes
from commol.context.stratification_condition import StratificationCondition
from commol.utils.security import validate_expression_security


class StratifiedRate(BaseModel):
    """
    Defines a rate for specific stratification categories.

    Attributes
    ----------
    conditions : list[StratificationCondition]
        List of stratification-category pairs that must match
    rate : str
        Rate expression for compartments matching these conditions
    """

    conditions: list[StratificationCondition] = Field(
        default=..., description="Stratification conditions that must match"
    )
    rate: str = Field(
        default=..., description="Rate expression for matching compartments"
    )
    absolute: bool | None = Field(
        default=None,
        description=(
            "When True, the flow is treated as absolute (not multiplied by source "
            "population). When None, inferred from whether the rate references "
            "any compartment variable."
        ),
    )

    @field_validator("conditions")
    @classmethod
    def validate_not_empty(
        cls, v: list[StratificationCondition]
    ) -> list[StratificationCondition]:
        """Ensure at least one stratification condition is specified."""
        if not v:
            raise ValueError(
                "conditions must contain at least one stratification condition"
            )
        return v

    @model_validator(mode="after")
    def validate_unique_stratifications(self) -> Self:
        """Ensure no duplicate stratification IDs."""
        strat_ids = [sc.stratification for sc in self.conditions]
        if len(strat_ids) != len(set(strat_ids)):
            duplicates = [s for s in strat_ids if strat_ids.count(s) > 1]
            raise ValueError(
                (
                    f"Duplicate stratifications found: {list(set(duplicates))}. "
                    f"Each stratification can only be specified once per rate."
                )
            )
        return self

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, value: str) -> str:
        """Perform security and syntax validation on the rate expression."""
        try:
            validate_expression_security(value)
            commol_rs.core.MathExpression(value).validate()
        except ValueError as e:
            raise ValueError(f"Validation failed for rate '{value}': {e}")
        return value


class Transition(BaseModel):
    """
    Defines a rule for system evolution.

    Attributes
    ----------
    id : str
        Id of the transition.
    source : list[str]
        The origin compartments.
    target : list[str]
        The destination compartments.
    rate : str | None
        Default mathematical formula, parameter name, or constant value for the flow.
        Used when no stratified rate matches. Numeric values are automatically
        converted to strings during validation.

        Operators: +, -, *, /, % (modulo), ^ or ** (power)
        Functions: sin, cos, tan, exp, ln, sqrt, abs, min, max, if, etc.
        Constants: pi, e

        Note: Both ^ and ** are supported for exponentiation (** is converted to ^).

        Examples:
        - "beta" (parameter reference)
        - "0.5" (constant, can also be passed as float 0.5)
        - "beta * S * I / N" (mathematical formula)
        - "0.3 * sin(2 * pi * t / 365)" (time-dependent formula)
        - "2^10" or "2**10" (power: both syntaxes work)
    stratified_rates : list[StratifiedRate] | None
        Stratification-specific rates. Each rate applies to compartments that match
        all specified stratification conditions.
    """

    id: str = Field(default=..., description="Id of the transition.")
    source: list[str] = Field(default=..., description="Origin compartments.")
    target: list[str] = Field(default=..., description="Destination compartments.")
    accumulators: list[str] = Field(
        default_factory=list,
        description="Cumulative event counters incremented by this transition flow.",
    )

    rate: str | None = Field(
        None,
        description=(
            "Default rate expression (fallback when no stratified rate matches). "
            "Can be a parameter reference (e.g., 'beta'), a constant (e.g., '0.5'), "
            "or a mathematical expression (e.g., 'beta * S * I / N'). "
            "Numeric values are automatically converted to strings during validation."
        ),
    )

    stratified_rates: list[StratifiedRate] | None = Field(
        default=None, description="List of stratification-specific rates"
    )

    per_compartment: bool | None = Field(
        default=None,
        description=(
            "When True, base compartment names in the rate expression "
            "are replaced with the specific stratified compartment name "
            "for each expanded transition flow."
        ),
    )

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, value: str | None) -> str | None:
        """
        Convert numeric rates to strings and perform security and syntax validation.
        """
        if value is None:
            return value
        try:
            validate_expression_security(value)
            commol_rs.core.MathExpression(value).validate()
        except ValueError as e:
            raise ValueError(f"Validation failed for rate '{value}': {e}")
        return value


class Dynamics(BaseModel):
    """
    Defines how the system evolves.

    Attributes
    ----------
    typology : Literal["DifferenceEquations", "DifferentialEquations"]
        The type of model.
    transitions : List[Transition]
        A list of rules for state changes.
    """

    typology: Literal[
        ModelTypes.DIFFERENCE_EQUATIONS, ModelTypes.DIFFERENTIAL_EQUATIONS
    ]
    transitions: list[Transition]

    @field_validator("transitions")
    @classmethod
    def validate_transitions_not_empty(cls, v: list[Transition]) -> list[Transition]:
        """Require at least one transition."""
        if not v:
            raise ValueError("At least one transition must be defined.")
        return v

    @model_validator(mode="after")
    def validate_unique_transition_ids(self) -> Self:
        """
        Validates that transition IDs are unique.
        """
        transition_ids = [t.id for t in self.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            duplicates = [
                item for item in set(transition_ids) if transition_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate transition IDs found: {duplicates}")
        return self
