"""Transitions of a process and their stratified rates."""

import math
from typing import Self

from pydantic import ConfigDict, BaseModel, Field, field_validator, model_validator

from commol.commol_rs import _commol_rs as commol_rs
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

    model_config = ConfigDict(extra="forbid")

    conditions: list[StratificationCondition] = Field(
        default=..., description="Stratification conditions that must match"
    )
    rate: str = Field(
        default=..., description="Rate expression for matching compartments"
    )
    absolute: bool | None = Field(
        default=None,
        description=(
            "When True, the flow is treated as absolute (not multiplied by the "
            "source amount). When None, inferred from whether the rate references "
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
        - "k1" (parameter reference)
        - "0.5" (constant, can also be passed as float 0.5)
        - "k1 * A * B / total" (mathematical formula)
        - "0.3 * sin(2 * pi * t / 365)" (formula that changes along the axis)
        - "2^10" or "2**10" (power: both syntaxes work)
    stratified_rates : list[StratifiedRate] | None
        Stratification-specific rates. Each rate applies to compartments that match
        all specified stratification conditions.
    """

    model_config = ConfigDict(extra="forbid")

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
            "Can be a parameter reference (e.g., 'k1'), a constant (e.g., '0.5'), "
            "or a mathematical expression (e.g., 'k1 * A * B / total'). "
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

    impulse_at: list[float] | None = Field(
        default=None,
        description=(
            "Positions on the axis where the transition is an impulse: at each "
            "one, the rate is a fraction (proportional) or an amount (absolute) "
            "moved at once. Between them the transition moves nothing."
        ),
    )

    impulse_every: tuple[float, float] | None = Field(
        default=None,
        description=(
            "(period, offset): the transition is an impulse at offset, "
            "offset + period, offset + 2 * period, ..."
        ),
    )

    @property
    def is_impulse(self) -> bool:
        """Whether the transition is an impulse rather than a continuous flow."""
        return self.impulse_at is not None or self.impulse_every is not None

    @model_validator(mode="after")
    def validate_impulse(self) -> Self:
        """Check the impulse schedule: one kind, valid positions and period."""
        if self.impulse_at is not None and self.impulse_every is not None:
            raise ValueError(
                f"Transition '{self.id}': set impulse_at or impulse_every, not both."
            )
        if self.impulse_at is not None:
            if not self.impulse_at:
                raise ValueError(
                    f"Transition '{self.id}': impulse_at needs at least one position."
                )
            bad = [p for p in self.impulse_at if not math.isfinite(p) or p < 0]
            if bad:
                raise ValueError(
                    f"Transition '{self.id}': impulse positions must be finite "
                    f"and not negative, got {bad}."
                )
        if self.impulse_every is not None:
            period, offset = self.impulse_every
            if not (math.isfinite(period) and period > 0):
                raise ValueError(
                    f"Transition '{self.id}': the impulse period must be "
                    f"positive, got {period}."
                )
            if not (math.isfinite(offset) and offset >= 0):
                raise ValueError(
                    f"Transition '{self.id}': the impulse offset must be finite "
                    f"and not negative, got {offset}."
                )
        return self

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
