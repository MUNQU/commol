"""Transitions of a process and their stratified rates."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from commol._model.stratification_condition import StratificationCondition
from commol.commol_rs import _commol_rs as commol_rs


def _checked_rate(value: str) -> str:
    """``value`` once the parser of the engine accepts it as a formula of the
    language (``commol_rs.core.MathExpression.check``)."""
    commol_rs.core.MathExpression(value).check()
    return value


class StratifiedRate(BaseModel):
    """
    The rate of the compartments of a transition that match some categories.

    ``ModelBuilder.add_transition`` takes a list of them as
    ``stratified_rates``. A compartment takes the rate of the stratified rate
    whose conditions it matches; the most specific match wins.

    Attributes
    ----------
    conditions : list[StratificationCondition]
        The categories that a compartment must match, at least one, each
        stratification at most once; the engine checks them when the model
        is built.
    rate : str | float
        The rate of the matching compartments: a formula, a parameter or a
        number, which is kept as its string. It can read ``$compartment``
        when the transition expands over its compartments.
    absolute : bool | None
        True: the flow is the rate, not multiplied by the source amount.
        False: the flow is the rate times the source amount. None: the rate
        is absolute when it reads a compartment, a bin total or a
        ``total_…`` variable.
    """

    model_config = ConfigDict(extra="forbid")

    conditions: list[StratificationCondition] = Field(
        default=..., description="Stratification conditions that must match"
    )
    rate: str | float = Field(
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

    @field_validator("rate")
    @classmethod
    def _validate_rate(cls, value: str | float) -> str:
        """The rate as its string, once the parser of the engine accepts it as
        the builder writes it: with ``$compartment`` set to the name of a
        compartment."""
        text = str(value)
        _checked_rate(text.replace("$compartment", "compartment"))
        return text


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
    accumulators : list[str]
        Ids of the accumulators that add up the flow of this transition.
    rate : str | None
        Default mathematical formula, parameter name, or constant value for the flow.
        Used when no stratified rate matches. ``ModelBuilder.add_transition``
        writes a number as its string.

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
    per_compartment : bool | None
        When True, the id of each source and target bin in a rate reads the
        compartment of each flow; any other bin keeps its total. A transition
        between categories ignores it.
    impulse_at : list[float] | None
        Positions on the axis where the transition is an impulse: at each one,
        the rate is a fraction (proportional) or an amount (absolute) moved at
        once. Between them the transition moves nothing.
    impulse_every : tuple[float, float] | None
        ``(period, offset)``: the transition is an impulse at ``offset``,
        ``offset + period``, ``offset + 2 * period``, and so on. At most one
        of `impulse_at` and `impulse_every` is set.
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
            "or a mathematical expression (e.g., 'k1 * A * B / total')."
        ),
    )

    stratified_rates: list[StratifiedRate] | None = Field(
        default=None, description="List of stratification-specific rates"
    )

    per_compartment: bool | None = Field(
        default=None,
        description=(
            "When True, the id of each source and target bin in a rate reads "
            "the compartment of each flow; any other bin keeps its total. A "
            "transition between categories ignores it."
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
    def _validate_impulse(self) -> Self:
        """
        Check the impulse with the rule of the engine (`check_impulse` of
        `crates/commol-model/src/rules.rs`): one of `impulse_at` and
        `impulse_every`, at least one position on the axis, a positive period
        and an offset that is not negative.
        """
        commol_rs.model.check_impulse(self.id, self.impulse_at, self.impulse_every)
        return self

    @field_validator("rate")
    @classmethod
    def _validate_rate(cls, value: str | None) -> str | None:
        """Check the rate, when there is one, with the parser of the engine."""
        if value is None:
            return value
        return _checked_rate(value)
