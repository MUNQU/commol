from pydantic import BaseModel, Field, field_validator

from epimodel.context.condition import Condition
from epimodel.utils.security import validate_expression_security


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
        Mathematical formula, parameter name, or constant value for the flow between
        compartments. Numeric values are automatically converted to strings during
        validation.

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
    condition : Condition | None
        Logical restrictions for the transition.
    """

    id: str = Field(..., description="Id of the transition.")
    source: list[str] = Field(..., description="Origin compartments.")
    target: list[str] = Field(..., description="Destination compartments.")

    rate: str | None = Field(
        None,
        description=(
            "Mathematical formula, parameter name, or constant value for the flow. "
            "Can be a parameter reference (e.g., 'beta'), a constant (e.g., '0.5'), "
            "or a mathematical expression (e.g., 'beta * S * I / N'). "
            "Numeric values are automatically converted to strings during validation."
        ),
    )

    condition: Condition | None = Field(
        None, description="Logical restrictions for the transition."
    )

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, value: str | None) -> str | None:
        """Convert numeric rates to strings and perform security validation."""
        if value is None:
            return value
        try:
            validate_expression_security(value)
        except ValueError as e:
            raise ValueError(f"Security validation failed for rate '{value}': {e}")
        return value
