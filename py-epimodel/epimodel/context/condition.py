from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from epimodel.constants import LogicOperators, VariablePrefixes


PREFIX_SEPARATOR: str = ":"
VALID_PREFIXES = [el.value for el in VariablePrefixes]


class Rule(BaseModel):
    """
    A simple logical rule that compares a variable with a value.

    Attributes
    ----------
    variable : str
        The variable to evaluate. Must follow the format '<prefix>:<variable_id>'. The 
        allowed prefixes are: ['states', 'strati'].
    operator : Literal["eq", "neq", "gt", "get", "lt", "let"]
        The comparison operator.
    value : Union[str, int, float, bool]
        The value to which the variable is compared.
    """
    variable: str = Field(
        ..., 
        description=(
            "Variable to evaluate. Must follow the format '<prefix>:<variable_id>'. "
            "The allowed prefixes are: ['states', 'strati']."
        )
    )
    operator: Literal[
        LogicOperators.EQ, 
        LogicOperators.NEQ, 
        LogicOperators.GT,
        LogicOperators.GET,
        LogicOperators.LT,
        LogicOperators.LET,
    ] = Field(
        ...,
        description="Comparison operator."
    )
    value: str | int | float | bool = Field(
        ..., 
        description="Value to which the variable is compared."
    )
    
    @model_validator(mode="after")
    def validate_variable_predecessor(self) -> Self:
        """
        Enforces that the 'variable' field is correctly formatted as 
        '<prefix>:<id>' and uses an allowed prefix.
        """
        parts = self.variable.split(PREFIX_SEPARATOR, 1)

        if len(parts) != 2:
            raise ValueError((
                f"Variable '{self.variable}' must contain exactly one "
                f"'{PREFIX_SEPARATOR}' to separate the predecessor and the id."
            ))

        predecessor, var_id = parts

        if predecessor not in VALID_PREFIXES:
            raise ValueError((
                f"Variable predecessor must be one of {VALID_PREFIXES}. "
                f"Found '{predecessor}' in variable '{self.variable}'."
            ))

        if not var_id:
            raise ValueError(
                f"Variable id must not be empty. Found empty id in '{self.variable}'."
            )

        return self


class Condition(BaseModel):
    """
    Defines a set of logical restrictions for a transition.

    Attributes
    ----------
    logic : Literal["and", "or"]
        How to combine the rules.
    rules : List[Rule]
        A list of rules that make up the condition.
    """
    logic: Literal[LogicOperators.AND, LogicOperators.OR] = Field(
        ..., description="How to combine the rules. Allowed operators: ['and', 'or']"
    )
    rules: list[Rule]
