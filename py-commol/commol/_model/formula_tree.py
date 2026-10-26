"""The tree of a formula, as the parser of the engine reads it.

The engine gives the tree of a formula as JSON
(``commol_rs.core.MathExpression.tree``, whose shape the module ``tree`` of
``crates/commol-core/src/math_expression/`` documents). This module reads it
into plain data, one frozen dataclass per kind of node, and writes a tree back
as text. Python reads a formula only as this tree: the grouping, the numbers
and the names are those of the parser, and the text of a formula is never
split or parsed here.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

# How tightly a node binds when a formula writes it without parentheses,
# from the loosest: a chain of each level, a sign, a power, then a number, a
# variable, a call or an `if`.
COMPARISON = 2
SUM = 3
PRODUCT = 4
SIGN = 5
POWER = 6
PRIMARY = 7


@dataclass(frozen=True, slots=True)
class Constant:
    """A number, with the value the parser read."""

    value: float


@dataclass(frozen=True, slots=True)
class Variable:
    """A variable: a parameter, a bin, a compartment or an engine variable."""

    name: str


@dataclass(frozen=True, slots=True)
class Chain:
    """
    Operators of one level, applied from the left: ``a + b - c`` is
    ``(a + b) - c``.

    Each operator of `rest` (``Add``, ``Sub``, ``Mul``, ``Div``, ``Mod``,
    ``Lt``, ``Gt``, ``Le``, ``Ge``, ``Eq``, ``Ne``) has the level of the
    first, and `rest` holds one pair or more.
    """

    first: "Node"
    rest: tuple[tuple[str, "Node"], ...]


@dataclass(frozen=True, slots=True)
class Power:
    """``base ^ exponent``; a chain of powers groups from the right."""

    base: "Node"
    exponent: "Node"


@dataclass(frozen=True, slots=True)
class UnaryOp:
    """A sign (``Neg``, ``-x``) or a negation of a condition (``Not``, ``!x``)."""

    op: str
    operand: "Node"


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """A call of a function of the language, such as ``max(a, b)``."""

    name: str
    args: tuple["Node", ...]


@dataclass(frozen=True, slots=True)
class Conditional:
    """``if(condition, true_expr, false_expr)``."""

    condition: "Node"
    true_expr: "Node"
    false_expr: "Node"


type Node = Constant | Variable | Chain | Power | UnaryOp | FunctionCall | Conditional
"""A node of the tree of a formula."""


def read_tree(text: str) -> Node:
    """
    The tree of a formula from the JSON of the engine.

    Parameters
    ----------
    text : str
        The JSON of ``commol_rs.core.MathExpression.tree``.

    Returns
    -------
    Node
        The root of the tree.
    """
    return _node(json.loads(text))


def _node(data: object) -> Node:
    """The node of one JSON object of the tree."""
    match data:
        case {"Constant": float() | int() | str() as value}:
            # A value that is not finite comes as its text, such as "Infinity".
            return Constant(float(value))
        case {"Variable": str() as name}:
            return Variable(name)
        case {"Chain": {"first": first, "rest": list() as rest}}:
            return Chain(
                _node(first), tuple((str(op), _node(operand)) for op, operand in rest)
            )
        case {"Power": {"base": base, "exponent": exponent}}:
            return Power(_node(base), _node(exponent))
        case {"UnaryOp": {"op": str() as op, "operand": operand}}:
            return UnaryOp(op, _node(operand))
        case {"FunctionCall": {"name": str() as name, "args": list() as args}}:
            return FunctionCall(name, tuple(_node(arg) for arg in args))
        case {
            "Conditional": {
                "condition": condition,
                "true_expr": true_expr,
                "false_expr": false_expr,
            }
        }:
            return Conditional(_node(condition), _node(true_expr), _node(false_expr))
    raise ValueError(f"Not a node of the tree of a formula: {data!r}")


OPERATORS: dict[str, tuple[str, int]] = {
    "Add": ("+", SUM),
    "Sub": ("-", SUM),
    "Mul": ("*", PRODUCT),
    "Div": ("/", PRODUCT),
    "Mod": ("%", PRODUCT),
    "Lt": ("<", COMPARISON),
    "Gt": (">", COMPARISON),
    "Le": ("<=", COMPARISON),
    "Ge": (">=", COMPARISON),
    "Eq": ("==", COMPARISON),
    "Ne": ("!=", COMPARISON),
}
"""The symbol and the level of each operator of a chain."""


def binding(node: Node) -> int:
    """How tightly `node` binds when a formula writes it without parentheses."""
    match node:
        case Chain(rest=rest):
            return OPERATORS[rest[0][0]][1]
        case UnaryOp():
            return SIGN
        case Power():
            return POWER
    return PRIMARY


def number_text(value: float) -> str:
    """A number as the shortest text that reads back as its value, without a
    decimal part when it has none: ``2``, ``0.5``, ``1e-05``."""
    return repr(value).removesuffix(".0")


class FormulaWriter:
    """
    Writes a tree as text, with the parentheses the tree needs and no other,
    so that the text reads back as the same tree.

    A subclass writes another notation: it changes the symbols, the variables
    and the nodes it writes in another way. `free` says whether a node stands
    outside every parenthesis of the formula.
    """

    SYMBOLS: dict[str, str] = {op: symbol for op, (symbol, _) in OPERATORS.items()}
    """The symbol of each operator of a chain."""

    UNARY: dict[str, str] = {"Neg": "-", "Not": "!"}
    """The symbol of each unary operator."""

    def __init__(self, variable: Callable[[str], str] | None = None) -> None:
        self._variable = variable or (lambda name: name)

    def write(self, node: Node) -> str:
        """The text of `node`, the root of a formula."""
        return self.node(node, free=True)

    def node(self, node: Node, free: bool) -> str:
        """The text of `node`."""
        match node:
            case Constant(value):
                return number_text(value)
            case Variable(name):
                return self._variable(name)
            case Chain(first, rest):
                return self.chain(first, rest, free)
            case Power(base, exponent):
                return self.power(base, exponent)
            case UnaryOp(op, operand):
                group = binding(operand) < SIGN
                return f"{self.UNARY[op]}{self.grouped(operand, group, free)}"
            case FunctionCall(name, args):
                return f"{name}({self._arguments(args)})"
            case Conditional(condition, true_expr, false_expr):
                return f"if({self._arguments((condition, true_expr, false_expr))})"

    def _arguments(self, args: tuple[Node, ...]) -> str:
        return ", ".join(self.node(arg, free=False) for arg in args)

    def grouped(self, node: Node, group: bool, free: bool) -> str:
        """The text of `node`, in parentheses when `group` holds."""
        if group:
            return f"({self.node(node, free=False)})"
        return self.node(node, free)

    def operand(self, node: Node, level: int, free: bool) -> str:
        """An operand of a chain of `level`: a chain of the same level, or of
        a looser one, is a group."""
        return self.grouped(node, binding(node) <= level, free)

    def chain(self, first: Node, rest: tuple[tuple[str, Node], ...], free: bool) -> str:
        """The text of a chain: each operand, with each operator before it."""
        level = OPERATORS[rest[0][0]][1]
        text = self.operand(first, level, free)
        for op, operand in rest:
            text += f" {self.SYMBOLS[op]} {self.operand(operand, level, free)}"
        return text

    def power(self, base: Node, exponent: Node) -> str:
        """The text of ``base ^ exponent``."""
        base_text = self.grouped(base, binding(base) < PRIMARY, free=False)
        exponent_text = self.grouped(exponent, binding(exponent) < SIGN, free=False)
        return f"{base_text} ^ {exponent_text}"


def formula_text(node: Node, variable: Callable[[str], str] | None = None) -> str:
    """
    A tree as the text of a formula, with spaces around each operator.

    Parameters
    ----------
    node : Node
        The root of the tree.
    variable : Callable[[str], str] | None
        The text of each variable, from its name; the name when None.

    Returns
    -------
    str
        The text, which reads back as the same tree.
    """
    return FormulaWriter(variable).write(node)


def operands(node: Node) -> tuple[Node, ...]:
    """The operands of `node` one level down, in the order the formula writes
    them."""
    match node:
        case Chain(first, rest):
            return (first, *(operand for _, operand in rest))
        case Power(base, exponent):
            return (base, exponent)
        case UnaryOp(_, operand):
            return (operand,)
        case FunctionCall(_, args):
            return args
        case Conditional(condition, true_expr, false_expr):
            return (condition, true_expr, false_expr)
    return ()


def variables(node: Node) -> list[str]:
    """The names of the variables of a tree, in the order it first reads them."""
    names: dict[str, None] = {}

    def visit(node: Node) -> None:
        if isinstance(node, Variable):
            names[node.name] = None
        for operand in operands(node):
            visit(operand)

    visit(node)
    return list(names)


def product(left: Node, right: Node) -> Node:
    """
    ``left * right``, as the tree the parser reads from its text.

    A product extends the chain of `left` when it is a product, which the
    parser reads from the left; it takes the operands of `right` when it is a
    product of ``*`` and ``/``, which gives the same value.

    Parameters
    ----------
    left, right : Node
        The factors.

    Returns
    -------
    Node
        A chain of the product level.
    """
    if isinstance(left, Chain) and binding(left) == PRODUCT:
        first, rest = left.first, list(left.rest)
    else:
        first, rest = left, []
    if (
        isinstance(right, Chain)
        and binding(right) == PRODUCT
        and all(op != "Mod" for op, _ in right.rest)
    ):
        rest += [("Mul", right.first), *right.rest]
    else:
        rest.append(("Mul", right))
    return Chain(first, tuple(rest))
