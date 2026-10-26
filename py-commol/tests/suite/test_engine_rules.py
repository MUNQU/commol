"""Check the rules of what the engine holds, which a program can judge: every
public setting of commol has the one name of its meaning, no pydantic model
declares a range of a setting, only the engine computes a window, only the
engine reads a formula, and the stub of the extension declares what the
extension has. The rules are in ``docs/codebase/conventions.md``."""

import ast
from collections.abc import Iterator
from pathlib import Path

from tests.suite.settings import declared_ranges, public_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "py-commol" / "commol"


def relative(path: Path) -> str:
    """The path of ``path`` from the root of the repository."""
    return path.relative_to(REPO_ROOT).as_posix()


# Names

OLD_NAMES = {"num_particles", "base_draws"}
"""Names of a count that every other method writes as a plain plural noun."""

QUANTILE_LEVELS = {"Quantile", "Quantile.__init__"}
"""The settings whose ``level`` is the level of a quantile, not of an interval."""


def test_a_setting_has_one_name() -> None:
    """A count is a plain plural noun and the level of an interval is
    ``confidence_level``, in every public setting of commol."""
    wrong = sorted(
        f"{owner}: {name}"
        for owner, name in public_settings()
        if name in OLD_NAMES or (name == "level" and owner not in QUANTILE_LEVELS)
    )
    assert not wrong, (
        "Name a count with a plural noun (particles, draws, replicates) and the "
        "level of an interval confidence_level:\n" + "\n".join(wrong)
    )


POLICY = "A range rule of a setting lives in the engine, and Python does not repeat it."
"""The policy line of the range rules in ``coverage.md``; ``conventions.md``
gives the rule with this test as its check."""

OLD_POLICY = "A check that the Python layer makes first is not repeated"
"""The policy that a range rule replaces: the binding skips a check of Python."""


def test_no_model_declares_a_range_of_a_setting() -> None:
    """No pydantic model of commol declares a range constraint (``ge``, ``gt``,
    ``le``, ``lt``, a constrained number type or ``allow_inf_nan=False``): the
    engine holds the range rule of each setting, and the documentation says
    so."""
    declared = sorted(declared_ranges())
    pages = {
        relative(page): page.read_text(encoding="utf-8")
        for page in (REPO_ROOT / "docs").rglob("*.md")
    }
    policy = {
        "docs/codebase/coverage.md": POLICY,
        "docs/codebase/conventions.md": "test_no_model_declares_a_range_of_a_setting",
    }
    without = [
        name
        for name, line in policy.items()
        if line not in " ".join(pages[name].split())
    ]
    old = [name for name, text in pages.items() if OLD_POLICY in " ".join(text.split())]

    assert not declared, (
        "Move the range rule to the engine (crates/*/src/ranges.rs):\n"
        + "\n".join(declared)
    )
    assert not without, f"Write the policy of the range rules in {without}"
    assert not old, f"Remove the old policy of the binding checks from {old}"


# Windows

WINDOW_NAMES = {"window", "tick_every"}
"""The names of the length of a window, and of the spacing of the ticks of a
figure, which is a grid of windows."""


def window_arithmetic(path: Path) -> Iterator[str]:
    """Each arithmetic operation of ``path`` on a length of a window, and each
    comparison of the start of a window that the engine gives."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.BinOp):
            names = {
                getattr(item, "id", getattr(item, "attr", None))
                for operand in (node.left, node.right)
                for item in ast.walk(operand)
            }
            if names & WINDOW_NAMES:
                yield f"{relative(path)}:{node.lineno}: {ast.unparse(node)}"
        elif isinstance(node, ast.Compare) and "window_start(" in ast.unparse(node):
            yield f"{relative(path)}:{node.lineno}: {ast.unparse(node)}"


def test_only_the_engine_computes_a_window() -> None:
    """The engine computes the start, the end, the index and the count of a
    window and checks its length (``crates/commol-calibration/src/observation.rs``):
    commol has no module of windows of its own and no arithmetic on the length
    of a window."""
    computed = [
        line
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        for line in window_arithmetic(path)
    ]

    assert not (PACKAGE_DIR / "context" / "_windows.py").exists()
    assert not computed, (
        "Ask the engine for the window (commol_rs.calibration):\n" + "\n".join(computed)
    )


# Formulas

FORMULA_TEXT = {"(", ")", " * ", " / ", " + ", " - ", "^", "**"}
"""The parentheses and the operators of the language, as a reader of the text
of a formula compares them."""

PYTHON_PARSERS = {"eval", "compile", "exec"}
"""The builtins that read a text as Python."""


def formula_reading(path: Path) -> Iterator[str]:
    """Each place of ``path`` that reads the text of a formula: a comparison
    with a parenthesis or an operator of the language, or a parse of a text as
    Python."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            reads = any(
                isinstance(operand, ast.Constant) and operand.value in FORMULA_TEXT
                for operand in operands
            )
        elif isinstance(node, ast.Call):
            function = node.func
            reads = (
                isinstance(function, ast.Name) and function.id in PYTHON_PARSERS
            ) or (ast.unparse(function) == "ast.parse")
        else:
            reads = False
        if reads:
            yield f"{relative(path)}:{node.lineno}: {ast.unparse(node)}"


def test_only_the_engine_reads_a_formula() -> None:
    """The parser of the engine (``crates/commol-core/src/math_expression/``)
    is the only reader and the only check of a formula: no module of commol
    tokenizes, checks or parses the text of a formula. The check of units and
    the description read the tree that the engine gives
    (``commol_rs.core.MathExpression.tree``)."""
    readers = [
        place
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        for place in formula_reading(path)
    ]

    assert not readers, (
        "Ask the engine (commol_rs.core.MathExpression) instead of reading the "
        "text of a formula:\n" + "\n".join(readers)
    )


# The stub of the extension

STUB = REPO_ROOT / "py-commol" / "commol" / "commol_rs" / "_commol_rs.pyi"
UNDECLARED = {"test_kind": "the queries of the test model kind, only in a test build"}
"""The members of the extension that the stub leaves out, and why."""


def declared(node: ast.Module | ast.ClassDef) -> dict[str, str]:
    """The names the stub declares in ``node``, each with its annotation."""
    return {
        item.target.id: ast.unparse(item.annotation)
        for item in node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    } | {
        item.name: ""
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name != "__init__"
    }


def test_the_stub_declares_what_the_extension_has() -> None:
    """Each module and class of ``_commol_rs.pyi`` declares the members of
    the built extension and nothing else."""
    from commol.commol_rs import _commol_rs as extension

    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    pending = [("_commol_rs", tree, extension)]
    wrong = []
    while pending:
        name, node, built = pending.pop()
        names = {key: value for key, value in declared(node).items() if key[:2] != "__"}
        members = {key for key in vars(built) if key[:2] != "__"} - set(UNDECLARED)
        if set(names) != members:
            wrong.append(
                f"{name}: only in the stub {sorted(set(names) - members)}, "
                f"only in the extension {sorted(members - set(names))}"
            )
        for member, annotation in names.items():
            protocol = annotation.removeprefix("type[").removesuffix("]")
            if protocol in classes and member in members:
                pending.append((member, classes[protocol], getattr(built, member)))
    assert not wrong, "Make the stub declare what the extension has:\n" + "\n".join(
        wrong
    )
