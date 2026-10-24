"""Check the rules of the Rust tests that a program can judge: every Rust
test checks a result, every ignored test says how it runs, and commol takes a
quantile in one place. The rules are in ``docs/codebase/tests.md`` and
``docs/codebase/decisions/results.md``."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUST_ROOTS = sorted(
    [*(REPO_ROOT / "crates").iterdir(), REPO_ROOT / "py-commol"],
    key=lambda root: root.name,
)


def relative(path: Path) -> str:
    """The path of ``path`` from the root of the repository."""
    return path.relative_to(REPO_ROOT).as_posix()


RUST_TEST = re.compile(r"#\[(?:tokio::)?test\]")
RUST_FUNCTION = re.compile(r"\bfn\s+(?P<name>\w+)")
RUST_CALL = re.compile(r"\b(?P<name>\w+)\s*(?:::<[^>]*>)?\s*\(")
RUST_CHECK = re.compile(r"\bassert\w*!|\bpanic!|\bunreachable!")
RUST_TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|r(?P<hashes>#*)".*?"(?P=hashes)|"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])'",
    re.DOTALL,
)


@dataclass
class RustFunction:
    """One function of a crate: its attributes, signature and body."""

    path: Path
    line: int
    name: str
    attributes: str
    signature: str
    body: str


def blank(text: str) -> str:
    """``text`` with its comments, strings and characters replaced by spaces."""
    return RUST_TOKEN.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), text)


def attributes_before(source: str, position: int) -> str:
    """The attributes on the lines above ``position``, past doc comments."""
    attributes = []
    for line in reversed(source[: source.rfind("\n", 0, position)].split("\n")):
        line = line.strip()
        if line.startswith("#["):
            attributes.append(line)
        elif not line.startswith("//"):
            break
    return "\n".join(reversed(attributes))


def rust_functions(path: Path) -> Iterator[RustFunction]:
    """Every function of ``path`` that has a body."""
    source = path.read_text(encoding="utf-8")
    text = blank(source)
    for match in RUST_FUNCTION.finditer(text):
        open_brace = text.find("{", match.end())
        semicolon = text.find(";", match.end())
        if open_brace == -1 or -1 < semicolon < open_brace:
            continue
        depth, end = 0, open_brace
        for end in range(open_brace, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                break
        yield RustFunction(
            path,
            text.count("\n", 0, match.start()) + 1,
            match.group("name"),
            attributes_before(source, match.start()),
            text[match.end() : open_brace],
            text[open_brace : end + 1],
        )


def rust_checks(root: Path) -> tuple[list[RustFunction], set[str]]:
    """The tests of the crate at ``root`` and the names of its functions that
    check a result, themselves or through another function of the crate."""
    functions = [
        function
        for path in sorted((root / "src").rglob("*.rs"))
        + sorted((root / "tests").rglob("*.rs"))
        for function in rust_functions(path)
    ]
    checking = {
        function.name for function in functions if RUST_CHECK.search(function.body)
    }
    changed = True
    while changed:
        changed = False
        for function in functions:
            if function.name not in checking and any(
                call.group("name") in checking
                for call in RUST_CALL.finditer(function.body)
            ):
                checking.add(function.name)
                changed = True
    tests = [
        function for function in functions if RUST_TEST.search(function.attributes)
    ]
    return tests, checking


def test_every_rust_test_asserts() -> None:
    """Every Rust test asserts, panics on purpose, returns a ``Result`` or calls
    a function of its crate that does."""
    unchecked = []
    for root in RUST_ROOTS:
        tests, checking = rust_checks(root)
        unchecked += [
            f"{relative(test.path)}:{test.line} {test.name}"
            for test in tests
            if test.name not in checking
            and "should_panic" not in test.attributes
            and "Result" not in test.signature
        ]
    assert not unchecked, "These Rust tests check no result:\n" + "\n".join(unchecked)


def test_every_ignored_rust_test_says_how_it_runs() -> None:
    """Every ``#[ignore]`` gives a reason with the command that runs the test."""
    unexplained = [
        f"{relative(test.path)}:{test.line} {test.name}"
        for root in RUST_ROOTS
        for test in rust_checks(root)[0]
        if "#[ignore" in test.attributes
        and not re.search(r'#\[ignore = ".*\bcargo test\b.*--ignored', test.attributes)
    ]
    assert not unexplained, (
        "Give each #[ignore] a reason with the cargo command that runs it:\n"
        + "\n".join(unexplained)
    )


RUST_SORT = re.compile(r"\.sort(?:_unstable)?(?:_by(?:_key)?)?\s*\(")
QUANTILE_FILES = {
    "crates/commol-statistics/src/statistics.rs": "the one quantile of commol",
    "crates/commol-inference/src/conformal/quantile.rs": (
        "the conformal rank, an order statistic that gives an interval its "
        "coverage, not an estimate of a quantile"
    ),
}
"""The files whose functions may sort values to take a quantile, and why."""


def test_commol_has_one_quantile() -> None:
    """No function outside ``statistics.rs`` sorts values to take a quantile:
    a function named for a quantile that sorts, itself or through a function
    of its file."""
    wrong = []
    for root in RUST_ROOTS:
        for path in sorted((root / "src").rglob("*.rs")):
            if relative(path) in QUANTILE_FILES:
                continue
            functions = list(rust_functions(path))
            sorting = {
                function.name
                for function in functions
                if RUST_SORT.search(function.body)
            }
            wrong += [
                f"{relative(path)}:{function.line} {function.name}"
                for function in functions
                if "quantile" in function.name
                and not RUST_TEST.search(function.attributes)
                and (
                    function.name in sorting
                    or any(
                        call.group("name") in sorting
                        for call in RUST_CALL.finditer(function.body)
                    )
                )
            ]
    assert not wrong, (
        "Take a quantile with statistics::quantiles, the one quantile of "
        "commol:\n" + "\n".join(wrong)
    )
