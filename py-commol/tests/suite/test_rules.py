"""Check the rules of the test suite that a program can judge: every test
checks a result, every expected failure names the issue that fixes it, and no
test takes over its time budget unless the test conventions give its reason.
The rules are in ``docs/codebase/tests.md``."""

import ast
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import coverage
import pytest

from tests.suite.durations import DURATIONS

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = REPO_ROOT / "py-commol" / "tests"
CONVENTIONS = REPO_ROOT / "docs" / "codebase" / "tests.md"
RUST_ROOTS = sorted(
    [*(REPO_ROOT / "crates").iterdir(), REPO_ROOT / "py-commol"],
    key=lambda root: root.name,
)

CHECKS = {"raises", "warns", "fail"}


def relative(path: Path) -> str:
    """The path of ``path`` from the root of the repository."""
    return path.relative_to(REPO_ROOT).as_posix()


# Python


@dataclass
class Module:
    """The functions of one test module and the names it imports."""

    path: Path
    name: str
    functions: dict[str, ast.FunctionDef] = field(default_factory=dict)
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)


def module_name(path: Path) -> str:
    """The dotted name of ``path``, from ``py-commol/``."""
    parts = path.relative_to(TESTS_DIR.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def read_module(path: Path) -> Module:
    """The functions of ``path``, methods by ``Class.method``, and its imports."""
    module = Module(path, module_name(path))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = (
        module.name if path.name == "__init__.py" else module.name.rpartition(".")[0]
    )
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            module.functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    module.functions[f"{node.name}.{item.name}"] = item
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            source = node.module
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                source = ".".join([*base, node.module])
            for alias in node.names:
                module.imports[alias.asname or alias.name] = (source, alias.name)
    return module


def calls(function: ast.FunctionDef) -> Iterator[str]:
    """The names that ``function`` calls: ``name`` or ``self.name``."""
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                yield node.func.id
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
            ):
                yield f"self.{node.func.attr}"


def checks_directly(function: ast.FunctionDef) -> bool:
    """``function`` has an ``assert``, a call of ``pytest.raises``,
    ``pytest.warns`` or ``pytest.fail``, or a call of ``np.testing.assert_*``."""
    for node in ast.walk(function):
        if isinstance(node, ast.Assert):
            return True
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = ast.unparse(node.func.value)
        if (owner == "pytest" and node.func.attr in CHECKS) or (
            owner.endswith("testing") and node.func.attr.startswith("assert_")
        ):
            return True
    return False


class Suite:
    """Every module of ``tests/`` and which of their functions check a result."""

    def __init__(self) -> None:
        self.modules = {
            module.name: module
            for module in map(read_module, sorted(TESTS_DIR.rglob("*.py")))
        }
        self.checking: dict[tuple[str, str], bool] = {}

    def resolve(
        self, module: Module, name: str, owner: str | None
    ) -> tuple[Module, str] | None:
        """The module and qualified name of the function ``name`` in ``module``."""
        if name.startswith("self."):
            method = f"{owner}.{name.removeprefix('self.')}"
            return (module, method) if owner and method in module.functions else None
        if name in module.functions:
            return module, name
        if name in module.imports:
            source, original = module.imports[name]
            if source in self.modules:
                return self.resolve(self.modules[source], original, None)
        return None

    def checks(self, module: Module, name: str) -> bool:
        """``name`` of ``module`` checks a result, itself or through a helper of
        ``tests/``."""
        key = (module.name, name)
        if key not in self.checking:
            self.checking[key] = False
            function = module.functions[name]
            owner = name.partition(".")[0] if "." in name else None
            self.checking[key] = checks_directly(function) or any(
                (target := self.resolve(module, called, owner)) is not None
                and self.checks(*target)
                for called in calls(function)
            )
        return self.checking[key]

    def tests(self) -> Iterator[tuple[Module, str]]:
        """Every test function and test method of a ``test_*.py`` module."""
        for module in self.modules.values():
            if module.path.name.startswith("test_"):
                for name in module.functions:
                    owner, _, method = name.rpartition(".")
                    if method.startswith("test_") and (
                        not owner or owner.startswith("Test")
                    ):
                        yield module, name


def test_every_test_asserts() -> None:
    """Every Python test asserts, expects an exception or a warning, or calls a
    helper of ``tests/`` that does."""
    suite = Suite()
    unchecked = [
        f"{relative(module.path)}::{name.replace('.', '::')}"
        for module, name in suite.tests()
        if not suite.checks(module, name)
    ]
    assert not unchecked, "These tests check no result:\n" + "\n".join(unchecked)


# Rust

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


# Expected failures

ISSUE = re.compile(r"#\d+")


def xfail_marks() -> Iterator[tuple[Path, ast.Call]]:
    """Every call of ``pytest.mark.xfail`` in ``tests/``."""
    for path in sorted(TESTS_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "pytest.mark.xfail"
            ):
                yield path, node


def test_every_xfail_is_strict_and_names_an_issue() -> None:
    """Every ``xfail`` is strict, so a fix fails it, and its reason names the
    open issue that fixes it."""
    loose = []
    for path, mark in xfail_marks():
        keywords = {keyword.arg: keyword.value for keyword in mark.keywords}
        strict = keywords.get("strict")
        reason = keywords.get("reason")
        if not (
            isinstance(strict, ast.Constant)
            and strict.value is True
            and isinstance(reason, ast.Constant)
            and ISSUE.search(str(reason.value))
        ):
            loose.append(f"{relative(path)}:{mark.lineno}")
    assert not loose, (
        "Mark each xfail strict=True and name its issue (#N) in the reason:\n"
        + "\n".join(loose)
    )


# Time

BUDGET = 1.0
"""The most seconds a test may take, setup and teardown included."""

SLOW_TABLE = re.compile(r"^\| `(?P<test>[^`]+)` \|", re.MULTILINE)


def listed_slow_tests() -> set[str]:
    """The tests that the conventions allow over the budget, with their reason."""
    text = CONVENTIONS.read_text(encoding="utf-8")
    section = text.partition("## Time budget")[2].partition("\n## ")[0]
    return {match.group("test") for match in SLOW_TABLE.finditer(section)}


def test_slow_tests_are_listed() -> None:
    """No test of this run takes over the budget unless the conventions list it.
    It runs after every other test (``durations.py``)."""
    if coverage.Coverage.current() is not None:
        pytest.skip("coverage slows the tests; the job Test of Build times them")
    if sys.platform != "linux":
        pytest.skip("the budget is set on Linux, where the job Test of Build runs")
    listed = listed_slow_tests()
    slow = [
        f"{nodeid}: {seconds:.2f} s"
        for nodeid, seconds in sorted(DURATIONS.items())
        if seconds > BUDGET and not {nodeid, nodeid.split("[")[0]} & listed
    ]
    assert not slow, (
        f"These tests take over {BUDGET} s: make their inputs smaller, or give "
        f"the reason in the time budget of {relative(CONVENTIONS)}:\n" + "\n".join(slow)
    )


# Coverage studies

COVERAGE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "coverage.yaml"
NODE_ID = re.compile(r"\btests/[\w/]+\.py(?:::\w+)*")


def coverage_studies() -> Iterator[str]:
    """The node id of every class and function marked ``coverage_study``."""
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef) and any(
                ast.unparse(decorator) == "pytest.mark.coverage_study"
                for decorator in node.decorator_list
            ):
                yield f"{path.relative_to(TESTS_DIR.parent).as_posix()}::{node.name}"


def test_every_coverage_study_runs_in_a_group_of_the_workflow() -> None:
    """Each study is in one group of the matrix of ``coverage.yaml``, so the
    workflow runs it."""
    groups = set(NODE_ID.findall(COVERAGE_WORKFLOW.read_text(encoding="utf-8")))
    studies = list(coverage_studies())
    assert studies, "No test is marked coverage_study"
    missing = [
        study
        for study in studies
        if not any(study == group or study.startswith(f"{group}::") for group in groups)
        and study.split("::")[0] not in groups
    ]
    assert not missing, (
        f"Add these coverage studies to a group of {relative(COVERAGE_WORKFLOW)}:\n"
        + "\n".join(missing)
    )


BUILD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build.yaml"
STUDY_ENGINE = {"commol-model", "commol-difference", "commol-stochastic"}
"""The engines of the models of the studies: ``flow_calibrator`` builds a
difference model, and the study of ABC a chain binomial."""


def workspace_dependencies(crate: str) -> set[str]:
    """``crate`` and the crates of the workspace it depends on, transitively."""
    found, pending = set(), [crate]
    while pending:
        name = pending.pop()
        if name not in found:
            found.add(name)
            manifest = tomllib.loads(
                (REPO_ROOT / "crates" / name / "Cargo.toml").read_text(encoding="utf-8")
            )
            pending += [
                d for d in manifest.get("dependencies", {}) if d.startswith("commol-")
            ]
    return found


def test_the_studies_run_for_a_change_of_any_crate_they_use() -> None:
    """Build runs the coverage studies for a change of the uncertainty methods,
    of every crate they depend on, and of the engine of their models."""
    pattern = re.search(
        r"studies='(?P<pattern>[^']+)'", BUILD_WORKFLOW.read_text(encoding="utf-8")
    )
    assert pattern, f"{relative(BUILD_WORKFLOW)} has no pattern 'studies'"
    used = workspace_dependencies("commol-inference") | STUDY_ENGINE
    missing = sorted(
        crate
        for crate in used
        if not re.search(pattern["pattern"], f"crates/{crate}/src/lib.rs")
    )
    assert not missing, (
        f"Add these crates to the pattern 'studies' of {relative(BUILD_WORKFLOW)}: "
        f"{missing}"
    )


# Test coverage

PACKAGE_DIR = REPO_ROOT / "py-commol" / "commol"
PYPROJECT = REPO_ROOT / "py-commol" / "pyproject.toml"
PRAGMA = re.compile(r"#\s*pragma:\s*no\s*cover\b(?P<reason>.*)", re.IGNORECASE)


def exclusion_patterns() -> Iterator[tuple[int, str, bool]]:
    """Each line of ``exclude_also`` in ``pyproject.toml``: its line number,
    the pattern and whether a comment above it gives the reason."""
    lines = PYPROJECT.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("exclude_also"))
    for number in range(start + 1, len(lines)):
        line = lines[number].strip()
        if line == "]":
            return
        if line.startswith(("'", '"')):
            above = lines[number - 1].strip()
            yield number + 1, line, above.startswith(("#", "'", '"'))


def test_every_coverage_exclusion_has_a_reason() -> None:
    """Every ``# pragma: no cover`` of the package says why on its line, and
    every pattern of ``exclude_also`` has a comment above its group."""
    unexplained = [
        f"{relative(path)}:{number}"
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if (match := PRAGMA.search(line))
        and not re.search(r"[a-z]", match["reason"], re.IGNORECASE)
    ]
    unexplained += [
        f"{relative(PYPROJECT)}:{number} {pattern}"
        for number, pattern, explained in exclusion_patterns()
        if not explained
    ]
    configured = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert configured["tool"]["coverage"]["report"]["exclude_also"]
    assert not unexplained, (
        "Give each coverage exclusion its reason (`# pragma: no cover: <why>`, "
        "or a comment above the pattern):\n" + "\n".join(unexplained)
    )


def test_no_coverage_profile_is_tracked() -> None:
    """An instrumented binary writes ``default_<binary>_<pid>.profraw`` where
    it runs when ``LLVM_PROFILE_FILE`` is not set; no such profile, nor a
    merged ``.profdata``, is part of the repository, and git ignores both."""

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    tracked = git("ls-files", "--", "*.profraw", "*.profdata").stdout.split()
    profiles = ["default_1_0_2.profraw", "py-commol/merged.profdata"]
    ignored = git("check-ignore", "--no-index", *profiles).stdout.split()

    assert not tracked, f"Remove the coverage profiles from git: {tracked}"
    assert ignored == profiles
