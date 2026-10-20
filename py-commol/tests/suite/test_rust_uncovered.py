"""Check ``.github/scripts/rust_uncovered.py``, the rule that a Rust line no
test runs carries a marker with its reason. The rule is in
``docs/codebase/coverage.md``."""

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"


def load(name: str) -> ModuleType:
    """The script ``name`` of ``.github/scripts/``, loaded as a module."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rust_uncovered = load("rust_uncovered")

SOURCE = """fn f(x: i32) -> Result<i32, String> {
    if x > 0 {
        return Ok(x);
    }
    // A comment.
    Err(format!(
        "{x}"
    ))
}
"""


def messages(source: str, counts: dict[int, int]) -> list[str]:
    """The problems of ``source`` with the counts ``counts``, as text."""
    return [str(p) for p in rust_uncovered.check_file("a.rs", source, counts)]


def test_lines_with_no_count_do_not_break_a_run() -> None:
    counts = {1: 1, 2: 1, 3: 1, 6: 0, 7: 0, 8: 0}
    assert rust_uncovered.runs(counts) == [(6, 8)]


def test_a_covered_line_ends_a_run() -> None:
    counts = {1: 0, 2: 1, 3: 0, 6: 0}
    assert rust_uncovered.runs(counts) == [(1, 1), (3, 6)]


def test_an_uncovered_run_with_no_marker_is_a_problem() -> None:
    counts = {1: 1, 2: 1, 3: 1, 6: 0, 7: 0, 8: 0}
    assert messages(SOURCE, counts) == [
        "a.rs:6-8: no test runs it, and it has no marker"
    ]


def test_a_marker_on_a_line_of_the_run_excuses_the_run() -> None:
    source = SOURCE.replace('"{x}"', '"{x}" // coverage: invariant, x is positive')
    assert messages(source, {1: 1, 2: 1, 3: 1, 6: 0, 7: 0, 8: 0}) == []


def test_a_marker_above_the_run_excuses_it() -> None:
    source = SOURCE.replace("// A comment.", "// coverage: invariant, x is positive")
    assert messages(source, {1: 1, 2: 1, 3: 1, 6: 0, 7: 0, 8: 0}) == []


def test_a_marker_excuses_only_its_own_run() -> None:
    source = SOURCE.replace("// A comment.", "// coverage: invariant, x is positive")
    counts = {1: 1, 2: 1, 3: 0, 6: 0, 7: 0, 8: 0}
    # Line 3 and lines 6 to 8 form one run: the comment has no count.
    assert messages(source, counts) == []
    counts = {1: 1, 2: 0, 3: 1, 6: 0, 7: 0, 8: 0}
    assert messages(source, counts) == ["a.rs:2: no test runs it, and it has no marker"]


def test_a_marker_with_no_reason_is_a_problem() -> None:
    source = SOURCE.replace("// A comment.", "// coverage:")
    assert messages(source, {1: 1, 2: 1, 3: 1, 6: 0, 7: 0, 8: 0}) == [
        "a.rs:5: the marker gives no reason"
    ]


def test_a_marker_on_covered_lines_is_a_problem() -> None:
    source = SOURCE.replace("// A comment.", "// coverage: invariant, x is positive")
    assert messages(source, {1: 1, 2: 1, 3: 1, 6: 1, 7: 1, 8: 1}) == [
        "a.rs:5: the marker excuses no uncovered line"
    ]


def test_the_report_joins_its_records_of_a_line(tmp_path: Path) -> None:
    report = (
        f"SF:{tmp_path / 'crates' / 'a.rs'}\nDA:1,0\nDA:2,0\nend_of_record\n"
        f"SF:{tmp_path / 'crates' / 'a.rs'}\nDA:1,3\nend_of_record\n"
    )
    assert rust_uncovered.read_lcov(report, tmp_path) == {"crates/a.rs": {1: 3, 2: 0}}


def test_a_marker_in_a_file_the_report_does_not_measure_is_a_problem(
    tmp_path: Path,
) -> None:
    measured = tmp_path / "crates" / "a.rs"
    other = tmp_path / "py-commol" / "src" / "b.rs"
    for path in (measured, other):
        path.parent.mkdir(parents=True)
    measured.write_text(SOURCE)
    other.write_text("fn g() {} // coverage: invariant\n")
    counts = {"crates/a.rs": {1: 1, 2: 1, 3: 1, 6: 1, 7: 1, 8: 1}}
    assert [str(p) for p in rust_uncovered.check(counts, tmp_path)] == [
        "py-commol/src/b.rs:1: the marker excuses no uncovered line"
    ]
