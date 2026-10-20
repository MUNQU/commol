"""Check that every Rust line no test runs carries the reason it may stay so.

The input is the LCOV report of ``rust_coverage.sh``, which joins every
instantiation of a line, so a line counts as run when any test runs it. A run
of uncovered lines (lines with no count of their own, such as comments, do
not break it) needs a marker ``// coverage: <reason>`` on one of its lines, or
on the lines with no count just above it. A marker that excuses no uncovered
line is an error too: its line now runs, so it no longer needs one. The kinds
of reason, and the rules, are in ``docs/codebase/coverage.md``.

Usage, from the root of the repository::

    python .github/scripts/rust_uncovered.py lcov.info
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MARKER = re.compile(r"//\s*coverage:(?P<reason>.*)$")
SOURCES = ("crates", "py-commol/src")


@dataclass(frozen=True)
class Problem:
    """A line of a source file that breaks the rule."""

    path: str
    first: int
    last: int
    message: str

    def __str__(self) -> str:
        """The problem as ``path:lines: message``."""
        lines = str(self.first)
        if self.last != self.first:
            lines += f"-{self.last}"
        return f"{self.path}:{lines}: {self.message}"


def read_lcov(text: str, root: Path) -> dict[str, dict[int, int]]:
    """
    The count of each instrumented line, by the path of its file.

    Parameters
    ----------
    text : str
        An LCOV report.
    root : Path
        The root of the repository; paths are made relative to it.

    Returns
    -------
    dict[str, dict[int, int]]
        For each file, its instrumented lines and how many times they ran.
    """
    counts: dict[str, dict[int, int]] = {}
    current: dict[int, int] = {}
    for line in text.splitlines():
        if line.startswith("SF:"):
            path = Path(line[3:])
            if path.is_absolute():
                path = path.relative_to(root)
            current = counts.setdefault(path.as_posix(), {})
        elif line.startswith("DA:"):
            number, count = line[3:].split(",")[:2]
            current[int(number)] = current.get(int(number), 0) + int(count)
    return counts


def runs(counts: dict[int, int]) -> list[tuple[int, int]]:
    """
    The runs of uncovered lines of a file, as their first and last lines.

    Parameters
    ----------
    counts : dict[int, int]
        The instrumented lines of the file and how many times they ran.

    Returns
    -------
    list[tuple[int, int]]
        Each run of uncovered lines; a covered line ends a run, a line with no
        count does not.
    """
    found: list[tuple[int, int]] = []
    previous = None
    for number in sorted(counts):
        if counts[number] == 0:
            if found and previous == found[-1][1]:
                found[-1] = (found[-1][0], number)
            else:
                found.append((number, number))
        previous = number
    return found


def check_file(path: str, source: str, counts: dict[int, int]) -> list[Problem]:
    """
    The problems of one source file.

    Parameters
    ----------
    path : str
        The path of the file, relative to the root of the repository.
    source : str
        The text of the file.
    counts : dict[int, int]
        Its instrumented lines and how many times they ran; empty when the
        report does not measure it.

    Returns
    -------
    list[Problem]
        Each run of uncovered lines with no marker, each marker with no
        reason, and each marker that excuses no uncovered line.
    """
    problems: list[Problem] = []
    markers: dict[int, str] = {}
    for number, line in enumerate(source.splitlines(), start=1):
        found = MARKER.search(line)
        if found:
            markers[number] = found.group("reason").strip()
            if not markers[number]:
                problems.append(
                    Problem(path, number, number, "the marker gives no reason")
                )
    excused: set[int] = set()
    lines = sorted(counts)
    for first, last in runs(counts):
        before = [n for n in lines if n < first]
        previous = before[-1] if before else 0
        marked = [n for n in markers if previous < n <= last]
        if marked:
            excused.update(marked)
        else:
            problems.append(
                Problem(path, first, last, "no test runs it, and it has no marker")
            )
    for number in sorted(set(markers) - excused):
        problems.append(
            Problem(path, number, number, "the marker excuses no uncovered line")
        )
    return problems


def check(counts: dict[str, dict[int, int]], root: Path) -> list[Problem]:
    """
    The problems of every Rust source file of the repository.

    Parameters
    ----------
    counts : dict[str, dict[int, int]]
        The report, as `read_lcov` returns it.
    root : Path
        The root of the repository.

    Returns
    -------
    list[Problem]
        The problems of the files that the report measures, and the markers
        of the files it does not.
    """
    problems: list[Problem] = []
    paths = set(counts)
    for directory in SOURCES:
        paths.update(
            p.relative_to(root).as_posix() for p in (root / directory).rglob("*.rs")
        )
    for path in sorted(paths):
        source = (root / path).read_text(encoding="utf-8")
        problems.extend(check_file(path, source, counts.get(path, {})))
    return problems


def main() -> int:
    """Check the report given on the command line, and print each problem."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lcov", type=Path, help="the LCOV report of rust_coverage.sh")
    args = parser.parse_args()
    root = Path.cwd()
    counts = read_lcov(args.lcov.read_text(encoding="utf-8"), root)
    problems = check(counts, root)
    for problem in problems:
        print(problem)
    lines = sum(len(c) for c in counts.values())
    uncovered = sum(1 for c in counts.values() for n in c.values() if n == 0)
    print(f"{lines} lines, {uncovered} not run, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
