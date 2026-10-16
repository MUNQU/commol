"""Which open proposals can be implemented in parallel.

Each proposal says what it touches in its section ``## Touches``:

    - Changes: `crates/commol-core/src/simulation_engine.rs::SimulationEngine`
    - Adds: `crates/commol-inference/src/mcmc/`, `py-commol/src/inference.rs::runners`

An item is a path, or a path and a symbol (``path::Symbol``); a path that
ends in ``/`` is a directory. *Changes* are edits to what exists: a behavior,
a signature, a contract that other code implements. *Adds* are new code next
to what exists: a module, a method of a shared class, a variant, an entry of a
registry. Two proposals can run in parallel when no chain of "blocked by" links
joins them and no *Changes* of one touches what the other changes or adds.
Two *Adds* to the same class never conflict: their merge only keeps both.

Usage, from the root of the repository::

    python.github / scripts / parallel.py  # print what can run in parallel
    python.github / scripts / parallel.py - -write  # also write it under Delivery
"""

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from itertools import combinations

REPO = "MUNQU/commol"
ITEM = re.compile(r"`([^`]+)`")
PARALLEL = re.compile(r"^- Parallel with:.*$", re.MULTILINE)


@dataclass
class Proposal:
    """An open proposal: its number, title, blockers and what it touches."""

    number: int
    title: str
    body: str
    blocked_by: set[int] = field(default_factory=set)
    changes: list[str] = field(default_factory=list)
    adds: list[str] = field(default_factory=list)


def gh_json(*args: str) -> list[dict]:
    """The JSON output of a gh command."""
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def touches(body: str, kind: str) -> list[str]:
    """The items of one line of the section Touches, such as ``- Adds: …``."""
    section = body.split("## Touches", 1)
    if len(section) < 2:
        return []
    text = section[1].split("\n## ", 1)[0]
    line = re.search(rf"^- {kind}:(.*)$", text, re.MULTILINE)
    return ITEM.findall(line.group(1)) if line else []


def split(item: str) -> tuple[str, str | None]:
    """The path and the symbol of an item, or None when it names no symbol."""
    path, _, symbol = item.partition("::")
    return path, symbol or None


def overlap(first: str, second: str) -> bool:
    """Whether two items name the same code: one path holds the other, and their
    symbols are equal or one of them has none."""
    path_a, symbol_a = split(first)
    path_b, symbol_b = split(second)

    def holds(outer: str, inner: str) -> bool:
        return outer == inner or (outer.endswith("/") and inner.startswith(outer))

    if not (holds(path_a, path_b) or holds(path_b, path_a)):
        return False
    return symbol_a is None or symbol_b is None or symbol_a == symbol_b


def conflict(a: Proposal, b: Proposal) -> list[tuple[str, str]]:
    """The pairs of items by which two proposals conflict."""
    found = []
    for x, y in ((a, b), (b, a)):
        for change in x.changes:
            for other in y.changes + y.adds:
                if overlap(change, other):
                    found.append((change, other))
    return found


def load() -> dict[int, Proposal]:
    """The open proposals and releases, with their open blockers."""
    issues = gh_json(
        "issue",
        "list",
        "--repo",
        REPO,
        "--state",
        "open",
        "--limit",
        "500",
        "--search",
        "label:proposal,release",
        "--json",
        "number,title,body",
    )
    found = {
        issue["number"]: Proposal(issue["number"], issue["title"], issue["body"] or "")
        for issue in issues
    }
    for proposal in found.values():
        url = f"repos/{REPO}/issues/{proposal.number}/dependencies/blocked_by"
        blockers = gh_json("api", url)
        open_blockers = {b["number"] for b in blockers if b["state"] == "open"}
        proposal.blocked_by = open_blockers & found.keys()
        proposal.changes = touches(proposal.body, "Changes")
        proposal.adds = touches(proposal.body, "Adds")
    return found


def ancestors(proposals: dict[int, Proposal]) -> dict[int, set[int]]:
    """Every issue that must close before each one, through any chain of links."""
    memo: dict[int, set[int]] = {}

    def visit(number: int) -> set[int]:
        if number not in memo:
            memo[number] = set()
            for blocker in proposals[number].blocked_by:
                memo[number] |= {blocker} | visit(blocker)
        return memo[number]

    for number in proposals:
        visit(number)
    return memo


def main() -> None:
    """Print, and with --write store, the proposals each one can run with."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write it under Delivery")
    write = parser.parse_args().write
    proposals = load()
    before = ancestors(proposals)
    candidates = [p for p in proposals.values() if "## Touches" in p.body]
    parallel: dict[int, list[int]] = {p.number: [] for p in candidates}
    for a, b in combinations(candidates, 2):
        if a.number in before[b.number] or b.number in before[a.number]:
            continue
        if conflict(a, b):
            continue
        parallel[a.number].append(b.number)
        parallel[b.number].append(a.number)
    ready = sorted(p.number for p in candidates if not p.blocked_by)
    print("Ready now:", ", ".join(f"#{n}" for n in ready) or "none")
    for number in sorted(parallel):
        others = sorted(parallel[number])
        print(f"#{number} {proposals[number].title}")
        print("    parallel with:", ", ".join(f"#{n}" for n in others) or "none")
        if not write:
            continue
        line = (
            "- Parallel with: "
            + (", ".join(f"#{n}" for n in others) if others else "none")
            + " (`python .github/scripts/parallel.py`)"
        )
        body = proposals[number].body
        if PARALLEL.search(body):
            body = PARALLEL.sub(line, body)
        else:
            body = body.replace("## Delivery\n\n", f"## Delivery\n\n{line}\n", 1)
        if body != proposals[number].body:
            subprocess.run(
                [
                    "gh",
                    "issue",
                    "edit",
                    str(number),
                    "--repo",
                    REPO,
                    "--body-file",
                    "-",
                ],
                input=body,
                text=True,
                check=True,
                capture_output=True,
            )


if __name__ == "__main__":
    main()
