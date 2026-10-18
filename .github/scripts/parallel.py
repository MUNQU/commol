"""Which open issues can be implemented in parallel, and the next ones to take.

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
A proposal blocked by an open issue of another repository is not ready.

A change has no section Touches: the paths of its section ``## Where`` count
as its *Changes*. An issue with an assignee is in progress. The next issues
(``--next``) are the open proposals and changes that nobody holds and nothing
blocks, that can run at the same time as every issue in progress and as each
other, by effective priority, then milestone, then number. The effective
priority is the highest of the field Priority of the issue in its project
(``Now``, ``Next``, ``Later``; no value counts as ``Next``) and of every open
issue it blocks, in any repository, through chains of "blocked by" links: an
issue that another issue needs inherits the priority of that issue. An open
milestone comes before none; the nearest due date first, then the oldest
milestone.

The script reads the repository from the git remote, and the project is the
one of the organization with the name of the repository. Usage, from the root
of the repository:
``python .github/scripts/parallel.py`` prints what can run in parallel,
``--write`` also writes it under the Delivery of each proposal, and
``--next N`` prints the next N issues (``.github/scripts/next.sh`` runs it).
"""

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations

from github import JsonValue, repository, rest

ITEM = re.compile(r"`([^`]+)`")
PRIORITIES = ("Now", "Next", "Later")
DEFAULT_PRIORITY = "Next"
PARALLEL = re.compile(r"^- Parallel with:.*$", re.MULTILINE)


@dataclass(frozen=True)
class Issue:
    """An issue as GitHub gives it: its number, title, body and state."""

    number: int
    title: str
    body: str
    state: str
    repository: str = ""


@dataclass
class Proposal:
    """An open proposal: its number, title, blockers and what it touches."""

    number: int
    title: str
    body: str
    blocked_by: set[int] = field(default_factory=set)
    blocked_elsewhere: int = 0
    changes: list[str] = field(default_factory=list)
    adds: list[str] = field(default_factory=list)
    kind: str = "proposal"
    held: bool = False
    milestone: tuple[str, int] | None = None


def text_field(item: dict[str, JsonValue], name: str) -> str:
    """Return a text field of a JSON object; a missing or null one is empty."""
    value = item.get(name)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SystemExit(f"GitHub returned a {name} that is not text: {value!r}")
    return value


def issue_of(item: JsonValue) -> Issue:
    """Return the issue of one JSON object of the REST API."""
    if not isinstance(item, dict):
        raise SystemExit(f"GitHub returned an issue that is not an object: {item!r}")
    number = item.get("number")
    if not isinstance(number, int):
        raise SystemExit(f"GitHub returned an issue without a number: {item!r}")
    return Issue(
        number,
        text_field(item, "title"),
        text_field(item, "body"),
        text_field(item, "state").upper(),
        text_field(item, "repository_url").removeprefix(
            "https://api.github.com/repos/"
        ),
    )


def issues_at(path: str) -> list[Issue]:
    """Return the issues of a REST path whose answer is a list, every page."""
    answer = rest(path, every_page=True)
    return [issue_of(item) for item in answer] if isinstance(answer, list) else []


def open_issues() -> list[JsonValue]:
    """Return the open issues of this repository, pull requests left out."""
    answer = rest(
        f"repos/{repository()}/issues?state=open&per_page=100", every_page=True
    )
    items = answer if isinstance(answer, list) else []
    return [i for i in items if isinstance(i, dict) and "pull_request" not in i]


def touches(body: str, kind: str) -> list[str]:
    """Return the items of one line of the section Touches, such as ``- Adds:``."""
    section = body.split("## Touches", 1)
    if len(section) < 2:
        return []
    text = section[1].split("\n## ", 1)[0]
    line = re.search(rf"^- {kind}:(.*)$", text, re.MULTILINE)
    return [str(item) for item in ITEM.findall(line.group(1))] if line else []


def where(body: str) -> list[str]:
    """Return the paths of the section Where of a change, in order."""
    section = body.split("## Where", 1)
    if len(section) < 2:
        return []
    text = section[1].split("\n## ", 1)[0]
    found = [str(item) for item in ITEM.findall(text)]
    return list(dict.fromkeys(item for item in found if names_a_path(item)))


# The suffixes of a file named without a directory (``pyproject.toml``), so
# that a dotted symbol (``Model.save``) is not taken for a file.
FILE_SUFFIXES = {
    "cfg",
    "css",
    "html",
    "ini",
    "j2",
    "js",
    "json",
    "lock",
    "md",
    "py",
    "rs",
    "sh",
    "toml",
    "txt",
    "yaml",
    "yml",
}


def names_a_path(item: str) -> bool:
    """Tell whether an item names a file or a directory, not a symbol or command."""
    path, _ = split(item)
    if not path or any(c.isspace() for c in path):
        return False
    return "/" in path or path.rsplit(".", 1)[-1] in FILE_SUFFIXES


def split(item: str) -> tuple[str, str | None]:
    """Return the path and the symbol of an item; None when it names no symbol."""
    path, _, symbol = item.partition("::")
    return path, symbol or None


def holds(outer: str, inner: str) -> bool:
    """Tell whether a path is another one or a directory that holds it."""
    return outer == inner or (outer.endswith("/") and inner.startswith(outer))


def overlap(first: str, second: str) -> bool:
    """Tell whether two items name the same code.

    One path holds the other, and their symbols are equal or one of them has
    none.
    """
    path_a, symbol_a = split(first)
    path_b, symbol_b = split(second)
    if not (holds(path_a, path_b) or holds(path_b, path_a)):
        return False
    return symbol_a is None or symbol_b is None or symbol_a == symbol_b


def conflict(a: Proposal, b: Proposal) -> list[tuple[str, str]]:
    """Return the pairs of items by which two proposals conflict."""
    found = []
    for x, y in ((a, b), (b, a)):
        for change in x.changes:
            for other in y.changes + y.adds:
                if overlap(change, other):
                    found.append((change, other))
    return found


def kind_and_holder(item: JsonValue) -> tuple[str, bool]:
    """Return the kind of an issue of gh (its label) and whether someone holds it."""
    if not isinstance(item, dict):
        raise SystemExit(f"gh returned an issue that is not an object: {item!r}")
    labels = item.get("labels")
    assignees = item.get("assignees")
    names = (
        {label.get("name") for label in labels or [] if isinstance(label, dict)}
        if isinstance(labels, list)
        else set()
    )
    kind = next((k for k in ("proposal", "change", "release") if k in names), "")
    return kind, isinstance(assignees, list) and bool(assignees)


def milestone_of(item: JsonValue) -> tuple[str, int] | None:
    """Return the due date (empty if none) and number of the milestone of an issue."""
    milestone = item.get("milestone") if isinstance(item, dict) else None
    if not isinstance(milestone, dict):
        return None
    number = milestone.get("number")
    if not isinstance(number, int):
        return None
    due = milestone.get("due_on")
    return (due if isinstance(due, str) else "", number)


def load() -> dict[int, Proposal]:
    """Return the open proposals, changes and releases, with their open blockers."""
    found: dict[int, Proposal] = {}
    for item in open_issues():
        if not kind_and_holder(item)[0]:
            continue
        issue = issue_of(item)
        kind, held = kind_and_holder(item)
        found[issue.number] = Proposal(
            issue.number,
            issue.title,
            issue.body,
            kind=kind,
            held=held,
            milestone=milestone_of(item),
        )
    for proposal in found.values():
        url = f"repos/{repository()}/issues/{proposal.number}/dependencies/blocked_by"
        blockers = issues_at(f"{url}?per_page=100")
        open_blockers = [b for b in blockers if b.state == "OPEN"]
        here = {b.number for b in open_blockers if b.repository in ("", repository())}
        proposal.blocked_by = here & found.keys()
        proposal.blocked_elsewhere = len(open_blockers) - len(here)
        if proposal.kind == "change":
            proposal.changes = where(proposal.body)
        else:
            proposal.changes = touches(proposal.body, "Changes")
            proposal.adds = touches(proposal.body, "Adds")
    return found


def ancestors(proposals: dict[int, Proposal]) -> dict[int, set[int]]:
    """Return every issue that must close before each one, through any chain."""
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


def write_line(proposal: Proposal, others: list[int]) -> None:
    """Write the line ``Parallel with`` under the Delivery of a proposal."""
    line = (
        "- Parallel with: "
        + (", ".join(f"#{n}" for n in others) if others else "none")
        + " (`python .github/scripts/parallel.py`)"
    )
    body = proposal.body
    if PARALLEL.search(body):
        body = PARALLEL.sub(line, body)
    else:
        body = body.replace("## Delivery\n\n", f"## Delivery\n\n{line}\n", 1)
    if body != proposal.body:
        rest(
            f"repos/{repository()}/issues/{proposal.number}",
            method="PATCH",
            data={"body": body},
        )


def project_number(owner: str, name: str) -> int | None:
    """Return the number of the project of a repository: the one with its name."""
    answer = rest(f"orgs/{owner}/projectsV2?per_page=100", every_page=True)
    for project in answer if isinstance(answer, list) else []:
        if isinstance(project, dict) and project.get("title") == name:
            number = project.get("number")
            return number if isinstance(number, int) else None
    return None


def priorities_of(items: JsonValue, repository_: str) -> dict[int, str]:
    """Return the field Priority of the issues of one repository in project items.

    ``items`` is the answer of ``GET /orgs/{org}/projectsV2/{number}/items``
    with the field Priority; an item with no Priority is left out.
    """
    found: dict[int, str] = {}
    prefix = "https://api.github.com/repos/"
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("content_type") != "Issue":
            continue
        content = item.get("content")
        fields = item.get("fields")
        if not isinstance(content, dict) or not isinstance(fields, list):
            continue
        url = content.get("repository_url")
        number = content.get("number")
        if not isinstance(url, str) or url.removeprefix(prefix) != repository_:
            continue
        for value in fields:
            if not isinstance(value, dict) or value.get("name") != "Priority":
                continue
            selected = value.get("value")
            name = selected.get("name") if isinstance(selected, dict) else None
            raw = name.get("raw") if isinstance(name, dict) else None
            if isinstance(number, int) and isinstance(raw, str) and raw in PRIORITIES:
                found[number] = raw
    return found


@cache
def project_priorities(repository_: str) -> dict[int, str]:
    """Return the field Priority of the issues of a repository, read once per run.

    The project of a repository is the project of its organization with the
    same name; a repository without one, or without the field, has none.
    """
    owner, name = repository_.split("/")
    number = project_number(owner, name)
    if number is None:
        return {}
    fields = rest(f"orgs/{owner}/projectsV2/{number}/fields?per_page=100")
    field_id = next(
        (
            f.get("id")
            for f in (fields if isinstance(fields, list) else [])
            if isinstance(f, dict) and f.get("name") == "Priority"
        ),
        None,
    )
    if field_id is None:
        return {}
    items = rest(
        f"orgs/{owner}/projectsV2/{number}/items?per_page=100&fields={field_id}",
        every_page=True,
    )
    return priorities_of(items, repository_)


@cache
def blocking(key: str) -> list[str]:
    """Return the open issues (``owner/name#N``) that an issue blocks."""
    path, number = key.split("#")
    url = f"repos/{path}/issues/{number}/dependencies/blocking?per_page=100"
    issues = issues_at(url)
    return [
        f"{issue.repository or path}#{issue.number}"
        for issue in issues
        if issue.state == "OPEN"
    ]


def effective_priority(
    key: str,
    own: Callable[[str], str | None],
    blocks: Callable[[str], list[str]],
) -> tuple[str, str]:
    """Return the effective priority of an issue and the issue it comes from.

    It is the highest of its own priority and of every issue it blocks, through
    any chain of links (each issue once, so a cycle ends). An issue that is
    already ``Now`` follows no link. The source is empty when the issue keeps
    its own priority.
    """
    best, source = own(key) or DEFAULT_PRIORITY, ""
    if best == PRIORITIES[0]:
        return best, source
    seen = {key}
    queue = list(blocks(key))
    while queue:
        other = queue.pop(0)
        if other in seen:
            continue
        seen.add(other)
        priority = own(other) or DEFAULT_PRIORITY
        if PRIORITIES.index(priority) < PRIORITIES.index(best):
            best, source = priority, other
        queue.extend(blocks(other))
    return best, source


type Skipped = tuple[Proposal, Proposal, list[tuple[str, str]]]
type Rank = tuple[str, str]


def ready(issue: Proposal) -> bool:
    """Return whether an issue is a proposal or change that nobody holds or blocks."""
    return (
        issue.kind in ("proposal", "change")
        and not issue.held
        and not issue.blocked_by
        and not issue.blocked_elsewhere
    )


def sort_key(issue: Proposal, rank: Rank) -> tuple[int, int, str, int, int]:
    """Sort by priority, then milestone (nearest due, then oldest), then number."""
    due, milestone = issue.milestone or ("", 0)
    return (
        PRIORITIES.index(rank[0]),
        0 if issue.milestone else 1,
        due or "~",
        milestone,
        issue.number,
    )


def next_issues(
    issues: dict[int, Proposal], ranks: dict[int, Rank], count: int
) -> tuple[list[Proposal], list[Skipped]]:
    """Return the next issues to take, and the ones skipped with their clash.

    The candidates are the proposals and changes that nobody holds and nothing
    blocks, by ``sort_key``; ``ranks`` gives the effective priority of each one
    (none is ``Next``). A candidate that touches what an issue in progress, or
    an issue already chosen, changes is skipped, with that issue and the items
    they share.
    """
    busy = [issue for issue in issues.values() if issue.held]
    candidates = sorted(
        (issue for issue in issues.values() if ready(issue)),
        key=lambda issue: sort_key(
            issue, ranks.get(issue.number, (DEFAULT_PRIORITY, ""))
        ),
    )
    chosen: list[Proposal] = []
    skipped: list[Skipped] = []
    for issue in candidates:
        if len(chosen) == count:
            break
        clash = next(
            ((other, pairs) for other in busy if (pairs := conflict(issue, other))),
            None,
        )
        if clash is not None:
            skipped.append((issue, *clash))
            continue
        chosen.append(issue)
        busy.append(issue)
    return chosen, skipped


def show_next(count: int) -> None:
    """Print the next issues with their priority, then the ones skipped and why."""
    issues = load()
    here = repository()

    def own(key: str) -> str | None:
        path, number = key.split("#")
        return project_priorities(path).get(int(number))

    ranks = {
        number: effective_priority(f"{here}#{number}", own, blocking)
        for number, issue in issues.items()
        if ready(issue)
    }
    chosen, skipped = next_issues(issues, ranks, count)
    for issue in chosen:
        priority, source = ranks[issue.number]
        origin = source.removeprefix(here) if source else ""
        label = f"{priority}, from {origin}" if origin else priority
        print(f"#{issue.number} [{label}] {issue.title}")
    for issue, other, pairs in skipped:
        where_ = ", ".join(sorted({a for a, _ in pairs} | {b for _, b in pairs}))
        state = "in progress" if other.held else "taken above"
        print(
            f"  skipped #{issue.number}: it touches #{other.number} ({state}): {where_}"
        )


def show_parallel(write: bool) -> None:
    """Print, and with ``write`` store, the issues each one can run with."""
    proposals = load()
    before = ancestors(proposals)
    candidates = [
        p for p in proposals.values() if "## Touches" in p.body or p.kind == "change"
    ]
    parallel: dict[int, list[int]] = {p.number: [] for p in candidates}
    for a, b in combinations(candidates, 2):
        if a.number in before[b.number] or b.number in before[a.number]:
            continue
        if conflict(a, b):
            continue
        parallel[a.number].append(b.number)
        parallel[b.number].append(a.number)
    ready = sorted(
        p.number for p in candidates if not p.blocked_by and not p.blocked_elsewhere
    )
    print("Ready now:", ", ".join(f"#{n}" for n in ready) or "none")
    for number in sorted(parallel):
        others = sorted(parallel[number])
        print(f"#{number} {proposals[number].title}")
        print("    parallel with:", ", ".join(f"#{n}" for n in others) or "none")
        if write:
            write_line(proposals[number], others)


def main() -> None:
    """Print what can run in parallel, or with --next the next issues to take."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write it under Delivery")
    parser.add_argument("--next", type=int, metavar="N", help="the next N issues")
    arguments = parser.parse_args()
    if arguments.next is not None:
        show_next(arguments.next)
    else:
        show_parallel(arguments.write)


if __name__ == "__main__":
    main()
