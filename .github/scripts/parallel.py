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

The precedence level (``--levels``) of an open proposal, change or release is
0 when no open issue blocks it, else one more than the highest level of its
blockers; an open blocker of another repository counts as level 0. The
script writes it into the number field ``Level`` of the project, only where
it changes, and empties the field of an issue that is no longer open. First
it adds to the project each open issue that is not an item of it.

An open proposal, change or release waits for the release before its
milestone (``--releases``): the open release issue of the nearest milestone
before its own, by the version that the titles of the milestones name. The
script writes that "blocked by" link, and removes the links from the issue to
every other open release, so that a new milestone in between (0.9.1 before
0.10.0) takes the place of the old blocker. Each open release also waits for
the open proposals and changes of its own milestone: the script writes those
links, and removes the links from the release to a proposal or change whose
milestone names another version, so that an issue moved to another milestone
leaves the release it no longer ships. It owns these links: an issue with no
version in its milestone keeps its links, and so does a release to an issue
with no version in its milestone or of another repository. It refuses a link
that closes a cycle, and names the issue to move. When a link of a release to
an issue it ships and a link to the release before close a cycle together, the
first one stays, whatever the numbers of the issues: a release needs its
issues, and the release before only orders the work.

The script reads the repository from the git remote, and the project is the
one of the organization with the name of the repository. Usage, from the root
of the repository:
``python .github/scripts/parallel.py`` prints what can run in parallel,
``--write`` also writes it under the Delivery of each proposal,
``--next N`` prints the next N issues (``.github/scripts/next.sh`` runs it),
``--releases`` writes the links to the release before each milestone, and
``--levels`` writes the precedence levels. Given together, the links come
first, then the levels, then the next issues (``next.sh`` and ``start.sh``
run them).
"""

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations

from github import JsonValue, repository, rest

ITEM = re.compile(r"`([^`]+)`")
PRIORITIES = ("Now", "Next", "Later")
DEFAULT_PRIORITY = "Next"
PARALLEL = re.compile(r"^- Parallel with:.*$", re.MULTILINE)
VERSION = re.compile(r"v?(\d+(?:\.\d+)*)")


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
    issue_id: int = 0
    version: tuple[int, ...] | None = None


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


def version_of(title: str) -> tuple[int, ...] | None:
    """Return the version that a milestone title names (``0.9.1``), or None.

    Trailing zeros do not count, so ``1.2`` and ``1.2.0`` are one version.
    """
    found = VERSION.fullmatch(title.strip())
    if found is None:
        return None
    parts = [int(part) for part in found.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def milestone_version(item: JsonValue) -> tuple[int, ...] | None:
    """Return the version of the milestone of an issue; None if it names none."""
    milestone = item.get("milestone") if isinstance(item, dict) else None
    title = milestone.get("title") if isinstance(milestone, dict) else None
    return version_of(title) if isinstance(title, str) else None


def load() -> dict[int, Proposal]:
    """Return the open proposals, changes and releases, with their open blockers."""
    found: dict[int, Proposal] = {}
    for item in open_issues():
        if not kind_and_holder(item)[0]:
            continue
        issue = issue_of(item)
        kind, held = kind_and_holder(item)
        issue_id = item.get("id") if isinstance(item, dict) else None
        found[issue.number] = Proposal(
            issue.number,
            issue.title,
            issue.body,
            kind=kind,
            held=held,
            milestone=milestone_of(item),
            issue_id=issue_id if isinstance(issue_id, int) else 0,
            version=milestone_version(item),
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


type Link = tuple[int, int]


def release_before(issue: Proposal, releases: list[Proposal]) -> Proposal | None:
    """Return the open release of the nearest milestone before the one of an issue."""
    if issue.version is None:
        return None
    earlier = [r for r in releases if r.version and r.version < issue.version]
    if not earlier:
        return None
    return max(earlier, key=lambda r: (r.version or (), r.number))


def reaches(issues: dict[int, Proposal], start: int, target: int) -> bool:
    """Tell whether a chain of "blocked by" links leads from one issue to another."""
    seen: set[int] = set()
    queue = [start]
    while queue:
        number = queue.pop()
        if number == target:
            return True
        if number in seen or number not in issues:
            continue
        seen.add(number)
        queue.extend(issues[number].blocked_by)
    return False


def links_to_release_before(
    issues: dict[int, Proposal], releases: list[Proposal]
) -> tuple[list[Link], list[Link]]:
    """Return the links to the release before each issue; remove the others.

    Each issue with a version in its milestone waits for ``release_before``,
    and for no other open release; its links to other releases leave its
    ``blocked_by`` and come back as the links removed.
    """
    numbers = {release.number for release in releases}
    wanted: list[Link] = []
    removed: list[Link] = []
    for number, issue in sorted(issues.items()):
        if issue.version is None:
            continue
        release = release_before(issue, releases)
        keep = release.number if release is not None else None
        if keep is not None:
            wanted.append((number, keep))
        for old in sorted(n for n in issue.blocked_by & numbers if n != keep):
            issue.blocked_by.discard(old)
            removed.append((number, old))
    return wanted, removed


def links_to_shipped_issues(
    issues: dict[int, Proposal], releases: list[Proposal]
) -> tuple[list[Link], list[Link]]:
    """Return the links of each release to the issues it ships; remove the others.

    Each release with a version waits for the open proposals and changes of
    its own milestone. Its links to proposals and changes whose milestone names
    another version leave its ``blocked_by`` and come back as the links
    removed; its links to releases, and to issues with no version in their
    milestone, stay.
    """
    numbers = {release.number for release in releases}
    wanted: list[Link] = []
    removed: list[Link] = []
    for release in sorted(releases, key=lambda r: r.number):
        if release.version is None:
            continue
        for number in sorted(release.blocked_by - numbers):
            issue = issues.get(number)
            if issue is not None and issue.version not in (None, release.version):
                release.blocked_by.discard(number)
                removed.append((release.number, number))
        wanted.extend(
            (release.number, number)
            for number, issue in sorted(issues.items())
            if issue.kind != "release" and issue.version == release.version
        )
    return wanted, removed


def release_links(
    issues: dict[int, Proposal],
) -> tuple[list[Link], list[Link], list[Link]]:
    """Return the links to add and to remove, and the ones refused.

    A link ``(issue, blocker)`` says that the blocker blocks the issue: the
    links of ``links_to_shipped_issues`` and ``links_to_release_before``. A
    link that would close a chain of links back to its issue is refused. The
    links of a release to the issues it ships come first: a release needs its
    issues, while the release before only orders the work. So when the two
    kinds of link close a cycle together, the link to the release before is
    the one refused, whatever the numbers of the issues, and the issue it
    names moves to the milestone of the release that needs it. The
    ``blocked_by`` of the issues change to what the links give.
    """
    releases = [issue for issue in issues.values() if issue.kind == "release"]
    shipped, removed_shipped = links_to_shipped_issues(issues, releases)
    before, removed_before = links_to_release_before(issues, releases)
    added: list[Link] = []
    refused: list[Link] = []
    for number, blocker in shipped + before:
        if blocker in issues[number].blocked_by:
            continue
        if reaches(issues, blocker, number):
            refused.append((number, blocker))
            continue
        issues[number].blocked_by.add(blocker)
        added.append((number, blocker))
    return sorted(added), sorted(removed_shipped + removed_before), refused


def write_release_links(issues: dict[int, Proposal]) -> None:
    """Write the links of each release to the issues it precedes and ships."""
    added, removed, refused = release_links(issues)
    path = f"repos/{repository()}/issues"
    for number, blocker in removed:
        blocker_id = issues[blocker].issue_id
        rest(f"{path}/{number}/dependencies/blocked_by/{blocker_id}", method="DELETE")
    for number, blocker in added:
        rest(
            f"{path}/{number}/dependencies/blocked_by",
            method="POST",
            data={"issue_id": issues[blocker].issue_id},
        )
    for number, blocker in refused:
        if issues[blocker].kind == "release":
            print(
                f"Releases: #{number} is in a later milestone than #{blocker},"
                f" which waits for it: move #{number} to that milestone, or"
                " remove the link"
            )
        else:
            print(
                f"Releases: #{blocker} waits for #{number} through its blockers:"
                f" move #{blocker} to a later milestone, or remove the link that"
                " makes it wait"
            )
    print(f"Releases: {len(added)} links added, {len(removed)} removed")


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


def project_field(repository_: str, name: str) -> tuple[str, int] | None:
    """Return the path of the project of a repository and the id of one field.

    The path is ``orgs/{owner}/projectsV2/{number}``. The project of a
    repository is the project of its organization with the same name; None
    when there is no such project or it has no field of that name.
    """
    owner, repository_name = repository_.split("/")
    number = project_number(owner, repository_name)
    if number is None:
        return None
    path = f"orgs/{owner}/projectsV2/{number}"
    fields = rest(f"{path}/fields?per_page=100")
    field_id = next(
        (
            f.get("id")
            for f in (fields if isinstance(fields, list) else [])
            if isinstance(f, dict) and f.get("name") == name
        ),
        None,
    )
    return (path, field_id) if isinstance(field_id, int) else None


@cache
def project_priorities(repository_: str) -> dict[int, str]:
    """Return the field Priority of the issues of a repository, read once per run.

    A repository without a project, or a project without the field, has none.
    """
    found = project_field(repository_, "Priority")
    if found is None:
        return {}
    path, field_id = found
    items = rest(f"{path}/items?per_page=100&fields={field_id}", every_page=True)
    return priorities_of(items, repository_)


def levels(issues: dict[int, Proposal]) -> dict[int, int]:
    """Return the precedence level of each issue.

    An issue that no open issue blocks is level 0; another is one level above
    its highest blocker. An open blocker of another repository counts as
    level 0. A cycle of links ends at the issue where it closes.
    """
    found: dict[int, int] = {}

    def visit(number: int, path: frozenset[int]) -> int:
        if number in found:
            return found[number]
        if number in path:
            return 0
        issue = issues[number]
        below = [visit(b, path | {number}) for b in issue.blocked_by]
        if issue.blocked_elsewhere:
            below.append(0)
        found[number] = 1 + max(below) if below else 0
        return found[number]

    for number in issues:
        visit(number, frozenset())
    return found


type Level = int | float | None


def levels_of(items: JsonValue, repository_: str) -> dict[int, tuple[int, Level]]:
    """Return the item and the field Level of the issues of one repository.

    ``items`` is the answer of ``GET /orgs/{org}/projectsV2/{number}/items``
    with the field Level; an issue with no Level has None.
    """
    found: dict[int, tuple[int, Level]] = {}
    prefix = "https://api.github.com/repos/"
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("content_type") != "Issue":
            continue
        content = item.get("content")
        fields = item.get("fields")
        item_id = item.get("id")
        if not isinstance(content, dict) or not isinstance(fields, list):
            continue
        url = content.get("repository_url")
        number = content.get("number")
        if not isinstance(url, str) or url.removeprefix(prefix) != repository_:
            continue
        if not isinstance(number, int) or not isinstance(item_id, int):
            continue
        value = next(
            (
                f.get("value")
                for f in fields
                if isinstance(f, dict) and f.get("name") == "Level"
            ),
            None,
        )
        found[number] = (item_id, value if isinstance(value, int | float) else None)
    return found


def level_changes(
    new: dict[int, int], current: dict[int, tuple[int, Level]]
) -> list[tuple[int, int | None]]:
    """Return the items whose Level changes, and the value each one gets.

    An open issue gets its level; an issue with a Level that is no longer
    open loses it. An open issue outside the project is left out.
    """
    changes: list[tuple[int, int | None]] = []
    for number, (item_id, value) in sorted(current.items()):
        level = new.get(number)
        if value != level:
            changes.append((item_id, level))
    return changes


def add_missing_items(
    path: str,
    issues: dict[int, Proposal],
    current: dict[int, tuple[int, Level]],
) -> dict[int, tuple[int, Level]]:
    """Add to the project each open issue that is not an item of it.

    The workflow of the project adds a new issue; this adds one that the
    workflow missed, as ``start.sh`` does when it starts an issue. Returns
    ``current`` with the new items, which have no Level yet.
    """
    found = dict(current)
    for number, issue in sorted(issues.items()):
        if number in found:
            continue
        item = rest(
            f"{path}/items",
            method="POST",
            data={"type": "Issue", "id": issue.issue_id},
        )
        item_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(item_id, int):
            raise SystemExit(f"GitHub: #{number} was added, but no item id came back")
        found[number] = (item_id, None)
        print(f"Project: #{number} added")
    return found


def write_levels(issues: dict[int, Proposal]) -> None:
    """Write the precedence level of each open issue into the field Level."""
    found = project_field(repository(), "Level")
    if found is None:
        print("Level: the project has no number field Level; no level written")
        return
    path, field_id = found
    items = rest(f"{path}/items?per_page=100&fields={field_id}", every_page=True)
    current = add_missing_items(path, issues, levels_of(items, repository()))
    changes = level_changes(levels(issues), current)
    for item_id, level in changes:
        rest(
            f"{path}/items/{item_id}",
            method="PATCH",
            data={"fields": [{"id": field_id, "value": level}]},
        )
    print(f"Level: {len(changes)} written")


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


def show_next(issues: dict[int, Proposal], count: int) -> None:
    """Print the next issues with their priority, then the ones skipped and why."""
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


def show_parallel(proposals: dict[int, Proposal], write: bool) -> None:
    """Print, and with ``write`` store, the issues each one can run with."""
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
    """Print what can run in parallel, or with --next the next issues to take.

    With --releases it first writes the links to the release before each
    milestone; a failure there stops it, because the blockers it reads after
    would be wrong. With --levels it then writes the precedence levels into
    the project. Alone, each does only that. A failure to write the levels
    prints one line and the rest goes on: the next issues, or what can run in
    parallel with --write.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write it under Delivery")
    parser.add_argument("--next", type=int, metavar="N", help="the next N issues")
    parser.add_argument(
        "--releases",
        action="store_true",
        help="link each issue to the release before its milestone",
    )
    parser.add_argument(
        "--levels", action="store_true", help="write the field Level of the project"
    )
    arguments = parser.parse_args()
    issues = load()
    if arguments.releases:
        write_release_links(issues)
    if arguments.levels:
        try:
            write_levels(issues)
        except SystemExit as error:
            print(f"Level: not written: {error}", file=sys.stderr)
    if arguments.next is not None:
        show_next(issues, arguments.next)
    elif arguments.write or not (arguments.levels or arguments.releases):
        show_parallel(issues, arguments.write)


if __name__ == "__main__":
    main()
