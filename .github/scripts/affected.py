"""Which open issues the changes of this branch affect.

An issue is affected when its body names a path that the branch changes: in its
section ``## Touches`` (what it will change or add) or anywhere else, such as its
current state. The changes are the files that differ from the merge base with
``main``, committed or not, and the new files that git does not ignore.

An item is a path, or a path and a symbol (``path::Symbol``); a path that ends in
``/`` is a directory, and a path may start below the root of the repository
(``_model/engine_state.py``). An item with a symbol matches only when
the changed lines of its file name the symbol (the last part of
``Class.method``). The indexes that every change
edits (the changelog, ``AGENTS.md``, the navigation of the docs, the lock files
and the pages of ``docs/codebase/``) are not read, nor are code blocks, comments
and directories of one level (``tests/``), which name too much to point at an
issue.

Each issue it lists needs its body updated where its current state, paths or
assumptions no longer hold, and a comment that says what changed.

Usage, from the root of the repository::

    python .github/scripts/affected.py               # against origin/main
    python .github/scripts/affected.py --closes 125  # skip the issue it delivers
    python .github/scripts/affected.py --base main
"""

import argparse
import re
import subprocess
from dataclasses import dataclass

from parallel import ITEM, issue_of, names_a_path, open_issues, split, touches

INDEXES = {
    "AGENTS.md",
    "CLAUDE.md",
    "Cargo.lock",
    "docs/about/changelog.md",
    "mkdocs.yaml",
    "py-commol/uv.lock",
    "uv.lock",
}
SHARED = "docs/codebase/"
UNREAD = re.compile(r"```.*?```|<!--.*?-->", re.DOTALL)
SHOWN = 3


@dataclass
class Match:
    """An item of an issue body, where it is, and the changed files it names."""

    item: str
    where: str
    files: list[str]


def git(*args: str) -> str:
    """The output of a git command."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout


def changed(base: str) -> dict[str, str]:
    """The files that the branch changes against its merge base with ``base``,
    each with its changed lines (all its lines for a new file)."""
    fork = git("merge-base", base, "HEAD").strip()
    lines: dict[str, str] = {}
    for part in git("diff", "-U0", fork).split("\ndiff --git ")[0:]:
        header = re.search(r"^(?:diff --git )?a/(\S+) b/", part)
        if header:
            lines[header.group(1)] = part
    for file in git("ls-files", "--others", "--exclude-standard").splitlines():
        with open(file, encoding="utf-8", errors="replace") as handle:
            lines[file] = handle.read()
    return {
        file: text
        for file, text in sorted(lines.items())
        if file not in INDEXES and not file.startswith(SHARED)
    }


def is_path(item: str) -> bool:
    """Whether an item names a file or a directory to read: a path for
    ``names_a_path`` of ``parallel.py`` (so ``Model.save`` is a symbol), and
    neither an index nor a directory of one level."""
    path, _ = split(item)
    return (
        names_a_path(item)
        and path not in INDEXES
        and not (path.endswith("/") and path.count("/") < 2)
    )


def names(item: str, file: str, lines: str) -> bool:
    """Whether an item names a changed file: the same path, a directory that
    holds it, or a path that ends the path of the file; and, when the item has a
    symbol, whether the changed lines name it."""
    path, symbol = split(item)
    path = path.removeprefix("./")
    if path.endswith("/"):
        found = file.startswith(path) or f"/{path}" in f"/{file}"
    else:
        found = file == path or file.endswith(f"/{path}")
    if not found or symbol is None:
        return found
    return re.search(rf"\b{re.escape(symbol.split('.')[-1])}\b", lines) is not None


def matches(body: str, files: dict[str, str]) -> list[Match]:
    """The items of an issue body that name a changed file."""
    body = UNREAD.sub("", body)
    sections = {
        item: kind for kind in ("Changes", "Adds") for item in touches(body, kind)
    }
    found = []
    for item in dict.fromkeys(ITEM.findall(body)):
        if not is_path(item):
            continue
        hits = [file for file, lines in files.items() if names(item, file, lines)]
        if hits:
            where = f"Touches ({sections[item]})" if item in sections else "body"
            found.append(Match(item, where, hits))
    return found


def main() -> None:
    """Print each open issue that the branch affects, with the items that match."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base", default="origin/main", help="the branch to compare with"
    )
    parser.add_argument(
        "--closes", type=int, action="append", default=[], help="an issue to skip"
    )
    options = parser.parse_args()
    files = changed(options.base)
    if not files:
        print(f"No changes against {options.base}.")
        return
    issues = [issue_of(item) for item in open_issues()]
    affected = 0
    for issue in sorted(issues, key=lambda issue: issue.number):
        if issue.number in options.closes:
            continue
        found = matches(issue.body, files)
        if not found:
            continue
        affected += 1
        print(f"#{issue.number} {issue.title}")
        for match in found:
            shown = ", ".join(match.files[:SHOWN])
            more = len(match.files) - SHOWN
            if more > 0:
                shown += f" and {more} more"
            print(f"    {match.where}: `{match.item}` <- {shown}")
    if not affected:
        print(f"No open issue names the {len(files)} changed file(s).")
        return
    print(
        f"\n{affected} issue(s). Update the body of each one whose current state, "
        "paths or assumptions changed, and add a comment that says what changed."
    )


if __name__ == "__main__":
    main()
