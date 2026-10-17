"""Check the layout of the maintainer documentation: one page per area of
decisions, one owner for each piece of shared content, and a size budget for
each page."""

import re
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS = REPO_ROOT / "AGENTS.md"
DOCS_DIR = REPO_ROOT / "docs"
CODEBASE_DIR = DOCS_DIR / "codebase"
DECISIONS_DIR = CODEBASE_DIR / "decisions"
DECISIONS_INDEX = CODEBASE_DIR / "design-decisions.md"

CODE_BLOCK = re.compile(r"^ *(```|~~~).*?^ *\1", re.MULTILINE | re.DOTALL)
HEADING = re.compile(r"^## (?P<title>.+)$", re.MULTILINE)


def relative(page: Path) -> str:
    """The path of ``page`` from the root of the repository."""
    return page.relative_to(REPO_ROOT).as_posix()


def anchor(title: str) -> str:
    """The anchor MkDocs gives a heading."""
    text = re.sub(r"[^\w\s-]", "", title.replace("`", "")).strip().lower()
    return re.sub(r"[\s-]+", "-", text)


@pytest.mark.docs
@pytest.mark.parametrize(
    "page", sorted(DECISIONS_DIR.glob("*.md")), ids=lambda page: page.name
)
def test_decision_areas_are_on_the_index(page: Path) -> None:
    """Every area page, and every decision on it, has a link on the index."""
    index = DECISIONS_INDEX.read_text(encoding="utf-8")
    target = f"decisions/{page.name}"
    assert f"]({target})" in index, (
        f"{relative(page)} has no row in {relative(DECISIONS_INDEX)}"
    )
    text = CODE_BLOCK.sub("", page.read_text(encoding="utf-8"))
    missing = [
        match.group("title")
        for match in HEADING.finditer(text)
        if f"]({target}#{anchor(match.group('title'))})" not in index
    ]
    assert not missing, (
        f"These decisions of {relative(page)} have no link on "
        f"{relative(DECISIONS_INDEX)}: {missing}"
    )


@pytest.mark.docs
def test_no_link_to_a_moved_decision() -> None:
    """A link to a decision points to its area page, not to the index."""
    stale = [
        f"{relative(page)}: {match.group(0)}"
        for page in [AGENTS, *sorted(DOCS_DIR.rglob("*.md"))]
        for match in re.finditer(
            r"design-decisions\.md#[\w-]+", page.read_text(encoding="utf-8")
        )
    ]
    assert not stale, (
        f"Link each decision to its page under docs/codebase/decisions/: {stale}"
    )


REPEAT_WORDS = 12


def shared_pages() -> list[Path]:
    """The pages that must not repeat one another."""
    return [
        AGENTS,
        *sorted(CODEBASE_DIR.rglob("*.md")),
        *sorted((DOCS_DIR / "development").rglob("*.md")),
    ]


@pytest.mark.docs
def test_no_text_is_repeated_across_pages() -> None:
    """No run of twelve words, outside code blocks, is in two pages."""
    runs: defaultdict[str, set[str]] = defaultdict(set)
    words_of: dict[str, list[str]] = {}
    for page in shared_pages():
        words = CODE_BLOCK.sub("", page.read_text(encoding="utf-8")).split()
        words_of[relative(page)] = words
        for start in range(len(words) - REPEAT_WORDS + 1):
            runs[" ".join(words[start : start + REPEAT_WORDS])].add(relative(page))
    repeated = []
    for name, words in words_of.items():
        start = 0
        while start <= len(words) - REPEAT_WORDS:
            others = runs[" ".join(words[start : start + REPEAT_WORDS])] - {name}
            if not others:
                start += 1
                continue
            end = start + REPEAT_WORDS
            while (
                end < len(words)
                and others & runs[" ".join(words[end - REPEAT_WORDS + 1 : end + 1])]
            ):
                end += 1
            if name < min(others):
                repeated.append(
                    f"{name} and {sorted(others)}: {' '.join(words[start:end])}"
                )
            start = end
    assert not repeated, (
        "Keep each text in one page and link to it from the others: "
        + "\n".join(repeated)
    )


SNIPPET = re.compile(
    r"^<!-- --8<-- \[start:(?P<name>\w+)\] -->\n(?P<text>.*?)"
    r"^<!-- --8<-- \[end:(?P=name)\] -->$",
    re.MULTILINE | re.DOTALL,
)


def assert_includes(page: Path, section: str, header: str) -> None:
    """``page`` includes the section ``section`` of ``AGENTS.md`` on the site."""
    sections = {
        match.group("name"): match.group("text")
        for match in SNIPPET.finditer(AGENTS.read_text(encoding="utf-8"))
    }
    assert section in sections, f"AGENTS.md has no snippet section {section!r}"
    assert header in sections[section], (
        f"The section {section!r} of AGENTS.md has no table {header!r}"
    )
    assert f'--8<-- "AGENTS.md:{section}"' in page.read_text(encoding="utf-8"), (
        f"{relative(page)} does not include the section {section!r} of AGENTS.md"
    )
    mkdocs = (REPO_ROOT / "mkdocs.yaml").read_text(encoding="utf-8")
    assert "pymdownx.snippets" in mkdocs, "mkdocs.yaml does not enable snippets"


@pytest.mark.docs
def test_repository_map_includes_the_task_table() -> None:
    """The repository map shows the task table of ``AGENTS.md``."""
    assert_includes(
        CODEBASE_DIR / "repository-map.md",
        "tasks",
        "| Task | Start with | Then check |",
    )


@pytest.mark.docs
def test_workflow_includes_the_commands() -> None:
    """The development workflow shows the commands of ``AGENTS.md``."""
    assert_includes(
        DOCS_DIR / "development" / "workflow.md", "commands", "| Task | Command |"
    )


def budget(page: Path) -> int:
    """The most lines ``page`` may have."""
    if page == AGENTS:
        return 110
    if page.parent == DECISIONS_DIR:
        return 250
    return 300


@pytest.mark.docs
@pytest.mark.parametrize(
    "page",
    [AGENTS, *sorted(CODEBASE_DIR.rglob("*.md"))],
    ids=relative,
)
def test_pages_stay_within_their_budget(page: Path) -> None:
    """Each page stays within the lines of its budget."""
    lines = len(page.read_text(encoding="utf-8").splitlines())
    assert lines <= budget(page), (
        f"{relative(page)} has {lines} lines, over its budget of "
        f"{budget(page)}: move a section to a page of its own"
    )
