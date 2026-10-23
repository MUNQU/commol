"""Check the layout of the documentation: every page in the navigation, one
page per area of decisions, one owner for each piece of text, and a size budget
for each maintainer page."""

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


NAV_PAGE = re.compile(r"^\s+- (?:.+: )?(?P<page>[\w./-]+\.md)\s*$", re.MULTILINE)
INCLUDE = re.compile(r'--8<-- "(?P<path>[^":]+\.md)')


def navigation() -> list[str]:
    """The pages of the ``nav`` of ``mkdocs.yaml``, relative to ``docs/``."""
    mkdocs = (REPO_ROOT / "mkdocs.yaml").read_text(encoding="utf-8")
    nav = re.search(r"^nav:\n(?P<items>(?:[ \t].*\n|\n)*)", mkdocs, re.MULTILINE)
    assert nav, "mkdocs.yaml has no nav"
    return [match.group("page") for match in NAV_PAGE.finditer(nav.group("items"))]


@pytest.mark.docs
def test_every_page_is_in_the_navigation() -> None:
    """Every page of ``docs/`` is in the navigation once, or is included in
    another page; every page of the navigation exists."""
    nav = navigation()
    pages = {page.relative_to(DOCS_DIR).as_posix() for page in DOCS_DIR.rglob("*.md")}
    included = {
        Path(match.group("path")).relative_to("docs").as_posix()
        for page in DOCS_DIR.rglob("*.md")
        for match in INCLUDE.finditer(page.read_text(encoding="utf-8"))
        if match.group("path").startswith("docs/")
    }
    missing = sorted(pages - set(nav) - included)
    assert not missing, f"Add these pages to the nav of mkdocs.yaml: {missing}"
    absent = sorted(set(nav) - pages)
    assert not absent, f"The nav of mkdocs.yaml names pages that do not exist: {absent}"
    twice = sorted({page for page in nav if nav.count(page) > 1})
    assert not twice, f"The nav of mkdocs.yaml names these pages twice: {twice}"


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
CHANGELOG = DOCS_DIR / "about" / "changelog.md"
TABLE_RULE = re.compile(r"^ *\|?( *:?-+:? *\|)+( *:?-+:? *)?$")


def documentation_pages() -> list[Path]:
    """The pages that must not repeat one another: ``AGENTS.md`` and every
    page of ``docs/`` but the changelog, whose entries name the changes."""
    return [
        AGENTS,
        *sorted(page for page in DOCS_DIR.rglob("*.md") if page != CHANGELOG),
    ]


def prose_words(page: Path) -> list[str]:
    """
    The words of ``page`` outside code blocks and the header rows of tables,
    without the bars between cells.

    A header row (``| Argument | Type | Default |``) and the bars are the
    layout that many tables share, not text.
    """
    lines = CODE_BLOCK.sub("", page.read_text(encoding="utf-8")).splitlines()
    kept = [
        line
        for line, after in zip(lines, [*lines[1:], ""], strict=True)
        if not TABLE_RULE.match(line) and not TABLE_RULE.match(after)
    ]
    return [word for word in " ".join(kept).split() if word.strip("|")]


@pytest.mark.docs
def test_no_text_is_repeated_across_pages() -> None:
    """No run of twelve words of text is in two pages."""
    runs: defaultdict[str, set[str]] = defaultdict(set)
    words_of: dict[str, list[str]] = {}
    keys_of: dict[str, list[str]] = {}
    for page in documentation_pages():
        name = relative(page)
        words = prose_words(page)
        # The run of twelve words that starts at each word, joined once.
        keys = [
            " ".join(words[start : start + REPEAT_WORDS])
            for start in range(len(words) - REPEAT_WORDS + 1)
        ]
        words_of[name], keys_of[name] = words, keys
        for key in keys:
            runs[key].add(name)
    repeated = []
    for name, keys in keys_of.items():
        start = 0
        while start < len(keys):
            if len(runs[keys[start]]) == 1:
                start += 1
                continue
            others = runs[keys[start]] - {name}
            end = start + 1
            while end < len(keys) and others & runs[keys[end]]:
                end += 1
            last_word = end - 1 + REPEAT_WORDS
            if name < min(others):
                repeated.append(
                    f"{name} and {sorted(others)}: "
                    f"{' '.join(words_of[name][start:last_word])}"
                )
            start = last_word
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
