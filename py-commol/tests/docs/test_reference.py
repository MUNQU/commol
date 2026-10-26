"""Check that the documentation names what exists: every public name has an
API entry, every public class writes its acronyms in capitals, and every site
link, crate, named test and cited path exists."""

import inspect
import re
import subprocess
from pathlib import Path

import pytest

import commol
from tests.suite.public_surface import RESULTS
from tests.support.surface import results

API_DIR = Path(__file__).resolve().parents[3] / "docs" / "api"
DIRECTIVE = re.compile(r"^:::\s*(?P<target>\S+)", re.MULTILINE)


def documented_targets() -> set[str]:
    """Dotted paths of every ``:::`` directive under ``docs/api/``."""
    return {
        match.group("target")
        for page in API_DIR.glob("*.md")
        for match in DIRECTIVE.finditer(page.read_text(encoding="utf-8"))
    }


@pytest.mark.docs
@pytest.mark.parametrize("name", sorted(commol.__all__))
def test_public_name_has_an_api_entry(name: str) -> None:
    """Every name in ``commol.__all__`` has a ``:::`` entry in the API reference."""
    obj = getattr(commol, name)
    path = f"{obj.__module__}.{getattr(obj, '__qualname__', name)}"
    assert path in documented_targets(), (
        f"commol.{name} ({path}) has no '::: {path}' entry under docs/api/"
    )


def pages_of_targets() -> dict[str, set[str]]:
    """The pages of ``docs/api/`` that document each ``:::`` target."""
    pages: dict[str, set[str]] = {}
    for page in API_DIR.glob("*.md"):
        for match in DIRECTIVE.finditer(page.read_text(encoding="utf-8")):
            pages.setdefault(match.group("target"), set()).add(page.name)
    return pages


def pages_of(item: object, pages: dict[str, set[str]]) -> set[str]:
    """The pages that document ``item``, by its path or by its module."""
    module = getattr(item, "__module__", "")
    path = f"{module}.{getattr(item, '__qualname__', '')}"
    return pages.get(path, set()) | pages.get(module, set())


@pytest.mark.docs
def test_each_result_is_on_the_page_of_its_object() -> None:
    """Each result of ``RESULTS`` is returned by an object of the surface, and
    has its ``:::`` entry on the page of an object that returns it."""
    pages = pages_of_targets()
    returned_by = {cls.__name__: owners for cls, owners in results().items()}
    classes = {cls.__name__: cls for cls in results()}
    wrong = []
    for name in sorted(RESULTS):
        if name not in returned_by:
            wrong.append(f"{name}: no member of the surface returns it")
            continue
        owners = set().union(*(pages_of(owner, pages) for owner in returned_by[name]))
        found = pages_of(classes[name], pages)
        if not found & owners:
            wrong.append(f"{name}: on {sorted(found)}, its objects on {sorted(owners)}")

    assert not wrong, "Document each result with its object:\n" + "\n".join(wrong)


@pytest.mark.docs
def test_every_public_name_is_on_the_api_index() -> None:
    """The table of ``docs/api/index.md`` names every name in ``commol.__all__``."""
    index = (API_DIR / "index.md").read_text(encoding="utf-8")
    missing = [name for name in commol.__all__ if f"`{name}`" not in index]
    assert not missing, f"docs/api/index.md does not name: {missing}"


@pytest.mark.docs
def test_no_public_name_is_private() -> None:
    """No name of ``commol.__all__`` starts with ``_``."""
    private = [name for name in commol.__all__ if name.startswith("_")]
    assert not private, f"commol.__all__ exports private names: {private}"


ACRONYMS = ("ABC", "SMC", "ODE", "PRCC", "PSO")
"""Acronyms that public class names use, each written in capitals."""


@pytest.mark.docs
def test_public_names_capitalize_acronyms() -> None:
    """No public class writes a known acronym as a CapWords word (``Abc``)."""
    mixed = {acronym.capitalize(): acronym for acronym in ACRONYMS}
    found = {
        name: [
            mixed[word] for word in re.findall(r"[A-Z][a-z0-9]+", name) if word in mixed
        ]
        for name in commol.__all__
        if inspect.isclass(getattr(commol, name))
    }
    offending = {name: words for name, words in found.items() if words}
    assert not offending, (
        f"public classes that do not write an acronym in capitals: {offending}"
    )


SITE_LINK = re.compile(
    r"https://munqu\.github\.io/commol/(?:latest|dev)/(?P<path>[^)#\s>]*)"
)
REPO_ROOT = API_DIR.parents[1]


@pytest.mark.docs
@pytest.mark.parametrize("source", ["README.md", "AGENTS.md"])
def test_site_links_point_to_existing_pages(source: str) -> None:
    """Every link to the published site names a page that exists under ``docs/``."""
    docs_dir = REPO_ROOT / "docs"
    text = (REPO_ROOT / source).read_text(encoding="utf-8")
    missing = []
    for match in SITE_LINK.finditer(text):
        path = match.group("path").strip("/")
        candidates = (
            [docs_dir / "index.md"]
            if not path
            else [docs_dir / f"{path}.md", docs_dir / path / "index.md"]
        )
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(match.group(0))
    assert not missing, f"{source} links to pages that do not exist: {missing}"


NAMED_TEST = re.compile(
    r"`(?P<file>(?:py-commol/)?(?:tests|crates)/[\w/.\-]+\.(?:py|rs))::(?P<name>\w+)"
)


def named_tests() -> list[tuple[str, str, str]]:
    """Each ``<file>::<name>`` reference under ``docs/``, with its page."""
    docs_dir = REPO_ROOT / "docs"
    return [
        (str(page.relative_to(docs_dir)), match.group("file"), match.group("name"))
        for page in sorted(docs_dir.rglob("*.md"))
        for match in NAMED_TEST.finditer(page.read_text(encoding="utf-8"))
    ]


@pytest.mark.docs
@pytest.mark.parametrize(("page", "file", "name"), named_tests())
def test_named_tests_exist(page: str, file: str, name: str) -> None:
    """Every ``<file>::<name>`` in the documentation names a definition in that file."""
    path = REPO_ROOT / file
    if not path.is_file():
        path = REPO_ROOT / "py-commol" / file
    assert path.is_file(), f"{page} names {file}, which does not exist"
    definition = re.compile(
        rf"^\s*(?:pub(?:\(\w+\))? )?(?:def|fn|class) {name}\b", re.MULTILINE
    )
    assert definition.search(path.read_text(encoding="utf-8")), (
        f"{page} names {file}::{name}, which {file} does not define"
    )


ARCHITECTURE_PAGE = REPO_ROOT / "docs" / "codebase" / "architecture.md"


@pytest.mark.docs
@pytest.mark.parametrize(
    "crate",
    sorted(path.parent.name for path in (REPO_ROOT / "crates").glob("*/Cargo.toml")),
)
def test_crate_is_on_the_architecture_page(crate: str) -> None:
    """Every crate under ``crates/`` is named on the architecture page."""
    text = ARCHITECTURE_PAGE.read_text(encoding="utf-8")
    assert f"`{crate}`" in text, (
        f"{crate} is not named in docs/codebase/architecture.md: add it to "
        "the diagrams and tables of that page"
    )


CODE_BLOCK = re.compile(r"^ *(```|~~~).*?^ *\1", re.MULTILINE | re.DOTALL)
CODE_SPAN = re.compile(r"`(?P<text>[^`\s]+)`")
CITED_PATH = re.compile(
    r"[\w.\-/]*(?:/|\.(?:py|pyi|rs|md|toml|yaml|yml|json|lock|sh|txt|html))"
)
LOCAL_FILES = {"CLAUDE.md", "AGENT.md"}
CODEBASE_DIR = REPO_ROOT / "docs" / "codebase"


def git_paths() -> tuple[set[str], set[str]]:
    """The files git tracks or does not ignore, and the directories that hold them."""
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = set(listing.stdout.splitlines())
    directories = {
        parent.as_posix() for file in files for parent in Path(file).parents
    } - {"."}
    return files, directories


def cited_paths() -> list[tuple[str, str]]:
    """Each path in code format outside a code block, with its page.

    The pages are ``AGENTS.md`` and ``docs/codebase/``. A path ends in ``/`` or
    in the extension of a file; ``::name`` after it is left out.
    """
    pages = [REPO_ROOT / "AGENTS.md", *sorted(CODEBASE_DIR.rglob("*.md"))]
    return [
        (page.relative_to(REPO_ROOT).as_posix(), path)
        for page in pages
        for match in CODE_SPAN.finditer(
            CODE_BLOCK.sub("", page.read_text(encoding="utf-8"))
        )
        if CITED_PATH.fullmatch(path := match.group("text").split("::")[0])
        and path not in LOCAL_FILES
    ]


def matches(path: str, tracked: set[str]) -> list[str]:
    """The paths of ``tracked`` that end with every part of ``path``."""
    name = path.removeprefix("../").rstrip("/")
    return sorted(
        candidate
        for candidate in tracked
        if candidate == name or candidate.endswith(f"/{name}")
    )


@pytest.mark.docs
def test_cited_paths_exist() -> None:
    """Every path a page cites names a file or a directory of the repository."""
    files, directories = git_paths()
    missing = [
        f"{page}: {path}"
        for page, path in cited_paths()
        if not matches(path, files if not path.endswith("/") else directories)
    ]
    assert not missing, f"These pages name paths that do not exist: {missing}"


@pytest.mark.docs
def test_cited_file_names_are_unique() -> None:
    """A bare file name a page cites matches one tracked file only."""
    files, _ = git_paths()
    ambiguous = [
        f"{page}: {path} matches {found}"
        for page, path in cited_paths()
        if "/" not in path and len(found := matches(path, files)) > 1
    ]
    assert not ambiguous, (
        f"Write enough of the directory to make each name unique: {ambiguous}"
    )
