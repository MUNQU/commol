"""Check that the documentation names what exists: every public name has an
API entry, and every site link, crate and named test exists."""

import re
from pathlib import Path

import pytest

import commol

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
    """Each ``<file>::<name>`` reference under ``docs/``, with its page.

    The dated design records (``development/design-*.md``) keep the names of
    their date, so they are left out.
    """
    docs_dir = REPO_ROOT / "docs"
    return [
        (str(page.relative_to(docs_dir)), match.group("file"), match.group("name"))
        for page in sorted(docs_dir.rglob("*.md"))
        if not page.match("development/design-*.md")
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
