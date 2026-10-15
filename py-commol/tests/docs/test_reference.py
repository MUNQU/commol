"""Check that the API reference documents every public name of the package."""

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
