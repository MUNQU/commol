"""Check the words of the user documentation and of the source of ``commol``:
they name no application field, so that a reader of any field reads abstract
names (``docs/codebase/conventions.md``)."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs"
SOURCE_DIR = REPO_ROOT / "py-commol" / "commol"

FIELD_WORDS = re.compile(
    r"\b(?:epidemi\w*|infect\w*|vaccin\w*|diseases?|bacteri\w*|pathogens?|"
    r"virus\w*|outbreaks?|pandemic\w*|susceptibles?|immun\w*|pharmac\w*|"
    r"drugs?|doses?|patients?|hospital\w*|predators?|prey)\b",
    re.IGNORECASE,
)
"""Words that belong to an application field."""

CITATIONS = {
    "docs/guide/describing-models.md": ["*Ecological Modelling*"],
    "py-commol/commol/api/inference/scores.py": [
        "epidemic forecasts in an interval format"
    ],
}
"""The title of a cited work or journal, which a page quotes as published."""


def relative(path: Path) -> str:
    """The path of ``path`` from the root of the repository."""
    return path.relative_to(REPO_ROOT).as_posix()


def checked_files() -> list[Path]:
    """The user pages and the source files of ``commol``.

    ``docs/codebase/`` and ``docs/development/`` state the rule itself, so they
    name the fields that it excludes."""
    pages = [
        DOCS_DIR / "index.md",
        *sorted((DOCS_DIR / "about").glob("*.md")),
        *(
            page
            for folder in ("getting-started", "guide", "advanced", "api")
            for page in sorted((DOCS_DIR / folder).rglob("*.md"))
        ),
    ]
    sources = [*sorted(SOURCE_DIR.rglob("*.py")), *sorted(SOURCE_DIR.rglob("*.pyi"))]
    return [*pages, *sources]


@pytest.mark.docs
@pytest.mark.parametrize("path", checked_files(), ids=relative)
def test_the_pages_name_no_application_field(path: Path) -> None:
    """No user page, docstring or comment of ``commol`` names a field."""
    name = relative(path)
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for title in CITATIONS.get(name, []):
            line = line.replace(title, "")
        found += [
            f"{name}:{number}: {match.group(0)}" for match in FIELD_WORDS.finditer(line)
        ]
    assert not found, (
        "Use abstract names or neutral process words, not the words of a "
        f"field: {found}"
    )
