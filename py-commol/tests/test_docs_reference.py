"""Check that the API reference documents every public name of the package."""

import re
from pathlib import Path

import pytest

import commol

API_DIR = Path(__file__).resolve().parents[2] / "docs" / "api"
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
