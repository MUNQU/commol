"""Check ``.github/scripts/release_notes.py``: the notes of a GitHub release,
from its section of the changelog. The rules are in
``docs/development/release.md``."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.support.planning import load

release_notes = load("release_notes")

CHANGELOG = """# Changelog

## [Unreleased]

## [0.4.0] - 2026-10-01

### Added

- A method that reads
  two lines ([Plotting](../guide/simulations.md#plotting)).

## [0.3.0] - 2026-09-01

- Old.

## Version History
"""


def test_the_notes_are_the_section_of_the_version(tmp_path: Path) -> None:
    changelog = tmp_path / "changelog.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")

    with patch.object(release_notes, "CHANGELOG", changelog):
        notes = release_notes.notes("0.4.0")

    assert notes == (
        "## What's New in 0.4.0\n\n"
        "### Added\n\n"
        "- A method that reads two lines ([Plotting]"
        "(https://munqu.github.io/commol/latest/guide/simulations/#plotting)).\n\n"
        "## Installation\n\n"
        "```bash\npip install --upgrade commol==0.4.0\n```\n\n"
        "## Full Changelog\n\n"
        "https://github.com/MUNQU/commol/compare/v0.3.0...v0.4.0\n"
    )


def test_a_version_without_a_section_stops(tmp_path: Path) -> None:
    changelog = tmp_path / "changelog.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")

    with (
        patch.object(release_notes, "CHANGELOG", changelog),
        pytest.raises(SystemExit, match=r"has no section \[0\.5\.0\]"),
    ):
        release_notes.notes("0.5.0")
