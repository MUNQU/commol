"""The planning scripts of ``.github/scripts/``, loaded as modules, and the
issues their tests build. No test here calls GitHub: each one patches ``rest``.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str) -> ModuleType:
    """The script ``name`` of ``.github/scripts/``, loaded as a module."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parallel = load("parallel")


def an_issue(
    number: int,
    *,
    changes: tuple[str, ...] = (),
    adds: tuple[str, ...] = (),
    held: bool = False,
    blocked_by: frozenset[int] = frozenset(),
    milestone: tuple[str, int] | None = None,
    kind: str = "change",
    version: str = "",
):
    """An open issue of the planning script, with what it touches.

    Its id is ten times its number; ``version`` is the title of its milestone.
    """
    issue = parallel.Proposal(number, f"issue {number}", "", kind=kind)
    issue.changes = list(changes)
    issue.adds = list(adds)
    issue.held = held
    issue.blocked_by = set(blocked_by)
    issue.milestone = milestone
    issue.issue_id = 10 * number
    issue.version = parallel.version_of(version)
    return issue
