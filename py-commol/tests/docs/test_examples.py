"""Run the Python examples of the documentation.

Each page under ``docs/``, and ``README.md``, runs as one script: its ``python``
blocks execute in order, in one namespace, from a temporary working directory,
in a copy of the ``contextvars`` context, so a setting such as
``commol.set_threads`` does not reach the next page.
A block preceded by
``<!-- doc-test: skip: <reason> -->`` is a fragment that cannot run alone; it is
not executed, and the reason is required.
"""

import contextvars
import re
import textwrap
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs"

FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```python[^\n]*\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
SKIP_MARKER = re.compile(r"<!--\s*doc-test:\s*skip:\s*(?P<reason>[^>]*?)\s*-->\s*$")


@dataclass(frozen=True)
class CodeBlock:
    """A ``python`` block of a documentation page."""

    line: int
    code: str
    skip_reason: str | None


def code_blocks(page: Path) -> Iterator[CodeBlock]:
    """
    Yield the ``python`` blocks of a page in order.

    Parameters
    ----------
    page : Path
        The Markdown page.

    Yields
    ------
    CodeBlock
        Each block with its first code line and its skip reason, if any.
    """
    text = page.read_text(encoding="utf-8")
    for match in FENCE.finditer(text):
        line = text.count("\n", 0, match.start()) + 2
        preceding = text[: match.start()].rstrip().splitlines()
        marker = SKIP_MARKER.search(preceding[-1]) if preceding else None
        yield CodeBlock(
            line=line,
            code=textwrap.dedent(match.group("body")),
            skip_reason=marker.group("reason") if marker else None,
        )


def pages_with_examples() -> list[Path]:
    """Pages, and ``README.md``, that contain at least one ``python`` block."""
    pages = [*sorted(DOCS_DIR.rglob("*.md")), REPO_ROOT / "README.md"]
    return [page for page in pages if any(True for _ in code_blocks(page))]


@pytest.mark.docs
@pytest.mark.parametrize(
    "page",
    pages_with_examples(),
    ids=lambda page: page.relative_to(REPO_ROOT).as_posix(),
)
def test_page_examples_run(
    page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every non-skipped block of the page runs, in order, without error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda *_args, **_kwargs: None)
    namespace: dict[str, object] = {"__name__": "__docs__"}
    context = contextvars.copy_context()
    relative = page.relative_to(REPO_ROOT).as_posix()
    try:
        for block in code_blocks(page):
            if block.skip_reason is not None:
                assert block.skip_reason, (
                    f"{relative}:{block.line}: a skipped block needs a reason"
                )
                continue
            try:
                code = compile(block.code, f"{relative}:{block.line}", "exec")
                context.run(exec, code, namespace)
            except Exception as error:
                filename = f"{relative}:{block.line}"
                frames = [
                    frame
                    for frame in traceback.extract_tb(error.__traceback__)
                    if frame.filename == filename
                ]
                line = (
                    block.line + (frames[-1].lineno or 1) - 1 if frames else block.line
                )
                pytest.fail(
                    f"{relative}:{line}: {type(error).__name__}: {error}",
                    pytrace=False,
                )
    finally:
        plt.close("all")
