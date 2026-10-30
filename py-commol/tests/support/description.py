"""The equations of a description, in the layout `print_equations` had, and
the snapshots of renderings.

`equation_forms` rebuilds the compact form, the expanded form and the
impulses from the blocks of the ``dynamics`` section, in text or in LaTeX, so
a test checks the same lines the equations of a model always had.
"""

from pathlib import Path

import pytest

from commol.description.blocks import Equation, Equations, Text
from commol._model.model import Model

BANNER = "=" * 40
SNAPSHOTS = Path(__file__).parent.parent / "description" / "snapshots"


def matches_snapshot(name: str, rendered: str) -> None:
    """`rendered` equals the snapshot `name`; a missing one is written, and fails."""
    path = SNAPSHOTS / name
    if not path.exists():
        SNAPSHOTS.mkdir(exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        pytest.fail(f"Wrote the snapshot {path}; review it and run again")
    assert rendered == path.read_text(encoding="utf-8")


def _banner(title: str) -> list[str]:
    return [BANNER, title, BANNER]


def _flow(equation: Equation, latex: bool) -> str:
    if latex:
        return f"  ${equation.left_latex}: {equation.right_latex}$"
    return f"  {equation.left}: {equation.right}"


def _equals(equation: Equation, latex: bool) -> str:
    if latex:
        return f"\\[{equation.left_latex} = {equation.right_latex}\\]"
    return f"{equation.left} = {equation.right}"


def equation_forms(model: Model, latex: bool = False) -> str:
    """
    The compact form, the expanded form and the impulses of a model.

    Parameters
    ----------
    model : Model
        An aggregate model.
    latex : bool
        Whether to write the equations in LaTeX.

    Returns
    -------
    str
        The lines, with a banner above each form.
    """
    blocks = model.describe().section("dynamics").blocks
    compact = [
        block
        for block in blocks
        if isinstance(block, Equations) and block.title is not None
    ]
    expanded = [
        block
        for block in blocks
        if isinstance(block, Equations) and block.heading == "Expanded form"
    ]
    impulses = [
        block
        for block in blocks
        if isinstance(block, Equations) and block.heading == "Impulses"
    ]
    size = next(block for block in blocks if isinstance(block, Text))

    lines = [*_banner("COMPACT FORM"), ""]
    for block in compact:
        lines.append(f"{block.title}:")
        for group in block.groups:
            lines.append(f"{group.heading}:")
            lines.extend(_flow(equation, latex) for equation in group.equations)
            if not group.equations:
                lines.append("  (none)")
            lines.append("")
    lines.extend([size.text, "", *_banner("EXPANDED FORM")])
    for block in expanded:
        for group in block.groups:
            lines.extend(_equals(equation, latex) for equation in group.equations)
    for block in impulses:
        lines.extend(["", *_banner("IMPULSES")])
        for group in block.groups:
            lines.append(f"{group.heading}:")
            lines.extend(_flow(equation, latex) for equation in group.equations)
    return "\n".join(lines)
