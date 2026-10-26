"""Renderers of a `ModelDescription`: text, Markdown, LaTeX, JSON and DOT.

A renderer reads the blocks of a description and never its kind of model, so
every kind renders the same way. The state graphs render as Mermaid in the
Markdown rendering (every graph that is not expanded: the bins and the strata;
an expanded graph can pass what a Mermaid diagram draws) and as Graphviz DOT
in a ``.dot`` file (every graph). The text, LaTeX and JSON renderings draw no
diagram; the JSON holds the graphs as data.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from commol._model.description import (
    Equation,
    Equations,
    Matrix,
    ModelDescription,
    StateGraph,
    Table,
    Text,
    _Block,
    _Cell,
)

SUFFIXES: dict[str, Literal["text", "markdown", "latex", "json", "dot"]] = {
    ".txt": "text",
    ".md": "markdown",
    ".tex": "latex",
    ".json": "json",
    ".dot": "dot",
}
"""The format of each file suffix `save` reads."""


def render(
    description: ModelDescription,
    format: Literal["text", "markdown", "latex", "json", "dot"],
) -> str:
    """
    A description as a string.

    Parameters
    ----------
    description : ModelDescription
        The description to render.
    format : {"text", "markdown", "latex", "json", "dot"}
        The format.

    Returns
    -------
    str
        The rendering.

    Raises
    ------
    ValueError
        If the format is not one of these.
    """
    renderers: dict[str, Callable[[ModelDescription], str]] = {
        "text": render_text,
        "markdown": render_markdown,
        "latex": render_latex,
        "json": lambda value: value.model_dump_json(indent=2),
        "dot": render_dot,
    }
    renderer = renderers.get(format)
    if renderer is None:
        raise ValueError(
            f"Unknown format '{format}'. The formats are {sorted(renderers)}"
        )
    return renderer(description)


def save(
    description: ModelDescription,
    path: Path,
    format: Literal["text", "markdown", "latex", "json", "dot"] | None,
) -> Path:
    """
    Write a description to a file, in the format of its suffix by default.

    Parameters
    ----------
    description : ModelDescription
        The description to write.
    path : Path
        The file to write.
    format : {"text", "markdown", "latex", "json", "dot"} | None
        The format; None takes it from the suffix of ``path``.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If ``format`` is None and the suffix is not one `SUFFIXES` knows.
    """
    if format is None:
        format = SUFFIXES.get(path.suffix.lower())  # noqa: A001
        if format is None:
            raise ValueError(
                f"The suffix '{path.suffix}' names no format; use one of "
                f"{sorted(SUFFIXES)} or pass format="
            )
    path.write_text(render(description, format), encoding="utf-8")
    return path


def _visible(description: ModelDescription) -> list:
    """The sections that have blocks."""
    return [section for section in description.sections if section.blocks]


def _cell(value: _Cell) -> str:
    """A cell as text: a number in its shortest form, a flag as yes or no."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _matrix_table(matrix: Matrix) -> Table:
    """The cells of a matrix; each renderer writes the heading of the block."""
    return Table(
        columns=["", *matrix.column_labels],
        rows=[
            [label, *values]
            for label, values in zip(matrix.row_labels, matrix.values, strict=True)
        ],
    )


# ----------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------


def render_text(description: ModelDescription) -> str:
    """
    A description as plain text.

    The equations keep their compact form (each transition and its flows) and
    their expanded form (one equation per state component); no diagram is
    drawn.

    Parameters
    ----------
    description : ModelDescription
        The description to render.

    Returns
    -------
    str
        The text, one line per row.
    """
    parts = [_underline(description.title, "=")]
    for section in _visible(description):
        blocks = [
            text for block in section.blocks if (text := _text_block(block)) is not None
        ]
        parts.append("\n\n".join([_underline(section.title, "-"), *blocks]))
    return "\n\n".join(parts) + "\n"


def _underline(title: str, mark: str) -> str:
    return f"{title}\n{mark * len(title)}"


def _text_block(block: _Block) -> str | None:
    if isinstance(block, StateGraph):
        return None
    lines = [_underline(block.heading, "~")] if block.heading else []
    if isinstance(block, Text):
        lines.append(block.text)
    elif isinstance(block, Table):
        lines.append(_text_table(block.columns, block.rows))
    elif isinstance(block, Matrix):
        table = _matrix_table(block)
        lines.append(_text_table(table.columns, table.rows))
    elif isinstance(block, Equations):
        lines.append(_text_equations(block))
    return "\n".join(lines)


def _text_table(columns: Sequence[str], rows: Sequence[Sequence[_Cell]]) -> str:
    cells = [list(columns), *[[_cell(value) for value in row] for row in rows]]
    widths = [max(len(row[index]) for row in cells) for index in range(len(columns))]

    def line(row: list[str]) -> str:
        return "  ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        ).rstrip()

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(cells[0]), rule, *[line(row) for row in cells[1:]]])


def _text_equation(equation: Equation) -> str:
    separator = ": " if equation.relation == "flow" else " = "
    return f"{equation.left}{separator}{equation.right}"


def _text_equations(block: Equations) -> str:
    lines = [f"{block.title}:"] if block.title else []
    for index, group in enumerate(block.groups):
        if group.heading is None:
            lines.extend(_text_equation(equation) for equation in group.equations)
            continue
        if index:
            lines.append("")
        lines.append(f"{group.heading}:")
        lines.extend(f"  {_text_equation(equation)}" for equation in group.equations)
        if not group.equations:
            lines.append("  (none)")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------


def render_markdown(description: ModelDescription) -> str:
    """
    A description as Markdown.

    Each state graph that is not expanded, the bins and the strata, is a
    Mermaid diagram, which GitHub, GitLab, MkDocs Material, Obsidian and
    JupyterLab (from 4.1) draw; an expanded graph is not drawn, and the
    rendering gives its size instead. The equations are LaTeX math.

    Parameters
    ----------
    description : ModelDescription
        The description to render.

    Returns
    -------
    str
        The Markdown.
    """
    parts = [f"# {description.title}"]
    for section in _visible(description):
        parts.append(f"## {section.title}")
        parts.extend(_markdown_block(block) for block in section.blocks)
    return "\n\n".join(parts) + "\n"


def _markdown_block(block: _Block) -> str:
    lines = [f"### {block.heading}"] if block.heading else []
    if isinstance(block, Text):
        lines.append(block.text)
    elif isinstance(block, Table):
        lines.append(_markdown_table(block.columns, block.rows))
    elif isinstance(block, Matrix):
        table = _matrix_table(block)
        lines.append(_markdown_table(table.columns, table.rows))
    elif isinstance(block, Equations):
        lines.append(_markdown_equations(block))
    elif isinstance(block, StateGraph):
        if block.expanded:
            lines.append(
                f"The graph of the expanded compartments has {len(block.nodes)} "
                f"nodes and {len(block.edges)} edges; it is not drawn here. "
                '`description.save("model.dot")` writes it as Graphviz DOT.'
            )
        else:
            lines.append(f"```mermaid\n{mermaid(block)}\n```")
    return "\n\n".join(lines)


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[_Cell]]) -> str:
    def line(values: Sequence[str]) -> str:
        return "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"

    return "\n".join(
        [
            line(columns),
            "|" + "|".join("---" for _ in columns) + "|",
            *[line([_cell(value) for value in row]) for row in rows],
        ]
    )


def _markdown_equations(block: Equations) -> str:
    lines = [f"**{block.title}**"] if block.title else []
    for group in block.groups:
        if group.heading is not None:
            lines.append(f"`{group.heading}`")
        if not group.equations:
            lines.append("(none)")
            continue
        rows = " \\\\\n".join(_latex_equation(equation) for equation in group.equations)
        lines.append(f"$$\n\\begin{{aligned}}\n{rows}\n\\end{{aligned}}\n$$")
    return "\n\n".join(lines)


def mermaid(graph: StateGraph) -> str:
    """
    A state graph as a Mermaid flowchart.

    Nodes take ids ``n0``, ``n1``, … so that no id of a model clashes with a
    word of the Mermaid grammar. Every inflow starts at one node labelled
    ``none``, and every outflow ends at another; an edge with an empty label
    has no label.

    Parameters
    ----------
    graph : StateGraph
        The graph.

    Returns
    -------
    str
        The flowchart, without the code fence.
    """
    ids = {node.id: f"n{index}" for index, node in enumerate(graph.nodes)}
    lines = ["flowchart LR"]
    lines.extend(
        f'    {ids[node.id]}["{_mermaid_text(node.label)}"]' for node in graph.nodes
    )
    lines.extend(f'    {nowhere}(("none"))' for nowhere in _nowhere(graph))
    for edge in graph.edges:
        source = "inflow" if edge.source is None else ids[edge.source]
        target = "outflow" if edge.target is None else ids[edge.target]
        label = f'|"{_mermaid_text(edge.label)}"|' if edge.label else ""
        lines.append(f"    {source} -->{label} {target}")
    return "\n".join(lines)


def _nowhere(graph: StateGraph) -> list[str]:
    """The ids of the node ``none`` of the inflows and of the outflows of a
    graph, for the sides that have one."""
    nowhere = []
    if any(edge.source is None for edge in graph.edges):
        nowhere.append("inflow")
    if any(edge.target is None for edge in graph.edges):
        nowhere.append("outflow")
    return nowhere


def _mermaid_text(text: str) -> str:
    """Text inside a quoted Mermaid label."""
    return text.replace('"', "#quot;")


# ----------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "×": r"$\times$",
    "Σ": r"$\Sigma$",
    "→": r"$\to$",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


def latex_text(text: str) -> str:
    """
    Plain text escaped for LaTeX.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    str
        The text, with every character LaTeX reads as a command escaped.
    """
    return "".join(_LATEX_SPECIAL.get(char, char) for char in text)


def render_latex(description: ModelDescription) -> str:
    """
    A description as a LaTeX document.

    The document needs only the packages of a standard distribution
    (``amsmath``, ``amssymb``, ``longtable``, ``fontenc``, ``geometry``); its
    equations are ``align*``
    environments, and it draws no diagram.

    Parameters
    ----------
    description : ModelDescription
        The description to render.

    Returns
    -------
    str
        The document, from ``\\documentclass`` to ``\\end{document}``.
    """
    lines = [
        r"\documentclass{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{longtable}",
        r"\usepackage[margin=2cm]{geometry}",
        rf"\title{{{latex_text(description.title)}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
    ]
    for section in _visible(description):
        lines.append(rf"\section*{{{latex_text(section.title)}}}")
        for block in section.blocks:
            lines.extend(_latex_block(block))
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def _latex_block(block: _Block) -> list[str]:
    if isinstance(block, StateGraph):
        return []
    lines = [rf"\subsection*{{{latex_text(block.heading)}}}"] if block.heading else []
    if isinstance(block, Text):
        lines.extend([latex_text(block.text), ""])
    elif isinstance(block, Table):
        lines.extend(_latex_table(block.columns, block.rows))
    elif isinstance(block, Matrix):
        table = _matrix_table(block)
        lines.extend(_latex_table(table.columns, table.rows))
    elif isinstance(block, Equations):
        lines.extend(_latex_equations(block))
    return lines


def _latex_table(columns: Sequence[str], rows: Sequence[Sequence[_Cell]]) -> list[str]:
    width = round(0.9 / len(columns), 3)
    spec = "".join(f"p{{{width}\\linewidth}}" for _ in columns)

    def line(values: Sequence[str]) -> str:
        return " & ".join(latex_text(value) for value in values) + r" \\"

    return [
        rf"\begin{{longtable}}{{{spec}}}",
        line(columns),
        r"\hline",
        r"\endhead",
        *[line([_cell(value) for value in row]) for row in rows],
        r"\end{longtable}",
    ]


def _latex_equation(equation: Equation) -> str:
    relation = "&:" if equation.relation == "flow" else "&="
    return f"{equation.left_latex} {relation} {equation.right_latex}"


def _latex_equations(block: Equations) -> list[str]:
    lines = [rf"\subsubsection*{{{latex_text(block.title)}}}"] if block.title else []
    for group in block.groups:
        if group.heading is not None:
            lines.append(rf"\noindent\texttt{{{latex_text(group.heading)}}}")
        if not group.equations:
            lines.extend(["(none)", ""])
            continue
        rows = " \\\\\n".join(_latex_equation(equation) for equation in group.equations)
        lines.extend([r"\begin{align*}", rows, r"\end{align*}"])
    return lines


# ----------------------------------------------------------------------
# Graphviz DOT
# ----------------------------------------------------------------------


def render_dot(description: ModelDescription) -> str:
    """
    Every state graph of a description in Graphviz DOT, one ``digraph`` each.

    Parameters
    ----------
    description : ModelDescription
        The description.

    Returns
    -------
    str
        The graphs; ``dot -Tpdf`` draws each one.
    """
    return "\n".join(dot(graph) for graph in description.graphs())


def dot(graph: StateGraph) -> str:
    """
    A state graph as a Graphviz ``digraph``.

    Every inflow starts at one point with no label, and every outflow ends at
    another.

    Parameters
    ----------
    graph : StateGraph
        The graph.

    Returns
    -------
    str
        The ``digraph``.
    """
    ids = {node.id: f"n{index}" for index, node in enumerate(graph.nodes)}
    lines = [f"digraph {_dot_text(graph.name)} {{", "    rankdir=LR;"]
    lines.extend(
        f"    {ids[node.id]} [label={_dot_text(node.label)}];" for node in graph.nodes
    )
    lines.extend(
        f'    {nowhere} [shape=point, label=""];' for nowhere in _nowhere(graph)
    )
    for edge in graph.edges:
        source = "inflow" if edge.source is None else ids[edge.source]
        target = "outflow" if edge.target is None else ids[edge.target]
        lines.append(f"    {source} -> {target} [label={_dot_text(edge.label)}];")
    lines.append("}")
    return "\n".join(lines)


def _dot_text(text: str) -> str:
    """A quoted DOT string."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
