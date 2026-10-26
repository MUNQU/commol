"""A description of a model, as data: sections of typed blocks.

`Model.describe` returns a `ModelDescription`. It is an ordered list of
`Section`, each with a key, a title and blocks from a closed set: `Text`,
`Table`, `Equations`, `StateGraph` and `Matrix`. Each kind of model builds its
sections from these blocks (`ModelStructure.describe`), and the renderers of
`commol.api.description` turn any description into text, Markdown, LaTeX, JSON
or Graphviz DOT without knowing the kind.
"""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type Cell = str | int | float | bool | None
"""The value of one cell of a `Table`."""

type DescriptionFormat = Literal["text", "markdown", "latex", "json", "dot"]
"""A format a description renders to."""

type Layout = Literal["default", "odd"]
"""The sections of a description and their order: ``"default"`` or ``"odd"``."""


class Text(BaseModel):
    """
    A paragraph.

    Attributes
    ----------
    type : Literal["text"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    text : str
        The paragraph, in plain text.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    heading: str | None = None
    text: str


class Table(BaseModel):
    """
    A table: named columns and rows of cells.

    Attributes
    ----------
    type : Literal["table"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    columns : list[str]
        The column names.
    rows : list[list[Cell]]
        The rows, each with one cell per column.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["table"] = "table"
    heading: str | None = None
    columns: list[str]
    rows: list[list[Cell]]

    @model_validator(mode="after")
    def check_row_lengths(self) -> Self:
        """Each row has one cell per column."""
        for index, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Row {index} of the table has {len(row)} cells; the table "
                    f"has {len(self.columns)} columns: {self.columns}"
                )
        return self


class Equation(BaseModel):
    """
    One equation, or one flow, in plain text and in LaTeX.

    Attributes
    ----------
    left : str
        The left side in plain text, such as ``dA/dt`` or ``A -> B``.
    right : str
        The right side in plain text.
    left_latex : str
        The left side in LaTeX, without math delimiters.
    right_latex : str
        The right side in LaTeX, without math delimiters.
    relation : Literal["equals", "flow"]
        ``"equals"``: the left side equals the right side. ``"flow"``: the
        left side names where an amount moves (``A -> B``) and the right side
        is the flow.
    """

    model_config = ConfigDict(extra="forbid")

    left: str
    right: str
    left_latex: str
    right_latex: str
    relation: Literal["equals", "flow"] = "equals"


class EquationGroup(BaseModel):
    """
    Equations under one heading, such as the flows of one transition.

    Attributes
    ----------
    heading : str | None
        The heading of the group, or None for equations with no heading.
    equations : list[Equation]
        The equations, in order. A group with a heading and no equations
        renders as ``(none)``.
    """

    model_config = ConfigDict(extra="forbid")

    heading: str | None = None
    equations: list[Equation]


class Equations(BaseModel):
    """
    A system of equations or of flows.

    Attributes
    ----------
    type : Literal["equations"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    title : str | None
        A line that introduces the groups, such as ``Bin Transitions``.
    groups : list[EquationGroup]
        The groups of equations, in order.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["equations"] = "equations"
    heading: str | None = None
    title: str | None = None
    groups: list[EquationGroup]


class GraphNode(BaseModel):
    """
    A node of a `StateGraph`.

    Attributes
    ----------
    id : str
        The id of the node, unique in its graph.
    label : str
        The label a diagram shows.
    group : str | None
        What the node is, such as ``"bin"`` or a stratification id.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    group: str | None = None


class GraphEdge(BaseModel):
    """
    An edge of a `StateGraph`: an amount that moves from a node to another.

    Attributes
    ----------
    source : str | None
        The id of the node the amount leaves, or None for an inflow.
    target : str | None
        The id of the node the amount enters, or None for an outflow.
    label : str
        The label a diagram shows, such as the rate.
    """

    model_config = ConfigDict(extra="forbid")

    source: str | None
    target: str | None
    label: str


class StateGraph(BaseModel):
    """
    The states of a model and the moves between them.

    Attributes
    ----------
    type : Literal["state_graph"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    name : str
        The name of the graph, such as ``states``; a DOT file names it so.
    nodes : list[GraphNode]
        The states.
    edges : list[GraphEdge]
        The moves between them.
    expanded : bool
        Whether it is the graph of the expanded compartments. The Markdown
        rendering draws only a graph that is not expanded: an expanded graph
        can pass what a Mermaid diagram draws.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["state_graph"] = "state_graph"
    heading: str | None = None
    name: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    expanded: bool = False


class Matrix(BaseModel):
    """
    A matrix with labelled rows and columns, such as the weights of groups.

    Attributes
    ----------
    type : Literal["matrix"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    row_labels : list[str]
        The label of each row.
    column_labels : list[str]
        The label of each column.
    values : list[list[float]]
        One row of values per row label, one value per column label.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["matrix"] = "matrix"
    heading: str | None = None
    row_labels: list[str]
    column_labels: list[str]
    values: list[list[float]]

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        """One row of values per row label, one value per column label."""
        shape = [len(row) for row in self.values]
        if shape != [len(self.column_labels)] * len(self.row_labels):
            raise ValueError(
                f"A matrix of {len(self.row_labels)} rows and "
                f"{len(self.column_labels)} columns has rows of {shape} values"
            )
        return self


type Block = Annotated[
    Text | Table | Equations | StateGraph | Matrix, Field(discriminator="type")
]
"""One block of a section: `Text`, `Table`, `Equations`, `StateGraph` or `Matrix`."""


class Section(BaseModel):
    """
    One section of a description.

    Attributes
    ----------
    key : str
        The key of the section, such as ``"dynamics"``.
    title : str
        The title a rendering shows.
    blocks : list[Block]
        The blocks, in order. A section with no blocks is left out of every
        rendering.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    blocks: list[Block] = Field(default_factory=list)


class ModelDescription(BaseModel):
    """
    A description of a model, as data.

    `Model.describe` builds it. `render` turns it into a string, `save` writes
    it to a file, and a notebook shows its Markdown rendering.

    Attributes
    ----------
    title : str
        The title, the name of the model.
    layout : str
        The layout the sections follow: ``"default"`` or ``"odd"``.
    sections : list[Section]
        The sections, in order.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    layout: str = "default"
    sections: list[Section]

    def section(self, key: str) -> Section:
        """
        The section with a key.

        Parameters
        ----------
        key : str
            The key of the section, such as ``"dynamics"``.

        Returns
        -------
        Section
            The section.

        Raises
        ------
        KeyError
            If the description has no section with that key; the message
            lists the keys.
        """
        for section in self.sections:
            if section.key == key:
                return section
        raise KeyError(
            f"The description has no section '{key}'. Its sections are "
            f"{[section.key for section in self.sections]}"
        )

    def graphs(self) -> list[StateGraph]:
        """
        Every state graph of the description, in order.

        Returns
        -------
        list[StateGraph]
            The graphs, the expanded ones included.
        """
        return [
            block
            for section in self.sections
            for block in section.blocks
            if isinstance(block, StateGraph)
        ]

    def render(self, format: DescriptionFormat = "text") -> str:
        """
        The description as a string.

        Parameters
        ----------
        format : {"text", "markdown", "latex", "json", "dot"}, default="text"
            ``"text"``: plain text, with the equations in their compact and
            expanded forms. ``"markdown"``: Markdown, with the graphs of the
            bins and of the strata as Mermaid diagrams. ``"latex"``: a LaTeX document.
            ``"json"``: the description as data. ``"dot"``: every state graph
            in Graphviz DOT, the expanded one included.

        Returns
        -------
        str
            The rendering.

        Raises
        ------
        ValueError
            If the format is not one of these.
        """
        from commol.api.description import render

        return render(self, format)

    def save(self, path: str | Path, format: DescriptionFormat | None = None) -> Path:
        """
        Write the description to a file.

        Parameters
        ----------
        path : str | Path
            The file to write.
        format : {"text", "markdown", "latex", "json", "dot"} | None
            The format; None takes it from the suffix of ``path``: ``.txt``,
            ``.md``, ``.tex``, ``.json`` or ``.dot``.

        Returns
        -------
        Path
            The path written.

        Raises
        ------
        ValueError
            If the format is None and the suffix is not one of these, or the
            format is not valid.
        """
        from commol.api.description import save

        return save(self, Path(path), format)

    def _repr_markdown_(self) -> str:
        """The Markdown rendering, which a notebook shows."""
        return self.render("markdown")
