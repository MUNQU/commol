# Description API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.description`: the `ModelDescription` that `Model.describe` returns, an ordered list of sections, and the blocks of a section. Guide: [Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md).

## ModelDescription

Bases: `BaseModel`

A description of a model, as data.

`Model.describe` builds it. `render` turns it into a string, `save` writes it to a file, and a notebook shows its Markdown rendering.

Attributes:

| Name       | Type            | Description                                         |
| ---------- | --------------- | --------------------------------------------------- |
| `title`    | `str`           | The title, the name of the model.                   |
| `layout`   | `str`           | The layout the sections follow: "default" or "odd". |
| `sections` | `list[Section]` | The sections, in order.                             |

Source code in `commol/description/blocks.py`

```python
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

    def render(
        self, format: Literal["text", "markdown", "latex", "json", "dot"] = "text"
    ) -> str:
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
        from commol.description.render import render

        return render(self, format)

    def save(
        self,
        path: str | Path,
        format: Literal["text", "markdown", "latex", "json", "dot"] | None = None,
    ) -> Path:
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
        from commol.description.render import save

        return save(self, Path(path), format)

    def _repr_markdown_(self) -> str:
        """The Markdown rendering, which a notebook shows."""
        return self.render("markdown")
```

### Methods:

#### section

```python
section(key: str) -> Section
```

The section with a key.

Parameters:

| Name  | Type  | Description                                 | Default    |
| ----- | ----- | ------------------------------------------- | ---------- |
| `key` | `str` | The key of the section, such as "dynamics". | *required* |

Returns:

| Type      | Description  |
| --------- | ------------ |
| `Section` | The section. |

Raises:

| Type       | Description                                                                  |
| ---------- | ---------------------------------------------------------------------------- |
| `KeyError` | If the description has no section with that key; the message lists the keys. |

Source code in `commol/description/blocks.py`

```python
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
```

#### graphs

```python
graphs() -> list[StateGraph]
```

Every state graph of the description, in order.

Returns:

| Type               | Description                             |
| ------------------ | --------------------------------------- |
| `list[StateGraph]` | The graphs, the expanded ones included. |

Source code in `commol/description/blocks.py`

```python
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
```

#### render

```python
render(format: Literal['text', 'markdown', 'latex', 'json', 'dot'] = 'text') -> str
```

The description as a string.

Parameters:

| Name     | Type                                 | Description                                                                                                                                                                                                                                                                                           | Default  |
| -------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `format` | `(text, markdown, latex, json, dot)` | "text": plain text, with the equations in their compact and expanded forms. "markdown": Markdown, with the graphs of the bins and of the strata as Mermaid diagrams. "latex": a LaTeX document. "json": the description as data. "dot": every state graph in Graphviz DOT, the expanded one included. | `"text"` |

Returns:

| Type  | Description    |
| ----- | -------------- |
| `str` | The rendering. |

Raises:

| Type         | Description                        |
| ------------ | ---------------------------------- |
| `ValueError` | If the format is not one of these. |

Source code in `commol/description/blocks.py`

```python
def render(
    self, format: Literal["text", "markdown", "latex", "json", "dot"] = "text"
) -> str:
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
    from commol.description.render import render

    return render(self, format)
```

#### save

```python
save(path: str | Path, format: Literal['text', 'markdown', 'latex', 'json', 'dot'] | None = None) -> Path
```

Write the description to a file.

Parameters:

| Name     | Type                                 | Description                                                                        | Default            |
| -------- | ------------------------------------ | ---------------------------------------------------------------------------------- | ------------------ |
| `path`   | \`str                                | Path\`                                                                             | The file to write. |
| `format` | `(text, markdown, latex, json, dot)` | The format; None takes it from the suffix of path: .txt, .md, .tex, .json or .dot. | `"text"`           |

Returns:

| Type   | Description       |
| ------ | ----------------- |
| `Path` | The path written. |

Raises:

| Type         | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| `ValueError` | If the format is None and the suffix is not one of these, or the format is not valid. |

Source code in `commol/description/blocks.py`

```python
def save(
    self,
    path: str | Path,
    format: Literal["text", "markdown", "latex", "json", "dot"] | None = None,
) -> Path:
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
    from commol.description.render import save

    return save(self, Path(path), format)
```

## Section

Bases: `BaseModel`

One section of a description.

Attributes:

| Name     | Type           | Description                                                                    |
| -------- | -------------- | ------------------------------------------------------------------------------ |
| `key`    | `str`          | The key of the section, such as "dynamics".                                    |
| `title`  | `str`          | The title a rendering shows.                                                   |
| `blocks` | `list[_Block]` | The blocks, in order. A section with no blocks is left out of every rendering. |

Source code in `commol/description/blocks.py`

```python
class Section(BaseModel):
    """
    One section of a description.

    Attributes
    ----------
    key : str
        The key of the section, such as ``"dynamics"``.
    title : str
        The title a rendering shows.
    blocks : list[_Block]
        The blocks, in order. A section with no blocks is left out of every
        rendering.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    blocks: list[_Block] = Field(default_factory=list)
```

## Text

Bases: `BaseModel`

A paragraph.

Attributes:

| Name      | Type              | Description                   |
| --------- | ----------------- | ----------------------------- |
| `type`    | `Literal['text']` | The tag of the block.         |
| `heading` | \`str             | None\`                        |
| `text`    | `str`             | The paragraph, in plain text. |

Source code in `commol/description/blocks.py`

```python
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
```

## Table

Bases: `BaseModel`

A table: named columns and rows of cells.

Attributes:

| Name      | Type                | Description                              |
| --------- | ------------------- | ---------------------------------------- |
| `type`    | `Literal['table']`  | The tag of the block.                    |
| `heading` | \`str               | None\`                                   |
| `columns` | `list[str]`         | The column names.                        |
| `rows`    | `list[list[_Cell]]` | The rows, each with one cell per column. |

Source code in `commol/description/blocks.py`

```python
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
    rows : list[list[_Cell]]
        The rows, each with one cell per column.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["table"] = "table"
    heading: str | None = None
    columns: list[str]
    rows: list[list[_Cell]]

    @model_validator(mode="after")
    def _check_row_lengths(self) -> Self:
        """Each row has one cell per column."""
        for index, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Row {index} of the table has {len(row)} cells; the table "
                    f"has {len(self.columns)} columns: {self.columns}"
                )
        return self
```

## Equations

Bases: `BaseModel`

A system of equations or of flows.

Attributes:

| Name      | Type                   | Description                        |
| --------- | ---------------------- | ---------------------------------- |
| `type`    | `Literal['equations']` | The tag of the block.              |
| `heading` | \`str                  | None\`                             |
| `title`   | \`str                  | None\`                             |
| `groups`  | `list[EquationGroup]`  | The groups of equations, in order. |

Source code in `commol/description/blocks.py`

```python
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
```

## EquationGroup

Bases: `BaseModel`

Equations under one heading, such as the flows of one transition.

Attributes:

| Name        | Type             | Description                                                                         |
| ----------- | ---------------- | ----------------------------------------------------------------------------------- |
| `heading`   | \`str            | None\`                                                                              |
| `equations` | `list[Equation]` | The equations, in order. A group with a heading and no equations renders as (none). |

Source code in `commol/description/blocks.py`

```python
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
```

## Equation

Bases: `BaseModel`

One equation, or one flow, in plain text and in LaTeX.

Attributes:

| Name          | Type                        | Description                                                                                                                               |
| ------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `left`        | `str`                       | The left side in plain text, such as dA/dt or A -> B.                                                                                     |
| `right`       | `str`                       | The right side in plain text.                                                                                                             |
| `left_latex`  | `str`                       | The left side in LaTeX, without math delimiters.                                                                                          |
| `right_latex` | `str`                       | The right side in LaTeX, without math delimiters.                                                                                         |
| `relation`    | `Literal['equals', 'flow']` | "equals": the left side equals the right side. "flow": the left side names where an amount moves (A -> B) and the right side is the flow. |

Source code in `commol/description/blocks.py`

```python
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
```

## StateGraph

Bases: `BaseModel`

The states of a model and the moves between them.

Attributes:

| Name       | Type                     | Description                                                                                                                                                                    |
| ---------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `type`     | `Literal['state_graph']` | The tag of the block.                                                                                                                                                          |
| `heading`  | \`str                    | None\`                                                                                                                                                                         |
| `name`     | `str`                    | The name of the graph, such as states; a DOT file names it so.                                                                                                                 |
| `nodes`    | `list[GraphNode]`        | The states.                                                                                                                                                                    |
| `edges`    | `list[GraphEdge]`        | The moves between them.                                                                                                                                                        |
| `expanded` | `bool`                   | Whether it is the graph of the expanded compartments. The Markdown rendering draws only a graph that is not expanded: an expanded graph can pass what a Mermaid diagram draws. |

Source code in `commol/description/blocks.py`

```python
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
```

## GraphNode

Bases: `BaseModel`

A node of a `StateGraph`.

Attributes:

| Name    | Type  | Description                              |
| ------- | ----- | ---------------------------------------- |
| `id`    | `str` | The id of the node, unique in its graph. |
| `label` | `str` | The label a diagram shows.               |
| `group` | \`str | None\`                                   |

Source code in `commol/description/blocks.py`

```python
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
```

## GraphEdge

Bases: `BaseModel`

An edge of a `StateGraph`: an amount that moves from a node to another.

Attributes:

| Name     | Type  | Description                                  |
| -------- | ----- | -------------------------------------------- |
| `source` | \`str | None\`                                       |
| `target` | \`str | None\`                                       |
| `label`  | `str` | The label a diagram shows, such as the rate. |

Source code in `commol/description/blocks.py`

```python
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
```

## Matrix

Bases: `BaseModel`

A matrix with labelled rows and columns, such as the weights of groups.

Attributes:

| Name            | Type                | Description                                                  |
| --------------- | ------------------- | ------------------------------------------------------------ |
| `type`          | `Literal['matrix']` | The tag of the block.                                        |
| `heading`       | \`str               | None\`                                                       |
| `row_labels`    | `list[str]`         | The label of each row.                                       |
| `column_labels` | `list[str]`         | The label of each column.                                    |
| `values`        | `list[list[float]]` | One row of values per row label, one value per column label. |

Source code in `commol/description/blocks.py`

```python
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
    def _check_shape(self) -> Self:
        """One row of values per row label, one value per column label."""
        shape = [len(row) for row in self.values]
        if shape != [len(self.column_labels)] * len(self.row_labels):
            raise ValueError(
                f"A matrix of {len(self.row_labels)} rows and "
                f"{len(self.column_labels)} columns has rows of {shape} values"
            )
        return self
```
