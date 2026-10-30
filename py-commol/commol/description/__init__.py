"""The description of a model: `ModelDescription`, which `Model.describe`
returns, and its sections and blocks (guide: Describing a Model)."""

from commol.description.blocks import (
    Equation,
    EquationGroup,
    Equations,
    GraphEdge,
    GraphNode,
    Matrix,
    ModelDescription,
    Section,
    StateGraph,
    Table,
    Text,
)

__all__ = [
    # Results
    "ModelDescription",
    "Section",
    "Text",
    "Table",
    "Equations",
    "EquationGroup",
    "Equation",
    "StateGraph",
    "GraphNode",
    "GraphEdge",
    "Matrix",
]
