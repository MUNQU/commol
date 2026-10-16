"""The sections an aggregate model adds to its description."""

from typing import TYPE_CHECKING

from commol.context._model import engine_state
from commol.context._model.equations import DynamicsDescription
from commol.context.description import Block, Matrix, Section, Table, Text
from commol.context.neighbors import Complete, Groups

if TYPE_CHECKING:
    from commol.context.aggregate import Aggregate
    from commol.context.model import Model


def describe_aggregate[S: "Aggregate"](model: "Model[S]") -> list[Section]:
    """
    The ``entities``, ``dynamics``, ``initialization`` and ``design_concepts``
    sections of an aggregate model.

    Parameters
    ----------
    model : Model[Aggregate]
        The model to describe.

    Returns
    -------
    list[Section]
        The sections, by key.
    """
    return [
        Section(key="entities", title="Entities", blocks=_entities(model)),
        Section(
            key="dynamics",
            title="Dynamics",
            blocks=[*DynamicsDescription(model).blocks(), *_neighbors(model)],
        ),
        Section(
            key="initialization", title="Initialization", blocks=_initialization(model)
        ),
        Section(
            key="design_concepts",
            title="Design concepts",
            blocks=_design_concepts(model),
        ),
    ]


def _entities[S: "Aggregate"](model: "Model[S]") -> list[Block]:
    process = model.process
    blocks: list[Block] = [
        Table(
            heading="Bins",
            columns=["Bin", "Name", "Unit"],
            rows=[[item.id, item.name, item.unit] for item in process.bins],
        )
    ]
    if process.accumulators:
        blocks.append(
            Table(
                heading="Accumulators",
                columns=["Accumulator", "Name"],
                rows=[[item.id, item.name] for item in process.accumulators],
            )
        )
    if process.stratifications:
        blocks.append(
            Table(
                heading="Stratifications",
                columns=["Stratification", "Categories", "Splits"],
                rows=[
                    [
                        stratification.id,
                        ", ".join(stratification.categories),
                        _conditions(stratification.conditions),
                    ]
                    for stratification in process.stratifications
                ],
            )
        )
    compartments, _ = engine_state.expanded_outputs(process)
    blocks.append(
        Table(
            heading="Compartments",
            columns=["Compartment", "Bin", "Categories"],
            rows=[
                [
                    compartment.name,
                    compartment.source,
                    ", ".join(
                        f"{stratification}={category}"
                        for stratification, category in compartment.categories.items()
                    ),
                ]
                for compartment in compartments
            ],
        )
    )
    return blocks


def _conditions(conditions: list | None) -> str:
    if not conditions:
        return "every compartment"
    return "where " + " and ".join(
        f"{condition.stratification} = {condition.category}"
        for condition in conditions
        if condition.category is not None
    )


def _neighbors[S: "Aggregate"](model: "Model[S]") -> list[Block]:
    neighbors = model.structure.neighbors
    if isinstance(neighbors, Complete):
        return [
            Text(
                heading="Neighbors",
                text=(
                    "Every unit is a neighbor of every other (Complete): "
                    "neighbors_<bin> is the bin total and neighbors is total."
                ),
            )
        ]
    if isinstance(neighbors, Groups):
        categories = next(
            stratification.categories
            for stratification in model.process.stratifications
            if stratification.id == neighbors.stratification
        )
        weights = neighbors.weights
        return [
            Text(
                heading="Neighbors",
                text=(
                    "Units are neighbors by the categories of "
                    f"'{neighbors.stratification}' (Groups): in a compartment of "
                    "category g, neighbors_<bin> is the sum over g' of "
                    "w[g][g'] times the amount of the bin in g', and neighbors "
                    "the same sum over every bin. The weights w[g][g'] are"
                    + (" the identity." if weights is None else ":")
                ),
            ),
            Matrix(
                row_labels=categories,
                column_labels=categories,
                values=[
                    [
                        (
                            float(own == other)
                            if weights is None
                            else weights[own][other]
                        )
                        for other in categories
                    ]
                    for own in categories
                ],
            ),
        ]
    return []


def _initialization[S: "Aggregate"](model: "Model[S]") -> list[Block]:
    structure = model.structure
    process = model.process
    total = structure.total
    blocks: list[Block] = [
        Table(
            heading="Bin fractions",
            columns=["Bin", "Fraction", "Amount"],
            rows=[
                [
                    bin_id,
                    "set by calibration" if fraction is None else fraction,
                    None if fraction is None else fraction * total,
                ]
                for bin_id, fraction in structure.bin_fractions.items()
            ],
        )
    ]
    if structure.stratification_fractions:
        rows = []
        for stratification, categories in structure.stratification_fractions.items():
            group = structure.get_conditioning_categories(process, stratification)
            for category, fraction in categories.items():
                rows.append(
                    [stratification, category, fraction, "+".join(group) or "total"]
                )
        blocks.append(
            Table(
                heading="Stratification fractions",
                columns=["Stratification", "Category", "Fraction", "Of"],
                rows=rows,
            )
        )
    if structure.compartment_fractions:
        blocks.append(
            Table(
                heading="Compartment overrides",
                columns=["Compartment", "Fraction of the total", "Amount"],
                rows=[
                    [compartment, fraction, fraction * total]
                    for compartment, fraction in structure.compartment_fractions.items()
                ],
            )
        )
    blocks.append(
        Table(
            heading=f"Initial state (total {total:g})",
            columns=["Compartment", "Amount"],
            rows=[
                [compartment.name, compartment.amount]
                for compartment in structure.initial_state(process)
            ],
        )
    )
    return blocks


def _design_concepts[S: "Aggregate"](model: "Model[S]") -> list[Block]:
    neighbors = model.structure.neighbors
    if neighbors is None:
        return []
    if isinstance(neighbors, Complete):
        interaction = (
            "A rate that reads neighbors_<bin> or neighbors depends on every "
            "unit: every unit is a neighbor of every other."
        )
    else:
        interaction = (
            "A rate that reads neighbors_<bin> or neighbors depends on the "
            "units of the categories of "
            f"'{neighbors.stratification}', weighted by the matrix of the "
            "neighbors."
        )
    return [
        Table(columns=["Element", "Description"], rows=[["Interaction", interaction]])
    ]
