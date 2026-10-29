"""The sections an aggregate model adds to its description."""

from typing import TYPE_CHECKING

from commol._model import engine_state
from commol._model.aggregate import split_groups
from commol._model.description import Matrix, Section, Table, Text, _Block
from commol._model.equations import DynamicsDescription
from commol._model.neighbors import Complete, Groups

if TYPE_CHECKING:
    from commol._model.aggregate import Aggregate
    from commol._model.model import Model


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


def _entities[S: "Aggregate"](model: "Model[S]") -> list[_Block]:
    process = model.process
    blocks: list[_Block] = [
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


def _neighbors[S: "Aggregate"](model: "Model[S]") -> list[_Block]:
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


def _initialization[S: "Aggregate"](model: "Model[S]") -> list[_Block]:
    structure = model.structure
    process = model.process

    def unit(bin_id: str) -> str | None:
        bin_item = process.bin(bin_id)
        return None if bin_item is None else bin_item.unit

    blocks: list[_Block] = [
        Table(
            heading="Bin amounts",
            columns=["Bin", "Amount", "Unit"],
            rows=[
                [
                    bin_id,
                    "set by calibration" if amount is None else amount,
                    unit(bin_id),
                ]
                for bin_id, amount in structure.bin_amounts.items()
            ],
        )
    ]
    if structure.stratification_fractions:
        rows = []
        for stratification in structure.stratification_fractions:
            for group, split in split_groups(structure, process, stratification):
                for category, fraction in split.items():
                    rows.append(
                        [
                            stratification,
                            category,
                            fraction,
                            "+".join(group) or "each bin",
                        ]
                    )
        blocks.append(
            Table(
                heading="Stratification fractions",
                columns=["Stratification", "Category", "Fraction", "Of"],
                rows=rows,
            )
        )
    state = structure.initial_state(process)
    bin_of = {compartment.name: compartment.bin for compartment in state}
    if structure.compartment_amounts:
        blocks.append(
            Table(
                heading="Compartment overrides",
                columns=["Compartment", "Amount", "Unit"],
                rows=[
                    [compartment, amount, unit(bin_of[compartment])]
                    for compartment, amount in structure.compartment_amounts.items()
                ],
            )
        )
    blocks.append(
        Table(
            heading="Initial state",
            columns=["Compartment", "Amount", "Unit"],
            rows=[
                [compartment.name, compartment.amount, unit(compartment.bin)]
                for compartment in state
            ],
        )
    )
    return blocks


def _design_concepts[S: "Aggregate"](model: "Model[S]") -> list[_Block]:
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
