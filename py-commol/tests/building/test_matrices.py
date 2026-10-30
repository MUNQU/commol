"""A matrix between the categories of a stratification: what a model holds,
what its description lists, and what a build refuses."""

import math
from pathlib import Path

import pytest

from commol import Aggregate, ModelBuilder
from commol.description.blocks import Matrix, Text
from commol._model.model import Model

VALUES = {"g1": {"g1": 2.0, "g2": 1.0}, "g2": {"g1": 1.0, "g2": 3.0}}
"""Rows ``g1`` 2 and 1, ``g2`` 1 and 3: eigenvalues ``(5 ± √5) / 2``."""


def _builder(
    values: dict[str, dict[str, float]] = VALUES, stratification: str = "group"
) -> ModelBuilder:
    return (
        ModelBuilder("matrices", bin_unit="item")
        .add_bin("A", "A")
        .add_bin("B", "B")
        .add_stratification("group", ["g1", "g2"])
        .add_matrix(
            "M",
            stratification,
            values,
            unit="1/day",
            description="A coupling between the groups",
        )
        .add_parameter("k", 0.1)
        .add_transition("ab", ["A"], ["B"], "k * matrix_sum(M, B / total)")
    )


def _build(builder: ModelBuilder) -> Model:
    return builder.build(
        Aggregate(
            bin_amounts={"A": 900, "B": 100},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
        ),
        engine="DifferenceEquations",
    )


def test_a_model_holds_a_matrix_between_categories() -> None:
    model = _build(_builder())

    (matrix,) = model.process.matrices
    assert (matrix.id, matrix.stratification, matrix.values) == ("M", "group", VALUES)
    assert matrix.unit == "1/day"
    blocks = model.describe().section("parameters").blocks
    text, values = blocks[1], blocks[2]
    assert isinstance(text, Text) and text.heading == "Matrix M"
    radius = (5 + math.sqrt(5)) / 2
    assert text.text == (
        "Between the categories of 'group' in 1/day. A coupling between the "
        "groups. A rate reads matrix_sum(M, e); a rate, a formula parameter or a "
        "calibration constraint reads its entries as "
        f"M_<row>_<column> and its spectral radius, {radius:.6g}, as M_radius."
    )
    assert values == Matrix(
        row_labels=["g1", "g2"], column_labels=["g1", "g2"], values=[[2, 1], [1, 3]]
    )


def test_the_odd_layout_lists_a_matrix_under_its_own_heading() -> None:
    builder = _builder()
    builder.reset().add_bin("A", "A").add_stratification("group", ["g1", "g2"])
    builder.add_matrix("M", "group", VALUES).add_transition("a", ["A"], [], "M_radius")
    model = builder.build(
        Aggregate(
            bin_amounts={"A": 1},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
        ),
        engine="DifferenceEquations",
    )

    blocks = model.describe(layout="odd").section("submodels").blocks
    headings = [block.heading for block in blocks]

    assert "Matrix M" in headings
    (text,) = [block for block in blocks if block.heading == "Matrix M"]
    assert isinstance(text, Text) and text.text.startswith(
        "Between the categories of 'group'. A rate reads"
    )


def test_a_saved_model_keeps_its_matrices(tmp_path: Path) -> None:
    model = _build(_builder())
    model.to_json(tmp_path / "model.json")

    loaded = Model.from_json(tmp_path / "model.json")

    assert loaded.process.matrices == model.process.matrices


def test_a_reset_builder_holds_no_matrix() -> None:
    builder = _builder()

    assert builder.reset().process().matrices == []
    assert _builder().clone().process().matrices[0].values == VALUES


@pytest.mark.parametrize(
    ("values", "stratification", "message"),
    [
        (
            {"g1": VALUES["g1"]},
            "group",
            "Matrix 'M' has no row for the category 'g2' of 'group'",
        ),
        (
            {"g1": VALUES["g1"], "g2": {"g2": 3.0}},
            "group",
            "Matrix 'M' has no value in row 'g2' for the category 'g1' of 'group'",
        ),
        (
            {**VALUES, "g3": {}},
            "group",
            "Matrix 'M' names 'g3', which is not a category of 'group'",
        ),
        (
            {"g1": VALUES["g1"], "g2": {"g1": math.nan, "g2": 3.0}},
            "group",
            "Matrix 'M' has NaN in row 'g2' and column 'g1'; each value must be a "
            "finite number",
        ),
        (
            VALUES,
            "other",
            "Matrix 'M' is between the categories of 'other', which the process "
            "does not have",
        ),
    ],
)
def test_a_matrix_with_a_missing_category_is_refused(
    values: dict[str, dict[str, float]], stratification: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(")):
        _build(_builder(values, stratification))


def _scaled(a1: float | None, scale: str = "a1") -> ModelBuilder:
    """`_builder` with the row of ``g1`` scaled by ``scale`` and the column of
    ``g2`` by ``b2`` at 1; ``a1`` is the value of ``a1``."""
    builder = _builder()
    builder.add_parameter("a1", a1).add_parameter("b2", 1.0)
    builder.add_parameter("f", "2 * k")
    builder.add_matrix(
        "N", "group", VALUES, row_scales={"g1": scale}, column_scales={"g2": "b2"}
    )
    return builder


def test_a_description_names_the_scales_and_the_radius_at_their_values() -> None:
    blocks = _build(_scaled(2.0)).describe().section("parameters").blocks
    (text,) = [b for b in blocks if isinstance(b, Text) and b.heading == "Matrix N"]

    # [[4, 2], [1, 3]]: the eigenvalues 5 and 2.
    assert text.text == (
        "Between the categories of 'group'. A parameter scales the row of 'g1' by a1, "
        "the column of 'g2' by b2; the values below are before the scales. A rate "
        "reads matrix_sum(N, e); a rate, a formula parameter or a calibration "
        "constraint reads its entries as N_<row>_<column> and its spectral radius, "
        "5, as N_radius."
    )
    blocks = _build(_scaled(None)).describe().section("parameters").blocks
    (text,) = [b for b in blocks if isinstance(b, Text) and b.heading == "Matrix N"]
    assert "its spectral radius, at the values of its scales, as N_radius." in text.text


def test_a_saved_model_keeps_the_scales_of_its_matrices(tmp_path: Path) -> None:
    model = _build(_scaled(2.0))
    model.to_json(tmp_path / "model.json")

    (_, matrix) = Model.from_json(tmp_path / "model.json").process.matrices
    assert (matrix.row_scales, matrix.column_scales) == ({"g1": "a1"}, {"g2": "b2"})
    assert matrix.scales() == [("row", "g1", "a1"), ("column", "g2", "b2")]


@pytest.mark.parametrize(
    ("scale", "message"),
    [
        (
            "a9",
            "Matrix 'N' scales the row of 'g1' by 'a9', which is not a parameter of "
            "the model; a scale is a constant or a parameter to calibrate",
        ),
        (
            "f",
            "Matrix 'N' scales the row of 'g1' by 'f', a formula parameter; a scale "
            "is a constant or a parameter to calibrate",
        ),
    ],
)
def test_a_scale_of_an_unknown_parameter_is_refused(scale: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build(_scaled(2.0, scale))


def test_a_scale_of_a_category_the_stratification_does_not_have_is_refused() -> None:
    builder = _builder().add_parameter("a1", 2.0)
    builder.add_matrix("N", "group", VALUES, column_scales={"g3": "a1"})

    with pytest.raises(
        ValueError,
        match="Matrix 'N' scales the column of 'g3' by 'a1', but 'g3' is not a "
        "category of 'group'",
    ):
        _build(builder)
