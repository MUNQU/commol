"""The unit of a matrix sum: the unit of the matrix times the unit of the
expression it reads in each category."""

import pytest

from commol._model.engine_state import formula_tree, transition_flows
from commol._model.unit_checks import (
    UnitConsistencyError,
    define_units,
    formula_unit,
    ureg,
)
from tests.support import FactoryBuilder, a_to_b_builder, build

VALUES = {"g1": {"g1": 2.0, "g2": 1.0}, "g2": {"g1": 1.0, "g2": 3.0}}


def _builder(
    rate: str, matrix_unit: str | None, values: dict[str, dict[str, float]] = VALUES
) -> FactoryBuilder:
    builder = a_to_b_builder(
        parameters={"k": 0.1},
        units={"k": "dimensionless"},
        rate=rate,
        stratifications={"group": {"g1": 0.5, "g2": 0.5}},
        bin_unit="item",
        step_unit="day",
    )
    builder.add_matrix("M", "group", values, unit=matrix_unit)
    return builder


def test_a_matrix_sum_has_the_unit_of_the_matrix_times_the_expression() -> None:
    tree = formula_tree("k * matrix_sum(M, B)")
    units = {"k": "dimensionless", "M": "1/day", "B": "item", "total": "item"}
    define_units("item")

    assert formula_unit(tree, units).units == ureg("item / day").units
    with pytest.raises(ValueError, match="the matrix 'M' has no unit"):
        formula_unit(tree, {"k": "1/day", "B": "item", "total": "item"})


@pytest.mark.parametrize("rate", ["k * matrix_sum(M, B / total)", "k * M_radius"])
def test_a_flow_through_a_matrix_checks_with_the_unit_of_the_matrix(
    rate: str,
) -> None:
    build(_builder(rate, "1/day")).check_unit_consistency()

    with pytest.raises(UnitConsistencyError, match="Unit mismatch"):
        build(_builder(rate, "1/week")).check_unit_consistency()


def test_a_matrix_without_a_unit_stops_the_unit_check() -> None:
    model = build(_builder("k * M_radius", None))

    with pytest.raises(UnitConsistencyError, match="matrices have no unit: M"):
        model.check_unit_consistency()


def test_a_row_of_zeros_keeps_the_unit_of_the_matrix_sum() -> None:
    zero_row = {"g1": {"g1": 0.0, "g2": 0.0}, "g2": {"g1": 1.0, "g2": 3.0}}

    model = build(_builder("k * matrix_sum(M, B / total)", "1/day", zero_row))

    model.check_unit_consistency()

    rates = [flow.rate for flow in transition_flows(model.process, model.structure)]
    assert rates[0] == "k * if(M_g1_g1 != 0, M_g1_g1 * (B_g1 / total_g1), 0)"


@pytest.mark.parametrize(("unit", "error"), [("dimensionless", None), ("1/day", "a1")])
def test_a_scale_of_a_matrix_has_no_unit(unit: str, error: str | None) -> None:
    builder = _builder("k * matrix_sum(M, B / total)", "1/day")
    builder.add_parameter("a1", 2.0, unit=unit)
    builder.add_matrix("N", "group", VALUES, unit="1/day", row_scales={"g1": "a1"})
    model = build(builder)

    if error is None:
        model.check_unit_consistency()
    else:
        with pytest.raises(
            UnitConsistencyError,
            match="Matrix 'N' scales the row of 'g1' by 'a1' in '1/day'; a scale is "
            "dimensionless",
        ):
            model.check_unit_consistency()
