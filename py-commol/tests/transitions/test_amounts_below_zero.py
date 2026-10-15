"""A difference step that leaves a compartment below zero stops the run."""

import pytest

from commol import InfeasibleStateError, Simulation
from tests.support import a_to_b_builder, build, build_ode


def test_a_step_that_moves_more_than_its_source_holds_raises() -> None:
    model = build(a_to_b_builder(parameters={"k": 1.5}))

    with pytest.raises(InfeasibleStateError) as raised:
        Simulation(model).run(3)

    assert str(raised.value).startswith(
        "The flows of step 0 leave compartment 'A' at -500, below zero: it held "
        "1000, and the flows out of it move 't_ab' 1500."
    )
    assert isinstance(raised.value, ValueError)


def test_a_rate_that_empties_its_source_exactly_runs() -> None:
    results = Simulation(build(a_to_b_builder(parameters={"k": 1.0}))).run(3)

    assert results["A"] == [1000.0, 0.0, 0.0, 0.0]


def test_run_at_raises_at_the_step_where_the_amount_drops_below_zero() -> None:
    model = build(a_to_b_builder(rate="if(step >= 4, 2, 0)"))

    with pytest.raises(InfeasibleStateError, match="step 4 leave compartment 'A'"):
        Simulation(model).run_at([2, 6])


def test_a_differential_model_is_not_checked() -> None:
    results = Simulation(build_ode(a_to_b_builder(parameters={"k": 1.5}))).run(3)

    assert min(results["A"]) >= 0.0
