"""Whole units: a stochastic engine starts from whole amounts.

A total that is not whole is an error. Initial amounts that are not whole
need a rounding in the settings of the engine, with no default:
``"largest_remainder"`` (the same whole amounts in every run) or ``"random"``
(a multinomial draw per run, whose mean is the amounts of the model).
"""

import math

import pytest

from commol import Simulation, StochasticEventsSettings, StochasticStepsSettings
from commol.context.engine import EngineName
from tests.support import STOCHASTIC_ENGINES, a_to_b_builder, build_stochastic

GROUPS = {"group": {f"g{index}": 0.1 for index in range(10)}}
"""Ten groups of a tenth each: half a unit of B in each group."""


def _seeded(
    engine: EngineName,
    settings: StochasticEventsSettings | StochasticStepsSettings | None = None,
):
    builder = a_to_b_builder(
        rate="k * B / total",
        amounts={"A": 995, "B": 5},
        stratifications=GROUPS,
    )
    return build_stochastic(builder, engine, settings)


def _settings(engine: EngineName, rounding: str):
    if engine == "StochasticEvents":
        return StochasticEventsSettings(rounding=rounding)  # ty: ignore[invalid-argument-type]
    return StochasticStepsSettings(rounding=rounding)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
class TestWholeUnits:
    def test_whole_amounts_need_no_rounding(self, engine: EngineName) -> None:
        model = build_stochastic(a_to_b_builder(), engine)

        first = Simulation(model).run(0, seed=1)

        assert (first["A"][0], first["B"][0]) == (1000, 0)

    def test_amounts_that_are_not_whole_need_a_rounding_and_list_both(
        self, engine: EngineName
    ) -> None:
        with pytest.raises(ValueError, match="largest_remainder") as error:
            _seeded(engine)

        assert '"random"' in str(error.value)
        assert engine in str(error.value)
        assert "'A_g0' (99.5)" in str(error.value)

    def test_a_bin_that_is_not_whole_names_the_engine_and_the_bin(
        self, engine: EngineName
    ) -> None:
        builder = a_to_b_builder(amounts={"A": 1000.5, "B": 0})

        with pytest.raises(ValueError, match=f"The {engine} engine") as error:
            build_stochastic(builder, engine, _settings(engine, "random"))

        assert "the amount of each bin must be a whole number" in str(error.value)
        assert "'A' (1000.5)" in str(error.value)

    def test_the_largest_remainder_starts_every_run_alike(
        self, engine: EngineName
    ) -> None:
        model = _seeded(engine, _settings(engine, "largest_remainder"))

        replicates = Simulation(model).replicates(5, seed=2).run(0)

        starts = [run.rows()[0] for run in replicates]
        assert all(start == starts[0] for start in starts)
        assert all(value == int(value) for value in starts[0])
        assert sum(starts[0]) == 1000
        amounts = model.structure.initial_state(model.process)
        for start, compartment in zip(starts[0], amounts, strict=True):
            assert abs(start - compartment.amount) < 1

    def test_a_random_rounding_starts_on_average_from_the_model(
        self, engine: EngineName
    ) -> None:
        model = _seeded(engine, _settings(engine, "random"))

        replicates = Simulation(model).replicates(2000, seed=3).run(0)

        for run in replicates:
            assert sum(run.rows()[0]) == 1000
        for index in range(10):
            values = [run[f"B_g{index}"][0] for run in replicates]
            mean = sum(values) / len(values)
            error = math.sqrt(0.5 * 0.5 / len(values))
            assert abs(mean - 0.5) < 4.5 * error, index
