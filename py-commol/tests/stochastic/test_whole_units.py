"""Whole units: a stochastic engine starts from whole amounts.

Initial amounts that are not whole, of a bin or of a compartment, need a
rounding in the settings of the engine, with no default:
``"largest_remainder"`` (the same whole amounts in every run) or ``"random"``
(a multinomial draw per run). Either makes each bin its nearest whole number
and splits it among its compartments.
A method that varies an initial condition needs a rounding too, and the
engine checks known values, such as a write-back or the draws of a sample,
when it writes them.
"""

import math

import pytest

from commol import Simulation
from commol.engines import StochasticEventsSettings, StochasticStepsSettings
from commol.engines.engine import EngineName
from commol.quantities import ValueAt
from commol.sensitivity import InputSpace, Propagation, SensitivityAnalysis
from commol.uncertainty import ParameterSample, SampleMethod
from tests.support import (
    STOCHASTIC_ENGINES,
    a_to_b_builder,
    bounded,
    build_stochastic,
    initial_condition_problem,
    needs_a_rounding,
)

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

    @pytest.mark.parametrize("rounding", ["largest_remainder", "random"])
    def test_a_rounding_makes_each_bin_its_nearest_whole_number(
        self, engine: EngineName, rounding: str
    ) -> None:
        # The shares of A sum to 10.499999999999998.
        builder = a_to_b_builder(
            amounts={"A": 10.5, "B": 0.4},
            stratifications={"group": {"g1": 0.15, "g2": 0.85}},
        )
        model = build_stochastic(builder, engine, _settings(engine, rounding))

        starts = Simulation(model).replicates(5, seed=2).run_at([0.0])

        # Each bin rounds on its own: a half rounds up whatever its split, and
        # a bin below half a unit starts empty in every run.
        bins = [
            [run[f"{bin_id}_g1"][0] + run[f"{bin_id}_g2"][0] for bin_id in ("A", "B")]
            for run in starts
        ]
        assert bins == [[11.0, 0.0]] * 5

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


def test_an_analysis_that_varies_an_amount_needs_a_rounding() -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
    inputs = InputSpace.uniform_on_bounds(
        [bounded("B", 0.0, 50.0, kind="initial_condition")]
    )

    with pytest.raises(ValueError) as error:
        SensitivityAnalysis(simulation, inputs, replicates=2, seed=1)

    assert str(error.value) == needs_a_rounding("StochasticSteps")


def _sample_of_b(draws: list[float], low: float = 0.0) -> InputSpace:
    """The initial amount of bin B in ``[low, 50]``, from a sample of ``draws``."""
    sample = ParameterSample.equally_weighted(
        ["B"], [(low, 50.0)], [[draw] for draw in draws], SampleMethod(name="grid")
    )
    return InputSpace.from_sample(
        sample, [bounded("B", low, 50.0, kind="initial_condition")]
    )


@pytest.mark.parametrize("low", [0.0, 0.5])
def test_an_analysis_of_a_sample_writes_its_draws_as_known_amounts(
    low: float,
) -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
    analysis = SensitivityAnalysis(
        simulation, _sample_of_b([10.0, 20.0, 30.0], low), replicates=2, seed=1
    )

    result = analysis.run(Propagation(quantities=[ValueAt(series="B", at=0)]))

    assert [run[0] for run in result.values] == [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]


def test_every_draw_of_a_sample_is_checked_when_the_analysis_is_built() -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))

    with pytest.raises(ValueError, match=r"^Parameter set 1: .*12\.3 is not whole"):
        SensitivityAnalysis(
            simulation, _sample_of_b([10.0, 12.3]), replicates=2, seed=1
        )


def test_a_known_amount_is_written_back_when_it_is_whole() -> None:
    model = build_stochastic(a_to_b_builder(), "StochasticSteps")
    problem = initial_condition_problem("B")

    with pytest.raises(ValueError, match=r"12\.3 is not whole"):
        model.apply_calibration_parameters({"B": 12.3}, problem)
    model.apply_calibration_parameters({"B": 12.0}, problem)
    start = Simulation(model).run_at([0.0], seed=1)

    assert start["B"] == [12.0]
