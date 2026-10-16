"""The stochastic engines of an aggregate model: one process, whole units.

A stochastic engine runs the process a deterministic engine runs, with no
change to it. Its mean is the run of its mean-field engine
(`MEAN_FIELD`): the differential engine for ``StochasticEvents``, the
difference engine for ``StochasticSteps``. The Rust tests of
``commol-stochastic`` check the closed forms; these check the layers.
"""

import math

import pytest

from commol import InfeasibleStateError, Simulation, simulation_report
from commol.api.plotter import SimulationPlotter
from commol.context.engine import EngineName
from tests.support import (
    MEAN_FIELD,
    STOCHASTIC_ENGINES,
    TIGHT,
    a_to_b_builder,
    build_stochastic,
)

POSITIONS = [1.0, 5.0, 10.0]


def _mean_field(engine: EngineName) -> list[float]:
    """The amount of A of the deterministic limit of `engine` at `POSITIONS`."""
    builder = a_to_b_builder()
    settings = TIGHT if MEAN_FIELD[engine] == "DifferentialEquations" else None
    model = builder.build(
        builder.structure, engine=MEAN_FIELD[engine], settings=settings
    )
    return Simulation(model).run_at(POSITIONS)["A"]


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
class TestOneProcess:
    def test_the_mean_of_many_runs_is_the_deterministic_run(
        self, engine: EngineName
    ) -> None:
        simulation = Simulation(build_stochastic(a_to_b_builder(), engine))

        replicates = simulation.replicates(400, seed=3).run_at(POSITIONS)

        expected = _mean_field(engine)
        for index, at in enumerate(POSITIONS):
            values = [run["A"][index] for run in replicates]
            mean = sum(values) / len(values)
            spread = math.sqrt(
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            )
            error = spread / math.sqrt(len(values))
            assert abs(mean - expected[index]) < 4.5 * error, at

    def test_every_run_moves_whole_units_and_keeps_the_total(
        self, engine: EngineName
    ) -> None:
        simulation = Simulation(build_stochastic(a_to_b_builder(), engine))

        replicates = simulation.replicates(20, seed=4).run(10)

        for run in replicates:
            for a, b in zip(run["A"], run["B"], strict=True):
                assert a == int(a) and b == int(b)
                assert a + b == 1000

    def test_one_seed_repeats_a_run_and_two_seeds_differ(
        self, engine: EngineName
    ) -> None:
        simulation = Simulation(build_stochastic(a_to_b_builder(), engine))

        first = simulation.run(10, seed=11)

        assert simulation.run(10, seed=11) == first
        assert simulation.run(10, seed=12) != first

    def test_a_plot_and_a_report_read_the_replicates(self, engine: EngineName) -> None:
        simulation = Simulation(build_stochastic(a_to_b_builder(), engine))
        replicates = simulation.replicates(30, seed=5).run(10)

        figure = SimulationPlotter(simulation, replicates).plot_series()
        report = simulation_report(
            simulation,
            replicates.mean(),
            at=[5.0, 10.0],
            window=5.0,
            num_windows=2,
            ensemble_runs=replicates,
            confidence_level=0.9,
        )

        line = figure.get_axes()[0].get_lines()[0]
        assert line.get_ydata().tolist() == replicates.mean()["A"]
        leaf = report["series"]["A"]
        assert leaf["ci_lower"] <= leaf["mean"] <= leaf["ci_upper"]


class TestPositions:
    def test_the_event_engine_reports_anywhere_and_the_step_engine_at_steps(
        self,
    ) -> None:
        events = Simulation(build_stochastic(a_to_b_builder(), "StochasticEvents"))
        steps = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))

        assert len(events.run_at([0.5, 2.25], seed=1)["A"]) == 2
        with pytest.raises(ValueError, match="integer steps"):
            steps.run_at([0.5], seed=1)


class TestProbabilities:
    def test_a_per_step_probability_above_one_names_its_flow(self) -> None:
        model = build_stochastic(a_to_b_builder(rate="1.2"), "StochasticSteps")

        with pytest.raises(InfeasibleStateError, match="'t_ab' 1.2") as error:
            Simulation(model).run(3, seed=1)

        assert "step 0" in str(error.value)
        assert "'A'" in str(error.value)
