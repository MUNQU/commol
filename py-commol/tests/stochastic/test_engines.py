"""The stochastic engines of an aggregate model: one process, whole units.

A stochastic engine runs the process a deterministic engine runs, with no
change to it. Its mean is the run of its mean-field engine
(`MEAN_FIELD`): the differential engine for ``StochasticEvents``, the
difference engine for ``StochasticSteps``. The Rust tests of
``commol-stochastic`` check the closed forms; these check the layers.
"""

import math

import pytest

from commol import (
    InfeasibleStateError,
    InputSpace,
    SensitivityAnalysis,
    Simulation,
)
from commol.api.results import (
    SimulationReplicates,
    SimulationResults,
)
from commol.context.engine import EngineName
from tests.support import (
    MEAN_FIELD,
    STOCHASTIC_ENGINES,
    TIGHT,
    a_to_b_builder,
    bounded,
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

        figure = replicates.plot.series()
        report = replicates.report(0.9, positions=[5.0, 10.0], window=5.0, windows=2)

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


class TestReplicates:
    @staticmethod
    def _replicates() -> SimulationReplicates:
        simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
        return simulation.replicates(3, seed=7).run_at(POSITIONS)

    def test_replicates_share_their_positions(self) -> None:
        runs = [
            SimulationResults([1.0], {"A": [1.0]}),
            SimulationResults([2.0], {"A": [1.0]}),
        ]

        with pytest.raises(ValueError, match="Replicate 1 is at other positions"):
            SimulationReplicates(runs, seed=1, positions=[1.0])

    @pytest.mark.parametrize(
        "statistic",
        [SimulationReplicates.mean, lambda r: r.quantiles([0.5])],
        ids=["mean", "quantiles"],
    )
    def test_a_statistic_of_no_replicates_is_an_error(self, statistic) -> None:
        with pytest.raises(ValueError, match="There are no replicates"):
            statistic(SimulationReplicates([], seed=1, positions=[1.0]))

    def test_a_quantile_level_is_a_probability(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1, got 1.5"):
            self._replicates().quantiles([0.5, 1.5])

    def test_the_statistics_of_an_unknown_output_name_the_outputs(self) -> None:
        with pytest.raises(KeyError, match=r"'D' is not an output.*\['A', 'B'\]"):
            self._replicates().member_statistics("D", confidence_level=0.9)

    def test_neighbouring_seeds_share_no_replicate(self) -> None:
        simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
        first = simulation.replicates(3, seed=42).run_at(POSITIONS)
        second = simulation.replicates(3, seed=43).run_at(POSITIONS)

        assert not {run.seed for run in first} & {run.seed for run in second}

    def test_replicates_print_their_count_their_seed_and_their_positions(
        self,
    ) -> None:
        assert repr(self._replicates()) == (
            "SimulationReplicates(3 replicates, seed=7, positions=[1.0, 5.0, 10.0])"
        )


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
class TestTheEngine:
    def test_a_step_continues_the_run_where_it_stopped(
        self, engine: EngineName
    ) -> None:
        simulation = Simulation(build_stochastic(a_to_b_builder(), engine))
        two_steps = simulation.run(2, seed=8)
        simulation.run(1, seed=8)

        simulation.engine.step()

        assert simulation.engine.state == [two_steps["A"][2], two_steps["B"][2]]

    def test_a_rate_that_is_not_a_number_stops_the_run(
        self, engine: EngineName
    ) -> None:
        simulation = Simulation(
            build_stochastic(a_to_b_builder(rate="k * sqrt(-1)"), engine)
        )

        with pytest.raises(
            RuntimeError, match="The rate of flow 't_ab' at .* is not a"
        ):
            simulation.run(2, seed=1)


def test_replicates_takes_a_count() -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))

    replicates = simulation.replicates(count=3, seed=7).run(5)

    assert len(replicates) == 3


def test_a_sensitivity_analysis_gets_the_seed_rule_of_the_engine() -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
    inputs = InputSpace.uniform_on_bounds([bounded("k", 0.01, 0.5)])

    with pytest.raises(ValueError) as error:
        SensitivityAnalysis(simulation, inputs, replicates=2)

    assert str(error.value) == (
        "The StochasticSteps engine is stochastic: set seed, so the replicates "
        "and the analysis are reproducible."
    )
