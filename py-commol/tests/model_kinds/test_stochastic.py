"""A stochastic engine: seeds, replicates, and the layers that read them.

The stochastic test engine (`TEST_STOCHASTIC_ENGINE`) moves each unit of an
output out after an exponential time at rate ``k``, so the mean of its runs is
``initial * exp(-k t)``, the curve of the deterministic test engine, and each
value is binomial: ``initial`` trials of probability ``exp(-k t)``.
"""

import math

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Quantile,
    Simulation,
)
from commol.api.plotter import SimulationPlotter
from commol.api.results import SimulationReplicates
from tests.support.kinds import TEST_STOCHASTIC_ENGINE, decay, decay_model
from tests.support.problems import bounded

INITIAL = 1000.0
RATE = 0.5
POSITIONS = [0.5, 1.0, 2.0, 4.0]


def _stochastic(rate: float | None = RATE) -> Simulation:
    return Simulation(
        decay_model(rate=rate, initial=INITIAL, engine=TEST_STOCHASTIC_ENGINE)
    )


class TestSeeds:
    def test_one_seed_repeats_a_run_and_two_seeds_differ(self) -> None:
        simulation = _stochastic()

        first = simulation.run_at(POSITIONS, seed=11)
        again = simulation.run_at(POSITIONS, seed=11)
        other = simulation.run_at(POSITIONS, seed=12)

        assert first == again
        assert first != other
        assert (first.seed, other.seed) == (11, 12)

    def test_a_run_without_a_seed_keeps_the_seed_it_drew(self) -> None:
        simulation = _stochastic()

        drawn = simulation.run(5)
        repeated = simulation.run(5, seed=drawn.seed)

        assert drawn.seed is not None
        assert repeated == drawn

    def test_a_deterministic_engine_ignores_the_seed(self) -> None:
        simulation = Simulation(decay_model(rate=RATE, initial=INITIAL))

        first = simulation.run_at(POSITIONS, seed=1)
        second = simulation.run_at(POSITIONS, seed=2)

        assert first == second
        assert first.seed is None
        assert not simulation.is_stochastic

    @pytest.mark.parametrize("seed", [-1, 2**64, 1.5, True])
    def test_a_seed_is_a_whole_number_of_64_bits(self, seed: object) -> None:
        with pytest.raises(ValueError, match="2\\*\\*64 - 1"):
            _stochastic().run(1, seed=seed)  # ty: ignore[invalid-argument-type]


class TestReplicates:
    def test_the_mean_of_many_replicates_converges_to_its_expectation(self) -> None:
        replicates = _stochastic().replicates(400, seed=3).run_at(POSITIONS)

        mean = replicates.mean()
        for index, position in enumerate(POSITIONS):
            survival = math.exp(-RATE * position)
            standard_error = math.sqrt(INITIAL * survival * (1 - survival) / 400)
            assert abs(mean["x"][index] - decay(INITIAL, RATE, position)) <= (
                5 * standard_error
            )

    def test_each_replicate_keeps_its_seed_and_repeats(self) -> None:
        simulation = _stochastic()

        replicates = simulation.replicates(8, seed=21).run_at(POSITIONS)
        fewer = simulation.replicates(3, seed=21).run_at(POSITIONS)

        assert isinstance(replicates, SimulationReplicates)
        assert len(replicates) == 8
        assert len(set(replicates.seeds)) == 8
        assert list(fewer) == list(replicates[:3])
        rerun = simulation.run_at(POSITIONS, seed=replicates[5].seed)
        assert rerun == replicates[5]

    def test_replicates_on_the_default_grid(self) -> None:
        replicates = _stochastic().replicates(4, seed=2).run(3)

        assert replicates.positions == [0.0, 1.0, 2.0, 3.0]
        assert all(run["x"][0] == INITIAL for run in replicates)

    def test_the_statistics_reduce_each_position_across_replicates(self) -> None:
        replicates = _stochastic().replicates(50, seed=9).run_at(POSITIONS)

        quantiles = replicates.quantiles([0.0, 0.5, 1.0])
        statistics = replicates.member_statistics("x", confidence_level=0.9)

        values = [[run["x"][index] for run in replicates] for index in range(4)]
        assert quantiles[0.0]["x"] == [min(column) for column in values]
        assert quantiles[1.0]["x"] == [max(column) for column in values]
        assert statistics["min"] == quantiles[0.0]["x"]
        assert statistics["mean"] == pytest.approx(replicates.mean()["x"])
        assert all(
            low <= high
            for low, high in zip(statistics["ci_lower"], statistics["ci_upper"])
        )

    def test_a_deterministic_engine_has_no_replicates(self) -> None:
        simulation = Simulation(decay_model())

        with pytest.raises(ValueError, match="TestEngine engine is deterministic"):
            simulation.replicates(10)

    def test_at_least_one_replicate(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            _stochastic().replicates(0)

    def test_an_uncalibrated_parameter_stops_the_replicates(self) -> None:
        with pytest.raises(ValueError, match="Parameters requiring calibration: k"):
            _stochastic(rate=None).replicates(2, seed=1).run(1)


class TestLayers:
    def test_a_plot_of_replicates_draws_their_mean_and_band(self) -> None:
        simulation = _stochastic()
        replicates = simulation.replicates(20, seed=4).run(4)

        figure = SimulationPlotter(simulation, replicates).plot_series()

        axis = figure.get_axes()[0]
        assert axis.get_lines()[0].get_ydata().tolist() == replicates.mean()["x"]
        assert len(axis.collections) >= 1
        labels = [text.get_text() for text in axis.get_legend().get_texts()]
        assert "Mean of 20 replicates" in labels

    def test_a_report_reads_replicates_as_ensemble_runs(self) -> None:
        simulation = _stochastic()
        replicates = simulation.replicates(30, seed=5).run(4)

        report = replicates.report(
            0.9,
            positions=[2.0, 4.0],
            window=2.0,
            windows=2,
            observed_data=[ObservedDataPoint(at=2.0, output="x", value=370.0)],
        )

        leaf = report["series"]["x"]
        assert set(leaf) >= {"mean", "ci_lower", "ci_upper"}


def _calibration_problem(
    summary: object,
    *,
    replicates: int | None = 40,
    seed: int | None = 7,
) -> CalibrationProblem:
    """Observations of the expected curve of rate 0.5, and ``k`` to find."""
    return CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=at, output="x", value=decay(INITIAL, RATE, at))
            for at in (0.5, 1.0, 2.0, 3.0)
        ],
        parameters=[bounded("k", 0.05, 2.0, initial_guess=1.0)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=300, sd_tolerance=1e-10),
        replicates=replicates,
        replicate_summary=summary,  # ty: ignore[invalid-argument-type]
        seed=seed,
    )


class TestCalibration:
    @pytest.mark.parametrize(
        "summary", ["mean", "median", Quantile(0.5), "crps"], ids=str
    )
    def test_each_summary_recovers_the_rate(self, summary: object) -> None:
        result = Calibrator(_stochastic(rate=None), _calibration_problem(summary)).run()

        # 40 replicates of 1000 units: the rate is found within a few percent.
        assert result.best_parameters["k"] == pytest.approx(RATE, rel=0.05)

    def test_one_seed_gives_the_same_calibration(self) -> None:
        first = Calibrator(_stochastic(rate=None), _calibration_problem("mean")).run()
        second = Calibrator(_stochastic(rate=None), _calibration_problem("mean")).run()

        assert first.best_parameters == second.best_parameters
        assert first.final_loss == second.final_loss

    def test_a_stochastic_calibration_needs_a_summary_and_lists_them(self) -> None:
        with pytest.raises(ValueError, match="crps") as error:
            Calibrator(_stochastic(rate=None), _calibration_problem(None))

        assert "Quantile(level)" in str(error.value)

    def test_a_stochastic_calibration_needs_a_seed(self) -> None:
        with pytest.raises(ValueError, match="set seed"):
            Calibrator(_stochastic(rate=None), _calibration_problem("mean", seed=None))

    def test_a_stochastic_calibration_needs_replicates(self) -> None:
        with pytest.raises(ValueError, match="set replicates"):
            Calibrator(
                _stochastic(rate=None), _calibration_problem("mean", replicates=None)
            )

    def test_a_deterministic_calibration_does_not_use_them(self) -> None:
        problem = _calibration_problem(None, replicates=None, seed=None)

        result = Calibrator(
            Simulation(decay_model(rate=None, initial=INITIAL)), problem
        ).run()

        assert result.best_parameters["k"] == pytest.approx(RATE, rel=1e-4)

    def test_a_quantile_level_is_between_zero_and_one(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to 1"):
            Quantile(1.5)

    def test_a_quantile_takes_no_other_field(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs"):
            Quantile(0.5, unknown=1)
