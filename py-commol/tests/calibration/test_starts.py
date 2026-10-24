"""Multi-start calibration: where each start begins, and what the result keeps."""

import logging
from types import SimpleNamespace

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model
from tests.support import abc_builder, bounded, build, observed

STEPS = range(10, 101, 10)


def _calibrator(
    truth_run: dict[str, list[float]],
    seed: int | None = 7,
    *,
    order: tuple[str, str] = ("k1", "k2"),
    verbose: bool = False,
    config: NelderMeadConfig | ParticleSwarmConfig | None = None,
) -> Calibrator:
    bounds = {"k1": (0.0, 0.5), "k2": (0.0, 0.3)}
    problem = CalibrationProblem(
        observed_data=observed(truth_run, "B", STEPS) + observed(truth_run, "C", STEPS),
        parameters=[bounded(name, *bounds[name]) for name in order],
        loss_function="sse",
        optimization_config=config
        or NelderMeadConfig(max_iterations=2000, sd_tolerance=1e-12, verbose=verbose),
        seed=seed,
    )
    model: Model = build(abc_builder(parameters={"k1": None, "k2": None}))
    return Calibrator(Simulation(model), problem)


class TestStarts:
    def test_one_start_begins_at_the_initial_guess(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        result = _calibrator(truth_run).run()

        # No guess: the midpoint of each bound.
        assert [start.start for start in result.starts] == [{"k1": 0.25, "k2": 0.15}]

    def test_the_other_starts_fill_one_stratum_of_each_parameter(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        result = _calibrator(truth_run).run(starts=5)

        assert len(result.starts) == 5
        for name, high in (("k1", 0.5), ("k2", 0.3)):
            strata = sorted(
                int(start.start[name] / high * 4) for start in result.starts[1:]
            )
            assert strata == [0, 1, 2, 3]

    def test_one_seed_repeats_the_starts_and_another_moves_them(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        first = _calibrator(truth_run, seed=3).run(starts=4)
        again = _calibrator(truth_run, seed=3).run(starts=4)
        other = _calibrator(truth_run, seed=4).run(starts=4)

        assert first.starts == again.starts
        assert [s.start for s in first.starts[1:]] != [
            s.start for s in other.starts[1:]
        ]

    def test_starts_from_distinct_points_reach_the_parameters_of_the_data(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        result = _calibrator(truth_run).run(starts=5)

        assert len({tuple(s.start.values()) for s in result.starts}) == 5
        for start in result.starts:
            assert start.end == pytest.approx({"k1": 0.1, "k2": 0.05}, abs=1e-5)
        assert result.best_parameters == pytest.approx(
            {"k1": 0.1, "k2": 0.05}, abs=1e-5
        )
        assert result.final_loss == min(start.loss for start in result.starts)

    def test_best_parameters_keep_the_order_of_the_problem(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        for _ in range(3):
            result = _calibrator(truth_run, order=("k2", "k1")).run(starts=3)
            assert list(result.best_parameters) == ["k2", "k1"]
            assert all(list(s.end) == ["k2", "k1"] for s in result.starts)

    def test_without_a_seed_the_drawn_seed_is_logged(
        self, truth_run: dict[str, list[float]], caplog: pytest.LogCaptureFixture
    ) -> None:
        calibrator = _calibrator(truth_run, seed=None)

        with caplog.at_level(logging.INFO, logger="commol.api.calibrator"):
            calibrator.run(starts=2)

        assert f"drew the seed {calibrator.seed}" in caplog.text

    def test_a_verbose_optimizer_shows_one_start(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        with pytest.raises(ValueError, match="verbose optimizer shows one start"):
            _calibrator(truth_run, verbose=True).run(starts=2)

    def test_a_verbose_swarm_repeats_its_fit_from_the_seed(
        self, truth_run: dict[str, list[float]], capfd: pytest.CaptureFixture[str]
    ) -> None:
        swarm = ParticleSwarmConfig(particles=6, max_iterations=4, verbose=True)
        first = _calibrator(truth_run, seed=5, config=swarm).run()
        again = _calibrator(truth_run, seed=5, config=swarm).run()
        quiet = _calibrator(
            truth_run, seed=5, config=swarm.model_copy(update={"verbose": False})
        ).run()

        assert "Particle Swarm Optimization (Verbose Mode)" in capfd.readouterr().out
        assert first.best_parameters == again.best_parameters
        assert first.final_loss == again.final_loss
        # A verbose calibration is the calibration that shows nothing.
        assert first.best_parameters == quiet.best_parameters

    def test_a_calibration_needs_a_start(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        with pytest.raises(
            ValueError, match="starts is a whole number of at least 1, got 0"
        ):
            _calibrator(truth_run).run(starts=0)


def _outcome(start: list[float], loss: float | None) -> SimpleNamespace:
    """A start outcome of the extension: a result, or the error of a failure."""
    if loss is None:
        return SimpleNamespace(start=start, result=None, error="Simulation failed: x")
    return SimpleNamespace(
        start=start,
        result=SimpleNamespace(
            best_parameters=[("k1", start[0]), ("k2", start[1])],
            final_loss=loss,
            iterations=3,
            converged=True,
            termination_reason="SolverConverged",
        ),
        error=None,
    )


class TestFailedStarts:
    """The calibrator reports the starts the extension says failed, and the
    start it chooses."""

    def _with_outcomes(
        self,
        truth_run: dict[str, list[float]],
        outcomes: list[SimpleNamespace],
        best: int = 0,
    ) -> Calibrator:
        calibrator = _calibrator(truth_run)
        calibrator._rust_problem = SimpleNamespace(  # type: ignore[assignment]
            optimize=lambda *_: SimpleNamespace(outcomes=outcomes, best=best),
            seed=calibrator.seed,
        )
        return calibrator

    def test_a_failed_start_is_in_failed(
        self, truth_run: dict[str, list[float]], caplog: pytest.LogCaptureFixture
    ) -> None:
        calibrator = self._with_outcomes(
            truth_run, [_outcome([0.2, 0.1], 4.0), _outcome([0.4, 0.2], None)]
        )

        with caplog.at_level(logging.WARNING, logger="commol.api.calibrator"):
            result = calibrator.run(starts=2)

        assert result.failed == ["Start 1 failed: Simulation failed: x"]
        assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
            "Start 1 failed: Simulation failed: x"
        ]
        assert result.best_parameters == {"k1": 0.2, "k2": 0.1}

    def test_the_result_is_the_start_the_engine_chooses(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        calibrator = self._with_outcomes(
            truth_run, [_outcome([0.2, 0.1], 1.0), _outcome([0.4, 0.2], 1.0)], best=1
        )

        assert calibrator.run(starts=2).best_parameters == {"k1": 0.4, "k2": 0.2}

    def test_every_start_failing_raises_with_each_reason(
        self, truth_run: dict[str, list[float]]
    ) -> None:
        # An expansion below 1, which Nelder-Mead rejects at every start.
        calibrator = _calibrator(truth_run, config=NelderMeadConfig(gamma=0.5))

        with pytest.raises(RuntimeError) as raised:
            calibrator.run(starts=2)

        message = str(raised.value)
        assert message.startswith("Every start failed: Start 0 failed: ")
        assert "; Start 1 failed: " in message
