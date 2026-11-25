"""An observation of the model plus a known background, which every reader of
the observation adds after its scale and its factor."""

import math
from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
    scores,
)
from commol._simulation.panels import (
    cumulative_panels,
    replicate_observation_panels,
    replicate_panels,
    run_observation_panels,
    run_panels,
)
from commol._simulation.results import SimulationResults
from commol.observations import (
    DistributionNoise,
    NoNoise,
    NormalObservations,
    PoissonObservations,
)
from commol.plotting.panels import ObservationPanel, Panel
from commol.profiles import PredictionProfile, PredictionSpec, Threshold
from commol.resampling import ResidualBootstrap
from commol.uncertainty import ParameterSample, Predictive, SampleMethod
from commol.uncertainty.panels import (
    predictive_cumulative_panels,
    predictive_observation_panels,
    predictive_panels,
)
from tests.support import (
    ENGINE_REL,
    a_to_b_builder,
    abc_builder,
    bounded,
    build,
    build_stochastic,
    line_calibrator,
    sum_calibrator,
)
from tests.support.plotted import fit, predictive_of

# The factor gives at most 48 of the model, so half of what is observed is
# background.
FACTOR, BACKGROUND = 0.3, 50.0

# log p(3) of a Poisson of mean 2.
THREE_AT_A_MEAN_OF_TWO = 3 * math.log(2) - 2 - math.log(6)

# ``5 + 2 t`` at t = 1, …, 10 plus a fixed draw of unit noise.
LINE = [7.4, 8.6, 11.3, 12.8, 15.1, 16.4, 19.9, 20.6, 23.5, 24.7]


def _fitted(run: dict[str, list[float]], background: float) -> dict[str, float]:
    """The fit of ``k1`` and of a factor to ``FACTOR · B + BACKGROUND`` of
    `run`, each observation with `background`."""
    observations = [
        ObservedDataPoint(
            at=at,
            output="B",
            value=FACTOR * run["B"][at] + BACKGROUND,
            factor_id="f",
            background=background,
        )
        for at in range(0, 100, 5)
    ]
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[
            bounded("k1", initial_guess=0.3),
            bounded("f", 0.01, 2.0, kind="factor", initial_guess=1.0),
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=2000, verbose=False),
        seed=42,
    )
    model = build(abc_builder(parameters={"k1": None, "k2": 0.05}))
    return Calibrator(Simulation(model), problem).run().best_parameters


def test_a_count_model_reads_the_background_where_the_model_gives_zero() -> None:
    # B is 0 at the start of the run.
    calibrator = sum_calibrator(
        [ObservedDataPoint(at=0, output="B", value=3.0, background=2.0)],
        {"B": PoissonObservations()},
    )

    (density,) = calibrator._engine_problem.pointwise_log_likelihood([0.1])

    assert density == pytest.approx(THREE_AT_A_MEAN_OF_TWO, rel=1e-12)


def test_a_calibration_reads_the_background(truth_run: dict[str, list[float]]) -> None:
    truth = {"k1": 0.1, "f": FACTOR}

    assert _fitted(truth_run, BACKGROUND) == pytest.approx(truth, rel=1e-4)

    without = _fitted(truth_run, 0.0)
    for name, value in truth.items():
        assert abs(without[name] / value - 1) > 0.1, name


@pytest.mark.parametrize("background", [-0.5, math.inf, math.nan])
@pytest.mark.parametrize(
    "reader",
    [
        lambda background: ObservedDataPoint(
            at=1, output="B", value=1.0, background=background
        ),
        lambda background: PredictionSpec(at=1, output="B", background=background),
    ],
    ids=["observation", "prediction"],
)
def test_an_invalid_background_is_refused(reader, background: float) -> None:
    message = "background is a finite number of at least 0, got"

    with pytest.raises(ValidationError, match=message):
        reader(background)


def _points(
    backgrounds: Sequence[float], at: Sequence[int] = (10, 20), **fields: object
) -> list[ObservedDataPoint]:
    """Observations of B, one per background."""
    return [
        ObservedDataPoint(
            at=position, output="B", value=1.0, background=background, **fields
        )
        for position, background in zip(at, backgrounds, strict=True)
    ]


def _series(source: str, observed: list[ObservedDataPoint]) -> Panel:
    """The panel of B of a run, of its replicates or of a predictive."""
    if source == "run":
        (panel,) = run_panels(_run(), observed, series=["B"])
    elif source == "replicates":
        (panel,) = replicate_panels(_replicates(), 0.9, observed, series=["B"])
    else:
        predictive = predictive_of([0.02, 0.05, 0.1])
        (panel,) = predictive_panels(predictive, 0.9, observed, series=["B"])
    return panel


def _observations(source: str, observed: list[ObservedDataPoint]) -> ObservationPanel:
    """The panel of the observed series of B, of the same results."""
    if source == "run":
        (panel,) = run_observation_panels(_run(), observed)
    elif source == "replicates":
        (panel,) = replicate_observation_panels(_replicates(), 0.9, observed)
    else:
        predictive = predictive_of([0.02, 0.05, 0.1])
        (panel,) = predictive_observation_panels(predictive, 0.9, observed)
    return panel


def _run() -> SimulationResults:
    return Simulation(build(a_to_b_builder())).run(40)


def _replicates():
    return Simulation(build_stochastic(a_to_b_builder())).replicates(5, seed=2).run(40)


def _assert_moved(
    panel: Panel | ObservationPanel,
    plain: Panel | ObservationPanel,
    by: float | Sequence[float],
) -> None:
    """The line, and the band if it has one, of `panel` are those of `plain`
    plus `by`."""
    line = panel.line if isinstance(panel, Panel) else panel.model
    before = plain.line if isinstance(plain, Panel) else plain.model
    assert line.positions == before.positions
    np.testing.assert_allclose(
        line.values, np.asarray(before.values) + by, rtol=ENGINE_REL
    )
    assert (panel.band is None) == (plain.band is None)
    if panel.band is not None and plain.band is not None:
        for bound in ("lower", "upper"):
            np.testing.assert_allclose(
                getattr(panel.band, bound),
                np.asarray(getattr(plain.band, bound)) + by,
                rtol=ENGINE_REL,
            )


class TestReaders:
    def test_a_predictive_reads_the_background(self) -> None:
        rates = [0.02, 0.1]
        observed = [
            ObservedDataPoint(
                at=10, output="events", value=1.0, window=5, background=2.0
            ),
            ObservedDataPoint(at=0, output="B", value=1.0, background=2.0),
        ]

        draws = predictive_of(rates).at_observations(observed, NoNoise()).values

        # ``A = 1000 (1 − k)^t``: what left it over the window, and B at 0.
        for draw, rate in zip(draws, rates, strict=True):
            gain = 1000 * ((1 - rate) ** 5 - (1 - rate) ** 10)
            assert draw == pytest.approx([gain + 2.0, 2.0], rel=ENGINE_REL)

    def test_a_score_reads_the_background(self) -> None:
        calibrator = sum_calibrator(
            [ObservedDataPoint(at=1, output="B", value=100.0)],
            {"B": PoissonObservations()},
        )
        sample = ParameterSample.equally_weighted(
            ["k"], [(0.001, 0.5)], [[0.1]], SampleMethod(name="point")
        )
        held_out = [ObservedDataPoint(at=0, output="B", value=3.0, background=2.0)]

        score = scores.log_score(
            Predictive(calibrator, sample, [0, 1]), held_out, DistributionNoise()
        )

        assert score.per_observation == [
            pytest.approx(THREE_AT_A_MEAN_OF_TWO, rel=1e-12)
        ]

    @pytest.mark.parametrize("source", ["run", "replicates", "predictive"])
    def test_the_figures_read_the_background(self, source: str) -> None:
        plain = _points([0.0, 0.0], scale=0.5)
        one = _points([2.0, 2.0], scale=0.5)
        each = _points([2.0, 5.0], scale=0.5)

        # One background: the line covers every position of the results.
        line = _series(source, plain)
        assert len(line.line.positions) == 41
        _assert_moved(_series(source, one), line, 2.0)
        # Backgrounds that differ: each observation reads its own.
        read = _series(source, each)
        assert read.line.positions == [10, 20]
        _assert_moved(
            _observations(source, each), _observations(source, plain), [2.0, 5.0]
        )
        np.testing.assert_allclose(
            read.line.values,
            _observations(source, each).model.values,
            rtol=ENGINE_REL,
        )

    def test_a_cumulative_figure_subtracts_the_background(self) -> None:
        # The last value is below its background: its difference is negative.
        observed = [
            ObservedDataPoint(at=10, output="B", value=12.0, background=2.0),
            ObservedDataPoint(
                at=20,
                output="B",
                value=12.0,
                scale=0.5,
                factor_id="f",
                background=2.0,
            ),
            ObservedDataPoint(at=30, output="B", value=1.0, background=2.0),
        ]
        predictive = predictive_of([0.02, 0.1])

        (of_run,) = cumulative_panels(_run(), observed, fit(f=0.5), ["B"])
        (of_predictive,) = predictive_cumulative_panels(
            predictive, 0.9, [observed[0], observed[2]], ["B"]
        )

        # 10, then 10 / 0.25 more, then 1 less.
        assert of_run.observed is not None and of_predictive.observed is not None
        assert of_run.observed.values == [10.0, 50.0, 49.0]
        assert of_predictive.observed.values == [10.0, 9.0]

    def test_a_report_records_the_backgrounds(self) -> None:
        observed = [
            ObservedDataPoint(at=5, output="B", value=1.0, background=2.0),
            ObservedDataPoint(at=15, output="B", value=1.0),
            ObservedDataPoint(at=5, output="C", value=1.0),
        ]

        report = SimulationResults([]).report([], 5, 4, observed)["observed"]

        assert report["B"]["backgrounds"] == [2.0, None, 0.0, None]
        assert "backgrounds" not in report["C"]

    @pytest.mark.parametrize(
        ("noise", "threshold"),
        [
            (NormalObservations(sd=1.0), Threshold.chi_square(0.9)),
            (
                None,
                Threshold.bootstrap_calibrated(
                    ResidualBootstrap(
                        replicates=5,
                        block_length=1,
                        residual_scale="additive",
                        scheme="moving",
                    ),
                    0.9,
                ),
            ),
        ],
        ids=["chi_square", "bootstrap_calibrated"],
    )
    def test_a_prediction_profile_reads_the_background(
        self, noise: NormalObservations | None, threshold: Threshold
    ) -> None:
        # Far above the prediction, so a step or a tolerance that read it
        # would move the points of the profile.
        background = 1e4
        calibrator = line_calibrator(
            LINE, observation_models={"B": noise} if noise else None
        )

        def profile(background: float):
            result = calibrator.uncertainty(
                PredictionProfile(
                    specs=[PredictionSpec(at=4, output="B", background=background)],
                    threshold=threshold,
                    max_steps=20,
                )
            )
            return result.profiles["B"], result.intervals["B"]

        plain, plain_interval = profile(0.0)
        moved, interval = profile(background)

        assert moved.center == pytest.approx(plain.center + background, rel=ENGINE_REL)
        assert moved.threshold == pytest.approx(plain.threshold, rel=1e-6)
        np.testing.assert_allclose(
            np.asarray(moved.values) - background, plain.values, rtol=1e-6
        )
        np.testing.assert_allclose(
            moved.statistics, plain.statistics, rtol=1e-4, atol=1e-6
        )
        assert interval.lower is not None and interval.upper is not None
        assert interval.lower - background == pytest.approx(
            plain_interval.lower, rel=1e-5
        )
        assert interval.upper - background == pytest.approx(
            plain_interval.upper, rel=1e-5
        )
