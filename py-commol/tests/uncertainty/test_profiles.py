"""Profile likelihood: parameter and prediction profiles, their thresholds,
intervals and verdicts."""

from statistics import NormalDist

import pytest

from commol import (
    NormalObservations,
    PredictionProfile,
    PredictionSpec,
    ProfileLikelihood,
    ResidualBootstrap,
    Threshold,
)
from commol.commol_rs import _commol_rs as commol_rs
from tests.support import line_calibrator

# ``5 + 2 t`` at t = 1, …, 10 plus a fixed draw of unit noise.
VALUES = [7.4, 8.6, 11.3, 12.8, 15.1, 16.4, 19.9, 20.6, 23.5, 24.7]
N = len(VALUES)
T_MEAN = (N + 1) / 2
STT = sum((t - T_MEAN) ** 2 for t in range(1, N + 1))


def _least_squares() -> tuple[float, float, float]:
    """The slope, the intercept and the sum of squares of the line fit."""
    y_mean = sum(VALUES) / N
    slope = (
        sum((t - T_MEAN) * (y - y_mean) for t, y in enumerate(VALUES, start=1)) / STT
    )
    intercept = y_mean - slope * T_MEAN
    sse = sum((y - intercept - slope * t) ** 2 for t, y in enumerate(VALUES, start=1))
    return slope, intercept, sse


class TestThresholds:
    def test_linear_normal_profiles_are_exact(self) -> None:
        calibrator = line_calibrator(
            VALUES, observation_models={"B": NormalObservations(sd=1.0)}
        )
        level = 0.9
        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c", "B"],
                threshold=Threshold.chi_square(level),
                max_steps=100,
            )
        )

        z = NormalDist().inv_cdf((1 + level) / 2)
        slope, intercept, _ = _least_squares()
        half_widths = {
            "c": z / STT**0.5,
            "B": z * (1 / N + T_MEAN**2 / STT) ** 0.5,
        }
        for name, center in (("c", slope), ("B", intercept)):
            interval = result.intervals[name]
            assert result.profiles[name].threshold == pytest.approx(z**2 / 2)
            assert interval.lower == pytest.approx(center - half_widths[name], rel=1e-6)
            assert interval.upper == pytest.approx(center + half_widths[name], rel=1e-6)

    def test_beale_threshold_gives_the_t_interval_of_a_linear_model(self) -> None:
        calibrator = line_calibrator(VALUES)
        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c"], threshold=Threshold.beale(0.95), max_steps=100
            )
        )

        # The 0.975 quantile of Student's t with 8 degrees of freedom.
        t_quantile = 2.306004135204166
        slope, _, sse = _least_squares()
        half_width = t_quantile * (sse / (N - 2) / STT) ** 0.5
        assert result.profiles["c"].threshold == pytest.approx(
            sse * t_quantile**2 / (N - 2)
        )
        assert result.intervals["c"].lower == pytest.approx(slope - half_width)
        assert result.intervals["c"].upper == pytest.approx(slope + half_width)

    def test_bootstrap_threshold_is_the_quantile_of_the_replicates(self) -> None:
        calibrator = line_calibrator(VALUES, seed=11)
        bootstrap = ResidualBootstrap(
            replicates=40,
            block_length=1,
            residual_scale="additive",
            scheme="moving",
        )
        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c"],
                threshold=Threshold.bootstrap_calibrated(bootstrap, 0.9),
                max_steps=100,
            )
        )

        # The same replicates: on the sum of squares of a line, the profile
        # of a replicate at the fitted slope rises by Stt (ĉ_b − ĉ)².
        replicates = calibrator.uncertainty(bootstrap)
        slope = replicates.fit.best_parameters["c"]
        rises = [STT * (row[0] - slope) ** 2 for row in replicates.sample.values]
        expected = commol_rs.inference.weighted_quantiles(
            rises, [1.0] * len(rises), [0.9]
        )[0]
        assert result.profiles["c"].threshold == pytest.approx(expected, rel=1e-6)
        assert result.profiles["c"].replicate_rises == pytest.approx(
            rises, rel=1e-4, abs=1e-8
        )

    def test_chi_square_threshold_needs_a_likelihood(self) -> None:
        calibrator = line_calibrator(VALUES)

        with pytest.raises(ValueError) as error:
            calibrator.uncertainty(
                ProfileLikelihood(
                    parameters=["c"], threshold=Threshold.chi_square(0.95)
                )
            )

        message = str(error.value)
        assert "observation models" in message
        assert "Threshold.beale" in message
        assert "Threshold.bootstrap_calibrated" in message

    def test_beale_threshold_needs_a_sum_of_squares(self) -> None:
        calibrator = line_calibrator(
            VALUES, observation_models={"B": NormalObservations(sd=1.0)}
        )

        with pytest.raises(ValueError, match="sum of squares"):
            calibrator.uncertainty(
                ProfileLikelihood(parameters=["c"], threshold=Threshold.beale(0.95))
            )

    def test_a_profile_names_parameters_of_the_problem(self) -> None:
        calibrator = line_calibrator(VALUES)

        with pytest.raises(ValueError, match="'k' is not a parameter"):
            calibrator.uncertainty(
                ProfileLikelihood(parameters=["k"], threshold=Threshold.beale(0.95))
            )

    @pytest.mark.parametrize("level", [0.0, 1.0])
    def test_a_level_is_strictly_between_0_and_1(self, level: float) -> None:
        with pytest.raises(ValueError, match="level"):
            Threshold.chi_square(level)


class TestPredictionProfiles:
    def test_prediction_profile_of_a_linear_model_is_exact(self) -> None:
        calibrator = line_calibrator(
            VALUES, observation_models={"B": NormalObservations(sd=1.0)}
        )
        level = 0.9
        # Inside the data and past them: a band of two positions.
        specs = [
            PredictionSpec(at=4, output="B", label="B at 4"),
            PredictionSpec(at=15, output="B", label="B at 15"),
        ]
        result = calibrator.uncertainty(
            PredictionProfile(specs=specs, threshold=Threshold.chi_square(level))
        )

        z = NormalDist().inv_cdf((1 + level) / 2)
        slope, intercept, _ = _least_squares()
        for spec in specs:
            mean = intercept + slope * spec.at
            half_width = z * (1 / N + (spec.at - T_MEAN) ** 2 / STT) ** 0.5
            interval = result.intervals[spec.label]
            assert result.profiles[spec.label].center == pytest.approx(mean)
            assert interval.lower == pytest.approx(mean - half_width, rel=1e-5)
            assert interval.upper == pytest.approx(mean + half_width, rel=1e-5)
