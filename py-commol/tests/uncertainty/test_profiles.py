"""Profile likelihood: parameter and prediction profiles, their thresholds
and intervals; the verdicts are in test_profile_verdicts.py."""

from statistics import NormalDist


import pytest

from pydantic import ValidationError

from commol import (
    BayesianBootstrap,
    Calibrator,
    IdentifiabilityReport,
    LogNormalObservations,
    NelderMeadConfig,
    NormalObservations,
    ParametricBootstrap,
    PredictionProfile,
    PredictionSpec,
    ProfileInterval,
    ProfileLikelihood,
    ResidualBootstrap,
    Threshold,
    WildBootstrap,
)
from commol.commol_rs import _commol_rs as commol_rs
from tests.support import (
    line_calibrator,
)

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

    @pytest.mark.parametrize(
        ("bootstrap", "models"),
        [
            (BayesianBootstrap(replicates=12, block_length=2), None),
            (WildBootstrap(replicates=12, block_length=2, multipliers="mammen"), None),
            (
                ParametricBootstrap(replicates=12),
                {"B": NormalObservations(sd=1.0)},
            ),
        ],
        ids=["bayesian", "wild", "parametric"],
    )
    def test_every_bootstrap_calibrates_a_threshold(
        self,
        bootstrap: BayesianBootstrap | WildBootstrap | ParametricBootstrap,
        models: dict[str, NormalObservations] | None,
    ) -> None:
        calibrator = line_calibrator(VALUES, observation_models=models)

        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c"],
                threshold=Threshold.bootstrap_calibrated(bootstrap, 0.8),
                max_steps=3,
            )
        )

        rises = result.profiles["c"].replicate_rises
        assert rises is not None and len(rises) == 12
        assert min(rises) >= 0.0
        expected = commol_rs.inference.weighted_quantiles(rises, [1.0] * 12, [0.8])
        assert result.profiles["c"].threshold == pytest.approx(expected[0])

    def test_a_replicate_that_fails_is_reported(self) -> None:
        # Small values with residuals of their size: the additive bootstrap
        # makes a value below 0, which a log-normal model cannot read.
        values = [0.3, 3.2, 0.2, 6.1, 4.0, 9.5, 6.2, 12.0, 9.0, 15.5]
        calibrator = line_calibrator(
            values,
            observation_models={"B": LogNormalObservations(sigma=0.5, center="median")},
        )
        bootstrap = ResidualBootstrap(
            replicates=20, block_length=1, residual_scale="additive", scheme="moving"
        )

        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c"],
                threshold=Threshold.bootstrap_calibrated(bootstrap, 0.9),
                max_steps=3,
            )
        )

        failed = result.failed
        assert failed and all("needs a value above 0" in reason for reason in failed)
        rises = result.profiles["c"].replicate_rises
        assert rises is not None and len(rises) == 20 - len(failed)

    def test_a_bootstrap_goes_only_with_its_threshold(self) -> None:
        bootstrap = BayesianBootstrap(replicates=5, block_length=1)

        with pytest.raises(ValidationError, match="no other"):
            Threshold(kind="beale", confidence_level=0.9, bootstrap=bootstrap)
        with pytest.raises(ValidationError, match="no other"):
            Threshold(kind="bootstrap_calibrated", confidence_level=0.9)


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


class TestResult:
    def test_profile_result_has_one_entry_per_parameter(self) -> None:
        calibrator = line_calibrator(
            VALUES, observation_models={"B": NormalObservations(sd=1.0)}
        )

        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["B", "c"],
                threshold=Threshold.chi_square(0.95),
                max_steps=50,
            )
        )

        assert list(result.profiles) == ["B", "c"]
        assert list(result.intervals) == ["B", "c"]
        assert result.verdicts == {"B": "identifiable", "c": "identifiable"}
        assert result.quantity == "parameter"
        for name, profile in result.profiles.items():
            assert profile.values == sorted(profile.values)
            fit = profile.values.index(result.fit.best_parameters[name])
            assert profile.statistics[fit] == 0.0
            assert profile.parameters[fit] == result.fit.best_parameters
            interval = result.intervals[name]
            assert interval.lower is not None and interval.upper is not None
            assert interval.lower < profile.center < interval.upper
            assert all(
                (s > profile.threshold) == (not interval.lower <= v <= interval.upper)
                for v, s in zip(profile.values, profile.statistics, strict=True)
            )


class TestSettings:
    def test_a_parameter_is_profiled_once(self) -> None:
        with pytest.raises(ValidationError, match="listed twice"):
            ProfileLikelihood(
                parameters=["c", "c"], threshold=Threshold.chi_square(0.9)
            )

    def test_each_prediction_has_a_label_of_its_own(self) -> None:
        with pytest.raises(ValidationError, match="label of its own"):
            PredictionProfile(
                specs=[
                    PredictionSpec(at=4, output="B"),
                    PredictionSpec(at=8, output="B"),
                ],
                threshold=Threshold.chi_square(0.9),
            )

    def test_the_bindings_refuse_a_bootstrap_they_cannot_build(self) -> None:
        settings = commol_rs.inference.BootstrapSettings
        with pytest.raises(ValueError, match="Unknown bootstrap 'jackknife'"):
            settings("jackknife", 10, 1)
        with pytest.raises(ValueError, match="A residual bootstrap needs scheme"):
            settings("residual", 10, 1, 1, residual_scale="additive")
        with pytest.raises(ValueError, match="bogus"):
            settings("residual", 10, 1, 1, residual_scale="bogus", scheme="moving")
        calibrator = line_calibrator(VALUES)
        with pytest.raises(ValueError, match="only when it is 'bootstrap'"):
            commol_rs.inference.trace_profiles(
                calibrator.rust_problem,
                calibrator.rust_optimizer,
                [2.0, 5.0],
                [0],
                [],
                "beale",
                0.9,
                5,
                1,
                bootstrap=settings("bayesian", 10, 1, 1),
            )


class TestUnreachable:
    def test_a_prediction_no_parameters_give_ends_the_profile(self) -> None:
        # With a noise of sd 1000 the data reject no value of B at 1, which
        # the bounds hold in [0, 100]: each side ends where no values give it.
        calibrator = line_calibrator(
            VALUES, observation_models={"B": NormalObservations(sd=1000.0)}
        )
        problem = calibrator.problem.model_copy(
            update={
                "optimization_config": NelderMeadConfig(
                    max_iterations=300, sd_tolerance=1e-8
                )
            }
        )

        result = Calibrator(calibrator.simulation, problem).uncertainty(
            PredictionProfile(
                specs=[PredictionSpec(at=1, output="B")],
                threshold=Threshold.chi_square(0.9),
            )
        )

        profile = result.profiles["B"]
        assert profile.ends == ("unreachable", "unreachable")
        assert 0.0 <= min(profile.values) and max(profile.values) <= 100.0
        assert result.intervals["B"] == ProfileInterval(lower=None, upper=None)
        assert result.verdicts["B"] == "practically_non_identifiable"


def test_a_threshold_takes_a_confidence_level() -> None:
    calibrator = line_calibrator(
        VALUES, observation_models={"B": NormalObservations(sd=1.0)}
    )
    threshold = Threshold.chi_square(confidence_level=0.9)

    result = calibrator.uncertainty(
        ProfileLikelihood(parameters=["c"], threshold=threshold, max_steps=100)
    )

    z = NormalDist().inv_cdf(0.95)
    assert result.profiles["c"].threshold == pytest.approx(z**2 / 2)


def test_the_identifiability_of_a_profile_is_a_report() -> None:
    calibrator = line_calibrator(
        VALUES, observation_models={"B": NormalObservations(sd=1.0)}
    )
    result = calibrator.uncertainty(
        ProfileLikelihood(
            parameters=["c"], threshold=Threshold.chi_square(0.9), max_steps=100
        )
    )

    report = result.identifiability()

    assert isinstance(report, IdentifiabilityReport)
    assert [section.confidence_level for section in report.profile] == [0.9]
    data = report.report()
    assert data["parameters"]["c"]["practical"] == {
        "verdict": "determined",
        "reasons": [
            "profile likelihood: identifiable. The verdict holds with the other "
            "parameters at their fitted values."
        ],
    }
    assert data["parameters"]["B"] == {"structural": None, "practical": None}


def test_a_prediction_profile_has_no_identifiability() -> None:
    calibrator = line_calibrator(
        VALUES, observation_models={"B": NormalObservations(sd=1.0)}
    )
    result = calibrator.uncertainty(
        PredictionProfile(
            specs=[PredictionSpec(at=4, output="B")],
            threshold=Threshold.chi_square(0.9),
            max_steps=2,
        )
    )

    assert result.structure == {}
    with pytest.raises(ValueError, match="predictions have no verdict on a parameter"):
        result.identifiability()
