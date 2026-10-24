"""The resampling methods through ``Calibrator.uncertainty``: their settings,
their results and the residual noise of a predictive.

The model is linear in its one parameter (``flow_calibrator``): each
observation is either ``c`` (windowed), so the fit is the mean of the values,
or ``c·t``, a regression through the origin.
"""

import logging
import math
import re
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    BayesianBootstrap,
    BlockJackknife,
    BootstrapCoverageEta,
    BootstrapResult,
    Calibrator,
    ConformalResult,
    GibbsPosterior,
    GibbsResult,
    JackknifePlus,
    JackknifeResult,
    LogNormalObservations,
    NoNoise,
    ObservedDataPoint,
    Predictive,
    Priors,
    ProfileLikelihood,
    ProfileResult,
    ResidualBootstrap,
    ResidualNoise,
    RollingConformal,
    Threshold,
    WildBootstrap,
)
from commol.api.inference import smc as smc_runners
from commol.commol_rs import _commol_rs as commol_rs
from tests.support import flow_calibrator, flow_observations, line_calibrator


def _noisy(count: int, seed: int, level: float = 10.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return (level + rng.normal(0.0, 1.0, count)).tolist()


class TestSettings:
    def test_block_length_is_required_and_checked(self) -> None:
        with pytest.raises(ValidationError, match="block_length"):
            BayesianBootstrap(replicates=4)  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="block_length"):
            BayesianBootstrap(replicates=4, block_length=0)
        with pytest.raises(ValidationError, match="block_length"):
            BayesianBootstrap(replicates=4, block_length="long")  # type: ignore[arg-type]

        calibrator = flow_calibrator(_noisy(10, 1), windowed=True)
        with pytest.raises(ValueError, match="block_length is 11.*10"):
            calibrator.uncertainty(BayesianBootstrap(replicates=2, block_length=11))

    def test_modelling_choices_have_no_default(self) -> None:
        with pytest.raises(ValidationError, match="residual_scale"):
            ResidualBootstrap(replicates=2, block_length=1, scheme="moving")  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="scheme"):
            ResidualBootstrap(  # type: ignore[call-arg]
                replicates=2, block_length=1, residual_scale="additive"
            )
        with pytest.raises(ValidationError, match="multipliers"):
            WildBootstrap(replicates=2, block_length=1)  # type: ignore[call-arg]


class TestBayesianBootstrap:
    def test_bayesian_bootstrap_returns_one_draw_per_replicate(self) -> None:
        calibrator = flow_calibrator(_noisy(20, 2), windowed=True, seed=4)

        result = calibrator.uncertainty(BayesianBootstrap(replicates=8, block_length=2))

        assert isinstance(result, BootstrapResult)
        assert len(result.sample) == 8
        assert result.sample.weights == pytest.approx([1 / 8] * 8)
        assert result.sample.groups == list(range(8))
        assert result.sample.method.name == "bayesian_bootstrap"
        assert result.sample.method.settings["seed"] == 4
        assert result.fit == calibrator.run()
        assert result.diagnostics.block_length == 2
        assert result.diagnostics.block_length_estimates is None
        assert result.failed == []

    def test_bayesian_bootstrap_matches_the_sandwich_variance(self) -> None:
        # y = c·t + e: least squares through the origin, whose sandwich
        # variance is Σ t² e² / (Σ t²)².
        t = np.arange(1.0, 41.0)
        rng = np.random.default_rng(7)
        y = 3.0 * t + rng.normal(0.0, 2.0, t.size)
        calibrator = flow_calibrator(y.tolist(), windowed=False, seed=11)
        replicates = 300

        result = calibrator.uncertainty(
            BayesianBootstrap(replicates=replicates, block_length=1)
        )

        estimate = float(t @ y / (t @ t))
        residuals = y - estimate * t
        sandwich = float((t**2) @ (residuals**2) / (t @ t) ** 2)
        variance = result.sample.covariance()[0][0]
        # The sample variance has a relative standard error of about √(2/R).
        assert variance == pytest.approx(sandwich, rel=3 * math.sqrt(2 / replicates))

    def test_automatic_length_reports_each_series(self) -> None:
        calibrator = flow_calibrator(_noisy(60, 3), windowed=True)

        result = calibrator.uncertainty(
            BayesianBootstrap(replicates=3, block_length="auto")
        )

        estimates = result.diagnostics.block_length_estimates
        assert estimates is not None and list(estimates) == ["B"]
        assert result.diagnostics.block_length == max(1, round(estimates["B"]))

    def test_the_monte_carlo_error_brackets_each_bound(self) -> None:
        calibrator = flow_calibrator(_noisy(30, 5), windowed=True)
        result = calibrator.uncertainty(
            BayesianBootstrap(replicates=200, block_length=1)
        )

        interval = result.sample.interval(0.9)["c"]
        error = result.interval_error(0.9)["c"]

        assert error.lower[0] <= interval[0] <= error.lower[1]
        assert error.upper[0] <= interval[1] <= error.upper[1]


class TestResidualBootstraps:
    def test_residuals_of_an_exact_fit_give_one_point(self) -> None:
        calibrator = flow_calibrator([4.0] * 12, windowed=True)

        for method in (
            ResidualBootstrap(
                replicates=5, block_length=3, residual_scale="additive", scheme="moving"
            ),
            WildBootstrap(replicates=5, block_length=3, multipliers="rademacher"),
        ):
            result = calibrator.uncertainty(method)
            assert [row[0] for row in result.sample.values] == pytest.approx(
                [4.0] * 5, abs=1e-7
            )

    def test_multiplicative_scale_needs_positive_values(self) -> None:
        values = _noisy(10, 1)
        values[3] = 0.0
        calibrator = flow_calibrator(values, windowed=True)

        with pytest.raises(ValueError, match=r"'B' at 4.*residual_scale=\"additive\""):
            calibrator.uncertainty(
                ResidualBootstrap(
                    replicates=2,
                    block_length=1,
                    residual_scale="multiplicative",
                    scheme="stationary",
                )
            )

    def test_a_negative_synthetic_value_is_counted(self) -> None:
        # y ≈ c·t with residuals of −6 late in the series: one moved to t = 1
        # gives a value below zero.
        values = [t + (6.0 if t % 2 else -6.0 if t >= 8 else 1.0) for t in range(1, 21)]
        calibrator = flow_calibrator(values, windowed=False, bounds=(0.0, 5.0))

        result = calibrator.uncertainty(
            ResidualBootstrap(
                replicates=10,
                block_length=1,
                residual_scale="additive",
                scheme="moving",
            )
        )

        assert result.diagnostics.negative_values > 0
        assert len(result.sample) + len(result.failed) == 10

    def test_one_seed_gives_one_result(self) -> None:
        values = _noisy(24, 9)
        for method in (
            BayesianBootstrap(replicates=6, block_length=2, starts=2),
            ResidualBootstrap(
                replicates=6,
                block_length=2,
                residual_scale="additive",
                scheme="stationary",
            ),
            WildBootstrap(replicates=6, block_length=2, multipliers="mammen"),
        ):
            first = flow_calibrator(values, windowed=True, seed=8).uncertainty(method)
            again = flow_calibrator(values, windowed=True, seed=8).uncertainty(method)
            other = flow_calibrator(values, windowed=True, seed=9).uncertainty(method)
            assert first == again
            assert first.sample != other.sample
            assert first.sample.method.settings["seed"] == 8


class TestBlockJackknife:
    def test_jackknife_of_a_mean_is_the_classical_standard_error(self) -> None:
        values = _noisy(15, 6)
        calibrator = flow_calibrator(values, windowed=True)

        result = calibrator.uncertainty(BlockJackknife(block_length=1))

        classical = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        assert result.standard_error["c"] == pytest.approx(classical, rel=1e-5)
        assert result.bias["c"] == pytest.approx(0.0, abs=1e-6)
        assert result.estimate["c"] == pytest.approx(np.mean(values), rel=1e-6)
        # The pseudo-values of a mean are the values.
        assert result.pseudo_values["c"] == pytest.approx(values, rel=1e-5)
        assert result.blocks == [(t, t) for t in range(1, 16)]

    def test_jackknife_influence_finds_the_outlier_block(self) -> None:
        values = _noisy(12, 2)
        values[7] += 15.0
        calibrator = flow_calibrator(values, windowed=True)

        result = calibrator.uncertainty(BlockJackknife(block_length=2))

        influence = [abs(v) for v in result.influence["c"]]
        assert influence.index(max(influence)) == 3
        assert result.blocks[3] == (7.0, 8.0)

    def test_jackknife_needs_two_blocks(self) -> None:
        calibrator = flow_calibrator(_noisy(4, 1), windowed=True)

        with pytest.raises(ValueError, match="block_length 4 makes 1 block"):
            calibrator.uncertainty(BlockJackknife(block_length=4))


class TestResidualNoise:
    def test_residual_noise_needs_residuals_of_the_series(self) -> None:
        calibrator = flow_calibrator(_noisy(10, 1), windowed=True)
        result = calibrator.uncertainty(BayesianBootstrap(replicates=3, block_length=1))
        predictive = Predictive(calibrator, result.sample, [1.0, 2.0])
        noise = ResidualNoise(fit=result.fit, block_length=2, residual_scale="additive")

        with pytest.raises(ValueError, match="no residuals of the series 'A'"):
            predictive.at_observations(
                [ObservedDataPoint(at=1, output="A", value=0.0)], noise
            )

    def test_residual_noise_widens_the_band_of_held_out_observations(self) -> None:
        values = _noisy(30, 4)
        calibrator = flow_calibrator(values, windowed=True)
        result = calibrator.uncertainty(
            BayesianBootstrap(replicates=50, block_length=1)
        )
        predictive = Predictive(calibrator, result.sample, [1.0])
        held_out = flow_observations([10.0] * 5, windowed=True, start=31)
        noise = ResidualNoise(fit=result.fit, block_length=1, residual_scale="additive")

        model = predictive.at_observations(held_out, NoNoise()).band(0.9)
        prediction = predictive.at_observations(held_out, noise).band(0.9)

        for i in range(len(held_out)):
            assert prediction.upper[i] - prediction.lower[i] > 2 * (
                model.upper[i] - model.lower[i]
            )


# Small values with residuals of their size: the additive bootstrap makes a
# value below 0, which a log-normal model cannot read.
_SMALL = [0.3, 3.2, 0.2, 6.1, 4.0, 9.5, 6.2, 12.0, 9.0, 15.5]
_BELOW_ZERO = ResidualBootstrap(
    replicates=20, block_length=1, residual_scale="additive", scheme="moving"
)
# A fit of a block or an origin fails only with a model that fails, which
# fails the fit of every observation too: the extension adds one failed fit.
_NO_WEIGHT = "Every weight is zero"


class _OneMoreFailed:
    """The run of the extension `run`, with one more failed fit, `entry`."""

    def __init__(self, run: object, entry: str) -> None:
        self._run = run
        self.failed = [*getattr(run, "failed"), entry]

    def __getattr__(self, name: str) -> object:
        return getattr(self._run, name)


class _ProblemWithAFailedFit:
    """The problem of the extension `problem`, whose `name` reports one more
    failed fit, `entry`."""

    def __init__(self, problem: object, name: str, entry: str) -> None:
        self._problem = problem
        self._name = name
        self._entry = entry

    def __getattr__(self, name: str) -> object:
        found = getattr(self._problem, name)
        if name != self._name:
            return found
        return lambda *arguments: _OneMoreFailed(found(*arguments), self._entry)


def _fails_one_fit(monkeypatch: pytest.MonkeyPatch, name: str, entry: str) -> None:
    """Make the function `name` of ``commol_rs.inference`` report one more
    failed fit, `entry`."""
    found = getattr(commol_rs.inference, name)
    monkeypatch.setattr(
        commol_rs.inference,
        name,
        lambda *arguments: _OneMoreFailed(found(*arguments), entry),
    )


type _Result = (
    BootstrapResult | JackknifeResult | ProfileResult | ConformalResult | GibbsResult
)


def _log_normal() -> Calibrator:
    return line_calibrator(
        _SMALL,
        observation_models={"B": LogNormalObservations(sigma=0.5, center="median")},
    )


def _bootstrap(_: pytest.MonkeyPatch) -> _Result:
    return _log_normal().uncertainty(_BELOW_ZERO)


def _jackknife(_: pytest.MonkeyPatch) -> _Result:
    calibrator = flow_calibrator(_noisy(9, 2), windowed=True)
    calibrator._rust_problem = _ProblemWithAFailedFit(
        calibrator.rust_problem, "jackknife", f"Block 1 failed: {_NO_WEIGHT}"
    )
    return calibrator.uncertainty(BlockJackknife(block_length=3))


def _profile(_: pytest.MonkeyPatch) -> _Result:
    return _log_normal().uncertainty(
        ProfileLikelihood(
            parameters=["c"],
            threshold=Threshold.bootstrap_calibrated(_BELOW_ZERO, 0.9),
            max_steps=3,
        )
    )


def _jackknife_plus(monkeypatch: pytest.MonkeyPatch) -> _Result:
    _fails_one_fit(monkeypatch, "jackknife_plus", f"Block 1 failed: {_NO_WEIGHT}")
    calibrator = flow_calibrator(_noisy(9, 2), windowed=True)
    return calibrator.uncertainty(
        JackknifePlus(
            block_length=3,
            score="absolute",
            by="series",
            confidence_level=0.5,
            positions=[10.0],
        )
    )


def _rolling(monkeypatch: pytest.MonkeyPatch) -> _Result:
    _fails_one_fit(monkeypatch, "rolling_conformal", f"Origin 1 failed: {_NO_WEIGHT}")
    calibrator = flow_calibrator(_noisy(9, 2), windowed=True)
    return calibrator.uncertainty(
        RollingConformal(
            origins=[2.0, 5.0],
            score="absolute",
            by="series",
            confidence_level=0.5,
            positions=[10.0],
        )
    )


def _gibbs(monkeypatch: pytest.MonkeyPatch) -> _Result:
    # A sampler run of a bootstrap replicate fails only with a model that
    # fails, which would fail the posterior too: the extension reports it.
    found = SimpleNamespace(
        eta=0.1, coverage=0.8, failed=[f"Replicate 2 failed: {_NO_WEIGHT}"]
    )
    monkeypatch.setattr(
        smc_runners,
        "rust",
        SimpleNamespace(
            bootstrap_coverage_eta=lambda *_: found,
            gibbs_posterior=commol_rs.inference.gibbs_posterior,
        ),
    )
    return flow_calibrator(_noisy(10, 3), windowed=True).uncertainty(
        GibbsPosterior(
            priors=Priors.uniform_on_bounds(),
            eta=BootstrapCoverageEta(
                replicates=3, block_length="auto", confidence_level=0.5
            ),
            particles=50,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.05,
        )
    )


@pytest.mark.parametrize(
    ("run", "unit", "reason"),
    [
        (_bootstrap, "Replicate", "needs a value above 0"),
        (_jackknife, "Block", _NO_WEIGHT),
        (_profile, "Replicate", "needs a value above 0"),
        (_jackknife_plus, "Block", _NO_WEIGHT),
        (_rolling, "Origin", _NO_WEIGHT),
        (_gibbs, "Replicate", _NO_WEIGHT),
    ],
    ids=["bootstrap", "jackknife", "profile", "jackknife_plus", "rolling", "gibbs"],
)
def test_every_method_lists_its_failed_replicates_in_failed(
    run: Callable[[pytest.MonkeyPatch], _Result],
    unit: str,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="commol.api.inference"):
        result = run(monkeypatch)

    failed = result.failed
    assert failed
    for entry in failed:
        assert re.fullmatch(rf"{unit} \d+ failed: .*{reason}.*", entry), entry
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == failed
    notes = (
        result.notes
        if isinstance(result, ProfileResult)
        else result.diagnostics.notes
        if isinstance(result, GibbsResult)
        else []
    )
    assert not any(entry in note for entry in failed for note in notes)
