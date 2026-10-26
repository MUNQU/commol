"""The priors of the sequential Monte Carlo methods: one per calibrated
parameter, on the bounds of its parameter, with the parameters each
distribution checks."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    Simulation,
)
from commol.commol_rs import _commol_rs as commol_rs
from commol.distributions import Beta, LogNormal, LogUniform, Normal, Priors, Uniform
from commol.smc import ABCSMC, GibbsPosterior, LossDistance
from tests.support import a_to_b_builder, build, flow_calibrator, flow_observations

DATA = [9.0, 11.0, 10.5, 8.5, 10.0]
"""Flows near 10: these tests read the priors, not the fit."""


def _gibbs(**settings: object) -> GibbsPosterior:
    defaults: dict[str, object] = {
        "priors": Priors.uniform_on_bounds(),
        "eta": 1.0,
        "particles": 50,
        "target_cess": 0.5,
        "resample_threshold": 0.5,
        "stay_probability": 0.05,
    }
    return GibbsPosterior.model_validate(defaults | settings)


def _unbounded_calibrator(low: float, high: float) -> Calibrator:
    """The flow model with the bounds ``[low, high]`` for ``c``."""
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
        )
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(DATA, windowed=True),
        parameters=[
            CalibrationParameter(
                id="c",
                parameter_type="parameter",
                min_bound=low,
                max_bound=high,
                initial_guess=10.0,
            )
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=200),
        seed=1,
    )
    return Calibrator(Simulation(model), problem)


class TestPriors:
    def test_priors_are_required_for_every_parameter(self) -> None:
        settings = _gibbs().model_dump(exclude={"priors"})
        with pytest.raises(ValueError, match=r"priors.*Priors\.uniform_on_bounds\(\)"):
            GibbsPosterior.model_validate(settings)
        with pytest.raises(ValueError, match=r"priors.*Priors\.uniform_on_bounds\(\)"):
            ABCSMC.model_validate(
                {
                    "distance": LossDistance(),
                    "particles": 100,
                    "target_cess": 0.5,
                    "min_acceptance": 0.05,
                    "stay_probability": 0.05,
                }
            )

        calibrator = flow_calibrator(DATA, windowed=True)
        with pytest.raises(
            ValueError, match=r"'c' has no prior.*Priors\.uniform_on_bounds\(\)"
        ):
            calibrator.uncertainty(_gibbs(priors={}))
        with pytest.raises(ValueError, match=r"'k' is not a calibrated parameter"):
            calibrator.uncertainty(_gibbs(priors={"c": Uniform(), "k": Uniform()}))

    def test_a_prior_on_the_bounds_needs_finite_bounds(self) -> None:
        open_above = _unbounded_calibrator(0.0, math.inf)
        for prior in (Uniform(), Beta(a=2.0, b=2.0), LogUniform()):
            with pytest.raises(
                ValueError,
                match=(
                    rf"{type(prior).__name__}\(\) of 'c' needs finite bounds.*"
                    r"\[0\.0, inf\].*Normal, LogNormal, Gamma"
                ),
            ):
                open_above.uncertainty(_gibbs(priors={"c": prior}))

        from_zero = _unbounded_calibrator(0.0, 50.0)
        with pytest.raises(
            ValueError,
            match=(
                r"LogUniform\(\) of 'c' needs a lower bound above 0.*"
                r"\[0\.0, 50\.0\].*Uniform, Beta, Normal, LogNormal, Gamma"
            ),
        ):
            from_zero.uncertainty(_gibbs(priors={"c": LogUniform()}))

    def test_a_prior_takes_the_bounds_of_its_parameter(self) -> None:
        bounds = (0.0, 50.0)
        half = Normal(mean=0.0, sd=1.0)
        assert half.cdf(1.0, (0.0, math.inf)) == pytest.approx(
            2 * 0.8413447460685429 - 1
        )
        assert Uniform().quantile(0.25, bounds) == pytest.approx(12.5)
        assert Uniform().log_density(60.0, bounds) == -math.inf
        draws = Beta(a=2.0, b=5.0).sample(500, bounds, seed=3)
        assert len(draws) == 500
        assert all(0.0 <= draw <= 50.0 for draw in draws)
        assert draws == Beta(a=2.0, b=5.0).sample(500, bounds, seed=3)
        assert np.mean(draws) == pytest.approx(50.0 * 2 / 7, rel=0.05)

    def test_every_distribution_reaches_rust(self) -> None:
        log_normal = LogNormal(mu=0.0, sigma=1.0)
        assert log_normal.cdf(1.0, (0.0, math.inf)) == pytest.approx(0.5)
        with pytest.raises(ValueError, match="between 0 and 1, got 1.5"):
            log_normal.quantile(1.5, (0.0, math.inf))
        with pytest.raises(ValueError, match="kind: unknown variant `cauchy`"):
            commol_rs.inference.Distribution('{"kind": "cauchy"}', 0.0, 1.0)
        with pytest.raises(ValidationError):
            GibbsPosterior.model_validate("not a configuration")

    def test_a_distribution_checks_its_parameters(self) -> None:
        with pytest.raises(ValidationError, match="sd"):
            Normal(mean=0.0, sd=0.0)
        with pytest.raises(ValidationError, match="a"):
            Beta(a=-1.0, b=1.0)
