"""A fit near a bound: `Laplace`, `Sandwich` and `LaplaceMixture` note a
bound less than 3 sd from the fit, on the scale of the parameters, and refuse
a fit on it; information matching gives `η` near a bound and on it.

Each test fits a constant flow ``c`` (``flow_calibrator``) to 24 windowed
values of mean 10 with a normal model of sd 1: the fit is 10 and the sd of
``c`` is ``1/√24``, about 0.2.
"""

import logging
import math
import re
import statistics

import numpy as np
import pytest

from commol import Calibrator
from commol.distributions import Priors
from commol.local import Laplace, LaplaceMixture, LikelihoodTarget, Sandwich
from commol.local.results import LocalResult
from commol.observations import NormalObservations
from commol.smc import GibbsPosterior, InformationMatchingEta
from tests.support import flow_calibrator

_VALUES = 10 + np.random.default_rng(3).normal(size=24)
_VALUES = (_VALUES - _VALUES.mean() + 10).tolist()
_SD = 1 / math.sqrt(24)
_NOTE = re.compile(r"'c' is (\S+) sd from its bound ([^\s,]+)")


def _calibrator(bounds: tuple[float, float]) -> Calibrator:
    return flow_calibrator(
        _VALUES,
        windowed=True,
        bounds=bounds,
        observation_models={"B": NormalObservations(sd=1.0)},
    )


def _noted(notes: list[str]) -> list[tuple[float, float]]:
    """The bound and the distance in sd of each note of a bound near the
    fit."""
    return [
        (float(match[2]), float(match[1]))
        for note in notes
        if (match := _NOTE.search(note))
    ]


def _distance(result: LocalResult, bound: tuple[float, float]) -> float:
    """The distance from the fit to the upper bound of `bound`, on the scale
    of the parameters, in units of the scale that the covariance of
    `result` gives there."""
    lower, upper = bound
    fit = result.fit.best_parameters["c"]
    derivative = (fit - lower) * (upper - fit) / (upper - lower)
    return (upper - fit) / (math.sqrt(result.covariance[0][0]) * derivative)


class TestLaplace:
    def test_a_fit_near_a_bound_is_noted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The upper bound is 0.49 sd from the fit.
        with caplog.at_level(logging.WARNING):
            result = _calibrator((0.0, 10.1)).uncertainty(Laplace(draws=200))

        [(bound, distance)] = _noted(result.notes)
        assert bound == 10.1
        assert distance == pytest.approx(0.1 / _SD, rel=0.01)
        assert _noted([caplog.text]) == [(bound, distance)]
        assert "ProfileLikelihood" in result.notes[0]

    @pytest.mark.parametrize(
        ("bounds", "expected"),
        [
            ((10 - 2 * _SD, 50.0), [(10 - 2 * _SD, 2.0)]),
            ((10 - 2 * _SD, 10 + _SD), [(10 - 2 * _SD, 2.0), (10 + _SD, 1.0)]),
            ((10 - 3.1 * _SD, 10 + 3.1 * _SD), []),
            ((0.0, 50.0), []),
        ],
        ids=["lower", "both", "from 3 sd", "far"],
    )
    def test_each_bound_within_3_sd_is_noted(
        self,
        bounds: tuple[float, float],
        expected: list[tuple[float, float]],
    ) -> None:
        result = _calibrator(bounds).uncertainty(Laplace(draws=200))

        noted = _noted(result.notes)
        assert [bound for bound, _ in noted] == [bound for bound, _ in expected]
        for (_, distance), (_, sd) in zip(noted, expected, strict=True):
            assert distance == pytest.approx(sd, rel=0.01)

    @pytest.mark.parametrize(
        "method",
        [Laplace(draws=200), Sandwich(block_length=3, draws=200)],
        ids=["laplace", "sandwich"],
    )
    def test_a_fit_on_its_bound_is_still_refused(
        self, method: Laplace | Sandwich
    ) -> None:
        calibrator = _calibrator((0.0, 10 * (1 + 1e-9)))

        with pytest.raises(ValueError, match="'c' is on its bound"):
            calibrator.uncertainty(method)


class TestSandwich:
    def test_a_sandwich_notes_the_mass_of_its_student_t_past_the_bound(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Two blocks of 12 give the t one degree of freedom: a bound 16 units
        # of its scale from the fit has the mass past it of a normal at 2 sd.
        bounds = (0.0, 11.0)
        with caplog.at_level(logging.WARNING):
            result = _calibrator(bounds).uncertainty(
                Sandwich(block_length=12, draws=200)
            )

        assert result.degrees_of_freedom["c"] == pytest.approx(1.0)
        scaled = _distance(result, bounds)
        # The upper tail of t₁ is ½ − atan(d)/π.
        z = statistics.NormalDist().inv_cdf(0.5 + math.atan(scaled) / math.pi)
        # The lower bound, 10 below the fit, is noted too: the t₁ puts 0.2 %
        # past it.
        noted = dict(_noted(result.notes))
        assert sorted(noted) == [0.0, 11.0]
        assert scaled > 3
        assert noted[11.0] == pytest.approx(z, abs=0.006)
        assert "Student t of 1.0 degrees of freedom" in result.notes[1]
        logged = {float(b): float(d) for d, b in _NOTE.findall(caplog.text)}
        assert logged == noted


class TestLaplaceMixture:
    def test_a_component_near_a_bound_is_noted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        method = LaplaceMixture(
            draws=200, target=LikelihoodTarget(priors=Priors.uniform_on_bounds())
        )
        with caplog.at_level(logging.WARNING):
            result = _calibrator((0.0, 10.1)).uncertainty(method)

        [(bound, distance)] = _noted(result.notes)
        assert bound == 10.1
        assert distance == pytest.approx(0.1 / _SD, rel=0.01)
        assert result.notes[0].startswith("At the end [")
        assert "the importance weights correct it" in result.notes[0]
        assert "ProfileLikelihood" not in result.notes[0]
        assert _noted([caplog.text]) == [(bound, distance)]


class TestInformationMatching:
    def test_information_matching_gives_eta_near_and_on_a_bound(self) -> None:
        # The curvature is read on the scale of the parameters, where a bound
        # changes nothing: `η` is that of a bound far from the fit, with no
        # note of the bound.
        method = GibbsPosterior(
            priors=Priors.uniform_on_bounds(),
            eta=InformationMatchingEta(block_length=3),
            particles=50,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.05,
        )
        far = _calibrator((0.0, 20.0)).uncertainty(method)

        for gap in (1e-3, 1e-9):
            result = _calibrator((0.0, 10 * (1 + gap))).uncertainty(method)

            assert result.eta == pytest.approx(far.eta, rel=1e-6)
            assert result.degrees_of_freedom == pytest.approx(
                far.degrees_of_freedom, rel=1e-6
            )
            assert not [n for n in result.diagnostics.notes if "bound" in n]
