"""Coverage study of a negative binomial series of sums of parts.

Each data set is a decay A → B from 2000 in A at the rate ``k``
(``a_to_b_builder``), whose flow into B is counted step by step: the first
20 steps are a series observed part by part (``"daily"``), and the next 20
windows of ``K`` steps are a series of totals (``"totals"``), each the sum of
its ``K`` steps. The counts of the steps are independent negative binomials of
one dispersion ``φ``, which the two series share. Over a window the flow
changes by at most 14 %, so its effective count of parts,
``(Σμₜ)²/Σμₜ²``, is ``K`` within 1 %.

The truth is drawn from the prior, ``k`` uniform on ``[0.004, 0.012]`` and
``φ`` uniform on ``[φ₀/2, 3φ₀/2]``, so the 90 % intervals of the posterior
cover it in 90 % of the data sets when the likelihood is the model of the
data: the study accepts ``sum="exact"`` within the tolerance of the harness,
reports ``"moments"`` beside it, and checks that the totals read without
``parts`` (a negative binomial of the window's mean and ``φ``) under-cover
``φ``. A study of a truth drawn
from the prior holds the truth by shifting each interval by it, so the truth
of the harness is 0. The studies carry the marker ``coverage_study``.
"""

import logging
from typing import Literal

import numpy as np
import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.distributions import Priors
from commol.observations import NegativeBinomialObservations
from commol.smc import BayesianPosterior
from tests.support import a_to_b_builder, bounded, build
from tests.support.coverage import CoverageStudy, Interval, coverage_study

AMOUNT, DAILY, TOTALS, LEVEL = 2000.0, 20, 20, 0.9
RATE = (0.004, 0.012)

Reading = Literal["exact", "moments", "none"]
"""How the totals are read: a sum of parts, read exactly or by its moments,
or one negative binomial of the window (``"none"``, no ``parts``)."""

logger = logging.getLogger(__name__)


def _observations(
    counts: np.ndarray, parts: int, reading: Reading
) -> list[ObservedDataPoint]:
    """The daily counts of the first steps, then the totals of `parts` steps."""
    daily = [
        ObservedDataPoint(
            at=t, output="B", label="daily", value=float(counts[t - 1]), window=1
        )
        for t in range(1, DAILY + 1)
    ]
    ends = [DAILY + parts * (index + 1) for index in range(TOTALS)]
    totals = [
        ObservedDataPoint(
            at=end,
            output="B",
            label="totals",
            value=float(counts[end - parts : end].sum()),
            window=parts,
            parts=1 if reading == "none" else parts,
        )
        for end in ends
    ]
    return daily + totals


def _calibrator(
    counts: np.ndarray, parts: int, phi: float, reading: Reading, seed: int
) -> Calibrator:
    model = build(a_to_b_builder(parameters={"k": None}, amounts={"A": AMOUNT, "B": 0}))
    problem = CalibrationProblem(
        observed_data=_observations(counts, parts, reading),
        parameters=[
            bounded("k", *RATE),
            bounded("phi", phi / 2, 3 * phi / 2, kind="noise"),
        ],
        observation_models={
            "daily": NegativeBinomialObservations(overdispersion="phi"),
            "totals": NegativeBinomialObservations(
                overdispersion="phi", sum=None if reading == "none" else reading
            ),
        },
        optimization_config=NelderMeadConfig(max_iterations=500),
        seed=seed,
    )
    return Calibrator(Simulation(model), problem)


def _study(parts: int, phi: float, reading: Reading) -> CoverageStudy:
    method = BayesianPosterior(
        priors=Priors.uniform_on_bounds(),
        particles=400,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.05,
    )

    def run(index: int, rng: np.random.Generator) -> dict[str, Interval]:
        truth = {
            "k": rng.uniform(*RATE),
            "phi": rng.uniform(phi / 2, 3 * phi / 2),
        }
        steps = np.arange(DAILY + TOTALS * parts)
        means = AMOUNT * truth["k"] * (1 - truth["k"]) ** steps
        size = 1 / truth["phi"]
        counts = rng.negative_binomial(size, size / (size + means))
        sample = _calibrator(counts, parts, phi, reading, index).uncertainty(method)
        intervals = sample.sample.interval(LEVEL)
        return {
            name: (lower - truth[name], upper - truth[name])
            for name, (lower, upper) in intervals.items()
        }

    # The same seed for every reading: each reads the same data sets.
    return coverage_study(run, {"k": 0.0, "phi": 0.0}, LEVEL, data_sets=200, seed=7)


@pytest.mark.coverage_study
class TestPartSums:
    @pytest.mark.parametrize("phi", [0.3, 1.0])
    @pytest.mark.parametrize("parts", [5, 7, 12])
    def test_an_exact_sum_is_calibrated(self, parts: int, phi: float) -> None:
        studies = {
            reading: _study(parts, phi, reading)
            for reading in ("exact", "moments", "none")
        }
        report = ", ".join(
            f"{reading}: "
            + " ".join(
                f"{name} {study.coverage(name):.3f}" for name in sorted(study.hits)
            )
            for reading, study in studies.items()
        )
        logger.info("parts=%d, phi=%s: %s", parts, phi, report)

        assert studies["exact"].calibrated(), report
        # Read as one negative binomial of the window, a total has the
        # variance of one part of its mean: the intervals of φ miss it.
        assert studies["none"].coverage("phi") < LEVEL - studies["none"].tolerance, (
            report
        )
