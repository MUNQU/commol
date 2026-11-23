"""The calibration of `η` by information matching: its value for correct and
for dependent observations, with few blocks and with blocks that share the
curvature unevenly, and its degrees of freedom."""

import logging
import math
from typing import Literal

import numpy as np
import pytest

from commol import ObservedDataPoint
from commol.distributions import Priors
from commol.observations import NegativeBinomialObservations, NormalObservations
from commol.smc import GibbsPosterior, GibbsResult, InformationMatchingEta
from tests.support import (
    bounded,
    decay_part_means,
    fan_calibrator,
    flow_calibrator,
    line_calibrator,
    sum_calibrator,
    normal_values,
)


def _matching(block_length: int | Literal["auto"]) -> GibbsPosterior:
    return GibbsPosterior(
        priors=Priors.uniform_on_bounds(),
        eta=InformationMatchingEta(block_length=block_length),
        particles=50,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.05,
    )


class TestInformationMatching:
    def test_information_matching_of_a_correct_likelihood_is_one(self) -> None:
        count = 200
        calibrator = flow_calibrator(
            normal_values(count, 8, sd=2.0),
            windowed=True,
            observation_models={"B": NormalObservations(sd=2.0)},
        )

        result = calibrator.uncertainty(_matching(block_length=1))

        # η = n sd² / Σ r²: its relative error is about √(2/n).
        assert result.eta == pytest.approx(1.0, rel=3 * math.sqrt(2 / count))

    def test_a_correct_model_with_sums_gives_one(self) -> None:
        # A decay A → B whose windows of two steps are each the sum of two
        # negative binomial parts of dispersion φ: the exact sum is the
        # correct model of the data, of the rate and of φ.
        k, phi, count, parts = 0.01, 0.5, 150, 2
        ends = [parts * (index + 1) for index in range(count)]
        means = np.array(decay_part_means(k, ends, parts))
        size = 1 / phi
        values = np.random.default_rng(21).negative_binomial(
            size, size / (size + means)
        )
        observations = [
            ObservedDataPoint(
                at=end, output="B", value=float(value), window=parts, parts=parts
            )
            for end, value in zip(ends, values.sum(axis=1), strict=True)
        ]
        calibrator = sum_calibrator(
            observations,
            {"B": NegativeBinomialObservations(overdispersion="phi", sum="exact")},
            noise=[bounded("phi", 0.01, 3.0, kind="noise")],
        )

        result = calibrator.uncertainty(_matching(block_length=1))

        # Across seeds of the data, η spreads by about 0.1 around one.
        assert result.eta == pytest.approx(1.0, rel=0.25)

    def test_information_matching_takes_a_likelihood(self) -> None:
        # The noise parameter sd adds its own information, 2n / sd², and the
        # variance of its scores.
        count = 200
        calibrator = flow_calibrator(
            normal_values(count, 9, sd=2.0),
            windowed=True,
            observation_models={"B": NormalObservations(sd="sd")},
            noise=[bounded("sd", 0.1, 10.0, kind="noise")],
        )

        result = calibrator.uncertainty(_matching(block_length=4))

        assert result.fit is not None
        assert result.eta == pytest.approx(1.0, rel=6 * math.sqrt(2 / count))

    def test_few_blocks_per_parameter_give_eta_near_its_value(self) -> None:
        # 10 constant flows at 35 positions with noise of sd 1: the sum of
        # squares has η = 1/(2σ²) = 0.5. Blocks of 3 and 7 positions make 12
        # and 5 blocks for 10 parameters.
        rng = np.random.default_rng(4)
        calibrator = fan_calibrator((5.0 + rng.normal(size=(10, 35))).tolist())

        for block_length in (3, 7):
            result = calibrator.uncertainty(_matching(block_length))

            assert result.eta == pytest.approx(0.5, rel=0.1), block_length

    def test_uneven_blocks_give_eta_near_its_value(self) -> None:
        # A line at 24 positions in 3 blocks of 8, whose outer blocks hold
        # most of the curvature of the slope: η corrects the trace by the
        # leverage of the blocks and by median(χ²_ν)/ν. The median of η over
        # data sets is checked in test_smc.rs; here, its closed form.
        positions = np.arange(1.0, 25.0)
        values = 10 + 2 * positions + np.random.default_rng(3).normal(size=24)
        calibrator = line_calibrator(values.tolist())

        eta = calibrator.uncertainty(_matching(block_length=8)).eta

        design = np.column_stack([positions, np.ones_like(positions)])
        curvature = 2 * design.T @ design
        inverse = np.linalg.inv(curvature)
        fit = np.linalg.lstsq(design, values)[0]
        gradients = -2 * (values - design @ fit)[:, None] * design
        blocks = [slice(start, start + 8) for start in (0, 8, 16)]
        shares = [inverse @ (2 * design[b].T @ design[b]) for b in blocks]
        trace = sum(g @ inverse @ g for g in (gradients[b].sum(axis=0) for b in blocks))
        squares = sum(np.trace(a @ a) for a in shares)
        spread = squares - 2 * sum(np.trace(a @ a @ a) for a in shares)
        spread += np.trace(np.linalg.matrix_power(sum(a @ a for a in shares), 2))
        nu = (2 - squares) ** 2 / spread
        expected = (2 - squares) * _chi_square_median(nu) / nu / trace
        assert nu == pytest.approx(1.48, abs=0.01)
        assert eta == pytest.approx(expected, rel=1e-6)

    def test_eta_gives_its_degrees_of_freedom_and_warns_when_few(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # One constant flow at 24 positions: blocks of one size read
        # ν = G − 1, 2 in 3 blocks of 8 and 23 in 24 blocks of 1.
        values = normal_values(24, 10, sd=1.0)
        calibrator = flow_calibrator(values, windowed=True)

        with caplog.at_level(logging.WARNING, logger="commol.smc"):
            few = calibrator.uncertainty(_matching(block_length=8))

        # P(|t₂| ≤ t) = t/√(2 + t²) at t = z √(2 / median(χ²₂)): 81 %.
        assert few.degrees_of_freedom == pytest.approx(2.0)
        (note,) = few.diagnostics.notes
        assert "ν = 2 (block_length 8)" in note
        assert "90% intervals are expected to cover about 81%" in note
        assert "Shorter blocks" in note
        assert _warnings(caplog) == [note]
        # Its log weights may change in the last digit (#535).
        read = GibbsResult.model_validate_json(few.model_dump_json())
        assert read.model_copy(update={"sample": few.sample}) == few

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="commol.smc"):
            enough = calibrator.uncertainty(_matching(block_length=1))

        assert enough.degrees_of_freedom == pytest.approx(23.0)
        assert enough.diagnostics.notes == []
        assert _warnings(caplog) == []

    def test_the_note_advises_only_what_the_blocks_allow(self) -> None:
        # Blocks of one position cannot be shorter; "auto" says it chose the
        # length.
        one = flow_calibrator(normal_values(6, 10, sd=1.0), windowed=True)
        (note,) = one.uncertainty(_matching(block_length=1)).diagnostics.notes
        assert "ν = 5 (block_length 1)" in note
        assert "Shorter blocks" not in note

        trend = (10 + np.repeat([-3.0, 3.0], 4)).tolist()
        automatic = flow_calibrator(trend, windowed=True).uncertainty(
            _matching(block_length="auto")
        )
        (note,) = automatic.diagnostics.notes
        assert f'(block_length {automatic.block_length}, from "auto")' in note


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The warnings of ``commol.smc``."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "commol.smc" and record.levelno == logging.WARNING
    ]


def _chi_square_median(nu: float) -> float:
    """The median of χ²_ν, by bisection on the series of ``P(ν/2, x/2)``."""

    def cdf(x: float) -> float:
        a, y = nu / 2, x / 2
        term = total = n = 1.0
        while term > 1e-17 * total:
            term, n = term * y / (a + n), n + 1
            total += term
        return math.exp(a * math.log(y) - y - math.lgamma(a + 1)) * total

    low, high = 0.0, 2 * nu + 2
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if cdf(middle) < 0.5 else (low, middle)
    return (low + high) / 2
