"""The degrees of freedom of a sandwich, read from the share of the curvature
that each block holds (Bell and McCaffrey 2002), by parameter and by output
of its band.

Two constant flows (``fan_calibrator``) fitted by the sum of squares: ``c0``
at the 40 positions 1 to 40, ``c1`` only at the first 10. In blocks of 5
positions every block holds the same share of the curvature of ``c0``, and
two of the eight hold all that of ``c1``.
"""

import math
import re

import numpy as np
import pytest

from commol import Calibrator, Simulation
from commol.local import Sandwich
from commol.local.results import SandwichResult
from tests.support import fan_calibrator, structural_zero_calibrator
from tests.support.problems import abc_builder, build

BLOCK, POSITION, LEVEL = 5, 4.0, 0.9

REFUSAL = (
    "block_length 5 makes 2 blocks with curvature, of {blocks} in all, for 2 parameters"
)
"""The start of the refusal of too few blocks with curvature."""


def _calibrator() -> Calibrator:
    rng = np.random.default_rng(7)
    return fan_calibrator(
        [
            (10.0 + rng.normal(0.0, 1.0, 40)).tolist(),
            (5.0 + rng.normal(0.0, 1.0, 10)).tolist(),
        ]
    )


def _block_curvatures() -> list[np.ndarray]:
    """The curvature of each block on the scale of the parameters, up to a
    factor that every block shares: one term per observation of each flow."""
    return [np.diag([BLOCK, BLOCK if b < 2 else 0]).astype(float) for b in range(8)]


def _degrees_of_freedom(direction: np.ndarray) -> float:
    """``(Σ_b v_bb)² / Σ_bc v_bc²`` with ``v_bc = cᵀ J⁻¹ V_bc J⁻¹ c`` and
    ``V_bc = J_b δ_bc − J_b J⁻¹ J_c``, from the whole matrix of the ``v_bc``."""
    blocks = _block_curvatures()
    inverse = np.linalg.inv(sum(blocks))
    u = inverse @ direction
    v = np.array(
        [
            [
                (u @ blocks[b] @ u if b == c else 0.0)
                - u @ blocks[b] @ inverse @ blocks[c] @ u
                for c in range(len(blocks))
            ]
            for b in range(len(blocks))
        ]
    )
    return float(np.trace(v) ** 2 / np.sum(v**2))


def _central_probability(nu: float, quantile: float) -> float:
    """``P(0 < T < quantile)`` of a Student t of `nu` degrees of freedom, by
    the trapezoidal rule on its density."""
    t = np.linspace(0.0, quantile, 20001)
    log_scale = (
        math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    )
    density = np.exp(log_scale - (nu + 1) / 2 * np.log1p(t**2 / nu))
    return float(np.trapezoid(density, t))


def _derivatives(result: SandwichResult) -> np.ndarray:
    """``dθ/du`` of each flow at the fit, on its bounds ``[0, 20]``."""
    fit = np.array([result.fit.best_parameters[name] for name in ("c0", "c1")])
    return fit * (20.0 - fit) / 20.0


def test_a_parameter_read_in_few_blocks_has_few_degrees_of_freedom() -> None:
    calibrator = _calibrator()

    result = calibrator.uncertainty(Sandwich(block_length=BLOCK, draws=10))

    assert result.degrees_of_freedom == pytest.approx({"c0": 7.0, "c1": 1.0}, rel=1e-9)
    # The influence of a block on each flow is the sum of its residuals over
    # the observations of the flow; the factor of c0 is 8/7, as with equal
    # blocks, and that of c1 is 2, as with its two blocks alone.
    fit = [result.fit.best_parameters[name] for name in ("c0", "c1")]
    series = [
        [o.value for o in calibrator.problem.observed_data if o.output == name]
        for name in ("B0", "B1")
    ]
    influences = np.zeros((8, 2))
    for i, values in enumerate(series):
        residuals = np.array(values) - fit[i]
        for b in range(len(values) // BLOCK):
            influences[b, i] = residuals[BLOCK * b : BLOCK * (b + 1)].sum() / len(
                values
            )
    influences /= _derivatives(result)
    roots = np.sqrt([8 / 7, 2.0])
    expected = np.outer(roots, roots) * (influences.T @ influences)
    np.testing.assert_allclose(result.covariance, expected, rtol=1e-6)


def test_a_band_reads_the_degrees_of_freedom_of_each_output() -> None:
    # At a position t, B0 = c0·t reads c0, B1 = c1·t reads c1 and
    # A = 10⁶ − (c0 + c1)·t reads both.
    calibrator = _calibrator()
    result = calibrator.uncertainty(Sandwich(block_length=BLOCK, draws=10))

    band = result.delta_band(calibrator, [POSITION], LEVEL)

    covariance = np.array(result.covariance)
    d = _derivatives(result)
    for output, direction in [
        ("B0", np.array([1.0, 0.0])),
        ("B1", np.array([0.0, 1.0])),
        ("A", np.array([-1.0, -1.0])),
    ]:
        gradient = POSITION * direction * d
        half = band.upper[output][0] - band.center[output][0]
        quantile = half / math.sqrt(gradient @ covariance @ gradient)
        nu = _degrees_of_freedom(direction)
        assert _central_probability(nu, quantile) == pytest.approx(
            LEVEL / 2, abs=1e-7
        ), (output, nu)


@pytest.mark.parametrize("weight", [1e-12, 1e-30, 1.0])
def test_blocks_of_little_weight_do_not_count_in_the_sandwich(weight: float) -> None:
    # Both flows at the 40 positions, of weight 1 at the first 10 and
    # `weight` after them: with weight 10⁻¹², 6 of the 8 blocks hold a share
    # of about 10⁻¹² of the curvature.
    rng = np.random.default_rng(5)
    base = fan_calibrator(
        [
            (10.0 + rng.normal(0.0, 1.0, 40)).tolist(),
            (5.0 + rng.normal(0.0, 1.0, 40)).tolist(),
        ]
    )
    data = [
        o.model_copy(update={"weight": 1.0 if o.at <= 10 else weight})
        for o in base.problem.observed_data
    ]
    problem = base.problem.model_copy(update={"observed_data": data})
    calibrator = Calibrator(base.simulation, problem)
    method = Sandwich(block_length=BLOCK, draws=10)

    if weight == 1.0:
        assert calibrator.uncertainty(method).blocks == 8
        return
    with pytest.raises(ValueError, match=f"^{re.escape(REFUSAL.format(blocks=8))}: "):
        calibrator.uncertainty(method)


def test_a_mean_near_zero_adds_no_block_with_curvature() -> None:
    # C fills from B at 10⁻¹⁴ · B up to step 10: its observations there,
    # at 0, carry a share of about 10⁻²⁸ of the curvature of k1.
    full = structural_zero_calibrator(1)
    model = build(
        abc_builder(
            parameters={"k1": None},
            ab="k1 * A",
            bc="if(step >= 10, 0.1 * B, 1e-14 * B)",
            amounts={"A": 1000, "B": 0, "C": 0},
        )
    )
    data = [o for o in full.problem.observed_data if o.label == "C" or o.at >= 11]
    problem = full.problem.model_copy(update={"observed_data": data})

    with pytest.raises(ValueError, match=f"^{re.escape(REFUSAL.format(blocks=4))}: "):
        Calibrator(Simulation(model), problem).uncertainty(
            Sandwich(block_length=BLOCK, draws=2)
        )


def test_two_identical_series_are_refused_at_the_fit_of_an_optimizer() -> None:
    # Both flows read the same series, so their scores move together in every
    # block; the fit of the optimizer is off the optimum, and the scores of
    # the blocks are read after one step to the zero of their sum.
    rng = np.random.default_rng(3)
    series = (10.0 + rng.normal(0.0, 1.0, 40)).tolist()
    calibrator = fan_calibrator([series, series])

    with pytest.raises(
        ValueError,
        match="^block_length 5 makes block scores of rank below the parameters: ",
    ):
        calibrator.uncertainty(Sandwich(block_length=BLOCK, draws=2))
