"""The steps of ``curvature="hessian"`` where the curvature is so large that a
step of ``δ^{1/4}`` moves the objective above the largest number: a flow ``c``
read by the observation "far" of a large scale (``far_calibrator``)."""

import pytest

from commol.local import Laplace
from tests.support import far_calibrator


@pytest.mark.parametrize(
    ("rate", "low"), [("c * A / A", 0.0), ("(c - 1e6) * A / A", 1e6)]
)
def test_a_hessian_reads_a_curvature_whose_differences_would_overflow(
    rate: str, low: float
) -> None:
    # At a scale of 1e156 the curvature on the unbounded scale is 2.5e317:
    # steps of δ^{1/4} there move the objective above the largest number. c in
    # [1e6, 1e6 + 100] gives the runs of c in [0, 100], and the shorter step,
    # 2⁻¹⁴, is far above the rounding of c and of the runs.
    calibrator = far_calibrator(1e156, rate=rate, low=low)

    expected = calibrator.uncertainty(Laplace(draws=2))
    observed = calibrator.uncertainty(Laplace(draws=2, curvature="hessian"))

    assert expected.covariance[0][0] == pytest.approx(4e-318, rel=1e-3)
    assert observed.covariance[0][0] == pytest.approx(
        expected.covariance[0][0], rel=1e-5
    )


@pytest.mark.parametrize(
    ("scale", "rate", "low", "amount", "far"),
    [
        (1e157, "c * A / A", 0.0, 1e12, 1e12 + 1000.0),
        (1e156, "(c + 1e11) * A / A", -50.0, 0.0, 2e12),
    ],
)
def test_a_hessian_refuses_a_step_that_the_rounding_of_the_runs_hides(
    scale: float, rate: str, low: float, amount: float, far: float
) -> None:
    # The runs of far, about 1e169 and 2e168, round by about their change at a
    # step that keeps the objective a number.
    calibrator = far_calibrator(scale, rate=rate, low=low, amount=amount, far=far)

    with pytest.raises(
        RuntimeError,
        match="too large for the Hessian: .* within the rounding of the "
        "predictions of the model at the fit",
    ):
        calibrator.uncertainty(Laplace(draws=2, curvature="hessian"))
