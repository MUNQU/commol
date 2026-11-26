"""A side of a profile that ends at `max_steps`: the profile gives a verdict
only when no longer side can change it, and a note names the side. And what
the resets of a stochastic profile say, flat or rising."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.identifiability import ProfileIdentifiability
from commol.observations import NormalObservations
from commol.profiles import ProfileLikelihood, Threshold
from commol.profiles.results import ProfileResult
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    build_stochastic,
    line_calibrator,
    observed,
)

# ``5 + 2 t`` at t = 1, …, 10 plus a fixed draw of unit noise.
VALUES = [7.4, 8.6, 11.3, 12.8, 15.1, 16.4, 19.9, 20.6, 23.5, 24.7]
# ``5 + 0.1 t`` at t = 1, …, 10 plus a fixed draw of noise: the slope sits
# near its lower bound, 0.
SHALLOW = [5.5, 4.8, 5.6, 5.2, 5.6, 5.2, 6.6, 5.4, 6.4, 5.7]


def test_a_side_at_max_steps_gives_no_verdict() -> None:
    calibrator = line_calibrator(
        VALUES, observation_models={"B": NormalObservations(sd=1.0)}
    )

    def profiled(max_steps: int) -> ProfileResult:
        return calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c"],
                threshold=Threshold.chi_square(0.95),
                max_steps=max_steps,
            )
        )

    short = profiled(1)
    full = profiled(100)

    assert short.profiles["c"].ends == ("max_steps", "max_steps")
    assert short.verdicts["c"] is None
    assert short.profiles["c"].undetermined == "max_steps"
    assert (
        "'c': the sides below and above the fit end at max_steps (1) below "
        "the threshold. A longer side could change the verdict, so there is no "
        "verdict. Raise max_steps to read one." in short.notes
    )
    practical = (
        calibrator.identifiability(ProfileIdentifiability(result=short))
        .verdicts["c"]
        .practical
    )
    assert practical is not None
    assert practical.verdict is None
    assert practical.reasons == [
        "profile likelihood: no verdict: the sides below and above the fit "
        "end at max_steps below the threshold, and a longer side could change "
        "the verdict. Raise max_steps to read one."
    ]
    assert full.profiles["c"].ends == ("crossed", "crossed")
    assert full.verdicts["c"] == "identifiable"
    assert full.profiles["c"].undetermined is None
    assert not any("max_steps" in note for note in full.notes)


def test_a_side_at_max_steps_keeps_a_verdict_it_cannot_change() -> None:
    # The side below rises to the bound of the slope, so no longer side
    # above makes the profile identifiable or flat.
    calibrator = line_calibrator(
        SHALLOW, observation_models={"B": NormalObservations(sd=1.0)}
    )

    result = calibrator.uncertainty(
        ProfileLikelihood(
            parameters=["c"], threshold=Threshold.chi_square(0.95), max_steps=3
        )
    )

    assert result.profiles["c"].ends == ("bound", "max_steps")
    assert result.verdicts["c"] == "practically_non_identifiable"
    assert result.profiles["c"].undetermined is None
    assert (
        "'c': the side above the fit ends at max_steps (3) below the "
        "threshold. A longer side cannot change the verdict." in result.notes
    )


def test_a_side_at_max_steps_below_the_tolerance_claims_nothing() -> None:
    # Exact data give Beale's threshold 0, and `u` changes nothing, so both
    # sides stay flat up to `max_steps`.
    truth = Simulation(build(a_to_b_builder(parameters={"k": 0.1}))).run(10)
    model = build(a_to_b_builder(parameters={"k": None, "u": None}, rate="k + 0 * u"))
    problem = CalibrationProblem(
        observed_data=observed(truth, "B", range(1, 11)),
        parameters=[bounded("k", 0.0, 1.0), bounded("u", 0.0, 1.0)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=500, sd_tolerance=1e-14),
        seed=1,
    )
    calibrator = Calibrator(Simulation(model), problem)

    for max_steps in (1, 3):
        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["u"], threshold=Threshold.beale(0.95), max_steps=max_steps
            )
        )

        assert result.profiles["u"].ends == ("max_steps", "max_steps")
        assert result.verdicts["u"] is None
        assert result.profiles["u"].undetermined == "below_tolerance"
        notes = [note for note in result.notes if "max_steps" in note]
        assert notes == [
            f"'u': the sides below and above the fit end at max_steps ({max_steps}) "
            "below the threshold."
        ]
        assert any("below the numerical tolerance" in note for note in result.notes)


def _stochastic_calibrator(k_bounds: tuple[float, float]) -> Calibrator:
    """The stochastic A→B model with the rate ``k + 0 * u``, fitted to the mean
    of 50 replicates of A, with `k` in `k_bounds` and `u` in ``[0, 1]``."""
    builder = a_to_b_builder(parameters={"k": None, "u": None}, rate="k + 0 * u")
    positions = [2.0, 4.0, 6.0, 8.0]
    data = (
        Simulation(build_stochastic(a_to_b_builder()))
        .replicates(50, seed=21)
        .run_at(positions)
        .mean()
    )
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=at, output="A", value=value)
            for at, value in zip(positions, data["A"], strict=True)
        ],
        parameters=[bounded("k", *k_bounds), bounded("u", 0.0, 1.0)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=200),
        replicates=4,
        replicate_summary="mean",
        seed=7,
    )
    return Calibrator(Simulation(build_stochastic(builder)), problem)


def test_a_flat_stochastic_profile_beside_a_bound_is_practical() -> None:
    # `u` changes nothing. A stochastic engine reads a flat profile as
    # practical, so with the side above at a bound no longer side below can
    # change the verdict.
    calibrator = _stochastic_calibrator((0.01, 0.5))

    result = calibrator.uncertainty(
        ProfileLikelihood(
            parameters=["u"], threshold=Threshold.beale(0.95), max_steps=5
        )
    )

    assert result.profiles["u"].ends == ("max_steps", "bound")
    assert result.verdicts["u"] == "practically_non_identifiable"
    assert result.profiles["u"].undetermined is None
    # As for a profile that ends at both bounds, the resets are not read.
    assert result.structure["u"].undetermined == "stochastic"
    assert any("'u'" in note and "stochastic engine" in note for note in result.notes)


@pytest.mark.parametrize(
    ("max_steps", "ends"), [(100, ("bound", "crossed")), (2, ("bound", "max_steps"))]
)
def test_a_rising_stochastic_profile_reads_not_flat(
    max_steps: int, ends: tuple[str, str]
) -> None:
    # The side above already rises, so a longer side cannot make it flat.
    calibrator = _stochastic_calibrator((0.098, 0.5))

    result = calibrator.uncertainty(
        ProfileLikelihood(
            parameters=["k"], threshold=Threshold.beale(0.95), max_steps=max_steps
        )
    )

    assert result.profiles["k"].ends == ends
    assert result.verdicts["k"] == "practically_non_identifiable"
    assert result.structure["k"].undetermined == "not_flat"
    assert not any("stochastic engine" in note for note in result.notes)
