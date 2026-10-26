"""One entry point: ``Calibrator.uncertainty`` runs the runner of a
configuration's type, and its result type follows that type."""

from collections.abc import Callable
from typing import assert_type

import pytest
from pydantic import ValidationError

from commol.resampling import BlockJackknife, ResidualBootstrap
from commol.smc import BootstrapCoverageEta, GibbsPosterior
from commol import (
    CalibrationConstraint,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.conformal import EnbPI, SplitConformal
from commol.sensitivity import InputSpace, PolynomialChaos, SensitivityAnalysis, Sobol
from commol.observations import LogNormalObservations, NormalObservations
from commol.local import Laplace
from commol.distributions import Priors
from commol.profiles import ProfileLikelihood, Threshold
from commol.quantities import Ratio, ValueAt
from commol.identifiability import StructuralRank
from commol._calibration.method import SensitivityMethod, UncertaintyMethod
from commol._calibration.registry import register, registered
from tests.support import (
    a_to_b_builder,
    abc_builder,
    bounded,
    build,
    build_stochastic,
    flow_calibrator,
    flow_observations,
    line_calibrator,
    rate_calibrator,
)
from tests.support.methods import StartsResult, StartsSample, run_starts_sample


class Unregistered(UncertaintyMethod[StartsResult]):
    """A method with no runner."""


class UnregisteredSensitivity(SensitivityMethod[StartsResult]):
    """A sensitivity method with no runner."""


class TestEntryPoint:
    def test_the_runner_of_the_configuration_type_runs(
        self, calibrator: Calibrator
    ) -> None:
        result = calibrator.uncertainty(StartsSample(starts=3))

        assert_type(result, StartsResult)
        assert isinstance(result, StartsResult)
        assert len(result.sample) == 3
        assert result.sample.method.settings == {"starts": 3}
        assert result.sample.names == ["k1", "k2"]

    def test_an_unregistered_configuration_names_the_registered_ones(
        self, calibrator: Calibrator
    ) -> None:
        with pytest.raises(ValueError, match=r"Unregistered.*StartsSample"):
            calibrator.uncertainty(Unregistered())

    def test_a_type_has_one_runner(self) -> None:
        with pytest.raises(ValueError, match="already has a runner"):
            register(StartsSample, run_starts_sample)

    def test_the_registry_lists_its_types(self) -> None:
        assert "StartsSample" in registered()

    def test_each_entry_point_refuses_the_other_kinds(
        self, calibrator: Calibrator
    ) -> None:
        analysis = SensitivityAnalysis(
            calibrator.simulation,
            InputSpace.uniform_on_bounds(calibrator.problem.parameters),
            seed=1,
        )
        with pytest.raises(TypeError, match=r"Calibrator\.uncertainty"):
            calibrator.identifiability(StartsSample(starts=2))  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match=r"Calibrator\.identifiability"):
            calibrator.uncertainty(
                StructuralRank(points=2, design="observations")  # ty: ignore[invalid-argument-type]
            )
        with pytest.raises(TypeError, match=r"SensitivityAnalysis\.run"):
            calibrator.uncertainty(UnregisteredSensitivity())  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match=r"Calibrator\.uncertainty"):
            analysis.run(StartsSample(starts=2))  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match=r"No method is registered"):
            analysis.run(UnregisteredSensitivity())


VALUE = ValueAt(series="B", at=5)


@pytest.mark.parametrize(
    ("build", "old"),
    [
        (
            lambda: SplitConformal(
                calibration_data=[ObservedDataPoint(at=1, output="B", value=1.0)],
                positions=[2.0],
                score="absolute",
                by="all",
                level=0.9,
            ),
            "level",
        ),
        (lambda: Threshold(kind="beale", level=0.9), "level"),
        (lambda: ParticleSwarmConfig(num_particles=5), "num_particles"),
        (
            lambda: Sobol(
                base_draws=8, replicates=2, confidence_level=0.9, quantities=[VALUE]
            ),
            "base_draws",
        ),
        (
            lambda: Sobol(
                draws=8, bootstrap=2, confidence_level=0.9, quantities=[VALUE]
            ),
            "bootstrap",
        ),
        (
            lambda: Ratio(numerator=VALUE, denominator=VALUE, replicates="within"),
            "replicates",
        ),
    ],
    ids=["conformal", "threshold", "swarm", "sobol draws", "sobol replicates", "ratio"],
)
def test_an_old_setting_name_is_rejected(build: Callable[[], object], old: str) -> None:
    with pytest.raises(
        ValidationError, match=rf"\n{old}\n  Extra inputs are not permitted"
    ):
        build()


# Every method maps its errors by one rule: a setting, an input, the data or
# the model that does not fit the method is a ValueError; a computation that
# could start and failed is a RuntimeError.

_FLOWS = [10.0, 11.0, 9.0, 10.5, 10.2, 9.8]


def _normal_flow(rate: str, values: list[float], sd: float) -> Calibrator:
    """A flow from A to B at `rate`, fitted by ``c`` in [0, 50] to `values`
    of a normal model of standard deviation `sd`."""
    model = build(
        a_to_b_builder(parameters={"c": None}, rate=rate, amounts={"A": 1e6, "B": 0})
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(values, windowed=True),
        parameters=[bounded("c", 0.0, 50.0, initial_guess=12.0)],
        observation_models={"B": NormalObservations(sd=sd)},
        optimization_config=NelderMeadConfig(max_iterations=1000),
        seed=1,
    )
    return Calibrator(Simulation(model), problem)


def _analysis(calibrator: Calibrator) -> SensitivityAnalysis:
    return SensitivityAnalysis(
        calibrator.simulation,
        InputSpace.uniform_on_bounds(calibrator.problem.parameters),
        seed=1,
    )


def _invalid_local() -> object:
    # Values below 5 put the fit of c on its lower bound.
    calibrator = flow_calibrator(
        [1.0, 2.0, 1.0],
        windowed=True,
        bounds=(5.0, 50.0),
        observation_models={"B": NormalObservations(sd=1.0)},
    )
    return calibrator.uncertainty(Laplace(draws=10))


def _invalid_resampling() -> object:
    calibrator = flow_calibrator(_FLOWS[:4], windowed=True)
    return calibrator.uncertainty(BlockJackknife(block_length=4))


def _invalid_profile() -> object:
    calibrator = _normal_flow("c * A / A", _FLOWS, 1.0)
    return calibrator.uncertainty(
        ProfileLikelihood(parameters=["c"], threshold=Threshold.beale(0.95))
    )


def _invalid_smc() -> object:
    # The flow is 0 for every c, as are the values: the loss at the fit is 0.
    calibrator = rate_calibrator("0 * c * A / A", [0.0] * 5)
    return calibrator.uncertainty(
        GibbsPosterior(
            priors=Priors.uniform_on_bounds(),
            eta=BootstrapCoverageEta(
                replicates=10, block_length="auto", confidence_level=0.8
            ),
            particles=50,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.05,
        )
    )


def _invalid_conformal() -> object:
    # Two replicates of 40 blocks leave some block out of neither.
    calibrator = flow_calibrator([float(t) for t in range(1, 41)], windowed=False)
    return calibrator.uncertainty(
        EnbPI(
            replicates=2,
            block_length=1,
            score="absolute",
            by="series",
            confidence_level=0.5,
            positions=[41.0],
        )
    )


def _invalid_identifiability() -> object:
    model = build_stochastic(
        abc_builder(parameters={"k1": None, "k2": 0.1}), "StochasticSteps"
    )
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=10, output="B", value=50.0)],
        parameters=[bounded("k1", 0.1, 0.6)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(),
        replicates=4,
        replicate_summary="mean",
        seed=2,
    )
    calibrator = Calibrator(Simulation(model), problem)
    return calibrator.identifiability(StructuralRank(points=2, design="observations"))


def _invalid_sensitivity() -> object:
    analysis = _analysis(flow_calibrator(_FLOWS, windowed=True))
    return analysis.run(
        PolynomialChaos(degree=3, draws=3, quantities=[ValueAt(series="B", at=5)])
    )


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (_invalid_local, "on its bound 5"),
        (_invalid_resampling, "makes 1 block; a jackknife needs at least two"),
        (_invalid_profile, "reads a sum of squares"),
        (_invalid_smc, "loss at the fit is 0"),
        (_invalid_conformal, r"[Bb]lock \[.*replicates"),
        (_invalid_identifiability, "SampleIdentifiability"),
        (_invalid_sensitivity, "needs more points than terms"),
    ],
    ids=[
        "local",
        "resampling",
        "profile",
        "smc",
        "conformal",
        "identifiability",
        "sensitivity",
    ],
)
def test_an_input_error_of_any_method_is_a_value_error(
    run: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run()


# The rate is not a number once c passes 20, or after a position.
_PAST_20 = "c * A / A + 0 * sqrt(20 - c)"


def _failed_local() -> object:
    # The Hessian steps from the fit on the constraint c ≤ 10 to an
    # infeasible point.
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
        )
    )
    problem = CalibrationProblem(
        observed_data=flow_observations([12.0, 13.0, 11.0, 12.5], windowed=True),
        parameters=[bounded("c", 0.0, 50.0, initial_guess=5.0)],
        observation_models={"B": NormalObservations(sd=1.0)},
        constraints=[CalibrationConstraint(id="at_most_10", expression="10 - c")],
        optimization_config=NelderMeadConfig(max_iterations=1000),
        seed=1,
    )
    calibrator = Calibrator(Simulation(model), problem)
    return calibrator.uncertainty(Laplace(draws=10, curvature="hessian"))


def _failed_resampling() -> object:
    # Values far from a line: every additive replicate has a value below 0,
    # which a log-normal model cannot read.
    calibrator = line_calibrator(
        [0.1, 20.0] * 4,
        observation_models={"B": LogNormalObservations(sigma=0.5, center="median")},
    )
    return calibrator.uncertainty(
        ResidualBootstrap(
            replicates=2, block_length=1, residual_scale="additive", scheme="moving"
        )
    )


def _failed_profile() -> object:
    # The fit is 12; with sd 10 the profile walks past 20.
    calibrator = _normal_flow(_PAST_20, [12.0, 12.5, 11.5, 12.0], 10.0)
    return calibrator.uncertainty(
        ProfileLikelihood(parameters=["c"], threshold=Threshold.chi_square(0.95))
    )


def _failed_smc() -> object:
    # The prior draws c up to 50, where the run fails.
    calibrator = rate_calibrator(_PAST_20, _FLOWS)
    return calibrator.uncertainty(
        GibbsPosterior(
            priors=Priors.uniform_on_bounds(),
            eta=1.0,
            particles=50,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.05,
        )
    )


def _failed_conformal() -> object:
    calibrator = rate_calibrator("c * A / A + 0 * sqrt(25.5 - t)", _FLOWS[:4])
    return calibrator.uncertainty(
        SplitConformal(
            calibration_data=flow_observations([10.0] * 30, windowed=True, start=5),
            score="absolute",
            by="series",
            confidence_level=0.5,
            positions=[40.0],
        )
    )


def _failed_identifiability() -> object:
    calibrator = rate_calibrator("c * A / A + 0 * sqrt(4.5 - t)", _FLOWS)
    return calibrator.identifiability(StructuralRank(points=2, design="observations"))


def _failed_sensitivity() -> object:
    analysis = _analysis(rate_calibrator("c * A / A + 0 * sqrt(2 - t)", _FLOWS))
    return analysis.run(
        PolynomialChaos(degree=1, draws=8, quantities=[ValueAt(series="B", at=5)])
    )


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (_failed_local, "The Hessian of the objective at the fit failed"),
        (_failed_resampling, "Every replicate failed"),
        (_failed_profile, "is not a number"),
        (_failed_smc, "is not a number"),
        (_failed_conformal, "Parameter set 0: Simulation failed"),
        (_failed_identifiability, "at 5 is not a number"),
        (_failed_sensitivity, "is not a number"),
    ],
    ids=[
        "local",
        "resampling",
        "profile",
        "smc",
        "conformal",
        "identifiability",
        "sensitivity",
    ],
)
def test_a_failed_computation_of_any_method_is_a_runtime_error(
    run: Callable[[], object], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        run()
