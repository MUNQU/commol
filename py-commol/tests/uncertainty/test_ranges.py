"""Every setting with a range rule gets the message of the engine when its
object is built with a value outside the range.

The cases come from the rules of the engine (``commol_rs.core.range_rules``):
each gives an object, a setting, a value outside its range and the message of
the rule. A count also gets ``-1``, which has the same wording.
"""

import math
import re
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from commol import (
    AdaptiveConformal,
    Increments,
    ABCSMC,
    PRCC,
    ABCRejection,
    AxisPattern,
    BayesianBootstrap,
    BayesianPosterior,
    Beta,
    BinomialObservations,
    BlockJackknife,
    BootstrapCoverageEta,
    CalibrationConstraint,
    CalibrationParameter,
    CalibrationProblem,
    CalibrationResult,
    Collinearity,
    EnbPI,
    Gain,
    Gamma,
    GibbsPosterior,
    GibbsTarget,
    InformationMatchingEta,
    JackknifePlus,
    Laplace,
    LaplaceMixture,
    LogNormal,
    LogNormalObservations,
    LossDistance,
    Morris,
    NegativeBinomialObservations,
    NelderMeadConfig,
    Normal,
    NormalObservations,
    ObservedDataPoint,
    ParameterSample,
    ParametricBootstrap,
    ParticleSwarmConfig,
    Peak,
    PeakPosition,
    PolynomialChaos,
    PredictionProfile,
    PredictionSpec,
    Priors,
    ProfileLikelihood,
    Propagation,
    ResidualBootstrap,
    ResidualNoise,
    RollingConformal,
    SampleIdentifiability,
    Sandwich,
    Sobol,
    SplitConformal,
    StartsIdentifiability,
    StructuralRank,
    Simulation,
    StudentTObservations,
    Threshold,
    Trajectory,
    ValueAt,
    WildBootstrap,
)
from commol.api.simulation import ReplicateRuns
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.inference.sample import SampleMethod
from commol.context.calibration import (
    PSOChaoticInertia,
    PSOConstantAcceleration,
    PSOConstantInertia,
    PSOImprovementStop,
    PSOMutationConfig,
    PSORadiusStop,
    PSOTimeVaryingAcceleration,
    PSOVelocityConfig,
)
from tests.support import a_to_b_builder, build_stochastic

PRIORS = Priors.uniform_on_bounds()
FIT = CalibrationResult(
    best_parameters={"c": 0.5},
    final_loss=0.0,
    iterations=1,
    converged=True,
    termination_reason="Test",
    seed=0,
)
POINT = ObservedDataPoint(at=1.0, output="B", value=1.0)
QUANTITIES = [ValueAt(series="B", at=1.0)]
THRESHOLD = Threshold(kind="chi_square", confidence_level=0.95)
CONFORMAL = {"score": "absolute", "by": "series", "confidence_level": 0.9}
SAMPLE = ParameterSample.equally_weighted(
    names=["c"],
    bounds=[(0.0, 1.0)],
    values=[[0.5]],
    method=SampleMethod(name="laplace"),
)


def replicate_runs() -> ReplicateRuns:
    """Two replicates of a stochastic model A -> B."""
    simulation = Simulation(
        build_stochastic(a_to_b_builder(amounts={"A": 10, "B": 0}), "StochasticSteps")
    )
    return ReplicateRuns(simulation, 2, None)


def pattern(name: str) -> Callable[[], BaseModel]:
    """A valid axis pattern of the private class ``name``."""
    patterns = {
        "_PulseAxisPattern": lambda: AxisPattern.pulse(at=1, value=1.0),
        "_PulsesAxisPattern": lambda: AxisPattern.pulses(at=[1, 2], value=1.0),
        "_PeriodicAxisPattern": lambda: AxisPattern.periodic(period=2, value=1.0),
        "_IntervalAxisPattern": lambda: AxisPattern.interval(
            start=0.0, end=1.0, value=1.0
        ),
        "_GaussianPulseAxisPattern": lambda: AxisPattern.gaussian_pulse(
            center=1.0, width=1.0, peak=1.0
        ),
        "_LinearRampAxisPattern": lambda: AxisPattern.linear_ramp(
            start=0.0, end=1.0, start_value=0.0, end_value=1.0
        ),
    }
    return patterns[name]


VALID: dict[str, Callable[[], object]] = {
    **{
        name: pattern(name)
        for name in (
            "_PulseAxisPattern",
            "_PulsesAxisPattern",
            "_PeriodicAxisPattern",
            "_IntervalAxisPattern",
            "_GaussianPulseAxisPattern",
            "_LinearRampAxisPattern",
        )
    },
    "ObservedDataPoint": lambda: POINT,
    "CalibrationConstraint": lambda: CalibrationConstraint(
        id="c", expression="c - 1", at=[1.0]
    ),
    "NormalObservations": lambda: NormalObservations(sd=1.0),
    "LogNormalObservations": lambda: LogNormalObservations(sigma=1.0, center="mean"),
    "StudentTObservations": lambda: StudentTObservations(df=3.0, scale=1.0),
    "NegativeBinomialObservations": lambda: NegativeBinomialObservations(
        overdispersion=0.1
    ),
    "BinomialObservations": lambda: BinomialObservations(trials=10),
    "NelderMeadConfig": NelderMeadConfig,
    "PSOConstantInertia": lambda: PSOConstantInertia(factor=0.7),
    "PSOChaoticInertia": lambda: PSOChaoticInertia(w_min=0.4, w_max=0.9),
    "PSOConstantAcceleration": lambda: PSOConstantAcceleration(
        cognitive=1.5, social=1.5
    ),
    "PSOTimeVaryingAcceleration": lambda: PSOTimeVaryingAcceleration(
        c1_initial=2.5, c1_final=0.5, c2_initial=0.5, c2_final=2.5
    ),
    "PSOMutationConfig": lambda: PSOMutationConfig(
        strategy="gaussian", scale=0.1, probability=0.1, application="all_particles"
    ),
    "PSOVelocityConfig": PSOVelocityConfig,
    "PSOImprovementStop": lambda: PSOImprovementStop(
        window=5, tol_abs=0.0, tol_rel=0.0
    ),
    "PSORadiusStop": lambda: PSORadiusStop(tol=0.1),
    "ParticleSwarmConfig": ParticleSwarmConfig,
    "CalibrationProblem": lambda: CalibrationProblem(
        observed_data=[POINT],
        parameters=[
            CalibrationParameter(
                id="c", parameter_type="parameter", min_bound=0.0, max_bound=1.0
            )
        ],
        optimization_config=NelderMeadConfig(),
        loss_function="sse",
    ),
    "CalibrationResult": lambda: FIT,
    "InformationMatchingEta": lambda: InformationMatchingEta(block_length=1),
    "BootstrapCoverageEta": lambda: BootstrapCoverageEta(
        replicates=2, block_length=1, confidence_level=0.9
    ),
    "GibbsPosterior": lambda: GibbsPosterior(
        priors=PRIORS,
        eta=1.0,
        particles=10,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.5,
    ),
    "BayesianPosterior": lambda: BayesianPosterior(
        priors=PRIORS,
        particles=10,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.5,
    ),
    "ABCSMC": lambda: ABCSMC(
        priors=PRIORS,
        distance=LossDistance(),
        particles=10,
        target_cess=0.5,
        min_acceptance=0.1,
        stay_probability=0.5,
    ),
    "ABCRejection": lambda: ABCRejection(
        priors=PRIORS, distance=LossDistance(), draws=10, quantile=0.5
    ),
    "BayesianBootstrap": lambda: BayesianBootstrap(replicates=2, block_length=1),
    "ParametricBootstrap": lambda: ParametricBootstrap(replicates=2),
    "ResidualBootstrap": lambda: ResidualBootstrap(
        replicates=2, block_length=1, residual_scale="additive", scheme="moving"
    ),
    "WildBootstrap": lambda: WildBootstrap(
        replicates=2, block_length=1, multipliers="rademacher"
    ),
    "BlockJackknife": lambda: BlockJackknife(block_length=1),
    "ResidualNoise": lambda: ResidualNoise(
        fit=FIT, block_length=1, residual_scale="additive"
    ),
    "SplitConformal": lambda: SplitConformal(
        **CONFORMAL, calibration_data=[POINT], positions=[1.0]
    ),
    "RollingConformal": lambda: RollingConformal(
        **CONFORMAL, origins=[1.0], positions=[2.0]
    ),
    "JackknifePlus": lambda: JackknifePlus(
        **CONFORMAL, block_length=1, positions=[1.0]
    ),
    "EnbPI": lambda: EnbPI(**CONFORMAL, replicates=2, block_length=1, positions=[1.0]),
    "AdaptiveConformal": lambda: AdaptiveConformal(
        **CONFORMAL, step=0.01, stream=[POINT]
    ),
    "Threshold": lambda: THRESHOLD,
    "PredictionSpec": lambda: PredictionSpec(at=1.0, output="B"),
    "ProfileLikelihood": lambda: ProfileLikelihood(
        parameters=["c"], threshold=THRESHOLD
    ),
    "PredictionProfile": lambda: PredictionProfile(
        specs=[PredictionSpec(at=1.0, output="B")], threshold=THRESHOLD
    ),
    "StructuralRank": lambda: StructuralRank(points=3, design="observations"),
    "Collinearity": lambda: Collinearity(at=FIT, max_subset_size=2, threshold=20.0),
    "SampleIdentifiability": lambda: SampleIdentifiability(
        sample=SAMPLE, reference="bounds", correlation_threshold=0.9
    ),
    "StartsIdentifiability": lambda: StartsIdentifiability(
        result=FIT, parameter_tolerance=0.1, loss_tolerance=0.1
    ),
    "Laplace": lambda: Laplace(draws=10),
    "Sandwich": lambda: Sandwich(block_length=1, draws=10),
    "GibbsTarget": lambda: GibbsTarget(priors=PRIORS, eta=1.0),
    "LaplaceMixture": lambda: LaplaceMixture(
        draws=25, target=GibbsTarget(priors=PRIORS, eta=1.0)
    ),
    "Propagation": Propagation,
    "PRCC": lambda: PRCC(quantities=QUANTITIES),
    "Morris": lambda: Morris(trajectories=2, levels=4, quantities=QUANTITIES),
    "Sobol": lambda: Sobol(
        draws=2, replicates=2, confidence_level=0.9, quantities=QUANTITIES
    ),
    "PolynomialChaos": lambda: PolynomialChaos(
        degree=1, draws=2, quantities=QUANTITIES
    ),
    "Beta": lambda: Beta(a=1.0, b=1.0),
    "Normal": lambda: Normal(mean=0.0, sd=1.0),
    "LogNormal": lambda: LogNormal(mu=0.0, sigma=1.0),
    "Gamma": lambda: Gamma(shape=1.0, rate=1.0),
    "ValueAt": lambda: QUANTITIES[0],
    "Gain": lambda: Gain(outputs=["B"], start=0.0, end=1.0),
    "Peak": lambda: Peak(series="B", positions=[1.0]),
    "PeakPosition": lambda: PeakPosition(series="B", positions=[1.0]),
    "Trajectory": lambda: Trajectory(series="B", positions=[1.0]),
    "Increments": lambda: Increments(outputs=["B"], window=1.0),
    "ReplicateRuns": replicate_runs,
}
"""A valid object of each object of the engine that has range rules."""


def cases() -> list[tuple[str, str, object, str]]:
    """Each rule with its value outside the range, and ``-1`` for a count."""
    found = []
    for owner, name, outside, message in commol_rs.core.range_rules():
        found.append((owner, name, outside, message))
        if isinstance(outside, int) and outside != -1:
            found.append((owner, name, -1, message.replace(f"got {outside}", "got -1")))
    return found


def outside_settings(model: object, name: str, outside: object) -> dict:
    """The settings of ``model`` (the fields of a pydantic model, or the
    attributes of another object) with ``name`` outside its range: the value,
    or a list of it for a setting that is a list."""
    fields = type(model).model_fields if isinstance(model, BaseModel) else vars(model)
    settings = {field: getattr(model, field) for field in fields}
    current = settings[name]
    settings[name] = [outside] if isinstance(current, list) else outside
    return settings


@pytest.mark.parametrize(
    ("owner", "name", "outside", "message"),
    cases(),
    ids=lambda value: str(value),
)
def test_a_setting_outside_its_range_gets_the_message_of_the_engine(
    owner: str, name: str, outside: object, message: str
) -> None:
    valid = VALID[owner]()
    settings = outside_settings(valid, name, outside)

    with pytest.raises(ValueError, match=re.escape(message)):
        type(valid)(**settings)


@pytest.mark.parametrize("name", ["NaN", "Infinity", "-Infinity"])
def test_a_parameter_named_like_a_number_is_not_a_number(name: str) -> None:
    assert NormalObservations(sd=name).sd == name
    assert StudentTObservations(df=name, scale=name).scale == name


@pytest.mark.parametrize(
    ("value", "shown"), [(math.nan, "NaN"), (math.inf, "inf"), (-math.inf, "-inf")]
)
def test_a_number_that_is_not_finite_gets_the_message_of_the_engine(
    value: float, shown: str
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"sd is a finite number greater than 0, got {shown}"),
    ):
        NormalObservations(sd=value)


@pytest.mark.parametrize(
    ("count", "shown"),
    [(10**40, "1" + "0" * 40), (10**400, "inf"), (-(10**400), "-inf")],
)
def test_a_count_beyond_the_whole_numbers_of_the_engine_is_rejected(
    count: int, shown: str
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"replicates is a whole number of at least 1, got {shown}"),
    ):
        ParametricBootstrap(replicates=count)


def test_each_number_of_a_tuple_is_checked() -> None:
    with pytest.raises(
        ValueError, match=re.escape("at is a finite number of at least 0, got -1")
    ):
        commol_rs.core.check_ranges("CalibrationConstraint", {"at": (1.0, -1.0)})
