"""A valid object of each data class whose JSON the engine reads
(``commol_rs.json.objects``), with settings away from their defaults where a
class has them, so that a test reads back more than the defaults."""

from collections.abc import Callable

from pydantic import BaseModel

from commol import (
    ABCSMC,
    PRCC,
    ABCRejection,
    AdaptiveConformal,
    BayesianBootstrap,
    BayesianPosterior,
    Beta,
    BinomialObservations,
    BlockJackknife,
    BootstrapCoverageEta,
    CalibrationConstraint,
    CalibrationParameter,
    CalibrationProblem,
    Collinearity,
    DistributionNoise,
    EnbPI,
    Gain,
    Gamma,
    GibbsPosterior,
    GibbsTarget,
    Increments,
    InformationMatchingEta,
    InputSpace,
    JackknifePlus,
    Laplace,
    LaplaceMixture,
    LikelihoodTarget,
    LogNormal,
    LogNormalObservations,
    LogUniform,
    LossDistance,
    Morris,
    NegativeBinomialObservations,
    NelderMeadConfig,
    NoNoise,
    Normal,
    NormalObservations,
    ObservedDataPoint,
    ParameterSample,
    ParametricBootstrap,
    ParticleSwarmConfig,
    Peak,
    PeakPosition,
    PoissonObservations,
    PolynomialChaos,
    PredictionProfile,
    PredictionSpec,
    Priors,
    ProfileLikelihood,
    Propagation,
    Quantile,
    Ratio,
    ResidualBootstrap,
    ResidualNoise,
    RollingConformal,
    SampleIdentifiability,
    SampleMethod,
    Sandwich,
    SeriesDistance,
    Sobol,
    SplitConformal,
    StartsIdentifiability,
    StructuralRank,
    StudentTObservations,
    Threshold,
    Trajectory,
    Uniform,
    ValueAt,
    WildBootstrap,
)
from commol.context.calibration import (
    CalibrationResult,
)

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
PARAMETER = CalibrationParameter(
    id="c", parameter_type="parameter", min_bound=0.0, max_bound=1.0
)
QUANTITIES = [ValueAt(series="B", at=1.0)]
THRESHOLD = Threshold(kind="chi_square", confidence_level=0.95)
CONFORMAL = {"score": "absolute", "by": "series", "confidence_level": 0.9}
SAMPLE = ParameterSample.equally_weighted(
    names=["c"],
    bounds=[(0.0, 1.0)],
    values=[[0.5]],
    method=SampleMethod(name="laplace"),
)


EXAMPLES: dict[str, Callable[[], BaseModel]] = {
    "ObservedDataPoint": lambda: POINT,
    "CalibrationParameter": lambda: PARAMETER,
    "CalibrationConstraint": lambda: CalibrationConstraint(
        id="c", expression="c - 1", at=[1.0]
    ),
    "CalibrationResult": lambda: FIT,
    "NelderMeadConfig": lambda: NelderMeadConfig(max_iterations=50, verbose=True),
    "ParticleSwarmConfig": lambda: ParticleSwarmConfig(
        particles=8,
        initialization="latin_hypercube",
        inertia="chaotic",
        inertia_min=0.4,
        inertia_max=0.9,
        acceleration="time_varying",
        cognitive_initial=2.5,
        cognitive_final=0.5,
        social_initial=0.5,
        social_final=2.5,
        mutation="cauchy",
        mutation_scale=0.1,
        mutation_probability=0.2,
        mutation_application="below_average",
        velocity_clamp_factor=0.5,
        improvement_window=10,
        improvement_tol_abs=1e-8,
        improvement_tol_rel=0.0,
        radius_tol=float("inf"),
    ),
    "Quantile": lambda: Quantile(level=0.9),
    "CalibrationProblem": lambda: CalibrationProblem(
        observed_data=[POINT],
        parameters=[PARAMETER],
        optimization_config=ParticleSwarmConfig(inertia="constant", inertia_factor=0.6),
        loss_function="sse",
        replicate_summary=Quantile(level=0.25),
        seed=3,
    ),
    "NormalObservations": lambda: NormalObservations(sd=1.0),
    "LogNormalObservations": lambda: LogNormalObservations(sigma=1.0, center="median"),
    "StudentTObservations": lambda: StudentTObservations(df=3.0, scale=1.0),
    "PoissonObservations": PoissonObservations,
    "NegativeBinomialObservations": lambda: NegativeBinomialObservations(
        overdispersion=0.5
    ),
    "BinomialObservations": lambda: BinomialObservations(trials=10),
    "Uniform": Uniform,
    "LogUniform": LogUniform,
    "Beta": lambda: Beta(a=1.0, b=2.0),
    "Normal": lambda: Normal(mean=0.0, sd=1.0),
    "LogNormal": lambda: LogNormal(mu=0.0, sigma=1.0),
    "Gamma": lambda: Gamma(shape=1.0, rate=2.0),
    "Priors": lambda: Priors(distributions={"c": Beta(a=1.0, b=2.0)}),
    "ValueAt": lambda: ValueAt(series=Increments("B", 0.5), at=1.0, label="b"),
    "Gain": lambda: Gain(outputs=["B"], start=0.0, end=1.0),
    "Peak": lambda: Peak(series="B", positions=[1.0, 2.0]),
    "PeakPosition": lambda: PeakPosition(series="B", positions=[1.0]),
    "Trajectory": lambda: Trajectory(series=["A", "B"], positions=[1.0]),
    "Ratio": lambda: Ratio(
        numerator=ValueAt(series="A", at=1.0),
        denominator=ValueAt(series="B", at=1.0),
        replicate_rule="pooled",
    ),
    "BayesianBootstrap": lambda: BayesianBootstrap(replicates=2, block_length=1),
    "ParametricBootstrap": lambda: ParametricBootstrap(replicates=2, starts=2),
    "ResidualBootstrap": lambda: ResidualBootstrap(
        replicates=2, block_length=1, residual_scale="additive", scheme="moving"
    ),
    "WildBootstrap": lambda: WildBootstrap(
        replicates=2, block_length=1, multipliers="rademacher"
    ),
    "BlockJackknife": lambda: BlockJackknife(block_length=1),
    "NoNoise": NoNoise,
    "ResidualNoise": lambda: ResidualNoise(
        fit=FIT, block_length=1, residual_scale="additive"
    ),
    "DistributionNoise": DistributionNoise,
    "InformationMatchingEta": lambda: InformationMatchingEta(block_length=1),
    "BootstrapCoverageEta": lambda: BootstrapCoverageEta(
        replicates=2, block_length=1, confidence_level=0.9
    ),
    "LossDistance": LossDistance,
    "SeriesDistance": lambda: SeriesDistance(metric="mae"),
    "GibbsPosterior": lambda: GibbsPosterior(
        priors=PRIORS,
        eta=InformationMatchingEta(block_length=2),
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
        distance=SeriesDistance(metric="rmse"),
        particles=10,
        target_cess=0.5,
        min_acceptance=0.1,
        stay_probability=0.5,
    ),
    "ABCRejection": lambda: ABCRejection(
        priors=PRIORS, distance=LossDistance(), draws=10, quantile=0.5
    ),
    "Laplace": lambda: Laplace(draws=10),
    "Sandwich": lambda: Sandwich(block_length=1, draws=10),
    "LikelihoodTarget": lambda: LikelihoodTarget(priors=PRIORS),
    "GibbsTarget": lambda: GibbsTarget(priors=PRIORS, eta=1.0),
    "LaplaceMixture": lambda: LaplaceMixture(
        draws=25, target=GibbsTarget(priors=PRIORS, eta=1.0)
    ),
    "SplitConformal": lambda: SplitConformal(
        **CONFORMAL, calibration_data=[POINT], positions=[1.0], sample=SAMPLE
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
    "Threshold": lambda: Threshold.bootstrap_calibrated(
        ResidualBootstrap(
            replicates=2, block_length=1, residual_scale="additive", scheme="moving"
        ),
        0.95,
    ),
    "PredictionSpec": lambda: PredictionSpec(at=1.0, output="B"),
    "ProfileLikelihood": lambda: ProfileLikelihood(
        parameters=["c"], threshold=THRESHOLD
    ),
    "PredictionProfile": lambda: PredictionProfile(
        specs=[PredictionSpec(at=1.0, output="B")], threshold=THRESHOLD
    ),
    "StructuralRank": lambda: StructuralRank(points=3, design="dense"),
    "Collinearity": lambda: Collinearity(at=FIT, max_subset_size=2, threshold=20.0),
    "SampleIdentifiability": lambda: SampleIdentifiability(
        sample=SAMPLE, reference="prior", priors=PRIORS, correlation_threshold=0.9
    ),
    "StartsIdentifiability": lambda: StartsIdentifiability(
        result=FIT, parameter_tolerance=0.1, loss_tolerance=0.1
    ),
    "InputSpace": lambda: InputSpace(
        parameters=[PARAMETER], distributions={"c": Beta(a=2.0, b=2.0)}
    ),
    "Propagation": lambda: Propagation(draws=4, quantities=QUANTITIES),
    "PRCC": lambda: PRCC(quantities=QUANTITIES),
    "Morris": lambda: Morris(trajectories=2, levels=4, quantities=QUANTITIES),
    "Sobol": lambda: Sobol(
        draws=2, replicates=2, confidence_level=0.9, quantities=QUANTITIES
    ),
    "PolynomialChaos": lambda: PolynomialChaos(
        degree=1, draws=2, quantities=QUANTITIES
    ),
}
"""A valid object of each class whose JSON the engine reads, by class name."""
