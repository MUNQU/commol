"""A result of each subclass of `ResultModel` on the public surface, by class
name: the return of a calibration, a method or a member of a result on a
small problem, so that a test reads back the values of a real run."""

from collections.abc import Callable

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    Simulation,
)
from commol._data.result_model import ResultModel
from commol.conformal import ConformalResult, JackknifePlus
from commol.distributions import Priors
from commol.engines import ODESolver
from commol.identifiability import (
    Collinearity,
    IdentifiabilityReport,
    ProfileIdentifiability,
    SampleIdentifiability,
    StartsIdentifiability,
    StructuralRank,
)
from commol.local import (
    GibbsTarget,
    ImportanceResult,
    Laplace,
    LaplaceMixture,
    LaplaceResult,
    Sandwich,
)
from commol.observations import NoNoise, NormalObservations
from commol.profiles import ProfileLikelihood, ProfileResult, Threshold
from commol.quantities import Peak, ValueAt
from commol.resampling import BayesianBootstrap, BlockJackknife, BootstrapResult
from commol.scores import interval_score
from commol.sensitivity import (
    PRCC,
    Morris,
    PolynomialChaos,
    Propagation,
    PropagationResult,
    SensitivityAnalysis,
    Sobol,
)
from commol._calibration.method import SensitivityMethod
from commol.smc import ABCSMC, GibbsPosterior, SeriesDistance, SMCResult
from commol.uncertainty import ParameterSample, Predictive, SampleMethod, Scenarios
from tests.support.models import a_to_b_builder, abc_builder, build
from tests.support.problems import (
    abc_rates,
    bounded,
    flow_calibrator,
    flow_observations,
    line_calibrator,
    rare_flow_calibrator,
)

FLOWS = [9.0, 11.0, 10.0, 12.0]
"""The flows of one step that `flow_calibrator` fits, at positions 1 to 4."""

LINE = [10.5, 12.0, 13.5, 15.0, 16.5]
"""The amounts of B that `line_calibrator` fits, at positions 1 to 5."""

QUANTITIES = [ValueAt(series="B", at=5), Peak(series="B", positions=[0, 5, 10])]
"""The quantities of the sensitivity methods and of a scenario."""

PRIORS = Priors.uniform_on_bounds()


def _flows() -> Calibrator:
    """`flow_calibrator` of `FLOWS`, which fits the sum of squares."""
    return flow_calibrator(FLOWS, windowed=True)


def _normal() -> Calibrator:
    """`flow_calibrator` of `FLOWS` with normal observations."""
    return flow_calibrator(
        FLOWS, windowed=True, observation_models={"B": NormalObservations(sd=1.0)}
    )


def _product() -> Calibrator:
    """`flow_calibrator` at the rate ``c * s * A / A``, which reads ``c`` and
    ``s`` only as their product, with normal observations."""
    model = build(
        a_to_b_builder(
            parameters={"c": None, "s": None},
            rate="c * s * A / A",
            amounts={"A": 1e6, "B": 0},
        )
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(FLOWS, windowed=True),
        parameters=[
            bounded("c", 1.0, 50.0, initial_guess=10.0),
            bounded("s", 0.5, 2.0, initial_guess=1.0),
        ],
        observation_models={"B": NormalObservations(sd=1.0)},
        optimization_config=NelderMeadConfig(max_iterations=500),
        seed=3,
    )
    return Calibrator(Simulation(model), problem)


def _bootstrap() -> BootstrapResult:
    return _flows().uncertainty(BayesianBootstrap(replicates=4, block_length=1))


def _laplace() -> LaplaceResult:
    return _normal().uncertainty(Laplace(draws=10))


def _mixture() -> ImportanceResult:
    return _flows().uncertainty(
        LaplaceMixture(draws=25, target=GibbsTarget(priors=PRIORS, eta=1.0))
    )


def _abc() -> SMCResult:
    return _flows().uncertainty(
        ABCSMC(
            priors=PRIORS,
            distance=SeriesDistance(metric="rmse"),
            particles=10,
            target_cess=0.5,
            min_acceptance=0.1,
            stay_probability=0.5,
        )
    )


def _conformal() -> ConformalResult:
    return _flows().uncertainty(
        JackknifePlus(
            score="absolute",
            by="series",
            confidence_level=0.9,
            block_length=1,
            positions=[1.0, 2.0],
        )
    )


def _profile(calibrator: Calibrator) -> ProfileResult:
    return calibrator.uncertainty(
        ProfileLikelihood(parameters=["c"], threshold=Threshold.chi_square(0.95))
    )


def _structural() -> IdentifiabilityReport:
    """The structural rank of `_product`: ``c`` and ``s`` are identifiable
    only in combination."""
    return _product().identifiability(StructuralRank(points=3, design="observations"))


def _tighter() -> IdentifiabilityReport:
    """A structural rank that the tolerance of the solver changes at one point."""
    return rare_flow_calibrator(ODESolver(rtol=1e-6, atol=1e-6)).identifiability(
        StructuralRank(points=3, design="observations")
    )


def _collinearity() -> IdentifiabilityReport:
    calibrator = line_calibrator(LINE)
    return calibrator.identifiability(
        Collinearity(at=calibrator.run(), max_subset_size=2, threshold=10.0)
    )


def _starts() -> IdentifiabilityReport:
    calibrator = line_calibrator(LINE)
    return calibrator.identifiability(
        StartsIdentifiability(
            result=calibrator.run(starts=3),
            parameter_tolerance=0.05,
            loss_tolerance=0.01,
        )
    )


def _correlated() -> IdentifiabilityReport:
    """`SampleIdentifiability` of a sample whose two parameters move together."""
    sample = ParameterSample.equally_weighted(
        ["c", "B"],
        [(0.0, 50.0), (0.0, 50.0)],
        [[1.0, 9.0], [1.5, 9.4], [2.0, 10.1], [2.5, 10.4]],
        SampleMethod(name="by hand", settings={"seed": 1}),
    )
    return line_calibrator(LINE).identifiability(
        SampleIdentifiability(
            sample=sample,
            reference="prior",
            priors=PRIORS,
            correlation_threshold=0.9,
        )
    )


def _profiled() -> IdentifiabilityReport:
    calibrator = _normal()
    return calibrator.identifiability(
        ProfileIdentifiability(result=_profile(calibrator))
    )


def _sensitivity[R](method: SensitivityMethod[R]) -> R:
    analysis = SensitivityAnalysis(
        Simulation(build(abc_builder())), abc_rates(), seed=7
    )
    return analysis.run(method)


def _propagation() -> PropagationResult:
    return _sensitivity(Propagation(draws=8, quantities=QUANTITIES))


def _predictive() -> Predictive:
    return Predictive(_flows(), _bootstrap().sample, [1.0, 2.0])


def _scenarios() -> Scenarios:
    """The sample of `_bootstrap` on `flow_calibrator` at the rate
    ``c * k * A / A``, with ``k`` at 1 and, in the scenario ``"faster"``,
    at 1.5."""
    model = build(
        a_to_b_builder(
            parameters={"c": None, "k": 1.0},
            rate="c * k * A / A",
            amounts={"A": 1e6, "B": 0},
        )
    )
    return Scenarios(
        Calibrator(Simulation(model), _flows().problem),
        _bootstrap().sample,
        {"fitted": {}, "faster": {"k": 1.5}},
        positions=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )


EXAMPLES_OF_RESULTS: dict[str, Callable[[], ResultModel | None]] = {
    # Calibrate
    "CalibrationResult": lambda: line_calibrator(LINE).run(starts=4),
    "StartSummary": lambda: line_calibrator(LINE).run(starts=4).starts[1],
    "CalibrationHistory": lambda: line_calibrator(LINE).run_with_history(),
    "CalibrationEvaluation": lambda: (
        line_calibrator(LINE).run_with_history().evaluations[2]
    ),
    # The sample, its predictive and its scenarios
    "ParameterSample": lambda: _bootstrap().sample,
    "SampleMethod": lambda: _bootstrap().sample.method,
    "ParameterSummary": lambda: _bootstrap().sample.summary(0.9)["c"],
    "PredictiveBand": lambda: _predictive().band(0.9),
    "SeriesBand": lambda: _predictive().series_band(["B"], 0.9, cumulative=True),
    "ScenarioSummary": lambda: _scenarios().summary("faster", QUANTITIES[:1], 0.9)[0],
    "Score": lambda: interval_score(
        _predictive(), flow_observations([9.0, 11.0], windowed=True), NoNoise(), 0.9
    ),
    # Resampling
    "BootstrapResult": _bootstrap,
    "ResamplingDiagnostics": lambda: _bootstrap().diagnostics,
    "BoundError": lambda: _bootstrap().interval_error(0.9)["c"],
    "JackknifeResult": lambda: _flows().uncertainty(BlockJackknife(block_length=1)),
    # Profiles
    "ProfileResult": lambda: _profile(_normal()),
    "Profile": lambda: _profile(_normal()).profiles["c"],
    "ProfileInterval": lambda: _profile(_normal()).intervals["c"],
    "ProfileStructure": lambda: _profile(_product()).structure["c"],
    # Local approximations
    "LaplaceResult": _laplace,
    "SandwichResult": lambda: _flows().uncertainty(Sandwich(block_length=1, draws=10)),
    "DeltaBand": lambda: _laplace().delta_band(_normal(), [1.0, 2.0], 0.9),
    "ImportanceResult": _mixture,
    "MixtureComponent": lambda: _mixture().components[0],
    # Sequential Monte Carlo
    "SMCResult": _abc,
    "SMCDiagnostics": lambda: _abc().diagnostics,
    "GibbsResult": lambda: _flows().uncertainty(
        GibbsPosterior(
            priors=PRIORS,
            eta=1.0,
            particles=10,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.5,
        )
    ),
    # Conformal
    "ConformalResult": _conformal,
    "PredictionIntervals": lambda: _conformal().intervals,
    "SeriesIntervals": lambda: _conformal().intervals.series["B"],
    # Identifiability
    "IdentifiabilityReport": _structural,
    "ParameterVerdicts": lambda: _structural().verdicts["c"],
    "StructuralVerdict": lambda: _structural().verdicts["c"].structural,
    "PracticalVerdict": lambda: _starts().verdicts["c"].practical,
    "StructuralSection": lambda: _structural().structural[0],
    "Combination": lambda: _structural().structural[0].combinations[0],
    "PointRank": lambda: _structural().structural[0].points[0],
    "Accuracy": lambda: _tighter().structural[0].points[0].accuracy,
    "TighterRank": lambda: _tighter().structural[0].points[0].tighter,
    "CollinearitySection": lambda: _collinearity().collinearity[0],
    "SubsetIndex": lambda: _collinearity().collinearity[0].best_by_size[0],
    "SampleSection": lambda: _correlated().sample[0],
    "CorrelatedPair": lambda: _correlated().sample[0].correlated_pairs[0],
    "StartsSection": lambda: _starts().starts[0],
    "ProfileSection": lambda: _profiled().profile[0],
    # Sensitivity
    "PropagationResult": _propagation,
    "QuantitySummary": lambda: _propagation().summary(0.9)[1],
    "PRCCResult": lambda: _sensitivity(PRCC(draws=12, quantities=QUANTITIES)),
    "MorrisResult": lambda: _sensitivity(
        Morris(trajectories=3, levels=4, quantities=QUANTITIES)
    ),
    "SobolResult": lambda: _sensitivity(
        Sobol(draws=8, replicates=4, confidence_level=0.9, quantities=QUANTITIES)
    ),
    "ChaosResult": lambda: _sensitivity(
        PolynomialChaos(degree=1, draws=8, quantities=QUANTITIES)
    ),
}
"""A result of each subclass of `ResultModel` on the public surface, by class
name. A field that may be ``None``, such as `TighterRank` of a point, gives
the result of a run where it is set; the test checks that it is."""
