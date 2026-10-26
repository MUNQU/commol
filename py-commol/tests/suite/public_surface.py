"""The public surface of commol: the names a user writes, and the results that
public objects return.

`tests/suite/test_architecture.py` checks the exports against `EXPORTS`, and
`tests/docs/test_reference.py` checks that each name of `RESULTS` is
documented on the page of an object that returns it. A name a user only
reads, passes as a string or never builds is not exported: a result is
reached from its object, a choice among names is a `Literal`, and a union, a
base or a part that the builder makes stays private.
"""

ROOT = frozenset(
    {
        # Build
        "ModelBuilder",
        "Model",
        "StratificationCondition",
        "StratifiedRate",
        "SeriesValue",
        "AxisUnits",
        "AxisPattern",
        # Structures and neighborhoods
        "Aggregate",
        "Complete",
        "Groups",
        # Simulate
        "Simulation",
        "InfeasibleStateError",
        # Calibrate
        "Calibrator",
        "CalibrationProblem",
        "CalibrationParameter",
        "CalibrationConstraint",
        "ObservedDataPoint",
        "NelderMeadConfig",
        "ParticleSwarmConfig",
        "Quantile",
        # Package
        "build_info",
    }
)
"""The main flow: build, the structures, simulate and calibrate."""

PAGES = {
    "engines": {"ODESolver", "StochasticEventsSettings", "StochasticStepsSettings"},
    "plotting": {"plot_compared", "PlotConfig"},
    "parallelism": {"set_threads", "threads", "get_threads", "get_levels"},
    "distributions": {
        "Uniform",
        "LogUniform",
        "Normal",
        "LogNormal",
        "Beta",
        "Gamma",
        "Priors",
    },
    "quantities": {
        "Level",
        "Increments",
        "ValueAt",
        "Gain",
        "Peak",
        "PeakPosition",
        "Trajectory",
        "Ratio",
        "quantity_schedule",
    },
    "observations": {
        "NormalObservations",
        "LogNormalObservations",
        "StudentTObservations",
        "PoissonObservations",
        "NegativeBinomialObservations",
        "BinomialObservations",
        "NoNoise",
        "ResidualNoise",
        "DistributionNoise",
    },
    "uncertainty": {
        "ParameterSample",
        "SampleMethod",
        "Predictive",
        "Scenarios",
        "weighted_quantiles",
        "member_statistics",
    },
    "resampling": {
        "BayesianBootstrap",
        "ParametricBootstrap",
        "ResidualBootstrap",
        "WildBootstrap",
        "BlockJackknife",
    },
    "profiles": {
        "ProfileLikelihood",
        "PredictionProfile",
        "PredictionSpec",
        "Threshold",
    },
    "local": {
        "Laplace",
        "Sandwich",
        "LaplaceMixture",
        "LikelihoodTarget",
        "GibbsTarget",
    },
    "smc": {
        "GibbsPosterior",
        "BayesianPosterior",
        "ABCSMC",
        "ABCRejection",
        "InformationMatchingEta",
        "BootstrapCoverageEta",
        "LossDistance",
        "SeriesDistance",
    },
    "conformal": {
        "SplitConformal",
        "RollingConformal",
        "JackknifePlus",
        "EnbPI",
        "AdaptiveConformal",
    },
    "identifiability": {
        "StructuralRank",
        "Collinearity",
        "SampleIdentifiability",
        "StartsIdentifiability",
        "ProfileIdentifiability",
    },
    "sensitivity": {
        "SensitivityAnalysis",
        "InputSpace",
        "Propagation",
        "PRCC",
        "Morris",
        "Sobol",
        "PolynomialChaos",
    },
    "scores": {
        "coverage",
        "crps",
        "interval_score",
        "log_score",
        "pit",
        "weighted_interval_score",
    },
}
"""The other public names, by the page of the guide that documents them: each
page is the module ``commol.<page>``."""

EXPORTS = {
    "commol": ROOT,
    **{f"commol.{page}": frozenset(names) for page, names in PAGES.items()},
}
"""The ``__all__`` of each public module of commol: the root, and one module
per page of the guide."""

RESULTS = frozenset(
    {
        "SimulationResults",
        "SimulationReplicates",
        "ReplicateRuns",
        "Process",
        "CalibrationResult",
        "CalibrationEvaluation",
        "CalibrationHistory",
        "StartSummary",
        "ModelDescription",
        "Section",
        "StateGraph",
        "BuildInfo",
        "ResultsPlot",
        "ReplicatesPlot",
        "PredictivePlot",
        "ParameterSummary",
        "PredictiveBand",
        "SeriesBand",
        "ObservationDraws",
        "ScenarioSummary",
        "Score",
        "BootstrapResult",
        "JackknifeResult",
        "ResamplingDiagnostics",
        "BoundError",
        "ProfileResult",
        "Profile",
        "ProfileInterval",
        "ProfileStructure",
        "SMCResult",
        "GibbsResult",
        "SMCDiagnostics",
        "LaplaceResult",
        "SandwichResult",
        "DeltaBand",
        "ImportanceResult",
        "MixtureComponent",
        "ConformalResult",
        "PredictionIntervals",
        "SeriesIntervals",
        "IdentifiabilityReport",
        "ParameterVerdicts",
        "StructuralSection",
        "CollinearitySection",
        "SampleSection",
        "StartsSection",
        "ProfileSection",
        "PropagationResult",
        "QuantitySummary",
        "PRCCResult",
        "MorrisResult",
        "SobolResult",
        "ChaosResult",
    }
)
"""The results that public objects return: each is typed as the return of the
member that gives it, and documented on the page of that object."""
