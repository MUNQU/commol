"""The public surface of commol: the names a user writes in a call (`ROOT` and
`PAGES`), and the results that public members return (`RESULTS`), which a
typed caller writes in an annotation.

`tests/suite/test_architecture.py` checks the exports against `EXPORTS` and
that every class a public member returns is exported, and
`tests/docs/test_reference.py` checks that each result is documented under its
public path on the page of an object that returns it. A result is exported so
that a typed caller can name it, not so that a user builds it. A choice among
names is a `Literal`, and a union or a base that no member returns stays
private.
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
        "UnitConsistencyError",
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
"""The main flow that a user writes: build, the structures, simulate and
calibrate."""

PAGES = {
    "engines": {
        "ODESolver",
        "StochasticEventsSettings",
        "StochasticStepsSettings",
    },
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

RESULTS = {
    "commol": frozenset(
        {
            # The model and its parts
            "Process",
            "Bin",
            "Accumulator",
            "Stratification",
            "CategoryMatrix",
            "Parameter",
            "Transition",
            "InitialCompartment",
            "SeriesLabel",
            "ReportSections",
            "AppliedParametersReport",
            "ParameterEntry",
            "EntrySpread",
            "InitialAmountEntry",
            "CategorySplitEntry",
            # Simulate
            "SimulationResults",
            "SimulationReplicates",
            "ReplicateRuns",
            "ResultsPlot",
            "ReplicatesPlot",
            "SimulationReport",
            "ObservedSeries",
            "AccumulatorReport",
            "WindowedAccumulator",
            "PredictionIntervalsReport",
            "SeriesIntervalsReport",
            # Calibrate
            "CalibrationResult",
            "StartSummary",
            "CalibrationHistory",
            "CalibrationEvaluation",
            # The package
            "BuildInfo",
        }
    ),
    "commol.engines": frozenset(
        {
            "ModelEngine",
        }
    ),
    "commol.uncertainty": frozenset(
        {
            "PredictivePlot",
            "ObservationDraws",
            "PredictiveBand",
            "SeriesBand",
            "ParameterSummary",
            "ScenarioSummary",
        }
    ),
    "commol.resampling": frozenset(
        {
            "BootstrapResult",
            "BoundError",
            "JackknifeResult",
            "ResamplingDiagnostics",
        }
    ),
    "commol.profiles": frozenset(
        {
            "Profile",
            "ProfileInterval",
            "ProfileResult",
            "ProfileStructure",
        }
    ),
    "commol.local": frozenset(
        {
            "DeltaBand",
            "ImportanceResult",
            "LaplaceResult",
            "MixtureComponent",
            "SandwichResult",
        }
    ),
    "commol.smc": frozenset(
        {
            "GibbsResult",
            "SMCDiagnostics",
            "SMCResult",
        }
    ),
    "commol.description": frozenset(
        {
            "ModelDescription",
            "Section",
            "Text",
            "Table",
            "Equations",
            "EquationGroup",
            "Equation",
            "StateGraph",
            "GraphNode",
            "GraphEdge",
            "Matrix",
        }
    ),
    "commol.conformal": frozenset(
        {
            "ConformalResult",
            "PredictionIntervals",
            "SeriesIntervals",
        }
    ),
    "commol.identifiability": frozenset(
        {
            "AccuracyData",
            "CollinearitySectionData",
            "CombinationData",
            "CorrelatedPairData",
            "IdentifiabilityReportData",
            "IdentifiabilitySections",
            "PointRankData",
            "PracticalVerdictEntry",
            "ProfileSectionData",
            "SampleSectionData",
            "StartsSectionData",
            "StructuralSectionData",
            "StructuralVerdictEntry",
            "SubsetIndexData",
            "TighterRankData",
            "VerdictsEntry",
            "Accuracy",
            "CollinearitySection",
            "Combination",
            "CorrelatedPair",
            "IdentifiabilityReport",
            "ParameterVerdicts",
            "PointRank",
            "PracticalVerdict",
            "ProfileSection",
            "SampleSection",
            "StartsSection",
            "StructuralSection",
            "StructuralVerdict",
            "SubsetIndex",
            "TighterRank",
        }
    ),
    "commol.sensitivity": frozenset(
        {
            "ChaosResult",
            "MorrisResult",
            "PRCCResult",
            "PropagationResult",
            "QuantitySummary",
            "SobolResult",
        }
    ),
    "commol.scores": frozenset(
        {
            "Score",
        }
    ),
}
"""The classes that public members return and that a user does not write in a
call, by the module that exports them: a typed caller names each one in an
annotation. A result of the main flow is in the root, and the result of a
method in the module of the method. Each is documented on the page of an
object that returns it."""

WRITTEN = {
    "commol": ROOT,
    **{f"commol.{page}": frozenset(names) for page, names in PAGES.items()},
}
"""The names a user writes in a call, by public module."""

EXPORTS = {
    module: WRITTEN.get(module, frozenset()) | RESULTS.get(module, frozenset())
    for module in {**WRITTEN, **RESULTS}
}
"""The ``__all__`` of each public module of commol: the root, and one module
per page of the guide, each with its results. A page whose names a user only
reads, such as ``commol.description``, has only results."""
