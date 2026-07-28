"""Probabilistic calibration configuration and results."""

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass
class CalibrationEvaluation:
    """A single calibration evaluation result.

    This dataclass represents a parameter set evaluation from calibration,
    including the parameters, loss value, and optionally predictions.

    Attributes
    ----------
    parameters : list[float]
        Parameter values for this evaluation
    loss : float
        Loss/objective function value
    parameter_names : list[str]
        Names of the parameters (in same order as parameters list)
    predictions : list[list[float]] | None
        Optional predictions matrix with shape (time_steps, compartments)
    """

    parameters: list[float]
    loss: float
    parameter_names: list[str]
    predictions: list[list[float]] | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert to a dictionary mapping parameter names to values.

        Returns
        -------
        dict[str, float]
            Dictionary with parameter names as keys and values as values
        """
        return {name: self.parameters[i] for i, name in enumerate(self.parameter_names)}


class EnsembleCandidate(BaseModel):
    """A calibrated candidate available to final ensemble selection."""

    parameters: dict[str, float]
    loss: float


class ProbEvaluationFilterConfig(BaseModel):
    """
    Configuration for processing calibration evaluations.

    Controls deduplication and filtering of parameter sets from calibration runs
    before clustering and ensemble selection.

    Attributes
    ----------
    deduplication_tolerance : float
        Absolute tolerance for identifying duplicate parameter sets (default: 1e-6).
        Parameter sets within this tolerance are considered identical.
    loss_percentile_filter : float
        Fraction (0.0, 1.0] of best solutions by loss to retain (default: 1.0).
        For example, 0.1 keeps only the best 10% of evaluations, filtering out
        poor-quality solutions that would widen confidence intervals.
    min_evaluations_required : int
        Minimum number of unique evaluations required for analysis (default: 5).
        Calibration fails if fewer unique evaluations remain after deduplication.
    evaluation_retention : Literal["all", "best_per_run", "top_k_per_run"]
        Amount of optimizer history retained from each run before crossing the
        Python/Rust boundary.
    top_k_per_run : int | None
        Number of evaluations retained per run when evaluation_retention is
        "top_k_per_run".
    max_loss_ratio : float | None
        Optional maximum loss relative to the best retained evaluation. Set this
        to exclude optimizer states that are too poor to represent calibrated
        uncertainty.
    tail_max_loss_ratio : float | None
        Optional broader maximum loss ratio for prediction-novel tail candidates.
        When enabled with ``tail_max_representatives``, evaluations beyond
        ``max_loss_ratio`` but within this wider band may be admitted only if
        they add observed-prediction diversity.
    tail_max_representatives : int
        Maximum number of additional prediction-novel tail representatives.
    """

    deduplication_tolerance: float = Field(
        default=1e-6, gt=0.0, description="Tolerance for parameter deduplication"
    )
    loss_percentile_filter: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction (0.0, 1.0] of best solutions (by loss) to keep before clustering"
        ),
    )
    min_evaluations_required: int = Field(
        default=5,
        ge=1,
        description="Minimum number of unique evaluations required for analysis",
    )
    evaluation_retention: Literal["all", "best_per_run", "top_k_per_run"] = Field(
        default="all",
        description="Optimizer evaluation history retention mode",
    )
    top_k_per_run: int | None = Field(
        default=None,
        ge=1,
        description="Number of evaluations to retain per run for top_k_per_run mode",
    )
    model_config = ConfigDict(extra="forbid")

    max_loss_ratio: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Optional maximum loss relative to the best retained evaluation. "
            "When set, only statistically/plausibly near-optimal evaluations are "
            "allowed into clustering and ensemble selection."
        ),
    )
    tail_max_loss_ratio: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Optional broader maximum loss ratio for prediction-novel tail "
            "candidates. When set with tail_max_representatives > 0, candidates "
            "outside max_loss_ratio but inside this band can be added after "
            "representative selection if they add observed-prediction diversity."
        ),
    )
    tail_max_representatives: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum number of additional prediction-novel tail representatives "
            "to add from tail_max_loss_ratio."
        ),
    )

    @model_validator(mode="after")
    def validate_evaluation_retention(self) -> Self:
        """Validate evaluation retention configuration."""
        if self.evaluation_retention == "top_k_per_run" and self.top_k_per_run is None:
            raise ValueError(
                "top_k_per_run must be specified when "
                "evaluation_retention='top_k_per_run'"
            )
        if (
            self.evaluation_retention != "top_k_per_run"
            and self.top_k_per_run is not None
        ):
            raise ValueError(
                "top_k_per_run should only be set when "
                "evaluation_retention='top_k_per_run'"
            )
        if self.tail_max_representatives > 0 and self.tail_max_loss_ratio is None:
            raise ValueError(
                "tail_max_loss_ratio must be specified when "
                "tail_max_representatives > 0"
            )
        if self.tail_max_representatives > 0 and self.max_loss_ratio is None:
            raise ValueError(
                "max_loss_ratio must be specified when tail_max_representatives > 0"
            )
        if (
            self.max_loss_ratio is not None
            and self.tail_max_loss_ratio is not None
            and self.tail_max_loss_ratio < self.max_loss_ratio
        ):
            raise ValueError(
                "tail_max_loss_ratio must be greater than or equal to max_loss_ratio"
            )
        return self


class ProbClusteringConfig(BaseModel):
    """
    Configuration for clustering calibration candidates.

    Clusters parameter vectors or transformed observed predictions to identify
    distinct calibrated behaviours and enable diverse representative selection.

    Attributes
    ----------
    n_clusters : int | None
        Number of clusters to use (default: None for automatic determination).
        If None, optimal number is found using silhouette analysis.
    min_evaluations_for_clustering : int
        Minimum evaluations needed to perform clustering (default: 10).
        Below this threshold, a single cluster is used.
    silhouette_threshold : float
        Minimum silhouette score for beneficial clustering (default: 0.2).
        Scores range from -1 to 1; values near 0 indicate overlapping clusters.
    silhouette_excellent_threshold : float
        Early stopping threshold for silhouette search (default: 0.7).
        Search stops if a score above this is found.
    identical_solutions_atol : float
        Absolute tolerance for detecting identical solutions (default: 1e-10).
        Used to detect when there's no variance in parameter space.
    kmeans_max_iter : int
        Maximum iterations for K-means clustering (default: 100).
    kmeans_algorithm : Literal["lloyd", "elkan", "auto", "full"]
        K-means algorithm variant (default: "elkan", faster for dense data).
    max_k : int
        Maximum K tested during automatic silhouette search.
    silhouette_sample_size : int | None
        Deterministic sample size used for silhouette scoring when the
        evaluation set is large.
    minibatch_kmeans_threshold : int | None
        Use MiniBatchKMeans at or above this evaluation count.
    """

    model_config = ConfigDict(extra="forbid")

    n_clusters: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of clusters (None for automatic determination using silhouette)"
        ),
    )
    min_evaluations_for_clustering: int = Field(
        default=100,
        ge=1,
        description="Minimum evaluations required for clustering analysis",
    )
    silhouette_threshold: float = Field(
        default=0.2,
        ge=-1.0,
        le=1.0,
        description="Silhouette score threshold for beneficial clustering",
    )
    silhouette_excellent_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Early stopping threshold for silhouette score search",
    )
    identical_solutions_atol: float = Field(
        default=1e-10,
        ge=0.0,
        description="Tolerance for detecting identical solutions",
    )
    kmeans_max_iter: int = Field(
        default=100, ge=1, description="Maximum iterations for K-means clustering"
    )
    kmeans_algorithm: Literal["lloyd", "elkan", "auto", "full"] = Field(
        default="elkan", description="K-means algorithm variant"
    )
    max_k: int = Field(
        default=10,
        ge=2,
        description="Maximum K to test during automatic silhouette search",
    )
    silhouette_sample_size: int | None = Field(
        default=None,
        ge=2,
        description="Sample size for silhouette scoring",
    )
    minibatch_kmeans_threshold: int | None = Field(
        default=None,
        ge=1,
        description="Evaluation count threshold for MiniBatchKMeans",
    )
    feature_space: Literal["parameters", "observed_predictions"] = Field(
        default="parameters",
        description=(
            "Feature space used for clustering. observed_predictions clusters "
            "the transformed quantities that are actually calibrated, preserving "
            "distinct predicted shapes rather than merely distinct parameter values."
        ),
    )


class ProbRepresentativeConfig(BaseModel):
    """
    Configuration for selecting representative parameter sets from clusters.

    Controls how diverse parameter sets are selected from each cluster for
    use in ensemble selection.

    Attributes
    ----------
    max_representatives : int
        Maximum total representatives across all clusters (default: 1500).
        These become candidates for ensemble selection.
    percentage_elite_cluster_selection : float
        Fraction [0.0, 1.0] of best solutions by loss to include from each
        cluster before diversity selection (default: 0.1).
        0.0 = only diversity, 1.0 = only quality.
    cluster_representative_strategy : Literal["proportional", "equal"]
        How to distribute representatives across clusters (default: "proportional").
        "proportional": allocate proportionally to cluster size.
        "equal": allocate equally to all clusters.
    cluster_selection_method : Literal["crowding_distance", "maximin_distance",
        "latin_hypercube"]
        Method for selecting diverse representatives (default: "crowding_distance").
        "crowding_distance": NSGA-II style, explores boundaries.
        "maximin_distance": uniform coverage, no boundary bias.
        "latin_hypercube": stratified space-filling selection.
    quality_temperature : float
        Temperature for quality weighting in maximin_distance (default: 1.0).
        Higher = more diversity, lower = stronger quality bias.
        Only used with cluster_selection_method="maximin_distance".
    k_neighbors_min : int
        Minimum k for k-nearest neighbors in density estimation (default: 5).
    k_neighbors_max : int
        Maximum k for k-nearest neighbors in density estimation (default: 10).
    sparsity_weight : float
        Exponential weight for sparsity bonus in maximin selection (default: 2.0).
        Higher values = stronger preference for sparse regions.
    stratum_fit_weight : float
        Weight for stratum fit vs quality in latin_hypercube (default: 10.0).
        Higher values prioritize space-filling over quality.
    """

    model_config = ConfigDict(extra="forbid")

    max_representatives: int = Field(
        default=1000,
        gt=0,
        description="Maximum total representatives for ensemble selection",
    )
    percentage_elite_cluster_selection: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of best solutions to include from each cluster",
    )
    cluster_representative_strategy: Literal["proportional", "equal"] = Field(
        default="proportional",
        description="Strategy for distributing representatives across clusters",
    )
    cluster_selection_method: Literal[
        "latin_hypercube", "crowding_distance", "maximin_distance"
    ] = Field(
        default="latin_hypercube",
        description="Method for selecting diverse representatives within clusters",
    )
    quality_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature for quality weighting (maximin_distance only)",
    )
    k_neighbors_min: int = Field(
        default=5, ge=1, description="Minimum k for density estimation"
    )
    k_neighbors_max: int = Field(
        default=10, ge=1, description="Maximum k for density estimation"
    )
    sparsity_weight: float = Field(
        default=2.0,
        gt=0.0,
        description="Exponential weight for sparsity bonus in maximin selection",
    )
    stratum_fit_weight: float = Field(
        default=10.0,
        gt=0.0,
        description="Weight for stratum fit vs quality in latin_hypercube",
    )


class ProbEnsembleSizeConfig(BaseModel):
    """Common ensemble-size configuration shared by selection algorithms.

    Attributes
    ----------
    ensemble_size_mode : Literal["fixed", "bounded", "automatic"]
        Mode for determining ensemble size (default: "automatic").
    ensemble_size : int | None
        Fixed ensemble size (required if mode="fixed").
    ensemble_size_min : int | None
        Minimum ensemble size (required if mode="bounded").
    ensemble_size_max : int | None
        Maximum ensemble size (required if mode="bounded").
    """

    model_config = ConfigDict(extra="forbid")

    ensemble_size_mode: Literal["fixed", "bounded", "automatic"] = Field(
        default="automatic", description="Mode for determining ensemble size"
    )
    ensemble_size: int | None = Field(
        default=None, ge=2, description="Fixed ensemble size (for mode='fixed')"
    )
    ensemble_size_min: int | None = Field(
        default=None, ge=2, description="Minimum ensemble size (for mode='bounded')"
    )
    ensemble_size_max: int | None = Field(
        default=None, ge=2, description="Maximum ensemble size (for mode='bounded')"
    )

    @model_validator(mode="after")
    def validate_ensemble_size_config(self) -> Self:
        """Validate ensemble size configuration."""
        if self.ensemble_size_mode == "fixed":
            self._validate_fixed_mode()
        elif self.ensemble_size_mode == "bounded":
            self._validate_bounded_mode()
        elif self.ensemble_size_mode == "automatic":
            self._validate_automatic_mode()
        else:
            raise ValueError(
                f"Invalid ensemble_size_mode: '{self.ensemble_size_mode}'. "
                "Must be one of: automatic, bounded, fixed"
            )
        return self

    def _validate_fixed_mode(self) -> None:
        """Validate fixed ensemble size mode."""
        if self.ensemble_size is None:
            raise ValueError(
                "ensemble_size must be specified when ensemble_size_mode='fixed'"
            )
        if self.ensemble_size_min is not None or self.ensemble_size_max is not None:
            raise ValueError(
                "ensemble_size_min and ensemble_size_max should not be set "
                "when ensemble_size_mode='fixed'"
            )

    def _validate_bounded_mode(self) -> None:
        """Validate bounded ensemble size mode."""
        if self.ensemble_size_min is None:
            raise ValueError(
                "ensemble_size_min must be specified when ensemble_size_mode='bounded'"
            )
        if self.ensemble_size_max is None:
            raise ValueError(
                "ensemble_size_max must be specified when ensemble_size_mode='bounded'"
            )
        if self.ensemble_size_max < self.ensemble_size_min:
            raise ValueError(
                f"ensemble_size_max ({self.ensemble_size_max}) must be >= "
                f"ensemble_size_min ({self.ensemble_size_min})"
            )

    def _validate_automatic_mode(self) -> None:
        """Validate automatic ensemble size mode."""
        if self.ensemble_size_min is not None or self.ensemble_size_max is not None:
            raise ValueError(
                "ensemble_size_min and ensemble_size_max should not be set "
                "when ensemble_size_mode='automatic'"
            )


class ProbNsga2Config(ProbEnsembleSizeConfig):
    """Configuration for the NSGA-II ensemble-selection algorithm.

    This is the default selection configuration. It balances compact
    observation-space interval width and observed-data coverage.
    """

    population_size: int = Field(
        default=100, gt=3, description="NSGA-II population size"
    )
    generations: int = Field(
        default=100, gt=0, description="Number of NSGA-II generations"
    )
    crossover_probability: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="NSGA-II crossover probability",
    )
    pareto_preference: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="NSGA-II preference for narrow intervals versus coverage",
    )


class ProbGreedyLocalSearchConfig(ProbEnsembleSizeConfig):
    """Configuration for the fit-gated greedy local-search algorithm."""

    central_fit_max_loss_ratio: float = Field(
        default=1.25,
        ge=1.0,
        description=(
            "Maximum central loss (the ensemble median scored with the problem's "
            "loss function) relative to the best selected member. The ensemble is "
            "rejected when it exceeds this limit."
        ),
    )
    search_beam_width: int = Field(
        default=32,
        ge=2,
        description=(
            "Partial ensembles retained per size during fit-gated beam search. "
            "Larger values improve combinatorial coverage at higher cost."
        ),
    )


class ProbabilisticCalibrationConfig(BaseModel):
    """
    Unified configuration for probabilistic calibration.

    This configuration groups all settings for the probabilistic calibration
    workflow, which finds an ensemble of parameter sets with uncertainty
    quantification instead of a single optimal solution.

    The workflow consists of:
    1. Run: Multiple independent calibration runs
    2. Evaluation Processing: Deduplication and filtering
    3. Clustering: Group similar solutions
    4. Representative Selection: Pick diverse solutions from clusters
    5. Ensemble Selection: ensemble subset selection
    6. Statistics: Calculate confidence intervals and coverage

    Attributes
    ----------
    n_runs : int
        Number of independent calibration runs to perform (default: 10).
        More runs provide better parameter space exploration but take longer.
    evaluation_processing : ProbEvaluationFilterConfig
        Configuration for evaluation deduplication and filtering
    clustering : ProbClusteringConfig
        Configuration for clustering calibration candidates
    representative_selection : ProbRepresentativeConfig
        Configuration for selecting representatives from clusters
    ensemble_selection : ProbNsga2Config | ProbGreedyLocalSearchConfig
        Configuration for the selected ensemble algorithm. NSGA-II is the
        default; use the greedy configuration to select the alternate backend.
    confidence_level : float
        Confidence interval level (default: 0.95 for 95% CI).
        Must be in range (0.0, 1.0).
    """

    model_config = ConfigDict(extra="forbid")

    n_runs: int = Field(
        default=10, gt=0, description="Number of calibration runs to perform"
    )
    evaluation_processing: ProbEvaluationFilterConfig = Field(
        default_factory=ProbEvaluationFilterConfig,
        description="Configuration for evaluation processing",
    )
    clustering: ProbClusteringConfig = Field(
        default_factory=ProbClusteringConfig,
        description="Configuration for clustering calibration candidates",
    )
    representative_selection: ProbRepresentativeConfig = Field(
        default_factory=ProbRepresentativeConfig,
        description="Configuration for representative selection",
    )
    ensemble_selection: ProbNsga2Config | ProbGreedyLocalSearchConfig = Field(
        default_factory=ProbNsga2Config,
        description="Configuration for ensemble selection",
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence interval level (e.g., 0.95 for 95% CI)",
    )
    include_ensemble_candidates: bool = Field(
        default=False,
        description=(
            "Include the parameter sets and losses available to final ensemble "
            "selection in the calibration result."
        ),
    )


class ParameterSetStatistics(BaseModel):
    """
    Statistics for a parameter across the ensemble.

    Attributes
    ----------
    mean : float
        Mean value across ensemble
    median : float
        Median value across ensemble
    std : float
        Standard deviation across ensemble
    percentile_lower : float
        Lower percentile bound (e.g., 2.5th for 95% CI)
    percentile_upper : float
        Upper percentile bound (e.g., 97.5th for 95% CI)
    min : float
        Minimum value in ensemble
    max : float
        Maximum value in ensemble
    """

    mean: float = Field(description="Mean value across ensemble")
    median: float = Field(description="Median value across ensemble")
    std: float = Field(description="Standard deviation across ensemble")
    percentile_lower: float = Field(
        description="Lower percentile bound of confidence interval"
    )
    percentile_upper: float = Field(
        description="Upper percentile bound of confidence interval"
    )
    min: float = Field(description="Minimum value in ensemble")
    max: float = Field(description="Maximum value in ensemble")


class EnsembleSelectionSummary(BaseModel):
    """Compact summary of one candidate ensemble from NSGA-II."""

    ensemble_size: int = Field(description="Number of selected parameter sets")
    selected_indices: list[int] = Field(description="Selected candidate indices")
    ci_width: float = Field(description="Standardized confidence interval width")
    coverage: float = Field(description="Observed-point coverage")
    central_loss: float = Field(description="Central median-prediction loss")


class EnsembleSolution(BaseModel):
    """
    A complete ensemble solution with statistics and predictions.

    Represents a single ensemble of parameter sets, containing the ensemble
    composition, parameter statistics, model predictions with confidence
    intervals, and performance metrics.

    Attributes
    ----------
    ensemble_size : int
        Number of parameter sets in this ensemble
    selected_indices : list[int]
        Indices of parameter sets selected for this ensemble
    ensemble_parameters : list[dict[str, float]]
        List of parameter dictionaries in this ensemble
    parameter_statistics : dict[str, ParameterSetStatistics]
        Statistics for each parameter across the ensemble
    prediction_median : dict[str, list[float]]
        Median predictions for each compartment over time
    prediction_ci_lower : dict[str, list[float]]
        Lower bound of confidence interval for each compartment over time
    prediction_ci_upper : dict[str, list[float]]
        Upper bound of confidence interval for each compartment over time
    coverage_percentage : float
        Percentage of observed data points within the confidence intervals
    average_ci_width : float
        Average width of confidence intervals across time and compartments
    ci_width : float
        Mean observed-data interval width normalized by observation magnitude.
    coverage : float
        Normalized coverage objective [0, 1] used in optimization
    point_parameters : dict[str, float]
        Parameters of the lowest-loss selected member. This is the only
        parameter dictionary suitable for constructing a single point model.
    point_loss : float
        Calibration loss of point_parameters.
    central_loss : float
        The memberwise-median prediction scored with the problem's loss function
        (the same loss the members were fit with), comparable to point_loss.
    observation_diagnostics : dict[str, dict[str, float]]
        Per-observation-series coverage and interval-width diagnostics.
    selection_diagnostics : dict[str, float]
        Candidate-search diagnostics produced by the selected backend.
    """

    ensemble_size: int = Field(description="Number of parameter sets in this ensemble")
    selected_indices: list[int] = Field(
        description="Indices of selected parameter sets"
    )
    ensemble_parameters: list[dict[str, float]] = Field(
        description="List of parameter dictionaries in this ensemble"
    )
    parameter_statistics: dict[str, ParameterSetStatistics] = Field(
        description="Statistics for each parameter across the ensemble"
    )
    prediction_median: dict[str, list[float]] = Field(
        description="Median predictions for each compartment over time"
    )
    prediction_ci_lower: dict[str, list[float]] = Field(
        description="Lower bound of confidence interval for each compartment over time"
    )
    prediction_ci_upper: dict[str, list[float]] = Field(
        description="Upper bound of confidence interval for each compartment over time"
    )
    windowed_prediction_median: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Correct windowed median for outputs with windowed observations",
    )
    windowed_prediction_ci_lower: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Correct windowed lower CI for outputs with windowed observations",
    )
    windowed_prediction_ci_upper: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Correct windowed upper CI for outputs with windowed observations",
    )
    coverage_percentage: float = Field(
        description="Percentage of observed data points within confidence intervals"
    )
    average_ci_width: float = Field(description="Average width of confidence intervals")
    ci_width: float = Field(
        description=(
            "Mean observed-data interval width normalized by observation magnitude"
        )
    )
    coverage: float = Field(description="Normalized coverage objective [0, 1]")
    point_parameters: dict[str, float] = Field(
        description="Parameters of the lowest-loss selected ensemble member"
    )
    point_loss: float = Field(description="Loss of the selected point member")
    central_loss: float = Field(
        description=(
            "Memberwise-median prediction scored with the problem's loss function"
        )
    )
    observation_diagnostics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Coverage and interval diagnostics grouped by observation series",
    )
    selection_diagnostics: dict[str, float] = Field(
        default_factory=dict,
        description="Ensemble selection diagnostics",
    )


class ProbabilisticCalibrationResult(BaseModel):
    """
    Complete result from probabilistic calibration with ensemble analysis.

    Contains the selected ensemble solution and metadata about the calibration
    and selection process.

    Attributes
    ----------
    selected_ensemble : EnsembleSolution
        The selected ensemble solution.
    n_runs_performed : int
        Number of calibration runs performed
    n_unique_evaluations : int
        Number of unique parameter evaluations after deduplication
    n_clusters_used : int
        Number of clusters identified in parameter space
    confidence_level : float
        Confidence level used for interval calculation (e.g., 0.95 for 95% CI)
    stage_timings : dict[str, float]
        Wall-clock timings in seconds for major probabilistic calibration stages.
    stage_counts : dict[str, int]
        Counts associated with major stages, useful for performance diagnostics.
    """

    selected_ensemble: EnsembleSolution = Field(
        description="The selected ensemble solution"
    )
    selection_algorithm: Literal["nsga2", "greedy_local_search"] = Field(
        default="nsga2", description="Algorithm used for ensemble subset selection"
    )
    pareto_front: list[EnsembleSelectionSummary] | None = Field(
        default=None,
        description="Compact NSGA-II Pareto-front summaries, when available",
    )
    selected_pareto_index: int | None = Field(
        default=None,
        description="Selected index in pareto_front for NSGA-II",
    )
    n_runs_performed: int = Field(description="Number of calibration runs performed")
    n_unique_evaluations: int = Field(
        description="Number of unique evaluations after deduplication"
    )
    n_clusters_used: int = Field(description="Number of clusters identified")
    confidence_level: float = Field(
        description="Confidence level used (e.g., 0.95 for 95% CI)"
    )
    stage_timings: dict[str, float] = Field(
        default_factory=dict,
        description="Wall-clock timings in seconds for major calibration stages",
    )
    stage_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts associated with major calibration stages",
    )
    ensemble_candidates: list[EnsembleCandidate] | None = Field(
        default=None,
        description=(
            "Calibrated candidates passed to final ensemble selection, when "
            "include_ensemble_candidates is enabled."
        ),
    )
