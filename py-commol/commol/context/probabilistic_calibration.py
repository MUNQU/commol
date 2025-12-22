"""Probabilistic calibration configuration and results."""

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


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


class ProbClusteringConfig(BaseModel):
    """
    Configuration for clustering parameter space.

    Clusters similar parameter sets together to identify distinct solution regions
    and enable diverse representative selection.

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
    """

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


class ProbRepresentativeConfig(BaseModel):
    """
    Configuration for selecting representative parameter sets from clusters.

    Controls how diverse parameter sets are selected from each cluster for
    use in ensemble selection.

    Attributes
    ----------
    max_representatives : int
        Maximum total representatives across all clusters (default: 1500).
        These become candidates for NSGA-II ensemble selection.
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

    max_representatives: int = Field(
        default=1000,
        gt=0,
        description="Maximum total representatives for NSGA-II ensemble selection",
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


class ProbEnsembleConfig(BaseModel):
    """
    Configuration for NSGA-II ensemble selection.

    Controls the multi-objective optimization that selects an optimal ensemble
    of parameter sets balancing narrow confidence intervals with good data coverage.

    NSGA-II Optimization
    --------------------
    The algorithm optimizes two objectives:
    - Minimize CI width (narrow intervals = more precise predictions)
    - Maximize coverage (intervals contain observed data points)

    To normalize the CI width objective, the algorithm establishes bounds by:
    1. Computing minimum CI width using the 2 most similar candidates
    2. Computing maximum CI width by testing random samples of different sizes
       specified in ci_sample_sizes (e.g., ensembles of 10, 20, 50, 100 members)
    3. Adding a safety margin (ci_margin_factor) to avoid edge cases

    These bounds ensure proper scaling during Pareto optimization, allowing
    meaningful comparison between narrow but precise vs. wide but comprehensive
    confidence intervals.

    Attributes
    ----------
    nsga_population_size : int
        NSGA-II population size (default: 100, must be > 3).
    nsga_generations : int
        Number of NSGA-II generations (default: 100).
    nsga_crossover_probability : float
        Crossover probability (default: 0.9).
        Higher values encourage more exploration through recombination.
    pareto_preference : float
        Preference for selecting from Pareto front (default: 0.5).
        0.0 = prefer narrow CI width (minimize uncertainty).
        1.0 = prefer high coverage (maximize data fit).
        0.5 = balanced trade-off (knee point).
    ensemble_size_mode : Literal["fixed", "bounded", "automatic"]
        Mode for determining ensemble size (default: "automatic").
        "fixed": use exact ensemble_size.
        "bounded": optimize within [ensemble_size_min, ensemble_size_max].
        "automatic": automatically determine optimal size.
    ensemble_size : int | None
        Fixed ensemble size (required if mode="fixed", must be >= 2).
    ensemble_size_min : int | None
        Minimum ensemble size (required if mode="bounded", must be >= 2).
        Not used for "automatic" or "fixed" modes.
    ensemble_size_max : int | None
        Maximum ensemble size
        (required if mode="bounded", must be >= ensemble_size_min).
        Not used for "automatic" or "fixed" modes.
    ci_margin_factor : float
        Safety margin for CI width bounds normalization (default: 0.1).
        Adds a buffer (e.g., 10%) to min/max CI width bounds to avoid
        numerical edge cases during optimization. Higher values provide
        more conservative bounds.
    ci_sample_sizes : list[int]
        Sample sizes for CI width estimation (default: [10, 20, 50, 100]).
        Used to explore how CI width varies with ensemble size when computing
        maximum CI width bounds. The algorithm tests random ensembles of these
        sizes to find the widest possible CI, ensuring proper normalization.
        Larger sizes explore wider CIs but increase computation time.
    """

    nsga_population_size: int = Field(
        default=100, gt=3, description="NSGA-II population size"
    )
    nsga_generations: int = Field(
        default=100, gt=0, description="NSGA-II number of generations"
    )
    nsga_crossover_probability: float = Field(
        default=0.9, ge=0.0, le=1.0, description="NSGA-II crossover probability"
    )
    pareto_preference: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Preference for Pareto front selection (0=narrow CI, 1=coverage)",
    )
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
    ci_margin_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Safety margin for CI width bounds normalization",
    )
    ci_sample_sizes: list[int] = Field(
        default=[10, 20, 50, 100],
        description="Sample sizes for CI width estimation",
    )

    @model_validator(mode="after")
    def validate_ensemble_size_config(self) -> Self:
        """Validate ensemble size configuration."""
        valid_modes = {"fixed", "bounded", "automatic"}
        if self.ensemble_size_mode not in valid_modes:
            raise ValueError(
                f"Invalid ensemble_size_mode: '{self.ensemble_size_mode}'. "
                f"Must be one of: {', '.join(sorted(valid_modes))}"
            )

        if self.ensemble_size_mode == "fixed":
            self._validate_fixed_mode()
        elif self.ensemble_size_mode == "bounded":
            self._validate_bounded_mode()
        elif self.ensemble_size_mode == "automatic":
            self._validate_automatic_mode()

        return self

    def _validate_fixed_mode(self) -> None:
        """Validate fixed ensemble size mode."""
        if self.ensemble_size is None:
            raise ValueError(
                "ensemble_size must be specified when ensemble_size_mode='fixed'"
            )
        if self.ensemble_size < 2:
            raise ValueError(f"ensemble_size must be >= 2, got {self.ensemble_size}")
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
        if self.ensemble_size_min < 2:
            raise ValueError(
                f"ensemble_size_min must be >= 2, got {self.ensemble_size_min}"
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
    5. Ensemble Selection: NSGA-II multi-objective optimization
    6. Statistics: Calculate confidence intervals and coverage

    Attributes
    ----------
    n_runs : int
        Number of independent calibration runs to perform (default: 50).
        More runs provide better parameter space exploration but take longer.
    evaluation_processing : ProbEvaluationFilterConfig
        Configuration for evaluation deduplication and filtering
    clustering : ProbClusteringConfig
        Configuration for clustering parameter space
    representative_selection : ProbRepresentativeConfig
        Configuration for selecting representatives from clusters
    ensemble_selection : ProbEnsembleConfig
        Configuration for NSGA-II ensemble selection
    confidence_level : float
        Confidence interval level (default: 0.95 for 95% CI).
        Must be in range (0.0, 1.0).
    """

    n_runs: int = Field(
        default=10, gt=0, description="Number of calibration runs to perform"
    )
    evaluation_processing: ProbEvaluationFilterConfig = Field(
        default_factory=ProbEvaluationFilterConfig,
        description="Configuration for evaluation processing",
    )
    clustering: ProbClusteringConfig = Field(
        default_factory=ProbClusteringConfig,
        description="Configuration for clustering parameter space",
    )
    representative_selection: ProbRepresentativeConfig = Field(
        default_factory=ProbRepresentativeConfig,
        description="Configuration for representative selection",
    )
    ensemble_selection: ProbEnsembleConfig = Field(
        default_factory=ProbEnsembleConfig,
        description="Configuration for NSGA-II ensemble selection",
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence interval level (e.g., 0.95 for 95% CI)",
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


class ParetoSolution(BaseModel):
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
        Normalized confidence interval width objective [0, 1] used in optimization
    coverage : float
        Normalized coverage objective [0, 1] used in optimization
    size_penalty : float
        Size constraint penalty applied during optimization [0, infinity]
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
    coverage_percentage: float = Field(
        description="Percentage of observed data points within confidence intervals"
    )
    average_ci_width: float = Field(description="Average width of confidence intervals")
    ci_width: float = Field(description="Normalized CI width objective [0, 1]")
    coverage: float = Field(description="Normalized coverage objective [0, 1]")
    size_penalty: float = Field(description="Size constraint penalty [0, infinity]")


class ProbabilisticCalibrationResult(BaseModel):
    """
    Complete result from probabilistic calibration with ensemble analysis.

    Contains the selected optimal ensemble solution, all Pareto-optimal
    solutions from multi-objective optimization, and metadata about the
    calibration process.

    Attributes
    ----------
    selected_ensemble : ParetoSolution
        The optimal ensemble solution selected based on pareto_preference
    pareto_front : list[ParetoSolution]
        All Pareto-optimal ensemble solutions from NSGA-II optimization
        These represent different trade-offs between CI width and coverage
    selected_pareto_index : int
        Index in pareto_front of the selected solution
    n_runs_performed : int
        Number of calibration runs performed
    n_unique_evaluations : int
        Number of unique parameter evaluations after deduplication
    n_clusters_used : int
        Number of clusters identified in parameter space
    confidence_level : float
        Confidence level used for interval calculation (e.g., 0.95 for 95% CI)
    """

    selected_ensemble: ParetoSolution = Field(
        description="The optimal ensemble solution selected based on preference"
    )
    pareto_front: list[ParetoSolution] = Field(
        description="All Pareto-optimal ensemble solutions from NSGA-II"
    )
    selected_pareto_index: int = Field(
        description="Index in pareto_front of the selected solution"
    )
    n_runs_performed: int = Field(description="Number of calibration runs performed")
    n_unique_evaluations: int = Field(
        description="Number of unique evaluations after deduplication"
    )
    n_clusters_used: int = Field(description="Number of clusters identified")
    confidence_level: float = Field(
        description="Confidence level used (e.g., 0.95 for 95% CI)"
    )
