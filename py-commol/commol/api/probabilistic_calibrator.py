"""Probabilistic calibrator for ensemble-based parameter estimation.

This module provides the main ProbabilisticCalibrator class that orchestrates
the probabilistic calibration workflow using focused helper classes.
"""

import logging
import random
import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.commol_rs.commol_rs import (
        CalibrationResultWithHistoryProtocol,
        EnsembleSelectionResultProtocol,
    )
    from commol.context.calibration import CalibrationProblem

from commol.api.probabilistic.calibration_runner import CalibrationRunner
from commol.api.probabilistic.ensemble_selector import EnsembleSelector
from commol.api.probabilistic.evaluation_processor import EvaluationProcessor
from commol.api.probabilistic.statistics_calculator import StatisticsCalculator
from commol.context.constants import CalibrationParameterType
from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    ParetoSolution,
    ProbabilisticCalibrationConfig,
    ProbabilisticCalibrationResult,
)

logger = logging.getLogger(__name__)


class ProbabilisticCalibrator:
    """Probabilistic calibration that finds an ensemble of parameter sets.

    This calibrator performs multiple calibration runs, clusters the results,
    and uses NSGA-II to find an optimal ensemble that balances narrow confidence
    intervals with good coverage of observed data.

    The workflow is orchestrated using focused helper classes:
    - CalibrationRunner: Runs multiple calibrations in parallel
    - EvaluationProcessor: Handles deduplication, filtering, and clustering
    - EnsembleSelector: Runs NSGA-II ensemble selection
    - StatisticsCalculator: Computes ensemble statistics and predictions

    Parameters
    ----------
    simulation : Simulation
        A fully initialized Simulation object with the model to calibrate.
    problem : CalibrationProblem
        A fully constructed and validated calibration problem definition.
        The `probabilistic_config` field on the problem should be set to
        configure probabilistic calibration behavior.
    """

    def __init__(
        self,
        simulation: "Simulation",
        problem: "CalibrationProblem",
    ):
        logger.info(
            f"Initializing ProbabilisticCalibrator for model: "
            f"'{simulation.model_definition.name}'"
        )
        self.simulation = simulation
        self.problem = problem

        # Get config from problem, or use defaults
        self.config = problem.probabilistic_config or ProbabilisticCalibrationConfig()

        # Generate master seed once - all components derive their seeds from this
        # Use seed from CalibrationProblem, not from config
        self._master_seed = (
            self.problem.seed if self.problem.seed is not None else secrets.randbits(32)
        )

        # Validate inputs
        self._validate_inputs()

        # Derive independent seeds for each stage using a deterministic PRNG
        # This ensures reproducibility when master_seed is fixed, while
        # guaranteeing statistical independence between components
        rng = random.Random(self._master_seed)
        calibration_seed = rng.getrandbits(32)
        evaluation_seed = rng.getrandbits(32)
        ensemble_seed = rng.getrandbits(32)

        # Initialize helper classes with derived seeds
        self._calibration_runner = CalibrationRunner(
            simulation, problem, seed=calibration_seed
        )
        self._evaluation_processor = EvaluationProcessor(
            deduplication_tolerance=self.config.evaluation_processing.deduplication_tolerance,
            seed=evaluation_seed,
            min_evaluations_for_clustering=self.config.clustering.min_evaluations_for_clustering,
            identical_solutions_atol=self.config.clustering.identical_solutions_atol,
            silhouette_threshold=self.config.clustering.silhouette_threshold,
            silhouette_excellent_threshold=self.config.clustering.silhouette_excellent_threshold,
            kmeans_max_iter=self.config.clustering.kmeans_max_iter,
            kmeans_algorithm=self.config.clustering.kmeans_algorithm,
        )
        self._ensemble_selector = EnsembleSelector(
            simulation, problem, seed=ensemble_seed
        )
        self._statistics_calculator = StatisticsCalculator(
            simulation, problem, self.config.confidence_level
        )

        logger.info(
            f"Probabilistic calibration configured with {self.config.n_runs} runs"
        )

    def _validate_inputs(self) -> None:
        """Validate that simulation and problem are compatible.

        Raises
        ------
        ValueError
            If validation fails due to incompatible inputs.
        """
        model_param_ids = {p.id for p in self.simulation.model_definition.parameters}
        model_bin_ids = {b.id for b in self.simulation.model_definition.population.bins}

        self._validate_calibration_parameters(model_param_ids, model_bin_ids)
        self._validate_observed_data(model_bin_ids)

        logger.debug("Input validation passed")

    def _validate_calibration_parameters(
        self, model_param_ids: set[str], model_bin_ids: set[str]
    ) -> None:
        """Validate that calibration parameters exist in the model."""
        for param in self.problem.parameters:
            if param.parameter_type == CalibrationParameterType.SCALE:
                continue

            if param.parameter_type == CalibrationParameterType.PARAMETER:
                if param.id not in model_param_ids:
                    raise ValueError(
                        f"Calibration parameter '{param.id}' not found in model. "
                        f"Available parameters: {sorted(model_param_ids)}"
                    )

            if param.parameter_type == CalibrationParameterType.INITIAL_CONDITION:
                if param.id not in model_bin_ids:
                    raise ValueError(
                        f"Initial condition parameter '{param.id}' not found in model "
                        f"bins. Available bins: {sorted(model_bin_ids)}"
                    )

    def _validate_observed_data(self, model_bin_ids: set[str]) -> None:
        """Validate that observed data compartments exist and have valid steps."""
        for obs in self.problem.observed_data:
            if obs.compartment not in model_bin_ids:
                raise ValueError(
                    f"Observed data compartment '{obs.compartment}' not found in "
                    f"model. Available compartments: {sorted(model_bin_ids)}"
                )

        if self.problem.observed_data:
            min_step = min(obs.step for obs in self.problem.observed_data)
            if min_step < 0:
                raise ValueError(
                    f"Observed data contains negative time step: {min_step}. "
                    "Time steps must be non-negative."
                )

    def run(self) -> ProbabilisticCalibrationResult:
        """Run probabilistic calibration.

        Returns
        -------
        ProbabilisticCalibrationResult
            Object containing the ensemble of parameter sets, statistics,
            predictions with confidence intervals, and coverage metrics.

        Raises
        ------
        RuntimeError
            If calibration or ensemble selection fails.
        """
        logger.info(
            f"Starting probabilistic calibration with {self.config.n_runs} runs"
        )

        # Run multiple calibrations
        all_results = self._run_calibrations()

        # Process evaluations (collect, deduplicate, filter)
        unique_evaluations = self._process_evaluations(all_results)

        # Cluster and select representatives
        representatives, optimal_k = self._cluster_and_select_representatives(
            unique_evaluations
        )

        # Run NSGA-II ensemble selection
        rust_ensemble_result = self._select_ensemble(representatives)

        # Build final result with statistics
        result = self._build_result(
            representatives=representatives,
            rust_ensemble_result=rust_ensemble_result,
            n_runs=len(all_results),
            n_unique=len(unique_evaluations),
            n_clusters=optimal_k,
        )

        # Log parameter intervals for selected ensemble
        logger.info("Parameter value intervals for selected ensemble:")
        for param_name, stats in result.selected_ensemble.parameter_statistics.items():
            logger.info(
                f"  {param_name}: "
                f"[{stats.percentile_lower:.6f}, {stats.percentile_upper:.6f}] "
                f"(mean: {stats.mean:.6f}; median: {stats.median:.6f})"
            )

        logger.info(
            f"Probabilistic calibration complete. "
            f"Ensemble size: {result.selected_ensemble.ensemble_size}, "
            f"Coverage: {result.selected_ensemble.coverage_percentage:.1f}%, "
            f"Average CI width: {result.selected_ensemble.average_ci_width:.4f}"
        )

        return result

    def _run_calibrations(self) -> list["CalibrationResultWithHistoryProtocol"]:
        """Run multiple calibration attempts."""
        logger.info("Running multiple calibration attempts...")
        all_results = self._calibration_runner.run_multiple(
            n_runs=self.config.n_runs,
        )
        logger.info(
            f"Completed {len(all_results)} successful calibration runs "
            f"out of {self.config.n_runs} attempts"
        )
        return all_results

    def _process_evaluations(
        self, all_results: list["CalibrationResultWithHistoryProtocol"]
    ) -> list[CalibrationEvaluation]:
        """Collect, deduplicate, and filter evaluations."""
        logger.info("Collecting and deduplicating evaluations...")

        # Collect evaluations from results
        all_evaluations = self._evaluation_processor.collect_evaluations(all_results)

        # Deduplicate
        unique_evaluations = self._evaluation_processor.deduplicate(all_evaluations)
        logger.info(
            f"Collected {len(all_evaluations)} evaluations, "
            f"{len(unique_evaluations)} unique after deduplication"
        )

        # Filter by loss percentile if configured
        if self.config.evaluation_processing.loss_percentile_filter < 1.0:
            unique_evaluations = self._evaluation_processor.filter_by_loss_percentile(
                unique_evaluations,
                self.config.evaluation_processing.loss_percentile_filter,
            )
            logger.info(
                "Filtered to best "
                f"{self.config.evaluation_processing.loss_percentile_filter * 100:.0f}"
                f"% by loss: {len(unique_evaluations)} evaluations remaining"
            )

        # Validate minimum evaluations
        if (
            len(unique_evaluations)
            < self.config.evaluation_processing.min_evaluations_required
        ):
            raise RuntimeError(
                f"Too few unique evaluations ({len(unique_evaluations)}). Need at least"
                f" {self.config.evaluation_processing.min_evaluations_required} for "
                "probabilistic calibration. Try increasing n_runs or decreasing "
                "deduplication_tolerance."
            )

        return unique_evaluations

    def _cluster_and_select_representatives(
        self, evaluations: list[CalibrationEvaluation]
    ) -> tuple[list[CalibrationEvaluation], int]:
        """Cluster evaluations and select representatives."""
        logger.info("Clustering parameter space...")

        # Determine number of clusters
        if self.config.clustering.n_clusters is not None:
            optimal_k = self.config.clustering.n_clusters
            logger.info(f"Using user-specified number of clusters: {optimal_k}")
        else:
            optimal_k = self._evaluation_processor.find_optimal_k(evaluations)
            logger.info(
                f"Automatically determined optimal number of clusters: {optimal_k}"
            )

        # Cluster evaluations
        cluster_labels = self._evaluation_processor.cluster_evaluations(
            evaluations, optimal_k
        )

        # Select representatives from clusters
        representative_indices = self._evaluation_processor.select_representatives(
            evaluations=evaluations,
            cluster_labels=cluster_labels,
            max_representatives=self.config.representative_selection.max_representatives,
            elite_fraction=self.config.representative_selection.percentage_elite_cluster_selection,
            strategy=self.config.representative_selection.cluster_representative_strategy,
            selection_method=self.config.representative_selection.cluster_selection_method,
            quality_temperature=self.config.representative_selection.quality_temperature,
            k_neighbors_min=self.config.representative_selection.k_neighbors_min,
            k_neighbors_max=self.config.representative_selection.k_neighbors_max,
            sparsity_weight=self.config.representative_selection.sparsity_weight,
            stratum_fit_weight=self.config.representative_selection.stratum_fit_weight,
        )

        representatives = [evaluations[i] for i in representative_indices]
        logger.info(f"Selected {len(representatives)} representative parameter sets")

        return representatives, optimal_k

    def _select_ensemble(
        self, representatives: list[CalibrationEvaluation]
    ) -> "EnsembleSelectionResultProtocol":
        """Run NSGA-II ensemble selection."""
        logger.info("Running NSGA-II ensemble selection...")

        rust_ensemble_result = self._ensemble_selector.select_ensemble(
            representatives=representatives,
            nsga_population_size=self.config.ensemble_selection.nsga_population_size,
            nsga_generations=self.config.ensemble_selection.nsga_generations,
            confidence_level=self.config.confidence_level,
            pareto_preference=self.config.ensemble_selection.pareto_preference,
            ensemble_size_mode=self.config.ensemble_selection.ensemble_size_mode,
            ensemble_size=self.config.ensemble_selection.ensemble_size,
            ensemble_size_min=self.config.ensemble_selection.ensemble_size_min,
            ensemble_size_max=self.config.ensemble_selection.ensemble_size_max,
            ci_margin_factor=self.config.ensemble_selection.ci_margin_factor,
            ci_sample_sizes=self.config.ensemble_selection.ci_sample_sizes,
            nsga_crossover_probability=self.config.ensemble_selection.nsga_crossover_probability,
        )

        # Log ensemble size information based on mode
        ensemble_size = len(rust_ensemble_result.selected_ensemble)
        if self.config.ensemble_selection.ensemble_size_mode == "fixed":
            logger.info(
                f"Selected ensemble of {ensemble_size} parameter sets "
                f"(target: {self.config.ensemble_selection.ensemble_size})"
            )
        elif self.config.ensemble_selection.ensemble_size_mode == "bounded":
            logger.info(
                f"Selected ensemble of {ensemble_size} parameter sets "
                f"(range: [{self.config.ensemble_selection.ensemble_size_min}, "
                f"{self.config.ensemble_selection.ensemble_size_max}])"
            )
        else:  # automatic
            logger.info(
                f"Selected ensemble of {ensemble_size} parameter sets (automatic)"
            )

        return rust_ensemble_result

    def _build_result(
        self,
        representatives: list[CalibrationEvaluation],
        rust_ensemble_result: "EnsembleSelectionResultProtocol",
        n_runs: int,
        n_unique: int,
        n_clusters: int,
    ) -> ProbabilisticCalibrationResult:
        """Calculate statistics and build final result with full Pareto front."""
        logger.info("Building complete Pareto front with statistics...")

        max_time_step = max(obs.step for obs in self.problem.observed_data)
        time_steps = max_time_step + 1
        compartment_ids = [
            bin.id for bin in self.simulation.model_definition.population.bins
        ]

        # Build full ParetoSolution objects for each solution in the Pareto front
        pareto_solutions = []
        for rust_sol in rust_ensemble_result.pareto_front:
            # Extract parameter sets for this solution
            solution_params = [representatives[i] for i in rust_sol.selected_indices]

            # Calculate statistics for this solution
            param_stats = self._statistics_calculator.calculate_parameter_statistics(
                solution_params
            )

            # Generate predictions for this solution
            all_preds = self._statistics_calculator.generate_ensemble_predictions(
                solution_params, compartment_ids, time_steps
            )

            # Calculate prediction intervals
            pred_median, pred_ci_lower, pred_ci_upper = (
                self._statistics_calculator.calculate_prediction_intervals(
                    all_preds, compartment_ids
                )
            )

            # Calculate coverage metrics
            cov_pct, avg_ci = self._statistics_calculator.calculate_coverage_metrics(
                pred_ci_lower, pred_ci_upper
            )

            # Convert to parameter dicts
            param_dicts = [ep.to_dict() for ep in solution_params]

            # Create complete ParetoSolution
            pareto_sol = ParetoSolution(
                ensemble_size=rust_sol.ensemble_size,
                selected_indices=rust_sol.selected_indices,
                ensemble_parameters=param_dicts,
                parameter_statistics=param_stats,
                prediction_median=pred_median,
                prediction_ci_lower=pred_ci_lower,
                prediction_ci_upper=pred_ci_upper,
                coverage_percentage=cov_pct,
                average_ci_width=avg_ci,
                ci_width=rust_sol.ci_width,
                coverage=rust_sol.coverage,
                size_penalty=rust_sol.size_penalty,
            )
            pareto_solutions.append(pareto_sol)

        # The selected ensemble is the one at selected_pareto_index
        selected_solution = pareto_solutions[rust_ensemble_result.selected_pareto_index]

        logger.info(
            f"Coverage: {selected_solution.coverage_percentage:.2f}%, "
            f"Average CI width: {selected_solution.average_ci_width:.4f}"
        )

        return ProbabilisticCalibrationResult(
            selected_ensemble=selected_solution,
            pareto_front=pareto_solutions,
            selected_pareto_index=rust_ensemble_result.selected_pareto_index,
            n_runs_performed=n_runs,
            n_unique_evaluations=n_unique,
            n_clusters_used=n_clusters,
            confidence_level=self.config.confidence_level,
        )
