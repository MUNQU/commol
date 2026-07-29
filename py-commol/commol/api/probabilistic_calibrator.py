"""Probabilistic calibrator for ensemble-based parameter estimation.

This module provides the main ProbabilisticCalibrator class that orchestrates
the probabilistic calibration workflow using focused helper classes.
"""

import logging
import random
import secrets
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.commol_rs._commol_rs import (
        CalibrationResultWithHistoryProtocol,
    )
    from commol.context.calibration import CalibrationProblem

from commol.api.probabilistic.calibration_runner import CalibrationRunner
from commol.api.probabilistic.ensemble_selector import (
    EnsembleSelector,
    FitGatedSelection,
)
from commol.api.probabilistic.evaluation_processor import EvaluationProcessor
from commol.api.probabilistic.statistics_calculator import StatisticsCalculator
from commol.context.constants import CalibrationParameterType
from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    EnsembleCandidate,
    EnsembleSolution,
    ProbGreedyLocalSearchConfig,
    ProbNsga2Config,
    ProbabilisticCalibrationConfig,
    ProbabilisticCalibrationResult,
)

logger = logging.getLogger(__name__)


class ProbabilisticCalibrator:
    """Probabilistic calibration that finds an ensemble of parameter sets.

    This calibrator performs multiple calibration runs, clusters the results,
    and uses ensemble selection to find an optimal ensemble that balances
    narrow confidence intervals with good coverage of observed data.

    The workflow is orchestrated using focused helper classes:
    - CalibrationRunner: Runs multiple calibrations in parallel
    - EvaluationProcessor: Handles deduplication, filtering, and clustering
    - EnsembleSelector: Runs ensemble selection
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
            max_k=self.config.clustering.max_k,
            silhouette_sample_size=self.config.clustering.silhouette_sample_size,
            minibatch_kmeans_threshold=self.config.clustering.minibatch_kmeans_threshold,
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
        stratifications = self.simulation.model_definition.population.stratifications
        model_binary_stratification_categories = {
            category
            for stratification in stratifications
            if len(stratification.categories) == 2
            for category in stratification.categories
        }
        engine_compartment_ids = set(self.simulation.engine.compartments)
        simulation_output_ids = set(self.simulation.simulation_outputs)

        self._validate_calibration_parameters(
            model_param_ids,
            model_bin_ids,
            model_binary_stratification_categories,
            engine_compartment_ids,
        )
        self._validate_observed_data(simulation_output_ids)
        self._warn_constraint_time_steps_beyond_observations()

        logger.debug("Input validation passed")

    def _validate_calibration_parameters(
        self,
        model_param_ids: set[str],
        model_bin_ids: set[str],
        model_binary_stratification_categories: set[str],
        engine_compartment_ids: set[str],
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
                if (
                    param.id not in model_bin_ids
                    and param.id not in model_binary_stratification_categories
                    and param.id not in engine_compartment_ids
                ):
                    raise ValueError(
                        f"Initial condition parameter '{param.id}' not found in model "
                        f"bins, stratification categories, or expanded compartments. "
                        f"Available bins: {sorted(model_bin_ids)}. "
                        f"Available binary stratification categories: "
                        f"{sorted(model_binary_stratification_categories)}. "
                        f"Available compartments: {sorted(engine_compartment_ids)}"
                    )

    def _validate_observed_data(self, simulation_output_ids: set[str]) -> None:
        """Validate that observed data targets exist and have valid steps."""
        for obs in self.problem.observed_data:
            observed_outputs = obs.compartments or [obs.compartment]
            missing_outputs = [
                output
                for output in observed_outputs
                if output not in simulation_output_ids
            ]
            if missing_outputs:
                raise ValueError(
                    f"Observed data output(s) {missing_outputs} for observation "
                    f"'{obs.compartment}' not found in model simulation outputs. "
                    f"Available outputs: {sorted(simulation_output_ids)}"
                )

        if self.problem.observed_data:
            min_step = min(obs.step for obs in self.problem.observed_data)
            if min_step < 0:
                raise ValueError(
                    f"Observed data contains negative time step: {min_step}. "
                    "Time steps must be non-negative."
                )

    def _warn_constraint_time_steps_beyond_observations(self) -> None:
        """Warn when time-dependent constraints extend beyond observed data."""
        if not self.problem.observed_data:
            return

        max_observed_step = max(obs.step for obs in self.problem.observed_data)
        for constraint in self.problem.constraints:
            if constraint.time_steps and max(constraint.time_steps) > max_observed_step:
                logger.warning(
                    "Constraint '%s' has time_steps beyond the maximum observed "
                    "step (%s); simulation will be extended to evaluate them.",
                    constraint.id,
                    max_observed_step,
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
            "Starting probabilistic calibration: %d runs across a 5-stage pipeline",
            self.config.n_runs,
        )
        total_start = perf_counter()
        stage_timings: dict[str, float] = {}
        stage_counts: dict[str, int] = {}

        # Stage 1: run multiple calibrations
        logger.info("[1/5] Running %d independent calibrations...", self.config.n_runs)
        stage_start = perf_counter()
        all_results = self._run_calibrations()
        stage_timings["calibration_runs_seconds"] = perf_counter() - stage_start
        stage_counts["n_runs"] = len(all_results)
        stage_counts["n_retained_evaluations"] = self._count_retained_evaluations(
            all_results
        )
        logger.info(
            "[1/5] Calibrations done in %.2fs: %d runs, %d retained evaluations",
            stage_timings["calibration_runs_seconds"],
            len(all_results),
            stage_counts["n_retained_evaluations"],
        )

        # Stage 2: process evaluations (collect, deduplicate, filter)
        logger.info("[2/5] Collecting, deduplicating and filtering evaluations...")
        stage_start = perf_counter()
        unique_evaluations, evaluation_counts = self._process_evaluations(all_results)
        stage_timings["evaluation_processing_seconds"] = perf_counter() - stage_start
        stage_counts.update(evaluation_counts)
        logger.info(
            "[2/5] Evaluation processing done in %.2fs: %d unique candidates",
            stage_timings["evaluation_processing_seconds"],
            len(unique_evaluations),
        )

        # Stage 3: cluster and select representatives
        logger.info("[3/5] Clustering candidates and selecting representatives...")
        stage_start = perf_counter()
        (
            representatives,
            optimal_k,
            representative_counts,
        ) = self._cluster_and_select_representatives(unique_evaluations)
        stage_timings["clustering_and_representative_selection_seconds"] = (
            perf_counter() - stage_start
        )
        stage_counts["n_clusters"] = optimal_k
        stage_counts["n_representatives"] = len(representatives)
        stage_counts.update(representative_counts)
        logger.info(
            "[3/5] Clustering done in %.2fs: %d clusters, %d representatives",
            stage_timings["clustering_and_representative_selection_seconds"],
            optimal_k,
            len(representatives),
        )
        ensemble_candidates = (
            [
                EnsembleCandidate(
                    parameters=representative.to_dict(),
                    loss=representative.loss,
                )
                for representative in representatives
            ]
            if self.config.include_ensemble_candidates
            else None
        )

        # Stage 4: run ensemble selection
        logger.info("[4/5] Generating predictions and selecting the ensemble...")
        stage_start = perf_counter()
        selection, representatives = self._select_ensemble(representatives)
        stage_timings["prediction_generation_and_ensemble_selection_seconds"] = (
            perf_counter() - stage_start
        )
        stage_counts["n_selection_observation_values"] = len(representatives) * len(
            self.problem.observed_data
        )
        stage_counts["selected_ensemble_size"] = selection.ensemble_size
        stage_counts["pareto_front_size"] = len(selection.pareto_front or [])
        logger.info(
            "[4/5] Ensemble selection done in %.2fs: %s backend, ensemble size %d",
            stage_timings["prediction_generation_and_ensemble_selection_seconds"],
            selection.algorithm,
            selection.ensemble_size,
        )

        # Stage 5: build final result with statistics
        logger.info("[5/5] Computing ensemble statistics and prediction intervals...")
        stage_start = perf_counter()
        result = self._build_result(
            representatives=representatives,
            selection=selection,
            n_runs=len(all_results),
            n_unique=len(unique_evaluations),
            n_clusters=optimal_k,
            stage_timings=stage_timings,
            stage_counts=stage_counts,
            ensemble_candidates=ensemble_candidates,
            result_stage_start=stage_start,
            total_start=total_start,
        )
        logger.info(
            "[5/5] Statistics done in %.2fs: %d ensemble members",
            result.stage_timings.get("result_construction_seconds", 0.0),
            result.selected_ensemble.ensemble_size,
        )

        logger.debug(
            "Probabilistic calibration stage timings: "
            f"{result.stage_timings}; counts: {result.stage_counts}"
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

    @staticmethod
    def _count_retained_evaluations(
        all_results: list["CalibrationResultWithHistoryProtocol"],
    ) -> int:
        """Count retained evaluations returned by all calibration runs."""
        count = 0
        for result in all_results:
            if hasattr(result, "evaluations") and len(result.evaluations) > 0:
                count += len(result.evaluations)
            else:
                count += 1
        return count

    def _run_calibrations(self) -> list["CalibrationResultWithHistoryProtocol"]:
        """Run multiple calibration attempts."""
        logger.info("Running multiple calibration attempts...")
        all_results = self._calibration_runner.run_multiple(
            n_runs=self.config.n_runs,
            evaluation_retention=self.config.evaluation_processing.evaluation_retention,
            top_k_per_run=self.config.evaluation_processing.top_k_per_run,
        )
        logger.info(
            f"Completed {len(all_results)} successful calibration runs "
            f"out of {self.config.n_runs} attempts"
        )
        return all_results

    def _process_evaluations(
        self, all_results: list["CalibrationResultWithHistoryProtocol"]
    ) -> tuple[list[CalibrationEvaluation], dict[str, int]]:
        """Collect, deduplicate, and filter evaluations."""
        logger.info("Collecting and deduplicating evaluations...")

        # Collect evaluations from results
        all_evaluations = self._evaluation_processor.collect_evaluations(all_results)

        # Deduplicate
        unique_evaluations = self._evaluation_processor.deduplicate(all_evaluations)
        counts = {
            "n_unique_evaluations_before_filters": len(unique_evaluations),
        }
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
        counts["n_evaluations_after_percentile_filter"] = len(unique_evaluations)

        evaluation_config = self.config.evaluation_processing
        max_loss_ratio = evaluation_config.max_loss_ratio
        relative_loss_ratio = max_loss_ratio
        if (
            evaluation_config.tail_max_representatives > 0
            and evaluation_config.tail_max_loss_ratio is not None
        ):
            relative_loss_ratio = max(
                ratio
                for ratio in (
                    max_loss_ratio,
                    evaluation_config.tail_max_loss_ratio,
                )
                if ratio is not None
            )

        if relative_loss_ratio is not None:
            unique_evaluations = self._evaluation_processor.filter_by_relative_loss(
                unique_evaluations,
                relative_loss_ratio,
            )
            logger.info(
                "Applied relative-loss gate (<= %.3fx best): %s evaluations remaining",
                relative_loss_ratio,
                len(unique_evaluations),
            )
        counts["n_evaluations_after_relative_loss_filter"] = len(unique_evaluations)
        if max_loss_ratio is not None:
            core_evaluations = self._evaluation_processor.filter_by_relative_loss(
                unique_evaluations,
                max_loss_ratio,
            )
            counts["n_core_evaluations_after_relative_loss_filter"] = len(
                core_evaluations
            )
            counts["n_tail_evaluations_after_relative_loss_filter"] = len(
                unique_evaluations
            ) - len(core_evaluations)

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

        counts["n_unique_evaluations"] = len(unique_evaluations)
        return unique_evaluations, counts

    def _cluster_and_select_representatives(
        self, evaluations: list[CalibrationEvaluation]
    ) -> tuple[list[CalibrationEvaluation], int, dict[str, int]]:
        """Cluster evaluations and select representatives."""
        logger.info("Clustering calibration candidates...")
        counts: dict[str, int] = {}

        all_feature_vectors: np.ndarray | None = None
        if self.config.clustering.feature_space == "observed_predictions":
            observation_predictions = (
                self._ensemble_selector.generate_observation_predictions(evaluations)
            )
            all_feature_vectors = np.asarray(
                [
                    [step[0] for step in evaluation.predictions or []]
                    for evaluation in observation_predictions
                ],
                dtype=float,
            )
            all_feature_vectors = self._standardize_feature_vectors(all_feature_vectors)
        else:
            all_feature_vectors = self._standardize_feature_vectors(
                EvaluationProcessor._feature_vectors(evaluations, None)
            )

        core_indices = self._core_evaluation_indices(evaluations)
        core_evaluations = [evaluations[index] for index in core_indices]
        feature_vectors = (
            all_feature_vectors[core_indices]
            if self.config.clustering.feature_space == "observed_predictions"
            else None
        )
        counts["n_core_candidate_evaluations"] = len(core_evaluations)

        # Determine number of clusters
        if self.config.clustering.n_clusters is not None:
            optimal_k = self.config.clustering.n_clusters
            logger.info(f"Using user-specified number of clusters: {optimal_k}")
        else:
            optimal_k = self._evaluation_processor.find_optimal_k(
                core_evaluations,
                feature_vectors=feature_vectors,
            )
            logger.info(
                f"Automatically determined optimal number of clusters: {optimal_k}"
            )

        # Cluster evaluations
        cluster_labels = self._evaluation_processor.cluster_evaluations(
            core_evaluations,
            optimal_k,
            feature_vectors=feature_vectors,
        )

        # Select representatives from clusters
        core_representative_indices = self._evaluation_processor.select_representatives(
            evaluations=core_evaluations,
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
            feature_vectors=feature_vectors,
        )
        representative_indices = [
            core_indices[index] for index in core_representative_indices
        ]
        counts["n_core_representatives"] = len(representative_indices)

        tail_indices = self._select_prediction_novel_tail_indices(
            evaluations=evaluations,
            selected_indices=representative_indices,
            feature_vectors=all_feature_vectors,
        )
        if tail_indices:
            representative_indices = [
                *representative_indices,
                *[
                    tail_index
                    for tail_index in tail_indices
                    if tail_index not in representative_indices
                ],
            ]
            logger.info(
                "Added %s prediction-novel tail representatives",
                len(tail_indices),
            )
        counts["n_tail_representatives"] = len(tail_indices)

        representatives = [evaluations[i] for i in representative_indices]
        logger.info(f"Selected {len(representatives)} representative parameter sets")

        return representatives, optimal_k, counts

    def _core_evaluation_indices(
        self, evaluations: list[CalibrationEvaluation]
    ) -> list[int]:
        """Return indices admitted to the core representative selection pool."""
        evaluation_config = self.config.evaluation_processing
        if (
            evaluation_config.tail_max_representatives <= 0
            or evaluation_config.max_loss_ratio is None
            or evaluation_config.tail_max_loss_ratio is None
        ):
            return list(range(len(evaluations)))

        best_loss = min(evaluation.loss for evaluation in evaluations)
        threshold = best_loss * evaluation_config.max_loss_ratio + np.finfo(float).eps
        return [
            index
            for index, evaluation in enumerate(evaluations)
            if evaluation.loss <= threshold
        ]

    def _select_prediction_novel_tail_indices(
        self,
        evaluations: list[CalibrationEvaluation],
        selected_indices: list[int],
        feature_vectors: np.ndarray,
    ) -> list[int]:
        """Select wider-loss candidates that add observed-prediction diversity."""
        evaluation_config = self.config.evaluation_processing
        if (
            evaluation_config.tail_max_representatives <= 0
            or evaluation_config.max_loss_ratio is None
            or evaluation_config.tail_max_loss_ratio is None
        ):
            return []

        best_loss = min(evaluation.loss for evaluation in evaluations)
        core_threshold = (
            best_loss * evaluation_config.max_loss_ratio + np.finfo(float).eps
        )
        tail_threshold = (
            best_loss * evaluation_config.tail_max_loss_ratio + np.finfo(float).eps
        )
        tail_candidate_indices = [
            index
            for index, evaluation in enumerate(evaluations)
            if core_threshold < evaluation.loss <= tail_threshold
            and index not in selected_indices
        ]

        return self._evaluation_processor.select_prediction_novel_candidates(
            selected_indices=selected_indices,
            candidate_indices=tail_candidate_indices,
            feature_vectors=feature_vectors,
            max_candidates=evaluation_config.tail_max_representatives,
        )

    def _select_ensemble(
        self, representatives: list[CalibrationEvaluation]
    ) -> tuple[FitGatedSelection, list[CalibrationEvaluation]]:
        """Run ensemble selection."""
        selection_config = self.config.ensemble_selection
        if isinstance(selection_config, ProbNsga2Config):
            algorithm_name = "nsga2"
        elif isinstance(selection_config, ProbGreedyLocalSearchConfig):
            algorithm_name = "greedy_local_search"
        else:
            raise TypeError(
                f"Unsupported ensemble selection config: {type(selection_config)!r}"
            )
        logger.info(
            "Running %s ensemble selection...",
            algorithm_name,
        )

        (
            ensemble_result,
            representatives_with_predictions,
        ) = self._ensemble_selector.select_ensemble_with_predictions(
            representatives=representatives,
            confidence_level=self.config.confidence_level,
            selection_config=selection_config,
        )

        # Log ensemble size information based on mode
        ensemble_size = ensemble_result.ensemble_size
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

        return ensemble_result, representatives_with_predictions

    @staticmethod
    def _standardize_feature_vectors(feature_vectors: np.ndarray) -> np.ndarray:
        """Standardize feature columns for distance-based diversity selection."""
        if feature_vectors.ndim != 2 or feature_vectors.shape[1] == 0:
            raise ValueError(
                "Observed-prediction clustering requires one feature per observation"
            )
        means = feature_vectors.mean(axis=0)
        scales = feature_vectors.std(axis=0)
        scales[scales <= np.finfo(float).eps] = 1.0
        return (feature_vectors - means) / scales

    def _build_result(
        self,
        representatives: list[CalibrationEvaluation],
        selection: FitGatedSelection,
        n_runs: int,
        n_unique: int,
        n_clusters: int,
        stage_timings: dict[str, float] | None = None,
        stage_counts: dict[str, int] | None = None,
        ensemble_candidates: list[EnsembleCandidate] | None = None,
        result_stage_start: float | None = None,
        total_start: float | None = None,
    ) -> ProbabilisticCalibrationResult:
        """Calculate statistics and build final result."""
        max_time_step = max(obs.step for obs in self.problem.observed_data)
        time_steps = max_time_step + 1
        simulation_output_ids = list(self.simulation.simulation_outputs)

        selected_solution = self._build_ensemble_solution(
            selection=selection,
            representatives=representatives,
            simulation_output_ids=simulation_output_ids,
            time_steps=time_steps,
        )

        logger.info(
            f"Coverage: {selected_solution.coverage_percentage:.2f}%, "
            f"Average CI width: {selected_solution.average_ci_width:.4f}"
        )

        final_stage_timings = dict(stage_timings or {})
        if result_stage_start is not None:
            final_stage_timings["result_construction_seconds"] = (
                perf_counter() - result_stage_start
            )
        if total_start is not None:
            final_stage_timings["total_seconds"] = perf_counter() - total_start

        return ProbabilisticCalibrationResult(
            selected_ensemble=selected_solution,
            selection_algorithm=selection.algorithm,
            pareto_front=selection.pareto_front,
            selected_pareto_index=selection.selected_pareto_index,
            n_runs_performed=n_runs,
            n_unique_evaluations=n_unique,
            n_clusters_used=n_clusters,
            confidence_level=self.config.confidence_level,
            stage_timings=final_stage_timings,
            stage_counts=stage_counts or {},
            ensemble_candidates=ensemble_candidates,
        )

    def _build_ensemble_solution(
        self,
        selection: FitGatedSelection,
        representatives: list[CalibrationEvaluation],
        simulation_output_ids: list[str],
        time_steps: int,
    ) -> EnsembleSolution:
        """Build the selected solution with parameter and prediction statistics."""
        solution_params = [representatives[i] for i in selection.selected_indices]
        param_stats = self._statistics_calculator.calculate_parameter_statistics(
            solution_params
        )
        all_preds = self._statistics_calculator.generate_ensemble_predictions(
            solution_params, simulation_output_ids, time_steps
        )
        pred_median, pred_ci_lower, pred_ci_upper = (
            self._statistics_calculator.calculate_prediction_intervals(
                all_preds, simulation_output_ids
            )
        )
        win_median, win_ci_lower, win_ci_upper, win_steps = (
            self._statistics_calculator.calculate_windowed_prediction_intervals(
                all_preds,
                solution_params,
            )
        )
        cov_pct, avg_ci = self._statistics_calculator.calculate_coverage_metrics(
            pred_ci_lower,
            pred_ci_upper,
            all_predictions=all_preds,
            ensemble_params=solution_params,
        )
        point_member = min(solution_params, key=lambda evaluation: evaluation.loss)
        central_loss = self._statistics_calculator.calculate_central_loss(
            all_preds,
            solution_params,
        )
        observation_diagnostics = (
            self._statistics_calculator.calculate_observation_diagnostics(
                all_preds,
                solution_params,
            )
        )

        return EnsembleSolution(
            ensemble_size=selection.ensemble_size,
            selected_indices=selection.selected_indices,
            ensemble_parameters=[ep.to_dict() for ep in solution_params],
            parameter_statistics=param_stats,
            prediction_median=pred_median,
            prediction_ci_lower=pred_ci_lower,
            prediction_ci_upper=pred_ci_upper,
            windowed_prediction_steps=win_steps,
            windowed_prediction_median=win_median,
            windowed_prediction_ci_lower=win_ci_lower,
            windowed_prediction_ci_upper=win_ci_upper,
            coverage_percentage=cov_pct,
            average_ci_width=avg_ci,
            ci_width=selection.ci_width,
            coverage=selection.coverage,
            point_parameters=point_member.to_dict(),
            point_loss=point_member.loss,
            central_loss=central_loss,
            observation_diagnostics=observation_diagnostics,
            selection_diagnostics=selection.diagnostics,
        )
