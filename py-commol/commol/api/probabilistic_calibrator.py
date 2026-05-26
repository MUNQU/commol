"""Probabilistic calibrator for ensemble-based parameter estimation.

This module provides the main ProbabilisticCalibrator class that orchestrates
the probabilistic calibration workflow using focused helper classes.
"""

import logging
import random
import secrets
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.commol_rs._commol_rs import (
        CalibrationResultWithHistoryProtocol,
        EnsembleSelectionResultProtocol,
        ParetoSolutionProtocol,
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
            if obs.compartment not in simulation_output_ids:
                raise ValueError(
                    f"Observed data compartment '{obs.compartment}' not found in "
                    f"model simulation outputs. Available outputs: "
                    f"{sorted(simulation_output_ids)}"
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
            f"Starting probabilistic calibration with {self.config.n_runs} runs"
        )
        total_start = perf_counter()
        stage_timings: dict[str, float] = {}
        stage_counts: dict[str, int] = {}

        # Run multiple calibrations
        stage_start = perf_counter()
        all_results = self._run_calibrations()
        stage_timings["calibration_runs_seconds"] = perf_counter() - stage_start
        stage_counts["n_runs"] = len(all_results)
        stage_counts["n_retained_evaluations"] = self._count_retained_evaluations(
            all_results
        )

        # Process evaluations (collect, deduplicate, filter)
        stage_start = perf_counter()
        unique_evaluations = self._process_evaluations(all_results)
        stage_timings["evaluation_processing_seconds"] = perf_counter() - stage_start
        stage_counts["n_unique_evaluations"] = len(unique_evaluations)

        # Cluster and select representatives
        stage_start = perf_counter()
        representatives, optimal_k = self._cluster_and_select_representatives(
            unique_evaluations
        )
        stage_timings["clustering_and_representative_selection_seconds"] = (
            perf_counter() - stage_start
        )
        stage_counts["n_clusters"] = optimal_k
        stage_counts["n_representatives"] = len(representatives)

        # Run ensemble selection
        stage_start = perf_counter()
        rust_ensemble_result, representatives = self._select_ensemble(representatives)
        stage_timings["prediction_generation_and_ensemble_selection_seconds"] = (
            perf_counter() - stage_start
        )
        stage_counts["n_generated_prediction_points"] = self._count_prediction_points(
            representatives
        )
        stage_counts["pareto_front_size"] = len(rust_ensemble_result.pareto_front)
        stage_counts["selected_ensemble_size"] = len(
            rust_ensemble_result.selected_ensemble
        )

        # Build final result with statistics
        stage_start = perf_counter()
        result = self._build_result(
            representatives=representatives,
            rust_ensemble_result=rust_ensemble_result,
            n_runs=len(all_results),
            n_unique=len(unique_evaluations),
            n_clusters=optimal_k,
            stage_timings=stage_timings,
            stage_counts=stage_counts,
            result_stage_start=stage_start,
            total_start=total_start,
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

    @staticmethod
    def _count_prediction_points(
        representatives: list[CalibrationEvaluation],
    ) -> int:
        """Count generated prediction scalar values attached to representatives."""
        total = 0
        for rep in representatives:
            if rep.predictions:
                total += sum(len(step) for step in rep.predictions)
        return total

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
    ) -> tuple["EnsembleSelectionResultProtocol", list[CalibrationEvaluation]]:
        """Run ensemble selection."""
        logger.info(
            "Running %s ensemble selection...",
            self.config.ensemble_selection.ensemble_algorithm,
        )

        (
            rust_ensemble_result,
            representatives_with_predictions,
        ) = self._ensemble_selector.select_ensemble_with_predictions(
            representatives=representatives,
            population_size=self.config.ensemble_selection.population_size,
            generations=self.config.ensemble_selection.generations,
            confidence_level=self.config.confidence_level,
            pareto_preference=self.config.ensemble_selection.pareto_preference,
            ensemble_size_mode=self.config.ensemble_selection.ensemble_size_mode,
            ensemble_size=self.config.ensemble_selection.ensemble_size,
            ensemble_size_min=self.config.ensemble_selection.ensemble_size_min,
            ensemble_size_max=self.config.ensemble_selection.ensemble_size_max,
            ci_margin_factor=self.config.ensemble_selection.ci_margin_factor,
            ci_sample_sizes=self.config.ensemble_selection.ci_sample_sizes,
            crossover_probability=self.config.ensemble_selection.crossover_probability,
            ci_width_scope=self.config.ensemble_selection.ci_width_scope,
            ensemble_algorithm=self.config.ensemble_selection.ensemble_algorithm,
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

        return rust_ensemble_result, representatives_with_predictions

    def _build_result(
        self,
        representatives: list[CalibrationEvaluation],
        rust_ensemble_result: "EnsembleSelectionResultProtocol",
        n_runs: int,
        n_unique: int,
        n_clusters: int,
        stage_timings: dict[str, float] | None = None,
        stage_counts: dict[str, int] | None = None,
        result_stage_start: float | None = None,
        total_start: float | None = None,
    ) -> ProbabilisticCalibrationResult:
        """Calculate statistics and build final result."""
        result_detail = self.config.result_detail
        logger.info("Building probabilistic result with detail mode: %s", result_detail)

        max_time_step = max(obs.step for obs in self.problem.observed_data)
        time_steps = max_time_step + 1
        simulation_output_ids = list(self.simulation.simulation_outputs)

        selected_rust_solution = rust_ensemble_result.pareto_front[
            rust_ensemble_result.selected_pareto_index
        ]
        selected_solution = self._build_full_pareto_solution(
            rust_sol=selected_rust_solution,
            representatives=representatives,
            simulation_output_ids=simulation_output_ids,
            time_steps=time_steps,
        )

        if result_detail == "full":
            pareto_solutions = [
                (
                    selected_solution
                    if idx == rust_ensemble_result.selected_pareto_index
                    else self._build_full_pareto_solution(
                        rust_sol=rust_sol,
                        representatives=representatives,
                        simulation_output_ids=simulation_output_ids,
                        time_steps=time_steps,
                    )
                )
                for idx, rust_sol in enumerate(rust_ensemble_result.pareto_front)
            ]
            selected_pareto_index = rust_ensemble_result.selected_pareto_index
        elif result_detail == "pareto_summary":
            pareto_solutions = [
                self._build_summary_pareto_solution(rust_sol)
                for rust_sol in rust_ensemble_result.pareto_front
            ]
            selected_pareto_index = rust_ensemble_result.selected_pareto_index
        else:
            pareto_solutions = [selected_solution]
            selected_pareto_index = 0

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
            pareto_front=pareto_solutions,
            selected_pareto_index=selected_pareto_index,
            n_runs_performed=n_runs,
            n_unique_evaluations=n_unique,
            n_clusters_used=n_clusters,
            confidence_level=self.config.confidence_level,
            stage_timings=final_stage_timings,
            stage_counts=stage_counts or {},
        )

    def _build_full_pareto_solution(
        self,
        rust_sol: "ParetoSolutionProtocol",
        representatives: list[CalibrationEvaluation],
        simulation_output_ids: list[str],
        time_steps: int,
    ) -> ParetoSolution:
        """Build a Pareto solution with parameter stats and prediction intervals."""
        solution_params = [representatives[i] for i in rust_sol.selected_indices]
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
        win_median, win_ci_lower, win_ci_upper = (
            self._statistics_calculator.calculate_windowed_prediction_intervals(
                all_preds
            )
        )
        cov_pct, avg_ci = self._statistics_calculator.calculate_coverage_metrics(
            pred_ci_lower, pred_ci_upper
        )

        return ParetoSolution(
            ensemble_size=rust_sol.ensemble_size,
            selected_indices=rust_sol.selected_indices,
            ensemble_parameters=[ep.to_dict() for ep in solution_params],
            parameter_statistics=param_stats,
            prediction_median=pred_median,
            prediction_ci_lower=pred_ci_lower,
            prediction_ci_upper=pred_ci_upper,
            windowed_prediction_median=win_median,
            windowed_prediction_ci_lower=win_ci_lower,
            windowed_prediction_ci_upper=win_ci_upper,
            coverage_percentage=cov_pct,
            average_ci_width=avg_ci,
            ci_width=rust_sol.ci_width,
            coverage=rust_sol.coverage,
            size_penalty=rust_sol.size_penalty,
        )

    @staticmethod
    def _build_summary_pareto_solution(
        rust_sol: "ParetoSolutionProtocol",
    ) -> ParetoSolution:
        """Build a compact Pareto solution containing objective summary only."""
        return ParetoSolution(
            ensemble_size=rust_sol.ensemble_size,
            selected_indices=rust_sol.selected_indices,
            ensemble_parameters=[],
            parameter_statistics={},
            prediction_median={},
            prediction_ci_lower={},
            prediction_ci_upper={},
            coverage_percentage=rust_sol.coverage * 100.0,
            average_ci_width=rust_sol.ci_width,
            ci_width=rust_sol.ci_width,
            coverage=rust_sol.coverage,
            size_penalty=rust_sol.size_penalty,
        )
