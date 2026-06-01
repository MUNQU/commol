"""Ensemble selector for probabilistic calibration.

This module handles the multi-objective optimization process that selects
an optimal ensemble of parameter sets balancing confidence interval width
and coverage of observed data.
"""

import logging
from typing import TYPE_CHECKING

from commol.commol_rs import _commol_rs as commol_rs
from commol.api.probabilistic.calibration_runner import CalibrationRunner

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.commol_rs._commol_rs import EnsembleSelectionResultProtocol
    from commol.context.calibration import CalibrationProblem

from commol.context.probabilistic_calibration import CalibrationEvaluation

logger = logging.getLogger(__name__)

MetricPoint = tuple[int, int]
ObservedTuple = tuple[int, int, float]
ObservationComponent = tuple[int, int | None]
ObservationSpec = tuple[list[ObservationComponent], str | None]


class EnsembleSelector:
    """Handles ensemble selection.

    This class is responsible for:
    - Generating predictions for candidate parameter sets
    - Running the configured ensemble selection algorithm
    - Selecting Pareto-optimal ensemble

    Parameters
    ----------
    simulation : Simulation
        A fully initialized Simulation object
    problem : CalibrationProblem
        The calibration problem definition
    seed : int
        Random seed for reproducibility in ensemble selection
    """

    def __init__(
        self,
        simulation: "Simulation",
        problem: "CalibrationProblem",
        seed: int,
    ):
        self.simulation = simulation
        self.problem = problem
        self.seed = seed
        self._calibration_runner = CalibrationRunner(simulation, problem, seed)
        self._simulation_output_to_idx = {
            output_name: idx
            for idx, output_name in enumerate(simulation.simulation_outputs)
        }

    def select_ensemble(
        self,
        representatives: list[CalibrationEvaluation],
        population_size: int,
        generations: int,
        confidence_level: float,
        pareto_preference: float,
        ensemble_size_mode: str,
        ensemble_size: int | None,
        ensemble_size_min: int | None,
        ensemble_size_max: int | None,
        ci_margin_factor: float,
        ci_sample_sizes: list[int],
        crossover_probability: float,
        ci_width_scope: str = "full_trajectory",
        ensemble_algorithm: str = "nsga2",
    ) -> "EnsembleSelectionResultProtocol":
        """Run ensemble selection.

        Parameters
        ----------
        representatives : list[CalibrationEvaluation]
            Candidate parameter sets for ensemble selection
        population_size : int
            Population size for population-based algorithms
        generations : int
            Iteration count for iterative algorithms
        confidence_level : float
            Confidence level for CI calculation (e.g., 0.95)
        pareto_preference : float
            Preference for Pareto front selection (0.0-1.0)
        ensemble_size_mode : str
            Mode for determining ensemble size ("fixed", "bounded", "automatic")
        ensemble_size : int | None
            Fixed ensemble size (required if mode='fixed')
        ensemble_size_min : int | None
            Minimum ensemble size (required if mode='bounded')
        ensemble_size_max : int | None
            Maximum ensemble size (required if mode='bounded')
        ci_margin_factor : float
            Margin factor for CI bounds estimation (e.g., 0.1 = 10% margin)
        ci_sample_sizes : list[int]
            Sample sizes used for CI bounds estimation
        crossover_probability : float
            Crossover probability for algorithms that use crossover

        Returns
        -------
        object
            Rust EnsembleSelectionResult object.
        """
        ensemble_result, _ = self.select_ensemble_with_predictions(
            representatives=representatives,
            population_size=population_size,
            generations=generations,
            confidence_level=confidence_level,
            pareto_preference=pareto_preference,
            ensemble_size_mode=ensemble_size_mode,
            ensemble_size=ensemble_size,
            ensemble_size_min=ensemble_size_min,
            ensemble_size_max=ensemble_size_max,
            ci_margin_factor=ci_margin_factor,
            ci_sample_sizes=ci_sample_sizes,
            crossover_probability=crossover_probability,
            ci_width_scope=ci_width_scope,
            ensemble_algorithm=ensemble_algorithm,
        )
        return ensemble_result

    def select_ensemble_with_predictions(
        self,
        representatives: list[CalibrationEvaluation],
        population_size: int,
        generations: int,
        confidence_level: float,
        pareto_preference: float,
        ensemble_size_mode: str,
        ensemble_size: int | None,
        ensemble_size_min: int | None,
        ensemble_size_max: int | None,
        ci_margin_factor: float,
        ci_sample_sizes: list[int],
        crossover_probability: float,
        ci_width_scope: str = "full_trajectory",
        ensemble_algorithm: str = "nsga2",
    ) -> tuple["EnsembleSelectionResultProtocol", list[CalibrationEvaluation]]:
        """Run ensemble selection and return generated predictions.

        Returns
        -------
        tuple[object, list[CalibrationEvaluation]]
            Rust EnsembleSelectionResult object and representatives with predictions.
        """

        logger.info(
            "Running %s ensemble selection on %s candidates",
            ensemble_algorithm,
            len(representatives),
        )

        # Generate predictions for each representative in parallel. Compact scopes
        # only generate metric points needed by the selector; full output trajectories
        # are generated later for selected/full result detail.
        requires_observation_level = self._requires_observation_level_selection()
        if ci_width_scope == "full_trajectory" and not requires_observation_level:
            representatives_for_selection = self._generate_predictions(representatives)
            representatives_with_predictions = representatives_for_selection
            observed_data_tuples = [
                (
                    obs.step,
                    self._simulation_output_to_idx[obs.compartment],
                    obs.value,
                )
                for obs in self.problem.observed_data
            ]
            selection_ci_width_scope = "full_trajectory"
        else:
            metric_scope = (
                "observed_points" if requires_observation_level else ci_width_scope
            )
            metric_points, observed_data_tuples, observation_specs = (
                self._selection_metric_points_for_scope(metric_scope)
            )
            representatives_for_selection = self._generate_compact_predictions(
                representatives,
                metric_points,
                observation_specs,
            )
            representatives_with_predictions = representatives
            selection_ci_width_scope = "full_trajectory"

        candidates = [
            commol_rs.calibration.CalibrationEvaluation(
                parameters=rep.parameters,
                loss=rep.loss,
                predictions=rep.predictions or [],
            )
            for rep in representatives_for_selection
        ]

        # Run ensemble selection
        logger.info("Running ensemble selection...")
        ensemble_result = commol_rs.calibration.select_optimal_ensemble(
            candidates=candidates,
            observed_data_tuples=observed_data_tuples,
            population_size=population_size,
            generations=generations,
            confidence_level=confidence_level,
            seed=self.seed,
            pareto_preference=pareto_preference,
            ensemble_size_mode=ensemble_size_mode,
            ensemble_size=ensemble_size,
            ensemble_size_min=ensemble_size_min,
            ensemble_size_max=ensemble_size_max,
            ci_margin_factor=ci_margin_factor,
            ci_sample_sizes=ci_sample_sizes,
            crossover_probability=crossover_probability,
            ci_width_scope=selection_ci_width_scope,
            ensemble_algorithm=ensemble_algorithm,
        )

        logger.info(
            f"Selected ensemble of {len(ensemble_result.selected_ensemble)} parameter "
            f"sets using {ensemble_algorithm}"
        )
        logger.info(
            f"Pareto front contains {len(ensemble_result.pareto_front)} solutions"
        )

        return ensemble_result, representatives_with_predictions

    def _generate_predictions(
        self,
        representatives: list[CalibrationEvaluation],
    ) -> list[CalibrationEvaluation]:
        """Generate predictions for representative parameter sets in parallel.

        Parameters
        ----------
        representatives : list[CalibrationEvaluation]
            List of parameter sets to generate predictions for

        Returns
        -------
        list[CalibrationEvaluation]
            Representatives with predictions attached
        """
        logger.info(
            "Generating predictions for representative parameter sets in parallel..."
        )

        max_time_step = max(obs.step for obs in self.problem.observed_data)
        time_steps = max_time_step + 1

        parameter_sets = [rep.parameters for rep in representatives]

        all_predictions = (
            commol_rs.calibration.generate_calibrated_predictions_parallel(
                self.simulation.engine,
                self._calibration_runner.build_rust_observed_data(),
                self._calibration_runner.build_rust_parameters(),
                self._calibration_runner.build_rust_constraints(),
                self._calibration_runner.build_loss_config(),
                self._calibration_runner.initial_population_size(),
                parameter_sets,
                time_steps,
            )
        )

        # Combine predictions with representative data
        result: list[CalibrationEvaluation] = []
        for rep, predictions in zip(representatives, all_predictions):
            result.append(
                CalibrationEvaluation(
                    parameters=rep.parameters,
                    loss=rep.loss,
                    parameter_names=rep.parameter_names,
                    predictions=predictions,
                )
            )

        return result

    def _metric_points_for_scope(
        self,
        ci_width_scope: str,
    ) -> tuple[list[MetricPoint], list[ObservedTuple]]:
        metric_points, observed_data_tuples, _ = (
            self._selection_metric_points_for_scope(ci_width_scope)
        )
        return metric_points, observed_data_tuples

    def _selection_metric_points_for_scope(
        self,
        ci_width_scope: str,
    ) -> tuple[list[MetricPoint], list[ObservedTuple], list[ObservationSpec] | None]:
        """Build compact metric points and remapped observed tuples."""
        metric_points = self._metric_points_for_ci_scope(ci_width_scope)
        deduplicated_points, point_to_compact_idx = self._deduplicate_metric_points(
            metric_points
        )

        if not self._requires_observation_level_selection():
            observed_tuples = self._raw_observed_tuples(point_to_compact_idx)
            return deduplicated_points, observed_tuples, None

        observed_tuples, observation_specs = self._observation_level_specs(
            point_to_compact_idx
        )
        return deduplicated_points, observed_tuples, observation_specs

    def _requires_observation_level_selection(self) -> bool:
        """Return whether observations need aggregate/window/scale transforms."""
        return any(
            obs.window_steps is not None
            or obs.compartments is not None
            or obs.scale_id is not None
            for obs in self.problem.observed_data
        )

    def _metric_points_for_ci_scope(self, ci_width_scope: str) -> list[MetricPoint]:
        """Build metric points for a compact CI-width scope."""
        if ci_width_scope == "observed_points":
            return self._observed_point_metric_points()
        if ci_width_scope == "observed_steps_all_compartments":
            observed_steps = sorted({obs.step for obs in self.problem.observed_data})
            return [
                (step, output_idx)
                for step in observed_steps
                for output_idx in range(len(self.simulation.simulation_outputs))
            ]
        raise ValueError(
            "ci_width_scope must be 'observed_points', "
            "'observed_steps_all_compartments', or 'full_trajectory'"
        )

    def _observed_point_metric_points(self) -> list[MetricPoint]:
        """Build compact metric points for observed outputs only."""
        metric_points = []
        for obs in self.problem.observed_data:
            output_ids = obs.compartments or [obs.compartment]
            for output_id in output_ids:
                output_idx = self._simulation_output_to_idx[output_id]
                metric_points.append((obs.step, output_idx))
                if obs.window_steps is not None:
                    metric_points.append((obs.step - obs.window_steps, output_idx))
        return metric_points

    @staticmethod
    def _deduplicate_metric_points(
        metric_points: list[MetricPoint],
    ) -> tuple[list[MetricPoint], dict[MetricPoint, int]]:
        """Return metric points with stable deduplication and index mapping."""
        point_to_compact_idx: dict[MetricPoint, int] = {}
        deduplicated_points: list[MetricPoint] = []
        for point in metric_points:
            if point not in point_to_compact_idx:
                point_to_compact_idx[point] = len(deduplicated_points)
                deduplicated_points.append(point)
        return deduplicated_points, point_to_compact_idx

    def _raw_observed_tuples(
        self,
        point_to_compact_idx: dict[MetricPoint, int],
    ) -> list[ObservedTuple]:
        """Build observed tuples that directly reference compact raw predictions."""
        return [
            (
                point_to_compact_idx[
                    (obs.step, self._simulation_output_to_idx[obs.compartment])
                ],
                0,
                obs.value,
            )
            for obs in self.problem.observed_data
        ]

    def _observation_level_specs(
        self,
        point_to_compact_idx: dict[MetricPoint, int],
    ) -> tuple[list[ObservedTuple], list[ObservationSpec]]:
        """Build observed tuples and transform specs for observation-level rows."""
        observed_data_tuples: list[ObservedTuple] = []
        observation_specs: list[ObservationSpec] = []
        for obs_index, obs in enumerate(self.problem.observed_data):
            output_ids = obs.compartments or [obs.compartment]
            observation_components = []
            for output_id in output_ids:
                output_idx = self._simulation_output_to_idx[output_id]
                current_point_idx = point_to_compact_idx[(obs.step, output_idx)]
                previous_point_idx = (
                    point_to_compact_idx[(obs.step - obs.window_steps, output_idx)]
                    if obs.window_steps is not None
                    else None
                )
                observation_components.append((current_point_idx, previous_point_idx))
            observed_data_tuples.append((obs_index, 0, obs.value))
            observation_specs.append((observation_components, obs.scale_id))

        return observed_data_tuples, observation_specs

    def _generate_compact_predictions(
        self,
        representatives: list[CalibrationEvaluation],
        metric_points: list[MetricPoint],
        observation_specs: list[ObservationSpec] | None = None,
    ) -> list[CalibrationEvaluation]:
        """Generate compact predictions for representative parameter sets."""
        logger.info(
            "Generating compact predictions for representative parameter sets..."
        )
        parameter_sets = [rep.parameters for rep in representatives]

        compact_predictions = (
            commol_rs.calibration.generate_calibrated_predictions_at_points_parallel(
                self.simulation.engine,
                self._calibration_runner.build_rust_observed_data(),
                self._calibration_runner.build_rust_parameters(),
                self._calibration_runner.build_rust_constraints(),
                self._calibration_runner.build_loss_config(),
                self._calibration_runner.initial_population_size(),
                parameter_sets,
                metric_points,
            )
        )

        result: list[CalibrationEvaluation] = []
        for rep, predictions in zip(representatives, compact_predictions):
            if observation_specs is None:
                compact_rows = [[value] for value in predictions]
            else:
                parameter_indices = {
                    param_id: idx for idx, param_id in enumerate(rep.parameter_names)
                }
                compact_rows = [
                    [
                        self._scale_observation_value(
                            sum(
                                predictions[current_idx]
                                - (
                                    predictions[previous_idx]
                                    if previous_idx is not None
                                    else 0.0
                                )
                                for current_idx, previous_idx in observation_components
                            ),
                            scale_id,
                            rep.parameters,
                            parameter_indices,
                        )
                    ]
                    for observation_components, scale_id in observation_specs
                ]
            result.append(
                CalibrationEvaluation(
                    parameters=rep.parameters,
                    loss=rep.loss,
                    parameter_names=rep.parameter_names,
                    predictions=compact_rows,
                )
            )

        return result

    @staticmethod
    def _scale_observation_value(
        value: float,
        scale_id: str | None,
        parameters: list[float],
        parameter_indices: dict[str, int],
    ) -> float:
        if scale_id is None:
            return value
        try:
            scale_idx = parameter_indices[scale_id]
        except KeyError as exc:
            raise ValueError(
                f"Scale parameter '{scale_id}' is not present in calibration "
                "evaluation parameter names."
            ) from exc
        return value * parameters[scale_idx]
