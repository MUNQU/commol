"""Statistics calculator for ensemble analysis.

This module handles computing statistics from the selected ensemble,
including parameter statistics, prediction intervals, and coverage metrics.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from commol.commol_rs import _commol_rs as commol_rs
from commol.api.probabilistic.calibration_runner import CalibrationRunner

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.context.calibration import CalibrationProblem

from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    ParameterSetStatistics,
)

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """Handles calculation of ensemble statistics.

    This class is responsible for:
    - Calculating parameter statistics across the ensemble
    - Generating ensemble predictions
    - Computing prediction intervals (median, CI bounds)
    - Calculating coverage metrics

    Parameters
    ----------
    simulation : Simulation
        A fully initialized Simulation object
    problem : CalibrationProblem
        The calibration problem definition
    confidence_level : float
        Confidence level for CI calculation (e.g., 0.95)
    """

    def __init__(
        self,
        simulation: "Simulation",
        problem: "CalibrationProblem",
        confidence_level: float = 0.95,
    ):
        self.simulation = simulation
        self.problem = problem
        self.confidence_level = confidence_level
        self._calibration_runner = CalibrationRunner(simulation, problem, seed=0)

    def calculate_parameter_statistics(
        self,
        ensemble_params: list[CalibrationEvaluation],
    ) -> dict[str, ParameterSetStatistics]:
        """Calculate statistics for each parameter across the ensemble.

        Parameters
        ----------
        ensemble_params : list[CalibrationEvaluation]
            List of parameter sets in the ensemble

        Returns
        -------
        dict[str, ParameterSetStatistics]
            Dictionary mapping parameter names to their statistics
        """
        param_names = ensemble_params[0].parameter_names
        param_values = {
            name: [p.parameters[i] for p in ensemble_params]
            for i, name in enumerate(param_names)
        }

        # Calculate percentile bounds based on confidence level
        ci_lower_percentile = (1.0 - self.confidence_level) / 2.0 * 100
        ci_upper_percentile = (1.0 + self.confidence_level) / 2.0 * 100

        param_statistics = {}
        for name, values in param_values.items():
            param_statistics[name] = ParameterSetStatistics(
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                std=float(np.std(values)),
                percentile_lower=float(np.percentile(values, ci_lower_percentile)),
                percentile_upper=float(np.percentile(values, ci_upper_percentile)),
                min=float(np.min(values)),
                max=float(np.max(values)),
            )
        return param_statistics

    def generate_ensemble_predictions(
        self,
        ensemble_params: list[CalibrationEvaluation],
        simulation_output_ids: list[str],
        time_steps: int,
    ) -> dict[str, list[list[float]]]:
        """Generate or reuse predictions for each ensemble member.

        Parameters
        ----------
        ensemble_params : list[CalibrationEvaluation]
            List of parameter sets in the ensemble
        simulation_output_ids : list[str]
            List of simulation output IDs to generate predictions for
        time_steps : int
            Number of time steps to simulate

        Returns
        -------
        dict[str, list[list[float]]]
            Dictionary mapping simulation output IDs to prediction trajectories
        """
        cached_predictions = [ep.predictions for ep in ensemble_params]
        n_outputs = len(self.simulation.simulation_outputs)
        if all(
            predictions is not None
            and len(predictions) == time_steps
            and all(len(step) >= n_outputs for step in predictions)
            for predictions in cached_predictions
        ):
            all_predictions_raw = [
                predictions
                for predictions in cached_predictions
                if predictions is not None
            ]
        else:
            parameter_sets = [ep.parameters for ep in ensemble_params]
            all_predictions_raw = (
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

        # Reorganize predictions by simulation output.
        # all_predictions_raw is list[list[list[float]]] where:
        # - outer list: one per parameter set
        # - middle list: one per time step
        # - inner list: one per simulation output
        all_predictions: dict[str, list[list[float]]] = {
            output_id: [] for output_id in simulation_output_ids
        }

        simulation_output_idx_map = {
            output_id: idx
            for idx, output_id in enumerate(self.simulation.simulation_outputs)
        }

        for predictions_per_param_set in all_predictions_raw:
            # predictions_per_param_set is list[list[float]]
            # where [time_step][simulation_output_idx]
            for output_id in simulation_output_ids:
                output_idx = simulation_output_idx_map[output_id]
                trajectory = [
                    predictions_per_param_set[t][output_idx]
                    for t in range(len(predictions_per_param_set))
                ]
                all_predictions[output_id].append(trajectory)

        return all_predictions

    def calculate_prediction_intervals(
        self,
        all_predictions: dict[str, list[list[float]]],
        simulation_output_ids: list[str],
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        """Calculate median and confidence intervals from ensemble predictions.

        Parameters
        ----------
        all_predictions : dict[str, list[list[float]]]
            Dictionary mapping compartment IDs to list of prediction trajectories
        simulation_output_ids : list[str]
            List of simulation output IDs

        Returns
        -------
        tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]
            Tuple of (median, lower CI, upper CI) dictionaries
        """
        prediction_median: dict[str, list[float]] = {}
        prediction_ci_lower: dict[str, list[float]] = {}
        prediction_ci_upper: dict[str, list[float]] = {}

        ci_lower_percentile = (1.0 - self.confidence_level) / 2.0 * 100
        ci_upper_percentile = (1.0 + self.confidence_level) / 2.0 * 100

        for output_id in simulation_output_ids:
            predictions_array = np.array(all_predictions[output_id])
            prediction_median[output_id] = np.median(predictions_array, axis=0).tolist()
            prediction_ci_lower[output_id] = np.percentile(
                predictions_array, ci_lower_percentile, axis=0
            ).tolist()
            prediction_ci_upper[output_id] = np.percentile(
                predictions_array, ci_upper_percentile, axis=0
            ).tolist()

        return prediction_median, prediction_ci_lower, prediction_ci_upper

    def calculate_coverage_metrics(
        self,
        prediction_ci_lower: dict[str, list[float]],
        prediction_ci_upper: dict[str, list[float]],
    ) -> tuple[float, float]:
        """Calculate coverage percentage and average CI width.

        Parameters
        ----------
        prediction_ci_lower : dict[str, list[float]]
            Lower CI bounds for each compartment
        prediction_ci_upper : dict[str, list[float]]
            Upper CI bounds for each compartment

        Returns
        -------
        tuple[float, float]
            Tuple of (coverage_percentage, average_ci_width)
        """
        points_in_ci = 0
        total_points = len(self.problem.observed_data)
        total_ci_width = 0.0

        for obs in self.problem.observed_data:
            comp_id = obs.compartment
            step = obs.step
            observed_value = obs.value

            if comp_id in prediction_ci_lower and step < len(
                prediction_ci_lower[comp_id]
            ):
                if obs.window_steps is None:
                    ci_lower = prediction_ci_lower[comp_id][step]
                    ci_upper = prediction_ci_upper[comp_id][step]
                else:
                    previous_step = step - obs.window_steps
                    ci_lower = (
                        prediction_ci_lower[comp_id][step]
                        - prediction_ci_upper[comp_id][previous_step]
                    )
                    ci_upper = (
                        prediction_ci_upper[comp_id][step]
                        - prediction_ci_lower[comp_id][previous_step]
                    )

                if ci_lower <= observed_value <= ci_upper:
                    points_in_ci += 1

                total_ci_width += ci_upper - ci_lower

        coverage_percentage = (
            (points_in_ci / total_points * 100) if total_points > 0 else 0.0
        )
        average_ci_width = total_ci_width / total_points if total_points > 0 else 0.0

        return coverage_percentage, average_ci_width
