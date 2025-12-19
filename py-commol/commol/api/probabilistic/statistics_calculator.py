"""Statistics calculator for ensemble analysis.

This module handles computing statistics from the selected ensemble,
including parameter statistics, prediction intervals, and coverage metrics.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from commol.commol_rs import commol_rs

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
        compartment_ids: list[str],
        time_steps: int,
    ) -> dict[str, list[list[float]]]:
        """Generate predictions for each ensemble member in parallel using Rust.

        Parameters
        ----------
        ensemble_params : list[CalibrationEvaluation]
            List of parameter sets in the ensemble
        compartment_ids : list[str]
            List of compartment IDs to generate predictions for
        time_steps : int
            Number of time steps to simulate

        Returns
        -------
        dict[str, list[list[float]]]
            Dictionary mapping compartment IDs to list of prediction trajectories
        """
        param_names = ensemble_params[0].parameter_names

        # Extract parameter sets
        parameter_sets = [ep.parameters for ep in ensemble_params]

        # Call Rust function to generate all predictions in parallel
        all_predictions_raw = commol_rs.calibration.generate_predictions_parallel(
            self.simulation.engine,
            parameter_sets,
            param_names,
            time_steps,
        )

        # Reorganize predictions by compartment
        # all_predictions_raw is list[list[list[float]]] where:
        # - outer list: one per parameter set
        # - middle list: one per time step
        # - inner list: one per compartment
        all_predictions: dict[str, list[list[float]]] = {
            comp_id: [] for comp_id in compartment_ids
        }

        compartment_idx_map = {
            bin.id: idx
            for idx, bin in enumerate(self.simulation.model_definition.population.bins)
        }

        for predictions_per_param_set in all_predictions_raw:
            # predictions_per_param_set is list[list[float]]
            # where [time_step][compartment_idx]
            for comp_id in compartment_ids:
                comp_idx = compartment_idx_map[comp_id]
                trajectory = [
                    predictions_per_param_set[t][comp_idx]
                    for t in range(len(predictions_per_param_set))
                ]
                all_predictions[comp_id].append(trajectory)

        return all_predictions

    def calculate_prediction_intervals(
        self,
        all_predictions: dict[str, list[list[float]]],
        compartment_ids: list[str],
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        """Calculate median and confidence intervals from ensemble predictions.

        Parameters
        ----------
        all_predictions : dict[str, list[list[float]]]
            Dictionary mapping compartment IDs to list of prediction trajectories
        compartment_ids : list[str]
            List of compartment IDs

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

        for comp_id in compartment_ids:
            predictions_array = np.array(all_predictions[comp_id])
            prediction_median[comp_id] = np.median(predictions_array, axis=0).tolist()
            prediction_ci_lower[comp_id] = np.percentile(
                predictions_array, ci_lower_percentile, axis=0
            ).tolist()
            prediction_ci_upper[comp_id] = np.percentile(
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
                ci_lower = prediction_ci_lower[comp_id][step]
                ci_upper = prediction_ci_upper[comp_id][step]

                if ci_lower <= observed_value <= ci_upper:
                    points_in_ci += 1

                total_ci_width += ci_upper - ci_lower

        coverage_percentage = (
            (points_in_ci / total_points * 100) if total_points > 0 else 0.0
        )
        average_ci_width = total_ci_width / total_points if total_points > 0 else 0.0

        return coverage_percentage, average_ci_width
