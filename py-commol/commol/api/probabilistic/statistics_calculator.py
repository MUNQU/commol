"""Statistics calculator for ensemble analysis.

This module handles computing statistics from the selected ensemble,
including parameter statistics, prediction intervals, and coverage metrics.
"""

import logging
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from commol.commol_rs import _commol_rs as commol_rs
from commol.api.probabilistic.calibration_runner import CalibrationRunner
from commol.api.probabilistic.intervals import ci_percentiles
from commol.api.probabilistic.normalization import (
    central_fit_loss,
    series_normalization_factors,
)
from commol.api.results import index_at, same_position

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.context.calibration import CalibrationProblem, ObservedDataPoint

from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    ParameterSetStatistics,
)

logger = logging.getLogger(__name__)


def prediction_positions(observed_data: Iterable["ObservedDataPoint"]) -> list[float]:
    """
    Positions the ensemble trajectories are reported at.

    The unit grid ``0, 1, ..., floor(last) + 1``, where ``last`` is the last
    observation position, and every observation position and window start,
    sorted and unique. A DifferenceEquations model observes at integers only,
    so its positions are the grid.

    Parameters
    ----------
    observed_data : Iterable[ObservedDataPoint]
        The observations of the problem; there is at least one.

    Returns
    -------
    list[float]
        Ascending positions.
    """
    points: list[float] = []
    for obs in observed_data:
        points.append(float(obs.at))
        if obs.window is not None:
            points.append(float(obs.at - obs.window))
    grid = [float(k) for k in range(math.floor(max(points)) + 2)]
    positions: list[float] = []
    for position in sorted(grid + points):
        if not positions or not same_position(positions[-1], position):
            positions.append(position)
    return positions


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
        self.positions: list[float] = prediction_positions(problem.observed_data)
        """Positions every ensemble trajectory holds, ascending."""

    def _member_value(
        self,
        all_predictions: dict[str, list[list[float]]],
        output: str,
        member_idx: int,
        at: float,
    ) -> float:
        """Value of one output for one ensemble member at a position."""
        return all_predictions[output][member_idx][index_at(self.positions, at)]

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

        ci_lower_percentile, ci_upper_percentile = ci_percentiles(self.confidence_level)

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
    ) -> dict[str, list[list[float]]]:
        """Generate the trajectory of each ensemble member at :attr:`positions`.

        Parameters
        ----------
        ensemble_params : list[CalibrationEvaluation]
            List of parameter sets in the ensemble
        simulation_output_ids : list[str]
            List of simulation output IDs to generate predictions for

        Returns
        -------
        dict[str, list[list[float]]]
            Each output mapped to one trajectory per member, with one value
            per position of :attr:`positions`.
        """
        runner = self._calibration_runner
        all_predictions_raw = (
            commol_rs.calibration.generate_calibrated_predictions_parallel(
                self.simulation.engine,
                runner.build_rust_observed_data(),
                runner.build_rust_parameters(),
                runner.build_rust_constraints(),
                runner.build_loss_config(),
                [ep.parameters for ep in ensemble_params],
                self.positions,
                initial_condition_remainder=runner.problem.initial_condition_remainder,
            )
        )

        # all_predictions_raw[member][position][output] -> output: member: position
        output_index = {
            output_id: idx
            for idx, output_id in enumerate(self.simulation.simulation_outputs)
        }
        return {
            output_id: [
                [row[output_index[output_id]] for row in member]
                for member in all_predictions_raw
            ]
            for output_id in simulation_output_ids
        }

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

        ci_lower_percentile, ci_upper_percentile = ci_percentiles(self.confidence_level)

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

    def calculate_windowed_prediction_intervals(
        self,
        all_predictions: dict[str, list[list[float]]],
        ensemble_params: list[CalibrationEvaluation],
    ) -> tuple[
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
        dict[str, list[float]],
    ]:
        """Calculate median and CI from windowed (per-period) trajectories.

        For outputs with windowed observations, computes the windowed value
        (the series at the position minus the series a window before) per
        ensemble member *before* taking percentiles, so the resulting CI is the
        percentile of differences rather than the difference of percentiles.

        Each value is taken at the position of an observation, using that
        observation's own window, so the result lines up with the observations
        of that output sorted by position and matches what the loss compared
        against.

        Also returns the position each windowed value was taken at.

        Returns empty dicts if no observed data sets a window.
        """
        windowed_observations: dict[
            str,
            tuple[list[str], str | None, list[tuple[float, float]]],
        ] = {}
        for obs in self.problem.observed_data:
            if obs.window is None:
                continue
            entry = windowed_observations.setdefault(
                obs.label,
                (obs.observed_outputs, obs.factor_id, []),
            )
            entry[2].append((float(obs.at), float(obs.window)))

        prediction_median: dict[str, list[float]] = {}
        prediction_ci_lower: dict[str, list[float]] = {}
        prediction_ci_upper: dict[str, list[float]] = {}
        prediction_positions: dict[str, list[float]] = {}

        ci_lower_percentile, ci_upper_percentile = ci_percentiles(self.confidence_level)

        for output_id, (
            component_ids,
            factor_id,
            observation_windows,
        ) in windowed_observations.items():
            if any(
                component_id not in all_predictions for component_id in component_ids
            ):
                continue
            windows = sorted(set(observation_windows))
            windowed = np.array(
                [
                    [
                        self._member_window_value(
                            all_predictions,
                            ensemble_params,
                            component_ids,
                            factor_id,
                            member_idx,
                            at,
                            window,
                        )
                        for member_idx in range(len(ensemble_params))
                    ]
                    for at, window in windows
                ]
            ).T

            prediction_positions[output_id] = [at for at, _ in windows]
            prediction_median[output_id] = np.median(windowed, axis=0).tolist()
            prediction_ci_lower[output_id] = np.percentile(
                windowed, ci_lower_percentile, axis=0
            ).tolist()
            prediction_ci_upper[output_id] = np.percentile(
                windowed, ci_upper_percentile, axis=0
            ).tolist()

        return (
            prediction_median,
            prediction_ci_lower,
            prediction_ci_upper,
            prediction_positions,
        )

    def _member_window_value(
        self,
        all_predictions: dict[str, list[list[float]]],
        ensemble_params: list[CalibrationEvaluation],
        component_ids: list[str],
        factor_id: str | None,
        member_idx: int,
        at: float,
        window: float,
    ) -> float:
        """Summed, optionally scaled windowed value of one ensemble member."""
        total = sum(
            self._member_value(all_predictions, component_id, member_idx, at)
            - self._member_value(all_predictions, component_id, member_idx, at - window)
            for component_id in component_ids
        )
        return self._apply_factor(total, factor_id, ensemble_params[member_idx])

    def calculate_central_loss(
        self,
        all_predictions: dict[str, list[list[float]]],
        ensemble_params: list[CalibrationEvaluation],
    ) -> float:
        """Evaluate the memberwise-median prediction with the optimizer's loss.

        Uses the same loss function and per-series normalization as the
        optimization loss and the fit-gated selection gate, so the reported
        central loss is comparable to the member losses it is gated against.
        """
        normalization = series_normalization_factors(
            self.problem.observed_data,
            self.problem.normalize_observations,
        )
        residuals: list[float] = []
        weights: list[float] = []
        factors: list[float] = []
        for observation in self.problem.observed_data:
            values = self._observation_member_values(
                observation,
                all_predictions,
                ensemble_params,
            )
            if not values:
                continue
            residuals.append(float(np.median(values)) - observation.value)
            weights.append(observation.weight)
            factors.append(normalization[observation.label])
        return central_fit_loss(residuals, weights, factors, self.problem.loss_function)

    def calculate_observation_diagnostics(
        self,
        all_predictions: dict[str, list[list[float]]],
        ensemble_params: list[CalibrationEvaluation],
    ) -> dict[str, dict[str, float]]:
        """Return coverage and interval width separately for each data series."""
        lower_percentile, upper_percentile = ci_percentiles(self.confidence_level)
        totals: dict[str, dict[str, float]] = {}

        for observation in self.problem.observed_data:
            values = self._observation_member_values(
                observation,
                all_predictions,
                ensemble_params,
            )
            if not values:
                continue
            lower = float(np.percentile(values, lower_percentile))
            upper = float(np.percentile(values, upper_percentile))
            diagnostics = totals.setdefault(
                observation.label,
                {"n_points": 0.0, "covered_points": 0.0, "total_ci_width": 0.0},
            )
            diagnostics["n_points"] += 1.0
            diagnostics["covered_points"] += float(lower <= observation.value <= upper)
            diagnostics["total_ci_width"] += upper - lower

        return {
            series: {
                "n_points": values["n_points"],
                "coverage_percentage": (
                    values["covered_points"] / values["n_points"] * 100.0
                ),
                "average_ci_width": values["total_ci_width"] / values["n_points"],
            }
            for series, values in totals.items()
            if values["n_points"] > 0.0
        }

    def calculate_coverage_metrics(
        self,
        prediction_ci_lower: dict[str, list[float]],
        prediction_ci_upper: dict[str, list[float]],
        all_predictions: dict[str, list[list[float]]] | None = None,
        ensemble_params: list[CalibrationEvaluation] | None = None,
    ) -> tuple[float, float]:
        """Calculate coverage percentage and average CI width.

        Parameters
        ----------
        prediction_ci_lower : dict[str, list[float]]
            Lower CI bounds for each compartment
        prediction_ci_upper : dict[str, list[float]]
            Upper CI bounds for each compartment
        all_predictions : dict[str, list[list[float]]] | None
            Optional selected-member trajectories. When provided, observation-level
            intervals are computed from member predictions directly, preserving
            aggregate, windowed, and scale-parameter correlations.
        ensemble_params : list[CalibrationEvaluation] | None
            Selected ensemble parameter sets, required for scaled observations.

        Returns
        -------
        tuple[float, float]
            Tuple of (coverage_percentage, average_ci_width)
        """
        if all_predictions is not None and ensemble_params is not None:
            return self._calculate_observation_level_coverage(
                all_predictions,
                ensemble_params,
            )

        points_in_ci = 0
        total_points = len(self.problem.observed_data)
        total_ci_width = 0.0

        for obs in self.problem.observed_data:
            component_ids = obs.observed_outputs
            if any(
                component_id not in prediction_ci_lower
                for component_id in component_ids
            ):
                continue
            end = index_at(self.positions, obs.at)
            if obs.window is None:
                ci_lower = sum(
                    prediction_ci_lower[component_id][end]
                    for component_id in component_ids
                )
                ci_upper = sum(
                    prediction_ci_upper[component_id][end]
                    for component_id in component_ids
                )
            else:
                start = index_at(self.positions, obs.at - obs.window)
                ci_lower = sum(
                    prediction_ci_lower[component_id][end]
                    - prediction_ci_upper[component_id][start]
                    for component_id in component_ids
                )
                ci_upper = sum(
                    prediction_ci_upper[component_id][end]
                    - prediction_ci_lower[component_id][start]
                    for component_id in component_ids
                )

            if ci_lower <= obs.value <= ci_upper:
                points_in_ci += 1

            total_ci_width += ci_upper - ci_lower

        coverage_percentage = (
            (points_in_ci / total_points * 100) if total_points > 0 else 0.0
        )
        average_ci_width = total_ci_width / total_points if total_points > 0 else 0.0

        return coverage_percentage, average_ci_width

    def _calculate_observation_level_coverage(
        self,
        all_predictions: dict[str, list[list[float]]],
        ensemble_params: list[CalibrationEvaluation],
    ) -> tuple[float, float]:
        points_in_ci = 0
        total_points = len(self.problem.observed_data)
        total_ci_width = 0.0

        ci_lower_percentile, ci_upper_percentile = ci_percentiles(self.confidence_level)

        for obs in self.problem.observed_data:
            values = self._observation_member_values(
                obs,
                all_predictions,
                ensemble_params,
            )
            if not values:
                continue

            ci_lower = float(np.percentile(values, ci_lower_percentile))
            ci_upper = float(np.percentile(values, ci_upper_percentile))
            if ci_lower <= obs.value <= ci_upper:
                points_in_ci += 1
            total_ci_width += ci_upper - ci_lower

        coverage_percentage = (
            (points_in_ci / total_points * 100) if total_points > 0 else 0.0
        )
        average_ci_width = total_ci_width / total_points if total_points > 0 else 0.0
        return coverage_percentage, average_ci_width

    def _observation_member_values(
        self,
        obs,
        all_predictions: dict[str, list[list[float]]],
        ensemble_params: list[CalibrationEvaluation],
    ) -> list[float]:
        component_ids = obs.observed_outputs
        if any(component_id not in all_predictions for component_id in component_ids):
            return []

        n_members = len(ensemble_params)
        if any(
            len(all_predictions[component_id]) < n_members
            for component_id in component_ids
        ):
            return []

        values = []
        for member_idx, params in enumerate(ensemble_params):
            value = 0.0
            for component_id in component_ids:
                value += self._member_value(
                    all_predictions, component_id, member_idx, obs.at
                )
                if obs.window is not None:
                    value -= self._member_value(
                        all_predictions,
                        component_id,
                        member_idx,
                        obs.at - obs.window,
                    )
            values.append(self._apply_factor(value, obs.factor_id, params))
        return values

    @staticmethod
    def _apply_factor(
        value: float,
        factor_id: str | None,
        params: CalibrationEvaluation,
    ) -> float:
        if factor_id is None:
            return value
        try:
            factor_idx = params.parameter_names.index(factor_id)
        except ValueError as exc:
            raise ValueError(
                f"Factor parameter '{factor_id}' is not present in calibration "
                "evaluation parameter names."
            ) from exc
        return value * params.parameters[factor_idx]
