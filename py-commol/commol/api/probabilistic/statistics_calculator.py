"""Statistics calculator for ensemble analysis.

This module handles computing statistics from the selected ensemble,
including parameter statistics, prediction intervals, and coverage metrics.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from commol.commol_rs import _commol_rs as commol_rs
from commol.api.probabilistic.calibration_runner import CalibrationRunner
from commol.api.probabilistic.intervals import ci_percentiles
from commol.api.probabilistic.normalization import (
    central_fit_loss,
    series_normalization_factors,
)

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.context.calibration import CalibrationProblem

from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    ParameterSetStatistics,
)

logger = logging.getLogger(__name__)


def _as_step(position: int | float) -> int | float:
    """A whole-number position as an int, so it indexes a trajectory."""
    if isinstance(position, float) and position.is_integer():
        return int(position)
    return position


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
        self._point_values_cache: (
            tuple[tuple[int, ...], dict[tuple[str, float], list[float]]] | None
        ) = None

    def _non_integer_points(self) -> list[tuple[str, float]]:
        """``(output, step)`` of every observation read at a non-integer step."""
        output_ids = set(self.simulation.simulation_outputs)
        points: list[tuple[str, float]] = []
        for obs in self.problem.observed_data:
            positions = [obs.step]
            if obs.window_steps is not None:
                positions.append(obs.step - obs.window_steps)
            for position in positions:
                if float(position).is_integer():
                    continue
                for output in obs.compartments or [obs.compartment]:
                    if output in output_ids and (output, position) not in points:
                        points.append((output, position))
        return points

    def _point_values(
        self,
        ensemble_params: list[CalibrationEvaluation],
    ) -> dict[tuple[str, float], list[float]]:
        """
        Exact value of each output at each non-integer step the problem uses.

        Ensemble trajectories hold integer steps only. A DifferentialEquations
        model may observe between them, so the engine reports those positions
        directly, once per ensemble. Keys are ``(output, step)``; each value
        has one entry per ensemble member.
        """
        cache_key = tuple(id(params) for params in ensemble_params)
        if self._point_values_cache is not None:
            cached_key, cached_values = self._point_values_cache
            if cached_key == cache_key:
                return cached_values

        points = self._non_integer_points()
        values: dict[tuple[str, float], list[float]] = {point: [] for point in points}
        if points:
            output_index = {
                name: idx for idx, name in enumerate(self.simulation.simulation_outputs)
            }
            runner = self._calibration_runner
            generate = (
                commol_rs.calibration.generate_calibrated_predictions_at_points_parallel
            )
            predictions = generate(
                self.simulation.engine,
                runner.build_rust_observed_data(),
                runner.build_rust_parameters(),
                runner.build_rust_constraints(),
                runner.build_loss_config(),
                runner.initial_population_size(),
                [params.parameters for params in ensemble_params],
                [(float(step), output_index[output]) for output, step in points],
                initial_condition_remainder=runner.problem.initial_condition_remainder,
            )
            for member in predictions:
                for point, value in zip(points, member, strict=True):
                    values[point].append(value)
        self._point_values_cache = (cache_key, values)
        return values

    def _member_value(
        self,
        all_predictions: dict[str, list[list[float]]],
        ensemble_params: list[CalibrationEvaluation],
        output: str,
        member_idx: int,
        step: int | float,
    ) -> float | None:
        """
        Value of one output for one ensemble member at a step, or None when
        the trajectory does not reach that step.
        """
        if isinstance(step, int):
            trajectory = all_predictions[output][member_idx]
            return trajectory[step] if step < len(trajectory) else None
        values = self._point_values(ensemble_params).get((output, step))
        return values[member_idx] if values else None

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
                    initial_condition_remainder=(
                        self._calibration_runner.problem.initial_condition_remainder
                    ),
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
        dict[str, list[int | float]],
    ]:
        """Calculate median and CI from windowed (per-period) trajectories.

        For outputs with windowed observations, computes the windowed value
        (series[t] - series[t - window_steps]) per ensemble member *before*
        taking percentiles, so the resulting CI is the percentile of differences
        rather than the difference of percentiles.

        Each value is taken at the step of an observation, using that
        observation's own window, so the result lines up with the observations
        of that output sorted by step and matches what the loss compared
        against.

        Also returns the step each windowed value was taken at.

        Returns empty dicts if no observed data has window_steps set.
        """
        windowed_observations: dict[
            str,
            tuple[list[str], str | None, list[tuple[int | float, int | float]]],
        ] = {}
        for obs in self.problem.observed_data:
            if obs.window_steps is None:
                continue
            entry = windowed_observations.setdefault(
                obs.compartment,
                (obs.compartments or [obs.compartment], obs.scale_id, []),
            )
            entry[2].append((obs.step, obs.window_steps))

        prediction_median: dict[str, list[float]] = {}
        prediction_ci_lower: dict[str, list[float]] = {}
        prediction_ci_upper: dict[str, list[float]] = {}
        prediction_steps: dict[str, list[int | float]] = {}

        ci_lower_percentile, ci_upper_percentile = ci_percentiles(self.confidence_level)

        for output_id, (
            component_ids,
            scale_id,
            observation_windows,
        ) in windowed_observations.items():
            if any(
                component_id not in all_predictions for component_id in component_ids
            ):
                continue
            windows = sorted(set(observation_windows))
            windowed_rows: list[list[float]] = []
            kept: list[tuple[int | float, int | float]] = []
            for step, window_steps in windows:
                row = [
                    self._member_window_value(
                        all_predictions,
                        ensemble_params,
                        component_ids,
                        scale_id,
                        member_idx,
                        step,
                        window_steps,
                    )
                    for member_idx in range(len(ensemble_params))
                ]
                values = [value for value in row if value is not None]
                if step - window_steps < 0 or len(values) != len(row):
                    continue
                kept.append((step, window_steps))
                windowed_rows.append(values)
            if not kept:
                continue
            windows = kept
            windowed = np.array(windowed_rows).T

            prediction_steps[output_id] = [step for step, _ in windows]
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
            prediction_steps,
        )

    def _member_window_value(
        self,
        all_predictions: dict[str, list[list[float]]],
        ensemble_params: list[CalibrationEvaluation],
        component_ids: list[str],
        scale_id: str | None,
        member_idx: int,
        step: int | float,
        window_steps: int | float,
    ) -> float | None:
        """Summed, optionally scaled windowed value of one ensemble member."""
        total = 0.0
        for component_id in component_ids:
            end = self._member_value(
                all_predictions, ensemble_params, component_id, member_idx, step
            )
            start = self._member_value(
                all_predictions,
                ensemble_params,
                component_id,
                member_idx,
                _as_step(step - window_steps),
            )
            if end is None or start is None:
                return None
            total += end - start
        return self._scale_observation_value(
            total, scale_id, ensemble_params[member_idx]
        )

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
            factors.append(normalization[observation.compartment])
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
                observation.compartment,
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
            comp_id = obs.compartment
            component_ids = obs.compartments or [comp_id]
            step = obs.step
            observed_value = obs.value
            previous_step = (
                _as_step(step - obs.window_steps) if obs.window_steps is not None else 0
            )
            # The intervals hold integer steps only.
            if not isinstance(step, int) or not isinstance(previous_step, int):
                continue

            if all(
                component_id in prediction_ci_lower
                and step < len(prediction_ci_lower[component_id])
                for component_id in component_ids
            ):
                if obs.window_steps is None:
                    ci_lower = sum(
                        prediction_ci_lower[component_id][step]
                        for component_id in component_ids
                    )
                    ci_upper = sum(
                        prediction_ci_upper[component_id][step]
                        for component_id in component_ids
                    )
                else:
                    ci_lower = sum(
                        prediction_ci_lower[component_id][step]
                        - prediction_ci_upper[component_id][previous_step]
                        for component_id in component_ids
                    )
                    ci_upper = sum(
                        prediction_ci_upper[component_id][step]
                        - prediction_ci_lower[component_id][previous_step]
                        for component_id in component_ids
                    )

                if ci_lower <= observed_value <= ci_upper:
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
        component_ids = obs.compartments or [obs.compartment]
        if any(component_id not in all_predictions for component_id in component_ids):
            return []

        n_members = len(ensemble_params)
        if any(
            len(all_predictions[component_id]) < n_members
            for component_id in component_ids
        ):
            return []

        values = []
        previous_step = (
            _as_step(obs.step - obs.window_steps)
            if obs.window_steps is not None
            else None
        )
        for member_idx, params in enumerate(ensemble_params):
            value = 0.0
            for component_id in component_ids:
                component_value = self._member_value(
                    all_predictions, ensemble_params, component_id, member_idx, obs.step
                )
                if component_value is None:
                    return []
                if previous_step is not None:
                    previous_value = self._member_value(
                        all_predictions,
                        ensemble_params,
                        component_id,
                        member_idx,
                        previous_step,
                    )
                    if previous_value is None:
                        return []
                    component_value -= previous_value
                value += component_value
            values.append(self._scale_observation_value(value, obs.scale_id, params))
        return values

    @staticmethod
    def _scale_observation_value(
        value: float,
        scale_id: str | None,
        params: CalibrationEvaluation,
    ) -> float:
        if scale_id is None:
            return value
        try:
            scale_idx = params.parameter_names.index(scale_id)
        except ValueError as exc:
            raise ValueError(
                f"Scale parameter '{scale_id}' is not present in calibration "
                "evaluation parameter names."
            ) from exc
        return value * params.parameters[scale_idx]
