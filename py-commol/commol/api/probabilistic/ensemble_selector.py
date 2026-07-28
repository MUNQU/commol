"""Ensemble selection for probabilistic calibration."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from commol.commol_rs import _commol_rs as commol_rs
from commol.api.probabilistic.calibration_runner import CalibrationRunner
from commol.api.probabilistic.normalization import series_normalization_factors

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.context.calibration import CalibrationProblem

from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    EnsembleSelectionSummary,
    ProbGreedyLocalSearchConfig,
    ProbNsga2Config,
)

logger = logging.getLogger(__name__)

MetricPoint = tuple[int, int]
ObservationComponent = tuple[int, int | None]
ObservationSpec = tuple[list[ObservationComponent], str | None]
EnsembleAlgorithmName = Literal["nsga2", "greedy_local_search"]


@dataclass(frozen=True)
class FitGatedSelection:
    """Python-side selection result for fit-constrained ensembles."""

    ensemble_size: int
    selected_indices: list[int]
    ci_width: float
    coverage: float
    diagnostics: dict[str, float]
    algorithm: EnsembleAlgorithmName = "greedy_local_search"
    pareto_front: list[EnsembleSelectionSummary] | None = None
    selected_pareto_index: int | None = None


class EnsembleSelector:
    """Build a diverse ensemble subject to a central-fit constraint."""

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

    def select_ensemble_with_predictions(
        self,
        representatives: list[CalibrationEvaluation],
        confidence_level: float,
        selection_config: ProbNsga2Config | ProbGreedyLocalSearchConfig,
    ) -> tuple[FitGatedSelection, list[CalibrationEvaluation]]:
        """Select members in observed-data space and return original members."""
        ensemble_algorithm = self._algorithm_name(selection_config)
        logger.info(
            "Running %s ensemble selection on %s compact candidates",
            ensemble_algorithm,
            len(representatives),
        )
        metric_points, observation_specs = self._observation_prediction_spec()
        representatives_for_selection = self._generate_compact_predictions(
            representatives,
            metric_points,
            observation_specs,
        )
        result = self._select_compact_ensemble(
            representatives_for_selection,
            confidence_level=confidence_level,
            selection_config=selection_config,
        )
        logger.info(
            "Selected %s ensemble of %s parameter sets",
            ensemble_algorithm,
            result.ensemble_size,
        )
        return result, representatives

    def generate_observation_predictions(
        self,
        evaluations: list[CalibrationEvaluation],
    ) -> list[CalibrationEvaluation]:
        """Generate calibrated values in the exact observed-data space.

        These compact vectors are used for prediction-space clustering. They
        contain weekly differences, aggregates, and member-specific scale
        parameters, rather than raw compartment values.
        """
        metric_points, observation_specs = self._observation_prediction_spec()
        return self._generate_compact_predictions(
            evaluations,
            metric_points,
            observation_specs,
        )

    def _select_compact_ensemble(
        self,
        representatives: list[CalibrationEvaluation],
        *,
        confidence_level: float,
        selection_config: ProbNsga2Config | ProbGreedyLocalSearchConfig,
    ) -> FitGatedSelection:
        """Run a Rust selector on the compact observation-space candidates."""
        ensemble_algorithm = self._algorithm_name(selection_config)

        # Pass the raw weights and normalization factors separately, plus the
        # optimizer's loss function, so the Rust central-fit gate scores the
        # ensemble median with the exact same loss the members were fit with.
        normalization = series_normalization_factors(
            self.problem.observed_data,
            self.problem.normalize_observations,
        )
        observed_values = [
            observation.value for observation in self.problem.observed_data
        ]
        weights = [observation.weight for observation in self.problem.observed_data]
        normalization_factors = [
            normalization[observation.compartment]
            for observation in self.problem.observed_data
        ]
        series_ids = [
            observation.compartment for observation in self.problem.observed_data
        ]
        rust_candidates = [
            commol_rs.calibration.CalibrationEvaluation(
                candidate.parameters,
                candidate.loss,
                candidate.predictions or [],
            )
            for candidate in representatives
        ]
        # Arguments common to both backends. Each backend then receives only the
        # parameters it consumes, so the inactive algorithm's knobs stay at their
        # Rust defaults instead of being passed placeholder values.
        common_args = (
            rust_candidates,
            observed_values,
            weights,
            normalization_factors,
            series_ids,
            confidence_level,
            self.seed,
            self.problem.loss_function,
            ensemble_algorithm,
            selection_config.ensemble_size_mode,
            selection_config.ensemble_size,
            selection_config.ensemble_size_min,
            selection_config.ensemble_size_max,
        )
        if isinstance(selection_config, ProbNsga2Config):
            rust_result = commol_rs.calibration.select_compact_ensemble(
                *common_args,
                population_size=selection_config.population_size,
                generations=selection_config.generations,
                crossover_probability=selection_config.crossover_probability,
                pareto_preference=selection_config.pareto_preference,
            )
        else:
            rust_result = commol_rs.calibration.select_compact_ensemble(
                *common_args,
                central_fit_max_loss_ratio=selection_config.central_fit_max_loss_ratio,
                search_beam_width=selection_config.search_beam_width,
            )
        pareto_front = [
            EnsembleSelectionSummary(
                ensemble_size=solution.ensemble_size,
                selected_indices=list(solution.selected_indices),
                ci_width=solution.ci_width,
                coverage=solution.coverage,
                central_loss=solution.central_loss,
            )
            for solution in rust_result.pareto_front
        ] or None
        return FitGatedSelection(
            ensemble_size=len(rust_result.selected_ensemble),
            selected_indices=list(rust_result.selected_ensemble),
            ci_width=rust_result.ci_width,
            coverage=rust_result.coverage,
            diagnostics=dict(rust_result.diagnostics),
            algorithm=ensemble_algorithm,
            pareto_front=pareto_front,
            selected_pareto_index=rust_result.selected_pareto_index,
        )

    @staticmethod
    def _algorithm_name(
        selection_config: ProbNsga2Config | ProbGreedyLocalSearchConfig,
    ) -> EnsembleAlgorithmName:
        """Return the Rust algorithm identifier for a typed config."""
        if isinstance(selection_config, ProbNsga2Config):
            return "nsga2"
        if isinstance(selection_config, ProbGreedyLocalSearchConfig):
            return "greedy_local_search"
        raise TypeError(
            f"Unsupported ensemble selection config: {type(selection_config)!r}"
        )

    def _observation_prediction_spec(
        self,
    ) -> tuple[list[MetricPoint], list[ObservationSpec]]:
        """Build compact points for the transformed calibration observations."""
        metric_points = self._observed_point_metric_points()
        deduplicated_points, point_to_compact_idx = self._deduplicate_metric_points(
            metric_points
        )
        return deduplicated_points, self._observation_level_specs(point_to_compact_idx)

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

    def _observation_level_specs(
        self,
        point_to_compact_idx: dict[MetricPoint, int],
    ) -> list[ObservationSpec]:
        """Build aggregate/window/scale transforms for each observation."""
        observation_specs: list[ObservationSpec] = []
        for obs in self.problem.observed_data:
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
            observation_specs.append((observation_components, obs.scale_id))
        return observation_specs

    def _generate_compact_predictions(
        self,
        representatives: list[CalibrationEvaluation],
        metric_points: list[MetricPoint],
        observation_specs: list[ObservationSpec],
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
