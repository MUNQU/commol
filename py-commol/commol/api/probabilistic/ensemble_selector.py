"""Ensemble selector for NSGA-II based ensemble optimization.

This module handles the multi-objective optimization process that selects
an optimal ensemble of parameter sets balancing confidence interval width
and coverage of observed data.
"""

import logging
from typing import TYPE_CHECKING

from commol.commol_rs import commol_rs

if TYPE_CHECKING:
    from commol.api.simulation import Simulation
    from commol.commol_rs.commol_rs import EnsembleSelectionResultProtocol
    from commol.context.calibration import CalibrationProblem

from commol.context.probabilistic_calibration import CalibrationEvaluation

logger = logging.getLogger(__name__)


class EnsembleSelector:
    """Handles NSGA-II ensemble selection.

    This class is responsible for:
    - Generating predictions for candidate parameter sets
    - Running NSGA-II multi-objective optimization
    - Selecting Pareto-optimal ensemble

    Parameters
    ----------
    simulation : Simulation
        A fully initialized Simulation object
    problem : CalibrationProblem
        The calibration problem definition
    seed : int
        Random seed for reproducibility in NSGA-II optimization
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
        self._compartment_name_to_idx = {
            bin.id: idx
            for idx, bin in enumerate(simulation.model_definition.population.bins)
        }

    def select_ensemble(
        self,
        representatives: list[CalibrationEvaluation],
        nsga_population_size: int,
        nsga_generations: int,
        confidence_level: float,
        pareto_preference: float,
        ensemble_size_mode: str,
        ensemble_size: int | None,
        ensemble_size_min: int | None,
        ensemble_size_max: int | None,
        ci_margin_factor: float,
        ci_sample_sizes: list[int],
        nsga_crossover_probability: float,
    ) -> "EnsembleSelectionResultProtocol":
        """Run NSGA-II ensemble selection.

        Parameters
        ----------
        representatives : list[CalibrationEvaluation]
            Candidate parameter sets for ensemble selection
        nsga_population_size : int
            NSGA-II population size
        nsga_generations : int
            Number of NSGA-II generations
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
        nsga_crossover_probability : float
            Crossover probability for NSGA-II (0.0-1.0)

        Returns
        -------
        object
            Rust EnsembleSelectionResult object
        """

        logger.info(
            f"Running NSGA-II ensemble selection on {len(representatives)} candidates"
        )

        # Generate predictions for each representative in parallel
        representatives_with_predictions = self._generate_predictions(representatives)

        # Convert to Rust CalibrationEvaluation objects
        candidates = [
            commol_rs.calibration.CalibrationEvaluation(
                parameters=rep.parameters,
                loss=rep.loss,
                predictions=rep.predictions or [],
            )
            for rep in representatives_with_predictions
        ]

        # Prepare observed data tuples
        observed_data_tuples = [
            (
                obs.step,
                self._compartment_name_to_idx[obs.compartment],
                obs.value,
            )
            for obs in self.problem.observed_data
        ]

        # Run NSGA-II ensemble selection
        logger.info("Running NSGA-II ensemble selection...")
        ensemble_result = commol_rs.calibration.select_optimal_ensemble(
            candidates=candidates,
            observed_data_tuples=observed_data_tuples,
            population_size=nsga_population_size,
            generations=nsga_generations,
            confidence_level=confidence_level,
            seed=self.seed,
            pareto_preference=pareto_preference,
            ensemble_size_mode=ensemble_size_mode,
            ensemble_size=ensemble_size,
            ensemble_size_min=ensemble_size_min,
            ensemble_size_max=ensemble_size_max,
            ci_margin_factor=ci_margin_factor,
            ci_sample_sizes=ci_sample_sizes,
            nsga_crossover_probability=nsga_crossover_probability,
        )

        logger.info(
            f"Selected ensemble of {len(ensemble_result.selected_ensemble)} parameter "
            "sets using NSGA-II"
        )
        logger.info(
            f"Pareto front contains {len(ensemble_result.pareto_front)} solutions"
        )

        return ensemble_result

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

        # Extract parameter sets and names
        parameter_sets = [rep.parameters for rep in representatives]
        parameter_names = representatives[0].parameter_names

        # Call Rust function to generate all predictions in parallel
        all_predictions = commol_rs.calibration.generate_predictions_parallel(
            self.simulation.engine,
            parameter_sets,
            parameter_names,
            time_steps,
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
