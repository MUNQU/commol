"""Integration tests for probabilistic calibration selection backends."""

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    ModelBuilder,
    ObservedDataPoint,
    ParticleSwarmConfig,
)
from commol.api.calibrator import Calibrator
from commol.api.simulation import Simulation
from commol.constants import ModelTypes
from commol.context.probabilistic_calibration import (
    ProbClusteringConfig,
    ProbEvaluationFilterConfig,
    ProbGreedyLocalSearchConfig,
    ProbNsga2Config,
    ProbabilisticCalibrationConfig,
    ProbRepresentativeConfig,
)

SEED = 42


@pytest.fixture(scope="module")
def model():
    return (
        ModelBuilder(name="Test SIR", version="1.0")
        .add_bin(id="S", name="Susceptible")
        .add_bin(id="I", name="Infected")
        .add_bin(id="R", name="Recovered")
        .add_parameter(id="beta", value=0.3)
        .add_parameter(id="gamma", value=0.1)
        .add_transition(
            id="infection",
            source=["S"],
            target=["I"],
            rate="beta * S * I / N",
        )
        .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "S", "fraction": 0.99},
                {"bin": "I", "fraction": 0.01},
                {"bin": "R", "fraction": 0.0},
            ],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


def _problem(
    model,
    ensemble_selection: ProbNsga2Config | ProbGreedyLocalSearchConfig | None = None,
    *,
    loss_function: str = "sse",
    feature_space: str = "observed_predictions",
) -> CalibrationProblem:
    simulation = Simulation(model)
    true_results = simulation.run(30, output_format="dict_of_lists")
    noise = np.random.default_rng(SEED).normal(0.0, 3.0, size=30)
    observations = [
        ObservedDataPoint(
            step=step,
            compartment="I",
            value=max(0.0, true_results["I"][step] + noise[step]),
        )
        for step in range(30)
    ]
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="gamma",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=0.5,
            ),
        ],
        loss_function=loss_function,
        optimization_config=ParticleSwarmConfig(
            num_particles=30,
            max_iterations=200,
            verbose=False,
        ),
    )
    if ensemble_selection is None:
        ensemble_selection = ProbNsga2Config(
            ensemble_size_mode="bounded",
            ensemble_size_min=4,
            ensemble_size_max=10,
        )
    problem.probabilistic_config = ProbabilisticCalibrationConfig(
        n_runs=6,
        evaluation_processing=ProbEvaluationFilterConfig(
            loss_percentile_filter=0.1,
            max_loss_ratio=1.5,
        ),
        clustering=ProbClusteringConfig(
            n_clusters=4,
            feature_space=feature_space,
        ),
        representative_selection=ProbRepresentativeConfig(
            max_representatives=80,
            cluster_selection_method="maximin_distance",
        ),
        ensemble_selection=ensemble_selection,
        include_ensemble_candidates=True,
    )
    problem.seed = SEED
    return problem


def test_fit_gated_calibration_preserves_central_fit(model) -> None:
    problem = _problem(
        model,
        ProbGreedyLocalSearchConfig(
            ensemble_size_mode="bounded",
            ensemble_size_min=4,
            ensemble_size_max=10,
            central_fit_max_loss_ratio=1.5,
        ),
    )
    result = Calibrator(Simulation(model), problem).run_probabilistic()
    ensemble = result.selected_ensemble

    assert 4 <= ensemble.ensemble_size <= 10
    assert ensemble.point_loss > 0.0
    assert ensemble.central_loss <= ensemble.point_loss * 1.5 + 1e-9
    assert ensemble.point_parameters in ensemble.ensemble_parameters
    assert ensemble.observation_diagnostics["I"]["n_points"] == 30.0
    assert result.ensemble_candidates is not None
    assert len(result.ensemble_candidates) == result.stage_counts["n_representatives"]
    assert all(candidate.parameters for candidate in result.ensemble_candidates)
    assert (
        ensemble.selection_diagnostics["max_feasible_ensemble_size"]
        >= ensemble.ensemble_size
    )
    assert len(ensemble.prediction_median["I"]) == 31


def test_nsga2_calibration_is_the_default_selection_backend(model) -> None:
    result = Calibrator(Simulation(model), _problem(model)).run_probabilistic()

    assert result.selection_algorithm == "nsga2"
    assert result.selected_pareto_index is not None
    assert result.pareto_front
    assert result.stage_counts["pareto_front_size"] == len(result.pareto_front)


def test_ensemble_configuration_defaults_to_nsga2() -> None:
    config = ProbNsga2Config()

    assert config.population_size == 100
    assert config.generations == 100
    assert config.pareto_preference == 0.5
    assert isinstance(
        ProbabilisticCalibrationConfig().ensemble_selection,
        ProbNsga2Config,
    )

    with pytest.raises(ValidationError, match="ensemble_algorithm"):
        ProbNsga2Config(ensemble_algorithm="unsupported")

    with pytest.raises(ValidationError, match="central_fit_max_loss_ratio"):
        ProbNsga2Config(central_fit_max_loss_ratio=1.5)

    with pytest.raises(ValidationError, match="population_size"):
        ProbGreedyLocalSearchConfig(population_size=100)

    with pytest.raises(ValidationError, match="result_detail"):
        ProbabilisticCalibrationConfig(result_detail="full")


def test_invalid_observed_compartment_is_rejected(model) -> None:
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(
                step=0,
                compartment="not-a-model-output",
                value=1.0,
            )
        ],
        parameters=[
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            )
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
    )

    with pytest.raises(ValueError, match="not found in model"):
        Calibrator(Simulation(model), problem)


def _single_parameter_problem(loss_function: str) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(step=0, compartment="I", value=1.0)],
        parameters=[
            CalibrationParameter(
                id="beta",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            )
        ],
        loss_function=loss_function,
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
    )


def test_probabilistic_config_accepts_any_loss_family() -> None:
    """Every loss is allowed; the central-fit gate adapts to the chosen loss."""
    for loss in ("sse", "weighted_sse", "rmse", "mae"):
        problem = _single_parameter_problem(loss)
        problem.probabilistic_config = ProbabilisticCalibrationConfig()
        assert problem.probabilistic_config is not None


def test_rmse_pipeline_gate_is_loss_coherent(model) -> None:
    """The greedy central-fit gate scores the median with the members' RMSE loss."""
    problem = _problem(
        model,
        ProbGreedyLocalSearchConfig(
            ensemble_size_mode="automatic",
            central_fit_max_loss_ratio=1.5,
        ),
        loss_function="rmse",
    )
    result = Calibrator(Simulation(model), problem).run_probabilistic()
    ensemble = result.selected_ensemble

    assert ensemble.point_loss > 0.0
    # central_loss is an RMSE (root-mean), directly comparable to the member
    # RMSE losses the gate constrains it against.
    assert ensemble.central_loss <= ensemble.point_loss * 1.5 + 1e-9


def test_mae_pipeline_runs_end_to_end(model) -> None:
    """A mean-absolute-error loss selects a coherent ensemble end-to-end."""
    problem = _problem(model, loss_function="mae")
    result = Calibrator(Simulation(model), problem).run_probabilistic()

    assert result.selected_ensemble.ensemble_size >= 4


def test_parameter_feature_space_pipeline(model) -> None:
    """Parameter-space clustering runs end-to-end with the NSGA-II backend."""
    problem = _problem(model, feature_space="parameters")
    result = Calibrator(Simulation(model), problem).run_probabilistic()

    assert result.selection_algorithm == "nsga2"
    assert 4 <= result.selected_ensemble.ensemble_size <= 10
    assert len(result.selected_ensemble.prediction_median["I"]) == 31


def test_greedy_automatic_size_mode_pipeline(model) -> None:
    """The greedy backend selects a fit-gated ensemble under automatic sizing."""
    problem = _problem(
        model,
        ProbGreedyLocalSearchConfig(
            ensemble_size_mode="automatic",
            central_fit_max_loss_ratio=1.5,
        ),
    )
    result = Calibrator(Simulation(model), problem).run_probabilistic()
    ensemble = result.selected_ensemble

    assert result.selection_algorithm == "greedy_local_search"
    assert result.pareto_front is None
    assert ensemble.ensemble_size >= 2
    assert ensemble.central_loss <= ensemble.point_loss * 1.5 + 1e-9


def test_weighted_sse_loss_pipeline(model) -> None:
    """A weighted-SSE loss is an accepted SSE-family loss for the pipeline."""
    problem = _problem(model, loss_function="weighted_sse")
    result = Calibrator(Simulation(model), problem).run_probabilistic()

    assert result.selected_ensemble.ensemble_size >= 4


def _windowed_problem(model, steps: list[int], window: int) -> CalibrationProblem:
    """Problem whose observations are windowed at the given steps."""
    simulation = Simulation(model)
    truth = simulation.run(max(steps))
    observations = [
        ObservedDataPoint(
            step=step,
            compartment="I",
            value=truth["I"][step] - truth["I"][step - window],
            window_steps=window,
        )
        for step in steps
    ]
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[
            CalibrationParameter(
                id="beta", parameter_type="parameter", min_bound=0.0, max_bound=1.0
            ),
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=10, max_iterations=20),
    )
    problem.probabilistic_config = ProbabilisticCalibrationConfig(
        n_runs=3,
        evaluation_processing=ProbEvaluationFilterConfig(min_evaluations_required=2),
        clustering=ProbClusteringConfig(n_clusters=2),
        representative_selection=ProbRepresentativeConfig(max_representatives=10),
        ensemble_selection=ProbNsga2Config(
            ensemble_size_mode="bounded", ensemble_size_min=2, ensemble_size_max=5
        ),
    )
    problem.seed = SEED
    return problem


def test_windowed_bands_are_taken_at_the_observation_steps(model) -> None:
    """
    Windowed values are anchored where the observations are.

    These steps are deliberately not multiples of the window, so a self-made
    grid of multiples would produce a different number of values, at different
    steps, from the ones the loss compared against.
    """
    steps = [9, 16, 23, 27]
    window = 7
    result = Calibrator(
        Simulation(model), _windowed_problem(model, steps, window)
    ).run_probabilistic()
    ensemble = result.selected_ensemble

    assert ensemble.windowed_prediction_steps["I"] == steps
    assert len(ensemble.windowed_prediction_median["I"]) == len(steps)
    assert len(ensemble.windowed_prediction_ci_lower["I"]) == len(steps)
    assert len(ensemble.windowed_prediction_ci_upper["I"]) == len(steps)


def test_windowed_median_matches_a_member_wise_reduction(model) -> None:
    steps = [9, 16, 23, 27]
    window = 7
    problem = _windowed_problem(model, steps, window)
    result = Calibrator(Simulation(model), problem).run_probabilistic()
    ensemble = result.selected_ensemble

    member_values: list[list[float]] = []
    for parameters in ensemble.ensemble_parameters:
        member = Simulation(model)
        member.model_definition.apply_calibration_parameters(parameters, problem)
        run = Simulation(member.model_definition).run(max(steps))
        member_values.append([run["I"][s] - run["I"][s - window] for s in steps])

    expected = [
        float(np.median([values[i] for values in member_values]))
        for i in range(len(steps))
    ]
    assert ensemble.windowed_prediction_median["I"] == pytest.approx(expected, rel=1e-9)
