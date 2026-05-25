from types import SimpleNamespace

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Model,
    ModelBuilder,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.api.probabilistic_calibrator import ProbabilisticCalibrator
from commol.api.probabilistic.ensemble_selector import EnsembleSelector
from commol.api.probabilistic.statistics_calculator import StatisticsCalculator
from commol.commol_rs import _commol_rs as commol_rs
from commol.constants import ModelTypes
from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    ProbClusteringConfig,
    ProbEnsembleConfig,
    ProbEvaluationFilterConfig,
    ProbabilisticCalibrationConfig,
)


def _generic_flow_model() -> Model:
    builder = (
        ModelBuilder(name="Generic Flow", version="1.0")
        .add_bin(id="waiting", name="Waiting")
        .add_bin(id="active", name="Active")
        .add_bin(id="done", name="Done")
        .add_parameter(id="start_rate", value=0.1)
        .add_parameter(id="finish_rate", value=0.2)
        .add_transition(
            id="start",
            source=["waiting"],
            target=["active"],
            rate="start_rate * waiting",
        )
        .add_transition(
            id="finish",
            source=["active"],
            target=["done"],
            rate="finish_rate * active",
        )
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "waiting", "fraction": 0.9},
                {"bin": "active", "fraction": 0.1},
                {"bin": "done", "fraction": 0.0},
            ],
        )
    )
    return builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)


def test_statistics_calculator_reuses_cached_predictions() -> None:
    model = _generic_flow_model()
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(step=1, compartment="active", value=18.0)],
        parameters=[
            CalibrationParameter(
                id="start_rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            )
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
    )
    calculator = StatisticsCalculator(Simulation(model), problem)
    ensemble_params = [
        CalibrationEvaluation(
            parameters=[0.1],
            loss=1.0,
            parameter_names=["start_rate"],
            predictions=[
                [900.0, 100.0, 0.0],
                [890.0, 90.0, 20.0],
            ],
        ),
        CalibrationEvaluation(
            parameters=[0.2],
            loss=2.0,
            parameter_names=["start_rate"],
            predictions=[
                [880.0, 120.0, 0.0],
                [860.0, 110.0, 30.0],
            ],
        ),
    ]

    predictions = calculator.generate_ensemble_predictions(
        ensemble_params,
        ["waiting", "active", "done"],
        time_steps=2,
    )

    assert predictions["waiting"] == [[900.0, 890.0], [880.0, 860.0]]
    assert predictions["active"] == [[100.0, 90.0], [120.0, 110.0]]
    assert predictions["done"] == [[0.0, 20.0], [0.0, 30.0]]


def test_statistics_calculator_rejects_compact_cached_predictions(monkeypatch) -> None:
    model = _generic_flow_model()
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(step=1, compartment="active", value=18.0)],
        parameters=[
            CalibrationParameter(
                id="start_rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            )
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
    )
    calculator = StatisticsCalculator(Simulation(model), problem)
    ensemble_params = [
        CalibrationEvaluation(
            parameters=[0.1],
            loss=1.0,
            parameter_names=["start_rate"],
            predictions=[[100.0], [90.0]],
        )
    ]

    def fake_generate_calibrated_predictions_parallel(*_args):
        return [[[900.0, 100.0, 0.0], [890.0, 90.0, 20.0]]]

    monkeypatch.setattr(
        commol_rs.calibration,
        "generate_calibrated_predictions_parallel",
        fake_generate_calibrated_predictions_parallel,
    )

    predictions = calculator.generate_ensemble_predictions(
        ensemble_params,
        ["waiting", "active", "done"],
        time_steps=2,
    )

    assert predictions["done"] == [[0.0, 20.0]]


def test_result_detail_pareto_summary_skips_heavy_pareto_payloads() -> None:
    model = _generic_flow_model()
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(step=1, compartment="active", value=18.0)],
        parameters=[
            CalibrationParameter(
                id="start_rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            )
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
    )
    problem.probabilistic_config = ProbabilisticCalibrationConfig(
        result_detail="pareto_summary"
    )
    calibrator = ProbabilisticCalibrator(Simulation(model), problem)
    representatives = [
        CalibrationEvaluation(
            parameters=[0.1],
            loss=1.0,
            parameter_names=["start_rate"],
            predictions=[
                [900.0, 100.0, 0.0],
                [890.0, 90.0, 20.0],
            ],
        ),
        CalibrationEvaluation(
            parameters=[0.2],
            loss=2.0,
            parameter_names=["start_rate"],
            predictions=[
                [880.0, 120.0, 0.0],
                [860.0, 110.0, 30.0],
            ],
        ),
        CalibrationEvaluation(
            parameters=[0.3],
            loss=3.0,
            parameter_names=["start_rate"],
            predictions=[
                [870.0, 130.0, 0.0],
                [840.0, 140.0, 20.0],
            ],
        ),
    ]
    rust_result = SimpleNamespace(
        pareto_front=[
            SimpleNamespace(
                ensemble_size=2,
                selected_indices=[0, 1],
                ci_width=0.1,
                coverage=1.0,
                size_penalty=0.0,
            ),
            SimpleNamespace(
                ensemble_size=2,
                selected_indices=[1, 2],
                ci_width=0.2,
                coverage=0.5,
                size_penalty=0.0,
            ),
        ],
        selected_pareto_index=0,
        selected_ensemble=[0, 1],
    )

    result = calibrator._build_result(
        representatives=representatives,
        rust_ensemble_result=rust_result,
        n_runs=2,
        n_unique=3,
        n_clusters=1,
        stage_timings={"calibration_runs_seconds": 0.01},
        stage_counts={"n_representatives": 3},
    )

    assert result.selected_ensemble.parameter_statistics
    assert result.selected_ensemble.prediction_median
    assert result.pareto_front[0].parameter_statistics == {}
    assert result.pareto_front[0].prediction_median == {}
    assert result.pareto_front[1].ensemble_parameters == []
    assert result.selected_pareto_index == 0
    assert result.stage_timings["calibration_runs_seconds"] == 0.01
    assert result.stage_counts["n_representatives"] == 3


def test_result_detail_selected_only_keeps_selected_solution_only() -> None:
    config = ProbabilisticCalibrationConfig(result_detail="selected_only")
    assert config.result_detail == "selected_only"


def test_performance_config_modes_validate() -> None:
    eval_config = ProbEvaluationFilterConfig(
        evaluation_retention="top_k_per_run",
        top_k_per_run=5,
    )
    ensemble_config = ProbEnsembleConfig(ci_width_scope="observed_points")
    clustering_config = ProbClusteringConfig(
        max_k=4,
        silhouette_sample_size=20,
        minibatch_kmeans_threshold=100,
    )

    assert eval_config.top_k_per_run == 5
    assert ensemble_config.ci_width_scope == "observed_points"
    assert clustering_config.max_k == 4


def test_compact_prediction_generation_returns_metric_points() -> None:
    model = _generic_flow_model()
    simulation = Simulation(model)

    predictions = commol_rs.calibration.generate_predictions_at_points_parallel(
        simulation.engine,
        parameter_sets=[[0.1, 0.2], [0.2, 0.2]],
        parameter_names=["start_rate", "finish_rate"],
        metric_points=[(0, 1), (1, 1)],
    )

    assert len(predictions) == 2
    assert len(predictions[0]) == 2


def test_compact_observed_point_predictions_match_full_predictions() -> None:
    model = _generic_flow_model()
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(step=0, compartment="waiting", value=900.0),
            ObservedDataPoint(step=1, compartment="active", value=110.0),
            ObservedDataPoint(step=2, compartment="done", value=40.0),
        ],
        parameters=[
            CalibrationParameter(
                id="start_rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="finish_rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
    )
    simulation = Simulation(model)
    selector = EnsembleSelector(simulation, problem, seed=42)
    representatives = [
        CalibrationEvaluation(
            parameters=[0.1, 0.2],
            loss=1.0,
            parameter_names=["start_rate", "finish_rate"],
        ),
        CalibrationEvaluation(
            parameters=[0.2, 0.2],
            loss=2.0,
            parameter_names=["start_rate", "finish_rate"],
        ),
    ]

    metric_points, _ = selector._metric_points_for_scope("observed_points")
    compact_representatives = selector._generate_compact_predictions(
        representatives,
        metric_points,
    )
    full_predictions = commol_rs.calibration.generate_predictions_parallel(
        simulation.engine,
        [rep.parameters for rep in representatives],
        representatives[0].parameter_names,
        3,
    )

    for rep_idx, compact_rep in enumerate(compact_representatives):
        assert compact_rep.predictions is not None
        for point_idx, (step, compartment_idx) in enumerate(metric_points):
            assert (
                compact_rep.predictions[point_idx][0]
                == full_predictions[rep_idx][step][compartment_idx]
            )


def test_selector_coverage_matches_regenerated_full_result_coverage() -> None:
    model = _generic_flow_model()
    simulation = Simulation(model)
    parameter_sets = [[0.1, 0.2], [0.2, 0.2], [0.3, 0.2]]
    parameter_names = ["start_rate", "finish_rate"]
    full_predictions = commol_rs.calibration.generate_predictions_parallel(
        simulation.engine,
        parameter_sets,
        parameter_names,
        2,
    )
    active_idx = simulation.engine.compartments.index("active")
    observed_value = full_predictions[1][1][active_idx]
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(step=1, compartment="active", value=observed_value),
        ],
        parameters=[
            CalibrationParameter(
                id="start_rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
            CalibrationParameter(
                id="finish_rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            ),
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
        probabilistic_config=ProbabilisticCalibrationConfig(
            result_detail="selected_only",
            ensemble_selection=ProbEnsembleConfig(
                ensemble_algorithm="greedy_local_search",
                ensemble_size_mode="fixed",
                ensemble_size=3,
                ci_width_scope="observed_points",
            ),
        ),
    )
    representatives = [
        CalibrationEvaluation(
            parameters=params,
            loss=float(idx),
            parameter_names=parameter_names,
        )
        for idx, params in enumerate(parameter_sets)
    ]
    selector = EnsembleSelector(simulation, problem, seed=42)
    rust_result, representatives_for_result = selector.select_ensemble_with_predictions(
        representatives=representatives,
        population_size=4,
        generations=4,
        confidence_level=0.95,
        pareto_preference=0.5,
        ensemble_size_mode="fixed",
        ensemble_size=3,
        ensemble_size_min=None,
        ensemble_size_max=None,
        ci_margin_factor=0.1,
        ci_sample_sizes=[3],
        crossover_probability=0.9,
        ci_width_scope="observed_points",
        ensemble_algorithm="greedy_local_search",
    )
    calibrator = ProbabilisticCalibrator(simulation, problem)
    result = calibrator._build_result(
        representatives=representatives_for_result,
        rust_ensemble_result=rust_result,
        n_runs=1,
        n_unique=len(representatives),
        n_clusters=1,
    )
    selected_rust_solution = rust_result.pareto_front[rust_result.selected_pareto_index]

    assert selected_rust_solution.coverage == 1.0
    assert result.selected_ensemble.coverage_percentage == 100.0
