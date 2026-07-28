import numpy as np

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    ObservedDataPoint,
    ParticleSwarmConfig,
)
from commol.api.probabilistic.ensemble_selector import EnsembleSelector
from commol.api.probabilistic.evaluation_processor import EvaluationProcessor
from commol.api.probabilistic.normalization import central_fit_loss
from commol.api.probabilistic.statistics_calculator import StatisticsCalculator
from commol.api.probabilistic_calibrator import ProbabilisticCalibrator
from commol.context.probabilistic_calibration import (
    CalibrationEvaluation,
    ProbClusteringConfig,
    ProbEvaluationFilterConfig,
    ProbGreedyLocalSearchConfig,
    ProbNsga2Config,
    ProbabilisticCalibrationConfig,
)


def _problem() -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[
            ObservedDataPoint(step=0, compartment="reported", value=10.5),
            ObservedDataPoint(step=1, compartment="reported", value=19.5),
            ObservedDataPoint(step=0, compartment="admissions", value=5.0),
        ],
        parameters=[
            CalibrationParameter(
                id="rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            )
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
    )


def _evaluation(loss: float, values: list[float]) -> CalibrationEvaluation:
    return CalibrationEvaluation(
        parameters=[loss],
        loss=loss,
        parameter_names=["rate"],
        predictions=[[value] for value in values],
    )


def _normalized_weight_problem(case_weight: float) -> CalibrationProblem:
    """Two equally scaled series with an explicit relative case weight."""
    return CalibrationProblem(
        observed_data=[
            ObservedDataPoint(
                step=0,
                compartment="reported",
                value=100.0,
                weight=case_weight,
            ),
            ObservedDataPoint(
                step=0,
                compartment="admissions",
                value=10.0,
                weight=1.0,
            ),
        ],
        parameters=[
            CalibrationParameter(
                id="rate",
                parameter_type="parameter",
                min_bound=0.0,
                max_bound=1.0,
            )
        ],
        loss_function="sse",
        normalize_observations=True,
        optimization_config=ParticleSwarmConfig(num_particles=4, max_iterations=1),
    )


def test_relative_loss_gate_keeps_distinct_near_optimal_candidates() -> None:
    evaluations = [
        _evaluation(10.0, [10.0, 20.0, 5.0]),
        _evaluation(12.5, [9.0, 19.0, 5.0]),
        _evaluation(13.0, [8.0, 18.0, 4.0]),
    ]

    retained = EvaluationProcessor.filter_by_relative_loss(evaluations, 1.25)

    assert [evaluation.loss for evaluation in retained] == [10.0, 12.5]


def test_prediction_feature_space_is_validated_and_standardized() -> None:
    evaluations = [
        _evaluation(1.0, [10.0, 20.0, 5.0]),
        _evaluation(2.0, [11.0, 22.0, 5.0]),
    ]
    processor = EvaluationProcessor(seed=42)
    vectors = np.array([[0.0, 10.0], [2.0, 14.0]])

    assert processor.find_optimal_k(evaluations, vectors) == 1
    labels = processor.cluster_evaluations(evaluations, 1, vectors)
    assert labels == [0, 0]


def test_representative_selection_accepts_prediction_space_features() -> None:
    """Representative diversity can be computed in observed-prediction space."""
    evaluations = [
        _evaluation(1.0, [10.0, 20.0, 5.0]),
        _evaluation(1.1, [11.0, 21.0, 5.0]),
        _evaluation(1.2, [12.0, 22.0, 5.0]),
    ]
    processor = EvaluationProcessor(seed=42)

    selected = processor.select_representatives(
        evaluations=evaluations,
        cluster_labels=[0, 0, 0],
        max_representatives=2,
        elite_fraction=0.0,
        strategy="equal",
        selection_method="maximin_distance",
        quality_temperature=1.0,
        k_neighbors_min=1,
        k_neighbors_max=2,
        sparsity_weight=2.0,
        stratum_fit_weight=10.0,
        feature_vectors=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 10.0]]),
    )

    assert len(selected) == 2
    assert set(selected).issubset({0, 1, 2})


def test_prediction_novel_tail_selection_prefers_new_observed_shapes() -> None:
    """Tail candidates are admitted for prediction novelty, not just count."""
    selected = EvaluationProcessor.select_prediction_novel_candidates(
        selected_indices=[0],
        candidate_indices=[1, 2, 3],
        feature_vectors=np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [3.0, 0.0],
                [0.0, 4.0],
            ]
        ),
        max_candidates=2,
    )

    assert selected == [3, 2]


def test_prediction_novel_tail_selection_can_start_without_core() -> None:
    """If no core representative exists, start from the prediction-space edge."""
    selected = EvaluationProcessor.select_prediction_novel_candidates(
        selected_indices=[],
        candidate_indices=[0, 1, 2],
        feature_vectors=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [10.0, 0.0],
            ]
        ),
        max_candidates=2,
    )

    assert selected == [2, 0]


def test_tail_mode_keeps_core_selection_inside_core_loss_gate() -> None:
    """The wider tail band must not bypass the core representative pool cap."""
    calibrator = ProbabilisticCalibrator.__new__(ProbabilisticCalibrator)
    calibrator.config = ProbabilisticCalibrationConfig(
        evaluation_processing=ProbEvaluationFilterConfig(
            max_loss_ratio=1.25,
            tail_max_loss_ratio=2.0,
            tail_max_representatives=2,
        )
    )
    evaluations = [
        _evaluation(10.0, [10.0, 20.0, 5.0]),
        _evaluation(12.5, [11.0, 20.0, 5.0]),
        _evaluation(15.0, [30.0, 20.0, 5.0]),
        _evaluation(20.0, [-10.0, 20.0, 5.0]),
    ]

    assert calibrator._core_evaluation_indices(evaluations) == [0, 1]


def _greedy_selector() -> EnsembleSelector:
    selector = EnsembleSelector.__new__(EnsembleSelector)
    object.__setattr__(selector, "problem", _problem())
    object.__setattr__(selector, "seed", 42)
    return selector


def test_greedy_selection_rejects_wide_bad_trajectory() -> None:
    """The fit gate keeps a wide, poorly-fitting member out of the ensemble."""
    selector = _greedy_selector()
    representatives = [
        _evaluation(1.0, [10.0, 20.0, 5.0]),
        _evaluation(1.1, [11.0, 19.0, 5.0]),
        _evaluation(100.0, [100.0, 100.0, 100.0]),
    ]

    result = selector._select_compact_ensemble(
        representatives,
        confidence_level=0.95,
        selection_config=ProbGreedyLocalSearchConfig(
            ensemble_size_mode="fixed",
            ensemble_size=2,
            central_fit_max_loss_ratio=1.5,
            search_beam_width=8,
        ),
    )

    assert result.selected_indices == [0, 1]
    assert result.ensemble_size == 2
    assert result.pareto_front is None
    assert result.diagnostics["n_single_additions_rejected_by_central_fit"] == 1.0


def test_greedy_selection_can_cross_an_infeasible_single_member_bridge() -> None:
    """Opposite CI tails may be jointly feasible although neither is alone."""
    selector = _greedy_selector()
    representatives = [
        _evaluation(1.0, [10.0, 20.0, 5.0]),
        _evaluation(1.1, [30.0, 20.0, 5.0]),
        _evaluation(1.1, [-10.0, 20.0, 5.0]),
    ]

    result = selector._select_compact_ensemble(
        representatives,
        confidence_level=0.95,
        selection_config=ProbGreedyLocalSearchConfig(
            ensemble_size_mode="fixed",
            ensemble_size=3,
            central_fit_max_loss_ratio=1.25,
            search_beam_width=8,
        ),
    )

    assert result.selected_indices == [0, 1, 2]
    assert result.ensemble_size == 3
    assert result.diagnostics["n_single_additions_rejected_by_central_fit"] == 2.0


def test_central_fit_loss_matches_configured_metric() -> None:
    """The central loss reproduces each optimizer loss formula (gate coherence)."""
    residuals = [3.0, 3.0]
    weights = [1.0, 1.0]
    normalization = [1.0, 1.0]

    assert np.isclose(central_fit_loss(residuals, weights, normalization, "sse"), 18.0)
    assert np.isclose(
        central_fit_loss(residuals, weights, normalization, "weighted_sse"), 18.0
    )
    assert np.isclose(central_fit_loss(residuals, weights, normalization, "rmse"), 3.0)
    assert np.isclose(central_fit_loss(residuals, weights, normalization, "mae"), 3.0)


def test_normalized_central_loss_respects_explicit_relative_series_weights() -> None:
    """After RMS normalization, weights control relative series importance."""
    predictions = {"reported": [[110.0]], "admissions": [[11.0]]}
    ensemble = [CalibrationEvaluation([0.5], 1.0, ["rate"])]

    calculator = StatisticsCalculator.__new__(StatisticsCalculator)
    calculator.confidence_level = 0.95

    calculator.problem = _normalized_weight_problem(1.0)
    equal_weight_loss = calculator.calculate_central_loss(predictions, ensemble)

    calculator.problem = _normalized_weight_problem(0.1)
    low_case_weight_loss = calculator.calculate_central_loss(predictions, ensemble)

    assert np.isclose(equal_weight_loss, 0.02)
    assert np.isclose(low_case_weight_loss, 0.0101)


def test_ensemble_selection_configs_expose_both_algorithms() -> None:
    config = ProbGreedyLocalSearchConfig(
        ensemble_size_mode="bounded",
        ensemble_size_min=2,
        ensemble_size_max=4,
        central_fit_max_loss_ratio=1.5,
        search_beam_width=12,
    )
    clustering = ProbClusteringConfig(feature_space="observed_predictions")
    filtering = ProbEvaluationFilterConfig(
        max_loss_ratio=1.25,
        tail_max_loss_ratio=2.0,
        tail_max_representatives=25,
    )
    result_config = ProbabilisticCalibrationConfig(include_ensemble_candidates=True)

    assert config.central_fit_max_loss_ratio == 1.5
    assert config.search_beam_width == 12
    assert isinstance(config, ProbGreedyLocalSearchConfig)
    assert isinstance(
        ProbabilisticCalibrationConfig().ensemble_selection,
        ProbNsga2Config,
    )
    assert clustering.feature_space == "observed_predictions"
    assert filtering.max_loss_ratio == 1.25
    assert filtering.tail_max_loss_ratio == 2.0
    assert filtering.tail_max_representatives == 25
    assert result_config.include_ensemble_candidates is True


def test_observation_diagnostics_reports_each_series() -> None:
    calculator = StatisticsCalculator.__new__(StatisticsCalculator)
    calculator.problem = _problem()
    calculator.confidence_level = 0.95
    ensemble = [
        CalibrationEvaluation([1.0], 1.0, ["rate"]),
        CalibrationEvaluation([1.0], 1.1, ["rate"]),
    ]
    predictions = {
        "reported": [[10.0, 20.0], [11.0, 19.0]],
        "admissions": [[5.0], [5.0]],
    }

    diagnostics = calculator.calculate_observation_diagnostics(predictions, ensemble)

    assert diagnostics["reported"]["n_points"] == 2.0
    assert diagnostics["reported"]["coverage_percentage"] == 100.0
    assert diagnostics["admissions"]["coverage_percentage"] == 100.0
