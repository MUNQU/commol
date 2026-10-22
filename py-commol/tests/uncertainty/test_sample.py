"""The parameter sample: its invariants and its weighted statistics."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from commol import ParameterSample, SampleMethod, weighted_quantiles
from commol import CalibrationProblem, NelderMeadConfig, ObservedDataPoint
from tests.support import abc_builder, bounded, build

METHOD = SampleMethod(name="test", settings={"draws": 3})
VALUES = [[1.0, 2.0], [2.0, 1.0], [4.0, 0.0]]


def _sample(log_weights: list[float], values: list[list[float]] = VALUES):
    return ParameterSample(
        names=["k1", "k2"],
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        values=values,
        log_weights=log_weights,
        method=METHOD,
    )


class TestInvariants:
    def test_the_log_weights_are_normalized(self) -> None:
        sample = _sample([5.0, 5.0 + math.log(2), 5.0 + math.log(3)])

        assert sample.weights == pytest.approx([1 / 6, 2 / 6, 3 / 6])
        assert sum(math.exp(w) for w in sample.log_weights) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("values", "log_weights", "message"),
        [
            ([[1.0, 11.0]], [0.0], "outside its bounds"),
            ([[1.0, float("nan")]], [0.0], "outside its bounds"),
            ([[1.0, 2.0]], [float("nan")], "finite or"),
            ([[1.0, 2.0]], [float("-inf")], "finite log weight"),
            ([[1.0]], [0.0], "1 values for 2 names"),
            ([[1.0, 2.0]], [0.0, 0.0], "one log weight per draw"),
        ],
    )
    def test_a_broken_invariant_is_rejected(
        self, values: list[list[float]], log_weights: list[float], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            _sample(log_weights, values)

    def test_a_name_appears_once(self) -> None:
        with pytest.raises(ValidationError, match="twice"):
            ParameterSample.equally_weighted(
                ["k1", "k1"], [(0, 1), (0, 1)], [[0.5, 0.5]], METHOD
            )

    def test_a_sample_is_read_from_a_mapping(self) -> None:
        with pytest.raises(ValidationError, match="a valid dictionary"):
            ParameterSample.model_validate([1.0, 2.0])

    def test_a_sample_equals_only_a_sample(self) -> None:
        sample = _sample([0.0, 0.0, 1.0])

        assert sample != sample.values
        assert sample == _sample([0.0, 0.0, 1.0])

    def test_a_sample_saved_to_json_loads_back_equal(self) -> None:
        sample = _sample([0.0, float("-inf"), 1.0])

        loaded = ParameterSample.model_validate_json(sample.model_dump_json())

        assert loaded == sample
        assert loaded.weights == sample.weights


class TestStatistics:
    def test_equal_weights_give_the_unweighted_statistics(self) -> None:
        sample = _sample([0.0, 0.0, 0.0])
        array = np.asarray(VALUES)

        assert sample.mean() == pytest.approx(dict(zip(["k1", "k2"], array.mean(0))))
        np.testing.assert_allclose(sample.covariance(), np.cov(array.T))
        np.testing.assert_allclose(sample.correlation(), np.corrcoef(array.T))
        assert sample.effective_size == pytest.approx(3.0)

    def test_weighted_statistics_are_their_closed_forms(self) -> None:
        weights = [0.5, 0.25, 0.25]
        sample = _sample([math.log(w) for w in weights])

        # Mean of k1: 0.5·1 + 0.25·2 + 0.25·4 = 2.
        assert sample.mean()["k1"] == pytest.approx(2.0)
        # Σw(x − 2)² = 0.5 + 0 + 1 = 1.5, over 1 − Σw² = 0.625.
        assert sample.covariance()[0][0] == pytest.approx(1.5 / 0.625)
        # Midpoints 1/4, 5/8 and 7/8 of k1 = 1, 2, 4.
        assert sample.quantiles([0.5])["k1"] == pytest.approx([1 + 0.25 / 0.375])

    def test_the_effective_size_is_a_property(self) -> None:
        sample = _sample([math.log(w) for w in [0.5, 0.25, 0.25]])

        # Kish: 1 / (0.5² + 0.25² + 0.25²).
        assert sample.effective_size == pytest.approx(1 / 0.375)

    def test_the_interval_is_the_central_pair_of_quantiles(self) -> None:
        sample = _sample([0.0, 0.0, 0.0])

        interval = sample.interval(0.5)

        assert interval["k1"] == pytest.approx(
            tuple(sample.quantiles([0.25, 0.75])["k1"])
        )

    @pytest.mark.parametrize("level", [-0.1, 2.0, float("nan")])
    def test_a_quantile_level_is_between_zero_and_one(self, level: float) -> None:
        with pytest.raises(ValueError, match="A quantile level is between 0 and 1"):
            _sample([0.0, 0.0, 0.0]).quantiles([0.5, level])
        with pytest.raises(ValueError, match="A quantile level is between 0 and 1"):
            weighted_quantiles([1.0, 2.0], levels=[level])

    def test_a_summary_holds_the_extremes_and_the_median(self) -> None:
        summary = _sample([0.0, 0.0, 0.0]).summary(0.5)["k1"]

        assert (summary.min, summary.median, summary.max) == (1.0, 2.0, 4.0)

    def test_draws_are_mappings_in_the_order_of_the_names(self) -> None:
        assert list(_sample([0.0, 0.0, 0.0]).draws())[2] == {"k1": 4.0, "k2": 0.0}

    def test_resampling_follows_the_weights(self) -> None:
        sample = _sample([0.0, float("-inf"), math.log(3)])

        drawn = sample.resample(4000, seed=1)

        assert drawn.weights == pytest.approx([1 / 4000] * 4000)
        share = sum(row == [4.0, 0.0] for row in drawn.values) / 4000
        assert share == pytest.approx(0.75, abs=0.03)
        assert [1.0, 2.0] in drawn.values and [2.0, 1.0] not in drawn.values
        assert drawn.values == sample.resample(4000, seed=1).values
        assert drawn.method == SampleMethod(
            name="resample",
            settings={"of": METHOD.name, "count": 4000, "seed": 1},
        )


def _k1_problem() -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="B", value=1.0)],
        parameters=[bounded("k1")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )


class TestNoSingleParameterSet:
    def test_a_sample_cannot_be_applied_to_a_model(self) -> None:
        model = build(abc_builder())
        sample = ParameterSample.equally_weighted(
            ["k1"], [(0.0, 1.0)], [[0.6], [0.7]], METHOD
        )

        with pytest.raises(TypeError, match=r"sample\.draws\(\)"):
            model.apply_calibration_parameters(sample, _k1_problem())  # ty: ignore[invalid-argument-type]

    def test_a_draw_can_be_applied_to_a_model(self) -> None:
        model = build(abc_builder())
        sample = ParameterSample.equally_weighted(
            ["k1"], [(0.0, 1.0)], [[0.6], [0.7]], METHOD
        )

        model.apply_calibration_parameters(next(sample.draws()), _k1_problem())

        assert {p.id: p.value for p in model.parameters}["k1"] == 0.6
