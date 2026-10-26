"""Two results compare by the values of their fields, a NaN equal to a NaN,
and a frozen result has the hash of those values."""

import json
import math
import sys
from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from commol import Simulation
from commol._data.result_model import ResultModel
from commol._fit.result import CalibrationResult, StartSummary
from commol.conformal.results import (
    ConformalResult,
    PredictionIntervals,
    SeriesIntervals,
)
from commol.quantities import ValueAt
from commol.sensitivity import PRCC, SensitivityAnalysis
from commol.sensitivity.results import PRCCResult
from commol.uncertainty import ParameterSample, SampleMethod
from commol.uncertainty.sample import ParameterSummary
from tests.suite.test_methods import method_classes
from tests.support import abc_rates

# With these 10 draws k2 orders B at 5 alone, so the PRCC of k1 is NaN.
NAN_SEED = 2496558405389616264


def _nan_prcc(simulation: Simulation) -> PRCCResult:
    method = PRCC(draws=10, quantities=[ValueAt(series="B", at=5)])
    return SensitivityAnalysis(simulation, abc_rates(), seed=NAN_SEED).run(method)


def _conformal(lower: float, scores: dict[str, list[float]]) -> ConformalResult:
    series = {
        label: SeriesIntervals(
            positions=[1.0, 2.0],
            prediction=[float("nan"), 2.0],
            lower=[lower, float("-inf")],
            upper=[float("nan"), float("inf")],
        )
        for label in scores
    }
    return ConformalResult(
        intervals=PredictionIntervals(
            confidence_level=0.9, series=series, unbounded={"B": [2.0]}
        ),
        scores=scores,
        guarantee="marginal",
        failed=[],
        not_converged=0,
        block_length=None,
        block_length_estimates=None,
    )


def _summary(mean: float) -> ParameterSummary:
    return ParameterSummary(
        mean=mean, median=mean, std=float("nan"), lower=0.0, upper=1.0, min=0.0, max=1.0
    )


def _calibration(loss: float) -> CalibrationResult:
    start = StartSummary(start={"k1": 0.1}, end={"k1": 0.2}, loss=loss, converged=True)
    return CalibrationResult(
        best_parameters={"k1": 0.2},
        final_loss=loss,
        iterations=3,
        converged=True,
        termination_reason="converged",
        starts=[start],
        seed=1,
    )


def test_two_results_with_one_nan_are_equal(simulation: Simulation) -> None:
    first = _nan_prcc(simulation)
    again = _nan_prcc(simulation)

    assert math.isnan(first.coefficients[0][0])
    assert first.coefficients[0][1] == -1.0
    assert first.coefficients[0][0] is not again.coefficients[0][0]
    assert first == again


def test_a_frozen_result_has_the_hash_of_its_values(simulation: Simulation) -> None:
    first = _nan_prcc(simulation)
    again = _nan_prcc(simulation)

    assert hash(first) == hash(again)
    assert len({first, again}) == 1


def test_nested_results_compare_by_value_in_any_key_order() -> None:
    first = _conformal(0.5, {"A": [float("nan")], "B": [1.0]})
    reordered = _conformal(0.5, {"B": [1.0], "A": [float("nan")]})
    other = _conformal(0.6, {"A": [float("nan")], "B": [1.0]})

    assert first == reordered
    assert hash(first) == hash(reordered)
    assert first != other


def test_zero_and_negative_zero_are_equal() -> None:
    assert _summary(0.0) == _summary(-0.0)
    assert hash(_summary(0.0)) == hash(_summary(-0.0))
    assert _summary(0.0) != _summary(1.0)


def test_a_sample_compares_its_fields_and_not_its_rust_sample() -> None:
    def sample(values: list[list[float]]) -> ParameterSample:
        return ParameterSample.equally_weighted(
            ["k1"],
            [(0.0, 1.0)],
            values,
            SampleMethod(name="test", settings={"seed": 1, "levels": [0.5]}),
        )

    first = sample([[0.2], [0.4]])
    again = sample([[0.2], [0.4]])

    assert first._rust is not again._rust
    assert first == again
    assert hash(first) == hash(again)
    assert first != sample([[0.2], [0.5]])


def test_a_result_that_can_change_compares_by_value_and_has_no_hash() -> None:
    first = _calibration(float("nan"))

    assert first == _calibration(float("nan"))
    assert first != _calibration(1.0)
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)


def test_results_of_other_types_are_not_equal() -> None:
    summary = _summary(0.0)

    assert summary != _calibration(0.0)
    assert summary != {"mean": 0.0}


def _result_type(method_type: type) -> type:
    """The result type a method declares, `ResultT` of its kind."""
    (result,) = next(
        base.__pydantic_generic_metadata__["args"]
        for base in method_type.__mro__
        if issubclass(base, BaseModel) and base.__pydantic_generic_metadata__["args"]
    )
    if isinstance(result, str):
        return getattr(sys.modules[method_type.__module__], result)
    return result


def test_every_method_of_commol_returns_a_result_model() -> None:
    results = {_result_type(method_type) for method_type in method_classes()}

    assert len(results) > 10
    assert [result for result in results if not issubclass(result, ResultModel)] == []


def _reject_constant(name: str) -> float:
    raise AssertionError(f"{name} is not JSON")


def test_a_result_saved_to_json_loads_back_equal_with_a_nan(
    simulation: Simulation,
) -> None:
    for result in (_conformal(float("-inf"), {"B": [1.0]}), _nan_prcc(simulation)):
        text = result.model_dump_json()

        json.loads(text, parse_constant=_reject_constant)
        assert type(result).model_validate_json(text) == result
    text = _conformal(float("-inf"), {"B": [1.0]}).model_dump_json()
    assert '"lower":["-Infinity","-Infinity"]' in text
    assert '"upper":["NaN","Infinity"]' in text


def test_every_result_writes_a_number_that_is_not_finite_as_its_text() -> None:
    def subclasses(cls: type) -> Iterator[type]:
        for subclass in cls.__subclasses__():
            yield subclass
            yield from subclasses(subclass)

    assert {
        cls.__name__
        for cls in subclasses(ResultModel)
        if issubclass(cls, BaseModel)
        and cls.model_config.get("ser_json_inf_nan") != "strings"
    } == set()


def test_a_result_saved_by_0_10_with_bare_constants_still_loads() -> None:
    result = _conformal(float("-inf"), {"B": [1.0]})
    saved = (
        result.model_dump_json()
        .replace('"-Infinity"', "-Infinity")
        .replace('"Infinity"', "Infinity")
        .replace('"NaN"', "NaN")
    )

    assert "NaN" in saved and '"NaN"' not in saved
    assert ConformalResult.model_validate_json(saved) == result
