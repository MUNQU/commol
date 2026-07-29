"""Tests for the applied-parameter report."""

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    CalibrationResult,
    Model,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSetStatistics,
    applied_parameters_report,
)
from commol.api.model_builder import BinFractionDict, StratificationFractionsDict
from commol.constants import ModelTypes


def _model(*, conditional: bool = False, s_fraction: float | None = 0.9) -> Model:
    builder = (
        ModelBuilder(name="Report", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k1", value=0.3)
        .add_parameter(id="k2", value="k1 * 2")
        .add_transition(id="flow", source=["A"], target=["B"], rate="k1")
        .add_stratification(id="group", categories=["group1", "group2"])
    )
    stratification_fractions: list[StratificationFractionsDict] = [
        {
            "stratification": "group",
            "fractions": [
                {"category": "group1", "fraction": 0.6},
                {"category": "group2", "fraction": 0.4},
            ],
        }
    ]
    if conditional:
        builder.add_stratification(
            id="sub",
            categories=["s1", "s2"],
            conditions=[{"stratification": "group", "category": "group1"}],
        )
        stratification_fractions.append(
            {
                "stratification": "sub",
                "fractions": [
                    {"category": "s1", "fraction": 0.25},
                    {"category": "s2", "fraction": 0.75},
                ],
            }
        )
    bin_fractions: list[BinFractionDict] = [
        {"bin": "A", "fraction": s_fraction},
        {"bin": "B", "fraction": 0.1},
    ]
    builder.set_initial_conditions(
        population_size=1000,
        bin_fractions=bin_fractions,
        stratification_fractions=stratification_fractions,
    )
    return builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)


def _problem(scale: bool = False) -> CalibrationProblem:
    parameters = [
        CalibrationParameter(
            id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0
        )
    ]
    if scale:
        parameters.append(
            CalibrationParameter(
                id="reporting_rate",
                parameter_type="scale",
                min_bound=0.01,
                max_bound=1.0,
            )
        )
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(step=1, compartment="A_group1", value=1.0)],
        parameters=parameters,
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )


def _statistics(*ids: str) -> dict[str, ParameterSetStatistics]:
    return {
        entry_id: ParameterSetStatistics(
            mean=1.0,
            median=1.0,
            std=0.1,
            percentile_lower=0.9,
            percentile_upper=1.1,
            min=0.8,
            max=1.2,
        )
        for entry_id in ids
    }


class TestParameters:
    def test_reports_every_numeric_parameter(self) -> None:
        report = applied_parameters_report(_model(), {"k1": 0.3})

        assert report["parameters"]["k1"] == {"value": 0.3, "calibrated": True}

    def test_flags_parameters_the_calibration_did_not_set(self) -> None:
        report = applied_parameters_report(_model(), {"k1": 0.3})

        assert report["parameters"]["k1"]["calibrated"] is True
        assert "k2" not in report["parameters"]

    def test_formula_parameters_are_omitted(self) -> None:
        model = _model()

        report = applied_parameters_report(model, {})

        assert "k2" not in report["parameters"]
        assert any(p.id == "k2" for p in model.parameters)

    def test_scale_parameters_come_from_the_problem(self) -> None:
        report = applied_parameters_report(
            _model(), {"k1": 0.3, "reporting_rate": 0.5}, _problem(scale=True)
        )

        assert report["parameters"]["reporting_rate"] == {
            "value": 0.5,
            "calibrated": True,
        }

    def test_scale_parameters_are_absent_without_the_problem(self) -> None:
        report = applied_parameters_report(_model(), {"k1": 0.3, "reporting_rate": 0.5})

        assert "reporting_rate" not in report["parameters"]


class TestInitialConditions:
    def test_population_size_is_reported_as_n(self) -> None:
        report = applied_parameters_report(_model(), {})

        assert report["initial_conditions"]["N"] == {
            "group": "population",
            "fraction": 1.0,
            "value": 1000.0,
            "calibrated": False,
        }

    def test_bin_fractions_are_taken_of_the_whole_population(self) -> None:
        report = applied_parameters_report(_model(), {})

        assert report["initial_conditions"]["A"] == {
            "group": "population",
            "fraction": 0.9,
            "value": pytest.approx(900.0),
            "calibrated": False,
        }

    def test_unconditional_categories_are_taken_of_the_whole_population(self) -> None:
        report = applied_parameters_report(_model(), {})

        entry = report["initial_conditions"]["group1"]
        assert entry["group"] == "population"
        assert entry["value"] == pytest.approx(600.0)

    def test_conditional_categories_are_taken_of_their_own_group(self) -> None:
        report = applied_parameters_report(_model(conditional=True), {})

        entry = report["initial_conditions"]["s1"]
        assert entry["group"] == "group1"
        assert entry["fraction"] == 0.25
        # 0.25 of the group1 subgroup, not of the whole population.
        assert entry["value"] == pytest.approx(0.25 * 600.0)

    def test_an_uncalibrated_fraction_reports_no_head_count(self) -> None:
        report = applied_parameters_report(_model(s_fraction=None), {})

        assert report["initial_conditions"]["A"]["fraction"] is None
        assert report["initial_conditions"]["A"]["value"] is None


class TestIntervals:
    def test_entries_gain_an_interval_when_statistics_are_given(self) -> None:
        report = applied_parameters_report(
            _model(),
            {"k1": 0.3},
            parameter_statistics=_statistics("k1"),
            confidence_level=0.95,
        )

        assert report["parameters"]["k1"]["interval"] == {
            "mean": 1.0,
            "median": 1.0,
            "ci_lower": 0.9,
            "ci_upper": 1.1,
            "min": 0.8,
            "max": 1.2,
            "std": 0.1,
        }

    def test_entries_without_statistics_have_no_interval(self) -> None:
        report = applied_parameters_report(
            _model(), {"k1": 0.3}, parameter_statistics=_statistics("k1")
        )

        assert "interval" not in report["initial_conditions"]["A"]

    def test_confidence_level_leads_the_report(self) -> None:
        report = applied_parameters_report(_model(), {}, confidence_level=0.95)

        assert list(report)[0] == "confidence_level"

    def test_no_confidence_level_key_without_one(self) -> None:
        report = applied_parameters_report(_model(), {})

        assert "confidence_level" not in report
        assert list(report) == ["parameters", "initial_conditions"]


class TestResultObjects:
    def test_accepts_a_calibration_result(self) -> None:
        result = CalibrationResult(
            best_parameters={"k1": 0.42},
            final_loss=1.0,
            iterations=1,
            converged=True,
            termination_reason="done",
        )

        report = applied_parameters_report(_model(), result)

        assert report["parameters"]["k1"]["calibrated"] is True
