"""``applied_parameters_report``: the values a calibration applied to a model."""

import time

import pytest

from commol import (
    CalibrationProblem,
    CalibrationResult,
    Model,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSetStatistics,
    applied_parameters_report,
)
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    initial_condition_problem,
    override_builder,
    shared_category_builder,
    stratify,
)


def _model(*, conditional: bool = False, s_fraction: float | None = 0.9) -> Model:
    """A numeric parameter k1, a formula k2, and a 60/40 split of ``group``."""
    stratifications = {"group": {"group1": 0.6, "group2": 0.4}}
    if conditional:
        stratifications["sub"] = {"s1": 0.25, "s2": 0.75}
    return build(
        a_to_b_builder(
            parameters={"k1": 0.3, "k2": "k1 * 2"},
            rate="k1",
            fractions={"A": s_fraction, "B": 0.1},
            stratifications=stratifications,
            conditions={"sub": {"group": "group1"}} if conditional else None,
        )
    )


def _problem(scale: bool = False) -> CalibrationProblem:
    parameters = [bounded("k1")]
    if scale:
        parameters.append(bounded("reporting_rate", 0.01, 1.0, kind="factor"))
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="A_group1", value=1.0)],
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

    def test_conditional_category_fractions_use_the_population_denominator(
        self,
    ) -> None:
        report = applied_parameters_report(_model(conditional=True), {})

        entry = report["initial_conditions"]["s1"]
        assert entry["group"] == "group1"
        # The head count is 0.25 of the group1 subgroup; the reported fraction
        # is that head count over the declared population size, so every row
        # shares one denominator.
        assert entry["value"] == pytest.approx(0.25 * 600.0)
        assert entry["fraction"] == pytest.approx(150.0 / 1000.0)

    def test_an_uncalibrated_fraction_reports_no_head_count(self) -> None:
        report = applied_parameters_report(_model(s_fraction=None), {})

        assert report["initial_conditions"]["A"]["fraction"] is None
        assert report["initial_conditions"]["A"]["value"] is None

    def test_shared_category_rows_are_qualified(self) -> None:
        model = build(shared_category_builder(age=(0.3, 0.7), sex=(0.6, 0.4)))

        report = applied_parameters_report(model, {"sex:y": 0.6})
        rows = report["initial_conditions"]

        assert rows["age:y"]["fraction"] == pytest.approx(0.3)
        assert rows["sex:y"]["fraction"] == pytest.approx(0.6)
        assert rows["sex:y"]["calibrated"] is True
        assert rows["age:y"]["calibrated"] is False
        assert "y" not in rows


class TestCompartmentOverrides:
    """``override_builder``: A_g1 is set, A_g2 keeps its uniform 0.5."""

    def test_declared_compartment_overrides_are_reported(self) -> None:
        report = applied_parameters_report(build(override_builder(0.2)), {})

        assert report["initial_conditions"]["A_g1"] == {
            "group": "population",
            "fraction": 0.2,
            "value": pytest.approx(200.0),
            "calibrated": False,
        }

    def test_population_group_rows_share_the_population_denominator(self) -> None:
        report = applied_parameters_report(build(override_builder(0.3)), {})
        rows = report["initial_conditions"]

        effective = 0.3 + 0.5  # A_g1 override + uniform A_g2
        assert rows["A"]["fraction"] + rows["B"]["fraction"] == pytest.approx(effective)
        assert rows["g1"]["fraction"] + rows["g2"]["fraction"] == pytest.approx(
            effective
        )
        assert rows["A"]["fraction"] == pytest.approx(0.8)
        assert rows["g1"]["fraction"] == pytest.approx(0.3)
        assert rows["g2"]["fraction"] == pytest.approx(0.5)

    def test_bin_rows_report_the_effective_head_count(self) -> None:
        report = applied_parameters_report(build(override_builder(0.3)), {})

        entry = report["initial_conditions"]["A"]
        assert entry["fraction"] == pytest.approx(0.8)
        assert entry["value"] == pytest.approx(800.0)

    def test_category_rows_report_the_effective_head_count(self) -> None:
        report = applied_parameters_report(build(override_builder(0.3)), {})

        assert report["initial_conditions"]["g1"]["value"] == pytest.approx(300.0)
        assert report["initial_conditions"]["g2"]["value"] == pytest.approx(500.0)

    def test_uncalibrated_bin_keeps_its_marker_with_overrides(self) -> None:
        model = build(
            a_to_b_builder(
                rate="k * A",
                fractions={"A": None, "B": 0.5},
                stratifications={"group": {"g1": 0.5, "g2": 0.5}},
                compartment_fractions=[{"compartment": "A_g1", "fraction": 0.3}],
            )
        )

        report = applied_parameters_report(model, {})

        assert report["initial_conditions"]["A"]["fraction"] is None
        assert report["initial_conditions"]["A"]["value"] is None
        # The override row still carries its own number.
        assert report["initial_conditions"]["A_g1"]["fraction"] == pytest.approx(0.3)
        assert report["initial_conditions"]["A_g1"]["value"] == pytest.approx(300.0)

    def test_exact_compartment_row_reports_calibrated(self) -> None:
        model = build(override_builder(0.2))
        problem = initial_condition_problem("A_g1", "A_g2")
        values = {"A_g1": 0.3, "A_g2": 0.7}

        model.apply_calibration_parameters(values, problem)
        report = applied_parameters_report(model, values, problem)

        entry = report["initial_conditions"]["A_g1"]
        assert entry["fraction"] == pytest.approx(0.3)
        assert entry["calibrated"] is True

    def test_aggregate_row_written_to_overrides_reports_calibrated(self) -> None:
        model = build(override_builder(0.3))
        problem = initial_condition_problem("A")

        model.apply_calibration_parameters({"A": 0.6}, problem)
        report = applied_parameters_report(model, {"A": 0.6}, problem)

        assert report["initial_conditions"]["A_g1"]["calibrated"] is True
        assert report["initial_conditions"]["A_g2"]["calibrated"] is True

    def test_aggregate_row_written_to_overrides_reports_the_engine_state(
        self,
    ) -> None:
        model = build(override_builder(0.3))
        problem = initial_condition_problem("A")

        # A = 0.6 over the override A_g1 = 0.3 and A_g2 = 0.5 is shared
        # 0.375 / 0.625, so the engine scores 225 and 375.
        model.apply_calibration_parameters({"A": 0.6}, problem)
        report = applied_parameters_report(model, {"A": 0.6}, problem)

        entry = report["initial_conditions"]["A"]
        assert entry["calibrated"] is True
        assert entry["fraction"] == pytest.approx(0.6)
        assert entry["value"] == pytest.approx(600.0)
        assert report["initial_conditions"]["A_g1"]["value"] == pytest.approx(225.0)

    def test_diverted_override_order_is_deterministic(self) -> None:
        model = build(override_builder(0.3))

        model.apply_calibration_parameters({"A": 0.6}, initial_condition_problem("A"))
        names = [
            override.compartment
            for override in model.population.initial_conditions.compartment_fractions
        ]

        assert names == sorted(names)


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

    def test_conditional_category_interval_is_rescaled_to_the_population(
        self,
    ) -> None:
        # A conditional category is calibrated as a fraction of its subgroup but
        # reported as a fraction of the whole population. The interval must be
        # rescaled by the group share so it brackets the reported fraction.
        statistics = {
            "s1": ParameterSetStatistics(
                mean=0.25,
                median=0.25,
                std=0.05,
                percentile_lower=0.2,
                percentile_upper=0.3,
                min=0.15,
                max=0.35,
            )
        }

        report = applied_parameters_report(
            _model(conditional=True),
            {},
            parameter_statistics=statistics,
            confidence_level=0.95,
        )

        entry = report["initial_conditions"]["s1"]
        interval = entry["interval"]
        assert entry["fraction"] == pytest.approx(0.25 * 0.6)
        assert interval["mean"] == pytest.approx(0.25 * 0.6)
        assert interval["ci_lower"] == pytest.approx(0.2 * 0.6)
        assert interval["ci_upper"] == pytest.approx(0.3 * 0.6)
        assert interval["ci_lower"] < entry["fraction"] < interval["ci_upper"]

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

        # The value is the one the model holds; the result marks it calibrated.
        assert report["parameters"]["k1"] == {"value": 0.3, "calibrated": True}


class TestReportPerformance:
    def test_report_with_many_compartments_and_overrides_is_bounded(self) -> None:
        builder = (
            ModelBuilder(name="Large", version="1.0")
            .add_bin(id="A", name="A")
            .add_parameter(id="k", value=0.1)
            .add_transition(
                id="flow", source=["A"], target=[], rate="k * A", per_compartment=True
            )
        )
        model = build(
            builder.set_initial_conditions(
                population_size=1000,
                bin_fractions=[{"bin": "A", "fraction": 1.0}],
                stratification_fractions=stratify(
                    builder,
                    {f"s{i}": {f"g{i}a": 0.5, f"g{i}b": 0.5} for i in range(8)},
                ),
            )
        )
        names = [compartment.name for compartment in model.initial_state()]
        assert len(names) == 256
        model.update_compartment_fractions({name: 1 / 256 for name in names[:128]})

        start = time.perf_counter()
        report = applied_parameters_report(model, {"g0a": 0.5})
        elapsed = time.perf_counter() - start

        assert report["initial_conditions"]["g0a"]["calibrated"] is True
        assert elapsed < 3.0
