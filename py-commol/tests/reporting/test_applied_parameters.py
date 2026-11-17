"""``Model.applied_parameters``: the values a calibration applied to a model."""

import time

import pytest

from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Model,
    NelderMeadConfig,
    ObservedDataPoint,
)
from commol._fit.result import CalibrationResult
from commol._model.engine_state import build_engine
from commol.uncertainty import ParameterSample, SampleMethod
from tests.support import (
    FactoryBuilder,
    a_to_b_builder,
    bounded,
    build,
    initial_condition_problem,
    mixed_units_builder,
    override_builder,
    shared_category_builder,
    stratify,
)


def _model(*, conditional: bool = False, a_amount: float | None = 900) -> Model:
    """A numeric parameter k1, a formula k2, and a 60/40 split of ``group``."""
    stratifications = {"group": {"group1": 0.6, "group2": 0.4}}
    if conditional:
        stratifications["sub"] = {"s1": 0.25, "s2": 0.75}
    return build(
        a_to_b_builder(
            parameters={"k1": 0.3, "k2": "k1 * 2"},
            rate="k1",
            amounts={"A": a_amount, "B": 100},
            stratifications=stratifications,
            conditions={"sub": {"group": "group1"}} if conditional else None,
        )
    )


def _problem(
    *parameters: CalibrationParameter, scale: bool = False
) -> CalibrationProblem:
    """A problem of ``parameters``, ``k1`` by default, and the factor
    ``reporting_rate`` with ``scale``."""
    declared = list(parameters) or [bounded("k1")]
    if scale:
        declared.append(bounded("reporting_rate", 0.01, 1.0, kind="factor"))
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="A_group1", value=1.0)],
        parameters=declared,
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )


def _sample(entry_id: str, values: list[float]) -> ParameterSample:
    """Equal draws of one parameter."""
    return ParameterSample.equally_weighted(
        [entry_id],
        [(0.0, 10.0)],
        [[value] for value in values],
        SampleMethod(name="test"),
    )


class TestParameters:
    def test_reports_every_numeric_parameter(self) -> None:
        report = _model().applied_parameters({"k1": 0.3}, _problem())

        assert report["parameters"]["k1"] == {"value": 0.3, "calibrated": True}

    def test_flags_parameters_the_calibration_did_not_set(self) -> None:
        report = _model().applied_parameters({"k1": 0.3}, _problem())

        assert report["parameters"]["k1"]["calibrated"] is True
        assert "k2" not in report["parameters"]

    def test_formula_parameters_are_omitted(self) -> None:
        model = _model()

        report = model.applied_parameters({}, _problem())

        assert "k2" not in report["parameters"]
        assert any(p.id == "k2" for p in model.parameters)

    def test_scale_parameters_come_from_the_problem(self) -> None:
        report = _model().applied_parameters(
            {"k1": 0.3, "reporting_rate": 0.5}, _problem(scale=True)
        )

        assert report["parameters"]["reporting_rate"] == {
            "value": 0.5,
            "calibrated": True,
        }

    def test_a_scale_parameter_gains_the_statistics_of_a_sample(self) -> None:
        report = _model().applied_parameters(
            {"k1": 0.3, "reporting_rate": 0.5},
            _problem(scale=True),
            sample=_sample("reporting_rate", [0.4, 0.6]),
            confidence_level=0.95,
        )

        spread = report["parameters"]["reporting_rate"]["spread"]
        assert spread["mean"] == pytest.approx(0.5)

    def test_a_scale_parameter_the_values_lack_is_absent(self) -> None:
        report = _model().applied_parameters({"k1": 0.3}, _problem(scale=True))

        assert "reporting_rate" not in report["parameters"]

    def test_a_value_the_problem_does_not_declare_is_refused(self) -> None:
        with pytest.raises(ValueError) as error:
            _model().applied_parameters({"k1": 0.3, "k9": 0.1}, _problem())

        assert str(error.value) == (
            "Values with no declared parameter type: ['k9']. Declared "
            "calibration parameters: ['k1']"
        )


class TestInitialAmounts:
    def test_amounts_of_different_units_are_reported_in_their_units(self) -> None:
        model = build(
            mixed_units_builder(
                stratifications={"zone": {"up": 0.25, "down": 0.75}},
                compartment_amounts={"C_up": 4e4},
            )
        )

        report = model.applied_parameters({}, _problem())

        assert report["initial_amounts"] == {
            "A": {"amount": 990, "unit": "item", "calibrated": False},
            "B": {"amount": 10, "unit": "item", "calibrated": False},
            "C": {"amount": 1.15e5, "unit": "mg/l", "calibrated": False},
            "C_up": {"amount": 4e4, "unit": "mg/l", "calibrated": False},
        }
        assert report["category_splits"] == {
            "up": {"group": [], "fraction": 0.25, "calibrated": False},
            "down": {"group": [], "fraction": 0.75, "calibrated": False},
        }

    def test_a_bin_reports_its_declared_amount(self) -> None:
        report = _model().applied_parameters({}, _problem())

        assert report["initial_amounts"]["A"] == {
            "amount": 900,
            "unit": None,
            "calibrated": False,
        }

    def test_an_uncalibrated_amount_reports_none(self) -> None:
        report = _model(a_amount=None).applied_parameters({}, _problem())

        assert report["initial_amounts"]["A"]["amount"] is None


class TestCategorySplits:
    def test_an_unconditional_category_splits_every_bin(self) -> None:
        report = _model().applied_parameters({}, _problem())

        assert report["category_splits"]["group1"] == {
            "group": [],
            "fraction": 0.6,
            "calibrated": False,
        }

    def test_a_conditional_category_reports_its_split_of_its_group(self) -> None:
        report = _model(conditional=True).applied_parameters({}, _problem())

        assert report["category_splits"]["s1"] == {
            "group": ["group1"],
            "fraction": 0.25,
            "calibrated": False,
        }

    def test_a_chain_of_conditions_names_every_category_of_its_group(self) -> None:
        model = build(
            a_to_b_builder(
                amounts={"A": 900, "B": 100},
                stratifications={
                    "group": {"group1": 0.6, "group2": 0.4},
                    "sub": {"s1": 0.25, "s2": 0.75},
                    "deep": {"d1": 0.5, "d2": 0.5},
                },
                conditions={
                    "sub": {"group": "group1"},
                    "deep": {"group": "group1", "sub": "s1"},
                },
            )
        )

        report = model.applied_parameters({}, _problem())["category_splits"]

        assert report["d1"]["group"] == ["group1", "s1"]
        assert report["d1"]["fraction"] == 0.5

    def test_a_parameter_with_the_id_of_a_category_marks_no_split(self) -> None:
        model = build(
            a_to_b_builder(
                parameters={"k": 0.1, "g1": 0.3},
                rate="k * g1",
                stratifications={"group": {"g1": 0.5, "g2": 0.5}},
            )
        )

        report = model.applied_parameters(
            {"g1": 0.3},
            _problem(bounded("g1")),
            sample=_sample("g1", [0.2, 0.4]),
            confidence_level=0.95,
        )

        assert report["parameters"]["g1"]["calibrated"] is True
        assert report["parameters"]["g1"]["spread"]["mean"] == pytest.approx(0.3)
        assert report["category_splits"]["g1"] == {
            "group": [],
            "fraction": 0.5,
            "calibrated": False,
        }

    def test_a_split_with_the_id_of_a_parameter_marks_no_parameter(self) -> None:
        model = build(
            a_to_b_builder(
                parameters={"k": 0.1, "g1": 0.3},
                rate="k * g1",
                stratifications={"group": {"g1": 0.5, "g2": 0.5}},
            )
        )
        problem = initial_condition_problem("g1")

        model.apply_calibration_parameters({"g1": 0.2}, problem)
        report = model.applied_parameters(
            {"g1": 0.2},
            problem,
            sample=_sample("g1", [0.1, 0.3]),
            confidence_level=0.95,
        )

        assert report["parameters"]["g1"] == {"value": 0.3, "calibrated": False}
        split = report["category_splits"]["g1"]
        assert split["fraction"] == pytest.approx(0.2)
        assert split["calibrated"] is True
        assert split["spread"]["mean"] == pytest.approx(0.2)

    def test_shared_category_rows_are_qualified(self) -> None:
        model = build(shared_category_builder(group=(0.3, 0.7), level=(0.6, 0.4)))

        report = model.applied_parameters(
            {"level:c1": 0.6}, initial_condition_problem("level:c1")
        )
        rows = report["category_splits"]

        assert rows["group:c1"]["fraction"] == pytest.approx(0.3)
        assert rows["level:c1"]["fraction"] == pytest.approx(0.6)
        # The split sets c1 of level and its complement, never group.
        assert rows["level:c1"]["calibrated"] is True
        assert rows["level:c2"]["calibrated"] is True
        assert rows["group:c1"]["calibrated"] is False
        assert "c1" not in rows


class TestCompartmentOverrides:
    """``override_builder``: A_g1 is set, A_g2 keeps its uniform 500."""

    def test_declared_compartment_overrides_are_reported(self) -> None:
        report = build(override_builder(200)).applied_parameters({}, _problem())

        assert report["initial_amounts"]["A_g1"] == {
            "amount": 200,
            "unit": None,
            "calibrated": False,
        }

    def test_bin_rows_report_the_effective_amount(self) -> None:
        report = build(override_builder(300)).applied_parameters({}, _problem())

        assert report["initial_amounts"]["A"]["amount"] == pytest.approx(800.0)

    def test_uncalibrated_bin_keeps_its_marker_with_overrides(self) -> None:
        model = build(
            a_to_b_builder(
                rate="k * A",
                amounts={"A": None, "B": 500},
                stratifications={"group": {"g1": 0.5, "g2": 0.5}},
                compartment_amounts={"A_g1": 300},
            )
        )

        report = model.applied_parameters({}, _problem())

        assert report["initial_amounts"]["A"]["amount"] is None
        # The override row still carries its own number.
        assert report["initial_amounts"]["A_g1"]["amount"] == 300

    def test_exact_compartment_row_reports_calibrated(self) -> None:
        model = build(override_builder(200))
        problem = initial_condition_problem("A_g1", "A_g2")
        values = {"A_g1": 300, "A_g2": 700}

        model.apply_calibration_parameters(values, problem)
        report = model.applied_parameters(values, problem)

        entry = report["initial_amounts"]["A_g1"]
        assert entry["amount"] == pytest.approx(300)
        assert entry["calibrated"] is True

    def test_aggregate_row_written_to_overrides_reports_the_engine_state(
        self,
    ) -> None:
        model = build(override_builder(300))
        problem = initial_condition_problem("A")

        # A = 600 over the override A_g1 = 300 and A_g2 = 500 is shared
        # 0.375 / 0.625, so the engine scores 225 and 375.
        model.apply_calibration_parameters({"A": 600}, problem)
        report = model.applied_parameters({"A": 600}, problem)
        rows = report["initial_amounts"]

        assert rows["A"]["calibrated"] is True
        assert rows["A"]["amount"] == pytest.approx(600.0)
        assert rows["A_g1"] == {"amount": 225.0, "unit": None, "calibrated": True}
        assert rows["A_g2"] == {"amount": 375.0, "unit": None, "calibrated": True}

    @pytest.mark.parametrize("category_id", ["g1", "group:g1"])
    def test_an_override_a_calibrated_category_covers_reports_calibrated(
        self, category_id: str
    ) -> None:
        model = build(override_builder(300))

        report = model.applied_parameters(
            {category_id: 0.5}, initial_condition_problem(category_id)
        )

        # The split writes both sides of the pair, and sets the complement.
        assert report["initial_amounts"]["A_g1"]["calibrated"] is True
        assert report["category_splits"]["g1"]["calibrated"] is True
        assert report["category_splits"]["g2"]["calibrated"] is True

    def test_the_overrides_a_calibrated_prefix_pins_report_calibrated(self) -> None:
        model = build(
            a_to_b_builder(
                stratifications={
                    "group": {"g1": 0.5, "g2": 0.5},
                    "test": {"t1": 0.5, "t2": 0.5},
                }
            )
        )
        problem = initial_condition_problem("A_g1")

        model.apply_calibration_parameters({"A_g1": 200.0}, problem)
        rows = model.applied_parameters({"A_g1": 200.0}, problem)["initial_amounts"]

        # The prefix sets half of A, so the bin keeps its amount and the
        # write-back pins the compartments of the prefix.
        assert rows["A_g1_t1"] == {"amount": 100.0, "unit": None, "calibrated": True}
        assert rows["A_g1_t2"] == {"amount": 100.0, "unit": None, "calibrated": True}
        assert rows["A"]["calibrated"] is False

    def test_the_amounts_of_the_remainder_report_calibrated(self) -> None:
        stratified = build(
            a_to_b_builder(stratifications={"group": {"g1": 0.5, "g2": 0.5}})
        )
        problem = initial_condition_problem("B_g1", remainder="A_g1")
        unstratified = build(a_to_b_builder())
        bins = initial_condition_problem("B", remainder="A")

        stratified.apply_calibration_parameters({"B_g1": 10.0}, problem)
        unstratified.apply_calibration_parameters({"B": 100.0}, bins)
        rows = stratified.applied_parameters({"B_g1": 10.0}, problem)
        bin_rows = unstratified.applied_parameters({"B": 100.0}, bins)

        assert rows["initial_amounts"]["A_g1"] == {
            "amount": 490.0,
            "unit": None,
            "calibrated": True,
        }
        assert rows["initial_amounts"]["A"]["calibrated"] is False
        assert bin_rows["initial_amounts"]["A"] == {
            "amount": 900.0,
            "unit": None,
            "calibrated": True,
        }

    def test_diverted_override_order_is_deterministic(self) -> None:
        model = build(override_builder(300))

        model.apply_calibration_parameters({"A": 600}, initial_condition_problem("A"))
        names = list(model.structure.compartment_amounts)

        assert names == sorted(names)


class TestSpreads:
    def test_an_entry_of_a_sample_has_a_spread(self) -> None:
        report = _model().applied_parameters(
            {"k1": 0.3},
            _problem(),
            sample=_sample("k1", [0.8, 1.0, 1.2]),
            confidence_level=0.95,
        )

        # Three equal draws: the 2.5 % and 97.5 % quantiles lie outside the
        # first and last midpoints (1/6, 5/6), so they are the extremes; the
        # sample variance is 0.08 / 2.
        assert report["parameters"]["k1"]["spread"] == pytest.approx(
            {
                "mean": 1.0,
                "median": 1.0,
                "ci_lower": 0.8,
                "ci_upper": 1.2,
                "min": 0.8,
                "max": 1.2,
                "std": 0.2,
            }
        )

    def test_the_spread_is_the_last_key_of_an_entry(self) -> None:
        report = _model().applied_parameters(
            {"k1": 0.3},
            _problem(),
            sample=_sample("k1", [0.8, 1.0, 1.2]),
            confidence_level=0.95,
        )

        assert list(report["parameters"]["k1"]) == ["value", "calibrated", "spread"]

    def test_a_sample_needs_a_confidence_level(self) -> None:
        with pytest.raises(ValueError) as error:
            _model().applied_parameters(
                {"k1": 0.3}, _problem(), sample=_sample("k1", [0.8, 1.2])
            )

        assert str(error.value) == "A report of a sample needs a confidence_level."

    def test_a_conditional_category_interval_is_in_the_split_it_reports(
        self,
    ) -> None:
        report = _model(conditional=True).applied_parameters(
            {"s1": 0.25},
            initial_condition_problem("s1"),
            sample=_sample("s1", [0.2, 0.25, 0.3]),
            confidence_level=0.95,
        )

        entry = report["category_splits"]["s1"]
        spread = entry["spread"]
        assert spread["mean"] == pytest.approx(0.25)
        assert spread["ci_lower"] == pytest.approx(0.2)
        assert spread["ci_upper"] == pytest.approx(0.3)
        assert spread["ci_lower"] < entry["fraction"] < spread["ci_upper"]

    def test_bin_category_and_override_rows_gain_their_statistics(self) -> None:
        sample = ParameterSample.equally_weighted(
            ["A", "g1", "A_g1"],
            [(0.0, 1000.0), (0.0, 1.0), (0.0, 1000.0)],
            [[200, 0.4, 100], [400, 0.6, 300]],
            SampleMethod(name="test"),
        )

        report = build(override_builder(300)).applied_parameters(
            {"A": 300, "g1": 0.5, "A_g1": 200},
            initial_condition_problem("A", "g1", "A_g1"),
            sample=sample,
            confidence_level=0.95,
        )

        assert report["initial_amounts"]["A"]["spread"]["mean"] == pytest.approx(300)
        assert report["category_splits"]["g1"]["spread"]["mean"] == pytest.approx(0.5)
        assert report["initial_amounts"]["A_g1"]["spread"]["mean"] == pytest.approx(200)
        # Only the row a value names takes its spread.
        assert "spread" not in report["category_splits"]["g2"]

    def test_entries_without_statistics_have_no_spread(self) -> None:
        report = _model().applied_parameters(
            {"k1": 0.3},
            _problem(),
            sample=_sample("k1", [0.8, 1.2]),
            confidence_level=0.95,
        )

        assert "spread" not in report["initial_amounts"]["A"]
        assert "spread" not in report["category_splits"]["group1"]

    def test_confidence_level_leads_the_report(self) -> None:
        report = _model().applied_parameters({}, _problem(), confidence_level=0.95)

        assert list(report)[0] == "confidence_level"
        assert report["confidence_level"] == 0.95

    def test_no_confidence_level_key_without_one(self) -> None:
        report = _model().applied_parameters({}, _problem())

        assert "confidence_level" not in report
        assert list(report) == ["parameters", "initial_amounts", "category_splits"]


class TestResultObjects:
    def test_accepts_a_calibration_result(self) -> None:
        result = CalibrationResult(
            best_parameters={"k1": 0.42},
            final_loss=1.0,
            iterations=1,
            converged=True,
            termination_reason="done",
            seed=0,
        )

        report = _model().applied_parameters(result, _problem())

        # The value is the one the model holds; the result marks it calibrated.
        assert report["parameters"]["k1"] == {"value": 0.3, "calibrated": True}


class TestReportPerformance:
    def test_report_with_many_compartments_and_overrides_is_bounded(self) -> None:
        builder = (
            FactoryBuilder(name="Large", version="1.0")
            .add_bin(id="A", name="A")
            .add_parameter(id="k", value=0.1)
            .add_transition(
                id="flow", source=["A"], target=[], rate="k * A", per_compartment=True
            )
        )
        model = build(
            builder.set_structure(
                Aggregate(
                    bin_amounts={"A": 1000},
                    stratification_fractions=stratify(
                        builder,
                        {f"s{i}": {f"g{i}a": 0.5, f"g{i}b": 0.5} for i in range(8)},
                    ),
                )
            )
        )
        names = [
            compartment.name
            for compartment in model.structure.initial_state(model.process)
        ]
        assert len(names) == 256
        model.structure.update_compartment_amounts(
            model.process, dict.fromkeys(names[:128], 1000 / 256)
        )

        start = time.perf_counter()
        build_engine(model)
        one_build = time.perf_counter() - start
        start = time.perf_counter()
        report = model.applied_parameters(
            {"g0a": 0.5}, initial_condition_problem("g0a")
        )
        elapsed = time.perf_counter() - start

        # The engine resolves the values once for the whole report, so the
        # report costs about one build of the engine, whatever the number of
        # rows and overrides; a bound relative to that build holds on a busy
        # machine and under the instrumentation of the Rust coverage.
        assert report["category_splits"]["g0a"]["calibrated"] is True
        assert elapsed < 2 * one_build + 0.1
