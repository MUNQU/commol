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
    Simulation,
    applied_parameters_report,
)
from commol.api.model_builder import BinFractionDict, StratificationFractionsDict
from commol.api.reporting import observed_report, simulation_report
from commol.constants import ModelTypes


def _override_report_model(fraction: float = 0.2) -> Model:
    return (
        ModelBuilder(name="OverrideReport", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=0.1)
        .add_transition(id="flow", source=["A"], target=["B"], rate="k * A")
        .add_stratification(id="group", categories=["group1", "group2"])
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 1.0},
                {"bin": "B", "fraction": 0.0},
            ],
            stratification_fractions=[
                {
                    "stratification": "group",
                    "fractions": [
                        {"category": "group1", "fraction": 0.5},
                        {"category": "group2", "fraction": 0.5},
                    ],
                }
            ],
            compartment_fractions=[{"compartment": "A_group1", "fraction": fraction}],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


def _initial_condition_problem(param_id: str) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(step=1, compartment="A_group1", value=1.0)],
        parameters=[
            CalibrationParameter(
                id=param_id,
                parameter_type="initial_condition",
                min_bound=0.0,
                max_bound=1.0,
            )
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )


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

    def test_population_group_rows_share_the_population_denominator(self) -> None:
        model = _override_report_model(fraction=0.3)
        report = applied_parameters_report(model, {})
        rows = report["initial_conditions"]

        effective = 0.3 + 0.5  # A_g1 override + uniform A_g2
        bin_total = rows["A"]["fraction"] + rows["B"]["fraction"]
        category_total = rows["group1"]["fraction"] + rows["group2"]["fraction"]

        assert bin_total == pytest.approx(effective)
        assert category_total == pytest.approx(effective)
        assert rows["A"]["fraction"] == pytest.approx(0.8)
        assert rows["group1"]["fraction"] == pytest.approx(0.3)
        assert rows["group2"]["fraction"] == pytest.approx(0.5)

    def test_uncalibrated_bin_keeps_its_marker_with_overrides(self) -> None:
        model = (
            ModelBuilder(name="UncalibratedOverride", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="flow", source=["A"], target=["B"], rate="k * A")
            .add_stratification(id="group", categories=["group1", "group2"])
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": None},
                    {"bin": "B", "fraction": 0.5},
                ],
                stratification_fractions=[
                    {
                        "stratification": "group",
                        "fractions": [
                            {"category": "group1", "fraction": 0.5},
                            {"category": "group2", "fraction": 0.5},
                        ],
                    }
                ],
                compartment_fractions=[{"compartment": "A_group1", "fraction": 0.3}],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        report = applied_parameters_report(model, {})

        assert report["initial_conditions"]["A"]["fraction"] is None
        assert report["initial_conditions"]["A"]["value"] is None
        # The override row still carries its own number.
        assert report["initial_conditions"]["A_group1"]["fraction"] == pytest.approx(
            0.3
        )
        assert report["initial_conditions"]["A_group1"]["value"] == pytest.approx(300.0)

    def test_an_uncalibrated_fraction_reports_no_head_count(self) -> None:
        report = applied_parameters_report(_model(s_fraction=None), {})

        assert report["initial_conditions"]["A"]["fraction"] is None
        assert report["initial_conditions"]["A"]["value"] is None

    def test_declared_compartment_overrides_are_reported(self) -> None:
        model = _override_report_model(fraction=0.2)

        report = applied_parameters_report(model, {})

        assert report["initial_conditions"]["A_group1"] == {
            "group": "population",
            "fraction": 0.2,
            "value": pytest.approx(200.0),
            "calibrated": False,
        }

    def test_exact_compartment_row_reports_calibrated(self) -> None:
        model = _override_report_model(fraction=0.2)
        problem = _initial_condition_problem("A_group1")

        model.apply_calibration_parameters({"A_group1": 0.3}, problem)
        report = applied_parameters_report(model, {"A_group1": 0.3}, problem)

        entry = report["initial_conditions"]["A_group1"]
        assert entry["fraction"] == pytest.approx(0.3)
        assert entry["calibrated"] is True

    def test_aggregate_row_written_to_overrides_reports_calibrated(self) -> None:
        model = _override_report_model(fraction=0.3)
        problem = _initial_condition_problem("A")

        model.apply_calibration_parameters({"A": 0.6}, problem)
        report = applied_parameters_report(model, {"A": 0.6}, problem)

        assert report["initial_conditions"]["A_group1"]["calibrated"] is True
        assert report["initial_conditions"]["A_group2"]["calibrated"] is True

    def test_single_overridden_compartment_sets_the_subgroup_population(self) -> None:
        model = (
            ModelBuilder(name="SingleOverride", version="1.0")
            .add_bin(id="A", name="A")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="flow", source=["A"], target=[], rate="k * A")
            .add_stratification(id="group", categories=["group1", "group2"])
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[{"bin": "A", "fraction": 1.0}],
                stratification_fractions=[
                    {
                        "stratification": "group",
                        "fractions": [
                            {"category": "group1", "fraction": 1.0},
                            {"category": "group2", "fraction": 0.0},
                        ],
                    }
                ],
                compartment_fractions=[{"compartment": "A_group1", "fraction": 0.2}],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        assert model.subgroup_population(("group1",)) == pytest.approx(200.0)

    def test_overrides_are_folded_into_multi_compartment_subgroups(self) -> None:
        model = _override_report_model(fraction=0.3)

        # group1 selects A_group1 (0.3) and B_group1 (0.0).
        assert model.subgroup_population(("group1",)) == pytest.approx(300.0)
        # group2 selects A_group2 (0.5) and B_group2 (0.0).
        assert model.subgroup_population(("group2",)) == pytest.approx(500.0)

    def test_bin_rows_report_the_effective_head_count_with_overrides(self) -> None:
        model = _override_report_model(fraction=0.3)

        report = applied_parameters_report(model, {})

        # A_g1 = 0.3 override plus the 0.5 uniform A_g2 gives an effective 0.8.
        entry = report["initial_conditions"]["A"]
        assert entry["fraction"] == pytest.approx(0.8)
        assert entry["value"] == pytest.approx(800.0)

    def test_category_rows_report_the_effective_head_count_with_overrides(
        self,
    ) -> None:
        model = _override_report_model(fraction=0.3)

        report = applied_parameters_report(model, {})

        assert report["initial_conditions"]["group1"]["value"] == pytest.approx(300.0)
        assert report["initial_conditions"]["group2"]["value"] == pytest.approx(500.0)

    def test_diverted_aggregate_row_does_not_show_the_stale_definition(
        self,
    ) -> None:
        model = _override_report_model(fraction=0.3)
        problem = _initial_condition_problem("A")

        model.apply_calibration_parameters({"A": 0.6}, problem)
        report = applied_parameters_report(model, {"A": 0.6}, problem)

        # The diverted write-back pins A_g1 = 0.225 and A_g2 = 0.375, so the
        # aggregate row must report 0.6 of the population, not the stale 1.0.
        entry = report["initial_conditions"]["A"]
        assert entry["calibrated"] is True
        assert entry["fraction"] == pytest.approx(0.6)
        assert entry["value"] == pytest.approx(600.0)

    def test_diverted_override_order_is_deterministic(self) -> None:
        model = _override_report_model(fraction=0.3)
        problem = _initial_condition_problem("A")

        model.apply_calibration_parameters({"A": 0.6}, problem)
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
        model = _model(conditional=True)
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
            model,
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

        assert report["parameters"]["k1"]["calibrated"] is True


def _accumulator_model() -> Simulation:
    bin_fractions: list[BinFractionDict] = [
        {"bin": "A", "fraction": 1.0},
        {"bin": "B", "fraction": 0.0},
    ]
    model = (
        ModelBuilder(name="Run", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_accumulator(id="events", name="Events")
        .add_parameter(id="k1", value=0.05)
        .add_transition(
            id="flow", source=["A"], target=["B"], rate="k1", accumulators=["events"]
        )
        .set_initial_conditions(population_size=1000, bin_fractions=bin_fractions)
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )
    return Simulation(model)


WINDOW = 5
WINDOWS = 8
STEPS = [(w + 1) * WINDOW for w in range(WINDOWS)]


class TestObservedReport:
    def test_values_land_on_their_window(self) -> None:
        observed = [
            ObservedDataPoint(step=5, compartment="events", value=1.0),
            ObservedDataPoint(step=15, compartment="events", value=3.0),
        ]

        report = observed_report(observed, WINDOW, WINDOWS)

        assert report["events"]["values"] == [
            1.0,
            None,
            3.0,
            None,
            None,
            None,
            None,
            None,
        ]

    def test_uniform_weights_are_omitted(self) -> None:
        observed = [ObservedDataPoint(step=5, compartment="events", value=1.0)]

        assert "weights" not in observed_report(observed, WINDOW, WINDOWS)["events"]

    def test_non_uniform_weights_are_kept(self) -> None:
        observed = [
            ObservedDataPoint(step=5, compartment="events", value=1.0, weight=2.0)
        ]

        assert observed_report(observed, WINDOW, WINDOWS)["events"]["weights"][0] == 2.0

    def test_aggregate_series_records_its_components_and_scale(self) -> None:
        observed = [
            ObservedDataPoint(
                step=5,
                compartment="events",
                value=1.0,
                compartments=["events"],
                scale_id="reporting_rate",
            )
        ]

        entry = observed_report(observed, WINDOW, WINDOWS)["events"]
        assert entry["compartments"] == ["events"]
        assert entry["scale_id"] == "reporting_rate"

    def test_observations_beyond_the_axis_are_dropped(self) -> None:
        observed = [ObservedDataPoint(step=500, compartment="events", value=1.0)]

        assert (
            observed_report(observed, WINDOW, WINDOWS)["events"]["values"]
            == [None] * WINDOWS
        )


class TestRunReport:
    def test_series_exclude_accumulator_outputs(self) -> None:
        simulation = _accumulator_model()
        results = simulation.run(max(STEPS))

        report = simulation_report(
            simulation, results, STEPS, WINDOW, WINDOWS, accumulators=["events"]
        )

        assert set(report["series"]) == {"A", "B"}
        assert set(report["accumulators"]) == {"events"}

    def test_series_are_sampled_at_the_given_steps(self) -> None:
        simulation = _accumulator_model()
        results = simulation.run(max(STEPS))

        report = simulation_report(simulation, results, STEPS, WINDOW, WINDOWS)

        assert report["series"]["A"] == pytest.approx(
            [results["A"][step] for step in STEPS]
        )

    def test_windowed_accumulator_is_the_increment_per_window(self) -> None:
        simulation = _accumulator_model()
        results = simulation.run(max(STEPS))

        report = simulation_report(
            simulation, results, STEPS, WINDOW, WINDOWS, accumulators=["events"]
        )

        windowed = report["accumulators"]["events"]["windowed"]["total"]
        assert isinstance(windowed, list)
        assert windowed == pytest.approx(
            [
                results["events"][step] - results["events"][step - WINDOW]
                for step in STEPS
            ]
        )

    def test_windowed_increments_sum_to_the_cumulative_total(self) -> None:
        simulation = _accumulator_model()
        results = simulation.run(max(STEPS))

        report = simulation_report(
            simulation, results, STEPS, WINDOW, WINDOWS, accumulators=["events"]
        )
        windowed = report["accumulators"]["events"]["windowed"]["total"]
        cumulative = report["accumulators"]["events"]["cumulative"]["total"]
        assert isinstance(windowed, list) and isinstance(cumulative, list)

        assert sum(windowed) == pytest.approx(cumulative[-1])

    def test_ensemble_leaves_become_statistics(self) -> None:
        simulation = _accumulator_model()
        results = simulation.run(max(STEPS))
        members = [results, results, results]

        report = simulation_report(
            simulation,
            results,
            STEPS,
            WINDOW,
            WINDOWS,
            accumulators=["events"],
            ensemble_runs=members,
            confidence_level=0.95,
        )

        leaf = report["series"]["A"]
        assert isinstance(leaf, dict)
        assert list(leaf) == ["mean", "median", "ci_lower", "ci_upper", "min", "max"]
        # Identical members give a degenerate interval around the single value.
        assert leaf["ci_lower"] == pytest.approx(leaf["ci_upper"])
        assert leaf["median"] == pytest.approx([results["A"][s] for s in STEPS])

    def test_layout_is_the_same_with_and_without_an_ensemble(self) -> None:
        simulation = _accumulator_model()
        results = simulation.run(max(STEPS))

        plain = simulation_report(
            simulation, results, STEPS, WINDOW, WINDOWS, accumulators=["events"]
        )
        ensemble = simulation_report(
            simulation,
            results,
            STEPS,
            WINDOW,
            WINDOWS,
            accumulators=["events"],
            ensemble_runs=[results],
            confidence_level=0.95,
        )

        assert list(plain) == list(ensemble)
        assert list(plain["accumulators"]["events"]) == list(
            ensemble["accumulators"]["events"]
        )


class TestReportPerformance:
    def test_report_with_many_compartments_and_overrides_is_bounded(self) -> None:
        import time

        builder = (
            ModelBuilder(name="Large", version="1.0")
            .add_bin(id="A", name="A")
            .add_parameter(id="k", value=0.1)
            .add_transition(
                id="flow", source=["A"], target=[], rate="k * A", per_compartment=True
            )
        )
        for index in range(8):
            builder.add_stratification(
                id=f"s{index}", categories=[f"g{index}a", f"g{index}b"]
            )
        model = builder.set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "A", "fraction": 1.0}],
            stratification_fractions=[
                {
                    "stratification": f"s{index}",
                    "fractions": [
                        {"category": f"g{index}a", "fraction": 0.5},
                        {"category": f"g{index}b", "fraction": 0.5},
                    ],
                }
                for index in range(8)
            ],
        ).build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        names = [name for name, _ in model._expanded_compartments()]
        assert len(names) == 256
        base = 1.0 / len(names)
        model.update_compartment_fractions({name: base for name in names[:128]})

        start = time.perf_counter()
        report = applied_parameters_report(model, {"g0a": 0.5})
        elapsed = time.perf_counter() - start

        assert report["initial_conditions"]["g0a"]["calibrated"] is True
        assert elapsed < 3.0
