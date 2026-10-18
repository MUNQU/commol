"""``observed_report`` and ``simulation_report``: series on a window grid."""

import pytest

from commol import ObservedDataPoint, Simulation, window_index
from commol.api.reporting import observed_report, simulation_report
from tests.support import a_to_b_builder, build, build_ode


def _accumulator_model() -> Simulation:
    builder = a_to_b_builder(parameters={"k1": 0.05}, rate="k1", accumulator="events")
    return Simulation(build(builder))


WINDOW = 5
WINDOWS = 8
STEPS = [(w + 1) * WINDOW for w in range(WINDOWS)]


class TestObservedReport:
    def test_a_non_integer_step_lands_on_the_window_that_holds_it(self) -> None:
        # Window w holds the steps (5 w, 5 (w + 1)].
        observed = [
            ObservedDataPoint(at=3.5, output="events", value=1.0),
            ObservedDataPoint(at=5.5, output="events", value=2.0),
        ]

        report = observed_report(observed, WINDOW, WINDOWS)

        assert report["events"]["values"][:3] == [1.0, 2.0, None]

    def test_a_differential_model_is_reported(self) -> None:
        builder = a_to_b_builder(
            parameters={"k1": 0.05}, rate="k1", accumulator="events"
        )
        simulation = Simulation(build_ode(builder))
        results = simulation.run(WINDOW * WINDOWS)

        report = simulation_report(
            simulation, results, STEPS, WINDOW, WINDOWS, accumulators=["events"]
        )

        assert set(report) >= {"observed", "series", "accumulators"}

    def test_values_land_on_their_window(self) -> None:
        observed = [
            ObservedDataPoint(at=5, output="events", value=1.0),
            ObservedDataPoint(at=15, output="events", value=3.0),
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
        observed = [ObservedDataPoint(at=5, output="events", value=1.0)]

        assert "weights" not in observed_report(observed, WINDOW, WINDOWS)["events"]

    def test_non_uniform_weights_are_kept(self) -> None:
        observed = [ObservedDataPoint(at=5, output="events", value=1.0, weight=2.0)]

        assert observed_report(observed, WINDOW, WINDOWS)["events"]["weights"][0] == 2.0

    def test_aggregate_series_records_its_components_and_scale(self) -> None:
        observed = [
            ObservedDataPoint(
                at=5,
                label="events",
                value=1.0,
                outputs=["events"],
                factor_id="reporting_rate",
            )
        ]

        entry = observed_report(observed, WINDOW, WINDOWS)["events"]
        assert entry["outputs"] == ["events"]
        assert entry["factor_id"] == "reporting_rate"

    def test_observations_beyond_the_axis_are_dropped(self) -> None:
        observed = [ObservedDataPoint(at=500, output="events", value=1.0)]

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

    def test_an_empty_ensemble_reports_the_run(self) -> None:
        simulation = _accumulator_model()
        results = simulation.run(max(STEPS))

        plain = simulation_report(
            simulation, results, STEPS, WINDOW, WINDOWS, accumulators=["events"]
        )
        empty = simulation_report(
            simulation,
            results,
            STEPS,
            WINDOW,
            WINDOWS,
            accumulators=["events"],
            ensemble_runs=[],
            confidence_level=0.95,
        )

        assert empty == plain

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


class TestNonIntegerPositions:
    """A differential model reports on a window of 2.5, read at its positions."""

    WINDOW = 2.5
    ENDS = [2.5 * (w + 1) for w in range(4)]

    def _simulation(self) -> Simulation:
        builder = a_to_b_builder(
            parameters={"k1": 0.05}, rate="k1", accumulator="events"
        )
        return Simulation(build_ode(builder))

    def test_series_and_windows_match_run_at(self) -> None:
        simulation = self._simulation()
        results = simulation.run_at([0.0, *self.ENDS])

        report = simulation_report(
            simulation, results, self.ENDS, self.WINDOW, 4, accumulators=["events"]
        )

        at = simulation.run_at(self.ENDS)
        assert report["series"]["A"] == pytest.approx(at["A"], rel=1e-9)
        windowed = report["accumulators"]["events"]["windowed"]["total"]
        assert windowed == pytest.approx(
            [b - a for a, b in zip([0.0, *at["events"]], at["events"])], rel=1e-9
        )

    def test_a_position_the_results_do_not_hold_is_an_error(self) -> None:
        simulation = self._simulation()
        results = simulation.run(10)

        with pytest.raises(ValueError, match="no value at 2.5"):
            simulation_report(simulation, results, self.ENDS, self.WINDOW, 4)


@pytest.mark.parametrize("window", [5, 2.5, 0.1])
def test_observed_report_places_each_observation_by_window_index(
    window: float,
) -> None:
    positions = [0, 0.05, 0.1, 1, 2.5, 3.3, 5, 7.5, 9.99]
    observed = [
        ObservedDataPoint(at=at, output="events", value=float(i))
        for i, at in enumerate(positions)
    ]
    windows = window_index(max(positions), window) + 1

    report = observed_report(observed, window, windows)

    expected: list[float | None] = [None] * windows
    for i, at in enumerate(positions):
        expected[window_index(at, window)] = float(i)
    assert report["events"]["values"] == expected
