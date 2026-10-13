"""``observed_report`` and ``simulation_report``: series on a window grid."""

import pytest

from commol import ObservedDataPoint, Simulation
from commol.api.reporting import observed_report, simulation_report
from tests.support import a_to_b_builder, build


def _accumulator_model() -> Simulation:
    builder = a_to_b_builder(parameters={"k1": 0.05}, rate="k1", accumulator="events")
    return Simulation(build(builder))


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
