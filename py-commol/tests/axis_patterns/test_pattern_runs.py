"""Axis patterns drive the flow of a run at the steps they name."""

import math

import pytest

from commol import ModelBuilder, Simulation, AxisPattern
from commol.context.model import Model
from tests.support import a_to_b_builder, assert_close, build, build_ode, run


def _run_ab(rate: str, *, steps: int = 20, pop: int = 1000) -> dict[str, list[float]]:
    """Run A → B with the given rate formula."""
    return run(a_to_b_builder(rate=rate, parameters={}, population=pop), steps)


class TestPulseIntegration:
    def test_pulse_fires_at_correct_step(self):
        # run(15) returns 16 entries; flow at step N shows as a[N] - a[N+1]
        result = _run_ab(str(AxisPattern.pulse(at=10, amount=0.5)), steps=15)
        a = result["A"]
        # Before pulse (steps 0-9): no change
        for t in range(10):
            assert a[t] - a[t + 1] == pytest.approx(0.0, abs=1e-6)
        # At pulse (step 10): A decreases by 0.5 * a[10]
        assert a[10] - a[11] == pytest.approx(0.5 * a[10], rel=1e-5)
        # After pulse (steps 11-14): no change
        for t in range(11, 15):
            assert a[t] - a[t + 1] == pytest.approx(0.0, abs=1e-6)

    def test_pulse_conserves_population(self):
        result = _run_ab(str(AxisPattern.pulse(at=5, amount=0.3)), steps=10, pop=1000)
        for t in range(11):
            assert result["A"][t] + result["B"][t] == pytest.approx(1000.0, abs=1e-4)


class TestPeriodicIntegration:
    def test_periodic_fires_at_correct_steps(self):
        # run(71) gives 72 entries (indices 0..71); flow at step N: a[N] - a[N+1]
        # multiples of 7 in range 0..70: 0, 7, 14, ..., 70
        result = _run_ab(str(AxisPattern.periodic(period=7, amount=0.1)), steps=71)
        a = result["A"]
        fire_steps = set(range(0, 71, 7))  # {0, 7, 14, ..., 70}
        nonzero = {t for t in range(71) if a[t] - a[t + 1] > 1e-6}
        assert nonzero == fire_steps

    def test_periodic_offset_shifts_first_fire(self):
        result = _run_ab(
            str(AxisPattern.periodic(period=7, amount=0.1, offset=3)), steps=30
        )
        a = result["A"]
        # Step 0 does not fire (offset=3)
        assert a[0] - a[1] == pytest.approx(0.0, abs=1e-6)
        # First flow appears at step 3
        assert a[3] - a[4] > 1e-6

    def test_period_one_fires_every_step(self):
        result = _run_ab(str(AxisPattern.periodic(period=1, amount=0.1)), steps=10)
        a = result["A"]
        for t in range(10):
            assert a[t] - a[t + 1] > 1e-6


class TestIntervalIntegration:
    def test_interval_boundaries(self):
        # flow at step N shows as a[N] - a[N+1]
        result = _run_ab(
            str(AxisPattern.interval(start=5, end=10, amount=0.2)), steps=15
        )
        a = result["A"]
        # Before window (steps 0-4): no change
        for t in range(5):
            assert a[t] - a[t + 1] == pytest.approx(0.0, abs=1e-6)
        # Inside window (steps 5-9): nonzero flow
        for t in range(5, 10):
            assert a[t] - a[t + 1] > 1e-6
        # At end and after (steps 10-14): no change
        for t in range(10, 15):
            assert a[t] - a[t + 1] == pytest.approx(0.0, abs=1e-6)


class TestDifferentialIntegration:
    """A pattern holds for one time unit: `amount` acts as a constant rate."""

    @staticmethod
    def _run_ode(pattern: AxisPattern, steps: int) -> list[float]:
        builder = a_to_b_builder(rate=str(pattern), parameters={})
        return Simulation(build_ode(builder)).run(steps)["A"]

    def test_interval_acts_on_its_unit_intervals(self):
        a = self._run_ode(AxisPattern.interval(start=2, end=4, amount=0.5), steps=6)
        assert_close(a[2], 1000.0, rel=1e-12)
        assert_close(a[4], 1000.0 * math.exp(-1.0), rel=1e-7)
        assert_close(a[6], a[4], rel=1e-12)

    def test_an_interval_opens_between_steps(self):
        # The rate 0.5 acts on [2.5, 4): A keeps exp(-1.5 * 0.5) of its start.
        a = self._run_ode(AxisPattern.interval(start=2.5, end=4, amount=0.5), steps=6)
        assert_close(a[2], 1000.0, rel=1e-12)
        assert_close(a[4], 1000.0 * math.exp(-0.75), rel=1e-7)
        assert_close(a[6], a[4], rel=1e-12)

    def test_a_ramp_matches_its_integral(self):
        # r(t) = 0.01 t on [0, 10), so A(t) = 1000 exp(-0.005 t^2) there.
        ramp = AxisPattern.linear_ramp(start=0, end=10, start_value=0.0, end_value=0.1)
        a = self._run_ode(ramp, steps=12)
        for t in (3, 5, 10):
            assert_close(a[t], 1000.0 * math.exp(-0.005 * t * t), rel=1e-7)
        assert_close(a[12], a[10], rel=1e-12)

    def test_periodic_acts_on_each_period_start(self):
        a = self._run_ode(AxisPattern.periodic(period=5, amount=0.5), steps=20)
        # [0, 1), [5, 6), [10, 11) and [15, 16): four time units.
        assert_close(a[20], 1000.0 * math.exp(-2.0), rel=1e-7)


class TestSeasonalIntegration:
    def test_seasonal_time_average_near_baseline(self):
        """Average flow over many full periods should approximate baseline × source."""
        # Use a small rate so A does not deplete to zero quickly
        result = _run_ab(
            str(AxisPattern.seasonal(amplitude=0.001, period=10, baseline=0.005)),
            steps=200,
            pop=10_000,
        )
        a = result["A"]
        # flow at step t = a[t] - a[t+1], per-capita rate ≈ 0.005 on average
        flows = [a[t] - a[t + 1] for t in range(200)]
        avg_flow = sum(flows) / len(flows)
        avg_a = sum(a[:200]) / 200
        # avg_flow ≈ baseline * avg_A (sin averages to 0 over full periods)
        assert avg_flow == pytest.approx(0.005 * avg_a, rel=0.05)


class TestCombineIntegration:
    def test_sum_is_additive_at_shared_step(self):
        """At a step that fires both sub-patterns, the sum applies."""
        # period=2 fires at steps 0,2,4,...; period=5 fires at steps 0,5,10,...
        # step 0: both fire → flow = (0.1 + 0.2) * a[0]
        # step 2: only period-2 fires → flow = 0.1 * a[2]
        p = AxisPattern.combine(
            AxisPattern.periodic(period=2, amount=0.1),
            AxisPattern.periodic(period=5, amount=0.2),
            mode=AxisPattern.SUM,
        )
        result = _run_ab(str(p), steps=15)
        a = result["A"]
        flow_shared = a[0] - a[1]  # step 0: both fire
        assert flow_shared == pytest.approx(0.3 * a[0], rel=1e-4)
        flow_single = a[2] - a[3]  # step 2: only period-2 fires
        assert flow_single == pytest.approx(0.1 * a[2], rel=1e-4)

    def test_max_gives_higher_value(self):
        """MAX combine: at the pulse step, the larger pulse dominates."""
        p = AxisPattern.combine(
            AxisPattern.pulse(at=5, amount=0.5),
            AxisPattern.pulse(at=5, amount=0.3),
            mode=AxisPattern.MAX,
        )
        result = _run_ab(str(p), steps=10)
        a = result["A"]
        # flow at step 5 = a[5] - a[6]
        assert a[5] - a[6] == pytest.approx(0.5 * a[5], rel=1e-4)


class TestGaussianPulseIntegration:
    def test_peak_at_center(self):
        """Flow at the center step should exceed all other steps."""
        center = 20
        # Small peak so depletion is negligible (< 5% total over all steps)
        p = AxisPattern.gaussian_pulse(center=float(center), width=3.0, peak=0.001)
        result = _run_ab(str(p), steps=40, pop=100_000)
        a = result["A"]
        flows = [a[t] - a[t + 1] for t in range(40)]
        assert flows[center] == max(flows)

    def test_symmetry_around_center(self):
        """Flows are approximately symmetric around the center."""
        center = 20
        p = AxisPattern.gaussian_pulse(center=float(center), width=3.0, peak=0.001)
        result = _run_ab(str(p), steps=40, pop=100_000)
        a = result["A"]
        # flow 3 steps left of center vs 3 steps right of center
        left_flow = a[center - 3] - a[center - 2]  # step center-3
        right_flow = a[center + 3] - a[center + 4]  # step center+3
        assert left_flow == pytest.approx(right_flow, rel=0.05)


class TestLinearRampIntegration:
    def test_ramp_inside_window(self):
        """Linear ramp produces increasing flow within the window."""
        p = AxisPattern.linear_ramp(start=0, end=10, start_value=0.0, end_value=0.1)
        result = _run_ab(str(p), steps=12)
        a = result["A"]
        # Per-capita rate increases from 0 at step 0 to ~0.09 at step 9.
        # Flow at step t = a[t] - a[t+1] should grow as we move from t=1 to t=8
        # (early flows dominated by small rate; later flows by larger rate).
        flow_early = a[1] - a[2]
        flow_late = a[8] - a[9]
        assert flow_late > flow_early

    def test_ramp_outside_window_is_zero(self):
        """Outside [start, end), flow is exactly zero."""
        p = AxisPattern.linear_ramp(start=5, end=10, start_value=1.0, end_value=0.0)
        result = _run_ab(str(p), steps=15)
        a = result["A"]
        for t in range(5):
            assert a[t] - a[t + 1] == pytest.approx(0.0, abs=1e-6)
        for t in range(10, 15):
            assert a[t] - a[t + 1] == pytest.approx(0.0, abs=1e-6)


class TestPulsesIntegration:
    def test_pulses_fire_at_each_listed_step(self):
        p = AxisPattern.pulses(at=[3, 7, 10], amount=0.1)
        result = _run_ab(str(p), steps=15)
        a = result["A"]
        nonzero = {t for t in range(15) if a[t] - a[t + 1] > 1e-6}
        assert nonzero == {3, 7, 10}

    def test_per_step_amounts_fire_with_distinct_magnitudes(self):
        p = AxisPattern.pulses(at=[2, 6, 11], amount=[0.1, 0.2, 0.4])
        result = _run_ab(str(p), steps=15)
        a = result["A"]
        # only the three listed steps trigger a flow
        nonzero = {t for t in range(15) if a[t] - a[t + 1] > 1e-9}
        assert nonzero == {2, 6, 11}
        # per-capita rates differ by the prescribed amounts (0.1, 0.2, 0.4)
        rate2 = (a[2] - a[3]) / a[2]
        rate6 = (a[6] - a[7]) / a[6]
        rate11 = (a[11] - a[12]) / a[11]
        assert rate6 == pytest.approx(rate2 * 2, rel=1e-6)
        assert rate11 == pytest.approx(rate2 * 4, rel=1e-6)


class TestPerCompartmentInteraction:
    """Verify AxisPattern formulas work correctly with per_compartment=True."""

    def test_per_compartment_with_pattern_substitutes_correctly(self):
        """With per_compartment=True the rate uses each stratified compartment."""
        # rate references base bin name `A`; per_compartment expands it per flow.
        p_str = f"{AxisPattern.pulse(at=5, amount=0.1)} * A"
        model = build(
            a_to_b_builder(
                rate=p_str,
                parameters={},
                per_compartment=True,
                population=2000,
                stratifications={"group": {"cat1": 0.5, "cat2": 0.5}},
            )
        )
        result = run(model, 10)
        # Each subgroup should transition independently at step 5 by 0.1 * its own A.
        # cat1 starts with 1000, cat2 with 1000 — both flows ≈ 100.
        flow_cat1 = result["A_cat1"][5] - result["A_cat1"][6]
        flow_cat2 = result["A_cat2"][5] - result["A_cat2"][6]
        assert flow_cat1 == pytest.approx(100.0, rel=1e-3)
        assert flow_cat2 == pytest.approx(100.0, rel=1e-3)


class TestAddTransitionAcceptsAxisPattern:
    """ModelBuilder.add_transition reads a AxisPattern directly via `rate=`."""

    def _build_ab(self, rate: str | float | AxisPattern | None) -> Model:
        return build(a_to_b_builder(rate=rate, parameters={}))

    def test_plain_pattern_via_rate(self):
        """A single AxisPattern can be passed as rate= without conversion."""
        model_dict = run(self._build_ab(rate=AxisPattern.pulse(at=5, amount=0.1)), 10)
        a = model_dict["A"]
        # Nonzero per-capita flow only at step 5; zero elsewhere.
        for t in range(10):
            delta = a[t] - a[t + 1]
            if t == 5:
                assert delta == pytest.approx(0.1 * a[5], rel=1e-5)
            else:
                assert delta == pytest.approx(0.0, abs=1e-6)

    def test_combined_rate_and_stratified_rates_raises(self):
        with pytest.raises(ValueError, match="not both"):
            (
                ModelBuilder("AB")
                .add_bin("A", "Source")
                .add_bin("B", "Sink")
                .add_transition(
                    "flow",
                    ["A"],
                    ["B"],
                    rate=AxisPattern.pulse(at=5, amount=0.1),
                    stratified_rates=[{"conditions": [], "rate": "0.0"}],
                )
            )

    def test_generated_time_series_parameter_collision_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            (
                ModelBuilder("AB")
                .add_bin("A", "Source")
                .add_bin("B", "Sink")
                .add_parameter("_series_0", 1.0)
                .add_transition(
                    "flow",
                    ["A"],
                    ["B"],
                    rate=AxisPattern.pulse(at=5, amount=0.1),
                )
            )
