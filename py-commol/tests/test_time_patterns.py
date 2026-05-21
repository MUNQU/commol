"""
Unit and integration tests for TimePattern.

Unit tests inspect the generated formula strings directly.
Integration tests feed patterns into an abstract A→B simulation.
"""

import pytest
from pydantic import ValidationError

from commol import ModelBuilder, Simulation, TimePattern
from commol.utils.security import SecurityConfig, SecurityError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_ab(rate: str, *, steps: int = 20, pop: int = 1000) -> dict:
    """Run a simple A→B model with the given rate formula and return results."""
    model = (
        ModelBuilder("AB")
        .add_bin("A", "Source")
        .add_bin("B", "Sink")
        .add_transition("flow", ["A"], ["B"], rate=rate)
        .set_initial_conditions(
            population_size=pop,
            bin_fractions=[
                {"bin": "A", "fraction": 1.0},
                {"bin": "B", "fraction": 0.0},
            ],
        )
        .build("DifferenceEquations")
    )
    return Simulation(model).run(steps)


def _net_flow(result: dict, step: int) -> float:
    """Return how much left A between step-1 and step."""
    return result["A"][step - 1] - result["A"][step]


# ---------------------------------------------------------------------------
# pulse — unit tests
# ---------------------------------------------------------------------------


class TestPulse:
    def test_formula_contains_step_and_amount(self):
        p = TimePattern.pulse(at=5, amount=1.0)
        assert "step == 5" in str(p)
        assert "1.0" in str(p)

    def test_formula_is_parenthesized(self):
        p = TimePattern.pulse(at=5, amount=1.0)
        formula = str(p)
        assert formula.startswith("(") and formula.endswith(")")

    def test_boundary_at_zero(self):
        p = TimePattern.pulse(at=0, amount=1.0)
        assert "step == 0" in str(p)

    def test_zero_amount_is_legal(self):
        p = TimePattern.pulse(at=0, amount=0.0)
        assert str(p)

    def test_negative_at_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.pulse(at=-1, amount=1.0)

    def test_string_amount(self):
        p = TimePattern.pulse(at=3, amount="alpha")
        assert "alpha" in str(p)


# ---------------------------------------------------------------------------
# pulses — unit tests
# ---------------------------------------------------------------------------


class TestPulses:
    def test_formula_contains_all_steps(self):
        p = TimePattern.pulses(at=[3, 7, 10], amount=1.0)
        formula = str(p)
        for s in (3, 7, 10):
            assert f"step == {s}" in formula

    def test_single_step_is_legal(self):
        p = TimePattern.pulses(at=[0], amount=0.5)
        assert "step == 0" in str(p)

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.pulses(at=[], amount=1.0)

    def test_negative_step_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.pulses(at=[-1, 5], amount=1.0)

    def test_too_many_steps_raises(self):
        config = SecurityConfig()
        max_pulses = config.max_function_calls
        with pytest.raises(ValidationError):
            TimePattern.pulses(at=list(range(max_pulses + 5)), amount=1.0)

    def test_duplicate_step_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.pulses(at=[3, 3, 7], amount=1.0)

    def test_amount_with_comma_rejected(self):
        with pytest.raises(ValueError):
            TimePattern.pulses(at=[1, 2], amount="1, 0")


# ---------------------------------------------------------------------------
# periodic — unit tests
# ---------------------------------------------------------------------------


class TestPeriodic:
    def test_formula_contains_period(self):
        p = TimePattern.periodic(period=7, amount=0.1)
        assert "7" in str(p)

    def test_period_zero_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.periodic(period=0, amount=0.1)

    def test_negative_period_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.periodic(period=-1, amount=0.1)

    def test_negative_offset_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.periodic(period=7, amount=0.1, offset=-1)

    def test_offset_equal_period_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.periodic(period=7, amount=0.1, offset=7)

    def test_formula_uses_floor_based_modulo(self):
        formula = str(TimePattern.periodic(period=7, amount=0.1))
        assert "floor" in formula
        assert "%" not in formula

    def test_offset_present_in_formula(self):
        formula = str(TimePattern.periodic(period=7, amount=0.1, offset=3))
        assert "3" in formula


# ---------------------------------------------------------------------------
# window — unit tests
# ---------------------------------------------------------------------------


class TestWindow:
    def test_formula_contains_start_and_end(self):
        p = TimePattern.window(start=5, end=10, amount=0.3)
        formula = str(p)
        assert "5" in formula and "10" in formula

    def test_start_equals_end_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.window(start=5, end=5, amount=0.3)

    def test_start_greater_than_end_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.window(start=10, end=5, amount=0.3)

    def test_formula_is_parenthesized(self):
        formula = str(TimePattern.window(start=0, end=1, amount=1.0))
        assert formula.startswith("(") and formula.endswith(")")


# ---------------------------------------------------------------------------
# seasonal — unit tests
# ---------------------------------------------------------------------------


class TestSeasonal:
    def test_formula_contains_sin(self):
        formula = str(TimePattern.seasonal(amplitude=1.0, period=4))
        assert "sin" in formula

    def test_period_zero_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.seasonal(amplitude=1.0, period=0)

    def test_string_period_zero_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.seasonal(amplitude=1.0, period="0")

    def test_string_period_symbolic_is_accepted(self):
        p = TimePattern.seasonal(amplitude=1.0, period="period_param")
        assert "period_param" in str(p)

    def test_formula_contains_all_params(self):
        formula = str(
            TimePattern.seasonal(amplitude=2.0, period=12.0, phase=3.0, baseline=1.0)
        )
        assert "2.0" in formula
        assert "12.0" in formula
        assert "3.0" in formula
        assert "1.0" in formula


# ---------------------------------------------------------------------------
# gaussian_pulse — unit tests
# ---------------------------------------------------------------------------


class TestGaussianPulse:
    def test_formula_contains_exp(self):
        formula = str(TimePattern.gaussian_pulse(center=10.0, width=2.0, peak=1.0))
        assert "exp" in formula

    def test_width_zero_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.gaussian_pulse(center=0.0, width=0.0, peak=1.0)

    def test_negative_width_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.gaussian_pulse(center=0.0, width=-1.0, peak=1.0)

    def test_formula_is_parenthesized(self):
        formula = str(TimePattern.gaussian_pulse(center=5.0, width=1.0, peak=1.0))
        assert formula.startswith("(") and formula.endswith(")")


# ---------------------------------------------------------------------------
# linear_ramp — unit tests
# ---------------------------------------------------------------------------


class TestLinearRamp:
    def test_formula_contains_start_and_end(self):
        p = TimePattern.linear_ramp(start=0, end=10, start_value=0.0, end_value=1.0)
        formula = str(p)
        assert "0" in formula and "10" in formula

    def test_start_equals_end_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.linear_ramp(start=5, end=5, start_value=0.0, end_value=1.0)

    def test_start_greater_than_end_raises(self):
        with pytest.raises(ValidationError):
            TimePattern.linear_ramp(start=10, end=5, start_value=0.0, end_value=1.0)


# ---------------------------------------------------------------------------
# combine — unit tests
# ---------------------------------------------------------------------------


class TestCombine:
    def test_sum_contains_plus(self):
        a = TimePattern.pulse(at=1, amount=0.1)
        b = TimePattern.pulse(at=2, amount=0.2)
        result = str(TimePattern.combine(a, b, mode=TimePattern.SUM))
        assert "+" in result

    def test_max_contains_max(self):
        a = TimePattern.pulse(at=1, amount=0.1)
        b = TimePattern.pulse(at=2, amount=0.2)
        result = str(TimePattern.combine(a, b, mode=TimePattern.MAX))
        assert "max" in result

    def test_min_contains_min(self):
        a = TimePattern.pulse(at=1, amount=0.1)
        b = TimePattern.pulse(at=2, amount=0.2)
        result = str(TimePattern.combine(a, b, mode=TimePattern.MIN))
        assert "min" in result

    def test_single_pattern_is_identity(self):
        """combine(p) returns p unchanged."""
        a = TimePattern.pulse(at=1, amount=0.1)
        assert TimePattern.combine(a) is a

    def test_zero_patterns_raises(self):
        with pytest.raises(ValueError):
            TimePattern.combine()

    def test_result_is_parenthesized(self):
        a = TimePattern.pulse(at=1, amount=0.1)
        b = TimePattern.pulse(at=2, amount=0.2)
        result = str(TimePattern.combine(a, b))
        assert result.startswith("(") and result.endswith(")")

    def test_three_patterns_sum(self):
        a = TimePattern.pulse(at=1, amount=0.1)
        b = TimePattern.pulse(at=2, amount=0.2)
        c = TimePattern.pulse(at=3, amount=0.3)
        result = str(TimePattern.combine(a, b, c, mode=TimePattern.SUM))
        assert result.count("+") >= 2

    def test_combine_propagates_conditions(self):
        """combine of grouped patterns preserves the shared group binding."""
        conds = [{"stratification": "group", "category": "cat1"}]
        a = TimePattern.pulse(at=1, amount=0.1).for_group(conds)
        b = TimePattern.pulse(at=2, amount=0.2).for_group(conds)
        result = TimePattern.combine(a, b)
        assert result.conditions == conds

    def test_combine_propagates_source_compartment(self):
        conds = [{"stratification": "group", "category": "cat1"}]
        a = TimePattern.pulse(at=1, amount=0.1).for_group(
            conds, source_compartment="A_cat1"
        )
        b = TimePattern.pulse(at=2, amount=0.2).for_group(
            conds, source_compartment="A_cat1"
        )
        result = TimePattern.combine(a, b)
        assert result.source_compartment == "A_cat1"

    def test_combine_rejects_inconsistent_conditions(self):
        a = TimePattern.pulse(at=1, amount=0.1).for_group(
            [{"stratification": "group", "category": "cat1"}]
        )
        b = TimePattern.pulse(at=2, amount=0.2).for_group(
            [{"stratification": "group", "category": "cat2"}]
        )
        with pytest.raises(ValueError, match="conditions"):
            TimePattern.combine(a, b)

    def test_combine_rejects_inconsistent_source_compartment(self):
        conds = [{"stratification": "group", "category": "cat1"}]
        a = TimePattern.pulse(at=1, amount=0.1).for_group(
            conds, source_compartment="A_cat1"
        )
        b = TimePattern.pulse(at=2, amount=0.2).for_group(
            conds, source_compartment="B_cat1"
        )
        with pytest.raises(ValueError, match="source_compartment"):
            TimePattern.combine(a, b)

    def test_combine_security_cap_raises_clear_error(self):
        """Repeatedly combining hits the security-length cap with a clear error."""
        # Each periodic formula is ~47 chars; ~11 of them saturates the 500-char cap.
        periodics = [TimePattern.periodic(period=7, amount=0.1) for _ in range(12)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            TimePattern.combine(*periodics)


# ---------------------------------------------------------------------------
# from_formula — unit tests
# ---------------------------------------------------------------------------


class TestFromFormula:
    def test_passthrough(self):
        p = TimePattern.from_formula("beta * t")
        assert "beta * t" in str(p)

    def test_wraps_in_parens(self):
        """from_formula always wraps in parens for safe composition."""
        p = TimePattern.from_formula("beta * t")
        formula = str(p)
        assert formula.startswith("(") and formula.endswith(")")

    def test_outer_multiplication_is_safe(self):
        """f-string multiplication binds to the whole expr, not the last term."""
        p = TimePattern.from_formula("a + b")
        composed = f"2 * {p}"
        # The result is "2 * (a + b)" rather than "2 * a + b".
        assert composed == "2 * (a + b)"

    def test_already_parenthesized_expression(self):
        p = TimePattern.from_formula("(x + y)")
        assert "(x + y)" in str(p)

    def test_empty_raises(self):
        with pytest.raises((ValueError, ValidationError)):
            TimePattern.from_formula("")

    def test_too_long_raises(self):
        config = SecurityConfig()
        with pytest.raises((ValueError, ValidationError)):
            TimePattern.from_formula("x" * (config.max_expression_length + 1))

    def test_dangerous_pattern_rejected(self):
        """Full security validation runs at construction time."""
        with pytest.raises(SecurityError):
            TimePattern.from_formula("__import__")


# ---------------------------------------------------------------------------
# for_group — unit tests
# ---------------------------------------------------------------------------


class TestForGroup:
    def test_conditions_are_attached(self):
        p = TimePattern.pulse(at=5, amount=0.1)
        grouped = p.for_group([{"stratification": "age", "category": "elderly"}])
        assert grouped.conditions == [{"stratification": "age", "category": "elderly"}]

    def test_original_is_unchanged(self):
        p = TimePattern.pulse(at=5, amount=0.1)
        p.for_group([{"stratification": "age", "category": "elderly"}])
        assert p.conditions is None

    def test_source_compartment_is_attached(self):
        p = TimePattern.pulse(at=5, amount=0.1)
        grouped = p.for_group(
            [{"stratification": "age", "category": "elderly"}],
            source_compartment="S_elderly",
        )
        assert grouped.source_compartment == "S_elderly"


# ---------------------------------------------------------------------------
# to_stratified_rate — unit tests
# ---------------------------------------------------------------------------


class TestToStratifiedRate:
    def test_without_source_compartment(self):
        p = TimePattern.pulse(at=5, amount=0.1).for_group(
            [{"stratification": "age", "category": "elderly"}]
        )
        sr = p.to_stratified_rate()
        assert sr["conditions"] == [{"stratification": "age", "category": "elderly"}]
        assert "S_" not in sr["rate"]

    def test_with_source_compartment(self):
        p = TimePattern.pulse(at=5, amount=0.1).for_group(
            [{"stratification": "age", "category": "elderly"}],
            source_compartment="S_elderly",
        )
        sr = p.to_stratified_rate()
        assert "S_elderly" in sr["rate"]

    def test_no_conditions_gives_empty_list(self):
        p = TimePattern.pulse(at=5, amount=0.1)
        sr = p.to_stratified_rate()
        assert sr["conditions"] == []


# ---------------------------------------------------------------------------
# Composition safety — unit tests
# ---------------------------------------------------------------------------


class TestCompositionSafety:
    def test_outer_multiplication_respects_parens(self):
        formula = str(TimePattern.pulse(at=5, amount=1.0))
        outer = f"2 * {formula}"
        # The whole pulse expression is multiplied, not just the last token
        assert outer.startswith("2 * (")

    def test_nested_combine_is_parenthesized(self):
        a = TimePattern.pulse(at=1, amount=0.1)
        b = TimePattern.pulse(at=2, amount=0.2)
        c = TimePattern.pulse(at=3, amount=0.3)
        inner = TimePattern.combine(a, b)
        result = str(TimePattern.combine(inner, c))
        assert result.startswith("(") and result.endswith(")")


# ---------------------------------------------------------------------------
# Integration tests (simulation-based)
# ---------------------------------------------------------------------------


class TestPulseIntegration:
    def test_pulse_fires_at_correct_step(self):
        # run(15) returns 16 entries; flow at step N shows as a[N] - a[N+1]
        result = _run_ab(str(TimePattern.pulse(at=10, amount=0.5)), steps=15)
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
        result = _run_ab(str(TimePattern.pulse(at=5, amount=0.3)), steps=10, pop=1000)
        for t in range(11):
            assert result["A"][t] + result["B"][t] == pytest.approx(1000.0, abs=1e-4)


class TestPeriodicIntegration:
    def test_periodic_fires_at_correct_steps(self):
        # run(71) gives 72 entries (indices 0..71); flow at step N: a[N] - a[N+1]
        # multiples of 7 in range 0..70: 0, 7, 14, ..., 70
        result = _run_ab(str(TimePattern.periodic(period=7, amount=0.1)), steps=71)
        a = result["A"]
        fire_steps = set(range(0, 71, 7))  # {0, 7, 14, ..., 70}
        nonzero = {t for t in range(71) if a[t] - a[t + 1] > 1e-6}
        assert nonzero == fire_steps

    def test_periodic_offset_shifts_first_fire(self):
        result = _run_ab(
            str(TimePattern.periodic(period=7, amount=0.1, offset=3)), steps=30
        )
        a = result["A"]
        # Step 0 does not fire (offset=3)
        assert a[0] - a[1] == pytest.approx(0.0, abs=1e-6)
        # First flow appears at step 3
        assert a[3] - a[4] > 1e-6

    def test_period_one_fires_every_step(self):
        result = _run_ab(str(TimePattern.periodic(period=1, amount=0.1)), steps=10)
        a = result["A"]
        for t in range(10):
            assert a[t] - a[t + 1] > 1e-6


class TestWindowIntegration:
    def test_window_boundaries(self):
        # flow at step N shows as a[N] - a[N+1]
        result = _run_ab(str(TimePattern.window(start=5, end=10, amount=0.2)), steps=15)
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


class TestSeasonalIntegration:
    def test_seasonal_time_average_near_baseline(self):
        """Average flow over many full periods should approximate baseline × source."""
        # Use a small rate so A does not deplete to zero quickly
        result = _run_ab(
            str(TimePattern.seasonal(amplitude=0.001, period=10, baseline=0.005)),
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
        p = TimePattern.combine(
            TimePattern.periodic(period=2, amount=0.1),
            TimePattern.periodic(period=5, amount=0.2),
            mode=TimePattern.SUM,
        )
        result = _run_ab(str(p), steps=15)
        a = result["A"]
        flow_shared = a[0] - a[1]  # step 0: both fire
        assert flow_shared == pytest.approx(0.3 * a[0], rel=1e-4)
        flow_single = a[2] - a[3]  # step 2: only period-2 fires
        assert flow_single == pytest.approx(0.1 * a[2], rel=1e-4)

    def test_max_gives_higher_value(self):
        """MAX combine: at the pulse step, the larger pulse dominates."""
        p = TimePattern.combine(
            TimePattern.pulse(at=5, amount=0.5),
            TimePattern.pulse(at=5, amount=0.3),
            mode=TimePattern.MAX,
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
        p = TimePattern.gaussian_pulse(center=float(center), width=3.0, peak=0.001)
        result = _run_ab(str(p), steps=40, pop=100_000)
        a = result["A"]
        flows = [a[t] - a[t + 1] for t in range(40)]
        assert flows[center] == max(flows)

    def test_symmetry_around_center(self):
        """Flows are approximately symmetric around the center."""
        center = 20
        p = TimePattern.gaussian_pulse(center=float(center), width=3.0, peak=0.001)
        result = _run_ab(str(p), steps=40, pop=100_000)
        a = result["A"]
        # flow 3 steps left of center vs 3 steps right of center
        left_flow = a[center - 3] - a[center - 2]  # step center-3
        right_flow = a[center + 3] - a[center + 4]  # step center+3
        assert left_flow == pytest.approx(right_flow, rel=0.05)


class TestLinearRampIntegration:
    def test_ramp_inside_window(self):
        """Linear ramp produces increasing flow within the window."""
        p = TimePattern.linear_ramp(start=0, end=10, start_value=0.0, end_value=0.1)
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
        p = TimePattern.linear_ramp(start=5, end=10, start_value=1.0, end_value=0.0)
        result = _run_ab(str(p), steps=15)
        a = result["A"]
        for t in range(5):
            assert a[t] - a[t + 1] == pytest.approx(0.0, abs=1e-6)
        for t in range(10, 15):
            assert a[t] - a[t + 1] == pytest.approx(0.0, abs=1e-6)


class TestPulsesIntegration:
    def test_pulses_fire_at_each_listed_step(self):
        p = TimePattern.pulses(at=[3, 7, 10], amount=0.1)
        result = _run_ab(str(p), steps=15)
        a = result["A"]
        nonzero = {t for t in range(15) if a[t] - a[t + 1] > 1e-6}
        assert nonzero == {3, 7, 10}


class TestPerCompartmentInteraction:
    """Verify TimePattern formulas work correctly with per_compartment=True."""

    def test_per_compartment_with_pattern_substitutes_correctly(self):
        """With per_compartment=True the rate uses each stratified compartment."""
        # rate references base bin name `A`; per_compartment expands it per flow.
        p_str = f"{TimePattern.pulse(at=5, amount=0.1)} * A"
        model = (
            ModelBuilder("AB")
            .add_bin("A", "Source")
            .add_bin("B", "Sink")
            .add_stratification("group", ["cat1", "cat2"])
            .add_transition("flow", ["A"], ["B"], rate=p_str, per_compartment=True)
            .set_initial_conditions(
                population_size=2000,
                bin_fractions=[
                    {"bin": "A", "fraction": 1.0},
                    {"bin": "B", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "group",
                        "fractions": [
                            {"category": "cat1", "fraction": 0.5},
                            {"category": "cat2", "fraction": 0.5},
                        ],
                    }
                ],
            )
            .build("DifferenceEquations")
        )
        result = Simulation(model).run(10)
        # Each subgroup should transition independently at step 5 by 0.1 * its own A.
        # cat1 starts with 1000, cat2 with 1000 — both flows ≈ 100.
        flow_cat1 = result["A_cat1"][5] - result["A_cat1"][6]
        flow_cat2 = result["A_cat2"][5] - result["A_cat2"][6]
        assert flow_cat1 == pytest.approx(100.0, rel=1e-3)
        assert flow_cat2 == pytest.approx(100.0, rel=1e-3)


class TestSecurityValidation:
    """Pattern-generated formulas must survive add_transition's security check."""

    @pytest.mark.parametrize(
        "pattern",
        [
            TimePattern.pulse(at=5, amount=0.1),
            TimePattern.pulses(at=[1, 3, 5], amount=0.1),
            TimePattern.periodic(period=7, amount=0.1),
            TimePattern.periodic(period=7, amount=0.1, offset=2),
            TimePattern.window(start=2, end=8, amount=0.1),
            TimePattern.seasonal(amplitude=0.05, period=10, baseline=0.01),
            TimePattern.gaussian_pulse(center=5.0, width=2.0, peak=0.1),
            TimePattern.linear_ramp(start=0, end=10, start_value=0.0, end_value=0.1),
            TimePattern.combine(
                TimePattern.pulse(at=1, amount=0.1),
                TimePattern.pulse(at=2, amount=0.2),
                mode=TimePattern.MAX,
            ),
        ],
    )
    def test_pattern_survives_security_validator(self, pattern: TimePattern):
        """Each pattern's formula passes the full security validator."""
        from commol.utils.security import validate_expression_security

        validate_expression_security(str(pattern))


# ---------------------------------------------------------------------------
# Schedule (multi-group) tests — uses TimePattern.add_group directly
# ---------------------------------------------------------------------------


def _cond(stratification: str, category: str, *, to: str | None = None) -> dict:
    d: dict = {"stratification": stratification, "category": category}
    if to is not None:
        d["to"] = to
    return d


def _make_schedule_ab(
    rate: TimePattern,
    *,
    categories: list[str] | None = None,
    steps: int = 20,
    pop: int = 1000,
) -> dict:
    """Run A→B model with one stratification 'group' and a TimePattern rate."""
    if categories is None:
        categories = ["cat1", "cat2"]
    model = (
        ModelBuilder("AB")
        .add_bin("A", "Source")
        .add_bin("B", "Sink")
        .add_stratification("group", categories)
        .add_transition("flow", ["A"], ["B"], rate=rate)
        .set_initial_conditions(
            population_size=pop,
            bin_fractions=[
                {"bin": "A", "fraction": 1.0},
                {"bin": "B", "fraction": 0.0},
            ],
            stratification_fractions=[
                {
                    "stratification": "group",
                    "fractions": [
                        {"category": c, "fraction": 1.0 / len(categories)}
                        for c in categories
                    ],
                }
            ],
        )
        .build("DifferenceEquations")
    )
    return Simulation(model).run(steps)


class TestScheduleUnit:
    def test_add_group_classlevel_creates_schedule(self):
        """TimePattern.add_group as classmethod returns a TimePattern instance."""
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.pulse(at=5, amount=0.1),
        )
        assert isinstance(rate, TimePattern)
        # Builder hook returns one stratified rate, no default.
        assert rate._builder_stratified_rates() == [
            {
                "conditions": [_cond("group", "cat1")],
                "rate": str(TimePattern.pulse(at=5, amount=0.1)),
            }
        ]
        assert rate._builder_rate() is None

    def test_add_group_chained_appends(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.pulse(at=5, amount=0.1),
        ).add_group(
            conditions=[_cond("group", "cat2")],
            schedule=TimePattern.pulse(at=10, amount=0.2),
        )
        srs = rate._builder_stratified_rates()
        assert len(srs) == 2
        assert srs[0]["conditions"] == [_cond("group", "cat1")]
        assert srs[1]["conditions"] == [_cond("group", "cat2")]

    def test_set_default_attaches_fallback(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.pulse(at=5, amount=0.1),
        ).set_default(TimePattern.from_formula("0.01"))
        assert rate._builder_rate() is not None
        assert "0.01" in rate._builder_rate()

    def test_add_group_on_single_pattern_raises(self):
        """Single TimePatterns reject instance-level add_group (use the class)."""
        p = TimePattern.pulse(at=5, amount=0.1)
        with pytest.raises(TypeError, match="class-level"):
            p.add_group(conditions=[_cond("group", "cat1")], schedule=p)

    def test_set_default_on_single_pattern_raises(self):
        p = TimePattern.pulse(at=5, amount=0.1)
        with pytest.raises(TypeError, match="schedule"):
            p.set_default(TimePattern.from_formula("0.01"))

    def test_add_group_empty_conditions_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            TimePattern.add_group(
                conditions=[], schedule=TimePattern.from_formula("0.1")
            )

    def test_add_group_duplicate_raises(self):
        with pytest.raises(ValueError, match="already been registered"):
            (
                TimePattern.add_group(
                    conditions=[_cond("group", "cat1")],
                    schedule=TimePattern.from_formula("0.1"),
                ).add_group(
                    conditions=[_cond("group", "cat1")],
                    schedule=TimePattern.from_formula("0.2"),
                )
            )

    def test_set_default_twice_raises(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.from_formula("0.1"),
        ).set_default(TimePattern.from_formula("0.05"))
        with pytest.raises(ValueError, match="already been registered"):
            rate.set_default(TimePattern.from_formula("0.02"))

    def test_set_default_rejects_conditions(self):
        with pytest.raises(ValueError, match="conditions"):
            TimePattern.add_group(
                conditions=[_cond("group", "cat1")],
                schedule=TimePattern.from_formula("0.1"),
            ).set_default(
                TimePattern.from_formula("0.05").for_group([_cond("group", "cat2")])
            )

    def test_set_default_rejects_source_compartment(self):
        default_with_comp = TimePattern.from_formula("0.1").model_copy(
            update={"source_compartment": "A_cat1"}
        )
        with pytest.raises(ValueError, match="source_compartment"):
            TimePattern.add_group(
                conditions=[_cond("group", "cat1")],
                schedule=TimePattern.from_formula("0.1"),
            ).set_default(default_with_comp)

    def test_malformed_condition_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            TimePattern.add_group(
                conditions=[{"strat": "g", "cat": "c1"}],  # typo'd keys
                schedule=TimePattern.from_formula("0.1"),
            )

    def test_schedule_has_no_single_formula(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.from_formula("0.1"),
        )
        with pytest.raises(TypeError, match="no single formula"):
            _ = rate.formula

    def test_chaining_returns_same_object(self):
        rate1 = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.from_formula("0.1"),
        )
        rate2 = rate1.add_group(
            conditions=[_cond("group", "cat2")],
            schedule=TimePattern.from_formula("0.2"),
        )
        assert rate1 is rate2

    def test_source_compartment_passes_through(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.from_formula("0.1"),
            source_compartment="A_cat1",
        )
        srs = rate._builder_stratified_rates()
        assert "A_cat1" in srs[0]["rate"]


class TestScheduleIntegration:
    """The simulation engine consumes a TimePattern via add_transition(rate=...)."""

    def test_conditional_exclusion_cat2_unchanged(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.periodic(period=7, amount=0.1),
        )
        result = _make_schedule_ab(rate, steps=30)
        assert all(v == pytest.approx(0.0, abs=1e-8) for v in result["B_cat2"])

    def test_differential_schedules(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.periodic(period=7, amount=0.1),
        ).add_group(
            conditions=[_cond("group", "cat2")],
            schedule=TimePattern.periodic(period=30, amount=0.1),
        )
        result = _make_schedule_ab(rate, steps=61)
        b1_events = sum(
            1 for t in range(61) if result["B_cat1"][t + 1] - result["B_cat1"][t] > 1e-6
        )
        b2_events = sum(
            1 for t in range(61) if result["B_cat2"][t + 1] - result["B_cat2"][t] > 1e-6
        )
        assert b1_events > b2_events

    def test_default_covers_unmatched_groups(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.pulse(at=5, amount=0.5),
        ).set_default(TimePattern.pulse(at=5, amount=0.2))
        result = _make_schedule_ab(rate, steps=10)
        # Both cat1 and cat2 fire at step 5 — flow appears between index 5 and 6.
        assert result["B_cat1"][6] > result["B_cat1"][5]
        assert result["B_cat2"][6] > result["B_cat2"][5]

    def test_cross_category_routing(self):
        """`to:` condition routes flow between categories of the same bin."""
        rate = TimePattern.add_group(
            conditions=[_cond("status", "s0", to="s1")],
            schedule=TimePattern.pulse(at=5, amount=0.5),
        )

        model = (
            ModelBuilder("A_status")
            .add_bin("A", "Compartment")
            .add_stratification("status", ["s0", "s1"])
            .add_transition("flow", ["A"], ["A"], rate=rate)
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[{"bin": "A", "fraction": 1.0}],
                stratification_fractions=[
                    {
                        "stratification": "status",
                        "fractions": [
                            {"category": "s0", "fraction": 1.0},
                            {"category": "s1", "fraction": 0.0},
                        ],
                    }
                ],
            )
            .build("DifferenceEquations")
        )
        result = Simulation(model).run(10)
        a_s0_drop = result["A_s0"][5] - result["A_s0"][6]
        a_s1_gain = result["A_s1"][6] - result["A_s1"][5]
        assert a_s0_drop > 0
        assert a_s1_gain == pytest.approx(a_s0_drop, rel=1e-4)
        for t in range(11):
            total = result["A_s0"][t] + result["A_s1"][t]
            assert total == pytest.approx(1000.0, abs=1e-4)

    def test_population_conservation(self):
        rate = TimePattern.add_group(
            conditions=[_cond("group", "cat1")],
            schedule=TimePattern.periodic(period=7, amount=0.1),
        ).add_group(
            conditions=[_cond("group", "cat2")],
            schedule=TimePattern.periodic(period=7, amount=0.1),
        )
        result = _make_schedule_ab(rate, steps=30, pop=1000)
        for t in range(31):
            total = (
                result["A_cat1"][t]
                + result["A_cat2"][t]
                + result["B_cat1"][t]
                + result["B_cat2"][t]
            )
            assert total == pytest.approx(1000.0, abs=1e-4)


class TestAddTransitionAcceptsTimePattern:
    """ModelBuilder.add_transition reads a TimePattern directly via `rate=`."""

    def _build_ab(self, **kwargs) -> dict:
        return (
            ModelBuilder("AB")
            .add_bin("A", "Source")
            .add_bin("B", "Sink")
            .add_transition("flow", ["A"], ["B"], **kwargs)
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": 1.0},
                    {"bin": "B", "fraction": 0.0},
                ],
            )
            .build("DifferenceEquations")
        )

    def test_plain_pattern_via_rate(self):
        """A single TimePattern can be passed as rate= without conversion."""
        model_dict = Simulation(
            self._build_ab(rate=TimePattern.pulse(at=5, amount=0.1))
        ).run(10)
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
                    rate=TimePattern.pulse(at=5, amount=0.1),
                    stratified_rates=[{"conditions": [], "rate": "0.0"}],
                )
            )
