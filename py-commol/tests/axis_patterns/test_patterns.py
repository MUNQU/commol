"""Axis patterns build the formulas they promise, and reject bad arguments."""

import pytest
from pydantic import ValidationError

from commol import ModelBuilder, AxisPattern
from commol.utils.security import (
    SecurityConfig,
    SecurityError,
    validate_expression_security,
)
from tests.axis_patterns.conditions import condition, rate_string


class TestPulse:
    def test_formula_contains_step_and_amount(self):
        p = AxisPattern.pulse(at=5, value=1.0)
        assert "step == 5" in str(p)
        assert "1.0" in str(p)

    def test_formula_is_parenthesized(self):
        p = AxisPattern.pulse(at=5, value=1.0)
        formula = str(p)
        assert formula.startswith("(") and formula.endswith(")")

    def test_boundary_at_zero(self):
        p = AxisPattern.pulse(at=0, value=1.0)
        assert "step == 0" in str(p)

    def test_zero_amount_is_legal(self):
        p = AxisPattern.pulse(at=0, value=0.0)
        assert str(p) == "(if(step == 0, 0.0, 0))"

    def test_negative_at_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.pulse(at=-1, value=1.0)

    def test_string_amount(self):
        p = AxisPattern.pulse(at=3, value="alpha")
        assert "alpha" in str(p)


class TestPulses:
    def test_formula_contains_all_steps(self):
        p = AxisPattern.pulses(at=[3, 7, 10], value=1.0)
        formula = str(p)
        for s in (3, 7, 10):
            assert f"step == {s}" in formula

    def test_single_step_is_legal(self):
        p = AxisPattern.pulses(at=[0], value=0.5)
        assert "step == 0" in str(p)

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.pulses(at=[], value=1.0)

    def test_negative_step_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.pulses(at=[-1, 5], value=1.0)

    def test_large_step_count_accepted(self):
        p = AxisPattern.pulses(at=list(range(50)), value=1.0)
        assert str(p).count("step ==") == 50

    def test_duplicate_step_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.pulses(at=[3, 3, 7], value=1.0)

    def test_amount_with_comma_rejected(self):
        with pytest.raises(ValueError):
            AxisPattern.pulses(at=[1, 2], value="1, 0")

    def test_per_step_amounts_formula(self):
        p = AxisPattern.pulses(at=[2, 5, 9], value=[0.1, 0.2, 0.3])
        formula = str(p)
        assert "step == 2, 0.1" in formula
        assert "step == 5, 0.2" in formula
        assert "step == 9, 0.3" in formula

    def test_per_step_amounts_string_values(self):
        p = AxisPattern.pulses(at=[1, 4], value=["alpha", "beta"])
        formula = str(p)
        assert "step == 1, alpha" in formula
        assert "step == 4, beta" in formula

    def test_per_step_amounts_length_mismatch_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.pulses(at=[1, 2, 3], value=[0.1, 0.2])

    def test_per_step_amounts_invalid_element_raises(self):
        with pytest.raises(ValueError):
            AxisPattern.pulses(at=[1, 2], value=[0.1, "1, 0"])


class TestSeriesParameterValue:
    def test_series_parameter_rejects_empty_data(self):
        with pytest.raises(ValidationError, match="at least one point"):
            ModelBuilder("M").add_parameter("x", [], mode="pulse")

    def test_series_parameter_rejects_negative_positions(self):
        with pytest.raises(ValidationError, match="non-negative"):
            ModelBuilder("M").add_parameter("x", [(-1, 1.0)], mode="pulse")

    def test_series_parameter_rejects_duplicate_positions(self):
        with pytest.raises(ValidationError, match="unique"):
            ModelBuilder("M").add_parameter("x", [(1, 1.0), (1, 2.0)], mode="pulse")

    def test_duplicate_user_parameter_id_raises_immediately(self):
        with pytest.raises(ValueError, match="already exists"):
            ModelBuilder("M").add_parameter("x", 1.0).add_parameter("x", 2.0)


class TestPeriodic:
    def test_formula_contains_period(self):
        p = AxisPattern.periodic(period=7, value=0.1)
        assert "7" in str(p)

    def test_period_zero_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.periodic(period=0, value=0.1)

    def test_negative_period_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.periodic(period=-1, value=0.1)

    def test_negative_offset_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.periodic(period=7, value=0.1, offset=-1)

    def test_offset_equal_period_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.periodic(period=7, value=0.1, offset=7)

    def test_formula_uses_floor_based_modulo(self):
        formula = str(AxisPattern.periodic(period=7, value=0.1))
        assert "floor" in formula
        assert "%" not in formula

    def test_offset_present_in_formula(self):
        formula = str(AxisPattern.periodic(period=7, value=0.1, offset=3))
        assert "3" in formula


class TestInterval:
    def test_formula_reads_t_between_its_bounds(self):
        p = AxisPattern.interval(start=5, end=10, value=0.3)
        assert str(p) == "(if(t >= 5, if(t < 10, 0.3, 0), 0))"

    def test_bounds_may_lie_between_steps(self):
        p = AxisPattern.interval(start=2.5, end=4, value=0.5)
        assert str(p) == "(if(t >= 2.5, if(t < 4, 0.5, 0), 0))"

    def test_start_equals_end_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.interval(start=5, end=5, value=0.3)

    def test_start_greater_than_end_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.interval(start=10, end=5, value=0.3)

    def test_formula_is_parenthesized(self):
        formula = str(AxisPattern.interval(start=0, end=1, value=1.0))
        assert formula.startswith("(") and formula.endswith(")")


class TestSeasonal:
    def test_formula_contains_sin(self):
        formula = str(AxisPattern.seasonal(amplitude=1.0, period=4))
        assert "sin" in formula

    def test_period_zero_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.seasonal(amplitude=1.0, period=0)

    def test_string_period_zero_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.seasonal(amplitude=1.0, period="0")

    def test_string_period_symbolic_is_accepted(self):
        p = AxisPattern.seasonal(amplitude=1.0, period="period_param")
        assert "period_param" in str(p)

    def test_formula_contains_all_params(self):
        formula = str(
            AxisPattern.seasonal(amplitude=2.0, period=12.0, phase=3.0, baseline=1.0)
        )
        assert "2.0" in formula
        assert "12.0" in formula
        assert "3.0" in formula
        assert "1.0" in formula


class TestGaussianPulse:
    def test_formula_contains_exp(self):
        formula = str(AxisPattern.gaussian_pulse(center=10.0, width=2.0, peak=1.0))
        assert "exp" in formula

    def test_width_zero_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.gaussian_pulse(center=0.0, width=0.0, peak=1.0)

    def test_negative_width_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.gaussian_pulse(center=0.0, width=-1.0, peak=1.0)

    def test_formula_is_parenthesized(self):
        formula = str(AxisPattern.gaussian_pulse(center=5.0, width=1.0, peak=1.0))
        assert formula.startswith("(") and formula.endswith(")")


class TestLinearRamp:
    def test_formula_contains_start_and_end(self):
        p = AxisPattern.linear_ramp(start=0, end=10, start_value=0.0, end_value=1.0)
        assert str(p) == "(if(t >= 0, if(t < 10, 0.0 + 1.0 * (t - 0) / 10, 0), 0))"

    def test_start_equals_end_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.linear_ramp(start=5, end=5, start_value=0.0, end_value=1.0)

    def test_start_greater_than_end_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.linear_ramp(start=10, end=5, start_value=0.0, end_value=1.0)


class TestCombine:
    def test_sum_contains_plus(self):
        a = AxisPattern.pulse(at=1, value=0.1)
        b = AxisPattern.pulse(at=2, value=0.2)
        result = str(AxisPattern.combine(a, b, mode=AxisPattern.SUM))
        assert "+" in result

    def test_max_contains_max(self):
        a = AxisPattern.pulse(at=1, value=0.1)
        b = AxisPattern.pulse(at=2, value=0.2)
        result = str(AxisPattern.combine(a, b, mode=AxisPattern.MAX))
        assert "max" in result

    def test_min_contains_min(self):
        a = AxisPattern.pulse(at=1, value=0.1)
        b = AxisPattern.pulse(at=2, value=0.2)
        result = str(AxisPattern.combine(a, b, mode=AxisPattern.MIN))
        assert "min" in result

    def test_single_pattern_is_identity(self):
        """combine(p) returns p unchanged."""
        a = AxisPattern.pulse(at=1, value=0.1)
        assert AxisPattern.combine(a) is a

    def test_zero_patterns_raises(self):
        with pytest.raises(ValueError):
            AxisPattern.combine()

    def test_result_is_parenthesized(self):
        a = AxisPattern.pulse(at=1, value=0.1)
        b = AxisPattern.pulse(at=2, value=0.2)
        result = str(AxisPattern.combine(a, b))
        assert result.startswith("(") and result.endswith(")")

    def test_three_patterns_sum(self):
        a = AxisPattern.pulse(at=1, value=0.1)
        b = AxisPattern.pulse(at=2, value=0.2)
        c = AxisPattern.pulse(at=3, value=0.3)
        result = str(AxisPattern.combine(a, b, c, mode=AxisPattern.SUM))
        assert result.count("+") >= 2

    def test_combine_propagates_conditions(self):
        """combine of grouped patterns preserves the shared group binding."""
        conds = [condition("group", "cat1")]
        a = AxisPattern.pulse(at=1, value=0.1).for_group(conds)
        b = AxisPattern.pulse(at=2, value=0.2).for_group(conds)
        result = AxisPattern.combine(a, b)
        assert result.conditions == conds

    def test_combine_propagates_source_compartment(self):
        conds = [condition("group", "cat1")]
        a = AxisPattern.pulse(at=1, value=0.1).for_group(
            conds, source_compartment="A_cat1"
        )
        b = AxisPattern.pulse(at=2, value=0.2).for_group(
            conds, source_compartment="A_cat1"
        )
        result = AxisPattern.combine(a, b)
        assert result.source_compartment == "A_cat1"

    def test_combine_rejects_inconsistent_conditions(self):
        a = AxisPattern.pulse(at=1, value=0.1).for_group([condition("group", "cat1")])
        b = AxisPattern.pulse(at=2, value=0.2).for_group([condition("group", "cat2")])
        with pytest.raises(ValueError, match="conditions"):
            AxisPattern.combine(a, b)

    def test_combine_rejects_inconsistent_source_compartment(self):
        conds = [condition("group", "cat1")]
        a = AxisPattern.pulse(at=1, value=0.1).for_group(
            conds, source_compartment="A_cat1"
        )
        b = AxisPattern.pulse(at=2, value=0.2).for_group(
            conds, source_compartment="B_cat1"
        )
        with pytest.raises(ValueError, match="source_compartment"):
            AxisPattern.combine(a, b)

    def test_combine_security_cap_raises_clear_error(self):
        """Repeatedly combining hits the security-length cap with a clear error."""
        # Each periodic formula is ~47 chars; need enough to exceed the 10000-char cap.
        periodics = [AxisPattern.periodic(period=7, value=0.1) for _ in range(220)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            AxisPattern.combine(*periodics)


class TestFromFormula:
    def test_passthrough(self):
        p = AxisPattern.from_formula("beta * t")
        assert "beta * t" in str(p)

    def test_wraps_in_parens(self):
        """from_formula always wraps in parens for safe composition."""
        p = AxisPattern.from_formula("beta * t")
        formula = str(p)
        assert formula.startswith("(") and formula.endswith(")")

    def test_outer_multiplication_is_safe(self):
        """f-string multiplication binds to the whole expr, not the last term."""
        p = AxisPattern.from_formula("a + b")
        composed = f"2 * {p}"
        # The result is "2 * (a + b)" rather than "2 * a + b".
        assert composed == "2 * (a + b)"

    def test_already_parenthesized_expression(self):
        p = AxisPattern.from_formula("(x + y)")
        assert "(x + y)" in str(p)

    def test_empty_raises(self):
        with pytest.raises((ValueError, ValidationError)):
            AxisPattern.from_formula("")

    def test_too_long_raises(self):
        config = SecurityConfig()
        with pytest.raises((ValueError, ValidationError)):
            AxisPattern.from_formula("x" * (config.max_expression_length + 1))

    def test_dangerous_pattern_rejected(self):
        """Full security validation runs at construction time."""
        with pytest.raises(SecurityError):
            AxisPattern.from_formula("__import__")


class TestForGroup:
    def test_conditions_are_attached(self):
        p = AxisPattern.pulse(at=5, value=0.1)
        grouped = p.for_group([{"stratification": "age", "category": "elderly"}])
        assert grouped.conditions == [{"stratification": "age", "category": "elderly"}]

    def test_original_is_unchanged(self):
        p = AxisPattern.pulse(at=5, value=0.1)
        p.for_group([{"stratification": "age", "category": "elderly"}])
        assert p.conditions is None

    def test_source_compartment_is_attached(self):
        p = AxisPattern.pulse(at=5, value=0.1)
        grouped = p.for_group(
            [{"stratification": "age", "category": "elderly"}],
            source_compartment="S_elderly",
        )
        assert grouped.source_compartment == "S_elderly"


class TestToStratifiedRate:
    def test_without_source_compartment(self):
        p = AxisPattern.pulse(at=5, value=0.1).for_group([condition("age", "elderly")])
        sr = p.to_stratified_rate()
        assert sr["conditions"] == [condition("age", "elderly")]
        assert "S_" not in rate_string(sr)

    def test_with_source_compartment(self):
        p = AxisPattern.pulse(at=5, value=0.1).for_group(
            [condition("age", "elderly")],
            source_compartment="S_elderly",
        )
        sr = p.to_stratified_rate()
        assert "S_elderly" in rate_string(sr)

    def test_no_conditions_gives_empty_list(self):
        p = AxisPattern.pulse(at=5, value=0.1)
        sr = p.to_stratified_rate()
        assert sr["conditions"] == []


class TestCompositionSafety:
    def test_outer_multiplication_respects_parens(self):
        formula = str(AxisPattern.pulse(at=5, value=1.0))
        outer = f"2 * {formula}"
        # The whole pulse expression is multiplied, not just the last token
        assert outer.startswith("2 * (")

    def test_nested_combine_is_parenthesized(self):
        a = AxisPattern.pulse(at=1, value=0.1)
        b = AxisPattern.pulse(at=2, value=0.2)
        c = AxisPattern.pulse(at=3, value=0.3)
        inner = AxisPattern.combine(a, b)
        result = str(AxisPattern.combine(inner, c))
        assert result.startswith("(") and result.endswith(")")


class TestSecurityValidation:
    """Pattern-generated formulas must survive add_transition's security check."""

    @pytest.mark.parametrize(
        "pattern",
        [
            AxisPattern.pulse(at=5, value=0.1),
            AxisPattern.pulses(at=[1, 3, 5], value=0.1),
            AxisPattern.periodic(period=7, value=0.1),
            AxisPattern.periodic(period=7, value=0.1, offset=2),
            AxisPattern.interval(start=2, end=8, value=0.1),
            AxisPattern.seasonal(amplitude=0.05, period=10, baseline=0.01),
            AxisPattern.gaussian_pulse(center=5.0, width=2.0, peak=0.1),
            AxisPattern.linear_ramp(start=0, end=10, start_value=0.0, end_value=0.1),
            AxisPattern.combine(
                AxisPattern.pulse(at=1, value=0.1),
                AxisPattern.pulse(at=2, value=0.2),
                mode=AxisPattern.MAX,
            ),
        ],
    )
    def test_pattern_survives_security_validator(self, pattern: AxisPattern):
        """Each pattern's formula passes the full security validator."""
        validate_expression_security(str(pattern))
