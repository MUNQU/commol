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
from tests.support import assert_accepted


class TestPulse:
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

    @pytest.mark.parametrize(
        "pattern",
        [
            lambda: AxisPattern.pulses(at=[1, 2], value="1, 0"),
            lambda: AxisPattern.pulse(at=1, value="1, 0"),
            lambda: AxisPattern.periodic(period=2, value="1, 0"),
            lambda: AxisPattern.interval(start=1, end=2, value="1, 0"),
        ],
    )
    def test_amount_with_comma_rejected(self, pattern):
        with pytest.raises(ValueError) as error:
            pattern()

        assert str(error.value) == (
            "'value' string must not contain commas (got '1, 0'); commas would "
            "break the surrounding if(...) expression"
        )

    def test_per_step_amounts_formula(self):
        p = AxisPattern.pulses(at=[2, 5, 9], value=[0.1, 0.2, 0.3])
        formula = str(p)
        assert "step == 2, 0.1" in formula
        assert "step == 5, 0.2" in formula
        assert "step == 9, 0.3" in formula

    def test_per_step_amounts_string_values(self):
        p = AxisPattern.pulses(at=[1, 4], value=["alpha", "k1"])
        formula = str(p)
        assert "step == 1, alpha" in formula
        assert "step == 4, k1" in formula

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

    @pytest.mark.parametrize(
        "pattern",
        [
            AxisPattern.pulse(at=1, value="k"),
            AxisPattern.pulses(at=[1, 2], value="k"),
            AxisPattern.pulses(at=[1, 2], value=[0.1, "k"]),
        ],
        ids=["pulse", "pulses", "one-formula-of-several"],
    )
    def test_a_pulse_of_a_formula_stays_a_formula(self, pattern: AxisPattern):
        builder = (
            ModelBuilder("M")
            .add_bin("A", "A")
            .add_bin("B", "B")
            .add_parameter("k", 0.1)
            .add_transition("t", ["A"], ["B"], rate=pattern)
        )

        assert [p.id for p in builder._parameters] == ["k"]
        assert builder._transitions[0].rate == pattern.formula

    def test_duplicate_user_parameter_id_raises_immediately(self):
        with pytest.raises(ValueError, match="already exists"):
            ModelBuilder("M").add_parameter("x", 1.0).add_parameter("x", 2.0)


class TestPeriodic:
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


class TestSinusoid:
    def test_period_zero_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.sinusoid(amplitude=1.0, period=0)

    def test_string_period_zero_raises(self):
        with pytest.raises(ValidationError):
            AxisPattern.sinusoid(amplitude=1.0, period="0")

    def test_string_period_symbolic_is_accepted(self):
        p = AxisPattern.sinusoid(amplitude=1.0, period="period_param")
        assert "period_param" in str(p)

    def test_a_sinusoid_starts_at_phase_zero_around_zero(self):
        formula = str(AxisPattern.sinusoid(amplitude=2.0, period=12.0))
        assert formula == "(0.0 + 2.0 * sin(2 * pi * (t - 0.0) / 12.0))"

    def test_formula_places_every_parameter(self):
        formula = str(
            AxisPattern.sinusoid(amplitude=2.0, period=12.0, phase=3.0, baseline=1.0)
        )
        assert formula == "(1.0 + 2.0 * sin(2 * pi * (t - 3.0) / 12.0))"


class TestGaussianPulse:
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
    @pytest.mark.parametrize(
        ("mode", "formula"),
        [
            (AxisPattern.SUM, "((a) + (b) + (c))"),
            (AxisPattern.MAX, "(max(max((a), (b)), (c)))"),
            (AxisPattern.MIN, "(min(min((a), (b)), (c)))"),
            ("min", "(min(min((a), (b)), (c)))"),
        ],
    )
    def test_a_mode_nests_its_function_over_the_patterns(self, mode, formula):
        a, b, c = (AxisPattern.from_formula(name) for name in "abc")
        assert str(AxisPattern.combine(a, b, c, mode=mode)) == formula

    def test_an_unknown_mode_is_rejected(self):
        a, b = AxisPattern.from_formula("a"), AxisPattern.from_formula("b")
        with pytest.raises(ValueError, match="'mean' is not a valid CombineMode"):
            AxisPattern.combine(a, b, mode="mean")  # ty: ignore[invalid-argument-type]

    def test_a_combined_formula_may_reach_the_length_cap(self):
        a, b = AxisPattern.from_formula("a"), AxisPattern.from_formula("b")
        length = len("((a) + (b))")

        combined = AxisPattern.combine(
            a, b, config=SecurityConfig(max_expression_length=length)
        )

        assert str(combined) == "((a) + (b))"
        with pytest.raises(ValueError) as error:
            AxisPattern.combine(
                a, b, config=SecurityConfig(max_expression_length=length - 1)
            )
        assert str(error.value) == (
            f"Combined formula length {length} exceeds maximum {length - 1} characters"
        )

    def test_a_formula_longer_than_the_default_cap_follows_its_configuration(self):
        expr = " + ".join(f"k{i}" for i in range(3000))
        config = SecurityConfig(max_expression_length=len(expr) + 2)

        pattern = AxisPattern.from_formula(expr, config=config)

        assert str(pattern) == f"({expr})"

    def test_a_formula_is_checked_with_its_configuration(self):
        with pytest.raises(SecurityError, match="6 characters \\(maximum: 4\\)"):
            AxisPattern.from_formula(
                "abcd", config=SecurityConfig(max_expression_length=4)
            )

    def test_single_pattern_is_identity(self):
        """combine(p) returns p unchanged."""
        a = AxisPattern.pulse(at=1, value=0.1)
        assert AxisPattern.combine(a) is a

    def test_zero_patterns_raises(self):
        with pytest.raises(ValueError) as error:
            AxisPattern.combine()

        assert str(error.value) == "combine() requires at least one pattern"

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
        with pytest.raises(ValueError) as error:
            AxisPattern.combine(a, b)

        assert str(error.value) == (
            "combine() requires all input patterns to share the same conditions "
            "(use for_group on the combined result instead)"
        )

    def test_combine_rejects_inconsistent_source_compartment(self):
        conds = [condition("group", "cat1")]
        a = AxisPattern.pulse(at=1, value=0.1).for_group(
            conds, source_compartment="A_cat1"
        )
        b = AxisPattern.pulse(at=2, value=0.2).for_group(
            conds, source_compartment="B_cat1"
        )
        with pytest.raises(ValueError) as error:
            AxisPattern.combine(a, b)

        assert str(error.value) == (
            "combine() requires all input patterns to share the same source_compartment"
        )


class TestFromFormula:
    def test_wraps_in_parens(self):
        """from_formula always wraps in parens for safe composition."""
        assert str(AxisPattern.from_formula("k1 * t")) == "(k1 * t)"

    def test_already_parenthesized_expression(self):
        p = AxisPattern.from_formula("(x + y)")
        assert "(x + y)" in str(p)

    @pytest.mark.parametrize("expr", ["", "   "])
    def test_empty_raises(self, expr: str):
        with pytest.raises(ValueError) as error:
            AxisPattern.from_formula(expr)

        assert str(error.value) == "'expr' must be a non-empty expression"

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
        grouped = p.for_group([{"stratification": "group", "category": "g2"}])
        assert grouped.conditions == [{"stratification": "group", "category": "g2"}]

    def test_original_is_unchanged(self):
        p = AxisPattern.pulse(at=5, value=0.1)
        p.for_group([{"stratification": "group", "category": "g2"}])
        assert p.conditions is None

    def test_source_compartment_is_attached(self):
        p = AxisPattern.pulse(at=5, value=0.1)
        grouped = p.for_group(
            [{"stratification": "group", "category": "g2"}],
            source_compartment="A_g2",
        )
        assert grouped.source_compartment == "A_g2"


class TestToStratifiedRate:
    def test_without_source_compartment(self):
        p = AxisPattern.pulse(at=5, value=0.1).for_group([condition("group", "g2")])
        sr = p.to_stratified_rate()
        assert sr["conditions"] == [condition("group", "g2")]
        assert "A_" not in rate_string(sr)

    def test_with_source_compartment(self):
        p = AxisPattern.pulse(at=5, value=0.1).for_group(
            [condition("group", "g2")],
            source_compartment="A_g2",
        )
        sr = p.to_stratified_rate()
        assert "A_g2" in rate_string(sr)

    def test_no_conditions_gives_empty_list(self):
        p = AxisPattern.pulse(at=5, value=0.1)
        sr = p.to_stratified_rate()
        assert sr["conditions"] == []


class TestCompositionSafety:
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
            AxisPattern.sinusoid(amplitude=0.05, period=10, baseline=0.01),
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
        assert_accepted(validate_expression_security, str(pattern))


class TestAbstract:
    def test_a_pattern_comes_from_a_factory(self):
        with pytest.raises(TypeError, match="abstract"):
            AxisPattern()
