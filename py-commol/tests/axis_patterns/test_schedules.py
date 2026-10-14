"""Schedules give each stratification category its own axis pattern."""

from typing import cast

import pytest

from commol import ModelBuilder, AxisPattern
from commol.api.patterns import ConditionDict
from tests.support import a_to_b_builder, run
from tests.axis_patterns.conditions import condition, rate_string, stratified_rates


def _make_schedule_ab(
    rate: AxisPattern,
    *,
    categories: list[str] | None = None,
    steps: int = 20,
    total: float = 1000,
) -> dict[str, list[float]]:
    """Run A → B split by ``group``, with a AxisPattern rate."""
    categories = categories or ["cat1", "cat2"]
    builder = a_to_b_builder(
        rate=rate,
        parameters={},
        total=total,
        stratifications={
            "group": {category: 1.0 / len(categories) for category in categories}
        },
    )
    return run(builder, steps)


class TestScheduleUnit:
    def test_add_group_classlevel_creates_schedule(self):
        """AxisPattern.add_group as classmethod returns a AxisPattern instance."""
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.pulse(at=5, value=0.1),
        )
        assert isinstance(rate, AxisPattern)
        # Builder hook returns one stratified rate, no default.
        assert rate._builder_stratified_rates() == [
            {
                "conditions": [condition("group", "cat1")],
                "rate": str(AxisPattern.pulse(at=5, value=0.1)),
            }
        ]
        assert rate._builder_rate() is None

    def test_add_group_chained_appends(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.pulse(at=5, value=0.1),
        ).add_group(
            conditions=[condition("group", "cat2")],
            schedule=AxisPattern.pulse(at=10, value=0.2),
        )
        srs = stratified_rates(rate)
        assert len(srs) == 2
        assert srs[0]["conditions"] == [condition("group", "cat1")]
        assert srs[1]["conditions"] == [condition("group", "cat2")]

    def test_set_default_attaches_fallback(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.pulse(at=5, value=0.1),
        ).set_default(AxisPattern.from_formula("0.01"))
        assert rate._builder_rate() is not None
        builder_rate = rate._builder_rate()
        assert isinstance(builder_rate, str)
        assert "0.01" in builder_rate

    def test_add_group_on_single_pattern_raises(self):
        """Single AxisPatterns reject instance-level add_group (use the class)."""
        p = AxisPattern.pulse(at=5, value=0.1)
        with pytest.raises(TypeError, match="class-level"):
            p.add_group(conditions=[condition("group", "cat1")], schedule=p)

    def test_set_default_on_single_pattern_raises(self):
        p = AxisPattern.pulse(at=5, value=0.1)
        with pytest.raises(TypeError, match="schedule"):
            p.set_default(AxisPattern.from_formula("0.01"))

    def test_add_group_empty_conditions_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            AxisPattern.add_group(
                conditions=[], schedule=AxisPattern.from_formula("0.1")
            )

    def test_add_group_duplicate_raises(self):
        with pytest.raises(ValueError, match="already been registered"):
            (
                AxisPattern.add_group(
                    conditions=[condition("group", "cat1")],
                    schedule=AxisPattern.from_formula("0.1"),
                ).add_group(
                    conditions=[condition("group", "cat1")],
                    schedule=AxisPattern.from_formula("0.2"),
                )
            )

    def test_set_default_twice_raises(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.from_formula("0.1"),
        ).set_default(AxisPattern.from_formula("0.05"))
        with pytest.raises(ValueError, match="already been registered"):
            rate.set_default(AxisPattern.from_formula("0.02"))

    def test_set_default_rejects_conditions(self):
        with pytest.raises(ValueError, match="conditions"):
            AxisPattern.add_group(
                conditions=[condition("group", "cat1")],
                schedule=AxisPattern.from_formula("0.1"),
            ).set_default(
                AxisPattern.from_formula("0.05").for_group([condition("group", "cat2")])
            )

    def test_set_default_rejects_source_compartment(self):
        default_with_comp = AxisPattern.from_formula("0.1").model_copy(
            update={"source_compartment": "A_cat1"}
        )
        with pytest.raises(ValueError, match="source_compartment"):
            AxisPattern.add_group(
                conditions=[condition("group", "cat1")],
                schedule=AxisPattern.from_formula("0.1"),
            ).set_default(default_with_comp)

    def test_malformed_condition_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            bad_condition: dict[str, str] = {"strat": "g", "cat": "c1"}
            AxisPattern.add_group(
                conditions=cast(list[ConditionDict], [bad_condition]),
                schedule=AxisPattern.from_formula("0.1"),
            )

    def test_schedule_has_no_single_formula(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.from_formula("0.1"),
        )
        with pytest.raises(TypeError, match="no single formula"):
            _ = rate.formula

    def test_chaining_returns_same_object(self):
        rate1 = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.from_formula("0.1"),
        )
        rate2 = rate1.add_group(
            conditions=[condition("group", "cat2")],
            schedule=AxisPattern.from_formula("0.2"),
        )
        assert rate1 is rate2

    def test_source_compartment_passes_through(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.from_formula("0.1"),
            source_compartment="A_cat1",
        )
        srs = stratified_rates(rate)
        assert "A_cat1" in rate_string(srs[0])

    def test_default_absolute_is_inferred(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.from_formula("0.001 * total"),
        )
        srs = stratified_rates(rate)
        assert "absolute" not in srs[0]

    def test_explicit_absolute_false_is_preserved(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.from_formula("0.001 * total"),
            absolute=False,
        )
        srs = stratified_rates(rate)
        assert srs[0]["absolute"] is False

    def test_absolute_true_is_preserved(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.pulse(at=5, value=1.0),
            absolute=True,
        )
        srs = stratified_rates(rate)
        assert srs[0]["absolute"] is True


class TestScheduleIntegration:
    """The simulation engine consumes a AxisPattern via add_transition(rate=...)."""

    def test_conditional_exclusion_cat2_unchanged(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.periodic(period=7, value=0.1),
        )
        result = _make_schedule_ab(rate, steps=30)
        assert all(v == pytest.approx(0.0, abs=1e-8) for v in result["B_cat2"])

    def test_differential_schedules(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.periodic(period=7, value=0.1),
        ).add_group(
            conditions=[condition("group", "cat2")],
            schedule=AxisPattern.periodic(period=30, value=0.1),
        )
        result = _make_schedule_ab(rate, steps=61)
        b1_events = sum(
            1 for t in range(61) if result["B_cat1"][t + 1] - result["B_cat1"][t] > 1e-6
        )
        b2_events = sum(
            1 for t in range(61) if result["B_cat2"][t + 1] - result["B_cat2"][t] > 1e-6
        )
        # Period 7 fires at 0, 7, ..., 56; period 30 at 0, 30 and 60.
        assert (b1_events, b2_events) == (9, 3)

    def test_default_covers_unmatched_groups(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.pulse(at=5, value=0.5),
        ).set_default(AxisPattern.pulse(at=5, value=0.2))
        result = _make_schedule_ab(rate, steps=10)
        # Both cat1 and cat2 fire at step 5 — flow appears between index 5 and 6.
        # Each group holds 500; cat1 moves half at step 5, cat2 the default 0.2.
        assert result["B_cat1"][6] - result["B_cat1"][5] == pytest.approx(250.0)
        assert result["B_cat2"][6] - result["B_cat2"][5] == pytest.approx(100.0)

    def test_cross_category_routing(self):
        """`to:` condition routes flow between categories of the same bin."""
        rate = AxisPattern.add_group(
            conditions=[condition("status", "s0", to="s1")],
            schedule=AxisPattern.pulse(at=5, value=0.5),
        )

        model = (
            ModelBuilder("A_status")
            .add_bin("A", "Compartment")
            .add_stratification("status", ["s0", "s1"])
            .add_transition("flow", ["A"], ["A"], rate=rate)
            .set_initial_conditions(
                total=1000,
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
        )
        result = run(model, 10)
        a_s0_drop = result["A_s0"][5] - result["A_s0"][6]
        a_s1_gain = result["A_s1"][6] - result["A_s1"][5]
        assert a_s0_drop == pytest.approx(500.0)
        assert a_s1_gain == pytest.approx(a_s0_drop, rel=1e-4)
        for t in range(11):
            total = result["A_s0"][t] + result["A_s1"][t]
            assert total == pytest.approx(1000.0, abs=1e-4)

    def test_target_only_override_without_category_routes_flow(self):
        """A target-only condition can match without filtering source categories."""
        model = (
            ModelBuilder("target_only")
            .add_bin("A", "Source")
            .add_bin("B", "Target")
            .add_stratification("group", ["g1", "g2"])
            .add_transition(
                "route",
                ["A"],
                ["B"],
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "group", "to": "g2"}],
                        "rate": 1.0,
                    }
                ],
            )
            .set_initial_conditions(
                total=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": 1.0},
                    {"bin": "B", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "group",
                        "fractions": [
                            {"category": "g1", "fraction": 0.5},
                            {"category": "g2", "fraction": 0.5},
                        ],
                    }
                ],
            )
        )

        result = run(model, 1)

        assert result["A_g1"][1] == pytest.approx(0.0)
        assert result["A_g2"][1] == pytest.approx(0.0)
        assert result["B_g1"][1] == pytest.approx(0.0)
        assert result["B_g2"][1] == pytest.approx(1000.0)

    def test_the_total_is_kept(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.periodic(period=7, value=0.1),
        ).add_group(
            conditions=[condition("group", "cat2")],
            schedule=AxisPattern.periodic(period=7, value=0.1),
        )
        result = _make_schedule_ab(rate, steps=30, total=1000)
        for t in range(31):
            total = (
                result["A_cat1"][t]
                + result["A_cat2"][t]
                + result["B_cat1"][t]
                + result["B_cat2"][t]
            )
            assert total == pytest.approx(1000.0, abs=1e-4)

    def test_mixed_absolute_and_relative_groups(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.pulse(at=5, value=10.0),
            absolute=True,
        ).add_group(
            conditions=[condition("group", "cat2")],
            schedule=AxisPattern.pulse(at=5, value=0.1),
            absolute=False,
        )

        result = _make_schedule_ab(rate, steps=10, total=1000)
        cat1_flow = result["A_cat1"][5] - result["A_cat1"][6]
        cat2_flow = result["A_cat2"][5] - result["A_cat2"][6]

        assert cat1_flow == pytest.approx(10.0, rel=1e-6)
        assert cat2_flow == pytest.approx(50.0, rel=1e-6)
