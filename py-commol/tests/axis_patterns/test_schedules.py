"""Schedules give each stratification category its own axis pattern."""

import pytest

from commol import Aggregate, AxisPattern, StratificationCondition, StratifiedRate
from tests.axis_patterns.conditions import condition, rate_string, stratified_rates
from tests.support import FactoryBuilder, a_to_b_builder, run


def _make_schedule_ab(
    rate: AxisPattern,
    *,
    categories: list[str] | None = None,
    steps: int = 20,
    amount: float = 1000,
) -> dict[str, list[float]]:
    """Run A → B split by ``group``, from `amount` in A, with a AxisPattern rate."""
    categories = categories or ["cat1", "cat2"]
    builder = a_to_b_builder(
        rate=rate,
        parameters={},
        amounts={"A": amount, "B": 0},
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
            StratifiedRate(
                conditions=[condition("group", "cat1")],
                rate=str(AxisPattern.pulse(at=5, value=0.1)),
            )
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
        assert srs[0].conditions == [condition("group", "cat1")]
        assert srs[1].conditions == [condition("group", "cat2")]

    def test_a_default_is_the_fallback_of_the_formula(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.pulse(at=5, value=0.1),
        ).set_default(AxisPattern.from_formula("0.01"))
        assert rate._builder_rate() == "(0.01)"

    @pytest.mark.parametrize(
        ("misuse", "error", "message"),
        [
            pytest.param(
                lambda: AxisPattern.pulse(at=5, value=0.1).add_group(
                    conditions=[condition("group", "cat1")], schedule=_rate("0.1")
                ),
                TypeError,
                "add_group on a single AxisPattern is only valid as a class-level "
                "factory (AxisPattern.add_group(...)); call it on the class to "
                "start a new schedule.",
                id="add_group-on-a-pattern",
            ),
            pytest.param(
                lambda: AxisPattern.pulse(at=5, value=0.1).set_default(_rate("0.01")),
                TypeError,
                "set_default is only valid on a schedule AxisPattern; start one "
                "with AxisPattern.add_group(...) first.",
                id="set_default-on-a-pattern",
            ),
            pytest.param(
                lambda: AxisPattern.add_group(conditions=[], schedule=_rate("0.1")),
                ValueError,
                "'conditions' must be non-empty. Use set_default() for the "
                "condition-free fallback.",
                id="no-conditions",
            ),
            pytest.param(
                lambda: _cat1().add_group(
                    conditions=[condition("group", "cat1")], schedule=_rate("0.2")
                ),
                ValueError,
                "A group with conditions [StratificationCondition(stratification="
                "'group', category='cat1', categories=None, to=None)] has already "
                "been registered.",
                id="same-group-twice",
            ),
            pytest.param(
                lambda: _cat1().set_default(_rate("0.05")).set_default(_rate("0.02")),
                ValueError,
                "A default pattern has already been registered.",
                id="two-defaults",
            ),
            pytest.param(
                lambda: _cat1().set_default(
                    _rate("0.05").for_group([condition("group", "cat2")])
                ),
                ValueError,
                "Default pattern must not have conditions. Use add_group() for "
                "sub-group-specific patterns.",
                id="default-with-conditions",
            ),
            pytest.param(
                lambda: _cat1().set_default(
                    _rate("0.1").model_copy(update={"source_compartment": "A_cat1"})
                ),
                ValueError,
                "Default pattern must not have source_compartment set. The default "
                "rate is applied to every unmatched compartment, so binding it to "
                "one compartment is almost certainly a bug.",
                id="default-with-a-compartment",
            ),
        ],
    )
    def test_a_misuse_of_a_schedule_says_how_to_fix_it(
        self, misuse, error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error) as raised:
            misuse()

        assert str(raised.value) == message

    def test_groups_that_differ_only_in_their_target_are_two_groups(self):
        rate = AxisPattern.add_group(
            conditions=[StratificationCondition(stratification="group", to="cat1")],
            schedule=_rate("0.1"),
        ).add_group(
            conditions=[StratificationCondition(stratification="group", to="cat2")],
            schedule=_rate("0.2"),
        )

        assert len(rate._builder_stratified_rates() or []) == 2

    def test_add_group_keeps_its_compartment_and_its_absolute_flag(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=_rate("0.1"),
            source_compartment="A_cat1",
            absolute=True,
        ).add_group(
            conditions=[condition("group", "cat2")],
            schedule=_rate("0.2"),
            source_compartment="A_cat2",
            absolute=False,
        )

        entries = rate._builder_stratified_rates() or []
        assert [entry.absolute for entry in entries] == [True, False]
        assert [entry.pattern.source_compartment for entry in rate.patterns] == [
            "A_cat1",
            "A_cat2",
        ]

    def test_a_schedule_prints_its_groups_and_its_default(self):
        assert str(_cat1()) == "<AxisPattern schedule with 1 pattern(s), default=unset>"
        assert str(_cat1().set_default(_rate("0.05"))) == (
            "<AxisPattern schedule with 2 pattern(s), default=set>"
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
        assert rate_string(stratified_rates(rate)[0]) == "(0.1) * A_cat1"

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
        assert srs[0].absolute is False

    def test_absolute_true_is_preserved(self):
        rate = AxisPattern.add_group(
            conditions=[condition("group", "cat1")],
            schedule=AxisPattern.pulse(at=5, value=1.0),
            absolute=True,
        )
        srs = stratified_rates(rate)
        assert srs[0].absolute is True


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
            FactoryBuilder("A_status")
            .add_bin("A", "Compartment")
            .add_stratification("status", ["s0", "s1"])
            .add_transition("flow", ["A"], ["A"], rate=rate)
            .set_structure(
                Aggregate(
                    bin_amounts={"A": 1000},
                    stratification_fractions={"status": {"s0": 1.0, "s1": 0.0}},
                )
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
            FactoryBuilder("target_only")
            .add_bin("A", "Source")
            .add_bin("B", "Target")
            .add_stratification("group", ["g1", "g2"])
            .add_transition(
                "route",
                ["A"],
                ["B"],
                stratified_rates=[
                    StratifiedRate(
                        conditions=[
                            StratificationCondition(stratification="group", to="g2")
                        ],
                        rate=1.0,
                    )
                ],
            )
            .set_structure(
                Aggregate(
                    bin_amounts={"A": 1000, "B": 0},
                    stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
                )
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
        result = _make_schedule_ab(rate, steps=30, amount=1000)
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

        result = _make_schedule_ab(rate, steps=10, amount=1000)
        cat1_flow = result["A_cat1"][5] - result["A_cat1"][6]
        cat2_flow = result["A_cat2"][5] - result["A_cat2"][6]

        assert cat1_flow == pytest.approx(10.0, rel=1e-6)
        assert cat2_flow == pytest.approx(50.0, rel=1e-6)


def _rate(formula: str) -> AxisPattern:
    return AxisPattern.from_formula(formula)


def _cat1() -> AxisPattern:
    """A schedule of one group, ``group = cat1``."""
    return AxisPattern.add_group(
        conditions=[condition("group", "cat1")], schedule=_rate("0.1")
    )
