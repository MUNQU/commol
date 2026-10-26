"""``$compartment`` expands one transition into one per source compartment."""

import pytest

from commol import (
    Aggregate,
    ModelBuilder,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)
from tests.support import (
    ENGINE_REL,
    FactoryBuilder,
    abc_builder,
    assert_close,
    build,
)


def _with_outflow(builder: ModelBuilder, sources: list[str]) -> ModelBuilder:
    return builder.add_parameter(id="d", value=0.01).add_transition(
        id="out", source=sources, target=[], rate="d * $compartment"
    )


class TestExpansion:
    def test_an_empty_target_removes_from_each_source(self) -> None:
        model = build(
            _with_outflow(
                abc_builder(ab=None, bc=None, parameters={}),
                ["A", "B", "C"],
            )
        )

        assert [
            (t.id, t.source, t.target, t.rate) for t in model.process.transitions
        ] == [
            ("out__A", ["A"], [], "d * A"),
            ("out__B", ["B"], [], "d * B"),
            ("out__C", ["C"], [], "d * C"),
        ]

    def test_a_single_target_receives_from_each_source(self) -> None:
        builder = (
            FactoryBuilder(name="Merge")
            .add_bin(id="B1", name="B1")
            .add_bin(id="B2", name="B2")
            .add_bin(id="C", name="C")
            .add_parameter(id="k", value=0.2)
            .add_transition(
                id="bc",
                source=["B1", "B2"],
                target=["C"],
                rate="k * $compartment",
            )
            .set_structure(Aggregate(bin_amounts={"B1": 600, "B2": 300, "C": 100}))
        )

        model = build(builder)

        assert [
            (t.id, t.source, t.target, t.rate) for t in model.process.transitions
        ] == [
            ("bc__B1", ["B1"], ["C"], "k * B1"),
            ("bc__B2", ["B2"], ["C"], "k * B2"),
        ]

    def test_every_occurrence_in_a_formula_is_replaced(self) -> None:
        model = build(
            abc_builder(
                ab=None, bc=None, parameters={"d": 0.01, "K": 1000}
            ).add_transition(
                id="out",
                source=["A", "B"],
                target=[],
                rate="d * $compartment * (1 + 0.1 * $compartment / K)",
            )
        )

        assert [t.rate for t in model.process.transitions] == [
            "d * A * (1 + 0.1 * A / K)",
            "d * B * (1 + 0.1 * B / K)",
        ]

    def test_stratified_rates_are_expanded_too(self) -> None:
        model = build(
            abc_builder(
                ab=None,
                bc=None,
                parameters={"d_g1": 0.005, "d_g2": 0.02, "d_base": 0.01},
                stratifications={"group": {"g1": 0.6, "g2": 0.4}},
            ).add_transition(
                id="out",
                source=["A", "B"],
                target=[],
                rate="d_base * $compartment",
                stratified_rates=[
                    StratifiedRate(
                        conditions=[
                            StratificationCondition(
                                stratification="group", category="g1"
                            )
                        ],
                        rate="d_g1 * $compartment",
                    ),
                    StratifiedRate(
                        conditions=[
                            StratificationCondition(
                                stratification="group", category="g2"
                            )
                        ],
                        rate="d_g2 * $compartment",
                    ),
                ],
            )
        )

        transitions = model.process.transitions
        assert [t.rate for t in transitions] == ["d_base * A", "d_base * B"]
        assert [
            [rate.rate for rate in t.stratified_rates or []] for t in transitions
        ] == [["d_g1 * A", "d_g2 * A"], ["d_g1 * B", "d_g2 * B"]]


class TestExpansionOverTargets:
    def test_a_source_feeds_each_target_and_keeps_its_schedule(self) -> None:
        builder = FactoryBuilder(name="Split")
        for bin_id in ("A", "B", "C"):
            builder.add_bin(id=bin_id, name=bin_id)

        builder.add_transition(
            id="split",
            source=["A"],
            target=["B", "C"],
            rate="k * $compartment",
            stratified_rates=[
                StratifiedRate(
                    conditions=[
                        StratificationCondition(stratification="group", category="g1")
                    ],
                    rate=0.5,
                )
            ],
            impulse_at=[2.0],
        )

        transitions = builder._transitions
        assert [
            (t.id, t.source, t.target, t.rate, t.impulse_at) for t in transitions
        ] == [
            ("split__B", ["A"], ["B"], "k * B", [2.0]),
            ("split__C", ["A"], ["C"], "k * C", [2.0]),
        ]
        assert [r.rate for t in transitions for r in t.stratified_rates or []] == [
            "0.5",
            "0.5",
        ]

    def test_an_expansion_over_sources_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        builder = FactoryBuilder(name="Merge")
        for bin_id in ("A", "B", "C"):
            builder.add_bin(id=bin_id, name=bin_id)

        caplog.clear()
        with caplog.at_level("INFO", logger="commol"):
            builder.add_transition(
                id="merge", source=["A", "B"], target=["C"], rate="k * $compartment"
            )

        assert caplog.messages == [
            "Expanding transition 'merge' with $compartment placeholder: 2 source "
            "compartments will create 2 separate transitions",
            "  Created expanded transition: id='merge__A', source=['A'], "
            "target=['C'], rate='k * A'",
            "  Created expanded transition: id='merge__B', source=['B'], "
            "target=['C'], rate='k * B'",
            "Successfully expanded transition 'merge' into 2 transitions",
        ]

    def test_each_expanded_transition_keeps_a_periodic_impulse(self) -> None:
        builder = FactoryBuilder(name="Split")
        for bin_id in ("A", "B", "C"):
            builder.add_bin(id=bin_id, name=bin_id)

        builder.add_transition(
            id="split",
            source=["A"],
            target=["B", "C"],
            rate="0.1 * $compartment",
            impulse_every=(7.0, 1.0),
        )

        assert [t.impulse_every for t in builder._transitions] == [(7.0, 1.0)] * 2

    def test_an_expansion_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        builder = FactoryBuilder(name="Split")
        for bin_id in ("A", "B", "C"):
            builder.add_bin(id=bin_id, name=bin_id)

        caplog.clear()
        with caplog.at_level("INFO", logger="commol"):
            builder.add_transition(
                id="split",
                source=["A"],
                target=["B", "C"],
                rate="k * $compartment",
                stratified_rates=[
                    StratifiedRate(
                        conditions=[
                            StratificationCondition(
                                stratification="group", category="g1"
                            )
                        ],
                        rate=0.5,
                    )
                ],
            )

        assert caplog.messages[-1] == (
            "Successfully expanded transition 'split' into 2 transitions"
        )
        assert caplog.messages[:2] == [
            "Expanding transition 'split' with $compartment placeholder: 2 target "
            "compartments will create 2 separate transitions",
            "  Created expanded transition: id='split__B', source=['B'], "
            "target=['B', 'C'], rate='k * B', 1 stratified rates",
        ]


class TestErrors:
    def test_a_single_source_is_rejected(self) -> None:
        builder = FactoryBuilder(name="Single").add_bin(id="A", name="A")

        with pytest.raises(ValueError) as error:
            builder.add_transition(
                id="out", source=["A"], target=[], rate="d * $compartment"
            )

        assert "$compartment placeholder requires either multiple source" in str(
            error.value
        )
        assert "use the compartment name directly" in str(error.value)

    def test_several_sources_and_several_targets_are_rejected(self) -> None:
        builder = FactoryBuilder(name="Ambiguous")
        for bin_id in ("A", "B", "R1", "R2"):
            builder.add_bin(id=bin_id, name=bin_id)

        with pytest.raises(ValueError) as error:
            builder.add_transition(
                id="bc",
                source=["A", "B"],
                target=["R1", "R2"],
                rate="k * $compartment",
            )

        assert "cannot be used with both multiple sources and multiple targets" in str(
            error.value
        )
        assert "ambiguous mappings" in str(error.value)


class TestRuns:
    def test_an_outflow_rate_on_every_compartment_shrinks_the_total_geometrically(
        self,
    ) -> None:
        model = build(
            _with_outflow(
                abc_builder(
                    parameters={"k1": 0.3, "k2": 0.2},
                    amounts={"A": 990, "B": 10, "C": 0},
                ),
                ["A", "B", "C"],
            )
        )

        results = Simulation(model).run(200)

        # ab and bc move amounts between bins; each step the outflow
        # rate removes 1 % of every bin, so total(t) = 1000 * 0.99**t.
        totals = [
            sum(results[bin_id][step] for bin_id in ("A", "B", "C"))
            for step in range(201)
        ]
        assert_close(
            totals, [1000.0 * 0.99**step for step in range(201)], rel=ENGINE_REL
        )

    def test_the_expansion_runs_like_transitions_written_by_hand(self) -> None:
        automatic = _with_outflow(abc_builder(), ["A", "B", "C"])
        by_hand = abc_builder().add_parameter(id="d", value=0.01)
        for bin_id in ("A", "B", "C"):
            by_hand.add_transition(
                id=f"out_{bin_id}", source=[bin_id], target=[], rate=f"d * {bin_id}"
            )

        assert_close(
            Simulation(build(automatic)).run(100),
            Simulation(build(by_hand)).run(100),
            rel=ENGINE_REL,
        )


def _two_bins_in_two_groups(parameters: dict[str, float]) -> FactoryBuilder:
    """A and B hold 100 each, split 90 and 10 by ``g``; C starts empty."""
    return abc_builder(
        ab=None,
        bc=None,
        parameters=parameters,
        amounts={"A": 100, "B": 100, "C": 0},
        stratifications={"g": {"g1": 0.9, "g2": 0.1}},
    )


def test_a_placeholder_with_per_compartment_reads_the_compartment_that_flows() -> None:
    model = build(
        _two_bins_in_two_groups({"k1": 0.1, "k2": 0.5}).add_transition(
            id="out",
            source=["A", "B"],
            target=[],
            rate="k1 * $compartment",
            per_compartment=True,
            stratified_rates=[
                StratifiedRate(
                    conditions=[
                        StratificationCondition(stratification="g", category="g2")
                    ],
                    rate="k2 * $compartment",
                )
            ],
        )
    )

    results = Simulation(model).run(1)

    # Each compartment loses its own rate times its own amount.
    assert_close(
        {name: results[name] for name in ("A_g1", "A_g2", "B_g1", "B_g2")},
        {
            "A_g1": [90.0, 90.0 * 0.9],
            "A_g2": [10.0, 10.0 * 0.5],
            "B_g1": [90.0, 90.0 * 0.9],
            "B_g2": [10.0, 10.0 * 0.5],
        },
        rel=ENGINE_REL,
    )


def test_a_placeholder_keeps_the_accumulators_of_its_transition() -> None:
    model = build(
        abc_builder(
            ab=None,
            bc=None,
            parameters={"k1": 0.1},
            amounts={"A": 100, "B": 100, "C": 0},
            accumulator="out",
        ).add_transition(
            id="t",
            source=["A", "B"],
            target=["C"],
            rate="k1 * $compartment",
            accumulators=["out"],
        )
    )

    results = Simulation(model).run(3)

    # Only t fills C, so the accumulator of t counts every amount C gains.
    assert_close(results["out"], [0.0, 20.0, 38.0, 54.2], rel=ENGINE_REL)
    assert_close(results["out"], results["C"], rel=ENGINE_REL)


def test_a_placeholder_keeps_the_absolute_flag_of_a_stratified_rate() -> None:
    model = build(
        _two_bins_in_two_groups({"k1": 0.1, "k2": 2.0}).add_transition(
            id="out",
            source=["A", "B"],
            target=[],
            rate="k1 * $compartment",
            stratified_rates=[
                StratifiedRate(
                    conditions=[
                        StratificationCondition(stratification="g", category="g2")
                    ],
                    rate="k2",
                    absolute=True,
                )
            ],
        )
    )

    results = Simulation(model).run(1)

    # g2 loses the amount k2; g1 loses k1 times the total of its bin.
    assert_close(results["A_g2"], [10.0, 8.0], rel=ENGINE_REL)
    assert_close(results["A_g1"], [90.0, 90.0 - 0.1 * 100.0], rel=ENGINE_REL)
