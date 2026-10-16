"""``$compartment`` expands one transition into one per source compartment."""

import pytest

from commol import Aggregate, ModelBuilder, Simulation
from tests.support import (
    FactoryBuilder,
    ENGINE_REL,
    assert_close,
    build,
    abc_builder,
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
            .set_structure(
                Aggregate(total=1000, bin_fractions={"B1": 0.6, "B2": 0.3, "C": 0.1})
            )
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
                    {
                        "conditions": [{"stratification": "group", "category": "g1"}],
                        "rate": "d_g1 * $compartment",
                    },
                    {
                        "conditions": [{"stratification": "group", "category": "g2"}],
                        "rate": "d_g2 * $compartment",
                    },
                ],
            )
        )

        transitions = model.process.transitions
        assert [t.rate for t in transitions] == ["d_base * A", "d_base * B"]
        assert [
            [rate.rate for rate in t.stratified_rates or []] for t in transitions
        ] == [["d_g1 * A", "d_g2 * A"], ["d_g1 * B", "d_g2 * B"]]


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
                    fractions={"A": 0.99, "B": 0.01, "C": 0.0},
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
