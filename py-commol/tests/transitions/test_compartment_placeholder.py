"""``$compartment`` expands one transition into one per source compartment."""

import pytest

from commol import ModelBuilder, Simulation
from tests.support import (
    ENGINE_REL,
    assert_close,
    build,
    sir_builder,
)


def _with_death(builder: ModelBuilder, sources: list[str]) -> ModelBuilder:
    return builder.add_parameter(id="d", value=0.01).add_transition(
        id="death", source=sources, target=[], rate="d * $compartment"
    )


class TestExpansion:
    def test_an_empty_target_removes_from_each_source(self) -> None:
        model = build(
            _with_death(
                sir_builder(infection=None, recovery=None, parameters={}),
                ["S", "I", "R"],
            )
        )

        assert [
            (t.id, t.source, t.target, t.rate) for t in model.dynamics.transitions
        ] == [
            ("death__S", ["S"], [], "d * S"),
            ("death__I", ["I"], [], "d * I"),
            ("death__R", ["R"], [], "d * R"),
        ]

    def test_a_single_target_receives_from_each_source(self) -> None:
        builder = (
            ModelBuilder(name="Treatment")
            .add_bin(id="I_mild", name="Mild")
            .add_bin(id="I_severe", name="Severe")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="treatment_rate", value=0.2)
            .add_transition(
                id="treatment",
                source=["I_mild", "I_severe"],
                target=["R"],
                rate="treatment_rate * $compartment",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "I_mild", "fraction": 0.6},
                    {"bin": "I_severe", "fraction": 0.3},
                    {"bin": "R", "fraction": 0.1},
                ],
            )
        )

        model = build(builder)

        assert [
            (t.id, t.source, t.target, t.rate) for t in model.dynamics.transitions
        ] == [
            ("treatment__I_mild", ["I_mild"], ["R"], "treatment_rate * I_mild"),
            ("treatment__I_severe", ["I_severe"], ["R"], "treatment_rate * I_severe"),
        ]

    def test_every_occurrence_in_a_formula_is_replaced(self) -> None:
        model = build(
            sir_builder(
                infection=None, recovery=None, parameters={"d": 0.01, "K": 1000}
            ).add_transition(
                id="complex_death",
                source=["S", "I"],
                target=[],
                rate="d * $compartment * (1 + 0.1 * $compartment / K)",
            )
        )

        assert [t.rate for t in model.dynamics.transitions] == [
            "d * S * (1 + 0.1 * S / K)",
            "d * I * (1 + 0.1 * I / K)",
        ]

    def test_stratified_rates_are_expanded_too(self) -> None:
        model = build(
            sir_builder(
                infection=None,
                recovery=None,
                parameters={"d_young": 0.005, "d_old": 0.02, "d_base": 0.01},
                stratifications={"age": {"young": 0.6, "old": 0.4}},
            ).add_transition(
                id="death",
                source=["S", "I"],
                target=[],
                rate="d_base * $compartment",
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "young"}],
                        "rate": "d_young * $compartment",
                    },
                    {
                        "conditions": [{"stratification": "age", "category": "old"}],
                        "rate": "d_old * $compartment",
                    },
                ],
            )
        )

        transitions = model.dynamics.transitions
        assert [t.rate for t in transitions] == ["d_base * S", "d_base * I"]
        assert [
            [rate.rate for rate in t.stratified_rates or []] for t in transitions
        ] == [["d_young * S", "d_old * S"], ["d_young * I", "d_old * I"]]


class TestErrors:
    def test_a_single_source_is_rejected(self) -> None:
        builder = ModelBuilder(name="Single").add_bin(id="S", name="S")

        with pytest.raises(ValueError) as error:
            builder.add_transition(
                id="death", source=["S"], target=[], rate="d * $compartment"
            )

        assert "$compartment placeholder requires either multiple source" in str(
            error.value
        )
        assert "use the compartment name directly" in str(error.value)

    def test_several_sources_and_several_targets_are_rejected(self) -> None:
        builder = ModelBuilder(name="Ambiguous")
        for bin_id in ("S", "I", "R1", "R2"):
            builder.add_bin(id=bin_id, name=bin_id)

        with pytest.raises(ValueError) as error:
            builder.add_transition(
                id="recovery",
                source=["S", "I"],
                target=["R1", "R2"],
                rate="recovery_rate * $compartment",
            )

        assert "cannot be used with both multiple sources and multiple targets" in str(
            error.value
        )
        assert "ambiguous mappings" in str(error.value)


class TestRuns:
    def test_a_death_rate_on_every_compartment_shrinks_the_total_geometrically(
        self,
    ) -> None:
        model = build(
            _with_death(
                sir_builder(
                    parameters={"beta": 0.3, "gamma": 0.2},
                    fractions={"S": 0.99, "I": 0.01, "R": 0.0},
                ),
                ["S", "I", "R"],
            )
        )

        results = Simulation(model).run(200)

        # Infection and recovery move people between bins; each step the
        # death rate removes 1 % of every bin, so N(t) = 1000 * 0.99**t.
        totals = [
            sum(results[bin_id][step] for bin_id in ("S", "I", "R"))
            for step in range(201)
        ]
        assert_close(
            totals, [1000.0 * 0.99**step for step in range(201)], rel=ENGINE_REL
        )

    def test_the_expansion_runs_like_transitions_written_by_hand(self) -> None:
        automatic = _with_death(sir_builder(), ["S", "I", "R"])
        by_hand = sir_builder().add_parameter(id="d", value=0.01)
        for bin_id in ("S", "I", "R"):
            by_hand.add_transition(
                id=f"death_{bin_id}", source=[bin_id], target=[], rate=f"d * {bin_id}"
            )

        assert_close(
            Simulation(build(automatic)).run(100),
            Simulation(build(by_hand)).run(100),
            rel=ENGINE_REL,
        )
