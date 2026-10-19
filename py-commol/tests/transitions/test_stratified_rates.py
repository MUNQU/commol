"""Stratified rates give each group of compartments its own rate."""

import pytest
from pydantic import ValidationError

from commol import Aggregate, Simulation
from tests.support import (
    FactoryBuilder,
    a_to_b_builder,
    ENGINE_REL,
    assert_close,
    build,
    reference_run,
    abc_builder,
    stratify,
    total,
)
from tests.support.reference import State


def _amount_a(state: State) -> float:
    return state["A_g1"] + state["A_g2"]


def _amount_b(state: State) -> float:
    return state["B_g1"] + state["B_g2"]


class TestFallbackRate:
    def test_a_group_without_a_stratified_rate_takes_the_rate(self) -> None:
        model = build(
            abc_builder(
                parameters={"k1_g1": 0.4, "k1_default": 0.2, "k2": 0.1},
                ab=None,
                stratifications={"group": {"g1": 0.5, "g2": 0.5}},
            ).add_transition(
                id="ab",
                source=["A"],
                target=["B"],
                rate="k1_default * A * B / total",
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "group", "category": "g1"}],
                        "rate": "k1_g1 * A * B / total",
                    }
                ],
            )
        )

        result = Simulation(model).run(5)

        # A and B are the bin totals, so both groups lose a share of the same
        # product; g1 at 0.4, g2 at the fallback 0.2.
        expected = reference_run(
            {
                "A_g1": 495.0,
                "A_g2": 495.0,
                "B_g1": 5.0,
                "B_g2": 5.0,
                "C_g1": 0.0,
                "C_g2": 0.0,
            },
            [
                (
                    "A_g1",
                    "B_g1",
                    lambda s, _: 0.4 * _amount_a(s) * _amount_b(s) / total(s),
                ),
                (
                    "A_g2",
                    "B_g2",
                    lambda s, _: 0.2 * _amount_a(s) * _amount_b(s) / total(s),
                ),
                ("B_g1", "C_g1", lambda s, _: 0.1 * _amount_b(s)),
                ("B_g2", "C_g2", lambda s, _: 0.1 * _amount_b(s)),
            ],
            5,
        )
        assert_close(result, expected, rel=ENGINE_REL)
        assert result["A_g1"][1] == pytest.approx(495.0 - 3.96)
        assert result["A_g2"][1] == pytest.approx(495.0 - 1.98)


class TestConditionalCompartments:
    def test_each_conditional_group_takes_its_own_rate(self) -> None:
        model = build(
            abc_builder(
                parameters={
                    "k1": 0.0,
                    "k2_g1": 0.0,
                    "k2_g2_t1": 0.5,
                    "k2_g2_t2": 0.8,
                },
                bc=None,
                amounts={"A": 0, "B": 1000, "C": 0},
                stratifications={
                    "group": {"g1": 0.5, "g2": 0.5},
                    "type": {"t1": 0.5, "t2": 0.5},
                },
                conditions={"type": {"group": "g2"}},
            ).add_transition(
                id="bc",
                source=["B"],
                target=["C"],
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "group", "category": "g1"}],
                        "rate": "k2_g1 * B_g1",
                    },
                    {
                        "conditions": [
                            {"stratification": "group", "category": "g2"},
                            {"stratification": "type", "category": "t1"},
                        ],
                        "rate": "k2_g2_t1 * B_g2_t1",
                    },
                    {
                        "conditions": [
                            {"stratification": "group", "category": "g2"},
                            {"stratification": "type", "category": "t2"},
                        ],
                        "rate": "k2_g2_t2 * B_g2_t2",
                    },
                ],
            )
        )

        result = Simulation(model).run(5)

        expected = reference_run(
            {"B_g1": 500.0, "B_g2_t1": 250.0, "B_g2_t2": 250.0},
            [
                ("B_g2_t1", None, lambda s, _: 0.5 * s["B_g2_t1"]),
                ("B_g2_t2", None, lambda s, _: 0.8 * s["B_g2_t2"]),
            ],
            5,
        )
        for name, values in expected.items():
            assert_close(result[name], values, rel=ENGINE_REL)
        assert result["C_g1"] == [0.0] * 6


class TestStratifiedRateValidation:
    def test_a_condition_the_source_cannot_meet_is_rejected_at_build(self) -> None:
        builder = (
            FactoryBuilder(name="Routing", version="1.0")
            .add_bin(id="A", name="A")
            .add_parameter(id="a", value=0.1)
            .add_transition(
                id="move",
                source=["A"],
                target=["A"],
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "group", "category": "g1", "to": "g2"},
                            {"stratification": "type", "category": "t1"},
                        ],
                        "rate": "a",
                    }
                ],
            )
        )
        builder.set_structure(
            Aggregate(
                bin_amounts={"A": 1000},
                stratification_fractions=stratify(
                    builder,
                    {"group": {"g1": 0.5, "g2": 0.5}, "type": {"t1": 0.5, "t2": 0.5}},
                    {"type": {"group": "g2"}},
                ),
            )
        )

        with pytest.raises(ValidationError, match="use 'to'"):
            build(builder)


class TestConditions:
    @pytest.mark.parametrize(
        ("conditions", "message"),
        [
            ([], "at least one stratification condition"),
            (
                [
                    {"stratification": "g", "category": "g1"},
                    {"stratification": "g", "category": "g2"},
                ],
                r"Duplicate stratifications found: \['g'\]",
            ),
            (
                [{"stratification": "z", "category": "g1"}],
                r"rate 0: Stratification 'z' not found. Available: \['g'\]",
            ),
            (
                [{"stratification": "g", "category": "g9"}],
                r"Category 'g9' not found in stratification 'g'",
            ),
        ],
        ids=["none", "repeated", "unknown stratification", "unknown category"],
    )
    def test_a_rate_names_each_stratification_once_by_a_category_it_has(
        self, conditions: list[dict[str, str]], message: str
    ) -> None:
        builder = a_to_b_builder(stratifications={"g": ["g1", "g2"]})

        with pytest.raises(ValidationError, match=message):
            builder.add_transition(
                id="t2",
                source=["A"],
                target=["B"],
                rate="k",
                stratified_rates=[{"conditions": conditions, "rate": "k"}],
            )
            build(builder)
