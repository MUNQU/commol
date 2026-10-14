"""Stratified rates give each group of compartments its own rate."""

import pytest
from pydantic import ValidationError

from commol import ModelBuilder, Simulation
from tests.support import (
    ENGINE_REL,
    assert_close,
    build,
    reference_run,
    sir_builder,
    stratify,
    total,
)
from tests.support.reference import State


def _susceptible(state: State) -> float:
    return state["S_young"] + state["S_old"]


def _infected(state: State) -> float:
    return state["I_young"] + state["I_old"]


class TestFallbackRate:
    def test_a_group_without_a_stratified_rate_takes_the_rate(self) -> None:
        model = build(
            sir_builder(
                parameters={"beta_young": 0.4, "beta_default": 0.2, "gamma": 0.1},
                infection=None,
                stratifications={"age": {"young": 0.5, "old": 0.5}},
            ).add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta_default * S * I / total",
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "young"}],
                        "rate": "beta_young * S * I / total",
                    }
                ],
            )
        )

        result = Simulation(model).run(5)

        # S and I are the bin totals, so both groups lose a share of the same
        # product; young at 0.4, old at the fallback 0.2.
        expected = reference_run(
            {
                "S_young": 495.0,
                "S_old": 495.0,
                "I_young": 5.0,
                "I_old": 5.0,
                "R_young": 0.0,
                "R_old": 0.0,
            },
            [
                (
                    "S_young",
                    "I_young",
                    lambda s, _: 0.4 * _susceptible(s) * _infected(s) / total(s),
                ),
                (
                    "S_old",
                    "I_old",
                    lambda s, _: 0.2 * _susceptible(s) * _infected(s) / total(s),
                ),
                ("I_young", "R_young", lambda s, _: 0.1 * _infected(s)),
                ("I_old", "R_old", lambda s, _: 0.1 * _infected(s)),
            ],
            5,
        )
        assert_close(result, expected, rel=ENGINE_REL)
        assert result["S_young"][1] == pytest.approx(495.0 - 3.96)
        assert result["S_old"][1] == pytest.approx(495.0 - 1.98)


class TestConditionalCompartments:
    def test_each_conditional_group_takes_its_own_rate(self) -> None:
        model = build(
            sir_builder(
                parameters={
                    "beta": 0.0,
                    "gamma_y60": 0.0,
                    "gamma_oe60_nv": 0.5,
                    "gamma_oe60_v": 0.8,
                },
                recovery=None,
                fractions={"S": 0.0, "I": 1.0, "R": 0.0},
                stratifications={
                    "age": {"y60": 0.5, "oe60": 0.5},
                    "vaccination": {"nv": 0.5, "v": 0.5},
                },
                conditions={"vaccination": {"age": "oe60"}},
            ).add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "y60"}],
                        "rate": "gamma_y60 * I_y60",
                    },
                    {
                        "conditions": [
                            {"stratification": "age", "category": "oe60"},
                            {"stratification": "vaccination", "category": "nv"},
                        ],
                        "rate": "gamma_oe60_nv * I_oe60_nv",
                    },
                    {
                        "conditions": [
                            {"stratification": "age", "category": "oe60"},
                            {"stratification": "vaccination", "category": "v"},
                        ],
                        "rate": "gamma_oe60_v * I_oe60_v",
                    },
                ],
            )
        )

        result = Simulation(model).run(5)

        expected = reference_run(
            {"I_y60": 500.0, "I_oe60_nv": 250.0, "I_oe60_v": 250.0},
            [
                ("I_oe60_nv", None, lambda s, _: 0.5 * s["I_oe60_nv"]),
                ("I_oe60_v", None, lambda s, _: 0.8 * s["I_oe60_v"]),
            ],
            5,
        )
        for name, values in expected.items():
            assert_close(result[name], values, rel=ENGINE_REL)
        assert result["R_y60"] == [0.0] * 6


class TestStratifiedRateValidation:
    def test_a_condition_the_source_cannot_meet_is_rejected_at_build(self) -> None:
        builder = (
            ModelBuilder(name="Routing", version="1.0")
            .add_bin(id="S", name="S")
            .add_parameter(id="a", value=0.1)
            .add_transition(
                id="aging",
                source=["S"],
                target=["S"],
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "age", "category": "y", "to": "o"},
                            {"stratification": "care", "category": "home"},
                        ],
                        "rate": "a",
                    }
                ],
            )
        )
        builder.set_initial_conditions(
            total=1000,
            bin_fractions=[{"bin": "S", "fraction": 1.0}],
            stratification_fractions=stratify(
                builder,
                {"age": {"y": 0.5, "o": 0.5}, "care": {"home": 0.5, "hosp": 0.5}},
                {"care": {"age": "o"}},
            ),
        )

        with pytest.raises(ValidationError, match="use 'to'"):
            build(builder)
