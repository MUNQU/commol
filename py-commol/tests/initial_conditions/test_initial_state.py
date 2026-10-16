"""The initial state: how the fractions expand into compartment amounts."""

import pytest

from commol import Simulation
from commol.commol_rs import _commol_rs as commol_rs
from tests.support import (
    a_to_b_builder,
    build,
    conditional_pathway_builder,
    initial_state,
    abc_builder,
)

GROUP = {"g1": 0.8, "g2": 0.2}
STATUS = {"s1": 0.7, "s2": 0.3}


class TestExpansion:
    def test_fractions_multiply_without_overrides(self) -> None:
        model = build(
            a_to_b_builder(
                rate="k * A",
                fractions={"A": 0.5, "B": 0.5},
                stratifications={"group": {"g1": 0.5, "g2": 0.5}},
            )
        )

        assert initial_state(model) == {
            "A_g1": 250.0,
            "A_g2": 250.0,
            "B_g1": 250.0,
            "B_g2": 250.0,
        }

    def test_a_conditional_stratification_splits_only_its_category(self) -> None:
        model = build(
            abc_builder(
                fractions={"A": 0.9, "B": 0.1, "C": 0.0},
                stratifications={"group": GROUP, "status": STATUS},
                conditions={"status": {"group": "g2"}},
            )
        )

        state = initial_state(model)

        for bin_id, share in (("A", 0.9), ("B", 0.1)):
            assert state[f"{bin_id}_g1"] == pytest.approx(1000 * share * 0.8)
            assert state[f"{bin_id}_g2_s1"] == pytest.approx(1000 * share * 0.2 * 0.7)
            assert state[f"{bin_id}_g2_s2"] == pytest.approx(1000 * share * 0.2 * 0.3)
        assert [state[f"C_{suffix}"] for suffix in ("g1", "g2_s1", "g2_s2")] == [
            0.0,
            0.0,
            0.0,
        ]

    def test_two_conditional_stratifications_split_their_category_twice(
        self,
    ) -> None:
        model = build(
            abc_builder(
                fractions={"A": 0.9, "B": 0.1, "C": 0.0},
                stratifications={
                    "group": GROUP,
                    "status": STATUS,
                    "level": {"l1": 0.8, "l2": 0.2},
                },
                conditions={
                    "status": {"group": "g2"},
                    "level": {"group": "g2"},
                },
            )
        )

        state = initial_state(model)

        assert state["A_g1"] == pytest.approx(720.0)
        assert state["B_g1"] == pytest.approx(80.0)
        assert state["A_g2_s1_l1"] == pytest.approx(100.8)
        assert state["A_g2_s1_l2"] == pytest.approx(25.2)
        assert state["A_g2_s2_l1"] == pytest.approx(43.2)
        assert state["A_g2_s2_l2"] == pytest.approx(10.8)


class TestInitialState:
    def test_initial_state_matches_the_engine(self) -> None:
        model = build(conditional_pathway_builder())
        engine = Simulation(model).engine

        state = model.structure.initial_state(model.process)

        assert [compartment.name for compartment in state] == engine.compartments
        assert [compartment.amount for compartment in state] == pytest.approx(
            engine.state
        )
        b_g1 = next(compartment for compartment in state if compartment.name == "B_g1")
        assert b_g1.bin == "B"
        assert b_g1.categories == {"group": "g1"}
        assert b_g1.amount == 0.0

    def test_the_binding_reports_bins_and_categories(self) -> None:
        model = build(conditional_pathway_builder())
        rust_process = commol_rs.core.Process.from_json(model.process.model_dump_json())
        rust_structure = commol_rs.core.Aggregate.from_json(
            model.structure.model_dump_json()
        )

        state = commol_rs.model.initial_state(rust_process, rust_structure)

        assert state[0] == ("A_g1", "A", [("group", "g1")], 450.0)
        assert [name for name, _, _, _ in state] == Simulation(
            model
        ).engine.compartments


class TestInitialConditionsBinding:
    """The PyO3 ``InitialConditions`` carries every field to the engine."""

    def test_overrides_reach_the_engine(self) -> None:
        core = commol_rs.core
        initial_conditions = core.InitialConditions(
            total=1000,
            bin_fractions=[("A", 1.0)],
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
            compartment_fractions=[("A_g1", 0.2)],
        )
        model = core.Model(
            name="LowLevel",
            process=core.Process(
                bins=[core.Bin(id="A", name="A")],
                stratifications=[
                    core.Stratification(id="group", categories=["g1", "g2"])
                ],
            ),
            structure=core.Aggregate(initial_conditions=initial_conditions),
            parameters=[],
            engine="DifferenceEquations",
        )

        engine = commol_rs.engine.Engine(model)

        assert initial_conditions.compartment_fractions == [("A_g1", 0.2)]
        assert engine.compartments == ["A_g1", "A_g2"]
        assert engine.state == [200.0, 500.0]

    def test_defaults_to_no_overrides(self) -> None:
        initial_conditions = commol_rs.core.InitialConditions(
            total=1000,
            bin_fractions=[("A", 1.0)],
        )

        assert initial_conditions.compartment_fractions == []

    def test_uncalibrated_bin_fraction_round_trips(self) -> None:
        initial_conditions = commol_rs.core.InitialConditions(
            total=1000,
            bin_fractions=[("A", 0.6), ("B", None)],
        )

        assert initial_conditions.bin_fractions == [("A", 0.6), ("B", None)]

    def test_every_declared_attribute_is_readable(self) -> None:
        initial_conditions = commol_rs.core.InitialConditions(
            total=1000,
            bin_fractions=[("A", 0.5), ("B", 0.5)],
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
            compartment_fractions=[("A_g1", 0.2)],
        )

        assert initial_conditions.total == 1000
        assert initial_conditions.bin_fractions == [("A", 0.5), ("B", 0.5)]
        assert {
            name: dict(categories)
            for name, categories in initial_conditions.stratification_fractions
        } == {"group": {"g1": 0.5, "g2": 0.5}}
        assert initial_conditions.compartment_fractions == [("A_g1", 0.2)]
