"""Transition flows: what the engine simulates is what is printed."""

import pytest
from pydantic import ValidationError

from commol import Aggregate
from commol._model.engine_state import transition_flows
from commol._model.formula_tree import formula_text
from commol._model.model import Model
from tests.support import (
    FactoryBuilder,
    a_to_b_builder,
    assert_total_conserved,
    build,
    run,
)


def _group_model() -> Model:
    """A → B in each group, and a move from g1 to g2."""
    return build(
        a_to_b_builder(
            name="Move",
            parameters={"g": 0.1, "a": 0.2},
            rate="g",
            transition_id="ab",
            stratifications={"group": {"g1": 0.5, "g2": 0.5}},
        ).add_transition(id="move", source=["g1"], target=["g2"], rate="a")
    )


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        (
            ["A"],
            ["Z"],
            "Transition 't2' names ids that are neither bins nor categories of a "
            'stratification: ["Z"]',
        ),
        (
            ["A"],
            ["g1"],
            'Transition \'t2\' mixes bins (["A"]) and categories (["g1"]); a '
            "transition moves between bins or between the categories of one "
            "stratification",
        ),
        (
            ["g1"],
            ["h1"],
            "Transition 't2' moves between the categories of several "
            'stratifications (["g", "h"]); a transition moves between the '
            "categories of one stratification",
        ),
    ],
    ids=["unknown", "bin and category", "two stratifications"],
)
def test_the_kind_of_a_transition_is_checked(
    source: list[str], target: list[str], message: str
) -> None:
    builder = a_to_b_builder(
        stratifications={"g": {"g1": 0.5, "g2": 0.5}, "h": {"h1": 0.5, "h2": 0.5}}
    )
    builder.add_transition(id="t2", source=source, target=target, rate="k")

    with pytest.raises(ValidationError) as error:
        build(builder)

    assert error.value.errors()[0]["msg"] == f"Value error, {message}"


class TestCategoryTransitions:
    def test_the_engine_moves_amounts_between_categories(self) -> None:
        result = run(_group_model(), 1)

        # ab moves 0.1 of each A compartment; move moves 0.2 of A_g1.
        assert result["A_g1"][1] == pytest.approx(500 - 50 - 100)
        assert result["A_g2"][1] == pytest.approx(500 - 50 + 100)
        assert_total_conserved(result, 1000.0)

    def test_printed_flows_are_the_simulated_ones(self) -> None:
        flows = {
            (flow.transition, flow.source, flow.target, formula_text(flow.term))
            for flow in transition_flows(
                _group_model().process, _group_model().structure
            )
        }

        assert ("move", "A_g1", "A_g2", "a * A_g1") in flows
        assert ("ab", "A_g1", "B_g1", "g * A_g1") in flows


class TestBinsSharingAPrefix:
    def test_a_transition_moves_only_its_own_bin(self) -> None:
        model = build(
            FactoryBuilder(name="Prefix", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="AB", name="AB")
            .add_bin(id="B", name="B")
            .add_bin(id="BB", name="BB")
            .add_parameter(id="k", value=0.5)
            .add_transition(id="t", source=["A"], target=["B"], rate="k")
            .set_structure(
                Aggregate(
                    bin_amounts={bin_id: 250 for bin_id in ("A", "AB", "B", "BB")}
                )
            )
        )

        result = run(model, 1)

        assert result["AB"][1] == pytest.approx(250.0)
        assert result["BB"][1] == pytest.approx(250.0)
        assert result["A"][1] == pytest.approx(125.0)
        assert result["B"][1] == pytest.approx(375.0)


class TestSeveralSourcesAndTargets:
    def test_every_source_loses_and_every_target_gains_the_flow(self) -> None:
        model = build(
            FactoryBuilder(name="Merge", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_bin(id="C", name="C")
            .add_parameter(id="k", value=0.001)
            .add_transition(
                id="forward", source=["A", "B"], target=["C", "C"], rate="k * A * B"
            )
            .set_structure(Aggregate(bin_amounts={"A": 50, "B": 50, "C": 0}))
        )

        result = run(model, 1)

        flow = 0.001 * 50 * 50
        assert result["A"][1] == pytest.approx(50 - flow)
        assert result["B"][1] == pytest.approx(50 - flow)
        assert result["C"][1] == pytest.approx(2 * flow)
