"""Transition flows: what the engine simulates is what is printed."""

import json

import pytest
from pydantic import ValidationError

from commol import Aggregate, ModelBuilder
from commol.context._model.engine_state import transition_flows
from commol.context.model import Model
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


class TestCategoryTransitions:
    def test_the_engine_moves_amounts_between_categories(self) -> None:
        result = run(_group_model(), 1)

        # ab moves 0.1 of each A compartment; move moves 0.2 of A_g1.
        assert result["A_g1"][1] == pytest.approx(500 - 50 - 100)
        assert result["A_g2"][1] == pytest.approx(500 - 50 + 100)
        assert_total_conserved(result, 1000.0)

    def test_printed_flows_are_the_simulated_ones(self) -> None:
        flows = {
            (flow.transition, flow.source, flow.target, flow.term)
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
                    total=1000,
                    bin_fractions={bin_id: 0.25 for bin_id in ("A", "AB", "B", "BB")},
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
            .set_structure(
                Aggregate(total=100, bin_fractions={"A": 0.5, "B": 0.5, "C": 0.0})
            )
        )

        result = run(model, 1)

        flow = 0.001 * 50 * 50
        assert result["A"][1] == pytest.approx(50 - flow)
        assert result["B"][1] == pytest.approx(50 - flow)
        assert result["C"][1] == pytest.approx(2 * flow)


class TestNoTransitionConditions:
    def test_add_transition_has_no_condition_argument(self) -> None:
        builder = FactoryBuilder(name="No condition").add_bin(id="A", name="A")

        with pytest.raises(TypeError, match="condition"):
            builder.add_transition(  # type: ignore[call-arg]
                id="t", source=["A"], target=[], rate="0.1", condition=None
            )

    def test_the_builder_has_no_create_condition(self) -> None:
        assert not hasattr(ModelBuilder, "create_condition")

    def test_a_model_file_with_a_condition_key_names_it(self, tmp_path) -> None:
        model = _group_model()
        data = model.model_dump(mode="json")
        data["process"]["transitions"][0]["condition"] = {
            "logic": "and",
            "rules": [{"variable": "state:A", "operator": "gt", "value": 0}],
        }
        path = tmp_path / "model.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValidationError, match="condition"):
            Model.from_json(path)
