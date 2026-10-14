"""Transition flows: what the engine simulates is what is printed."""

import json

import pytest
from pydantic import ValidationError

from commol import ModelBuilder
from commol.context._model.engine_state import transition_flows
from commol.context.model import Model
from tests.support import a_to_b_builder, assert_total_conserved, build, run


def _age_model() -> Model:
    """A → B in each age group, and aging from young to old."""
    return build(
        a_to_b_builder(
            name="Aging",
            parameters={"g": 0.1, "a": 0.2},
            rate="g",
            transition_id="rec",
            stratifications={"age": {"young": 0.5, "old": 0.5}},
        ).add_transition(id="aging", source=["young"], target=["old"], rate="a")
    )


class TestCategoryTransitions:
    def test_the_engine_moves_amounts_between_categories(self) -> None:
        result = run(_age_model(), 1)

        # rec moves 0.1 of each A compartment; aging moves 0.2 of A_young.
        assert result["A_young"][1] == pytest.approx(500 - 50 - 100)
        assert result["A_old"][1] == pytest.approx(500 - 50 + 100)
        assert_total_conserved(result, 1000.0)

    def test_printed_flows_are_the_simulated_ones(self) -> None:
        flows = {
            (flow.transition, flow.source, flow.target, flow.term)
            for flow in transition_flows(_age_model().structure)
        }

        assert ("aging", "A_young", "A_old", "a * A_young") in flows
        assert ("rec", "A_young", "B_young", "g * A_young") in flows


class TestBinsSharingAPrefix:
    def test_a_transition_moves_only_its_own_bin(self) -> None:
        model = build(
            ModelBuilder(name="Prefix", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="AB", name="AB")
            .add_bin(id="B", name="B")
            .add_bin(id="BB", name="BB")
            .add_parameter(id="k", value=0.5)
            .add_transition(id="t", source=["A"], target=["B"], rate="k")
            .set_initial_conditions(
                total=1000,
                bin_fractions=[
                    {"bin": bin_id, "fraction": 0.25}
                    for bin_id in ("A", "AB", "B", "BB")
                ],
            )
        )

        result = run(model, 1)

        assert result["AB"][1] == pytest.approx(250.0)
        assert result["BB"][1] == pytest.approx(250.0)
        assert result["A"][1] == pytest.approx(125.0)
        assert result["B"][1] == pytest.approx(375.0)


class TestStoichiometricTransitions:
    def test_every_source_loses_and_every_target_gains_the_flow(self) -> None:
        model = build(
            ModelBuilder(name="Reaction", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_bin(id="C", name="C")
            .add_parameter(id="k", value=0.001)
            .add_transition(
                id="forward", source=["A", "B"], target=["C", "C"], rate="k * A * B"
            )
            .set_initial_conditions(
                total=100,
                bin_fractions=[
                    {"bin": "A", "fraction": 0.5},
                    {"bin": "B", "fraction": 0.5},
                    {"bin": "C", "fraction": 0.0},
                ],
            )
        )

        result = run(model, 1)

        flow = 0.001 * 50 * 50
        assert result["A"][1] == pytest.approx(50 - flow)
        assert result["B"][1] == pytest.approx(50 - flow)
        assert result["C"][1] == pytest.approx(2 * flow)


class TestNoTransitionConditions:
    def test_add_transition_has_no_condition_argument(self) -> None:
        builder = ModelBuilder(name="No condition").add_bin(id="A", name="A")

        with pytest.raises(TypeError, match="condition"):
            builder.add_transition(  # type: ignore[call-arg]
                id="t", source=["A"], target=[], rate="0.1", condition=None
            )

    def test_the_builder_has_no_create_condition(self) -> None:
        assert not hasattr(ModelBuilder, "create_condition")

    def test_a_model_file_with_a_condition_key_names_it(self, tmp_path) -> None:
        model = _age_model()
        data = model.model_dump(mode="json")
        data["structure"]["transitions"][0]["condition"] = {
            "logic": "and",
            "rules": [{"variable": "state:A", "operator": "gt", "value": 0}],
        }
        path = tmp_path / "model.json"
        path.write_text(json.dumps(data))

        with pytest.raises(ValidationError, match="condition"):
            Model.from_json(path)
