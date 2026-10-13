"""Transition flows: what the engine simulates is what is printed."""

import json

import pytest
from pydantic import ValidationError

from commol import Simulation
from commol.api.model_builder import ModelBuilder
from commol.constants import ModelTypes
from commol.context._model.engine_state import transition_flows
from commol.context.model import Model


def _age_model() -> Model:
    return (
        ModelBuilder(name="Aging", version="1.0")
        .add_bin(id="S", name="S")
        .add_bin(id="I", name="I")
        .add_parameter(id="g", value=0.1)
        .add_parameter(id="a", value=0.2)
        .add_stratification(id="age", categories=["young", "old"])
        .add_transition(id="rec", source=["S"], target=["I"], rate="g")
        .add_transition(id="aging", source=["young"], target=["old"], rate="a")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "S", "fraction": 1.0},
                {"bin": "I", "fraction": 0.0},
            ],
            stratification_fractions=[
                {
                    "stratification": "age",
                    "fractions": [
                        {"category": "young", "fraction": 0.5},
                        {"category": "old", "fraction": 0.5},
                    ],
                }
            ],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


class TestCategoryTransitions:
    def test_the_engine_moves_population_between_categories(self) -> None:
        result = Simulation(_age_model()).run(1)

        # rec moves 0.1 of each S compartment; aging moves 0.2 of S_young.
        assert result["S_young"][1] == pytest.approx(500 - 50 - 100)
        assert result["S_old"][1] == pytest.approx(500 - 50 + 100)
        assert sum(values[1] for values in result.values()) == pytest.approx(1000.0)

    def test_printed_flows_are_the_simulated_ones(self) -> None:
        flows = {
            (flow.transition, flow.source, flow.target, flow.term)
            for flow in transition_flows(_age_model())
        }

        assert ("aging", "S_young", "S_old", "a * S_young") in flows
        assert ("rec", "S_young", "I_young", "g * S_young") in flows


class TestBinsSharingAPrefix:
    def test_a_transition_moves_only_its_own_bin(self) -> None:
        model = (
            ModelBuilder(name="Prefix", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="AB", name="AB")
            .add_bin(id="B", name="B")
            .add_bin(id="BB", name="BB")
            .add_parameter(id="k", value=0.5)
            .add_transition(id="t", source=["A"], target=["B"], rate="k")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": bin_id, "fraction": 0.25}
                    for bin_id in ("A", "AB", "B", "BB")
                ],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        result = Simulation(model).run(1)

        assert result["AB"][1] == pytest.approx(250.0)
        assert result["BB"][1] == pytest.approx(250.0)
        assert result["A"][1] == pytest.approx(125.0)
        assert result["B"][1] == pytest.approx(375.0)


class TestStoichiometricTransitions:
    def test_every_source_loses_and_every_target_gains_the_flow(self) -> None:
        model = (
            ModelBuilder(name="Reaction", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_bin(id="C", name="C")
            .add_parameter(id="k", value=0.001)
            .add_transition(
                id="forward", source=["A", "B"], target=["C", "C"], rate="k * A * B"
            )
            .set_initial_conditions(
                population_size=100,
                bin_fractions=[
                    {"bin": "A", "fraction": 0.5},
                    {"bin": "B", "fraction": 0.5},
                    {"bin": "C", "fraction": 0.0},
                ],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        result = Simulation(model).run(1)

        flow = 0.001 * 50 * 50
        assert result["A"][1] == pytest.approx(50 - flow)
        assert result["B"][1] == pytest.approx(50 - flow)
        assert result["C"][1] == pytest.approx(2 * flow)


class TestStratifiedRateValidation:
    def test_a_condition_the_source_cannot_meet_is_rejected_at_build(self) -> None:
        builder = (
            ModelBuilder(name="Routing", version="1.0")
            .add_bin(id="S", name="S")
            .add_parameter(id="a", value=0.1)
            .add_stratification(id="age", categories=["y", "o"])
            .add_stratification(
                id="care",
                categories=["home", "hosp"],
                conditions=[{"stratification": "age", "category": "o"}],
            )
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
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[{"bin": "S", "fraction": 1.0}],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "y", "fraction": 0.5},
                            {"category": "o", "fraction": 0.5},
                        ],
                    },
                    {
                        "stratification": "care",
                        "fractions": [
                            {"category": "home", "fraction": 0.5},
                            {"category": "hosp", "fraction": 0.5},
                        ],
                    },
                ],
            )
        )

        with pytest.raises(ValidationError, match="use 'to'"):
            builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)


class TestNoTransitionConditions:
    def test_add_transition_has_no_condition_argument(self) -> None:
        builder = ModelBuilder(name="No condition").add_bin(id="A", name="A")

        with pytest.raises(TypeError, match="condition"):
            builder.add_transition(  # type: ignore[call-arg]
                id="t", source=["A"], target=[], rate="0.1", condition=None
            )

    def test_the_builder_has_no_create_condition(self) -> None:
        assert not hasattr(ModelBuilder, "create_condition")

    def test_a_model_file_with_a_condition_key_still_loads(self, tmp_path) -> None:
        model = _age_model()
        data = model.model_dump(mode="json")
        data["dynamics"]["transitions"][0]["condition"] = {
            "logic": "and",
            "rules": [{"variable": "state:S", "operator": "gt", "value": 0}],
        }
        path = tmp_path / "model.json"
        path.write_text(json.dumps(data))

        loaded = Model.from_json(path)

        assert Simulation(loaded).run(1) == Simulation(model).run(1)
