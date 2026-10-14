"""The interface of a model kind, and the check that an engine runs its kind."""

from typing import Literal, Self

import pytest
from pydantic import ValidationError

from commol import CompartmentalStructure, Model, ModelStructure, Simulation
from commol.context.engine import ModelEngine
from tests.support import a_to_b_builder, build
from tests.support.kinds import DecayStructure, decay_model


class TestKindCheck:
    def test_a_compartmental_engine_rejects_a_test_model(self) -> None:
        with pytest.raises(ValidationError) as error:
            decay_model(engine="DifferenceEquations")

        for name in ("DifferenceEquations", "compartmental", "test"):
            assert name in str(error.value)

    def test_the_test_engine_rejects_a_compartmental_model(self) -> None:
        compartmental = build(a_to_b_builder())

        with pytest.raises(ValidationError) as error:
            Model(
                name="wrong",
                parameters=compartmental.parameters,
                engine=ModelEngine(type="TestEngine"),
                structure=compartmental.structure,
            )

        for name in ("TestEngine", "compartmental", "test"):
            assert name in str(error.value)

    def test_an_unknown_engine_lists_the_engines(self) -> None:
        with pytest.raises(ValidationError, match="DifferenceEquations"):
            ModelEngine(type="Stochastic")


class TestStructureInterface:
    def test_a_kind_that_misses_a_method_cannot_be_made(self) -> None:
        class Incomplete(ModelStructure):
            kind: Literal["incomplete"] = "incomplete"

            def check_model(self, model: Model[Self]) -> None:  # noqa: ARG002
                return None

        with pytest.raises(TypeError, match="outputs_by_source"):
            Incomplete()

    def test_a_simulation_rejects_a_structure_that_is_not_a_model_structure(
        self,
    ) -> None:
        model = build(a_to_b_builder())
        broken = Model.model_construct(
            name="broken",
            parameters=model.parameters,
            engine=model.engine,
            structure={"kind": "compartmental"},
        )

        with pytest.raises(TypeError, match="not a ModelStructure"):
            Simulation(broken)

    def test_a_structure_of_an_unknown_kind_lists_the_kinds(self) -> None:
        data = decay_model().model_dump(mode="json")
        data["structure"]["kind"] = "network"

        with pytest.raises(ValidationError, match="compartmental"):
            Model.model_validate(data)

    def test_a_model_reads_back_as_the_class_of_its_kind(self, tmp_path) -> None:
        path = tmp_path / "decay.json"
        decay_model().to_json(path)

        loaded = Model.from_json(path)

        assert isinstance(loaded.structure, DecayStructure)
        assert loaded == decay_model()

    def test_each_kind_declares_the_values_calibration_may_set(self) -> None:
        compartmental = build(a_to_b_builder()).structure

        assert isinstance(compartmental, CompartmentalStructure)
        assert compartmental.calibration_value_types() == {"initial_condition"}
        assert decay_model().structure.calibration_value_types() == frozenset()

    def test_a_kind_without_equations_says_so(self) -> None:
        with pytest.raises(NotImplementedError, match="test model"):
            decay_model().print_equations()
