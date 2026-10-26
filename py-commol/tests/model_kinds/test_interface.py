"""The interface of a model kind, its variants, and the check that an engine
runs its kind."""

from typing import Literal, Self

import pytest
from pydantic import ValidationError

from commol import (
    Aggregate,
    Model,
    Simulation,
)
from commol._model.process import Process
from commol._model.structure import ModelStructure, SeriesLabel
from commol.engines.engine import ModelEngine
from tests.support import a_to_b_builder, abc_builder, build
from tests.support.kinds import DecayStructure, UnstratifiedVariant, decay_model


class TestKindCheck:
    def test_an_aggregate_engine_rejects_a_test_model(self) -> None:
        with pytest.raises(ValidationError) as error:
            decay_model(engine="DifferenceEquations")

        for name in ("DifferenceEquations", "aggregate", "test", "TestEngine"):
            assert name in str(error.value)

    def test_the_test_engine_rejects_an_aggregate_model_and_lists_its_engines(
        self,
    ) -> None:
        aggregate = build(a_to_b_builder())

        with pytest.raises(ValidationError) as error:
            Model(
                name="wrong",
                parameters=aggregate.parameters,
                engine=ModelEngine(type="TestEngine"),
                process=aggregate.process,
                structure=aggregate.structure,
            )

        for name in (
            "TestEngine",
            "aggregate",
            "test",
            "DifferenceEquations",
            "DifferentialEquations",
        ):
            assert name in str(error.value)

    def test_a_process_the_structure_cannot_run_fails_at_build(self) -> None:
        aggregate = build(a_to_b_builder())

        with pytest.raises(ValidationError, match="runs no process"):
            Model(
                name="wrong",
                parameters=decay_model().parameters,
                engine=ModelEngine(type="TestEngine"),
                process=aggregate.process,
                structure=decay_model().structure,
            )

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

    def test_a_concrete_kind_is_the_structure_of_its_kind(self, monkeypatch) -> None:
        monkeypatch.setattr(ModelStructure, "_kinds", dict(ModelStructure._kinds))

        class Registered(DecayStructure):
            kind: Literal["registered"] = "registered"  # ty: ignore[invalid-assignment]

        assert ModelStructure.of_kind("registered") is Registered

    def test_an_abstract_kind_is_no_structure(self, monkeypatch) -> None:
        monkeypatch.setattr(ModelStructure, "_kinds", dict(ModelStructure._kinds))

        class Abstract(ModelStructure):
            kind: Literal["test"] = "test"

        assert Abstract.__abstractmethods__
        assert ModelStructure.of_kind("test") is DecayStructure

    def test_a_kind_is_the_structure_of_one_class(self, monkeypatch) -> None:
        monkeypatch.setattr(ModelStructure, "_kinds", dict(ModelStructure._kinds))

        with pytest.raises(
            TypeError,
            match="Model kind 'test' with variant None is already the structure "
            "of DecayStructure",
        ):

            class Other(DecayStructure):
                pass

    def test_a_class_defined_again_stays_the_structure_of_its_kind(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(ModelStructure, "_kinds", dict(ModelStructure._kinds))

        def define() -> type[ModelStructure]:
            class Again(DecayStructure):
                kind: Literal["again"] = "again"  # ty: ignore[invalid-assignment]

            return Again

        define()
        again = define()
        assert ModelStructure.of_kind("again") is again

    def test_a_simulation_rejects_a_structure_that_is_not_a_model_structure(
        self,
    ) -> None:
        model = build(a_to_b_builder())
        broken = Model.model_construct(
            name="broken",
            parameters=model.parameters,
            engine=model.engine,
            structure={"kind": "aggregate"},
        )

        with pytest.raises(TypeError, match="not a ModelStructure"):
            Simulation(broken)

    def test_a_structure_of_an_unknown_kind_lists_the_kinds(self) -> None:
        data = decay_model().model_dump(mode="json")
        data["structure"]["kind"] = "network"

        with pytest.raises(ValidationError, match="aggregate"):
            Model.model_validate(data)

    def test_a_model_reads_back_as_the_class_of_its_kind(self, tmp_path) -> None:
        path = tmp_path / "decay.json"
        decay_model().to_json(path)

        loaded = Model.from_json(path)

        assert isinstance(loaded.structure, DecayStructure)
        assert loaded == decay_model()

    def test_each_kind_declares_the_values_calibration_may_set(self) -> None:
        aggregate = build(a_to_b_builder()).structure

        assert isinstance(aggregate, Aggregate)
        assert aggregate.calibration_value_types() == {"initial_condition"}
        assert decay_model().structure.calibration_value_types() == frozenset()

    def test_the_compartment_queries_of_an_engine_name_it(self) -> None:
        engine = Simulation(decay_model()).engine

        with pytest.raises(ValueError, match="TestEngine engine of this model"):
            engine.compartments
        with pytest.raises(ValueError, match="only an aggregate model"):
            engine.state

    def test_a_kind_without_a_unit_check_says_so(self) -> None:
        with pytest.raises(NotImplementedError) as error:
            decay_model().check_unit_consistency()

        assert str(error.value) == "A test model has no unit check"

    def test_a_kind_names_a_series_by_its_id_unless_it_knows_better(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(ModelStructure, "_kinds", dict(ModelStructure._kinds))

        class Plain(DecayStructure):
            kind: Literal["plain"] = "plain"  # ty: ignore[invalid-assignment]
            series_label = ModelStructure.series_label

        structure = Plain(outputs=[])

        assert structure.series_label(decay_model(), "x") == SeriesLabel("x", None)

    def test_a_kind_adds_no_sections_unless_it_has_some(self, monkeypatch) -> None:
        monkeypatch.setattr(ModelStructure, "_kinds", dict(ModelStructure._kinds))

        class Plain(DecayStructure):
            kind: Literal["plain"] = "plain"  # ty: ignore[invalid-assignment]
            report_sections = ModelStructure.report_sections
            describe = ModelStructure.describe

        structure = Plain(outputs=[])

        assert structure.report_sections(decay_model(), set(), None) == {}
        assert structure.describe(decay_model()) == []

    def test_a_kind_describes_itself_through_the_interface(self) -> None:
        description = decay_model().describe()

        assert description.section("dynamics").blocks
        assert "dx/dt = -k * x" in description.render("text")


class TestVariants:
    """A variant keeps the kind, the fields and the engines of its structure."""

    @staticmethod
    def _variant() -> UnstratifiedVariant:
        return UnstratifiedVariant(bin_amounts={"A": 990, "B": 10, "C": 0})

    def test_a_variant_saves_its_variant_and_loads_back_as_its_class(
        self, tmp_path
    ) -> None:
        model = build_variant(self._variant())
        path = tmp_path / "variant.json"
        model.to_json(path)

        loaded = Model.from_json(path)

        assert model.model_dump(mode="json")["structure"]["variant"] == "test"
        assert model.model_dump(mode="json")["structure"]["kind"] == "aggregate"
        assert isinstance(loaded.structure, UnstratifiedVariant)
        assert isinstance(loaded.structure, Aggregate)
        assert loaded == model

    @pytest.mark.parametrize("engine", ["DifferenceEquations", "DifferentialEquations"])
    def test_a_variant_runs_every_engine_of_its_structure(self, engine: str) -> None:
        plain = abc_builder()
        expected = Simulation(
            plain.build(plain.structure, engine=engine)  # ty: ignore[invalid-argument-type]
        ).run(3)

        results = Simulation(build_variant(self._variant(), engine=engine)).run(3)

        # The models differ, so the runs compare their positions and values.
        assert results.positions == expected.positions
        assert dict(results) == dict(expected)

    def test_the_engine_runs_the_checks_of_the_variant(self) -> None:
        builder = abc_builder(stratifications={"group": {"g1": 0.5, "g2": 0.5}})
        variant = UnstratifiedVariant(
            bin_amounts={"A": 990, "B": 10, "C": 0},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
        )

        with pytest.raises(ValidationError, match="test aggregate structure"):
            builder.build(variant, engine="DifferenceEquations")

    def test_an_unknown_variant_lists_the_variants_of_the_kind(self) -> None:
        data = build_variant(self._variant()).model_dump(mode="json")
        data["structure"]["variant"] = "no_such_variant"

        with pytest.raises(ValidationError, match="'test'"):
            Model.model_validate(data)

    def test_a_variant_needs_the_structure_of_its_kind(self) -> None:
        with pytest.raises(TypeError, match="needs the structure of its kind"):

            class Orphan(DecayStructure):
                kind: Literal["orphan"] = "orphan"  # ty: ignore[invalid-assignment]
                variant: Literal["v"] = "v"  # ty: ignore[invalid-assignment]


def build_variant(
    structure: UnstratifiedVariant, engine: str = "DifferenceEquations"
) -> Model[UnstratifiedVariant]:
    """The A → B → C process of ``abc_builder``, laid out by ``structure``."""
    builder = abc_builder()
    return builder.build(structure, engine=engine)  # ty: ignore[invalid-argument-type]


def test_the_process_of_a_model_is_saved_apart_from_its_structure() -> None:
    model = build(a_to_b_builder())

    data = model.model_dump(mode="json")

    assert set(data["process"]) == {
        "bins",
        "accumulators",
        "stratifications",
        "transitions",
    }
    assert "bins" not in data["structure"]
    assert data["structure"]["bin_amounts"] == {"A": 1000, "B": 0}
    assert Process.model_validate(data["process"]) == model.process
