"""The builder assembles bins, parameters, transitions and initial conditions."""

from commol.context.model import Model
from tests.support import a_to_b_builder, build, sir_builder


class TestBuild:
    def test_build_simple_model(self) -> None:
        model = build(sir_builder(name="TestModel"))

        assert isinstance(model, Model)
        assert model.name == "TestModel"
        assert model.version == "1.0"
        assert [bin_.id for bin_ in model.structure.bins] == ["S", "I", "R"]
        assert [parameter.id for parameter in model.parameters] == ["beta", "gamma"]
        assert [t.id for t in model.structure.transitions] == ["infection", "recovery"]
        assert model.structure.initial_conditions.population_size == 1000


class TestCloneAndReset:
    def test_a_clone_builds_the_same_model(self) -> None:
        original = a_to_b_builder(accumulator="moved")

        clone = original.clone()

        assert build(clone) == build(original)

    def test_a_clone_is_independent(self) -> None:
        original = a_to_b_builder(accumulator="moved")
        clone = original.clone()

        clone.add_accumulator(id="other", name="Other")

        accumulators = build(original).structure.accumulators
        assert [accumulator.id for accumulator in accumulators] == ["moved"]

    def test_reset_removes_every_accumulator(self) -> None:
        builder = a_to_b_builder(accumulator="moved").reset()

        model = build(
            builder.add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="t_ab", source=["A"], target=["B"], rate="k")
            .set_initial_conditions(
                population_size=100,
                bin_fractions=[
                    {"bin": "A", "fraction": 1.0},
                    {"bin": "B", "fraction": 0.0},
                ],
            )
        )

        assert model.structure.accumulators == []
