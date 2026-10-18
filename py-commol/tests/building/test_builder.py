"""The builder assembles bins, parameters, transitions and initial conditions."""

from commol.context.model import Model
from tests.support import a_to_b_builder, build, abc_builder
from commol import Aggregate


class TestBuild:
    def test_build_simple_model(self) -> None:
        model = build(abc_builder(name="TestModel"))

        assert isinstance(model, Model)
        assert model.name == "TestModel"
        assert model.version == "1.0"
        assert [bin_.id for bin_ in model.process.bins] == ["A", "B", "C"]
        assert [parameter.id for parameter in model.parameters] == ["k1", "k2"]
        assert [t.id for t in model.process.transitions] == ["ab", "bc"]
        assert model.structure.total == 1000


class TestSummary:
    def test_a_summary_lists_what_was_added(self) -> None:
        builder = abc_builder(
            name="ABC", stratifications={"group": {"g1": 0.5, "g2": 0.5}}
        )

        assert builder.get_summary() == {
            "name": "ABC",
            "description": None,
            "version": "1.0",
            "bins_count": 3,
            "bin_ids": ["A", "B", "C"],
            "stratifications_count": 1,
            "stratification_ids": ["group"],
            "transitions_count": 2,
            "transition_ids": ["ab", "bc"],
            "parameters_count": 2,
            "parameter_ids": ["k1", "k2"],
        }


class TestCloneAndReset:
    def test_a_clone_builds_the_same_model(self) -> None:
        original = a_to_b_builder(accumulator="moved")

        clone = original.clone()

        assert build(clone) == build(original)

    def test_a_clone_is_independent(self) -> None:
        original = a_to_b_builder(accumulator="moved")
        clone = original.clone()

        clone.add_accumulator(id="other", name="Other")

        accumulators = build(original).process.accumulators
        assert [accumulator.id for accumulator in accumulators] == ["moved"]

    def test_reset_removes_every_accumulator(self) -> None:
        builder = a_to_b_builder(accumulator="moved").reset()

        model = build(
            builder.add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="t_ab", source=["A"], target=["B"], rate="k")
            .set_structure(Aggregate(total=100, bin_fractions={"A": 1.0, "B": 0.0}))
        )

        assert model.process.accumulators == []
