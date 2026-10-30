"""The builder assembles bins, parameters, transitions and initial conditions."""

import pytest
from pydantic import ValidationError

from commol import Aggregate
from commol._model.bin import Bin
from commol._model.model import Model
from tests.support import FactoryBuilder, a_to_b_builder, abc_builder, build


class TestBuild:
    def test_build_simple_model(self) -> None:
        model = build(abc_builder(name="TestModel"))

        assert isinstance(model, Model)
        assert model.name == "TestModel"
        assert model.version == "1.0"
        assert [bin_.id for bin_ in model.process.bins] == ["A", "B", "C"]
        assert [parameter.id for parameter in model.parameters] == ["k1", "k2"]
        assert [t.id for t in model.process.transitions] == ["ab", "bc"]
        assert sum(model.structure.bin_amounts.values()) == 1000


class TestSummary:
    def test_a_summary_lists_what_was_added(self) -> None:
        builder = abc_builder(
            name="ABC", stratifications={"group": {"g1": 0.5, "g2": 0.5}}
        )
        builder.add_matrix(
            "M", "group", {"g1": {"g1": 1, "g2": 0}, "g2": {"g1": 0, "g2": 1}}
        )

        assert builder.get_summary() == {
            "name": "ABC",
            "description": None,
            "version": "1.0",
            "bins_count": 3,
            "bin_ids": ["A", "B", "C"],
            "stratifications_count": 1,
            "stratification_ids": ["group"],
            "matrices_count": 1,
            "matrix_ids": ["M"],
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
            .set_structure(Aggregate(bin_amounts={"A": 100, "B": 0}))
        )

        assert model.process.accumulators == []


@pytest.mark.parametrize(
    ("add", "message"),
    [
        (
            lambda b: b.add_bin(id="A", name="A2"),
            "Two bins share the id 'A'; give each bin its own id",
        ),
        (
            lambda b: b.add_accumulator(id="n", name="n2"),
            "Two accumulators share the id 'n'; give each accumulator its own id",
        ),
        (
            lambda b: b.add_stratification(id="g", categories=["x"]),
            "Two stratifications share the id 'g'; give each stratification its own id",
        ),
        (
            lambda b: b.add_transition(id="t_ab", source=["A"], target=["B"], rate="k"),
            "Two transitions share the id 't_ab'; give each transition its own id",
        ),
        (
            lambda b: b.add_parameter(id="k", value=0.2),
            "Two parameters share the id 'k'; give each parameter its own id",
        ),
    ],
    ids=["bin", "accumulator", "stratification", "transition", "parameter"],
)
def test_a_repeated_id_is_refused(add, message: str) -> None:
    builder = a_to_b_builder(
        accumulator="n", stratifications={"g": {"g1": 0.5, "g2": 0.5}}
    )
    add(builder)

    with pytest.raises(ValidationError) as error:
        build(builder)

    assert error.value.errors()[0]["msg"] == f"Value error, {message}"


class TestProcessChecks:
    def test_bins_of_one_id_are_one_member_of_a_set(self) -> None:
        bins = {Bin(id="A", name="first"), Bin(id="A", name="second")}

        assert bins == {Bin(id="A", name="any")}

    @pytest.mark.parametrize(
        ("bins", "message"),
        [
            (
                [],
                "The process has no bin; an aggregate model needs at least one bin",
            ),
            (
                ["A"],
                "The process has no transition; an aggregate model needs at least "
                "one transition",
            ),
        ],
    )
    def test_a_process_has_a_bin_and_a_transition(
        self, bins: list[str], message: str
    ) -> None:
        builder = FactoryBuilder(name="m")
        for bin_id in bins:
            builder.add_bin(bin_id, bin_id)
        builder.set_structure(Aggregate(bin_amounts=dict.fromkeys(bins, 10)))

        with pytest.raises(ValidationError) as error:
            build(builder)

        assert error.value.errors()[0]["msg"] == f"Value error, {message}"
