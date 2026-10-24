"""The builder assembles bins, parameters, transitions and initial conditions."""

import pytest
from pydantic import ValidationError

from commol.context.bin import Bin
from commol.context.model import Model
from tests.support import FactoryBuilder, a_to_b_builder, build, abc_builder
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
        assert sum(model.structure.bin_amounts.values()) == 1000


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
            .set_structure(Aggregate(bin_amounts={"A": 100, "B": 0}))
        )

        assert model.process.accumulators == []


class TestProcessChecks:
    @pytest.mark.parametrize(
        ("add", "message"),
        [
            (lambda b: b.add_bin(id="A", name="A2"), r"Duplicate bin IDs.*'A'"),
            (
                lambda b: b.add_stratification(id="g", categories=["x"]),
                r"Duplicate stratification IDs.*'g'",
            ),
            (
                lambda b: b.add_transition(
                    id="t_ab", source=["A"], target=["B"], rate="k"
                ),
                r"Duplicate transition IDs.*'t_ab'",
            ),
        ],
        ids=["bin", "stratification", "transition"],
    )
    def test_an_id_names_one_item_of_its_kind(self, add, message: str) -> None:
        builder = a_to_b_builder(stratifications={"g": {"g1": 0.5, "g2": 0.5}})
        add(builder)

        with pytest.raises(ValidationError, match=message):
            build(builder)

    @pytest.mark.parametrize(
        ("source", "target", "message"),
        [
            (["A"], ["Z"], r"'t2' contains invalid ids: \{'Z'\}"),
            (["A"], ["g1"], "'t2' mixes id types"),
            (["g1"], ["h1"], "categories from multiple stratifications: g, h"),
        ],
        ids=["unknown", "bin and category", "two stratifications"],
    )
    def test_a_transition_moves_between_bins_or_categories_of_one_stratification(
        self, source: list[str], target: list[str], message: str
    ) -> None:
        builder = a_to_b_builder(
            stratifications={"g": {"g1": 0.5, "g2": 0.5}, "h": {"h1": 0.5, "h2": 0.5}}
        )
        builder.add_transition(id="t2", source=source, target=target, rate="k")

        with pytest.raises(ValidationError, match=message):
            build(builder)

    def test_bins_of_one_id_are_one_member_of_a_set(self) -> None:
        bins = {Bin(id="A", name="first"), Bin(id="A", name="second")}

        assert bins == {Bin(id="A", name="any")}

    @pytest.mark.parametrize(
        ("bins", "message"),
        [([], "At least one bin"), (["A"], "At least one transition")],
    )
    def test_a_process_has_a_bin_and_a_transition(
        self, bins: list[str], message: str
    ) -> None:
        builder = FactoryBuilder(name="m")
        for bin_id in bins:
            builder.add_bin(bin_id, bin_id)
        builder.set_structure(Aggregate(bin_amounts=dict.fromkeys(bins, 10)))

        with pytest.raises(ValidationError, match=message):
            build(builder)
