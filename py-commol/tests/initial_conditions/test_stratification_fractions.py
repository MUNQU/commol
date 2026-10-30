"""Stratification fractions: splits by category, updating them, and reading
categories by id."""

import pytest
from pydantic import ValidationError

from commol import Aggregate, CalibrationParameter
from commol.description.blocks import Table
from commol._model.engine_state import calibrated_initial_state
from commol._model.model import Model
from tests.support import (
    a_to_b_builder,
    build,
    shared_category_builder,
    stratify,
)

# ``type`` splits ``g1`` 0.25 / 0.75 and ``g2`` 0.5 / 0.5.
TYPE_BY_GROUP = {
    "by": "group",
    "g1": {"t1": 0.25, "t2": 0.75},
    "g2": {"t1": 0.5, "t2": 0.5},
}


def _by_group(
    type_fractions: dict[str, float] | dict[str, str | dict[str, float]],
) -> Model:
    """``group`` splits 0.4 / 0.6 into ``g1`` / ``g2``, then ``type`` splits
    by ``type_fractions``; 1000 starts in A."""
    builder = a_to_b_builder()
    fractions = stratify(
        builder,
        {"group": {"g1": 0.4, "g2": 0.6}, "type": {"t1": 0.5, "t2": 0.5}},
    )
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0},
            stratification_fractions={
                "group": fractions["group"],
                "type": type_fractions,
            },
        )
    )
    return build(builder)


def _amounts(model: Model) -> dict[str, float]:
    return {
        compartment.name: compartment.amount
        for compartment in model.structure.initial_state(model.process)
        if compartment.amount
    }


def _fraction_rows(model: Model) -> list[list[object]]:
    """The rows of the table of fractions of the description."""
    return [
        row
        for block in model.describe().section("initialization").blocks
        if isinstance(block, Table) and block.heading == "Stratification fractions"
        for row in block.rows
    ]


def test_a_stratification_splits_each_category_of_an_earlier_one() -> None:
    model = _by_group(TYPE_BY_GROUP)

    assert _amounts(model) == pytest.approx(
        {"A_g1_t1": 100.0, "A_g1_t2": 300.0, "A_g2_t1": 300.0, "A_g2_t2": 300.0}
    )


def test_one_split_per_stratification_is_unchanged() -> None:
    model = _by_group({"t1": 0.25, "t2": 0.75})
    t1 = CalibrationParameter(
        id="t1", parameter_type="initial_condition", min_bound=0.0, max_bound=1.0
    )

    # The amount is the product in the order of the stratifications.
    assert _amounts(model) == {
        "A_g1_t1": 1000 * 0.4 * 0.25,
        "A_g1_t2": 1000 * 0.4 * 0.75,
        "A_g2_t1": 1000 * 0.6 * 0.25,
        "A_g2_t2": 1000 * 0.6 * 0.75,
    }
    assert calibrated_initial_state(model, [t1], [0.5]).state == pytest.approx(
        {"A_g1_t1": 200.0, "A_g1_t2": 200.0, "A_g2_t1": 300.0, "A_g2_t2": 300.0}
        | {f"B_{g}_{t}": 0.0 for g in ("g1", "g2") for t in ("t1", "t2")}
    )
    assert _fraction_rows(model)[2:] == [
        ["type", "t1", 0.25, "each bin"],
        ["type", "t2", 0.75, "each bin"],
    ]
    assert model.applied_parameters({})["category_splits"]["t1"] == {
        "group": [],
        "fraction": 0.25,
        "calibrated": False,
    }
    model.structure.update_stratification_fractions({"t1": 0.5})
    assert model.structure.get_category_fraction("t2") == 0.5


class TestSplitByCategory:
    def test_the_description_gives_each_split_of_its_category(self) -> None:
        model = _by_group(TYPE_BY_GROUP)

        assert _fraction_rows(model)[2:] == [
            ["type", "t1", 0.25, "g1"],
            ["type", "t2", 0.75, "g1"],
            ["type", "t1", 0.5, "g2"],
            ["type", "t2", 0.5, "g2"],
        ]

    def test_the_report_gives_a_row_per_category_of_each_split(self) -> None:
        model = _by_group(TYPE_BY_GROUP)

        rows = model.applied_parameters({})["category_splits"]

        assert rows["t2 of g1"] == {
            "group": ["g1"],
            "fraction": 0.75,
            "calibrated": False,
        }
        assert rows["t1 of g2"]["fraction"] == 0.5
        assert rows["g1"]["group"] == []

    def test_its_categories_resolve_by_name(self) -> None:
        model = _by_group(TYPE_BY_GROUP)

        assert model.structure.resolve_category("t1") == ("type", "t1")
        assert model.structure.subgroup_total(model.process, ("t1",)) == (
            pytest.approx(400.0)
        )

    def test_one_fraction_of_it_is_not_read(self) -> None:
        model = _by_group(TYPE_BY_GROUP)

        with pytest.raises(ValueError, match="differ by the category of 'group'"):
            model.structure.get_category_fraction("t1")

    def test_an_update_sets_its_value_in_every_split(self) -> None:
        model = _by_group(TYPE_BY_GROUP)

        model.structure.update_stratification_fractions({"t1": 0.3})

        assert model.structure.stratification_fractions["type"] == {
            "by": "group",
            "g1": {"t1": 0.3, "t2": 0.7},
            "g2": {"t1": 0.3, "t2": 0.7},
        }

    def test_it_reads_back_from_its_json(self) -> None:
        model = _by_group(TYPE_BY_GROUP)

        reloaded = Model.model_validate_json(model.model_dump_json())

        assert reloaded.structure.stratification_fractions == (
            model.structure.stratification_fractions
        )
        assert _amounts(reloaded) == _amounts(model)

    def test_a_fraction_of_a_category_gets_the_message_of_the_engine(self) -> None:
        with pytest.raises(ValidationError) as error:
            Aggregate(
                bin_amounts={"A": 1000, "B": 0},
                stratification_fractions={
                    "type": {**TYPE_BY_GROUP, "g2": {"t1": 0.5, "t2": 0.4}}
                },
            )

        assert (
            "Stratification 'type' within category 'g2' of 'group': the fractions "
            "of its categories sum to 1 within 1e-6, got 0.9" in str(error.value)
        )

    def test_a_split_by_a_later_stratification_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not a stratification declared"):
            _by_group({"by": "type", "t1": {"t1": 1.0, "t2": 0.0}})


def _on_g2_g3(
    type_fractions: dict[str, float] | dict[str, str | dict[str, float]],
) -> Model:
    """``group`` splits 0.2 / 0.3 / 0.5 into ``g1`` / ``g2`` / ``g3``, and
    ``type`` splits ``g2`` and ``g3`` by ``type_fractions``; 1000 starts in A."""
    builder = a_to_b_builder()
    fractions = stratify(
        builder,
        {"group": {"g1": 0.2, "g2": 0.3, "g3": 0.5}, "type": {"t1": 0.5, "t2": 0.5}},
        conditions={"type": {"group": ["g2", "g3"]}},
    )
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0},
            stratification_fractions={**fractions, "type": type_fractions},
        )
    )
    return build(builder)


def test_a_split_by_category_names_the_categories_it_reaches() -> None:
    model = _on_g2_g3(
        {"by": "group", "g2": {"t1": 0.25, "t2": 0.75}, "g3": {"t1": 0.5, "t2": 0.5}}
    )

    assert _amounts(model) == pytest.approx(
        {"A_g1": 200.0, "A_g2_t1": 75.0, "A_g2_t2": 225.0, "A_g3_t1": 250.0}
        | {"A_g3_t2": 250.0}
    )
    assert _fraction_rows(model)[3:5] == [
        ["type", "t1", 0.25, "g2"],
        ["type", "t2", 0.75, "g2"],
    ]
    assert model.applied_parameters({})["category_splits"]["t1 of g2"]["group"] == [
        "g2"
    ]


def test_one_split_of_a_set_is_of_its_categories() -> None:
    model = _on_g2_g3({"t1": 0.25, "t2": 0.75})

    assert _fraction_rows(model)[3] == ["type", "t1", 0.25, "g2|g3"]
    assert model.applied_parameters({})["category_splits"]["t1"]["group"] == [
        "g2",
        "g3",
    ]
    assert model.structure.subgroup_total(model.process, ["g2", "g3"]) == (
        pytest.approx(800.0)
    )


def _stratified_model() -> Model:
    """One binary and one ternary stratification."""
    return build(
        a_to_b_builder(
            amounts={"A": 900, "B": 100},
            stratifications={
                "group": {"group1": 0.6, "group2": 0.4},
                "tier": {"low": 0.5, "mid": 0.3, "high": 0.2},
            },
        )
    )


def _fractions_of(model: Model, stratification_id: str) -> dict[str, float]:
    return dict(model.structure.stratification_fractions[stratification_id])


class TestUpdateStratificationFractions:
    def test_writing_one_of_two_categories_fills_the_other(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions({"group1": 0.25})

        assert _fractions_of(model, "group") == {"group1": 0.25, "group2": 0.75}

    def test_writing_the_other_category_is_symmetric(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions({"group2": 0.25})

        assert _fractions_of(model, "group") == {"group1": 0.75, "group2": 0.25}

    def test_all_categories_of_a_ternary_stratification_can_be_set(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions(
            {"low": 0.2, "mid": 0.3, "high": 0.5}
        )

        assert _fractions_of(model, "tier") == {"low": 0.2, "mid": 0.3, "high": 0.5}

    def test_one_omitted_category_takes_the_remainder(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions({"low": 0.2, "mid": 0.3})

        assert _fractions_of(model, "tier") == {"low": 0.2, "mid": 0.3, "high": 0.5}

    def test_several_stratifications_update_in_one_call(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions(
            {"group1": 0.25, "low": 0.1, "mid": 0.4}
        )

        assert _fractions_of(model, "group") == {"group1": 0.25, "group2": 0.75}
        assert _fractions_of(model, "tier") == {"low": 0.1, "mid": 0.4, "high": 0.5}

    def test_two_omitted_categories_are_rejected(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="left unnamed"):
            model.structure.update_stratification_fractions({"low": 0.4})

        assert _fractions_of(model, "tier") == {"low": 0.5, "mid": 0.3, "high": 0.2}

    def test_fully_specified_values_must_sum_to_one(self) -> None:
        model = _stratified_model()

        with pytest.raises(
            ValueError,
            match="Stratification 'tier': the fractions of its categories sum to 1",
        ):
            model.structure.update_stratification_fractions(
                {"low": 0.2, "mid": 0.2, "high": 0.2}
            )

        assert _fractions_of(model, "tier") == {"low": 0.5, "mid": 0.3, "high": 0.2}

    def test_values_exceeding_one_are_rejected(self) -> None:
        model = _stratified_model()

        with pytest.raises(
            ValueError,
            match="the fractions of its categories sum to 1 within 1e-6, got 1.29",
        ):
            model.structure.update_stratification_fractions({"low": 0.7, "mid": 0.6})

    def test_unknown_category_is_rejected_and_lists_available(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="Category 'nope' not found") as excinfo:
            model.structure.update_stratification_fractions({"nope": 0.4})

        assert "group1" in str(excinfo.value)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_out_of_range_fraction_gets_the_message_of_the_engine(
        self, value: float
    ) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError) as error:
            model.structure.update_stratification_fractions(
                {"group1": value, "low": 0.1, "mid": 0.4}
            )

        assert str(error.value) == (
            f"Stratification 'group', category 'group1': fraction is a number "
            f"from 0 to 1, got {value}"
        )
        assert _fractions_of(model, "tier") == {"low": 0.5, "mid": 0.3, "high": 0.2}

    def test_a_complete_update_may_miss_one_by_its_tolerance(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions(
            {"group1": 0.6000005, "group2": 0.4}
        )

        assert _fractions_of(model, "group") == {"group1": 0.6000005, "group2": 0.4}

    def test_a_partial_update_may_pass_one_by_its_tolerance(self) -> None:
        model = _stratified_model()
        near_one = 1.0 + 1e-6 - 0.5

        model.structure.update_stratification_fractions({"low": 0.5, "mid": near_one})

        assert _fractions_of(model, "tier") == {
            "low": 0.5,
            "mid": near_one,
            "high": 0.0,
        }

    def test_a_partial_update_just_below_one_leaves_its_remainder(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions({"low": 0.5, "mid": 0.4999995})

        assert _fractions_of(model, "tier")["high"] == pytest.approx(5e-7, abs=1e-12)

    def test_boundary_values_are_allowed(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions({"group1": 0.0})

        assert _fractions_of(model, "group") == {"group1": 0.0, "group2": 1.0}

    def test_untouched_stratifications_keep_their_fractions(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions({"group1": 0.25})

        assert _fractions_of(model, "tier") == {"low": 0.5, "mid": 0.3, "high": 0.2}


def test_an_ambiguous_category_is_refused() -> None:
    model = build(shared_category_builder())

    with pytest.raises(ValueError) as error:
        model.structure.update_stratification_fractions({"c1": 0.6})

    assert str(error.value) == (
        "Category 'c1' is ambiguous: it is declared by stratifications "
        "[\"group\", \"level\"]. Qualify it as 'group:c1' or 'level:c1'."
    )


class TestQualifiedCategoryReads:
    """Read methods resolve category ids like calibration does."""

    def test_bare_shared_name_is_ambiguous(self) -> None:
        model = build(shared_category_builder())

        with pytest.raises(ValueError, match="ambiguous"):
            model.structure.subgroup_total(model.process, ["c1"])
        with pytest.raises(ValueError, match="ambiguous"):
            model.structure.get_category_fraction("c1")

    def test_qualified_name_selects_its_stratification(self) -> None:
        model = build(shared_category_builder())
        model.structure.update_stratification_fractions({"level:c1": 0.8})

        assert model.structure.subgroup_total(
            model.process, ["level:c1"]
        ) == pytest.approx(800.0)
        assert model.structure.subgroup_total(
            model.process, ["group:c1"]
        ) == pytest.approx(500.0)
        assert model.structure.get_category_fraction("level:c1") == 0.8
        assert model.structure.get_category_fraction("group:c1") == 0.5

    def test_qualified_name_with_overrides_sums_the_engine_state(self) -> None:
        model = build(shared_category_builder(compartment_amounts={"A_c1_c1": 0}))

        # A_c1_c1 is pinned to 0, so only A_c2_c1 holds level:c1.
        assert model.structure.subgroup_total(
            model.process, ["level:c1"]
        ) == pytest.approx(250.0)
        assert model.structure.subgroup_total(
            model.process, ["group:c2", "level:c1"]
        ) == pytest.approx(250.0)

    def test_conditioning_categories_are_qualified_when_shared(self) -> None:
        model = build(
            a_to_b_builder(
                stratifications={
                    "group": {"c1": 0.5, "c2": 0.5},
                    "level": {"c1": 0.5, "c2": 0.5},
                    "type": {"t1": 0.5, "t2": 0.5},
                },
                conditions={"type": {"level": "c1"}},
            )
        )

        assert model.structure.get_conditioning_categories(model.process, "type") == (
            "level:c1",
        )
        assert model.structure.subgroup_total(
            model.process, ("level:c1",)
        ) == pytest.approx(500.0)


class TestNames:
    @pytest.mark.parametrize(
        ("fractions", "message"),
        [
            ({}, r'stratification ids. Missing ids: \["g"\], Extra ids: \[\]'),
            (
                {"g": {"g1": 1.0, "g3": 0.0}},
                r'Missing categories: \["g2"\], Extra categories: \["g3"\]',
            ),
        ],
        ids=["stratification", "category"],
    )
    def test_the_fractions_of_a_stratification_name_its_categories(
        self, fractions: dict[str, dict[str, float]], message: str
    ) -> None:
        builder = a_to_b_builder(stratifications={"g": {"g1": 0.5, "g2": 0.5}})
        builder.set_structure(
            Aggregate(
                bin_amounts={"A": 10, "B": 0},
                stratification_fractions=fractions,
            )
        )

        with pytest.raises(ValidationError, match=message):
            build(builder)
