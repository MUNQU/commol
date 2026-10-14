"""Stratification fractions: updating them, and reading categories by id."""

import pytest

from commol.context.model import Model
from tests.support import (
    a_to_b_builder,
    build,
    shared_category_builder,
)


def _stratified_model() -> Model:
    """One binary and one ternary stratification."""
    return build(
        a_to_b_builder(
            fractions={"A": 0.9, "B": 0.1},
            stratifications={
                "group": {"group1": 0.6, "group2": 0.4},
                "tier": {"low": 0.5, "mid": 0.3, "high": 0.2},
            },
        )
    )


def _fractions_of(model: Model, stratification_id: str) -> dict[str, float]:
    for stratification in model.structure.initial_conditions.stratification_fractions:
        if stratification.stratification == stratification_id:
            return {f.category: f.fraction for f in stratification.fractions}
    raise AssertionError(f"stratification '{stratification_id}' not found")


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

    def test_fractions_still_sum_to_one(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions(
            {"group1": 0.123, "low": 0.077, "mid": 0.401}
        )

        assert sum(_fractions_of(model, "group").values()) == pytest.approx(1.0)
        assert sum(_fractions_of(model, "tier").values()) == pytest.approx(1.0)

    def test_two_omitted_categories_are_rejected(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="left unnamed"):
            model.structure.update_stratification_fractions({"low": 0.4})

        assert _fractions_of(model, "tier") == {"low": 0.5, "mid": 0.3, "high": 0.2}

    def test_fully_specified_values_must_sum_to_one(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            model.structure.update_stratification_fractions(
                {"low": 0.2, "mid": 0.2, "high": 0.2}
            )

    def test_values_exceeding_one_are_rejected(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="negative fraction"):
            model.structure.update_stratification_fractions({"low": 0.7, "mid": 0.6})

    def test_unknown_category_is_rejected_and_lists_available(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="Category 'nope' not found") as excinfo:
            model.structure.update_stratification_fractions({"nope": 0.4})

        assert "group1" in str(excinfo.value)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_out_of_range_fraction_is_rejected(self, value: float) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            model.structure.update_stratification_fractions({"group1": value})

    def test_boundary_values_are_allowed(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions({"group1": 0.0})

        assert _fractions_of(model, "group") == {"group1": 0.0, "group2": 1.0}

    def test_untouched_stratifications_keep_their_fractions(self) -> None:
        model = _stratified_model()

        model.structure.update_stratification_fractions({"group1": 0.25})

        assert _fractions_of(model, "tier") == {"low": 0.5, "mid": 0.3, "high": 0.2}

    def test_a_bare_shared_category_is_rejected_as_ambiguous(self) -> None:
        model = build(shared_category_builder())

        with pytest.raises(ValueError, match="ambiguous"):
            model.structure.update_stratification_fractions({"y": 0.6})


class TestQualifiedCategoryReads:
    """Read methods resolve category ids like calibration does."""

    def test_bare_shared_name_is_ambiguous(self) -> None:
        model = build(shared_category_builder())

        with pytest.raises(ValueError, match="ambiguous"):
            model.structure.subgroup_total(["y"])
        with pytest.raises(ValueError, match="ambiguous"):
            model.structure.initial_conditions.get_category_fraction("y")

    def test_qualified_name_selects_its_stratification(self) -> None:
        model = build(shared_category_builder())
        model.structure.update_stratification_fractions({"sex:y": 0.8})

        assert model.structure.subgroup_total(["sex:y"]) == pytest.approx(800.0)
        assert model.structure.subgroup_total(["age:y"]) == pytest.approx(500.0)

    def test_qualified_name_with_overrides_sums_the_engine_state(self) -> None:
        model = build(
            shared_category_builder(
                compartment_fractions=[{"compartment": "A_y_y", "fraction": 0.0}]
            )
        )

        # A_y_y is pinned to 0, so only A_o_y holds sex:y.
        assert model.structure.subgroup_total(["sex:y"]) == pytest.approx(250.0)
        assert model.structure.subgroup_total(["age:o", "sex:y"]) == pytest.approx(
            250.0
        )

    def test_conditioning_categories_are_qualified_when_shared(self) -> None:
        model = build(
            a_to_b_builder(
                stratifications={
                    "age": {"y": 0.5, "o": 0.5},
                    "sex": {"y": 0.5, "o": 0.5},
                    "care": {"home": 0.5, "hosp": 0.5},
                },
                conditions={"care": {"sex": "y"}},
            )
        )

        assert model.structure.get_conditioning_categories("care") == ("sex:y",)
        assert model.structure.subgroup_total(("sex:y",)) == pytest.approx(500.0)
