"""Per-compartment initial conditions: the type, the builder and the model."""

import pytest
from pydantic import ValidationError

from collections.abc import Mapping

from commol import Aggregate, Simulation
from commol.context.model import Model
from tests.support import (
    FactoryBuilder,
    a_to_b_builder,
    build,
    conditional_pathway_builder,
    override_builder,
    run,
)


def _all_in_g1(overrides: Mapping[str, float]) -> FactoryBuilder:
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        stratifications={"group": {"g1": 1.0, "g2": 0.0}},
        compartment_fractions=overrides,
    )


def _overrides(model: Model) -> dict[str, float]:
    return dict(model.structure.compartment_fractions)


class TestCompartmentFractionType:
    @pytest.mark.parametrize("fraction", [1.5, -0.1])
    def test_rejects_a_fraction_outside_zero_and_one(self, fraction: float) -> None:
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            Aggregate(
                total=1000,
                bin_fractions={"A": 1.0},
                compartment_fractions={"A_g1": fraction},
            )


class TestBuilderCompartmentFractions:
    def test_unknown_compartment_is_rejected_by_the_engine_at_build(self) -> None:
        builder = _all_in_g1({"A_missing": 0.0})

        with pytest.raises(ValueError, match="Unknown compartment 'A_missing'"):
            build(builder)

    def test_over_allocation_is_rejected_by_the_engine_at_build(self) -> None:
        builder = _all_in_g1({"A_g2": 0.5})

        with pytest.raises(ValueError, match="exceed the total"):
            build(builder)

    def test_uncalibrated_bin_fractions_count_as_zero(self) -> None:
        model = build(
            a_to_b_builder(
                rate="k * A",
                fractions={"A": None, "B": 0.5},
                compartment_fractions={"A": 0.4},
            )
        )

        assert _overrides(model) == {"A": 0.4}

    def test_uncalibrated_override_exceeding_the_total_is_rejected(self) -> None:
        builder = a_to_b_builder(
            rate="k * A",
            fractions={"A": None, "B": 0.5},
            compartment_fractions={"A": 1.0},
        )

        with pytest.raises(ValueError, match="exceed the total"):
            build(builder)


class TestOverrideInARun:
    def test_zero_override_keeps_the_other_stratum_empty(self) -> None:
        results = run(conditional_pathway_builder(), 10)

        assert results["B_g1"] == [0.0] * 11
        # Only g2 moves from A to B. `k1 * A` uses the A total, 900, so
        # each g2 compartment moves 0.01 * 900 = 9 in the first step.
        assert results["A_g1"][1] == pytest.approx(450.0)
        assert results["A_g2_t2"][1] == pytest.approx(225.0 - 9.0)
        assert results["B_g2_t2"][1] == pytest.approx(25.0 + 9.0)


class TestCompartmentFractionValidation:
    def test_model_rejects_an_unknown_compartment_name(self) -> None:
        model = build(override_builder())

        with pytest.raises(ValueError, match="Unknown compartment 'A_missing'"):
            model.structure.update_compartment_fractions(
                model.process, {"A_missing": 0.1}
            )

    def test_an_update_replaces_a_fraction_and_appends_a_new_one(self) -> None:
        model = build(
            a_to_b_builder(
                fractions={"A": 0.6, "B": 0.4}, compartment_fractions={"A": 0.1}
            )
        )

        model.structure.update_compartment_fractions(
            model.process, {"A": 0.5, "B": 0.2}
        )

        assert _overrides(model) == {"A": 0.5, "B": 0.2}

    def test_an_out_of_range_update_is_rejected(self) -> None:
        model = build(a_to_b_builder())

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            model.structure.update_compartment_fractions(model.process, {"A": 1.5})

    def test_model_rejects_over_allocation(self) -> None:
        model = build(override_builder(0.3))

        # Uniform A_g2 = 0.5, so an override of 0.9 on A_g1 totals 1.4.
        with pytest.raises(ValueError, match="exceed the total"):
            model.structure.update_compartment_fractions(model.process, {"A_g1": 0.9})

        assert _overrides(model) == {"A_g1": 0.3}

    def test_model_allows_overrides_within_the_total(self) -> None:
        model = build(override_builder(0.3))

        model.structure.update_compartment_fractions(model.process, {"A_g1": 0.4})

        assert _overrides(model) == pytest.approx({"A_g1": 0.4})

    def test_json_with_over_allocation_is_rejected(self) -> None:
        model = build(override_builder(0.3))
        payload = model.model_dump_json().replace('"A_g1":0.3', '"A_g1":0.9')

        with pytest.raises(ValidationError, match="exceed the total"):
            Model.model_validate_json(payload)

    def test_an_unknown_override_fails_when_the_simulation_is_built(self) -> None:
        model = build(override_builder())
        model.structure.compartment_fractions["A_missing"] = 0.0

        with pytest.raises(RuntimeError, match="A_missing"):
            Simulation(model)


class TestManyCompartmentFractions:
    def test_thousands_of_unique_names_still_validate(self) -> None:
        structure = Aggregate(
            total=1_000_000,
            bin_fractions={"A": 1.0},
            compartment_fractions={f"A_c{index}": 1e-6 for index in range(4000)},
        )

        assert len(structure.compartment_fractions) == 4000
