"""Per-compartment initial conditions: the type, the builder and the model."""

import pytest
from pydantic import ValidationError

from commol import Simulation
from commol.api.model_builder import CompartmentFractionDict, ModelBuilder
from commol.context import CompartmentFraction
from commol.context.initial_conditions import BinFraction, InitialConditions
from commol.context.model import Model
from tests.support import (
    a_to_b_builder,
    build,
    conditional_pathway_builder,
    override_builder,
    run,
)


def _all_in_g1(overrides: list[CompartmentFractionDict]) -> ModelBuilder:
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        stratifications={"group": {"g1": 1.0, "g2": 0.0}},
        compartment_fractions=overrides,
    )


def _overrides(model: Model) -> dict[str, float]:
    return {
        fraction.compartment: fraction.fraction
        for fraction in model.structure.initial_conditions.compartment_fractions
    }


class TestCompartmentFractionType:
    def test_is_exported_from_context(self) -> None:
        assert CompartmentFraction.__name__ == "CompartmentFraction"

    def test_rejects_fraction_above_one(self) -> None:
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            CompartmentFraction(compartment="A_g1", fraction=1.5)

    def test_rejects_negative_fraction(self) -> None:
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            CompartmentFraction(compartment="A_g1", fraction=-0.1)

    def test_rejects_duplicate_compartment_names(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate compartment fractions"):
            InitialConditions(
                population_size=1000,
                bin_fractions=[BinFraction(bin="A", fraction=1.0)],
                compartment_fractions=[
                    CompartmentFraction(compartment="A_g1", fraction=0.1),
                    CompartmentFraction(compartment="A_g1", fraction=0.2),
                ],
            )

    def test_defaults_to_empty_list(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[BinFraction(bin="A", fraction=1.0)],
        )
        assert conditions.compartment_fractions == []


class TestBuilderCompartmentFractions:
    def test_valid_override_is_stored(self) -> None:
        model = build(_all_in_g1([{"compartment": "A_g2", "fraction": 0.0}]))

        assert _overrides(model) == {"A_g2": 0.0}

    def test_unknown_compartment_is_rejected_by_the_engine_at_build(self) -> None:
        builder = _all_in_g1([{"compartment": "A_missing", "fraction": 0.0}])

        with pytest.raises(ValueError, match="Unknown compartment 'A_missing'"):
            build(builder)

    def test_duplicate_compartment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate compartment fraction"):
            _all_in_g1(
                [
                    {"compartment": "A_g2", "fraction": 0.0},
                    {"compartment": "A_g2", "fraction": 0.1},
                ]
            )

    def test_out_of_range_fraction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            _all_in_g1([{"compartment": "A_g2", "fraction": 1.5}])

    def test_over_allocation_is_rejected_by_the_engine_at_build(self) -> None:
        builder = _all_in_g1([{"compartment": "A_g2", "fraction": 0.5}])

        with pytest.raises(ValueError, match="exceed population_size"):
            build(builder)

    def test_empty_compartment_fractions_use_uniform_expansion(self) -> None:
        model = build(_all_in_g1([]))

        assert model.structure.initial_conditions.compartment_fractions == []

    def test_uncalibrated_bin_fractions_count_as_zero(self) -> None:
        model = build(
            a_to_b_builder(
                rate="k * A",
                fractions={"A": None, "B": 0.5},
                compartment_fractions=[{"compartment": "A", "fraction": 0.4}],
            )
        )

        assert _overrides(model) == {"A": 0.4}

    def test_uncalibrated_override_exceeding_population_is_rejected(self) -> None:
        builder = a_to_b_builder(
            rate="k * A",
            fractions={"A": None, "B": 0.5},
            compartment_fractions=[{"compartment": "A", "fraction": 1.0}],
        )

        with pytest.raises(ValueError, match="exceed population_size"):
            build(builder)


class TestOverrideInARun:
    def test_zero_override_keeps_the_other_stratum_empty(self) -> None:
        results = run(conditional_pathway_builder(), 10)

        assert results["IH_y60"] == [0.0] * 11
        # Only oe60 moves from S to IH. `beta * S` uses the S total, 900, so
        # each oe60 compartment moves 0.01 * 900 = 9 in the first step.
        assert results["S_y60"][1] == pytest.approx(450.0)
        assert results["S_oe60_hosp"][1] == pytest.approx(225.0 - 9.0)
        assert results["IH_oe60_hosp"][1] == pytest.approx(25.0 + 9.0)


class TestCompartmentFractionValidation:
    def test_model_rejects_an_unknown_compartment_name(self) -> None:
        model = build(override_builder())

        with pytest.raises(ValueError, match="Unknown compartment 'A_missing'"):
            model.structure.update_compartment_fractions({"A_missing": 0.1})

    def test_initial_conditions_upsert_without_the_expansion(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[
                BinFraction(bin="A", fraction=0.6),
                BinFraction(bin="B", fraction=0.4),
            ],
            compartment_fractions=[CompartmentFraction(compartment="A", fraction=0.1)],
        )

        conditions.update_compartment_fractions({"A": 0.5, "B": 0.2})

        assert [
            (fraction.compartment, fraction.fraction)
            for fraction in conditions.compartment_fractions
        ] == [("A", 0.5), ("B", 0.2)]

    def test_initial_conditions_reject_an_out_of_range_upsert(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[BinFraction(bin="A", fraction=1.0)],
        )

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            conditions.update_compartment_fractions({"A": 1.5})

    def test_model_rejects_over_allocation(self) -> None:
        model = build(override_builder(0.3))

        # Uniform A_g2 = 0.5, so an override of 0.9 on A_g1 totals 1.4.
        with pytest.raises(ValueError, match="exceed population_size"):
            model.structure.update_compartment_fractions({"A_g1": 0.9})

        assert _overrides(model) == {"A_g1": 0.3}

    def test_model_allows_overrides_within_the_population(self) -> None:
        model = build(override_builder(0.3))

        model.structure.update_compartment_fractions({"A_g1": 0.4})

        assert _overrides(model) == pytest.approx({"A_g1": 0.4})

    def test_json_with_over_allocation_is_rejected(self) -> None:
        model = build(override_builder(0.3))
        payload = model.model_dump_json().replace('"fraction":0.3}', '"fraction":0.9}')

        with pytest.raises(ValidationError, match="exceed population_size"):
            Model.model_validate_json(payload)

    def test_an_unknown_override_fails_when_the_simulation_is_built(self) -> None:
        model = build(override_builder())
        model.structure.initial_conditions.compartment_fractions.append(
            CompartmentFraction(compartment="A_missing", fraction=0.0)
        )

        with pytest.raises(RuntimeError, match="A_missing"):
            Simulation(model)


class TestManyCompartmentFractions:
    def test_thousands_of_unique_names_still_validate(self) -> None:
        conditions = InitialConditions(
            population_size=1_000_000,
            bin_fractions=[BinFraction(bin="A", fraction=1.0)],
            compartment_fractions=[
                CompartmentFraction(compartment=f"A_c{index}", fraction=1e-6)
                for index in range(4000)
            ],
        )

        assert len(conditions.compartment_fractions) == 4000

    def test_duplicates_are_still_detected_in_large_inputs(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate compartment fractions"):
            InitialConditions(
                population_size=1_000_000,
                bin_fractions=[BinFraction(bin="A", fraction=1.0)],
                compartment_fractions=[
                    CompartmentFraction(compartment="A_c0", fraction=1e-6),
                    *(
                        CompartmentFraction(compartment=f"A_c{index}", fraction=1e-6)
                        for index in range(1, 2000)
                    ),
                    CompartmentFraction(compartment="A_c0", fraction=1e-6),
                ],
            )
