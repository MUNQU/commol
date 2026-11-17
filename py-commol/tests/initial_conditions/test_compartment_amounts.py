"""Per-compartment initial conditions: the type, the builder and the model."""

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from commol import Aggregate, Simulation
from commol._model.engine_state import calibrated_initial_state
from commol._model.model import Model
from tests.support import (
    FactoryBuilder,
    a_to_b_builder,
    build,
    conditional_pathway_builder,
    initial_condition_problem,
    initial_state,
    override_builder,
    run,
)


def _all_in_g1(overrides: Mapping[str, float]) -> FactoryBuilder:
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        stratifications={"group": {"g1": 1.0, "g2": 0.0}},
        compartment_amounts=overrides,
    )


def _overrides(model: Model) -> dict[str, float]:
    return dict(model.structure.compartment_amounts)


def _a_left_none_with_a_g1() -> Model:
    """A left None and B = 1000, split in halves, with the override A_g1 = 500."""
    return build(
        a_to_b_builder(
            amounts={"A": None, "B": 1000},
            stratifications={"group": {"g1": 0.5, "g2": 0.5}},
            compartment_amounts={"A_g1": 500},
        )
    )


class TestBuilderCompartmentAmounts:
    def test_unknown_compartment_is_rejected_by_the_engine_at_build(self) -> None:
        builder = _all_in_g1({"A_missing": 0.0})

        with pytest.raises(ValueError, match="Unknown compartment 'A_missing'"):
            build(builder)

    def test_an_override_sets_its_amount_whatever_its_bin_holds(self) -> None:
        model = build(_all_in_g1({"A_g2": 5000}))

        assert initial_state(model)["A_g2"] == 5000


class TestOverrideOnABinLeftNone:
    def test_an_override_gives_its_compartment_a_value_on_a_bin_left_none(self) -> None:
        # A_g1 = 500 states that amount of A, left None: a run names only
        # A_g2, and the remainder A keeps the 500. A_g2 holds a 0 no one
        # stated, so the remainder splits the 500 by the fractions.
        model = _a_left_none_with_a_g1()

        kept = calibrated_initial_state(model, [], [], "A").state
        with pytest.raises(ValueError) as error:
            Simulation(model).run(1)

        assert kept == {"A_g1": 250, "A_g2": 250, "B_g1": 500, "B_g2": 500}
        assert str(error.value).startswith(
            'The model cannot run: the amounts of the compartments ["A_g2"] have no '
            "value."
        )

    def test_a_bin_left_none_starts_from_an_override_on_each_compartment(
        self,
    ) -> None:
        results = run(
            a_to_b_builder(
                rate="k * A",
                amounts={"A": None, "B": 500},
                compartment_amounts={"A": 400},
            ),
            1,
        )

        assert [results["A"][0], results["B"][0]] == [400, 500]

    def test_a_calibration_of_the_other_compartments_is_written_back(self) -> None:
        # The override gives A_g1 its value, so a value of A_g2 completes A.
        model = _a_left_none_with_a_g1()

        model.apply_calibration_parameters(
            {"A_g2": 200}, initial_condition_problem("A_g2")
        )
        results = run(model, 1)

        assert {name: series[0] for name, series in results.items()} == {
            "A_g1": 500,
            "A_g2": 200,
            "B_g1": 500,
            "B_g2": 500,
        }


class TestOverrideInARun:
    def test_zero_override_keeps_the_other_stratum_empty(self) -> None:
        results = run(conditional_pathway_builder(), 10)

        assert results["B_g1"] == [0.0] * 11
        # Only g2 moves from A to B. `k1 * A` uses the A total, 900, so
        # each g2 compartment moves 0.01 * 900 = 9 in the first step.
        assert results["A_g1"][1] == pytest.approx(450.0)
        assert results["A_g2_t2"][1] == pytest.approx(225.0 - 9.0)
        assert results["B_g2_t2"][1] == pytest.approx(25.0 + 9.0)


class TestCompartmentAmountValidation:
    def test_model_rejects_an_unknown_compartment_name(self) -> None:
        model = build(override_builder())

        with pytest.raises(ValueError, match="Unknown compartment 'A_missing'"):
            model.structure.update_compartment_amounts(
                model.process, {"A_missing": 100}
            )

        assert _overrides(model) == {"A_g1": 300}

    def test_an_update_replaces_an_amount_and_appends_a_new_one(self) -> None:
        model = build(
            a_to_b_builder(amounts={"A": 600, "B": 400}, compartment_amounts={"A": 100})
        )

        model.structure.update_compartment_amounts(model.process, {"A": 500, "B": 200})

        assert _overrides(model) == {"A": 500, "B": 200}

    def test_a_negative_update_is_rejected(self) -> None:
        model = build(a_to_b_builder())

        with pytest.raises(ValueError) as error:
            model.structure.update_compartment_amounts(model.process, {"A": -1.5})

        assert str(error.value) == (
            "Compartment 'A': initial amount is a finite number of at least 0, got -1.5"
        )

    def test_json_with_an_unknown_override_is_rejected(self) -> None:
        model = build(override_builder())
        payload = model.model_dump_json().replace('"A_g1":300', '"A_x":300')

        with pytest.raises(ValidationError, match="Unknown compartment 'A_x'"):
            Model.model_validate_json(payload)

    def test_an_unknown_override_fails_when_the_simulation_is_built(self) -> None:
        model = build(override_builder())
        model.structure.compartment_amounts["A_missing"] = 0.0

        with pytest.raises(RuntimeError, match="A_missing"):
            Simulation(model)


class TestManyCompartmentAmounts:
    def test_thousands_of_unique_names_still_validate(self) -> None:
        structure = Aggregate(
            bin_amounts={"A": 1_000_000},
            compartment_amounts={f"A_c{index}": 1.0 for index in range(4000)},
        )

        assert len(structure.compartment_amounts) == 4000
