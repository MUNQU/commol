"""Scored amounts: the start a calibration scored where the bins and the
fractions build it only to its last bits, and how an edit makes one stale."""

import json
import math
import random
from pathlib import Path

import pytest

from commol import (
    Aggregate,
    CalibrationProblem,
    Model,
    NelderMeadConfig,
    ObservedDataPoint,
    ScoredAmount,
    Simulation,
)
from commol._model.engine_state import calibrated_initial_state
from commol.engines import StochasticStepsSettings
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    build_stochastic,
    initial_condition_problem,
    initial_state,
)

SPLIT = {"g1": 0.7, "g2": 0.2, "g3": 0.1}

# B = 1.5 split 0.7, 0.2, 0.1: the calibration scores B_g1 = 1.05, and the
# bin and its split build 1.0499999999999998.
BUILT = 1.05 - 2**-52
SCORED = 1.05


def _written(amount: float = 1.5, model: Model | None = None) -> Model:
    """The model of `SPLIT` with the calibrated ``B = amount`` written back,
    the remainder in A."""
    model = model or build(a_to_b_builder(stratifications={"group": SPLIT}))
    model.apply_calibration_parameters(
        {"B": amount}, initial_condition_problem("B", remainder="A")
    )
    return model


def _bits(value: float) -> str:
    return value.hex()


def test_a_write_back_records_the_last_bits_it_corrects() -> None:
    model = _written()

    assert model.structure.compartment_amounts == {}
    assert model.structure.scored_amounts == {
        "B_g1": ScoredAmount(built=BUILT, scored=SCORED)
    }
    assert initial_state(model)["B_g1"] == SCORED
    # The report lists the bins, and no row for a scored amount.
    report = model.applied_parameters(
        {"B": 1.5}, initial_condition_problem("B", remainder="A")
    )
    assert list(report["initial_amounts"]) == ["A", "B"]


def test_an_edit_that_rebuilds_the_same_bits_keeps_only_a_last_bit_scored_amount() -> (
    None
):
    model = _written()
    calibrated = build(a_to_b_builder(stratifications={"group": SPLIT}))
    calibrated.apply_calibration_parameters(
        {"B_g1": 0.9}, initial_condition_problem("B_g1")
    )

    model.structure.update_bin_amounts({"B": 3.0})
    stale = initial_state(model)["B_g1"]
    model.structure.update_bin_amounts({"B": 1.5})
    calibrated.structure.update_bin_amounts({"B": 3.0})

    # The edit makes the scored amount stale, and B back at 1.5 builds its
    # bits again, so the scored amount holds: it moves B_g1 by one unit in the last
    # place. A value set by a compartment id is an override, a declaration
    # that every edit keeps, never a scored amount a coincidence brings back.
    assert stale == 3.0 * 0.7
    assert initial_state(model)["B_g1"] == SCORED
    assert calibrated.structure.compartment_amounts == {"B_g1": 0.9}
    assert calibrated.structure.scored_amounts == {}
    assert initial_state(calibrated)["B_g1"] == 0.9


def test_a_scored_amount_never_gives_a_value_to_a_bin_left_none() -> None:
    model = _written(-0.0)

    model.structure.bin_amounts["B"] = None

    # B = -0.0 scores B_g1 = -0.0, which the bin of 0.0 builds as 0.0. A bin
    # left None also builds 0.0, but the scored amount is stale there.
    amount = model.structure.scored_amounts["B_g1"]
    assert (amount.built.hex(), amount.scored.hex()) == ("0x0.0p+0", "-0x0.0p+0")
    assert math.copysign(1.0, initial_state(model)["B_g1"]) == 1.0


def test_a_declared_override_beats_a_scored_amount_of_its_compartment() -> None:
    model = _written()

    model.structure.compartment_amounts = {"B_g1": BUILT}

    assert _bits(initial_state(model)["B_g1"]) == _bits(BUILT)


def test_an_override_through_the_method_drops_the_scored_amount_it_makes_stale() -> (
    None
):
    model = _written(0.37)
    corrected = set(model.structure.scored_amounts)

    model.structure.update_compartment_amounts(model.process, {"A_g1": 600.0})
    kept = dict(model.structure.scored_amounts)
    model.structure.update_compartment_amounts(model.process, {"B_g2": 1.0})

    assert corrected == {"B_g2", "B_g3"}
    assert set(kept) == corrected
    assert set(model.structure.scored_amounts) == {"B_g3"}
    assert initial_state(model)["B_g2"] == 1.0


def test_an_update_of_a_bin_keeps_its_overrides_and_drops_its_scored_amounts() -> None:
    declared = build(
        a_to_b_builder(
            amounts={"A": 988.0, "B": 12.0},
            stratifications={"group": {"g1": 0.5, "g2": 0.5}},
            compartment_amounts={"B_g1": 8.4},
        )
    )
    written = _written()

    declared.structure.update_bin_amounts({"B": 20.0})
    written.structure.update_bin_amounts({"B": 20.0})

    # The override is a declaration and stays, so B sums to 20 only where no
    # override holds; the scored amount of the write-back stops applying.
    state = initial_state(declared)
    assert (state["B_g1"], state["B_g2"]) == (8.4, 10.0)
    assert initial_state(written)["B_g1"] == 20.0 * 0.7


def test_a_new_split_makes_the_scored_amounts_of_its_compartments_stale() -> None:
    model = _written()

    model.structure.update_stratification_fractions({"g1": 0.5, "g2": 0.3})

    state = model.structure.initial_state(model.process)
    assert "B_g1" in model.structure.scored_amounts
    assert not any(compartment.scored for compartment in state)
    assert initial_state(model)["B_g1"] == 1.5 * 0.5


def test_both_numbers_of_a_scored_amount_read_back_bit_for_bit(tmp_path: Path) -> None:
    model = _written()
    model.to_json(tmp_path / "model.json")
    plain = build(a_to_b_builder(stratifications={"group": SPLIT}))
    plain.to_json(tmp_path / "plain.json")

    loaded = Model.from_json(tmp_path / "model.json")

    (amount,) = loaded.structure.scored_amounts.values()
    assert (_bits(amount.built), _bits(amount.scored)) == (_bits(BUILT), _bits(SCORED))
    assert _bits(initial_state(loaded)["B_g1"]) == _bits(SCORED)
    # A model with no scored amount writes no field, so an older commol reads it.
    structure = json.loads((tmp_path / "plain.json").read_text())["structure"]
    assert "scored_amounts" not in structure


def test_a_scored_amount_further_than_its_last_bits_is_refused() -> None:
    with pytest.raises(ValueError, match="at most 64 units in the last place"):
        Aggregate(
            bin_amounts={"A": 1000.0, "B": 0.0},
            scored_amounts={"A_g1": ScoredAmount(built=700.0, scored=300.0)},
        )


def test_a_stochastic_run_after_a_new_split_starts_from_the_new_split() -> None:
    settings = StochasticStepsSettings(rounding="random")
    builder = a_to_b_builder(stratifications={"group": SPLIT})
    model = _written(1.5, build_stochastic(builder, "StochasticSteps", settings))
    corrected = set(model.structure.scored_amounts)

    model.structure.update_stratification_fractions({"g1": 0.5, "g2": 0.3})
    fresh = model.model_copy(deep=True)
    fresh.structure.scored_amounts = {}

    # The scored amount of B_g1 is stale under the new split, so each
    # replicate starts as the same model with no scored amount does.
    state = model.structure.initial_state(model.process)
    assert corrected == {"B_g1"}
    assert not any(compartment.scored for compartment in state)
    assert initial_state(model)["B_g1"] == 1.5 * 0.5
    outputs = [f"{bin_id}_{group}" for bin_id in ("A", "B") for group in SPLIT]
    runs, expected = (
        Simulation(runnable).replicates(3, seed=7).run_at([0.0, 4.0])
        for runnable in (model, fresh)
    )
    assert [[run[output] for output in outputs] for run in runs] == [
        [run[output] for output in outputs] for run in expected
    ]


def test_a_write_back_of_four_stratifications_writes_no_override() -> None:
    builder = a_to_b_builder()
    rng = random.Random(1)
    fractions = {}
    for index in range(4):
        categories = [f"s{index}c{c}" for c in range(3)]
        builder.add_stratification(id=f"s{index}", categories=categories)
        weights = [rng.random() for _ in categories]
        shares = [weight / sum(weights) for weight in weights]
        shares[-1] = 1.0 - shares[0] - shares[1]
        fractions[f"s{index}"] = dict(zip(categories, shares, strict=True))
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": 1000.0, "B": 0.0}, stratification_fractions=fractions
        )
    )
    model = build(builder)
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=0.0, output="B", value=1.0)],
        parameters=[bounded("B", 0.0, 500.0, kind="initial_condition")],
        initial_condition_remainder="A",
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )
    scored = calibrated_initial_state(model, problem.parameters, [123.456], "A")

    model.apply_calibration_parameters({"B": 123.456}, problem)

    # The bins and the splits rebuild most of the 162 compartments only to
    # their last bits: every one of them is a scored amount, none an override.
    assert model.structure.compartment_amounts == {}
    assert len(model.structure.scored_amounts) > 81
    assert initial_state(model) == scored.state
