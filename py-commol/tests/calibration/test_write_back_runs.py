"""A written-back model on a deterministic engine runs as calibration scored it."""

import math
from collections.abc import Callable
from pathlib import Path

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol._model.model import Model
from tests.support import a_to_b_builder, bounded, build, build_ode, initial_state

SPLIT = {"g1": 0.7, "g2": 0.2, "g3": 0.1}
OUTPUTS = [f"{bin_id}_{group}" for bin_id in ("A", "B") for group in SPLIT]
POSITIONS = [0.0, 3.0, 5.0]


def _written_back(
    build_model: Callable[..., Model], rate: str, amount: float, tmp_path: Path
) -> tuple[list[list[float]], Model, Model]:
    """The outputs a calibration scores for ``B = amount``, the model it writes
    back, and that model saved and loaded."""
    model = build_model(a_to_b_builder(stratifications={"group": SPLIT}, rate=rate))
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=at, output=output, value=1.0)
            for at in POSITIONS
            for output in OUTPUTS
        ],
        parameters=[bounded("B", 0.0, 50.0, kind="initial_condition")],
        initial_condition_remainder="A",
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )
    calibrator = Calibrator(Simulation(model), problem)
    (scored,) = calibrator._engine_problem.predict_observations([[amount]])

    model.apply_calibration_parameters({"B": amount}, problem)
    model.to_json(tmp_path / "model.json")
    return scored, model, Model.from_json(tmp_path / "model.json")


def _outputs(model: Model) -> list[float]:
    run = Simulation(model).run_at(POSITIONS)
    return [run[output][index] for index in range(len(POSITIONS)) for output in OUTPUTS]


@pytest.mark.parametrize("build_model", [build, build_ode], ids=["difference", "ode"])
def test_a_deterministic_write_back_runs_as_the_calibration_scored_it(
    build_model: Callable[..., Model], tmp_path: Path
) -> None:
    scored, written, loaded = _written_back(
        build_model, "if(B_g1 >= 1.05, 0.2, 0.01)", 1.5, tmp_path
    )

    # The calibration scores B_g1 = 1.05, at the step of the rate, so B_g2 is
    # 1.30 at 5; B = 1.5 split by 0.7 rebuilds 1.0499999999999998, below the
    # step, which gives 1.11 on the difference engine and outputs off in their
    # last bits on the ODE engine. The model, saved and loaded or not, takes
    # the step and gives every scored output bit for bit.
    b_g2_at_5 = 2 * len(OUTPUTS) + OUTPUTS.index("B_g2")
    for runnable in (written, loaded):
        outputs = _outputs(runnable)
        assert outputs[b_g2_at_5] == pytest.approx(1.30)
        assert [outputs] == scored


def test_a_write_back_keeps_the_sign_of_a_scored_zero(tmp_path: Path) -> None:
    scored, written, loaded = _written_back(
        build, "if(1 / B_g1 < 0, 0.5, 0.01)", -0.0, tmp_path
    )

    # B = -0.0 scores B_g1 = -0.0, whose inverse is below 0, so A_g1 loses
    # half of itself in the first step: 699.46 at 5, not 699.95. The bin
    # rebuilds 0.0, equal to -0.0 but not the same number.
    a_g1_at_5 = 2 * len(OUTPUTS) + OUTPUTS.index("A_g1")
    assert written.structure.compartment_amounts == {}
    assert {
        name: (amount.built.hex(), amount.scored.hex())
        for name, amount in written.structure.scored_amounts.items()
    } == {f"B_{g}": ("0x0.0p+0", "-0x0.0p+0") for g in SPLIT}
    for runnable in (written, loaded):
        assert math.copysign(1.0, initial_state(runnable)["B_g1"]) == -1.0
        outputs = _outputs(runnable)
        assert outputs[a_g1_at_5] == pytest.approx(699.46)
        assert [outputs] == scored


@pytest.mark.parametrize("build_model", [build, build_ode], ids=["difference", "ode"])
def test_a_compartment_the_user_overrides_starts_as_the_calibration_scored_it(
    build_model: Callable[..., Model],
) -> None:
    builder = a_to_b_builder(
        amounts={"A": 1000.0, "B": 1.5},
        stratifications={"s": {"x": 0.5, "y": 0.5}},
        compartment_amounts={"B_x": 0.45, "B_y": 1.05},
        rate="if(B_x < 0.45, 0.5, 0.01)",
    )
    model = build_model(builder)
    outputs = ["A_x", "A_y", "B_x", "B_y"]
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=at, output=output, value=1.0)
            for at in (0.0, 5.0)
            for output in outputs
        ],
        parameters=[bounded("x", 0.0, 1.0, kind="initial_condition")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )
    calibrator = Calibrator(Simulation(model), problem)
    (scored,) = calibrator._engine_problem.predict_observations([[0.3]])

    model.apply_calibration_parameters({"x": 0.3}, problem)

    # The split 0.3 of the declared 1.5 scores B_x = 0.44999999999999996,
    # below the step of the rate. B_x has an override, which a scored amount
    # never beats, so the write-back writes the override with the scored
    # amount, and the model takes the step as the calibration did.
    run = Simulation(model).run_at([0.0, 5.0])
    b_x = scored[0][outputs.index("B_x")]
    assert b_x == 0.44999999999999996
    assert model.structure.compartment_amounts["B_x"] == b_x
    assert model.structure.scored_amounts == {}
    assert [[run[output][index] for index in range(2) for output in outputs]] == scored
