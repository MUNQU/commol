"""Initial-condition ids are resolved by the engine when a calibrator is built."""

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.api.probabilistic_calibrator import ProbabilisticCalibrator
from commol.context.model import Model


def _model() -> Model:
    def fractions(stratification: str, categories: list[str]) -> dict:
        return {
            "stratification": stratification,
            "fractions": [
                {"category": category, "fraction": 1 / len(categories)}
                for category in categories
            ],
        }

    return (
        ModelBuilder(name="Ids")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=0.1)
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_stratification(id="age", categories=["y", "m", "o"])
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 0.9},
                {"bin": "B", "fraction": 0.1},
            ],
            stratification_fractions=[
                fractions("group", ["g1", "g2"]),
                fractions("age", ["y", "m", "o"]),
            ],
        )
        .build(typology="DifferenceEquations")
    )


def _problem(parameter_id: str) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(step=1, compartment="B_g1_y", value=10.0)],
        parameters=[
            CalibrationParameter(
                id=parameter_id,
                parameter_type="initial_condition",
                min_bound=0.0,
                max_bound=0.5,
            )
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=5),
    )


@pytest.mark.parametrize("parameter_id", ["y", "age:y"])
def test_a_category_of_a_non_binary_stratification_is_rejected(
    parameter_id: str,
) -> None:
    with pytest.raises(ValueError, match="needs a binary stratification"):
        Calibrator(Simulation(_model()), _problem(parameter_id))


def test_the_probabilistic_calibrator_gives_the_same_message() -> None:
    with pytest.raises(ValueError, match="needs a binary stratification"):
        ProbabilisticCalibrator(Simulation(_model()), _problem("y"))


@pytest.mark.parametrize("parameter_id", ["B", "B_g1", "B_g1_y", "g1", "group:g1"])
def test_every_documented_id_kind_is_accepted(parameter_id: str) -> None:
    Calibrator(Simulation(_model()), _problem(parameter_id))
    ProbabilisticCalibrator(Simulation(_model()), _problem(parameter_id))


def test_an_unknown_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid bin, compartment or category 'X'"):
        Calibrator(Simulation(_model()), _problem("X"))
