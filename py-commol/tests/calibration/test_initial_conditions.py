"""Calibrating initial conditions: which ids name them, and fitting them."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model
from tests.support import (
    a_to_b_builder,
    assert_accepted,
    bounded,
    build,
    observed,
    abc_builder,
)

SEED = 42


def _model() -> Model:
    """A binary and a ternary stratification."""
    return build(
        a_to_b_builder(
            name="Ids",
            fractions={"A": 0.9, "B": 0.1},
            stratifications={
                "group": {"g1": 0.5, "g2": 0.5},
                "age": {"y": 1 / 3, "m": 1 / 3, "o": 1 / 3},
            },
        )
    )


def _problem(parameter_id: str) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="B_g1_y", value=10.0)],
        parameters=[bounded(parameter_id, 0.0, 0.5, kind="initial_condition")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=5),
    )


class TestIds:
    """Initial-condition ids are resolved by the engine when a calibrator is built."""

    @pytest.mark.parametrize("parameter_id", ["y", "age:y"])
    def test_a_category_of_a_non_binary_stratification_is_rejected(
        self, parameter_id: str
    ) -> None:
        with pytest.raises(ValueError, match="needs a binary stratification"):
            Calibrator(Simulation(_model()), _problem(parameter_id))

    @pytest.mark.parametrize("parameter_id", ["B", "B_g1", "B_g1_y", "g1", "group:g1"])
    def test_every_documented_id_kind_is_accepted(self, parameter_id: str) -> None:
        assert_accepted(Calibrator, Simulation(_model()), _problem(parameter_id))

    def test_an_unknown_id_is_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="Invalid bin, compartment or category 'X'"
        ):
            Calibrator(Simulation(_model()), _problem("X"))

    def test_an_unknown_bin_of_an_unstratified_model_is_rejected(self) -> None:
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=10, output="B", value=50.0)],
            parameters=[bounded("X", 0.0, 100.0, kind="initial_condition")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=100),
        )

        with pytest.raises(ValueError, match="Invalid bin, compartment or category .X"):
            Calibrator(Simulation(build(abc_builder())), problem)


def _sir(i_fraction: float | None, k1: float | None = 0.3) -> Model:
    return build(
        abc_builder(
            parameters={"k1": k1, "k2": 0.1},
            fractions={"A": 0.98, "B": i_fraction, "C": 0.0},
        )
    )


@pytest.fixture
def b_every_5_steps() -> list[ObservedDataPoint]:
    """B of ``abc_builder`` with 2 % in B at the start, at steps 0, 5, …, 45."""
    return observed(Simulation(_sir(0.02)).run(50), "B", range(0, 50, 5))


class TestFitting:
    def test_an_initial_fraction_is_recovered(self, b_every_5_steps) -> None:
        model = _sir(None)
        problem = CalibrationProblem(
            observed_data=b_every_5_steps,
            parameters=[
                bounded("B", 0.0, 0.1, kind="initial_condition", initial_guess=0.01)
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(
                max_iterations=5000, sd_tolerance=1e-9, verbose=False
            ),
        )

        result = Calibrator(Simulation(model), problem).run()

        assert result.converged
        assert result.best_parameters["B"] == pytest.approx(0.02, rel=1e-9)

    def test_a_parameter_and_an_initial_fraction_together(
        self, b_every_5_steps
    ) -> None:
        config = (
            ParticleSwarmConfig(num_particles=40, max_iterations=1000, verbose=False)
            .acceleration(
                "time_varying",
                c1_initial=2.5,
                c1_final=0.5,
                c2_initial=0.5,
                c2_final=2.5,
            )
            .velocity(clamp_factor=0.2)
            .mutation(
                "gaussian", scale=0.1, probability=0.05, application="global_best"
            )
        )
        problem = CalibrationProblem(
            observed_data=b_every_5_steps,
            parameters=[
                bounded("k1"),
                bounded("B", 0.0, 0.1, kind="initial_condition"),
            ],
            loss_function="sse",
            optimization_config=config,
            seed=SEED,
        )

        result = Calibrator(Simulation(_sir(None, k1=None)), problem).run()

        assert result.best_parameters == pytest.approx(
            {"k1": 0.3, "B": 0.02}, abs=0.001
        )
