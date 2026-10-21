"""A model kind that is not compartmental runs through every generic layer.

The test kind (`tests.support.kinds`) is written only against
`ModelStructure`: a simulation, a calibration, a report and a plot use it
without knowing it.
"""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.api.plotter import SimulationPlotter
from tests.support import assert_close
from tests.support.kinds import decay, decay_model
from tests.support.problems import bounded


def _problem(*, kind: str = "parameter") -> CalibrationProblem:
    """Observations of ``x`` with rate 0.3, and ``k`` to calibrate."""
    return CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=at, output="x", value=decay(10.0, 0.3, at))
            for at in (0.5, 1.0, 2.0, 4.0)
        ],
        parameters=[bounded("k", 0.0, 2.0, kind=kind, initial_guess=1.0)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=400, sd_tolerance=1e-12),
    )


class TestSimulation:
    def test_a_run_reports_the_values_of_the_kind(self) -> None:
        results = Simulation(decay_model()).run_at([0.0, 1.5, 3.0])

        assert_close(
            results["x"], [decay(10.0, 0.5, at) for at in (0, 1.5, 3)], rel=1e-12
        )

    def test_the_outputs_come_from_the_structure(self) -> None:
        simulation = Simulation(decay_model())

        assert simulation.simulation_outputs == ["x"]
        assert simulation.outputs_for("x") == ["x"]

    def test_an_uncalibrated_parameter_stops_a_run(self) -> None:
        with pytest.raises(ValueError, match="Parameters requiring calibration: k"):
            Simulation(decay_model(rate=None)).run(1)


class TestCalibration:
    def test_a_calibration_finds_the_rate(self) -> None:
        model = decay_model(rate=None)
        problem = _problem()

        result = Calibrator(Simulation(model), problem).run()
        model.apply_calibration_parameters(result, problem)

        assert result.best_parameters["k"] == pytest.approx(0.3, rel=1e-4)
        assert model.parameters[0].value == pytest.approx(0.3, rel=1e-4)

    def test_a_value_type_the_kind_does_not_accept_names_the_kind(self) -> None:
        with pytest.raises(ValueError, match="test model does not accept"):
            Calibrator(Simulation(decay_model()), _problem(kind="initial_condition"))


class TestReport:
    def test_the_report_has_the_sections_of_the_kind(self) -> None:
        model = decay_model(rate=0.3)

        report = model.applied_parameters({"k": 0.3})

        assert report["parameters"] == {"k": {"value": 0.3, "calibrated": True}}
        assert dict(report) == {
            "parameters": report["parameters"],
            "outputs": {
                "x": {"initial": 10.0, "rate_calibrated": True, "has_interval": False}
            },
        }


class TestPlot:
    def test_a_plot_names_the_series_as_the_kind_does(self) -> None:
        simulation = Simulation(decay_model())

        figure = SimulationPlotter(simulation, simulation.run(5)).plot_series()

        axis = figure.get_axes()[0]
        assert axis.get_title() == "Output x"
        assert axis.get_ylabel() == "unit"
