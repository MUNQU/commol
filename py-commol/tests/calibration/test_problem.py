"""The calibration problem: losses, normalization, and the whole workflow."""

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol._data.engine_json import engine_json
from commol._fit.result import CalibrationResult
from commol._model.model import Model
from commol.commol_rs import _commol_rs as commol_rs
from tests.support import abc_builder, bounded, build, observed

SEED = 42


class TestLosses:
    @pytest.mark.parametrize("loss_function", ["sse", "rmse", "mae"])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_loss_config_carries_normalization_flag(
        self, truth: Model, loss_function: str, normalize: bool
    ) -> None:
        """The problem's loss and normalize_observations reach the engine."""
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=0, output="B", value=10.0)],
            parameters=[bounded("k1")],
            loss_function=loss_function,
            normalize_observations=normalize,
            optimization_config=NelderMeadConfig(max_iterations=1, verbose=False),
        )

        Calibrator(Simulation(truth), problem)
        read = json.loads(
            commol_rs.json.as_read("CalibrationProblem", engine_json(problem))
        )

        assert read["loss_function"] == loss_function
        assert read["normalize_observations"] is normalize

    def test_normalization_changes_deterministic_loss(
        self, truth: Model, truth_run
    ) -> None:
        """Normalization divides each residual by its series' RMS."""
        # Biased targets: no parameter choice drives the residuals to zero,
        # otherwise both losses collapse to 0 and cannot be told apart.
        observations = [
            ObservedDataPoint(at=i, output="B", value=truth_run["B"][i] + 25.0)
            for i in range(100)
        ]

        def final_loss(normalize: bool) -> float:
            problem = CalibrationProblem(
                observed_data=observations,
                parameters=[bounded("k1"), bounded("k2")],
                loss_function="sse",
                normalize_observations=normalize,
                optimization_config=ParticleSwarmConfig(
                    max_iterations=50, verbose=False
                ),
                seed=SEED,
            )
            return Calibrator(Simulation(truth), problem).run().final_loss

        rms = math.sqrt(
            sum(point.value**2 for point in observations) / len(observations)
        )
        assert rms > 1.0
        assert final_loss(normalize=True) < final_loss(normalize=False)


class TestWorkflow:
    def test_calibrate_write_back_and_simulate(self) -> None:
        """
        The quick path of the guide: a model whose unknowns are None cannot
        run; calibration finds them; written back, the model runs again.
        """
        truth = Simulation(build(abc_builder(parameters={"k1": 0.1, "k2": 0.05}))).run(
            100
        )
        model = build(abc_builder(parameters={"k1": None, "k2": None}))
        with pytest.raises(ValueError, match="The model cannot run"):
            Simulation(model).run(100)
        problem = CalibrationProblem(
            observed_data=observed(truth, "B", range(100)),
            parameters=[bounded("k1"), bounded("k2")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1000, verbose=False),
        )

        result = Calibrator(Simulation(model), problem).run()
        model.update_parameters(result.best_parameters)
        rerun = Simulation(model).run(100)

        assert result.best_parameters == pytest.approx(
            {"k1": 0.1, "k2": 0.05}, abs=1e-5
        )
        assert rerun["B"] == pytest.approx(truth["B"], rel=1e-3)


class TestValidation:
    @pytest.mark.parametrize(
        ("bounds", "message"),
        [
            ({"min_bound": 1.0, "max_bound": 1.0}, "max_bound .* greater than"),
            (
                {"min_bound": 0.0, "max_bound": 1.0, "initial_guess": 2.0},
                "initial_guess .* between",
            ),
        ],
    )
    def test_a_parameter_needs_an_interval_that_holds_its_guess(
        self, bounds: dict[str, float], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            CalibrationParameter(id="k1", parameter_type="parameter", **bounds)

    def test_a_parameter_type_is_one_of_the_known_ones(self) -> None:
        with pytest.raises(
            ValidationError,
            match="Input should be 'parameter', 'initial_condition', 'factor' or",
        ):
            CalibrationParameter(
                id="k1", parameter_type="rate", min_bound=0.0, max_bound=1.0
            )

    def test_a_loss_function_is_one_of_the_known_ones(self) -> None:
        with pytest.raises(
            ValidationError, match="Input should be .sse., .rmse. or .mae."
        ):
            CalibrationProblem(
                observed_data=[ObservedDataPoint(at=0, output="B", value=1.0)],
                parameters=[bounded("k1")],
                loss_function="mse",
                optimization_config=NelderMeadConfig(max_iterations=1),
            )

    def test_a_parameter_is_fitted_once(self) -> None:
        with pytest.raises(ValidationError, match=r"Duplicate parameter IDs.*'k1'"):
            CalibrationProblem(
                observed_data=[ObservedDataPoint(at=0, output="B", value=1.0)],
                parameters=[bounded("k1"), bounded("k1")],
                loss_function="sse",
                optimization_config=NelderMeadConfig(max_iterations=1),
            )


class TestResult:
    def test_a_result_prints_its_outcome(self) -> None:
        result = CalibrationResult(
            best_parameters={"k1": 0.1},
            final_loss=0.25,
            iterations=12,
            converged=True,
            termination_reason="tolerance",
            seed=7,
        )

        assert str(result) == (
            "CalibrationResult(\n"
            "  converged=True,\n"
            "  final_loss=0.250000,\n"
            "  iterations=12,\n"
            "  best_parameters={'k1': 0.1},\n"
            "  termination_reason='tolerance',\n"
            "  starts=0,\n"
            "  failed=0,\n"
            "  seed=7\n"
            ")"
        )

    def test_a_result_records_the_seed_of_its_calibration(
        self, truth: Model, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        def calibrator(seed: int | None) -> Calibrator:
            problem = CalibrationProblem(
                observed_data=[ObservedDataPoint(at=1, output="B", value=1.0)],
                parameters=[bounded("k1")],
                loss_function="sse",
                optimization_config=NelderMeadConfig(max_iterations=20),
                seed=seed,
            )
            return Calibrator(Simulation(truth), problem)

        with caplog.at_level("INFO", logger="commol"):
            drawn = calibrator(None)
        result = drawn.run()

        assert result.seed == drawn.seed
        assert f"drew the seed {drawn.seed}" in caplog.text
        assert calibrator(result.seed).run() == result
        assert calibrator(SEED).run().seed == SEED
        result.to_json(tmp_path / "fit.json")
        read = CalibrationResult.from_json(tmp_path / "fit.json")
        assert calibrator(read.seed).run() == read


class TestProblemAgainstTheModel:
    @staticmethod
    def _calibrator(
        truth: Model,
        parameters: list[CalibrationParameter],
        observed_data: list[ObservedDataPoint] | None = None,
    ) -> Calibrator:
        problem = CalibrationProblem(
            observed_data=observed_data
            or [ObservedDataPoint(at=0, output="B", value=1.0)],
            parameters=parameters,
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        return Calibrator(Simulation(truth), problem)

    def test_a_calibrator_states_the_size_of_its_problem(self, truth: Model) -> None:
        observations = [
            ObservedDataPoint(at=at, output="B", value=1.0) for at in (0, 5, 10)
        ]

        calibrator = self._calibrator(
            truth, [bounded("k1"), bounded("k2")], observations
        )

        assert calibrator.num_parameters == 2
        assert calibrator.num_observations == 3
        assert calibrator.parameter_names == ["k1", "k2"]

    def test_the_engine_rejects_an_unknown_calibration_parameter(
        self, truth: Model
    ) -> None:
        with pytest.raises(
            ValueError,
            match="^Parameter 'k9' is not in the model; the parameters that can be "
            "set are k1, k2$",
        ):
            self._calibrator(truth, [bounded("k9")])

    def test_the_engine_rejects_a_formula_calibration_parameter(self) -> None:
        model = build(abc_builder(parameters={"k1": 0.1, "k2": "2 * k1"}))

        with pytest.raises(
            ValueError, match="^Parameter 'k2' cannot be set: its formula sets it"
        ):
            self._calibrator(model, [bounded("k2")])

    @pytest.mark.parametrize(
        "engine",
        [
            "DifferenceEquations",
            "DifferentialEquations",
            "StochasticSteps",
            "StochasticEvents",
        ],
    )
    def test_a_value_that_the_calibration_leaves_missing_is_refused_at_once(
        self, engine: str
    ) -> None:
        builder = abc_builder(parameters={"k1": 0.3, "k2": 0.1, "unused": None})
        model = builder.build(builder.structure, engine=engine)

        with pytest.raises(
            ValueError,
            match=r'^The model cannot run: the parameters \["unused"\] have no value',
        ):
            self._calibrator(model, [bounded("k1")])

    def test_a_factor_has_positive_bounds(self, truth: Model) -> None:
        with pytest.raises(ValueError, match="'f' must have positive bounds"):
            self._calibrator(truth, [bounded("f", 0.0, 1.0, kind="factor")])

    def test_an_unknown_observed_output_gets_the_message_of_the_engine(
        self, truth: Model
    ) -> None:
        with pytest.raises(
            ValueError,
            match="^Invalid observed output name 'X' for observation 'series1' "
            r"\(available: A, B, C\)$",
        ):
            self._calibrator(
                truth,
                [bounded("k1")],
                [ObservedDataPoint(at=0, output="X", label="series1", value=1.0)],
            )
