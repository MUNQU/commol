"""The calibration problem: losses, normalization, and the whole workflow."""

import math

import pytest

from pydantic import ValidationError

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    CalibrationResult,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model
from tests.support import bounded, build, observed, abc_builder

SEED = 42


class TestLosses:
    @pytest.mark.parametrize("loss_function", ["sse", "rmse", "mae"])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_loss_config_carries_normalization_flag(
        self, truth: Model, loss_function: str, normalize: bool
    ) -> None:
        """The problem's normalize_observations must reach the Rust LossConfig."""
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=0, output="B", value=10.0)],
            parameters=[bounded("k1")],
            loss_function=loss_function,
            normalize_observations=normalize,
            optimization_config=NelderMeadConfig(max_iterations=1, verbose=False),
        )

        loss_config = Calibrator(Simulation(truth), problem)._build_loss_config()

        assert loss_config.normalize_observations is normalize

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
        with pytest.raises(ValueError, match="Cannot run Simulation"):
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
        with pytest.raises(ValidationError, match="Invalid parameter_type: rate"):
            CalibrationParameter(
                id="k1", parameter_type="rate", min_bound=0.0, max_bound=1.0
            )

    def test_a_loss_function_is_one_of_the_known_ones(self) -> None:
        with pytest.raises(ValidationError, match="Invalid loss_function: mse"):
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
        self, truth: Model, caplog: pytest.LogCaptureFixture
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

        drawn = calibrator(None)
        with caplog.at_level("INFO", logger="commol"):
            result = drawn.run()

        assert result.seed == drawn.seed
        assert f"drew the seed {drawn.seed}" in caplog.text
        assert calibrator(SEED).run().seed == SEED
        assert CalibrationResult.model_validate_json(result.model_dump_json()) == result


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

    def test_a_parameter_is_one_of_the_model(self, truth: Model) -> None:
        with pytest.raises(ValueError, match=r"'k9' not found .*\['k1', 'k2'\]"):
            self._calibrator(truth, [bounded("k9")])

    def test_a_factor_has_positive_bounds(self, truth: Model) -> None:
        with pytest.raises(ValueError, match="'f' must have positive bounds"):
            self._calibrator(truth, [bounded("f", 0.0, 1.0, kind="factor")])

    def test_an_observation_is_of_an_output_of_the_model(self, truth: Model) -> None:
        with pytest.raises(ValueError, match=r"\['D'\] for observation 'D' not found"):
            self._calibrator(
                truth,
                [bounded("k1")],
                [ObservedDataPoint(at=0, output="D", value=1.0)],
            )
