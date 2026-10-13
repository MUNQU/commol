"""Constraints keep the fitted parameters and outputs inside stated limits."""

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationConstraint,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model
from tests.support import bounded, build, observed, sir_builder

SEED = 42


def _fit(
    truth: Model,
    truth_run: dict[str, list[float]],
    constraints: list[CalibrationConstraint],
    *,
    gamma_bound: float | None = 1.0,
    config: NelderMeadConfig | ParticleSwarmConfig | None = None,
) -> dict[str, float]:
    """Fit beta, and gamma unless ``gamma_bound`` is None, to every 10th I."""
    parameters = [bounded("beta")]
    if gamma_bound is not None:
        parameters.append(bounded("gamma", high=gamma_bound))
    problem = CalibrationProblem(
        observed_data=observed(truth_run, "I", range(0, 100, 10)),
        parameters=parameters,
        constraints=constraints,
        loss_function="sse",
        optimization_config=config
        or NelderMeadConfig(max_iterations=200, verbose=False),
        seed=SEED,
    )
    return Calibrator(Simulation(truth), problem).run().best_parameters


def _run_with(model: Model, fitted: dict[str, float]) -> dict[str, list[float]]:
    calibrated = model.model_copy(deep=True)
    calibrated.update_parameters(fitted)
    return Simulation(calibrated).run(100)


class TestParameterConstraints:
    def test_a_ratio_bound(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(
            id="r0_bound", expression="5.0 - beta/gamma", weight=1.0
        )

        fitted = _fit(truth, truth_run, [constraint], gamma_bound=0.5)

        assert fitted["beta"] / fitted["gamma"] <= 5.0

    def test_a_linear_sum_bound(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(
            id="sum_bound", expression="0.2 - (beta + gamma)"
        )

        fitted = _fit(truth, truth_run, [constraint])

        assert fitted["beta"] + fitted["gamma"] <= 0.21

    def test_an_ordering(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(
            id="beta_ge_gamma", expression="beta - gamma"
        )

        fitted = _fit(truth, truth_run, [constraint])

        assert fitted["beta"] >= fitted["gamma"] - 1e-6

    def test_several_constraints_hold_together(self, truth: Model, truth_run) -> None:
        constraints = [
            CalibrationConstraint(id="r0_bound", expression="5.0 - beta/gamma"),
            CalibrationConstraint(id="ordering", expression="beta - gamma"),
            CalibrationConstraint(id="sum_bound", expression="0.6 - (beta + gamma)"),
        ]

        fitted = _fit(
            truth,
            truth_run,
            constraints,
            gamma_bound=0.5,
            config=ParticleSwarmConfig(
                num_particles=20, max_iterations=200, verbose=False
            ),
        )

        assert fitted["beta"] / fitted["gamma"] <= 5.1
        assert fitted["beta"] >= fitted["gamma"] - 1e-6
        assert fitted["beta"] + fitted["gamma"] <= 0.61


class TestOutputConstraints:
    def test_an_output_bound_at_given_steps(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(
            id="peak_infected",
            expression="100.0 - I",
            at=[10, 20, 30, 40, 50],
        )

        fitted = _fit(truth, truth_run, [constraint], gamma_bound=None)

        calibrated = _run_with(truth, fitted)
        for step in [10, 20, 30, 40, 50]:
            assert calibrated["I"][step] <= 105.0, step

    def test_a_bound_on_a_sum_of_outputs(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(
            id="min_recovered", expression="950.0 - (S + I)", at=[30, 50, 70]
        )

        fitted = _fit(
            truth,
            truth_run,
            [constraint],
            gamma_bound=0.5,
            config=ParticleSwarmConfig(
                num_particles=20,
                max_iterations=200,
                verbose=False,
                initialization="latin_hypercube",
            ),
        )

        calibrated = _run_with(truth, fitted)
        for step in [30, 50, 70]:
            assert calibrated["S"][step] + calibrated["I"][step] <= 955.0, step
            assert calibrated["R"][step] >= 45.0, step

    def test_n_is_the_population_without_accumulators(self) -> None:
        """
        ``N`` in a constraint must be the population total only.

        Accumulators are appended to each output row after the compartments.
        If they were summed into ``N``, a bound written as a fraction of ``N``
        would loosen as the accumulator grew.
        """
        model = build(
            sir_builder(
                parameters={"beta": 0.1, "gamma": 0.05},
                accumulator="cumulative_infections",
            )
        )
        results = Simulation(model).run(200)
        # The accumulator must be big enough that inflating N would be visible.
        population = sum(results[b][-1] for b in ("S", "I", "R"))
        assert results["cumulative_infections"][-1] > 0.2 * population
        problem = CalibrationProblem(
            observed_data=observed(results, "I", range(200)),
            parameters=[bounded("beta", 0.05, 1.0)],
            constraints=[
                CalibrationConstraint(
                    id="max_cumulative_infections",
                    expression="0.5 * N - cumulative_infections",
                    at=[200],
                ),
            ],
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                num_particles=12, max_iterations=40, verbose=False
            ),
            seed=SEED,
        )

        fitted = Calibrator(Simulation(model), problem).run().best_parameters

        model.update_parameters(fitted)
        rerun = Simulation(model).run(200)
        rerun_population = sum(rerun[b][-1] for b in ("S", "I", "R"))
        assert rerun["cumulative_infections"][-1] <= 0.5 * rerun_population + 1e-6


class TestExpressionSecurity:
    @pytest.mark.parametrize(
        ("expression", "at"),
        [
            ("eval(beta)", None),
            ("import os", None),
            ("unsafe { beta }", None),
            (r"beta + \x41", None),
            ("eval(I)", [10, 20, 30]),
        ],
    )
    def test_a_dangerous_expression_is_rejected(
        self, expression: str, at: list[int] | None
    ) -> None:
        with pytest.raises(ValidationError, match="Security validation failed"):
            CalibrationConstraint(id="evil", expression=expression, at=at)

    @pytest.mark.parametrize(
        ("expression", "at"),
        [("5.0 - beta/gamma", None), ("500.0 - I", [10, 20, 30])],
    )
    def test_a_safe_expression_is_kept(
        self, expression: str, at: list[int] | None
    ) -> None:
        constraint = CalibrationConstraint(id="safe", expression=expression, at=at)

        assert constraint.expression == expression
