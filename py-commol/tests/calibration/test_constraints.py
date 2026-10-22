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
from tests.support import assert_accepted, bounded, build, observed, abc_builder

SEED = 42


def _fit(
    truth: Model,
    truth_run: dict[str, list[float]],
    constraints: list[CalibrationConstraint],
    *,
    k2_bound: float | None = 1.0,
    config: NelderMeadConfig | ParticleSwarmConfig | None = None,
) -> dict[str, float]:
    """Fit k1, and k2 unless ``k2_bound`` is None, to every 10th B."""
    parameters = [bounded("k1")]
    if k2_bound is not None:
        parameters.append(bounded("k2", high=k2_bound))
    problem = CalibrationProblem(
        observed_data=observed(truth_run, "B", range(0, 100, 10)),
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
            id="ratio_bound", expression="5.0 - k1/k2", weight=1.0
        )

        fitted = _fit(truth, truth_run, [constraint], k2_bound=0.5)

        assert fitted["k1"] / fitted["k2"] <= 5.0

    def test_a_linear_sum_bound(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(id="sum_bound", expression="0.2 - (k1 + k2)")

        fitted = _fit(truth, truth_run, [constraint])

        assert fitted["k1"] + fitted["k2"] <= 0.21

    def test_an_ordering(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(id="k1_ge_k2", expression="k1 - k2")

        fitted = _fit(truth, truth_run, [constraint])

        assert fitted["k1"] >= fitted["k2"] - 1e-6

    def test_several_constraints_hold_together(self, truth: Model, truth_run) -> None:
        constraints = [
            CalibrationConstraint(id="ratio_bound", expression="5.0 - k1/k2"),
            CalibrationConstraint(id="ordering", expression="k1 - k2"),
            CalibrationConstraint(id="sum_bound", expression="0.6 - (k1 + k2)"),
        ]

        fitted = _fit(
            truth,
            truth_run,
            constraints,
            k2_bound=0.5,
            config=ParticleSwarmConfig(particles=20, max_iterations=200, verbose=False),
        )

        assert fitted["k1"] / fitted["k2"] <= 5.1
        assert fitted["k1"] >= fitted["k2"] - 1e-6
        assert fitted["k1"] + fitted["k2"] <= 0.61


class TestOutputConstraints:
    def test_an_output_bound_at_given_steps(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(
            id="peak_b",
            expression="100.0 - B",
            at=[10, 20, 30, 40, 50],
        )

        fitted = _fit(truth, truth_run, [constraint], k2_bound=None)

        calibrated = _run_with(truth, fitted)
        for step in [10, 20, 30, 40, 50]:
            assert calibrated["B"][step] <= 105.0, step

    def test_a_bound_on_a_sum_of_outputs(self, truth: Model, truth_run) -> None:
        constraint = CalibrationConstraint(
            id="min_c", expression="950.0 - (A + B)", at=[30, 50, 70]
        )

        fitted = _fit(
            truth,
            truth_run,
            [constraint],
            k2_bound=0.5,
            config=ParticleSwarmConfig(
                particles=20,
                max_iterations=200,
                verbose=False,
                initialization="latin_hypercube",
            ),
        )

        calibrated = _run_with(truth, fitted)
        for step in [30, 50, 70]:
            assert calibrated["A"][step] + calibrated["B"][step] <= 955.0, step
            assert calibrated["C"][step] >= 45.0, step

    def test_total_excludes_the_accumulators(self) -> None:
        """
        ``total`` in a constraint must be the sum of the compartments only.

        Accumulators are appended to each output row after the compartments.
        If they were summed into ``total``, a bound written as a fraction of ``total``
        would loosen as the accumulator grew.
        """
        model = build(
            abc_builder(
                parameters={"k1": 0.1, "k2": 0.05},
                accumulator="cumulative_ab",
            )
        )
        results = Simulation(model).run(200)
        # The accumulator must be big enough that inflating total would be visible.
        compartments = sum(results[b][-1] for b in ("A", "B", "C"))
        assert results["cumulative_ab"][-1] > 0.2 * compartments
        problem = CalibrationProblem(
            observed_data=observed(results, "B", range(200)),
            parameters=[bounded("k1", 0.05, 1.0)],
            constraints=[
                CalibrationConstraint(
                    id="max_cumulative_ab",
                    expression="0.5 * total - cumulative_ab",
                    at=[200],
                ),
            ],
            loss_function="sse",
            optimization_config=ParticleSwarmConfig(
                particles=12, max_iterations=40, verbose=False
            ),
            seed=SEED,
        )

        fitted = Calibrator(Simulation(model), problem).run().best_parameters

        model.update_parameters(fitted)
        rerun = Simulation(model).run(200)
        rerun_total = sum(rerun[b][-1] for b in ("A", "B", "C"))
        assert rerun["cumulative_ab"][-1] <= 0.5 * rerun_total + 1e-6


class TestExpressionSecurity:
    @pytest.mark.parametrize(
        ("expression", "at"),
        [
            ("eval(k1)", None),
            ("import os", None),
            ("unsafe { k1 }", None),
            (r"k1 + \x41", None),
            ("eval(B)", [10, 20, 30]),
        ],
    )
    def test_a_dangerous_expression_is_rejected(
        self, expression: str, at: list[int] | None
    ) -> None:
        with pytest.raises(ValidationError, match="Security validation failed"):
            CalibrationConstraint(id="evil", expression=expression, at=at)

    @pytest.mark.parametrize(
        ("expression", "at"),
        [("5.0 - k1/k2", None), ("500.0 - B", [10, 20, 30])],
    )
    def test_a_safe_expression_is_accepted(
        self, expression: str, at: list[int] | None
    ) -> None:
        assert_accepted(CalibrationConstraint, id="safe", expression=expression, at=at)


class TestPositions:
    @pytest.mark.parametrize(
        ("at", "message"),
        [
            ([], "must not be empty if specified"),
            ([10, -1], "must all be finite and non-negative"),
            ([float("inf")], "must all be finite and non-negative"),
        ],
    )
    def test_a_constraint_is_checked_at_positions_of_the_axis(
        self, at: list[float], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=f"'cap' {message}"):
            CalibrationConstraint(id="cap", expression="500.0 - B", at=at)
