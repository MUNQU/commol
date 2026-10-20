"""The verdicts of profiles, what a flat profile says about the structure,
and the identifiability report they give."""

import numpy as np
import pytest

from commol import (
    CalibrationProblem,
    CalibrationResult,
    Calibrator,
    NelderMeadConfig,
    NormalObservations,
    ObservedDataPoint,
    PredictionProfile,
    PredictionSpec,
    Profile,
    ProfileInterval,
    ProfileLikelihood,
    ProfileResult,
    ResidualBootstrap,
    Simulation,
    StructuralRank,
    Threshold,
)
from commol.context.inference.identifiability import ParameterVerdicts
from commol.context.inference.profiles import ProfileStructure
from tests.support import (
    a_to_b_builder,
    abc_builder,
    bounded,
    build,
    build_ode,
    build_stochastic,
    line_calibrator,
    observed,
)

# ``5 + 2 t`` at t = 1, …, 10 plus a fixed draw of unit noise.
VALUES = [7.4, 8.6, 11.3, 12.8, 15.1, 16.4, 19.9, 20.6, 23.5, 24.7]
RESET = "with the other parameters at their fitted values"


def _noisy_b(k1: float, steps: int = 40) -> list[ObservedDataPoint]:
    """B of ``abc_builder`` every 4 steps, with normal noise of sd 10."""
    truth = Simulation(build(abc_builder(parameters={"k1": k1, "k2": 0.1}))).run(steps)
    rng = np.random.default_rng(3)
    return [
        ObservedDataPoint(
            at=step,
            output="B",
            value=max(truth["B"][step] + rng.normal(0.0, 10.0), 0.0),
        )
        for step in range(4, steps + 1, 4)
    ]


def _noisy(
    results: dict[str, list[float]], output: str, at: range, sd: float
) -> list[ObservedDataPoint]:
    """`output` of `results` at `at`, with normal noise of `sd`."""
    rng = np.random.default_rng(5)
    return [
        ObservedDataPoint(
            at=step,
            output=output,
            value=max(results[output][step] + rng.normal(0.0, sd), 0.0),
        )
        for step in at
    ]


def _guide_calibrator(
    engine: str = "DifferenceEquations",
    optimizer: NelderMeadConfig | None = None,
) -> Calibrator:
    """The product model of the guide: the rate reads ``k1 * s``, with ``k1``
    in ``[0.2, 0.8]`` and ``s`` in ``[0.5, 2]``; by default Nelder-Mead runs
    to an sd_tolerance of 1e-15."""
    builder = abc_builder(
        ab="k1 * s * A * B / total",
        parameters={"k1": None, "s": None, "k2": None},
    )
    model = builder.build(builder.structure, engine=engine)
    problem = CalibrationProblem(
        observed_data=_noisy_b(0.4),
        parameters=[
            bounded("k1", 0.2, 0.8),
            bounded("s", 0.5, 2.0),
            bounded("k2", 0.0, 1.0),
        ],
        observation_models={"B": NormalObservations(sd=10.0)},
        optimization_config=optimizer
        or NelderMeadConfig(max_iterations=4000, sd_tolerance=1e-15),
        seed=42,
    )
    return Calibrator(Simulation(model), problem)


def _unread_k3(
    ode: bool,
    optimizer: NelderMeadConfig | None = None,
    k2: tuple[float, float] = (0.0, 1.0),
) -> Calibrator:
    """``abc_builder`` with ``k3`` emptying C into D, and only B observed;
    `k2` are the bounds of ``k2``."""
    builder = abc_builder(
        parameters={"k1": None, "k2": None, "k3": None},
        amounts={"A": 990, "B": 10, "C": 0, "D": 0},
    )
    builder.add_bin(id="D", name="State D")
    builder.add_transition(id="cd", source=["C"], target=["D"], rate="k3 * C")
    model = (
        builder.build(builder.structure, engine="DifferentialEquations")
        if ode
        else build(builder)
    )
    problem = CalibrationProblem(
        observed_data=_noisy_b(0.3),
        parameters=[
            bounded("k1", 0.0, 1.0),
            bounded("k2", *k2),
            bounded("k3", 0.0, 1.0),
        ],
        observation_models={"B": NormalObservations(sd=10.0)},
        optimization_config=optimizer
        or NelderMeadConfig(max_iterations=4000, sd_tolerance=1e-15),
        seed=42,
    )
    return Calibrator(Simulation(model), problem)


def _profile(statistics: list[float], threshold: float = 2.0) -> Profile:
    values = [float(i) for i in range(len(statistics))]
    return Profile(
        center=1.0,
        threshold=threshold,
        values=values,
        statistics=statistics,
        parameters=[{"k1": v, "s": 1.0} for v in values],
        multipliers=[None] * len(values),
        converged=[True] * len(values),
        ends=("bound", "bound"),
        evaluations=10,
    )


def _result(structure: ProfileStructure) -> ProfileResult:
    """A result whose profile of ``k1`` is flat, with `structure`."""
    return ProfileResult(
        fit=CalibrationResult(
            best_parameters={"k1": 1.0, "s": 1.0},
            final_loss=2.0,
            iterations=10,
            converged=True,
            termination_reason="converged",
        ),
        quantity="parameter",
        threshold=Threshold.chi_square(0.9),
        profiles={"k1": _profile([1e-9, 0.0, 1e-9])},
        intervals={"k1": ProfileInterval(lower=None, upper=None)},
        verdicts={"k1": "structurally_non_identifiable"},
        structure={"k1": structure},
        tolerance=1e-6,
    )


class TestShapes:
    def test_product_model_gives_flat_profiles(self) -> None:
        # The model of the fixture `product` of test_identifiability.py: the
        # rate reads `k1 * s`, with data of noise of sd 1 scored by a normal
        # model, so the threshold and the noise floor are above 0.
        truth = Simulation(build(abc_builder(parameters={"k1": 0.3, "k2": 0.1}))).run(
            50
        )
        model = build(
            abc_builder(
                ab="k1 * s * A * B / total",
                parameters={"k1": None, "s": None, "k2": None},
            )
        )
        problem = CalibrationProblem(
            observed_data=_noisy(truth, "B", range(5, 51, 5), 1.0),
            parameters=[
                bounded("k1", 0.15, 0.6),
                bounded("s", 0.5, 2.0),
                bounded("k2", 0.02, 0.3),
            ],
            observation_models={"B": NormalObservations(sd=1.0)},
            # A refit on the ridge runs to max_iterations: 1500 still reach
            # the floor of the ridge.
            optimization_config=NelderMeadConfig(
                max_iterations=1500, sd_tolerance=1e-15
            ),
            seed=3,
        )
        calibrator = Calibrator(Simulation(model), problem)

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["k1"], threshold=Threshold.chi_square(0.95))
        )

        assert result.verdicts["k1"] == "structurally_non_identifiable"
        assert result.structure["k1"].verdict == "in_combination"
        assert result.structure["k1"].compensated_by == ["s"]
        report = result.report()
        assert report.parameters == ["k1", "s", "k2"]
        structural = report.verdicts["k1"].structural
        assert structural is not None
        assert structural.verdict == "in_combination"
        assert structural.combinations == []
        assert "k1, compensated by s along its profile" in structural.reasons[0]
        assert RESET in structural.reasons[0]
        assert report.verdicts["k2"] == ParameterVerdicts()
        assert report.profile[0].verdicts == result.verdicts

    def test_late_parameter_gives_a_one_sided_profile(self) -> None:
        # B starts to fill at `tau`; the data stop at 10 and see no flow, so
        # every `tau` past 10 fits them, and an earlier one does not.
        model = build_ode(
            a_to_b_builder(parameters={"tau": None}, rate="if(t > tau, 0.2, 0)")
        )
        problem = CalibrationProblem(
            observed_data=observed({"B": [0.0] * 11}, "B", range(1, 11)),
            parameters=[bounded("tau", 0.0, 30.0, initial_guess=20.0)],
            observation_models={"B": NormalObservations(sd=1.0)},
            optimization_config=NelderMeadConfig(max_iterations=500),
            seed=3,
        )
        calibrator = Calibrator(Simulation(model), problem)

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["tau"], threshold=Threshold.chi_square(0.95))
        )

        assert result.profiles["tau"].ends == ("crossed", "bound")
        assert result.intervals["tau"].upper is None
        lower = result.intervals["tau"].lower
        assert lower is not None and 9.0 < lower < 10.0
        assert result.verdicts["tau"] == "practically_non_identifiable"
        assert result.structure["tau"].undetermined == "not_flat"
        practical = result.report().verdicts["tau"].practical
        assert practical is not None
        assert practical.verdict == "poorly_determined"

    def test_a_threshold_below_the_tolerance_gives_no_verdict(self) -> None:
        # Values on the line: the sum of squares of the fit, and Beale's
        # threshold with it, are 0.
        calibrator = line_calibrator([5.0 + 2.0 * t for t in range(1, 11)])

        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c"], threshold=Threshold.beale(0.95), max_steps=5
            )
        )

        assert result.profiles["c"].threshold <= result.tolerance
        assert result.verdicts["c"] is None
        assert any(
            "'c'" in note and "below the numerical tolerance" in note
            for note in result.notes
        )
        assert result.report().verdicts["c"].practical is None

    def test_a_refit_below_the_fit_is_noted(self) -> None:
        # Three iterations of Nelder-Mead leave the fit far from the optimum;
        # the refits of the profile, which start from it, go below it.
        calibrator = line_calibrator(
            VALUES, observation_models={"B": NormalObservations(sd=1.0)}
        )
        problem = calibrator.problem.model_copy(
            update={"optimization_config": NelderMeadConfig(max_iterations=3)}
        )

        result = Calibrator(calibrator.simulation, problem).uncertainty(
            ProfileLikelihood(
                parameters=["c"], threshold=Threshold.chi_square(0.9), max_steps=3
            )
        )

        assert min(result.profiles["c"].statistics) < -result.tolerance
        assert any("below the fit" in note for note in result.notes)

    def test_only_parameter_profiles_report_on_identifiability(self) -> None:
        calibrator = line_calibrator(
            VALUES, observation_models={"B": NormalObservations(sd=1.0)}
        )
        result = calibrator.uncertainty(
            PredictionProfile(
                specs=[PredictionSpec(at=4, output="B")],
                threshold=Threshold.chi_square(0.9),
                max_steps=2,
            )
        )

        assert result.structure == {}
        with pytest.raises(ValueError, match="profiles of predictions"):
            result.report()


class TestStructure:
    def test_the_ridge_of_a_product_agrees_with_the_structural_rank(self) -> None:
        calibrator = _guide_calibrator()

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["k1"], threshold=Threshold.chi_square(0.95))
        )

        # The ridge `k1 · s ≈ 0.4` leaves the box where `s` reaches 0.5 or
        # 2, and the profile rises past it.
        assert max(result.profiles["k1"].statistics) > result.tolerance
        assert result.verdicts == {"k1": "structurally_non_identifiable"}
        notes = " ".join(result.notes)
        assert "'k1': the ridge leaves the box" in notes
        assert "where 's' reaches its bound" in notes
        structure = result.structure["k1"]
        assert structure.verdict == "in_combination"
        assert structure.compensated_by == ["s"]
        assert structure.largest_rise is not None
        assert structure.largest_rise > result.profiles["k1"].threshold
        rank = calibrator.identifiability(
            StructuralRank(points=5, design="observations")
        )
        merged = rank.merge(result.report())
        for name in ("k1", "s"):
            structural = merged.verdicts[name].structural
            assert structural is not None
            assert structural.verdict == "in_combination"
            assert structural.combinations == ["k1 · s"]

    @pytest.mark.parametrize("ode", [False, True], ids=["difference", "ode"])
    def test_a_parameter_with_no_effect_is_non_identifiable(self, ode: bool) -> None:
        # With the ODE engine k3 enters the right-hand side, at the default
        # tolerances of the solver, and moves B only by solver error. The
        # resets compare evaluations with no optimizer, so the default
        # sd_tolerance of Nelder-Mead is enough.
        calibrator = _unread_k3(ode, NelderMeadConfig(max_iterations=4000))

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["k3"], threshold=Threshold.chi_square(0.95))
        )

        assert result.verdicts["k3"] == "structurally_non_identifiable"
        structure = result.structure["k3"]
        assert structure.verdict == "non_identifiable"
        assert structure.compensated_by == []
        rank = calibrator.identifiability(
            StructuralRank(points=5, design="observations")
        )
        merged = rank.merge(result.report())
        structural = merged.verdicts["k3"].structural
        assert structural is not None
        assert structural.verdict == "non_identifiable"
        assert RESET in structural.reasons[-1]

    def test_a_flat_profile_beside_a_fit_on_a_bound_gives_no_verdict(self) -> None:
        # B is drawn with k2 = 0.1, below the bounds of k2: its fit is on 0.2,
        # and what lies past the bound is not seen.
        calibrator = _unread_k3(False, k2=(0.2, 1.0))

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["k3"], threshold=Threshold.chi_square(0.95))
        )

        assert result.fit.best_parameters["k2"] == pytest.approx(0.2)
        structure = result.structure["k3"]
        assert structure.verdict is None
        assert structure.undetermined == "on_bound"
        assert structure.on_bound == "k2"

    def test_a_sum_dominated_by_one_term_conflicts_with_the_rank(self) -> None:
        # The rate reads k1 + k2 with k1 near 0.1 and k2 below 0.005: on the
        # scale of each parameter, the rank finds k2 in the null space and
        # k1 out of it (its cut-offs 0.95 and 0.05), while along the profile
        # of k2 the other term compensates it.
        model = build(
            a_to_b_builder(parameters={"k1": None, "k2": None}, rate="k1 + k2")
        )
        truth = Simulation(build(a_to_b_builder(parameters={"k": 0.1}))).run(20)
        problem = CalibrationProblem(
            observed_data=_noisy(truth, "B", range(2, 21, 2), 5.0),
            parameters=[bounded("k1", 0.05, 0.5), bounded("k2", 0.0, 0.005)],
            observation_models={"B": NormalObservations(sd=5.0)},
            optimization_config=NelderMeadConfig(
                max_iterations=4000, sd_tolerance=1e-15
            ),
            seed=42,
        )
        calibrator = Calibrator(Simulation(model), problem)

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["k2"], threshold=Threshold.chi_square(0.95))
        )
        rank = calibrator.identifiability(
            StructuralRank(points=5, design="observations")
        )

        assert result.structure["k2"].verdict == "in_combination"
        assert result.structure["k2"].compensated_by == ["k1"]
        rank_verdict = rank.verdicts["k2"].structural
        assert rank_verdict is not None and rank_verdict.verdict == "non_identifiable"
        merged = rank.merge(result.report()).verdicts["k2"].structural
        assert merged is not None and merged.verdict == "conflicting"

    def test_a_stochastic_engine_gives_no_structural_verdict(self) -> None:
        # u multiplies nothing: its profile is flat, and a stochastic engine
        # still reads no structure from it.
        builder = a_to_b_builder(parameters={"k": None, "u": None}, rate="k + 0 * u")
        data = (
            Simulation(build_stochastic(a_to_b_builder()))
            .replicates(50, seed=21)
            .run_at([2.0, 4.0, 6.0, 8.0])
            .mean()
        )
        problem = CalibrationProblem(
            observed_data=[
                ObservedDataPoint(at=at, output="A", value=value)
                for at, value in zip([2.0, 4.0, 6.0, 8.0], data["A"], strict=True)
            ],
            parameters=[bounded("k", 0.01, 0.5), bounded("u", 0.0, 1.0)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=200),
            replicates=4,
            replicate_summary="mean",
            seed=7,
        )
        calibrator = Calibrator(Simulation(build_stochastic(builder)), problem)

        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["u"], threshold=Threshold.beale(0.95), max_steps=10
            )
        )

        assert result.verdicts["u"] == "practically_non_identifiable"
        assert result.structure["u"].verdict is None
        assert any("'u'" in note and "stochastic" in note for note in result.notes)
        report = result.report()
        assert report.verdicts["u"].structural is None


class TestDefaults:
    """Nelder-Mead at its default sd_tolerance (1e-6), the ODE solver at its
    default tolerances."""

    @pytest.mark.parametrize("ode", [False, True], ids=["difference", "ode"])
    def test_a_parameter_with_no_effect_is_non_identifiable(self, ode: bool) -> None:
        calibrator = _unread_k3(ode, NelderMeadConfig())

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["k3"], threshold=Threshold.chi_square(0.95))
        )

        assert result.structure["k3"].verdict == "non_identifiable"
        assert result.verdicts["k3"] == "structurally_non_identifiable"

    def test_the_ridge_of_an_ode_model_is_in_combination(self) -> None:
        # The solver's tolerance sets a floor above the error of a refit, so
        # the profiles are flat; the ridge leaves the box where the other
        # parameter reaches its bound, and the crossing past it is the box's.
        calibrator = _guide_calibrator("DifferentialEquations", NelderMeadConfig())

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["k1"], threshold=Threshold.chi_square(0.95))
        )

        assert result.structure["k1"].verdict == "in_combination"
        assert result.profiles["k1"].box_crossings == (None, "s")
        assert any("where 's' reaches a bound" in note for note in result.notes)

    def test_the_ridge_of_a_difference_model_gives_no_structural_verdict(
        self,
    ) -> None:
        # The error of a refit, near the sd_tolerance, is above the floor of a
        # difference model: the profile is not flat, and a refit lands below
        # the fit.
        calibrator = _guide_calibrator(optimizer=NelderMeadConfig())

        result = calibrator.uncertainty(
            ProfileLikelihood(parameters=["k1"], threshold=Threshold.chi_square(0.95))
        )

        assert result.verdicts["k1"] == "practically_non_identifiable"
        assert result.structure["k1"].verdict is None
        assert result.report().verdicts["k1"].practical.verdict == "poorly_determined"


class TestNotes:
    def test_a_loss_with_no_floor_says_so(self) -> None:
        calibrator = line_calibrator(VALUES)
        problem = calibrator.problem.model_copy(update={"loss_function": "mae"})

        result = Calibrator(calibrator.simulation, problem).uncertainty(
            ProfileLikelihood(
                parameters=["c"],
                threshold=Threshold.bootstrap_calibrated(
                    ResidualBootstrap(
                        replicates=5,
                        block_length=1,
                        residual_scale="additive",
                        scheme="moving",
                    ),
                    0.9,
                ),
                max_steps=3,
            )
        )

        assert result.evaluation_floor is None
        assert any("cannot be computed" in note for note in result.notes)

    def test_a_refit_less_accurate_than_the_tolerance_is_noted(self) -> None:
        calibrator = line_calibrator(
            VALUES, observation_models={"B": NormalObservations(sd=1.0)}
        )
        problem = calibrator.problem.model_copy(
            update={"optimization_config": NelderMeadConfig(max_iterations=5)}
        )

        result = Calibrator(calibrator.simulation, problem).uncertainty(
            ProfileLikelihood(
                parameters=["c"], threshold=Threshold.chi_square(0.9), max_steps=3
            )
        )

        refit = result.profiles["c"].refit_error
        assert refit is not None and refit > result.tolerance
        assert any("a refit at the fit lands" in note for note in result.notes)


class TestReport:
    def test_no_structural_verdict_still_gives_a_practical_one(self) -> None:
        result = _result(
            ProfileStructure(
                verdict=None,
                largest_rise=0.5,
                compensated_by=[],
                joint_rise=None,
                undetermined="undecided",
            )
        )

        verdicts = result.report().verdicts["k1"]

        assert verdicts.structural is None
        assert verdicts.practical is not None
        assert verdicts.practical.verdict == "poorly_determined"
        assert RESET in verdicts.practical.reasons[0]
        assert "0.5" in verdicts.practical.reasons[0]

    def test_a_list_of_names_that_may_be_incomplete_says_so(self) -> None:
        result = _result(
            ProfileStructure(
                verdict="in_combination",
                largest_rise=40.0,
                compensated_by=["s"],
                joint_rise=3.5,
                undetermined=None,
            )
        )

        structural = result.report().verdicts["k1"].structural

        assert structural is not None
        assert structural.verdict == "in_combination"
        assert "may be incomplete (largest rise 3.5)" in structural.reasons[0]

    def test_in_combination_with_no_parameter_named(self) -> None:
        result = _result(
            ProfileStructure(
                verdict="in_combination",
                largest_rise=40.0,
                compensated_by=[],
                joint_rise=40.0,
                undetermined=None,
            )
        )

        reason = result.report().verdicts["k1"].structural.reasons[0]

        assert "compensated by" not in reason
        assert RESET in reason
