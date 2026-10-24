"""Every count a binding takes names its setting when it is negative."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from commol import SensitivityAnalysis, Simulation
from commol.api.calibrator import Calibrator
from commol.commol_rs import _commol_rs as commol_rs
from tests.support import abc_builder, abc_rates, build, line_calibrator

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import ForwardAnalysisProtocol

inference = commol_rs.inference
calibration = commol_rs.calibration
FIT = [20.0, 5.0]


@dataclass(frozen=True)
class Bindings:
    """The objects whose methods the cases call."""

    calibrator: Calibrator
    analysis: "ForwardAnalysisProtocol"


@pytest.fixture(scope="module")
def bindings() -> Bindings:
    calibrator = line_calibrator([7.4, 8.6, 11.3, 12.8])
    analysis = SensitivityAnalysis(
        Simulation(build(abc_builder())), abc_rates(), seed=1
    )._rust
    return Bindings(calibrator, analysis)


def _problem(b: Bindings):
    return b.calibrator.rust_problem


def _config(b: Bindings):
    return b.calibrator.rust_optimizer


def _sample():
    return inference.ParameterSample(["c"], [(0.0, 1.0)], [[0.5]], [0.0], None)


CASES: dict[str, tuple[str, Callable[[Bindings], object]]] = {
    "trials of ObservedDataPoint": (
        "trials",
        lambda _: calibration.ObservedDataPoint(1.0, "B", ["B"], 1.0, trials=-1),
    ),
    "trials of ObservationModel.binomial": (
        "trials",
        lambda _: calibration.ObservationModel.binomial(trials=-1),
    ),
    "count of LossConfig.with_replicates": (
        "count",
        lambda _: calibration.LossConfig.sse().with_replicates(count=-1),
    ),
    "max_iterations of NelderMeadConfig": (
        "max_iterations",
        lambda _: calibration.NelderMeadConfig(max_iterations=-1),
    ),
    "header_interval of NelderMeadConfig": (
        "header_interval",
        lambda _: calibration.NelderMeadConfig(header_interval=-1),
    ),
    "particles of ParticleSwarmConfig": (
        "particles",
        lambda _: calibration.ParticleSwarmConfig(particles=-1),
    ),
    "max_iterations of ParticleSwarmConfig": (
        "max_iterations",
        lambda _: calibration.ParticleSwarmConfig(max_iterations=-1),
    ),
    "window of ParticleSwarmConfig": (
        "window",
        lambda _: calibration.ParticleSwarmConfig(improvement_stop=(-1, 0.1, 0.1)),
    ),
    "num_steps of Engine.run": (
        "num_steps",
        lambda b: b.calibrator.simulation.engine.run(-1),
    ),
    "count of Engine.run_replicates": (
        "count",
        lambda b: b.calibrator.simulation.engine.run_replicates(None, -1, 5),
    ),
    "starts of CalibrationProblem.optimize": (
        "starts",
        lambda b: _problem(b).optimize(_config(b), -1, 1),
    ),
    "replicates of parametric_bootstrap": (
        "replicates",
        lambda b: _problem(b).parametric_bootstrap(_config(b), FIT, -1, 1, 1),
    ),
    "starts of parametric_bootstrap": (
        "starts",
        lambda b: _problem(b).parametric_bootstrap(_config(b), FIT, 2, -1, 1),
    ),
    "replicates of bayesian_bootstrap": (
        "replicates",
        lambda b: _problem(b).bayesian_bootstrap(_config(b), FIT, -1, None, 1, 1),
    ),
    "block_length of bayesian_bootstrap": (
        "block_length",
        lambda b: _problem(b).bayesian_bootstrap(_config(b), FIT, 2, -1, 1, 1),
    ),
    "starts of bayesian_bootstrap": (
        "starts",
        lambda b: _problem(b).bayesian_bootstrap(_config(b), FIT, 2, None, -1, 1),
    ),
    "replicates of residual_bootstrap": (
        "replicates",
        lambda b: _problem(b).residual_bootstrap(
            _config(b), FIT, -1, None, 1, 1, "additive", "iid"
        ),
    ),
    "block_length of residual_bootstrap": (
        "block_length",
        lambda b: _problem(b).residual_bootstrap(
            _config(b), FIT, 2, -1, 1, 1, "additive", "iid"
        ),
    ),
    "starts of residual_bootstrap": (
        "starts",
        lambda b: _problem(b).residual_bootstrap(
            _config(b), FIT, 2, None, -1, 1, "additive", "iid"
        ),
    ),
    "replicates of wild_bootstrap": (
        "replicates",
        lambda b: _problem(b).wild_bootstrap(_config(b), FIT, -1, None, 1, 1, "x"),
    ),
    "block_length of wild_bootstrap": (
        "block_length",
        lambda b: _problem(b).wild_bootstrap(_config(b), FIT, 2, -1, 1, 1, "x"),
    ),
    "starts of wild_bootstrap": (
        "starts",
        lambda b: _problem(b).wild_bootstrap(_config(b), FIT, 2, None, -1, 1, "x"),
    ),
    "block_length of jackknife": (
        "block_length",
        lambda b: _problem(b).jackknife(_config(b), FIT, -1, 1, 1),
    ),
    "starts of jackknife": (
        "starts",
        lambda b: _problem(b).jackknife(_config(b), FIT, None, -1, 1),
    ),
    "block_length of observation_draws": (
        "block_length",
        lambda b: _problem(b).observation_draws(
            _sample(), [], 1, residual_noise=(FIT, -1, "additive")
        ),
    ),
    "count of ParameterSample.resample": (
        "count",
        lambda _: _sample().resample(-1, 1),
    ),
    "starts of rolling_conformal": (
        "starts",
        lambda b: inference.rolling_conformal(
            _problem(b), _config(b), FIT, [], [], "x", "x", 0.9, -1, 1
        ),
    ),
    "block_length of jackknife_plus": (
        "block_length",
        lambda b: inference.jackknife_plus(
            _problem(b), _config(b), FIT, -1, [], "x", "x", 0.9, 1, 1
        ),
    ),
    "starts of jackknife_plus": (
        "starts",
        lambda b: inference.jackknife_plus(
            _problem(b), _config(b), FIT, None, [], "x", "x", 0.9, -1, 1
        ),
    ),
    "replicates of enbpi": (
        "replicates",
        lambda b: inference.enbpi(
            _problem(b), _config(b), FIT, -1, None, [], "x", "x", 0.9, 1, 1
        ),
    ),
    "block_length of enbpi": (
        "block_length",
        lambda b: inference.enbpi(
            _problem(b), _config(b), FIT, 2, -1, [], "x", "x", 0.9, 1, 1
        ),
    ),
    "starts of enbpi": (
        "starts",
        lambda b: inference.enbpi(
            _problem(b), _config(b), FIT, 2, None, [], "x", "x", 0.9, -1, 1
        ),
    ),
    "points of structural_rank": (
        "points",
        lambda b: inference.structural_rank(_problem(b), -1, False, 1),
    ),
    "max_subset_size of collinearity": (
        "max_subset_size",
        lambda b: inference.collinearity(_problem(b), [], [], -1, 20.0),
    ),
    "draws of laplace": (
        "draws",
        lambda b: inference.laplace(_problem(b), FIT, "x", -1, 1),
    ),
    "block_length of sandwich": (
        "block_length",
        lambda b: inference.sandwich(_problem(b), FIT, "x", -1, 1, 1),
    ),
    "draws of sandwich": (
        "draws",
        lambda b: inference.sandwich(_problem(b), FIT, "x", None, -1, 1),
    ),
    "draws of laplace_mixture": (
        "draws",
        lambda b: inference.laplace_mixture(_problem(b), None, None, [], -1, "x", 1),
    ),
    "replicates of BootstrapSettings": (
        "replicates",
        lambda _: inference.BootstrapSettings("bayesian", -1, 1),
    ),
    "starts of BootstrapSettings": (
        "starts",
        lambda _: inference.BootstrapSettings("bayesian", 2, -1),
    ),
    "block_length of BootstrapSettings": (
        "block_length",
        lambda _: inference.BootstrapSettings("bayesian", 2, 1, -1),
    ),
    "max_steps of trace_profiles": (
        "max_steps",
        lambda b: inference.trace_profiles(
            _problem(b), _config(b), FIT, [0], [], "beale", 0.9, -1, 1
        ),
    ),
    "replicates of ForwardAnalysis": (
        "replicates",
        lambda b: inference.ForwardAnalysis(
            b.calibrator.simulation.engine, [], replicates=-1, seed=1
        ),
    ),
    "draws of ForwardAnalysis.propagate": (
        "draws",
        lambda b: b.analysis.propagate(-1, "[]"),
    ),
    "draws of ForwardAnalysis.prcc": ("draws", lambda b: b.analysis.prcc(-1, "[]")),
    "trajectories of ForwardAnalysis.morris": (
        "trajectories",
        lambda b: b.analysis.morris(-1, 4, "[]"),
    ),
    "levels of ForwardAnalysis.morris": (
        "levels",
        lambda b: b.analysis.morris(4, -1, "[]"),
    ),
    "draws of ForwardAnalysis.sobol": (
        "draws",
        lambda b: b.analysis.sobol(-1, 2, 0.9, "[]"),
    ),
    "replicates of ForwardAnalysis.sobol": (
        "replicates",
        lambda b: b.analysis.sobol(8, -1, 0.9, "[]"),
    ),
    "degree of ForwardAnalysis.chaos": (
        "degree",
        lambda b: b.analysis.chaos(-1, 1.0, None, 8, "[]"),
    ),
    "max_interaction of ForwardAnalysis.chaos": (
        "max_interaction",
        lambda b: b.analysis.chaos(2, 1.0, -1, 8, "[]"),
    ),
    "draws of ForwardAnalysis.chaos": (
        "draws",
        lambda b: b.analysis.chaos(2, 1.0, None, -1, "[]"),
    ),
    "degree of chaos_evaluate": (
        "degree",
        lambda _: inference.chaos_evaluate([], [], -1, 1.0, None, [], []),
    ),
    "max_interaction of chaos_evaluate": (
        "max_interaction",
        lambda _: inference.chaos_evaluate([], [], 2, 1.0, -1, [], []),
    ),
    "count of Distribution.sample": (
        "count",
        lambda _: inference.Distribution("uniform", [], 0.0, 1.0).sample(-1, 1),
    ),
    "particles of gibbs_posterior": (
        "particles",
        lambda b: inference.gibbs_posterior(
            _problem(b), None, 1.0, -1, 0.5, 0.5, 0.5, 1
        ),
    ),
    "particles of bayesian_posterior": (
        "particles",
        lambda b: inference.bayesian_posterior(_problem(b), None, -1, 0.5, 0.5, 0.5, 1),
    ),
    "block_length of information_matching_eta": (
        "block_length",
        lambda b: inference.information_matching_eta(_problem(b), FIT, -1),
    ),
    "replicates of bootstrap_coverage_eta": (
        "replicates",
        lambda b: inference.bootstrap_coverage_eta(
            _problem(b), None, FIT, -1, None, 0.9, 10, 0.5, 0.5, 0.5, 1
        ),
    ),
    "block_length of bootstrap_coverage_eta": (
        "block_length",
        lambda b: inference.bootstrap_coverage_eta(
            _problem(b), None, FIT, 2, -1, 0.9, 10, 0.5, 0.5, 0.5, 1
        ),
    ),
    "particles of bootstrap_coverage_eta": (
        "particles",
        lambda b: inference.bootstrap_coverage_eta(
            _problem(b), None, FIT, 2, None, 0.9, -1, 0.5, 0.5, 0.5, 1
        ),
    ),
    "particles of abc_smc": (
        "particles",
        lambda b: inference.abc_smc(_problem(b), None, None, -1, 0.5, 0.5, 0.5, 1),
    ),
    "draws of abc_rejection": (
        "draws",
        lambda b: inference.abc_rejection(_problem(b), None, None, -1, 0.5, 1),
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_a_negative_count_names_its_setting(bindings: Bindings, case: str) -> None:
    setting, call = CASES[case]

    with pytest.raises(ValueError) as error:
        call(bindings)

    assert str(error.value) == f"{setting} is a whole number from 0, got -1"


def test_a_count_above_its_type_names_its_largest_value(bindings: Bindings) -> None:
    with pytest.raises(ValueError) as error:
        bindings.calibrator.simulation.engine.run(2**32)

    assert str(error.value) == (
        f"num_steps is a whole number from 0 to {2**32 - 1}, got {2**32}"
    )


def test_a_count_that_is_not_a_whole_number_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        calibration.NelderMeadConfig(max_iterations=1.5)  # ty: ignore[invalid-argument-type]
