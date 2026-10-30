"""A particle swarm polished by Nelder-Mead: each start runs the swarm, then
Nelder-Mead from the best point of the swarm."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ParticleSwarmConfig,
    Simulation,
)
from commol._model.model import Model
from tests.support import abc_builder, bounded, build, observed

STEPS = range(5, 51, 5)
TRUTH = {"k1": 0.4, "k2": 0.1}
POLISH = NelderMeadConfig(max_iterations=500)


def _calibrator(
    config: ParticleSwarmConfig | NelderMeadConfig,
    guesses: dict[str, float] | None = None,
) -> Calibrator:
    """
    Fit ``k1`` and ``k2`` of ``abc_builder``, in ``[0, 1]``, to B and C of
    its run at ``k1 = 0.4``, ``k2 = 0.1``, from the `guesses`, with the seed 7.
    """
    truth = Simulation(build(abc_builder(parameters=dict(TRUTH)))).run(50)
    guesses = guesses or {}
    problem = CalibrationProblem(
        observed_data=observed(truth, "B", STEPS) + observed(truth, "C", STEPS),
        parameters=[
            bounded(name, 0.0, 1.0, initial_guess=guesses.get(name)) for name in TRUTH
        ],
        loss_function="sse",
        optimization_config=config,
        seed=7,
    )
    model: Model = build(abc_builder(parameters={"k1": None, "k2": None}))
    return Calibrator(Simulation(model), problem)


def _swarm(polish: NelderMeadConfig | None) -> ParticleSwarmConfig:
    return ParticleSwarmConfig(particles=8, max_iterations=3, polish=polish)


def test_a_swarm_polishes_each_start() -> None:
    result = _calibrator(_swarm(POLISH)).run(starts=4)

    assert len(result.starts) == 4
    assert all(start.converged for start in result.starts)
    assert result.converged
    assert result.termination_reason == "Terminated(SolverConverged)"
    assert result.best_parameters == pytest.approx(TRUTH, abs=1e-4)


def test_a_polish_is_nelder_mead_from_the_end_of_the_swarm() -> None:
    polished = _calibrator(_swarm(POLISH)).run(starts=4)
    swarm = _calibrator(_swarm(None)).run(starts=4)

    for alone, start in zip(swarm.starts, polished.starts, strict=True):
        by_hand = _calibrator(POLISH, guesses=alone.end).run()
        assert start.start == alone.start
        assert start.end == by_hand.best_parameters
        assert start.loss == by_hand.final_loss
        assert start.converged == by_hand.converged
    # The swarm alone stops at its 3 iterations without converging.
    assert not any(start.converged for start in swarm.starts)


def test_a_polished_swarm_counts_the_iterations_of_both_stages() -> None:
    polished = _calibrator(_swarm(POLISH)).run()
    swarm = _calibrator(_swarm(None)).run()
    by_hand = _calibrator(POLISH, guesses=swarm.best_parameters).run()

    assert swarm.iterations == 3
    assert polished.iterations == swarm.iterations + by_hand.iterations


def test_a_polish_is_nelder_mead() -> None:
    with pytest.raises(ValueError, match="polish"):
        ParticleSwarmConfig(polish=_swarm(None))  # type: ignore[arg-type]


def test_each_stage_writes_its_own_table(capfd: pytest.CaptureFixture[str]) -> None:
    verbose_polish = POLISH.model_copy(update={"verbose": True, "header_interval": 5})
    _calibrator(_swarm(verbose_polish)).run()
    polish_only = capfd.readouterr().out
    quiet = _calibrator(_swarm(POLISH)).run()
    nothing = capfd.readouterr().out
    both = ParticleSwarmConfig(
        particles=8, max_iterations=3, verbose=True, polish=verbose_polish
    )
    loud = _calibrator(both).run()
    swarm_and_polish = capfd.readouterr().out

    assert "Polish: Nelder-Mead from the best point of the swarm" in polish_only
    assert nothing == ""
    # The swarm writes rows 0 to 3; Nelder-Mead starts its own table at 0.
    rows = [line.split("|")[0].strip() for line in swarm_and_polish.splitlines()]
    assert rows.count("0") == 2
    assert polish_only.count("Iteration") < swarm_and_polish.count("Iteration")
    # A calibration that writes is the calibration that writes nothing.
    assert loud.best_parameters == quiet.best_parameters
