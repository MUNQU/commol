"""Calibration problems that many tests share."""

from collections.abc import Iterable, Mapping, Sequence

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.sensitivity import InputSpace
from commol.observations import NormalObservations
from commol.observations.models import _ObservationModel

from .models import a_to_b_builder, abc_builder, build


def initial_condition_problem(
    *ids: str, remainder: str | None = None
) -> CalibrationProblem:
    """
    A problem that declares each id as an initial condition in ``[0, 10⁶]``,
    wide enough for an amount and for a category split.

    It is meant for writing values back, not for a run: it has one placeholder
    observation and one Nelder-Mead iteration.
    """
    return CalibrationProblem(
        initial_condition_remainder=remainder,
        observed_data=[ObservedDataPoint(at=1, output="A", value=1.0)],
        parameters=[
            CalibrationParameter(
                id=entry_id,
                parameter_type="initial_condition",
                min_bound=0.0,
                max_bound=1e6,
            )
            for entry_id in ids
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )


def bounded(
    entry_id: str,
    low: float = 0.0,
    high: float = 1.0,
    *,
    kind: str = "parameter",
    initial_guess: float | None = None,
) -> CalibrationParameter:
    """A calibration parameter of ``kind`` searched in ``[low, high]``."""
    return CalibrationParameter(
        id=entry_id,
        parameter_type=kind,
        min_bound=low,
        max_bound=high,
        initial_guess=initial_guess,
    )


def abc_rates() -> InputSpace:
    """The inputs ``k1`` of ``abc_builder``, uniform on ``[0.1, 0.5]``, and
    ``k2``, uniform on ``[0, 1]``."""
    return InputSpace.uniform_on_bounds([bounded("k1", 0.1, 0.5), bounded("k2")])


def observed(
    results: dict[str, list[float]],
    output: str,
    at: Iterable[int],
    *,
    factor: float = 1.0,
    factor_id: str | None = None,
) -> list[ObservedDataPoint]:
    """Observations of one output of a run, at the given integer positions."""
    return [
        ObservedDataPoint(
            at=step,
            output=output,
            value=results[output][step] * factor,
            factor_id=factor_id,
        )
        for step in at
    ]


def flow_observations(
    values: Sequence[float], *, windowed: bool, start: int = 1
) -> list[ObservedDataPoint]:
    """
    Observations of B at positions ``start``, ``start + 1``, …: with
    `windowed`, of the flow of one step each; otherwise of the amount of B.
    """
    return [
        ObservedDataPoint(
            at=start + index,
            output="B",
            value=float(value),
            window=1 if windowed else None,
        )
        for index, value in enumerate(values)
    ]


def flow_calibrator(
    values: Sequence[float],
    *,
    windowed: bool,
    seed: int = 1,
    bounds: tuple[float, float] = (0.0, 50.0),
    observation_models: Mapping[str, _ObservationModel] | None = None,
    noise: Sequence[CalibrationParameter] = (),
) -> Calibrator:
    """
    Fit a constant flow ``c`` from A to B (``a_to_b_builder`` with the rate
    ``c * A / A``, absolute because it names A, and 10⁶ in A) to
    `values` at positions 1, 2, ….

    A model linear in its parameter: with `windowed`, each observation is the
    flow of one step and its prediction is ``c``, so the fit is the mean of
    the values; otherwise each observation is the amount of B at ``t`` and
    its prediction is ``c·t``, a regression through the origin.

    The fit minimizes the sum of squares, or, with `observation_models` (of
    the series ``"B"``), maximizes their likelihood, with the `noise`
    parameters calibrated after ``c``.
    """
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
        )
    )
    observations = flow_observations(values, windowed=windowed)
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[bounded("c", *bounds, initial_guess=sum(bounds) / 2), *noise],
        loss_function=None if observation_models else "sse",
        observation_models=dict(observation_models) if observation_models else None,
        optimization_config=NelderMeadConfig(max_iterations=1000, sd_tolerance=1e-14),
        seed=seed,
    )
    return Calibrator(Simulation(model), problem)


def rate_calibrator(rate: str, flows: Sequence[float]) -> Calibrator:
    """
    A flow from A to B at `rate` (``a_to_b_builder``, 10⁶ in A), fitted by
    ``c`` in [0, 50] from 10 to `flows`, each the flow of one step, at
    positions 1, 2, ….
    """
    model = build(
        a_to_b_builder(parameters={"c": None}, rate=rate, amounts={"A": 1e6, "B": 0})
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(flows, windowed=True),
        parameters=[bounded("c", 0.0, 50.0, initial_guess=10.0)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1000, sd_tolerance=1e-14),
        seed=1,
    )
    return Calibrator(Simulation(model), problem)


TWO_WELLS_FIT = 2.492033
"""The best ``c`` of ``two_wells_calibrator``: ``c³ − 3c + 2`` at the mean."""


def two_wells_calibrator() -> Calibrator:
    """
    Fit the flow ``g(c) = c³ − 3c + 2`` from A to B (``a_to_b_builder``, 10⁶
    in A) to eight windowed observations of mean 10, with a normal model of
    ``sd = 1``, ``c`` in ``[−1.5, 4]`` and the seed 2.

    ``g`` has a local maximum ``g(−1) = 4`` below the data, so the initial
    guess ``c = −1`` is a local minimum of ``−log L`` and a fit from it alone
    stays there; the best fit is ``TWO_WELLS_FIT``, where ``g`` is the mean.
    Every row of a Latin hypercube in the upper part of the bounds reaches it.
    """
    model = build(
        a_to_b_builder(
            parameters={"c": None},
            rate="(c * c * c - 3 * c + 2) * A / A",
            amounts={"A": 1e6, "B": 0},
        )
    )
    values = [10.4, 9.7, 10.2, 9.9, 10.1, 9.6, 10.3, 9.8]
    problem = CalibrationProblem(
        observed_data=flow_observations(values, windowed=True),
        parameters=[bounded("c", -1.5, 4.0, initial_guess=-1.0)],
        observation_models={"B": NormalObservations(sd=1.0)},
        optimization_config=NelderMeadConfig(max_iterations=1000, sd_tolerance=1e-12),
        seed=2,
    )
    return Calibrator(Simulation(model), problem)


def line_calibrator(
    values: Sequence[float],
    *,
    observation_models: Mapping[str, _ObservationModel] | None = None,
    seed: int = 1,
) -> Calibrator:
    """
    Fit the line ``B(t) = b + c·t`` to `values` of B at positions 1, 2, ….

    ``a_to_b_builder`` with the constant flow ``c * A / A`` from 10⁶ in A,
    and the initial amount ``b`` of B as an initial condition: a model
    linear in its two parameters, ``c`` in ``[0, 50]`` and ``B`` in
    ``[0, 50]``. The fit minimizes the sum of squares, or, with
    `observation_models` (of the series ``"B"``), maximizes their
    likelihood.
    """
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": None}
        )
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(values, windowed=False),
        parameters=[
            bounded("c", 0.0, 50.0, initial_guess=25.0),
            bounded("B", 0.0, 50.0, kind="initial_condition", initial_guess=25.0),
        ],
        loss_function=None if observation_models else "sse",
        observation_models=dict(observation_models) if observation_models else None,
        optimization_config=NelderMeadConfig(max_iterations=2000, sd_tolerance=1e-14),
        seed=seed,
    )
    return Calibrator(Simulation(model), problem)


def swarm(
    particles: int = 12, iterations: int = 15, *, mutation: bool = True
) -> ParticleSwarmConfig:
    """A quiet swarm, with a Gaussian mutation of the particles below average."""
    if not mutation:
        return ParticleSwarmConfig(
            particles=particles, max_iterations=iterations, verbose=False
        )
    return ParticleSwarmConfig(
        particles=particles,
        max_iterations=iterations,
        verbose=False,
        mutation="gaussian",
        mutation_scale=0.1,
        mutation_probability=0.3,
        mutation_application="below_average",
    )


def abc_swarm_calibrator(config: ParticleSwarmConfig) -> Calibrator:
    """
    Fit ``k1`` and ``k2`` of ``abc_builder`` (in ``[0, 0.5]`` and
    ``[0, 0.3]``) to B and C of its run at ``k1 = 0.1``, ``k2 = 0.05``,
    every 10 steps to 100, with the swarm `config` and the seed 7.
    """
    truth = build(abc_builder(parameters={"k1": 0.1, "k2": 0.05}))
    truth_run = Simulation(truth).run(100)
    steps = range(10, 101, 10)
    problem = CalibrationProblem(
        observed_data=observed(truth_run, "B", steps) + observed(truth_run, "C", steps),
        parameters=[bounded("k1", 0.0, 0.5), bounded("k2", 0.0, 0.3)],
        loss_function="sse",
        optimization_config=config,
        seed=7,
    )
    model = build(abc_builder(parameters={"k1": None, "k2": None}))
    return Calibrator(Simulation(model), problem)
