"""Calibration of a model against observed data."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from commol._model.model import Model
    from commol.commol_rs._commol_rs import (
        CalibrationProblemProtocol,
        EngineProtocol,
        ObservationDrawsProtocol,
        PredictiveProtocol,
    )

from commol._calibration.method import (
    IdentifiabilityMethod,
    UncertaintyMethod,
    run_method,
)
from commol._data.engine_json import engine_json
from commol._fit.result import (
    CalibrationEvaluation,
    CalibrationHistory,
    CalibrationResult,
)
from commol._fit.starts import fit_result
from commol._model import engine_state
from commol._problem.problem import (
    CalibrationParameter,
    CalibrationProblem,
    ObservedDataPoint,
)
from commol._simulation.simulation import Simulation
from commol.commol_rs import _commol_rs as commol_rs
from commol.observations.noise import _ObservationNoise
from commol.uncertainty.sample import ParameterSample, SampleMethod

logger = logging.getLogger("commol")

ResultT = TypeVar("ResultT", bound=BaseModel)


class Calibrator:
    """
    Calibrate the parameters of a simulation against observed data.

    The calibration problem is built in Rust once, when the calibrator is
    created, and every optimization, prediction and uncertainty method reads
    that one problem.
    """

    def __init__(
        self,
        simulation: Simulation,
        problem: CalibrationProblem,
    ) -> None:
        """
        Initializes the calibration from a Simulation and CalibrationProblem.

        Parameters
        ----------
        simulation : Simulation
            A fully initialized Simulation object with the model to calibrate.
        problem : CalibrationProblem
            A fully constructed and validated calibration problem definition.

        Raises
        ------
        ValueError
            If a parameter, observation, position or constraint does not fit
            the model, or a setting of the problem is outside its rule; the
            message names the path of the setting, such as
            ``optimization_config.inertia_factor``.
        """
        logger.info(
            f"Initializing Calibration for model: '{simulation.model_definition.name}'"
        )
        self.simulation: Simulation = simulation
        self.problem: CalibrationProblem = problem
        self._engine: "EngineProtocol" = simulation.engine

        logger.info(
            (
                f"Calibration initialized with {len(problem.parameters)} parameters "
                f"and {len(problem.observed_data)} observed data points."
            )
        )

        self._validate_calibration_parameters()
        self._warn_constraint_positions_beyond_observations()
        self._engine_problem: "CalibrationProblemProtocol" = (
            commol_rs.inference.CalibrationProblem(self._engine, engine_json(problem))
        )
        if problem.seed is None:
            logger.info(
                "No seed set; drew the seed %d, which repeats this calibration.",
                self._engine_problem.seed,
            )

    def _predictive(
        self, sample: ParameterSample, positions: list[float]
    ) -> "PredictiveProtocol":
        """
        The run of every draw of `sample` at `positions`, for a `Predictive`:
        each replicate of a stochastic engine has a seed of its own, derived
        from the calibrator's seed.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        positions : list[float]
            Positions to report at, sorted and unique.

        Returns
        -------
        PredictiveProtocol
            The runs, computed in Rust.
        """
        return self._engine_problem.predictive(sample._rust, positions)

    def with_model(self, model: "Model") -> "Calibrator":
        """
        The same calibration problem on another model, with the same seed.

        The draws of a predictive of the copy take the seeds they take here,
        so `Scenarios` runs every scenario on common random numbers.

        Parameters
        ----------
        model : Model
            The model, such as a copy of this one with other parameter values.

        Returns
        -------
        Calibrator
            A calibrator of `model`, with `problem` and its seed set to
            `seed`.

        Raises
        ------
        ValueError
            If a parameter, observation, position or constraint of the problem
            does not fit `model`.
        """
        return Calibrator(
            Simulation(model), self.problem.model_copy(update={"seed": self.seed})
        )

    def point_sample(self, result: CalibrationResult) -> ParameterSample:
        """
        The point estimate of a fit as a sample of one draw.

        A `Predictive` or `Scenarios` of it runs the model at the fit, so a
        fit is read as any sample is.

        Parameters
        ----------
        result : CalibrationResult
            A fit of this calibration problem.

        Returns
        -------
        ParameterSample
            One draw at ``result.best_parameters``, with the names and bounds
            of the parameters of the problem and
            ``SampleMethod(name="point", settings={"seed": result.seed})``.

        Raises
        ------
        ValueError
            If the names of `result` are not those of the problem; the message
            names the names that differ.
        """
        names = self.parameter_names
        missing = [name for name in names if name not in result.best_parameters]
        extra = [name for name in result.best_parameters if name not in names]
        if missing or extra:
            raise ValueError(
                f"The result is not a fit of this problem: it lacks {missing} and "
                f"has {extra}, which the problem does not have."
            )
        return ParameterSample.equally_weighted(
            names,
            [(p.min_bound, p.max_bound) for p in self.problem.parameters],
            [[result.best_parameters[name] for name in names]],
            SampleMethod(name="point", settings={"seed": result.seed}),
        )

    def _observation_draws(
        self,
        sample: ParameterSample,
        observed_data: list[ObservedDataPoint],
        noise: _ObservationNoise,
    ) -> "ObservationDrawsProtocol":
        """
        Every draw's value of `observed_data` with `noise`, for a `Predictive`.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        observed_data : list[ObservedDataPoint]
            The observations.
        noise : NoNoise | ResidualNoise | DistributionNoise
            The observation noise, drawn from the calibrator's seed.

        Returns
        -------
        ObservationDrawsProtocol
            The draws, computed in Rust.
        """
        return self._engine_problem.observation_draws(
            sample._rust, engine_json(observed_data), engine_json(noise)
        )

    def _check_observations(self, observed_data: list[ObservedDataPoint]) -> None:
        """
        Check `observed_data` against the problem in Rust, with no run.

        Parameters
        ----------
        observed_data : list[ObservedDataPoint]
            The observations.

        Raises
        ------
        ValueError
            If the engine rejects a position, or an observation names an
            unknown output or factor, or a parameter that is not a factor.
        """
        self._engine_problem.check_observations(engine_json(observed_data))

    def _log_score(
        self, sample: ParameterSample, observed_data: list[ObservedDataPoint]
    ) -> list[float]:
        """
        The log score of each observation under the predictive of `sample`,
        for `commol.scores.log_score`.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the problem's order.
        observed_data : list[ObservedDataPoint]
            The observations.

        Returns
        -------
        list[float]
            One log score per observation, computed in Rust.
        """
        return self._engine_problem.log_score(sample._rust, engine_json(observed_data))

    @property
    def seed(self) -> int:
        """
        The seed of the calibration's random numbers.

        Returns
        -------
        int
            ``problem.seed``, or the seed the engine drew when the
            calibrator was built, which it logs.
        """
        return self._engine_problem.seed

    def run(self, starts: int = 1) -> CalibrationResult:
        """
        Runs the calibration optimization from one or more starts.

        Start 0 is the initial guess of each parameter. Each further start is a
        row of a Latin hypercube over the bounds, drawn from `seed`, so the
        starts cover the parameter space; a particle swarm takes a seed of
        its own at each start, also from `seed`. They run in parallel, on the
        threads of `commol.parallelism.threads`; the result is the best of them, the
        first start of lowest loss, and lists every start. Distinct ends with
        equal losses are a sign that the data do not identify the
        parameters.

        Parameters
        ----------
        starts : int, optional
            Number of starts, at least 1 (default 1). A verbose optimizer
            shows one start, so it needs ``starts=1``.

        Returns
        -------
        CalibrationResult
            The best start: the optimized parameter values, in the order of
            the problem's parameters, its final loss and convergence, every
            start (``starts``), each start that failed (``failed``, also
            logged as warnings) and the seed (``seed``).

        Raises
        ------
        ValueError
            If `starts` is below 1, or a verbose optimizer has more than one
            start.
        RuntimeError
            If every start fails.
        """
        measure = (
            f"{self.problem.loss_function} loss function"
            if self.problem.loss_function is not None
            else "the negative log-likelihood of its observation models"
        )
        logger.info(
            f"Starting calibration with {self.problem.optimization_config.kind} "
            f"algorithm, {measure} and {starts} start(s)."
        )
        result = fit_result(
            self._engine_problem.optimize(starts),
            self.parameter_names,
            self.seed,
            logger,
        )
        logger.info(
            (
                f"Calibration finished after {result.iterations} iterations. "
                f"Final loss: {result.final_loss:.6f}"
            )
        )
        return result

    def run_with_history(self) -> CalibrationHistory:
        """
        Runs the calibration from the initial guess and keeps every evaluation.

        Returns
        -------
        CalibrationHistory
            The result, and every evaluation of a feasible point the optimizer
            made, in order. Infeasible points, which the objective penalizes,
            are not kept.

        Raises
        ------
        RuntimeError
            If optimization fails.
        """
        logger.info("Starting calibration with history tracking")
        history = self._engine_problem.optimize_with_history()
        names = self.parameter_names
        best = history.result
        result = CalibrationResult(
            best_parameters=dict(best.best_parameters),
            final_loss=best.final_loss,
            iterations=best.iterations,
            converged=best.converged,
            termination_reason=best.termination_reason,
            seed=self.seed,
        )
        evaluations = [
            CalibrationEvaluation(
                parameters=dict(zip(names, evaluation.parameters, strict=True)),
                loss=evaluation.loss,
            )
            for evaluation in history.evaluations
        ]
        logger.info(
            f"Calibration finished after {result.iterations} iterations. "
            f"Final loss: {result.final_loss:.6f}. "
            f"Collected {len(evaluations)} evaluations."
        )
        return CalibrationHistory(result=result, evaluations=evaluations)

    def uncertainty(self, method: UncertaintyMethod[ResultT]) -> ResultT:
        """
        Run an uncertainty method on this calibration problem.

        Each method is a configuration (a subclass of `UncertaintyMethod`)
        whose type sets the result type and declares its runner. The guide's
        table "Which method
        answers which question" lists them.

        Parameters
        ----------
        method : UncertaintyMethod[ResultT]
            The configuration of the method.

        Returns
        -------
        ResultT
            The result of the method.

        Raises
        ------
        TypeError
            If `method` is an identifiability method; the message names
            ``Calibrator.identifiability``.
        ValueError
            If the configuration's type declares no runner, or
            a setting, an input, the data or the model does not fit the
            method; the message says why.
        RuntimeError
            If a computation of the method that could start fails: an engine
            run, an optimizer, every replicate, or a sampler that does not
            reach its target.
        InfeasibleStateError
            If a draw of a prediction leaves the values its model allows.
        """
        return run_method(self, method, UncertaintyMethod)

    def identifiability(self, method: IdentifiabilityMethod[ResultT]) -> ResultT:
        """
        Run an identifiability method on this calibration problem.

        Each method answers which parameters the model and the data
        determine, from another kind of evidence: `StructuralRank`,
        `Collinearity`, `SampleIdentifiability` or `StartsIdentifiability`.
        Each returns an `IdentifiabilityReport`; `IdentifiabilityReport.merge`
        joins them.

        Parameters
        ----------
        method : IdentifiabilityMethod[ResultT]
            The configuration of the method.

        Returns
        -------
        ResultT
            The result of the method.

        Raises
        ------
        TypeError
            If `method` is an uncertainty method; the message names
            ``Calibrator.uncertainty``.
        ValueError
            If the configuration's type declares no runner, or
            a setting, an input, the data or the model does not fit the
            method; the message says why.
        RuntimeError
            If a computation of the method that could start fails: an engine
            run or an optimizer.
        """
        return run_method(self, method, IdentifiabilityMethod)

    @property
    def num_parameters(self) -> int:
        """Number of parameters being calibrated."""
        return len(self.problem.parameters)

    @property
    def num_observations(self) -> int:
        """Number of observed data points."""
        return len(self.problem.observed_data)

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters being calibrated."""
        return [param.id for param in self.problem.parameters]

    def _validate_calibration_parameters(self) -> None:
        """
        Validate that all calibration parameters exist in the model.

        Raises
        ------
        ValueError
            If a parameter ID doesn't exist in the model or if a bin ID is invalid.
        """
        check_values(
            self.simulation.model_definition,
            self._engine,
            self.problem.parameters,
            self.problem.initial_condition_remainder,
        )

    def _warn_constraint_positions_beyond_observations(self) -> None:
        """Warn when constraints at positions extend beyond observed data."""
        max_observed_position = max(obs.at for obs in self.problem.observed_data)
        for constraint in self.problem.constraints:
            if constraint.at and max(constraint.at) > max_observed_position:
                logger.warning(
                    "Constraint '%s' has positions beyond the maximum observed "
                    "position (%s); simulation will be extended to evaluate them.",
                    constraint.id,
                    max_observed_position,
                )


def check_values(
    model: "Model",
    engine: "EngineProtocol",
    parameters: Sequence[CalibrationParameter],
    remainder: str | None,
) -> None:
    """
    Check that a model has every value a calibration or an analysis sets.

    The engine checks that a factor has positive bounds, and each parameter
    as its ``set_parameter`` does: one the model does not have, or one that
    a formula or a series sets, is not a value to set. The structure and the
    engine check the values of the model kind, such as initial conditions,
    and the remainder, and the engine checks that no value is missing once
    they are written.

    Parameters
    ----------
    model : Model
        The model the values are set on.
    engine : EngineProtocol
        The engine built from it.
    parameters : Sequence[CalibrationParameter]
        The values set, with their types and bounds.
    remainder : str | None
        The id that takes what the initial conditions leave, if any.

    Raises
    ------
    ValueError
        If a factor has a bound at or below 0, or the engine rejects a
        parameter or the structure or the engine rejects a value; the message
        is theirs.
    """
    engine_state.check_model_values(engine, parameters, remainder)
    model.structure.check_calibration_values(model, engine, parameters, remainder)
