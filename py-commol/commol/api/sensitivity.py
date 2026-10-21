"""Sensitivity analysis: what a model does when its inputs vary, with no data."""

import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from commol.api.calibrator import check_values
from commol.api.inference.registry import run_method
from commol.api.simulation import Simulation, check_seed, draw_seed
from commol.commol_rs import _commol_rs as commol_rs
from commol.context._model import engine_state
from commol.context.inference.global_sensitivity import InputSpace
from commol.context.calibration import ObservedDataPoint
from commol.context.inference.method import SensitivityMethod
from commol.context.inference.noise import NoNoise, ObservationNoise
from commol.context.inference.sample import ParameterSample

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import (
        ForwardAnalysisProtocol,
        ObservationDrawsProtocol,
        PredictiveProtocol,
    )

logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT", bound=BaseModel)


class SensitivityAnalysis:
    """
    The forward questions of a model: how uncertainty in its inputs moves its
    outputs, and which inputs matter most.

    A calibration asks which inputs fit the data; an analysis needs no data.
    It holds the simulation, what varies (`InputSpace`), the replicates of a
    stochastic engine and the seed of every random number, and every method
    of one analysis reads the same inputs. The forward model is built in Rust
    once, when the analysis is created: the values of the inputs reach the
    engine as a calibration writes them.
    """

    def __init__(
        self,
        simulation: Simulation,
        inputs: InputSpace,
        *,
        replicates: int | None = None,
        seed: int | None = None,
    ):
        """
        Build the analysis of `inputs` on `simulation`.

        Parameters
        ----------
        simulation : Simulation
            The model to analyse.
        inputs : InputSpace
            What varies.
        replicates : int | None, optional
            For a stochastic engine, required, at least 1: how many replicates
            each point runs. A point of the design of PRCC, Morris, Sobol or
            polynomial chaos holds the mean of its replicates, on the same
            seeds at every point; a propagation keeps every replicate, each
            with a seed of its own. A deterministic engine does not use it, so
            the same analysis runs on either engine.
        seed : int | None, optional
            The seed of the designs and the replicates. Required with a
            stochastic engine; otherwise None draws one from the operating
            system, which the analysis logs.

        Raises
        ------
        ValueError
            If an input is not a value of the model, the replicates or the
            seed do not fit the engine, `replicates` is below 1, or the seed
            is not a whole number from 0 to ``2**64 - 1``.
        """
        engine_type = simulation.model_definition.engine.type
        stochastic = commol_rs.engine.EngineType(engine_type).is_stochastic
        if replicates is not None and replicates < 1:
            raise ValueError(f"replicates is at least 1, got {replicates}")
        if stochastic and replicates is None:
            raise ValueError(
                f"The {engine_type} engine is stochastic: set replicates, how many "
                "replicates each point of a design runs."
            )
        if stochastic and seed is None:
            raise ValueError(
                f"The {engine_type} engine is stochastic: set seed, so the "
                "replicates and the analysis are reproducible."
            )
        self.simulation: Simulation = simulation
        self.inputs: InputSpace = inputs
        self.replicates: int | None = replicates
        if seed is not None:
            check_seed(seed)
        self._seed: int | None = seed
        check_values(
            simulation.model_definition,
            simulation.engine,
            inputs.parameters,
            inputs.initial_condition_remainder,
        )
        self._rust: "ForwardAnalysisProtocol" = commol_rs.inference.ForwardAnalysis(
            simulation.engine,
            engine_state.rust_parameters(inputs.parameters),
            distributions=inputs.rust_distributions(),
            sample=None if inputs.sample is None else inputs.sample.rust_sample,
            initial_condition_remainder=inputs.initial_condition_remainder,
            replicates=replicates if stochastic else None,
            seed=self.seed,
        )

    @property
    def rust_analysis(self) -> "ForwardAnalysisProtocol":
        """
        The forward analysis built in Rust, for the runners of methods.

        Returns
        -------
        ForwardAnalysisProtocol
            The forward model and the inputs every method reads.
        """
        return self._rust

    def rust_predictive(
        self, sample: ParameterSample, positions: list[float]
    ) -> "PredictiveProtocol":
        """
        The run of every draw of `sample` at `positions`, for a `Predictive`:
        each replicate of a stochastic engine has a seed of its own, derived
        from the seed of the analysis, as in a propagation.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the order of the inputs.
        positions : list[float]
            Positions to report at, sorted and unique.

        Returns
        -------
        PredictiveProtocol
            The runs, computed in Rust.
        """
        return self._rust.predictive(sample.rust_sample, positions)

    def rust_observation_draws(
        self,
        sample: ParameterSample,
        observed_data: list[ObservedDataPoint],
        noise: ObservationNoise,
    ) -> "ObservationDrawsProtocol":
        """
        Every draw's value of `observed_data`, for a `Predictive`.

        Parameters
        ----------
        sample : ParameterSample
            The draws, in the order of the inputs.
        observed_data : list[ObservedDataPoint]
            The observations.
        noise : NoNoise | ResidualNoise | DistributionNoise
            The observation noise; only ``NoNoise()``.

        Returns
        -------
        ObservationDrawsProtocol
            The draws, computed in Rust.

        Raises
        ------
        ValueError
            If the noise is not ``NoNoise()``: residual noise needs the
            residuals of a fit, distribution noise the observation models of a
            problem, and an analysis has neither.
        """
        if not isinstance(noise, NoNoise):
            raise ValueError(
                f"{type(noise).__name__} needs a calibration: residual noise reads "
                "the residuals of a fit and distribution noise the observation "
                "models of a problem, and a sensitivity analysis has no data. Read "
                "the observations with NoNoise()."
            )
        return self._rust.observation_draws(
            sample.rust_sample, engine_state.rust_observed_data(observed_data)
        )

    def rust_log_score(
        self,
        sample: ParameterSample,  # noqa: ARG002
        observed_data: list[ObservedDataPoint],  # noqa: ARG002
    ) -> list[float]:
        """
        Refuse the log score, which `commol.scores.log_score` asks for.

        Parameters
        ----------
        sample : ParameterSample
            The draws.
        observed_data : list[ObservedDataPoint]
            The observations.

        Raises
        ------
        ValueError
            Always: the log score reads the observation models of a problem,
            and an analysis has no data.
        """
        raise ValueError(
            "The log score needs the observation models of a calibration problem, "
            "and a sensitivity analysis has no data."
        )

    @property
    def seed(self) -> int:
        """
        The seed of the analysis' random numbers.

        Returns
        -------
        int
            The seed given, or a seed drawn from the operating system once
            per analysis and logged.
        """
        if self._seed is not None:
            return self._seed
        if not hasattr(self, "_drawn_seed"):
            self._drawn_seed = draw_seed()
            logger.info(
                "No seed set; drew the seed %d, which repeats this analysis.",
                self._drawn_seed,
            )
        return self._drawn_seed

    def run(self, method: SensitivityMethod[ResultT]) -> ResultT:
        """
        Run a sensitivity method on the inputs of this analysis.

        Parameters
        ----------
        method : SensitivityMethod[ResultT]
            The configuration of the method; its type sets the result type.

        Returns
        -------
        ResultT
            The method's result.

        Raises
        ------
        TypeError
            If `method` is not a sensitivity method; the message names its
            entry point.
        ValueError
            If the method does not take these inputs, such as a method of
            independent inputs on inputs from a sample.
        """
        return run_method(self, method, SensitivityMethod)
