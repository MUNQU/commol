"""Simulation of a model with the Rust engine, and access to its outputs."""

import logging
import secrets
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from commol.commol_rs import _commol_rs as commol_rs

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import EngineProtocol

from commol.api.results import SimulationReplicates, SimulationResults
from commol.api.windows import windowed_totals
from commol.context._model.engine_state import build_engine
from commol.context.model import Model
from commol.context.structure import ModelStructure

logger = logging.getLogger(__name__)

SEED_LIMIT = 2**64
"""Seeds are whole numbers from 0 to ``2**64 - 1``."""


def draw_seed() -> int:
    """
    A seed drawn from the randomness of the operating system.

    Returns
    -------
    int
        A whole number from 0 to ``2**64 - 1``.
    """
    return secrets.randbits(64)


def check_seed(seed: int) -> int:
    """
    Check that a seed is a whole number from 0 to ``2**64 - 1``.

    Parameters
    ----------
    seed : int
        The seed.

    Returns
    -------
    int
        The seed.

    Raises
    ------
    ValueError
        If the seed is not such a number.
    """
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < SEED_LIMIT
    ):
        raise ValueError(f"A seed is a whole number from 0 to 2**64 - 1, got {seed!r}.")
    return seed


class Simulation:
    """
    A Facade for running a simulation from a defined Model.

    The engine follows the model: ``run`` and ``engine`` rebuild it when
    ``model_definition`` no longer serializes to the JSON the engine was built
    from, for example after ``Model.update_parameters``.
    """

    def __init__(self, model: Model):
        """
        Initializes the simulation engine from a Pydantic Model definition.

        Parameters
        ----------
        model : Model
            A fully constructed and validated model object, of any kind.
            None values for parameters/initial conditions if used for calibration.

        Raises
        ------
        TypeError
            If the structure of the model is not a `ModelStructure`.

        """
        if not isinstance(model.structure, ModelStructure):
            raise TypeError(
                f"The structure of model '{model.name}' is a "
                f"{type(model.structure).__name__}, not a ModelStructure. A model "
                "holds the structure of its kind, such as an Aggregate."
            )
        logger.info(f"Initializing Simulation with model: '{model.name}'")
        self.model_definition: Model = model
        self._build_engine(model.model_dump_json())

    def _build_engine(self, model_json: str) -> None:
        """Build the engine and its output names from a model JSON."""
        self._model_json: str = model_json
        self._engine: "EngineProtocol" = build_engine(model_json)
        self._simulation_outputs: list[str] = self._engine.output_names
        logger.info(
            f"Simulation engine ready. Outputs: {len(self._simulation_outputs)}"
        )

    def _sync_engine(self) -> None:
        """Rebuild the engine when the model changed since it was built."""
        model_json = self.model_definition.model_dump_json()
        if model_json != self._model_json:
            logger.info("Model changed since the engine was built; rebuilding it.")
            self._build_engine(model_json)

    def _validate_all_parameters_calibrated(self, model: Model) -> None:
        """
        Validates that every parameter, and every value of the model kind, has
        a value (is calibrated).

        Parameters
        ----------
        model : Model
            The model to validate.

        Raises
        ------
        ValueError
            If any parameter, or any value of the model kind such as an
            initial condition, has a None value.
        """
        uncalibrated_params = model.get_uncalibrated_parameters()
        uncalibrated_ics = model.structure.uncalibrated_values()

        errors = []
        if uncalibrated_params:
            errors.append(
                f"Parameters requiring calibration: {', '.join(uncalibrated_params)}"
            )
        if uncalibrated_ics:
            errors.append(
                f"Initial conditions requiring calibration: "
                f"{', '.join(uncalibrated_ics)}"
            )

        if errors:
            raise ValueError(
                f"Cannot run Simulation: {'; '.join(errors)}. "
                f"Please calibrate these values before running a simulation."
            )

    @property
    def is_stochastic(self) -> bool:
        """Whether the engine of the model draws random numbers."""
        return commol_rs.engine.EngineType(
            self.model_definition.engine.type
        ).is_stochastic

    def _seed_run(self, seed: int | None) -> int | None:
        """Seed the engine of a stochastic model; None for a deterministic one."""
        if not self.is_stochastic:
            return None
        seed = draw_seed() if seed is None else check_seed(seed)
        self._engine.set_seed(seed)
        return seed

    def run(self, num_steps: int, *, seed: int | None = None) -> SimulationResults:
        """
        Runs the simulation from the initial state on its default grid.

        Every call starts at position 0 with the current ``model_definition``,
        so two calls with the same model, and the same seed for a stochastic
        engine, give the same result. To continue a run, call
        ``simulation.engine.step()``.

        Parameters
        ----------
        num_steps : int
            The number of steps: the results hold the positions
            ``0, 1, ..., num_steps``.
        seed : int | None, optional
            The seed of a stochastic engine, a whole number from 0 to
            ``2**64 - 1``. None draws one, which ``results.seed`` keeps, so
            every run can be repeated. A deterministic engine ignores it.

        Returns
        -------
        SimulationResults
            One value per position for each output, the positions, and the
            seed of a stochastic run.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value, or the seed is
            not a whole number from 0 to ``2**64 - 1``.
        RuntimeError
            If a rate cannot be evaluated or is not a number; the message
            names the flow and the position.
        """
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        used_seed = self._seed_run(seed)
        self._engine.reset()
        logger.info(f"Running simulation for {num_steps} steps.")
        start = time.time()
        positions, rows = self._engine.run(num_steps)
        logger.info(f"Simulation complete. It took {time.time() - start} seconds.")
        return SimulationResults.from_rows(
            positions, self._simulation_outputs, rows, seed=used_seed
        )

    def run_at(
        self, at: Sequence[float], *, seed: int | None = None
    ) -> SimulationResults:
        """
        Runs the simulation from the initial state and reports at given positions.

        The positions are on the axis, the independent variable of the model.
        The engine decides which positions its model type reports at: a
        DifferentialEquations model at any finite, non-negative position, from
        the dense output of its solver; a DifferenceEquations model only at
        integer steps.

        Parameters
        ----------
        at : Sequence[float]
            Sorted, unique positions to report at.
        seed : int | None, optional
            The seed of a stochastic engine; see `run`.

        Returns
        -------
        SimulationResults
            One value per position for each output, the positions, and the
            seed of a stochastic run.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value, the positions are
            not sorted and unique, or the engine does not report at one of them;
            the message is the engine's.
        RuntimeError
            If a rate cannot be evaluated or is not a number; the message
            names the flow and the position.
        """
        positions = [float(position) for position in at]
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        used_seed = self._seed_run(seed)
        self._engine.reset()
        rows = self._engine.run_at(positions)
        return SimulationResults.from_rows(
            positions, self._simulation_outputs, rows, seed=used_seed
        )

    def replicates(self, n: int, *, seed: int | None = None) -> "ReplicateRuns":
        """
        Replicates of the runs of a stochastic engine.

        The returned object has ``run`` and ``run_at``, as `Simulation`, and
        they return `SimulationReplicates`: ``n`` runs, each with its own seed,
        run in parallel in the engine. Replicate ``i`` takes the seed derived
        from ``seed`` and ``i`` (the SplitMix64 of ``seed + i``), so it gives
        the same run however many replicates run.

        Parameters
        ----------
        n : int
            How many replicates, at least 1.
        seed : int | None, optional
            The seed the replicate seeds derive from, a whole number from 0 to
            ``2**64 - 1``. None draws one, which the replicates keep.

        Returns
        -------
        ReplicateRuns
            The runner of the replicates.

        Raises
        ------
        ValueError
            If the engine is deterministic (every replicate would be equal),
            ``n`` is below 1, or the seed is not valid.
        """
        if not self.is_stochastic:
            raise ValueError(
                f"The {self.model_definition.engine.type} engine is deterministic: "
                "every replicate would be equal. Replicates need a stochastic engine."
            )
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"The number of replicates is at least 1, got {n!r}.")
        return ReplicateRuns(self, n, draw_seed() if seed is None else check_seed(seed))

    @property
    def engine(self) -> "EngineProtocol":
        """
        The engine for the current ``model_definition``.

        It is rebuilt, at step 0, when the model changed since it was built.
        Otherwise it is the same object, in the state the last ``run`` or
        ``step`` left it.
        """
        self._sync_engine()
        return self._engine

    @property
    def simulation_outputs(self) -> list[str]:
        """Names of the columns returned by simulation runs."""
        self._sync_engine()
        return self._simulation_outputs

    def outputs_for(self, id: str) -> list[str]:
        """
        Names of the outputs a source id expands into.

        The structure of the model groups its outputs by source
        (`ModelStructure.outputs_by_source`): for an aggregate model, by
        bin and accumulator.

        Parameters
        ----------
        id : str
            A source id, such as a bin id or an accumulator id.

        Returns
        -------
        list[str]
            Output names for that id, in the order the engine reports them.

        Raises
        ------
        KeyError
            If `id` is not a source of the model.
        """
        model = self.model_definition
        outputs = model.structure.outputs_by_source(model)
        if id not in outputs:
            raise KeyError(
                f"'{id}' is not a bin or accumulator of model "
                f"'{self.model_definition.name}'. Available: {sorted(outputs)}"
            )
        return outputs[id]

    def group_outputs(self, ids: Iterable[str]) -> dict[str, list[str]]:
        """
        Group output names by the bin or accumulator they belong to.

        Parameters
        ----------
        ids : Iterable[str]
            Bin and accumulator ids.

        Returns
        -------
        dict[str, list[str]]
            Each id mapped to its output names, keeping the order given.

        Raises
        ------
        KeyError
            If an id is not a bin or accumulator of the model.
        """
        return {id: self.outputs_for(id) for id in ids}

    def _series_names(self, ids: Iterable[str]) -> list[str]:
        """
        The outputs the ids stand for, in the order given.

        A source id, such as a bin or accumulator id, stands for all its
        outputs (`outputs_for`); an output name stands for itself.

        Raises
        ------
        KeyError
            If an id is neither a source nor an output of the model.
        """
        model = self.model_definition
        by_source = model.structure.outputs_by_source(model)
        outputs = [name for names in by_source.values() for name in names]
        names: list[str] = []
        for id in ids:
            if id in by_source:
                names.extend(by_source[id])
            elif id in outputs:
                names.append(id)
            else:
                raise KeyError(
                    f"'{id}' is not a bin, accumulator or output of model "
                    f"'{model.name}'. Bins and accumulators: {sorted(by_source)}. "
                    f"Outputs: {outputs}"
                )
        return names

    def total_series(
        self,
        results: Mapping[str, Sequence[float]],
        ids: Iterable[str],
    ) -> list[float]:
        """
        Sum the given outputs at each position.

        Parameters
        ----------
        results : Mapping[str, Sequence[float]]
            Simulation results, such as those of `run` or `run_at`.
        ids : Iterable[str]
            Bin and accumulator ids, each for all its outputs
            (`outputs_for`), and output names, each for itself. Their outputs
            are summed together.

        Returns
        -------
        list[float]
            One value per position of `results`.

        Raises
        ------
        KeyError
            If an id is not a bin, accumulator or output of the model, or an
            expected output is missing from `results`.
        """
        names = self._series_names(ids)
        missing = [name for name in names if name not in results]
        if missing:
            raise KeyError(f"Missing output series in results: {sorted(missing)}")
        if not names:
            return []
        return [
            sum(values)
            for values in zip(*(results[name] for name in names), strict=True)
        ]

    def windowed_totals(
        self,
        results: SimulationResults,
        ids: Iterable[str],
        window: float,
        at: Iterable[float] | None = None,
    ) -> list[float]:
        """
        Amount the given accumulators gained over each window.

        Parameters
        ----------
        results : SimulationResults
            Simulation results, with their positions.
        ids : Iterable[str]
            Accumulator ids, each for all its outputs, and output names, each
            for itself. Their outputs are summed before windowing.
        window : float
            Length of one window on the axis.
        at : Iterable[float] | None, optional
            Positions at which windows close. Defaults to every complete
            window of the results (:func:`commol.window_ends`). Pass the
            observation positions to reproduce exactly the values a
            calibration used.

        Returns
        -------
        list[float]
            One increment per window.

        Raises
        ------
        KeyError
            If an id is not a bin, accumulator or output of the model, or an
            expected output is missing from `results`.
        ValueError
            If `window` is not positive, or the results have no value at the
            end or the start of a requested window.
        """
        return windowed_totals(
            self.total_series(results, ids), results.positions, window, at
        )


class ReplicateRuns:
    """
    Runs replicates of a stochastic model; `Simulation.replicates` makes it.

    Attributes
    ----------
    simulation : Simulation
        The simulation whose model the replicates run.
    n : int
        How many replicates.
    seed : int
        The seed the replicate seeds derive from.
    """

    def __init__(self, simulation: Simulation, n: int, seed: int):
        """
        The replicates of a simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation whose model the replicates run.
        n : int
            How many replicates.
        seed : int
            The seed the replicate seeds derive from.
        """
        self.simulation = simulation
        self.n = n
        self.seed = seed

    def _replicates(self, grid: int | list[float]) -> SimulationReplicates:
        """The replicates on ``grid``: a number of steps or the positions."""
        simulation = self.simulation
        simulation._validate_all_parameters_calibrated(simulation.model_definition)
        engine = simulation.engine
        positions, seeds, runs = engine.run_replicates(self.seed, self.n, grid)
        outputs = simulation.simulation_outputs
        return SimulationReplicates(
            (
                SimulationResults.from_rows(
                    positions, outputs, rows, seed=replicate_seed
                )
                for replicate_seed, rows in zip(seeds, runs, strict=True)
            ),
            seed=self.seed,
            positions=positions,
        )

    def run(self, num_steps: int) -> SimulationReplicates:
        """
        Every replicate from the initial state, on the default grid.

        Parameters
        ----------
        num_steps : int
            The number of steps: each replicate holds the positions
            ``0, 1, ..., num_steps``.

        Returns
        -------
        SimulationReplicates
            One `SimulationResults` per replicate, each with its seed.

        Raises
        ------
        InfeasibleStateError
            If a replicate leaves the values the model allows.
        ValueError
            If a parameter or a value of the model has no value.
        """
        return self._replicates(num_steps)

    def run_at(self, at: Sequence[float]) -> SimulationReplicates:
        """
        Every replicate from the initial state, reported at given positions.

        Parameters
        ----------
        at : Sequence[float]
            Sorted, unique positions to report at.

        Returns
        -------
        SimulationReplicates
            One `SimulationResults` per replicate, each with its seed.

        Raises
        ------
        InfeasibleStateError
            If a replicate leaves the values the model allows.
        ValueError
            If a parameter has no value, the positions are not sorted and
            unique, or the engine does not report at one of them.
        """
        return self._replicates([float(position) for position in at])
