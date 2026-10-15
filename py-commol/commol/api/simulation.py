"""Simulation of a model with the Rust engine, and access to its outputs."""

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import EngineProtocol

from commol.api.results import SimulationResults
from commol.api.windows import windowed_totals
from commol.context._model.engine_state import build_engine
from commol.context.model import Model
from commol.context.structure import ModelStructure

logger = logging.getLogger(__name__)


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
                "holds the structure of its kind, such as a CompartmentalStructure."
            )
        logging.info(f"Initializing Simulation with model: '{model.name}'")
        self.model_definition: Model = model
        self._build_engine(model.model_dump_json())

    def _build_engine(self, model_json: str) -> None:
        """Build the engine and its output names from a model JSON."""
        self._model_json: str = model_json
        self._engine: "EngineProtocol" = build_engine(model_json)
        self._simulation_outputs: list[str] = self._engine.output_names
        logging.info(
            f"Simulation engine ready. Outputs: {len(self._simulation_outputs)}"
        )

    def _sync_engine(self) -> None:
        """Rebuild the engine when the model changed since it was built."""
        model_json = self.model_definition.model_dump_json()
        if model_json != self._model_json:
            logging.info("Model changed since the engine was built; rebuilding it.")
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

    def run(self, num_steps: int) -> SimulationResults:
        """
        Runs the simulation from the initial state on its default grid.

        Every call starts at position 0 with the current ``model_definition``,
        so two calls with the same model give the same result. To continue a
        run, call ``simulation.engine.step()``.

        Parameters
        ----------
        num_steps : int
            The number of steps: the results hold the positions
            ``0, 1, ..., num_steps``.

        Returns
        -------
        SimulationResults
            One value per position for each output, and the positions.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value.
        """
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        self._engine.reset()
        logging.info(f"Running simulation for {num_steps} steps.")
        start = time.time()
        positions, rows = self._engine.run(num_steps)
        logging.info(f"Simulation complete. It took {time.time() - start} seconds.")
        return SimulationResults.from_rows(positions, self._simulation_outputs, rows)

    def run_at(self, at: Sequence[float]) -> SimulationResults:
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

        Returns
        -------
        SimulationResults
            One value per position for each output, and the positions.

        Raises
        ------
        InfeasibleStateError
            If a step leaves an amount below zero; the message names the
            compartment, the step and the flows out of it.
        ValueError
            If a parameter or initial condition has no value, the positions are
            not sorted and unique, or the engine does not report at one of them;
            the message is the engine's.
        """
        positions = [float(position) for position in at]
        self._validate_all_parameters_calibrated(self.model_definition)
        self._sync_engine()
        self._engine.reset()
        rows = self._engine.run_at(positions)
        return SimulationResults.from_rows(positions, self._simulation_outputs, rows)

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
        (`ModelStructure.outputs_by_source`): for a compartmental model, by
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
        outputs = self.model_definition.structure.outputs_by_source()
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

    def total_series(
        self,
        results: Mapping[str, Sequence[float]],
        ids: Iterable[str],
    ) -> list[float]:
        """
        Sum the outputs of the given bins or accumulators at each position.

        Parameters
        ----------
        results : Mapping[str, Sequence[float]]
            Simulation results, such as those of `run` or `run_at`.
        ids : Iterable[str]
            Bin and accumulator ids whose outputs are summed together.

        Returns
        -------
        list[float]
            One value per position of `results`.

        Raises
        ------
        KeyError
            If an id is not a bin or accumulator of the model, or an expected
            output is missing from `results`.
        """
        names = [name for id in ids for name in self.outputs_for(id)]
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
            Accumulator ids whose outputs are summed before windowing.
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
            If an id is not a bin or accumulator of the model, or an expected
            output is missing from `results`.
        ValueError
            If `window` is not positive, or the results have no value at the
            end or the start of a requested window.
        """
        return windowed_totals(
            self.total_series(results, ids), results.positions, window, at
        )
