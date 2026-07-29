import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, assert_never, overload

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import (
        DifferenceEquationsProtocol,
        RustModelProtocol,
    )

try:
    from commol.commol_rs import _commol_rs as commol_rs

    core = commol_rs.core
    difference = commol_rs.difference
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e

from commol.api.windows import windowed_totals
from commol.constants import ModelTypes
from commol.context.model import Model

logger = logging.getLogger(__name__)


class Simulation:
    """
    A Facade for running a simulation from a defined Model.
    """

    def __init__(self, model: Model):
        """
        Initializes the simulation engine from a Pydantic Model definition.

        Parameters
        ----------
        model : Model
            A fully constructed and validated model object.
            None values for parameters/initial conditions if used for calibration.

        """
        logging.info(f"Initializing Simulation with model: '{model.name}'")
        self.model_definition: Model = model

        self._engine: "DifferenceEquationsProtocol" = self._initialize_engine()

        self._compartments: list[str] = self._engine.compartments
        self._simulation_outputs: list[str] = getattr(
            self._engine,
            "output_names",
            self._compartments,
        )
        logging.info(
            f"Simulation engine ready. Total compartments: {len(self._compartments)}"
        )

    def _validate_all_parameters_calibrated(self, model: Model) -> None:
        """
        Validates that all parameters and initial conditions have values
        (are calibrated).

        Parameters
        ----------
        model : Model
            The model to validate.

        Raises
        ------
        ValueError
            If any parameter or initial condition has a None value.
        """
        uncalibrated_params = model.get_uncalibrated_parameters()
        uncalibrated_ics = model.get_uncalibrated_initial_conditions()

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

    def _initialize_engine(self) -> "DifferenceEquationsProtocol":
        """Internal method to set up the Rust backend."""
        logging.info("Preparing model definition for Rust serialization...")
        model_json = self.model_definition.model_dump_json()

        rust_model_instance: "RustModelProtocol" = core.Model.from_json(model_json)
        logging.info("Rust model instance created from JSON.")

        # This could be extended if you have more engine types
        if self.model_definition.dynamics.typology == ModelTypes.DIFFERENCE_EQUATIONS:
            logging.info("Initializing DifferenceEquations engine.")
            return difference.DifferenceEquations(rust_model_instance)

        raise NotImplementedError(
            (
                f"Engine for typology '{self.model_definition.dynamics.typology}' "
                f"not implemented."
            )
        )

    def _run_raw(self, num_steps: int) -> list[list[float]]:
        """
        Runs the simulation and returns the raw, high-performance output.
        This is the fastest method, returning a list of lists of floats.
        """
        logging.info(f"Running raw simulation for {num_steps} steps.")
        start = time.time()
        results = self._engine.run(num_steps)
        end = time.time()
        logging.info(f"Raw simulation complete. It tool {end - start} seconds.")
        return results

    @overload
    def run(
        self, num_steps: int, output_format: Literal["list_of_lists"]
    ) -> list[list[float]]: ...
    @overload
    def run(
        self, num_steps: int, output_format: Literal["dict_of_lists"]
    ) -> dict[str, list[float]]: ...
    @overload
    def run(self, num_steps: int) -> dict[str, list[float]]: ...
    def run(
        self,
        num_steps: int,
        output_format: Literal["dict_of_lists", "list_of_lists"] = "dict_of_lists",
    ) -> dict[str, list[float]] | list[list[float]]:
        """
        Runs the simulation and returns the output in the specified format.

        Parameters
        ----------
        num_steps : int
            The number of steps for the simulation.
        output_format : {'dict_of_lists', 'list_of_lists'}, default 'dict_of_lists'
            - 'dict_of_lists': Returns a dictionary of lists, with compartment names
                as keys.
            - 'list_of_lists': Returns a list of lists of floats, where the first level
                is the step and the second level the comptarment.

        Returns
        -------
        dict[str, list[float]] | list[list[float]]
            The simulation results in the specified format.
        """
        self._validate_all_parameters_calibrated(self.model_definition)
        raw_results = self._run_raw(num_steps)
        if output_format == "list_of_lists":
            logging.info("Returning results in 'list_of_lists' format.")
            return raw_results

        elif output_format == "dict_of_lists":
            logging.info("Transposing raw results to 'dict_of_lists' format.")
            if not raw_results:
                return {c: [] for c in self._simulation_outputs}
            transposed_results = zip(*raw_results)
            return {
                column: list(values)
                for column, values in zip(self._simulation_outputs, transposed_results)
            }

        else:
            assert_never(output_format)

    @property
    def engine(self) -> "DifferenceEquationsProtocol":
        """Get the underlying simulation engine."""
        return self._engine

    @property
    def simulation_outputs(self) -> list[str]:
        """Names of the columns returned by simulation runs."""
        return self._simulation_outputs

    def outputs_for(self, source_id: str) -> list[str]:
        """
        Names of the outputs a bin or accumulator expands into.

        Parameters
        ----------
        source_id : str
            A bin id or an accumulator id.

        Returns
        -------
        list[str]
            Output names for that id, in the order the engine reports them.

        Raises
        ------
        KeyError
            If `source_id` is not a bin or accumulator of the model.
        """
        outputs = self.model_definition.get_outputs_by_source()
        if source_id not in outputs:
            raise KeyError(
                f"'{source_id}' is not a bin or accumulator of model "
                f"'{self.model_definition.name}'. Available: {sorted(outputs)}"
            )
        return outputs[source_id]

    def group_outputs(self, source_ids: Iterable[str]) -> dict[str, list[str]]:
        """
        Group output names by the bin or accumulator they belong to.

        Parameters
        ----------
        source_ids : Iterable[str]
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
        return {source_id: self.outputs_for(source_id) for source_id in source_ids}

    def total_series(
        self,
        results: Mapping[str, Sequence[float]],
        source_ids: Iterable[str],
    ) -> list[float]:
        """
        Sum the outputs of the given bins or accumulators at each step.

        Parameters
        ----------
        results : Mapping[str, Sequence[float]]
            Simulation results in `dict_of_lists` form.
        source_ids : Iterable[str]
            Bin and accumulator ids whose outputs are summed together.

        Returns
        -------
        list[float]
            One value per step.

        Raises
        ------
        KeyError
            If an id is not a bin or accumulator of the model, or an expected
            output is missing from `results`.
        """
        names = [
            name for source_id in source_ids for name in self.outputs_for(source_id)
        ]
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
        results: Mapping[str, Sequence[float]],
        source_ids: Iterable[str],
        window_steps: int,
        at_steps: Iterable[int] | None = None,
    ) -> list[float]:
        """
        Amount the given accumulators gained over each window.

        Parameters
        ----------
        results : Mapping[str, Sequence[float]]
            Simulation results in `dict_of_lists` form.
        source_ids : Iterable[str]
            Accumulator ids whose outputs are summed before windowing.
        window_steps : int
            Length of one window, in simulation steps.
        at_steps : Iterable[int] | None, optional
            Steps at which windows close. Defaults to every complete window of
            the run. Pass the observation steps to reproduce exactly the values
            a calibration used.

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
            If `window_steps` is not positive, or a requested step has no
            complete window inside the run.
        """
        return windowed_totals(
            self.total_series(results, source_ids),
            window_steps,
            at_steps,
        )
