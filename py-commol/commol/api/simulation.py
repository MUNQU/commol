import logging
import time
from typing import TYPE_CHECKING, Literal, overload, assert_never

if TYPE_CHECKING:
    from commol.commol_rs.commol_rs import (
        DifferenceEquationsProtocol,
        RustModelProtocol,
    )

try:
    from commol.commol_rs import commol_rs

    core = commol_rs.core
    difference = commol_rs.difference
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e

from commol.context.model import Model
from commol.constants import ModelTypes


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
                return {c: [] for c in self._compartments}
            transposed_results = zip(*raw_results)
            return {
                compartment: list(values)
                for compartment, values in zip(self._compartments, transposed_results)
            }

        else:
            assert_never(output_format)

    @property
    def engine(self) -> "DifferenceEquationsProtocol":
        """Get the underlying simulation engine."""
        return self._engine
