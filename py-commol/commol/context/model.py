"""The `Model` definition and its update, inspection and serialization methods."""

from collections.abc import Mapping
from pathlib import Path
from typing import Generic, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

try:
    from commol.commol_rs import _commol_rs as commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e

from commol.constants import PrintEquationsOutputFormat
from commol.context._model.serialization import render_json
from commol.context.calibration import (
    CalibrationProblem,
    CalibrationResult,
    calibrated_values,
)
from commol.context.constants import CalibrationParameterType
from commol.context.engine import ModelEngine
from commol.context.parameter import Parameter
from commol.context.probabilistic_calibration import ProbabilisticCalibrationResult
from commol.context.structure import ModelStructure

StructureT = TypeVar("StructureT", bound=ModelStructure)
"""The structure class of a model, which sets its kind."""


class Model(BaseModel, Generic[StructureT]):
    """
    A model of any kind: a header, an engine and the structure of its kind.

    ``Model[CompartmentalStructure]`` is a compartmental model, the model
    `ModelBuilder` builds. A model loaded without a structure class reads the
    class from the ``kind`` of its structure.

    Attributes
    ----------
    name : str
        A unique name that identifies the model.
    description : str | None
        A human-readable description of the model's purpose and function.
    version : str | None
        The version number of the model.
    parameters : list[Parameter]
        A list of global model parameters.
    engine : ModelEngine
        The engine that runs the model, and its settings.
    structure : StructureT
        The part of the model that depends on its kind, such as the bins,
        stratifications and transitions of a compartmental model.
    step_unit : str | None
        Unit of one step on the axis, of any single dimension (``"day"``,
        ``"meter"``). The unit check uses it; when None, the check infers it
        from the parameter units of dimension time (or uses ``step``).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name which identifies the model.")
    description: str | None = Field(
        None,
        description="Human-readable description of the model's purpose and function.",
    )
    version: str | None = Field(None, description="Version number of the model.")

    parameters: list[Parameter]
    engine: ModelEngine
    structure: SerializeAsAny[StructureT]
    step_unit: str | None = Field(
        None,
        description=(
            "Unit of one step on the axis (the independent variable, of any "
            "dimension), for the unit check. "
            "Inferred from the parameter units when None."
        ),
    )

    @field_validator("structure", mode="before")
    @classmethod
    def read_structure_kind(cls, value: object) -> object:
        """Read a structure given as a mapping as the class of its ``kind``."""
        if not isinstance(value, Mapping):
            return value
        kind = value.get("kind")
        if not isinstance(kind, str):
            raise ValueError(
                "A model structure needs a 'kind'. Known kinds: "
                f"{sorted(ModelStructure._kinds)}"
            )
        return ModelStructure.of_kind(kind).model_validate(value)

    @classmethod
    def from_json(cls, file_path: str | Path) -> Self:
        """
        Loads a model from a JSON file.

        The method reads the specified JSON file, parses its content, and validates
        it against the Model schema.

        Parameters
        ----------
        file_path : str | Path
            The path to the JSON file.

        Returns
        -------
        Model
            A validated Model instance.

        Raises
        ------
        FileNotFoundError
            If the file at `file_path` does not exist.
        pydantic.ValidationError
            If the JSON content does not conform to the Model schema.
        """
        with open(file_path, "r") as f:
            json_data = f.read()

        return cls.model_validate_json(json_data)

    def to_json(self, file_path: str | Path, indent: int = 2) -> None:
        """
        Save the model to a JSON file.

        Every field is written, including those left unset, so the file always
        reloads through :meth:`from_json` to an equal model.

        Long numeric arrays, such as a series parameter, are written on a
        single line.

        Parameters
        ----------
        file_path : str | Path
            Path of the file to write.
        indent : int, optional
            Indentation width for nested structures.
        """
        payload = self.model_dump(mode="json")
        Path(file_path).write_text(
            render_json(payload, indent=indent), encoding="utf-8"
        )

    @model_validator(mode="after")
    def validate_engine_runs_structure(self) -> Self:
        """Ask the engine whether it runs the kind of the structure."""
        commol_rs.engine.EngineType(self.engine.type).check_kind(self.structure.kind)
        return self

    @model_validator(mode="after")
    def validate_unique_parameter_ids(self) -> Self:
        """
        Validates that parameter IDs are unique.
        """
        parameter_ids = [p.id for p in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            duplicates = [
                item for item in set(parameter_ids) if parameter_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate parameter IDs found: {duplicates}")
        return self

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        """Check the structure against the parameters and the engine."""
        self.structure.check_model(self)
        return self

    def update_parameters(self, parameter_values: Mapping[str, float | None]) -> None:
        """
        Update parameter values in the model.

        Parameters
        ----------
        parameter_values : Mapping[str, float | None]
            Dictionary mapping parameter IDs to their new values.

        Raises
        ------
        ValueError
            If a parameter ID in the dictionary doesn't exist in the model.
        """
        param_dict = {param.id: param for param in self.parameters}

        for param_id, value in parameter_values.items():
            if param_id not in param_dict:
                raise ValueError(
                    (
                        f"Parameter '{param_id}' not found in model. "
                        f"Available parameters: {', '.join(param_dict.keys())}"
                    )
                )
            param_dict[param_id].value = value

    def get_uncalibrated_parameters(self) -> list[str]:
        """
        Get a list of parameter IDs that have None values (need calibration).

        Returns
        -------
        list[str]
            List of parameter IDs that require calibration.
        """
        return [param.id for param in self.parameters if param.value is None]

    def apply_calibration_parameters(
        self,
        result: CalibrationResult
        | ProbabilisticCalibrationResult
        | Mapping[str, float],
        problem: CalibrationProblem,
    ) -> None:
        """
        Write calibrated values back onto the model.

        Each value is routed by the ``parameter_type`` declared for it in
        ``problem``: ``parameter`` values update model parameters, ``factor``
        values are ignored because they apply to observations rather than to
        the model, and the structure writes back the values of its kind
        (`ModelStructure.with_calibrated_values`). A compartmental structure
        writes ``initial_condition`` values so that the model rebuilds the
        initial population the engine scored for them.

        Parameters
        ----------
        result : CalibrationResult | ProbabilisticCalibrationResult | Mapping
            The calibration outcome, or a mapping of parameter id to value.
            For a probabilistic result the point parameters of the selected
            ensemble are applied.
        problem : CalibrationProblem
            The problem the result came from, used to resolve parameter types.

        Raises
        ------
        ValueError
            If an id is not declared in ``problem``, or the structure cannot
            write back a value of its kind.
        """
        values = calibrated_values(result)
        types = {param.id: param.parameter_type for param in problem.parameters}
        unknown = sorted(set(values) - set(types))
        if unknown:
            raise ValueError(
                f"Cannot apply values with no declared parameter type: {unknown}. "
                f"Declared calibration parameters: {sorted(types)}"
            )

        parameters: dict[str, float | None] = {
            param_id: value
            for param_id, value in values.items()
            if types[param_id] == CalibrationParameterType.PARAMETER
        }
        # Computed before any mutation, so a rejected value leaves the model
        # unchanged.
        structure = self.structure.with_calibrated_values(self, problem, values)

        if parameters:
            self.update_parameters(parameters)
        self.structure = structure

    def print_equations(
        self,
        output_file: str | None = None,
        format: str = PrintEquationsOutputFormat.TEXT,
    ) -> None:
        """
        Prints the equations of the model in mathematical form.

        The structure prints them (`ModelStructure.print_equations`). A
        compartmental model displays its metadata and the system of equations
        in both compact (mathematical notation) and expanded (individual
        equations) forms.

        For DifferentialEquations models, displays equations as dX/dt = ...
        For DifferenceEquations models,
        displays equations as [X(t+Dt) - X(t)] / Dt = ...

        Parameters
        ----------
        output_file : str | None
            If provided, writes the equations to this file path instead of printing
            to console. If None, prints to console.
        format : str, default="text"
            Output format for equations. Must be one of:
            - "text": Plain text format (default)
            - "latex": LaTeX mathematical notation format

        Raises
        ------
        ValueError
            If format is not "text" or "latex"
        NotImplementedError
            If the kind of the model has no equations to print.

        Examples
        --------
        >>> model.print_equations()  # Print to console in text format
        >>> model.print_equations(output_file="equations.txt")  # Save text format
        >>> model.print_equations(format="latex")  # Print LaTeX to console
        >>> model.print_equations(
        ...     output_file="equations.txt", format="latex"
        ... )  # Save LaTeX
        """
        self.structure.print_equations(self, output_file, format)

    def check_unit_consistency(self, verbose: bool = False) -> None:
        """
        Check unit consistency of all equations in the model.

        The structure checks them (`ModelStructure.check_unit_consistency`).
        For a compartmental model, the engine applies every flow once per step
        and converts no value between units. So the check infers one step unit
        from the parameter
        units: the one unit of time (``day``, ``week``, ``step``, …) that they
        use, or ``step`` when they use none. Every flow must then be exactly
        ``<bin unit>/<step unit>``, such as ``person/day``.

        Parameters
        ----------
        verbose : bool, default=False
            If True, prints a success message when all units are consistent.

        Raises
        ------
        UnitConsistencyError
            If a constant parameter or a bin has no unit, if the parameters use
            more than one unit of time (``1/day`` and ``1/week``), or if a flow
            does not have the unit ``<bin unit>/<step unit>``.
        ValueError
            If the model type doesn't support unit checking.
        NotImplementedError
            If the kind of the model has no unit check.

        Notes
        -----
        - Predefined variables (N, N_young, etc.) have the bin unit
        - Time step variables (t, step) are dimensionless
        - Mathematical constants (pi, e) are dimensionless
        """
        self.structure.check_unit_consistency(self, verbose)
