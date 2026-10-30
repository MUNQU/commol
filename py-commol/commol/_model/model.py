"""The `Model` definition and its update, inspection and serialization methods."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from commol._data.engine_json import engine_json
from commol._fit.result import CalibrationResult
from commol._model.axis_units import AxisUnits, Origin, seconds_per_step
from commol._model.applied_parameters import applied_parameters_report
from commol.description.blocks import ModelDescription
from commol._model.parameter import Parameter, SeriesValue
from commol._model.process import Process
from commol._model.serialization import render_json
from commol._model.structure import ModelStructure
from commol._model.units import DEFAULT_STEP_UNIT
from commol._problem.problem import CalibrationProblem, calibrated_values
from commol.commol_rs import _commol_rs as commol_rs
from commol.engines.engine import ModelEngine

if TYPE_CHECKING:
    from commol._model.applied_parameters import AppliedParametersReport
    from commol._model.report_entry import SummarizedSample

StructureT = TypeVar("StructureT", bound=ModelStructure)
"""The structure class of a model, which sets its kind."""


class Model(BaseModel, Generic[StructureT]):
    """
    A model of any kind: a header, a process, a structure and an engine.

    The process (bins, stratifications, transitions) is written once; the
    structure lays it out and changes between kinds of model, so
    ``Model[Aggregate]`` is a model of amounts per compartment.
    `ModelBuilder.build` joins its process with a structure. A model loaded
    without a structure class reads the class from the ``kind`` and the
    ``variant`` of its structure.

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
    process : Process
        What changes and how: the bins, accumulators, stratifications and
        transitions, written once for every structure.
    structure : StructureT
        How the process is laid out, which changes between kinds of model,
        such as the amounts per compartment of an `Aggregate`.
    step_unit : str | None
        Unit of one step on the axis, of any single dimension (``"day"``,
        ``"meter"``). The unit check and `axis_units` use it; None means
        ``step``, the default increment of any axis.
    origin : datetime | None
        Calendar moment of position 0, with no time zone, for the dates of
        `axis_units`. It needs a ``step_unit`` of time whose length is a
        whole number of days or a whole fraction of a day. None means the
        axis has no dates.
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
    process: Process = Field(default_factory=Process)
    structure: SerializeAsAny[StructureT]
    step_unit: str | None = Field(
        None,
        description=(
            "Unit of one step on the axis (the independent variable, of any "
            "dimension), for the unit check and axis_units. "
            "None means 'step', the default increment of any axis."
        ),
    )
    origin: Origin | None = Field(
        None,
        description=(
            "Calendar moment of position 0, with no time zone, for the dates "
            "of axis_units. None means the axis has no dates."
        ),
    )

    @model_validator(mode="after")
    def _check_origin_unit(self) -> Self:
        """Check that an origin has a step unit of fixed calendar length."""
        if self.origin is not None:
            seconds_per_step(self.step_unit or DEFAULT_STEP_UNIT)
        return self

    @field_validator("structure", mode="before")
    @classmethod
    def _read_structure_kind(cls, value: JsonValue) -> JsonValue | ModelStructure:
        """Read a structure mapping as the class of its ``kind`` and ``variant``."""
        if not isinstance(value, Mapping):
            return value
        kind = value.get("kind")
        variant = value.get("variant")
        if not isinstance(kind, str):
            raise ValueError(
                "A model structure needs a 'kind'. Known kinds: "
                f"{sorted({known for known, _ in ModelStructure._kinds})}"
            )
        if variant is not None and not isinstance(variant, str):
            raise ValueError(f"The variant of a structure is a name, got {variant!r}")
        return ModelStructure._of_kind(kind, variant).model_validate(value)

    @classmethod
    def from_json(cls, file_path: str | Path) -> Self:
        """
        Loads a model from a JSON file.

        The method reads the specified JSON file as UTF-8, the encoding of
        JSON, parses its content, and validates it against the Model schema.

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
        return cls.model_validate_json(Path(file_path).read_text(encoding="utf-8"))

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
    def _validate_engine_runs_structure(self) -> Self:
        """Ask the engine whether it runs the kind of the structure."""
        commol_rs.engine.EngineType(engine_json(self.engine.type)).check_kind(
            engine_json(self.structure.kind)
        )
        return self

    @model_validator(mode="after")
    def _validate_unique_parameter_ids(self) -> Self:
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
    def _validate_structure(self) -> Self:
        """Check the structure against the parameters and the engine."""
        self.structure.check_model(self)
        return self

    def update_parameters(
        self,
        parameter_values: Mapping[
            str, float | str | list[tuple[float, float]] | SeriesValue | None
        ],
    ) -> None:
        """
        Update parameter values in the model.

        The model takes every value or none: the values pass the checks of
        `Parameter` and of the model, as when the model is made, before any is
        written.

        Parameters
        ----------
        parameter_values : Mapping[str, value of a parameter]
            Each parameter id mapped to its new value: a number, a formula, a
            `SeriesValue`, or None to leave it for calibration. A list of
            (position, value) points replaces the points of a series parameter
            and keeps its mode; a `SeriesValue` sets the points and the mode.

        Raises
        ------
        ValueError
            If an id is not a parameter of the model, a list is given for a
            parameter that is not a series, a value is not valid (the message
            names the parameter), or the model rejects the new values.
        """
        current = {param.id: param for param in self.parameters}
        updated = dict(current)
        for param_id, value in parameter_values.items():
            if param_id not in current:
                raise ValueError(
                    (
                        f"Parameter '{param_id}' not found in model. "
                        f"Available parameters: {', '.join(current.keys())}"
                    )
                )
            updated[param_id] = current[param_id].with_value(value)
        parameters = list(updated.values())
        self.structure.check_model(self.model_copy(update={"parameters": parameters}))
        self.parameters = parameters

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
        result: CalibrationResult | Mapping[str, float],
        problem: CalibrationProblem,
    ) -> None:
        """
        Write calibrated values back onto the model.

        Each value is routed by the ``parameter_type`` declared for it in
        ``problem``: ``parameter`` values update model parameters, ``factor``
        and ``noise`` values are ignored because they apply to observations
        rather than to the model, and the structure writes back the values of its kind
        (`ModelStructure.with_calibrated_values`). An aggregate structure
        writes ``initial_condition`` values so that the model rebuilds the
        initial state the engine scored for them.

        Parameters
        ----------
        result : CalibrationResult | Mapping
            The calibration result, or a mapping of parameter id to value such
            as one draw of a `ParameterSample` (``sample.draws()``).
        problem : CalibrationProblem
            The problem the result came from, used to resolve parameter types.

        Raises
        ------
        ValueError
            If an id is not declared in ``problem``, or the structure cannot
            write back a value of its kind.
        TypeError
            If `result` is a `ParameterSample`, which has no single set of
            parameters.
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
            if types[param_id] == "parameter"
        }
        # Computed before any mutation, so a rejected value leaves the model
        # unchanged.
        structure = self.structure.with_calibrated_values(self, problem, values)

        if parameters:
            self.update_parameters(parameters)
        self.structure = structure

    def applied_parameters(
        self,
        values: CalibrationResult | Mapping[str, float],
        problem: CalibrationProblem | None = None,
        sample: "SummarizedSample | None" = None,
        confidence_level: float | None = None,
    ) -> "AppliedParametersReport":
        """
        Report every parameter of the model, and the values of its kind.

        Call it on the model **after** `apply_calibration_parameters`. Each
        entry carries the value the model holds and whether calibration set
        it, so entries the calibration left alone appear alongside the fitted
        ones. The structure of the model adds the sections of its kind
        (`ModelStructure.report_sections`); an aggregate model adds
        ``initial_amounts`` and ``category_splits``.

        Parameters that are not a single number, such as a formula or a
        series, are omitted: they are inputs rather than fitted values.
        ``factor`` and ``noise`` parameters have no place in the model, so
        they are reported from `problem` when it is given.

        ``initial_amounts`` gives each bin and each overridden compartment of
        an aggregate model its ``amount`` the engine builds and the ``unit``
        of its bin, so bins of different units are reported side by side; no
        row is a share of a total. A bin left uncalibrated keeps a ``None``
        amount as its "needs calibration" marker even when overrides touch
        some of its compartments. ``category_splits`` gives each category its
        ``fraction`` of the group it splits, and that ``group``: the
        categories a conditional stratification is conditioned on, empty when
        it splits every bin. A stratification whose fractions differ by the
        category of an earlier one gives a row per split, under the key
        ``t1 of g1``, with that category in its ``group``.

        With a `sample` each entry gains a ``spread`` holding the weighted
        statistics of its draws (`ParameterSample.summary`). Every entry is in
        the unit it was calibrated in: a parameter in its own, an amount in
        the unit of its bin, and a category split as a fraction of its group.

        Parameters
        ----------
        values : CalibrationResult | Mapping
            The calibration result, or a mapping of parameter id to value such
            as one draw of a sample.
        problem : CalibrationProblem | None, optional
            The problem the values came from, used to report factor and noise
            parameters.
        sample : ParameterSample | None, optional
            Draws of the parameters, such as the sample of an uncertainty
            method.
        confidence_level : float | None, optional
            Confidence level of the intervals; required with a sample.

        Returns
        -------
        AppliedParametersReport
            Key ``parameters`` and the sections of the kind
            (``initial_amounts`` and ``category_splits`` for an aggregate
            model), preceded by ``confidence_level`` when there is one.

        Raises
        ------
        ValueError
            If a sample is given without a confidence level.
        """
        return applied_parameters_report(
            self, values, problem, sample, confidence_level
        )

    def describe(
        self, layout: Literal["default", "odd"] = "default"
    ) -> ModelDescription:
        """
        A description of the model, as data.

        The description is a list of sections of typed blocks: tables,
        equations, state graphs and text. `ModelDescription.render` turns it
        into text, Markdown, LaTeX or JSON, and `ModelDescription.save` writes
        it to a file; in a notebook it shows as Markdown, with the state graph
        of the model drawn.

        Parameters
        ----------
        layout : {"default", "odd"}, default="default"
            ``"default"``: overview, entities, dynamics, schedule,
            initialization, parameters and outputs. ``"odd"``: the ODD
            protocol for describing individual- and agent-based models
            (Grimm et al. 2006, 2010, 2020): purpose, entities, process
            overview and scheduling, design concepts, initialization, input
            data and submodels; an element commol cannot fill says "To be
            written by the modeller.".

        Returns
        -------
        ModelDescription
            The description.

        Raises
        ------
        ValueError
            If the layout is not one of these.

        Examples
        --------
        >>> description = model.describe()
        >>> print(description.render("text"))
        >>> description.save("model.md")
        >>> model.describe(layout="odd").save("model.tex")
        """
        from commol._model.describe import describe_model

        return describe_model(self, layout)

    def axis_units(self) -> AxisUnits:
        """
        The unit of one step of the model, and its engine, to convert values.

        The step unit is ``step_unit`` or, when the model declares none,
        ``step``, the default increment of any axis. The origin is
        ``origin``, for the dates of the axis. The engine is the one of the
        model, so the rates of `AxisUnits` mean what this engine reads.

        Returns
        -------
        AxisUnits
            The converter of the model.

        Examples
        --------
        >>> units = model.axis_units()
        >>> units.steps(2, "week")  # with step_unit="day"
        14.0
        >>> k = units.rate_from_mean_span(5, "day", keep="survival")
        """
        return AxisUnits(
            step_unit=self.step_unit or DEFAULT_STEP_UNIT,
            engine=self.engine.type,
            origin=self.origin,
        )

    def check_unit_consistency(self, verbose: bool = False) -> None:
        """
        Check unit consistency of all equations in the model.

        The structure checks them (`ModelStructure.check_unit_consistency`).
        For an aggregate model, the engine converts no value between units.
        So the check uses one step unit, ``step_unit`` or else ``step``, and
        every flow must be exactly ``<bin unit>/<step unit>``, such as
        ``item/day``. The unit is the same for every engine; what the value of
        a flow means is the engine's.

        Parameters
        ----------
        verbose : bool, default=False
            If True, prints a success message when all units are consistent.

        Raises
        ------
        UnitConsistencyError
            If a constant parameter or a bin has no unit, if ``step_unit`` is
            not one unit of one dimension, or if a flow does not have the unit
            ``<bin unit>/<step unit>``. Without ``step_unit``, the message
            says to declare it.
        ValueError
            If the model type doesn't support unit checking.
        NotImplementedError
            If the kind of the model has no unit check.

        Notes
        -----
        - Predefined variables (total, total_g1, etc.) have the bin unit
        - ``t``, a position on the axis, has the step unit; a number added
          to, subtracted from or compared with it is read in that unit
        - ``step``, a count of increments, is dimensionless
        - Mathematical constants (pi, e) are dimensionless
        """
        self.structure.check_unit_consistency(self, verbose)
