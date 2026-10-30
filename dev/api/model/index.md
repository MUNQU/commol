# Model API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The validated model that `ModelBuilder.build` returns: a header, the process written once, the structure that lays it out (`Aggregate`, with its neighbors `Complete` and `Groups`), the entries of `Aggregate.initial_state()`, the `SeriesValue` of a series parameter, the parts of the process, and `UnitConsistencyError`, the error of `Model.check_unit_consistency`. Guide: [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md), [Building Models](https://munqu.github.io/commol/latest/guide/building-models/#the-structure-aggregate), [Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md), [Units](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md). The tree of types is in [Model Structure](https://munqu.github.io/commol/latest/advanced/model-structure/index.md). `Model.describe` returns a `ModelDescription` of the [`commol.description`](https://munqu.github.io/commol/latest/api/description/index.md) module.

## Model

Bases: `BaseModel`, `Generic[StructureT]`

A model of any kind: a header, a process, a structure and an engine.

The process (bins, stratifications, transitions) is written once; the structure lays it out and changes between kinds of model, so `Model[Aggregate]` is a model of amounts per compartment. `ModelBuilder.build` joins its process with a structure. A model loaded without a structure class reads the class from the `kind` and the `variant` of its structure.

Attributes:

| Name          | Type              | Description                                                                                                             |
| ------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `name`        | `str`             | A unique name that identifies the model.                                                                                |
| `description` | \`str             | None\`                                                                                                                  |
| `version`     | \`str             | None\`                                                                                                                  |
| `parameters`  | `list[Parameter]` | A list of global model parameters.                                                                                      |
| `engine`      | `ModelEngine`     | The engine that runs the model, and its settings.                                                                       |
| `process`     | `Process`         | What changes and how: the bins, accumulators, stratifications and transitions, written once for every structure.        |
| `structure`   | `StructureT`      | How the process is laid out, which changes between kinds of model, such as the amounts per compartment of an Aggregate. |
| `step_unit`   | \`str             | None\`                                                                                                                  |
| `origin`      | \`datetime        | None\`                                                                                                                  |

Source code in `commol/_model/model.py`

```python
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
```

### Methods:

#### from_json

```python
from_json(file_path: str | Path) -> Self
```

Loads a model from a JSON file.

The method reads the specified JSON file as UTF-8, the encoding of JSON, parses its content, and validates it against the Model schema.

Parameters:

| Name        | Type  | Description | Default                    |
| ----------- | ----- | ----------- | -------------------------- |
| `file_path` | \`str | Path\`      | The path to the JSON file. |

Returns:

| Type    | Description                 |
| ------- | --------------------------- |
| `Model` | A validated Model instance. |

Raises:

| Type                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `FileNotFoundError` | If the file at file_path does not exist.                  |
| `ValidationError`   | If the JSON content does not conform to the Model schema. |

Source code in `commol/_model/model.py`

```python
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
```

#### to_json

```python
to_json(file_path: str | Path, indent: int = 2) -> None
```

Save the model to a JSON file.

Every field is written, including those left unset, so the file always reloads through :meth:`from_json` to an equal model.

Long numeric arrays, such as a series parameter, are written on a single line.

Parameters:

| Name        | Type  | Description                              | Default                    |
| ----------- | ----- | ---------------------------------------- | -------------------------- |
| `file_path` | \`str | Path\`                                   | Path of the file to write. |
| `indent`    | `int` | Indentation width for nested structures. | `2`                        |

Source code in `commol/_model/model.py`

```python
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
```

#### update_parameters

```python
update_parameters(parameter_values: Mapping[str, float | str | list[tuple[float, float]] | SeriesValue | None]) -> None
```

Update parameter values in the model.

The model takes every value or none: the values pass the checks of `Parameter` and of the model, as when the model is made, before any is written.

Parameters:

| Name               | Type                                 | Description                                                                                                                                                                                                                                                      | Default    |
| ------------------ | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `parameter_values` | `Mapping[str, value of a parameter]` | Each parameter id mapped to its new value: a number, a formula, a SeriesValue, or None to leave it for calibration. A list of (position, value) points replaces the points of a series parameter and keeps its mode; a SeriesValue sets the points and the mode. | *required* |

Raises:

| Type         | Description                                                                                                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If an id is not a parameter of the model, a list is given for a parameter that is not a series, a value is not valid (the message names the parameter), or the model rejects the new values. |

Source code in `commol/_model/model.py`

```python
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
```

#### get_uncalibrated_parameters

```python
get_uncalibrated_parameters() -> list[str]
```

Get a list of parameter IDs that have None values (need calibration).

Returns:

| Type        | Description                                     |
| ----------- | ----------------------------------------------- |
| `list[str]` | List of parameter IDs that require calibration. |

Source code in `commol/_model/model.py`

```python
def get_uncalibrated_parameters(self) -> list[str]:
    """
    Get a list of parameter IDs that have None values (need calibration).

    Returns
    -------
    list[str]
        List of parameter IDs that require calibration.
    """
    return [param.id for param in self.parameters if param.value is None]
```

#### apply_calibration_parameters

```python
apply_calibration_parameters(result: CalibrationResult | Mapping[str, float], problem: CalibrationProblem) -> None
```

Write calibrated values back onto the model.

Each value is routed by the `parameter_type` declared for it in `problem`: `parameter` values update model parameters, `factor` and `noise` values are ignored because they apply to observations rather than to the model, and the structure writes back the values of its kind (`ModelStructure.with_calibrated_values`). An aggregate structure writes `initial_condition` values so that the model rebuilds the initial state the engine scored for them.

Parameters:

| Name      | Type                 | Description                                                        | Default                                                                                                               |
| --------- | -------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `result`  | \`CalibrationResult  | Mapping\`                                                          | The calibration result, or a mapping of parameter id to value such as one draw of a ParameterSample (sample.draws()). |
| `problem` | `CalibrationProblem` | The problem the result came from, used to resolve parameter types. | *required*                                                                                                            |

Raises:

| Type         | Description                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------- |
| `ValueError` | If an id is not declared in problem, or the structure cannot write back a value of its kind. |
| `TypeError`  | If result is a ParameterSample, which has no single set of parameters.                       |

Source code in `commol/_model/model.py`

```python
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
```

#### applied_parameters

```python
applied_parameters(values: CalibrationResult | Mapping[str, float], problem: CalibrationProblem | None = None, sample: SummarizedSample | None = None, confidence_level: float | None = None) -> AppliedParametersReport
```

Report every parameter of the model, and the values of its kind.

Call it on the model **after** `apply_calibration_parameters`. Each entry carries the value the model holds and whether calibration set it, so entries the calibration left alone appear alongside the fitted ones. The structure of the model adds the sections of its kind (`ModelStructure.report_sections`); an aggregate model adds `initial_amounts` and `category_splits`.

Parameters that are not a single number, such as a formula or a series, are omitted: they are inputs rather than fitted values. `factor` and `noise` parameters have no place in the model, so they are reported from `problem` when it is given.

`initial_amounts` gives each bin and each overridden compartment of an aggregate model its `amount` the engine builds and the `unit` of its bin, so bins of different units are reported side by side; no row is a share of a total. A bin left uncalibrated keeps a `None` amount as its "needs calibration" marker even when overrides touch some of its compartments. `category_splits` gives each category its `fraction` of the group it splits, and that `group`: the categories a conditional stratification is conditioned on, empty when it splits every bin. A stratification whose fractions differ by the category of an earlier one gives a row per split, under the key `t1 of g1`, with that category in its `group`.

With a `sample` each entry gains a `spread` holding the weighted statistics of its draws (`ParameterSample.summary`). Every entry is in the unit it was calibrated in: a parameter in its own, an amount in the unit of its bin, and a category split as a fraction of its group.

Parameters:

| Name               | Type                 | Description | Default                                                                                     |
| ------------------ | -------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| `values`           | \`CalibrationResult  | Mapping\`   | The calibration result, or a mapping of parameter id to value such as one draw of a sample. |
| `problem`          | \`CalibrationProblem | None\`      | The problem the values came from, used to report factor and noise parameters.               |
| `sample`           | \`ParameterSample    | None\`      | Draws of the parameters, such as the sample of an uncertainty method.                       |
| `confidence_level` | \`float              | None\`      | Confidence level of the intervals; required with a sample.                                  |

Returns:

| Type                      | Description                                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AppliedParametersReport` | Key parameters and the sections of the kind (initial_amounts and category_splits for an aggregate model), preceded by confidence_level when there is one. |

Raises:

| Type         | Description                                      |
| ------------ | ------------------------------------------------ |
| `ValueError` | If a sample is given without a confidence level. |

Source code in `commol/_model/model.py`

```python
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
```

#### describe

```python
describe(layout: Literal['default', 'odd'] = 'default') -> ModelDescription
```

A description of the model, as data.

The description is a list of sections of typed blocks: tables, equations, state graphs and text. `ModelDescription.render` turns it into text, Markdown, LaTeX or JSON, and `ModelDescription.save` writes it to a file; in a notebook it shows as Markdown, with the state graph of the model drawn.

Parameters:

| Name     | Type             | Description                                                                                                                                                                                                                                                                                                                                                                               | Default     |
| -------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `layout` | `(default, odd)` | "default": overview, entities, dynamics, schedule, initialization, parameters and outputs. "odd": the ODD protocol for describing individual- and agent-based models (Grimm et al. 2006, 2010, 2020): purpose, entities, process overview and scheduling, design concepts, initialization, input data and submodels; an element commol cannot fill says "To be written by the modeller.". | `"default"` |

Returns:

| Type               | Description      |
| ------------------ | ---------------- |
| `ModelDescription` | The description. |

Raises:

| Type         | Description                        |
| ------------ | ---------------------------------- |
| `ValueError` | If the layout is not one of these. |

Examples:

```pycon
>>> description = model.describe()
>>> print(description.render("text"))
>>> description.save("model.md")
>>> model.describe(layout="odd").save("model.tex")
```

Source code in `commol/_model/model.py`

```python
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
```

#### axis_units

```python
axis_units() -> AxisUnits
```

The unit of one step of the model, and its engine, to convert values.

The step unit is `step_unit` or, when the model declares none, `step`, the default increment of any axis. The origin is `origin`, for the dates of the axis. The engine is the one of the model, so the rates of `AxisUnits` mean what this engine reads.

Returns:

| Type        | Description                 |
| ----------- | --------------------------- |
| `AxisUnits` | The converter of the model. |

Examples:

```pycon
>>> units = model.axis_units()
>>> units.steps(2, "week")  # with step_unit="day"
14.0
>>> k = units.rate_from_mean_span(5, "day", keep="survival")
```

Source code in `commol/_model/model.py`

```python
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
```

#### check_unit_consistency

```python
check_unit_consistency(verbose: bool = False) -> None
```

Check unit consistency of all equations in the model.

The structure checks them (`ModelStructure.check_unit_consistency`). For an aggregate model, the engine converts no value between units. So the check uses one step unit, `step_unit` or else `step`, and every flow must be exactly `<bin unit>/<step unit>`, such as `item/day`. The unit is the same for every engine; what the value of a flow means is the engine's.

Parameters:

| Name      | Type   | Description                                                      | Default |
| --------- | ------ | ---------------------------------------------------------------- | ------- |
| `verbose` | `bool` | If True, prints a success message when all units are consistent. | `False` |

Raises:

| Type                   | Description                                                                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `UnitConsistencyError` | If a constant parameter or a bin has no unit, if step_unit is not one unit of one dimension, or if a flow does not have the unit <bin unit>/<step unit>. Without step_unit, the message says to declare it. |
| `ValueError`           | If the model type doesn't support unit checking.                                                                                                                                                            |
| `NotImplementedError`  | If the kind of the model has no unit check.                                                                                                                                                                 |

Notes

- Predefined variables (total, total_g1, etc.) have the bin unit
- `t`, a position on the axis, has the step unit; a number added to, subtracted from or compared with it is read in that unit
- `step`, a count of increments, is dimensionless
- Mathematical constants (pi, e) are dimensionless

Source code in `commol/_model/model.py`

```python
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
```

## Process

Bases: `BaseModel`

What changes in a model and how, written once for every structure.

Attributes:

| Name              | Type                   | Description                                                                               |
| ----------------- | ---------------------- | ----------------------------------------------------------------------------------------- |
| `bins`            | `list[Bin]`            | The states of the process. Stratifications expand each one into compartments.             |
| `accumulators`    | `list[Accumulator]`    | Counters of the flow of the transitions that name them.                                   |
| `stratifications` | `list[Stratification]` | Categorical subdivisions of the bins, in expansion order.                                 |
| `matrices`        | `list[CategoryMatrix]` | Matrices between the categories of a stratification, which rates read through matrix_sum. |
| `transitions`     | `list[Transition]`     | Flows between bins, or between the categories of one stratification.                      |

Source code in `commol/_model/process.py`

```python
class Process(BaseModel):
    """
    What changes in a model and how, written once for every structure.

    Attributes
    ----------
    bins : list[Bin]
        The states of the process. Stratifications expand each one into
        compartments.
    accumulators : list[Accumulator]
        Counters of the flow of the transitions that name them.
    stratifications : list[Stratification]
        Categorical subdivisions of the bins, in expansion order.
    matrices : list[CategoryMatrix]
        Matrices between the categories of a stratification, which rates
        read through ``matrix_sum``.
    transitions : list[Transition]
        Flows between bins, or between the categories of one stratification.
    """

    model_config = ConfigDict(extra="forbid")

    bins: list[Bin] = Field(default_factory=list)
    accumulators: list[Accumulator] = Field(default_factory=list)
    stratifications: list[Stratification] = Field(default_factory=list)
    matrices: list[CategoryMatrix] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> Self:
        """Validate that bin, stratification and transition ids are unique."""
        for label, ids in (
            ("bin", [item.id for item in self.bins]),
            ("stratification", [item.id for item in self.stratifications]),
            ("transition", [item.id for item in self.transitions]),
        ):
            if len(ids) != len(set(ids)):
                duplicates = [item for item in set(ids) if ids.count(item) > 1]
                raise ValueError(f"Duplicate {label} IDs found: {duplicates}")
        return self

    @model_validator(mode="after")
    def _validate_transition_ids(self) -> Self:
        """
        Check that each transition moves between bins, or between the
        categories of one stratification.
        """
        bin_ids = {bin_item.id for bin_item in self.bins}
        category_to_stratification = {
            cat: strat.id for strat in self.stratifications for cat in strat.categories
        }
        categories_ids = set(category_to_stratification)
        bin_and_categories_ids = bin_ids | categories_ids

        for transition in self.transitions:
            transition_ids = set(transition.source) | set(transition.target)

            if not transition_ids.issubset(bin_and_categories_ids):
                invalid_ids = transition_ids - bin_and_categories_ids
                raise ValueError(
                    f"Transition '{transition.id}' contains invalid ids: "
                    f"{invalid_ids}. Ids must be defined in Bin ids "
                    f"or Stratification Categories."
                )

            is_bin_flow = transition_ids.issubset(bin_ids)
            is_stratification_flow = transition_ids.issubset(categories_ids)

            if not is_bin_flow and not is_stratification_flow:
                raise ValueError(
                    f"Transition '{transition.id}' mixes id types. "
                    f"Found Bin ids ({transition_ids & bin_ids}) and "
                    f"Stratification Categories ids "
                    f"({transition_ids & categories_ids}). "
                    "Transitions must be purely Bin flow or purely "
                    f"Stratification flow."
                )

            if is_stratification_flow:
                parent_stratification_ids = {
                    category_to_stratification[cat_id] for cat_id in transition_ids
                }
                if len(parent_stratification_ids) > 1:
                    raise ValueError(
                        f"Transition '{transition.id}' is a Stratification flow "
                        f"but involves categories from multiple stratifications: "
                        f"{', '.join(sorted(parent_stratification_ids))}. A single "
                        f"transition must only move between categories belonging "
                        f"to the same parent stratification."
                    )
        return self

    def is_empty(self) -> bool:
        """
        Whether the process holds nothing.

        Returns
        -------
        bool
            True when it has no bin, accumulator, stratification, matrix or
            transition.
        """
        return not (
            self.bins
            or self.accumulators
            or self.stratifications
            or self.matrices
            or self.transitions
        )

    def bin(self, bin_id: str) -> Bin | None:
        """
        The bin with an id.

        Parameters
        ----------
        bin_id : str
            Id of a bin.

        Returns
        -------
        Bin | None
            The bin, or None when the process has no bin with that id.
        """
        return next((item for item in self.bins if item.id == bin_id), None)
```

### Methods:

#### is_empty

```python
is_empty() -> bool
```

Whether the process holds nothing.

Returns:

| Type   | Description                                                                 |
| ------ | --------------------------------------------------------------------------- |
| `bool` | True when it has no bin, accumulator, stratification, matrix or transition. |

Source code in `commol/_model/process.py`

```python
def is_empty(self) -> bool:
    """
    Whether the process holds nothing.

    Returns
    -------
    bool
        True when it has no bin, accumulator, stratification, matrix or
        transition.
    """
    return not (
        self.bins
        or self.accumulators
        or self.stratifications
        or self.matrices
        or self.transitions
    )
```

#### bin

```python
bin(bin_id: str) -> Bin | None
```

The bin with an id.

Parameters:

| Name     | Type  | Description  | Default    |
| -------- | ----- | ------------ | ---------- |
| `bin_id` | `str` | Id of a bin. | *required* |

Returns:

| Type  | Description |
| ----- | ----------- |
| \`Bin | None\`      |

Source code in `commol/_model/process.py`

```python
def bin(self, bin_id: str) -> Bin | None:
    """
    The bin with an id.

    Parameters
    ----------
    bin_id : str
        Id of a bin.

    Returns
    -------
    Bin | None
        The bin, or None when the process has no bin with that id.
    """
    return next((item for item in self.bins if item.id == bin_id), None)
```

## ModelStructure

Bases: `BaseModel`, `ABC`

How the process of a model is laid out: the part that depends on its kind.

Each kind subclasses it and sets `kind` to a `Literal` default, its name in the engine's list of kinds. A variant subclasses the structure of its kind and sets `variant` to a `Literal` default. A subclass that does not implement an abstract method cannot be made: the error names the method.

Attributes:

| Name      | Type  | Description                                               |
| --------- | ----- | --------------------------------------------------------- |
| `kind`    | `str` | Name of the model kind, the tag of the structure in JSON. |
| `variant` | \`str | None\`                                                    |

Source code in `commol/_model/structure.py`

```python
class ModelStructure(BaseModel, ABC):
    """
    How the process of a model is laid out: the part that depends on its kind.

    Each kind subclasses it and sets ``kind`` to a ``Literal`` default, its
    name in the engine's list of kinds. A variant subclasses the structure of
    its kind and sets ``variant`` to a ``Literal`` default. A subclass that
    does not implement an abstract method cannot be made: the error names
    the method.

    Attributes
    ----------
    kind : str
        Name of the model kind, the tag of the structure in JSON.
    variant : str | None
        Name of the variant, or None for the structure of the kind itself.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    variant: str | None = None

    _kinds: ClassVar[dict[tuple[str, str | None], type["ModelStructure"]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Register a concrete subclass under its ``kind`` and ``variant``."""
        super().__pydantic_init_subclass__(**kwargs)
        kind = cls.model_fields["kind"].default
        variant = cls.model_fields["variant"].default
        if inspect.isabstract(cls) or not isinstance(kind, str):
            return
        key = (kind, variant)
        known = ModelStructure._kinds.get(key)
        if known is not None and known.__qualname__ != cls.__qualname__:
            raise TypeError(
                f"Model kind '{kind}' with variant {variant!r} is already the "
                f"structure of {known.__qualname__}"
            )
        if variant is not None and (kind, None) not in ModelStructure._kinds:
            raise TypeError(
                f"The variant {variant!r} of {cls.__qualname__} needs the "
                f"structure of its kind '{kind}' first"
            )
        ModelStructure._kinds[key] = cls

    @classmethod
    def _of_kind(cls, kind: str, variant: str | None = None) -> type["ModelStructure"]:
        """
        The structure class of a model kind, or of one of its variants.

        Parameters
        ----------
        kind : str
            Name of the kind, such as ``"aggregate"``.
        variant : str | None, default=None
            Name of a variant of the kind, or None for the kind itself.

        Returns
        -------
        type[ModelStructure]
            The subclass registered for them.

        Raises
        ------
        ValueError
            If no structure class has that kind and variant.
        """
        structure = cls._kinds.get((kind, variant))
        if structure is not None:
            return structure
        kinds = sorted({known for known, _ in cls._kinds})
        if kind not in kinds:
            raise ValueError(f"Unknown model kind '{kind}'. Known kinds: {kinds}")
        variants = sorted(
            known for known_kind, known in cls._kinds if known_kind == kind and known
        )
        raise ValueError(
            f"Unknown variant {variant!r} of the {kind} kind. Known variants: "
            f"{variants}"
        )

    @abstractmethod
    def check_model(self, model: "Model[Self]") -> None:
        """
        Check the structure against the rest of its model.

        The model calls it when it is made, so a model that cannot be run as
        written is rejected then.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Raises
        ------
        ValueError
            If the structure does not agree with the parameters or the engine.
        """

    @abstractmethod
    def outputs_by_source(self, model: "Model[Self]") -> dict[str, list[str]]:
        """
        The output names of the model, grouped by the id they come from.

        Reports and plots name a group of outputs by this id and sum them.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Returns
        -------
        dict[str, list[str]]
            Each id mapped to its output names, in engine order.
        """

    def series_label(
        self,
        model: "Model[Self]",  # noqa: ARG002
        series_id: str,
    ) -> SeriesLabel:
        """
        How a plot names a series and its unit.

        Parameters
        ----------
        model : Model
            The model that holds this structure.
        series_id : str
            An output name or an id of `outputs_by_source`.

        Returns
        -------
        SeriesLabel
            The id itself and no unit, unless the kind knows better.
        """
        return SeriesLabel(series_id, None)

    def calibration_value_types(self) -> frozenset[str]:
        """
        Types of value of this kind that calibration may set.

        ``parameter`` and ``factor`` are not here: every kind accepts them.
        The engine's list of kinds declares the others.

        Returns
        -------
        frozenset[str]
            For example ``{"initial_condition"}`` for an aggregate model.
        """
        return frozenset(
            name
            for name in commol_rs.engine.ModelKind(engine_json(self.kind)).value_types()
        )

    def uncalibrated_values(self) -> list[str]:
        """
        Ids of the values of this kind that need calibration before a run.

        Returns
        -------
        list[str]
            Empty unless the kind has values left unset.
        """
        return []

    def check_calibration_values(
        self,
        model: "Model[Self]",  # noqa: ARG002
        engine: "EngineProtocol",  # noqa: ARG002
        parameters: Sequence["CalibrationParameter"],  # noqa: ARG002
        remainder: str | None,  # noqa: ARG002
    ) -> None:
        """
        Check the values of this kind that a calibration or a sensitivity
        analysis sets.

        The engine has resolved each value and the remainder when this runs;
        a kind checks here what only the Python definition knows, such as the
        units of the bins. The default checks nothing.

        Parameters
        ----------
        model : Model
            The model the values are set on.
        engine : EngineProtocol
            The engine built from it.
        parameters : Sequence[CalibrationParameter]
            The values set, with their types and bounds.
        remainder : str | None
            The id that takes what the initial conditions leave of the sum
            the kind keeps, if any.
        """

    def with_calibrated_values(
        self,
        model: "Model[Self]",  # noqa: ARG002
        problem: "CalibrationProblem",  # noqa: ARG002
        values: Mapping[str, float],  # noqa: ARG002
    ) -> Self:
        """
        The structure with the calibrated values of its kind written back.

        Parameters
        ----------
        model : Model
            The model the problem was built on. It is not changed.
        problem : CalibrationProblem
            The problem, for the types of its parameters.
        values : Mapping[str, float]
            Calibrated value per parameter id.

        Returns
        -------
        Self
            A structure; this one when the kind sets no values of its own.
        """
        return self

    def report_sections(
        self,
        model: "Model[Self]",  # noqa: ARG002
        calibrated_ids: set[str],  # noqa: ARG002
        parameter_statistics: "Mapping[str, EntrySummary] | None",  # noqa: ARG002
    ) -> ReportSections:
        """
        Sections of this kind in a report of applied values.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, EntrySummary] | None
            Statistics of each id from a sample, or None without one.

        Returns
        -------
        ReportSections
            Each section by its key in the report; none by default.
        """
        return {}

    def describe(
        self,
        model: "Model[Self]",  # noqa: ARG002
    ) -> "list[Section]":
        """
        The sections of a description that this kind fills.

        `Model.describe` adds them to the sections every kind shares
        (overview, schedule, parameters, outputs). A kind fills the sections
        it knows, by key: ``entities`` (what holds the state), ``dynamics``
        (how it changes), ``initialization`` (the state it starts from) and
        ``design_concepts`` (a `Table` of elements of the ODD protocol and
        their text). A section with no blocks is left out of a rendering.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Returns
        -------
        list[Section]
            The sections of the kind; none by default.
        """
        return []

    def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
        """
        Check the units of the model.

        Raises
        ------
        NotImplementedError
            If the kind has no unit check.
        """
        raise NotImplementedError(f"A {self.kind} model has no unit check")
```

### Methods:

#### __pydantic_init_subclass__

```python
__pydantic_init_subclass__(**kwargs: object) -> None
```

Register a concrete subclass under its `kind` and `variant`.

Source code in `commol/_model/structure.py`

```python
@classmethod
def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
    """Register a concrete subclass under its ``kind`` and ``variant``."""
    super().__pydantic_init_subclass__(**kwargs)
    kind = cls.model_fields["kind"].default
    variant = cls.model_fields["variant"].default
    if inspect.isabstract(cls) or not isinstance(kind, str):
        return
    key = (kind, variant)
    known = ModelStructure._kinds.get(key)
    if known is not None and known.__qualname__ != cls.__qualname__:
        raise TypeError(
            f"Model kind '{kind}' with variant {variant!r} is already the "
            f"structure of {known.__qualname__}"
        )
    if variant is not None and (kind, None) not in ModelStructure._kinds:
        raise TypeError(
            f"The variant {variant!r} of {cls.__qualname__} needs the "
            f"structure of its kind '{kind}' first"
        )
    ModelStructure._kinds[key] = cls
```

#### check_model

```python
check_model(model: Model[Self]) -> None
```

Check the structure against the rest of its model.

The model calls it when it is made, so a model that cannot be run as written is rejected then.

Parameters:

| Name    | Type    | Description                          | Default    |
| ------- | ------- | ------------------------------------ | ---------- |
| `model` | `Model` | The model that holds this structure. | *required* |

Raises:

| Type         | Description                                                        |
| ------------ | ------------------------------------------------------------------ |
| `ValueError` | If the structure does not agree with the parameters or the engine. |

Source code in `commol/_model/structure.py`

```python
@abstractmethod
def check_model(self, model: "Model[Self]") -> None:
    """
    Check the structure against the rest of its model.

    The model calls it when it is made, so a model that cannot be run as
    written is rejected then.

    Parameters
    ----------
    model : Model
        The model that holds this structure.

    Raises
    ------
    ValueError
        If the structure does not agree with the parameters or the engine.
    """
```

#### outputs_by_source

```python
outputs_by_source(model: Model[Self]) -> dict[str, list[str]]
```

The output names of the model, grouped by the id they come from.

Reports and plots name a group of outputs by this id and sum them.

Parameters:

| Name    | Type    | Description                          | Default    |
| ------- | ------- | ------------------------------------ | ---------- |
| `model` | `Model` | The model that holds this structure. | *required* |

Returns:

| Type                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `dict[str, list[str]]` | Each id mapped to its output names, in engine order. |

Source code in `commol/_model/structure.py`

```python
@abstractmethod
def outputs_by_source(self, model: "Model[Self]") -> dict[str, list[str]]:
    """
    The output names of the model, grouped by the id they come from.

    Reports and plots name a group of outputs by this id and sum them.

    Parameters
    ----------
    model : Model
        The model that holds this structure.

    Returns
    -------
    dict[str, list[str]]
        Each id mapped to its output names, in engine order.
    """
```

#### series_label

```python
series_label(model: Model[Self], series_id: str) -> SeriesLabel
```

How a plot names a series and its unit.

Parameters:

| Name        | Type    | Description                                   | Default    |
| ----------- | ------- | --------------------------------------------- | ---------- |
| `model`     | `Model` | The model that holds this structure.          | *required* |
| `series_id` | `str`   | An output name or an id of outputs_by_source. | *required* |

Returns:

| Type          | Description                                              |
| ------------- | -------------------------------------------------------- |
| `SeriesLabel` | The id itself and no unit, unless the kind knows better. |

Source code in `commol/_model/structure.py`

```python
def series_label(
    self,
    model: "Model[Self]",  # noqa: ARG002
    series_id: str,
) -> SeriesLabel:
    """
    How a plot names a series and its unit.

    Parameters
    ----------
    model : Model
        The model that holds this structure.
    series_id : str
        An output name or an id of `outputs_by_source`.

    Returns
    -------
    SeriesLabel
        The id itself and no unit, unless the kind knows better.
    """
    return SeriesLabel(series_id, None)
```

#### calibration_value_types

```python
calibration_value_types() -> frozenset[str]
```

Types of value of this kind that calibration may set.

`parameter` and `factor` are not here: every kind accepts them. The engine's list of kinds declares the others.

Returns:

| Type             | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `frozenset[str]` | For example {"initial_condition"} for an aggregate model. |

Source code in `commol/_model/structure.py`

```python
def calibration_value_types(self) -> frozenset[str]:
    """
    Types of value of this kind that calibration may set.

    ``parameter`` and ``factor`` are not here: every kind accepts them.
    The engine's list of kinds declares the others.

    Returns
    -------
    frozenset[str]
        For example ``{"initial_condition"}`` for an aggregate model.
    """
    return frozenset(
        name
        for name in commol_rs.engine.ModelKind(engine_json(self.kind)).value_types()
    )
```

#### uncalibrated_values

```python
uncalibrated_values() -> list[str]
```

Ids of the values of this kind that need calibration before a run.

Returns:

| Type        | Description                                  |
| ----------- | -------------------------------------------- |
| `list[str]` | Empty unless the kind has values left unset. |

Source code in `commol/_model/structure.py`

```python
def uncalibrated_values(self) -> list[str]:
    """
    Ids of the values of this kind that need calibration before a run.

    Returns
    -------
    list[str]
        Empty unless the kind has values left unset.
    """
    return []
```

#### check_calibration_values

```python
check_calibration_values(model: Model[Self], engine: EngineProtocol, parameters: Sequence[CalibrationParameter], remainder: str | None) -> None
```

Check the values of this kind that a calibration or a sensitivity analysis sets.

The engine has resolved each value and the remainder when this runs; a kind checks here what only the Python definition knows, such as the units of the bins. The default checks nothing.

Parameters:

| Name         | Type                             | Description                                  | Default                                                                                |
| ------------ | -------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| `model`      | `Model`                          | The model the values are set on.             | *required*                                                                             |
| `engine`     | `EngineProtocol`                 | The engine built from it.                    | *required*                                                                             |
| `parameters` | `Sequence[CalibrationParameter]` | The values set, with their types and bounds. | *required*                                                                             |
| `remainder`  | \`str                            | None\`                                       | The id that takes what the initial conditions leave of the sum the kind keeps, if any. |

Source code in `commol/_model/structure.py`

```python
def check_calibration_values(
    self,
    model: "Model[Self]",  # noqa: ARG002
    engine: "EngineProtocol",  # noqa: ARG002
    parameters: Sequence["CalibrationParameter"],  # noqa: ARG002
    remainder: str | None,  # noqa: ARG002
) -> None:
    """
    Check the values of this kind that a calibration or a sensitivity
    analysis sets.

    The engine has resolved each value and the remainder when this runs;
    a kind checks here what only the Python definition knows, such as the
    units of the bins. The default checks nothing.

    Parameters
    ----------
    model : Model
        The model the values are set on.
    engine : EngineProtocol
        The engine built from it.
    parameters : Sequence[CalibrationParameter]
        The values set, with their types and bounds.
    remainder : str | None
        The id that takes what the initial conditions leave of the sum
        the kind keeps, if any.
    """
```

#### with_calibrated_values

```python
with_calibrated_values(model: Model[Self], problem: CalibrationProblem, values: Mapping[str, float]) -> Self
```

The structure with the calibrated values of its kind written back.

Parameters:

| Name      | Type                  | Description                                            | Default    |
| --------- | --------------------- | ------------------------------------------------------ | ---------- |
| `model`   | `Model`               | The model the problem was built on. It is not changed. | *required* |
| `problem` | `CalibrationProblem`  | The problem, for the types of its parameters.          | *required* |
| `values`  | `Mapping[str, float]` | Calibrated value per parameter id.                     | *required* |

Returns:

| Type   | Description                                                    |
| ------ | -------------------------------------------------------------- |
| `Self` | A structure; this one when the kind sets no values of its own. |

Source code in `commol/_model/structure.py`

```python
def with_calibrated_values(
    self,
    model: "Model[Self]",  # noqa: ARG002
    problem: "CalibrationProblem",  # noqa: ARG002
    values: Mapping[str, float],  # noqa: ARG002
) -> Self:
    """
    The structure with the calibrated values of its kind written back.

    Parameters
    ----------
    model : Model
        The model the problem was built on. It is not changed.
    problem : CalibrationProblem
        The problem, for the types of its parameters.
    values : Mapping[str, float]
        Calibrated value per parameter id.

    Returns
    -------
    Self
        A structure; this one when the kind sets no values of its own.
    """
    return self
```

#### report_sections

```python
report_sections(model: Model[Self], calibrated_ids: set[str], parameter_statistics: Mapping[str, EntrySummary] | None) -> ReportSections
```

Sections of this kind in a report of applied values.

Parameters:

| Name                   | Type                         | Description                              | Default                                                   |
| ---------------------- | ---------------------------- | ---------------------------------------- | --------------------------------------------------------- |
| `model`                | `Model`                      | The model that holds the applied values. | *required*                                                |
| `calibrated_ids`       | `set[str]`                   | Ids that calibration set.                | *required*                                                |
| `parameter_statistics` | \`Mapping[str, EntrySummary] | None\`                                   | Statistics of each id from a sample, or None without one. |

Returns:

| Type             | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `ReportSections` | Each section by its key in the report; none by default. |

Source code in `commol/_model/structure.py`

```python
def report_sections(
    self,
    model: "Model[Self]",  # noqa: ARG002
    calibrated_ids: set[str],  # noqa: ARG002
    parameter_statistics: "Mapping[str, EntrySummary] | None",  # noqa: ARG002
) -> ReportSections:
    """
    Sections of this kind in a report of applied values.

    Parameters
    ----------
    model : Model
        The model that holds the applied values.
    calibrated_ids : set[str]
        Ids that calibration set.
    parameter_statistics : Mapping[str, EntrySummary] | None
        Statistics of each id from a sample, or None without one.

    Returns
    -------
    ReportSections
        Each section by its key in the report; none by default.
    """
    return {}
```

#### describe

```python
describe(model: Model[Self]) -> list[Section]
```

The sections of a description that this kind fills.

`Model.describe` adds them to the sections every kind shares (overview, schedule, parameters, outputs). A kind fills the sections it knows, by key: `entities` (what holds the state), `dynamics` (how it changes), `initialization` (the state it starts from) and `design_concepts` (a `Table` of elements of the ODD protocol and their text). A section with no blocks is left out of a rendering.

Parameters:

| Name    | Type    | Description                          | Default    |
| ------- | ------- | ------------------------------------ | ---------- |
| `model` | `Model` | The model that holds this structure. | *required* |

Returns:

| Type            | Description                                |
| --------------- | ------------------------------------------ |
| `list[Section]` | The sections of the kind; none by default. |

Source code in `commol/_model/structure.py`

```python
def describe(
    self,
    model: "Model[Self]",  # noqa: ARG002
) -> "list[Section]":
    """
    The sections of a description that this kind fills.

    `Model.describe` adds them to the sections every kind shares
    (overview, schedule, parameters, outputs). A kind fills the sections
    it knows, by key: ``entities`` (what holds the state), ``dynamics``
    (how it changes), ``initialization`` (the state it starts from) and
    ``design_concepts`` (a `Table` of elements of the ODD protocol and
    their text). A section with no blocks is left out of a rendering.

    Parameters
    ----------
    model : Model
        The model that holds this structure.

    Returns
    -------
    list[Section]
        The sections of the kind; none by default.
    """
    return []
```

#### check_unit_consistency

```python
check_unit_consistency(model: Model[Self], verbose: bool) -> None
```

Check the units of the model.

Raises:

| Type                  | Description                    |
| --------------------- | ------------------------------ |
| `NotImplementedError` | If the kind has no unit check. |

Source code in `commol/_model/structure.py`

```python
def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
    """
    Check the units of the model.

    Raises
    ------
    NotImplementedError
        If the kind has no unit check.
    """
    raise NotImplementedError(f"A {self.kind} model has no unit check")
```

## Aggregate

Bases: `ModelStructure`, `InitialConditions`

Amounts per compartment: how much starts where, and who the neighbors of a unit are.

The bins of the process expand over its stratifications into compartments. `bin_amounts` gives the amount of each bin at the start, in the unit of the bin, `stratification_fractions` splits it among the categories, and `compartment_amounts` overrides single compartments. Bins of different units, such as a count and a concentration, need no total over them.

Attributes:

| Name                       | Type                   | Description                                                                                                                                                                                                                                                                                           |
| -------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`                     | `Literal['aggregate']` | The model kind.                                                                                                                                                                                                                                                                                       |
| `variant`                  | \`str                  | None\`                                                                                                                                                                                                                                                                                                |
| `bin_amounts`              | \`dict\[str, float     | None\]\`                                                                                                                                                                                                                                                                                              |
| `stratification_fractions` | `dict[str, dict]`      | The fraction of each category, by stratification id, a dict[str, float] such as {"group": {"g1": 0.3, "g2": 0.7}}: the split of the amount of every bin it applies to. A stratification may instead split each category of a stratification declared before it by its own fractions, a dict\[str, str |
| `compartment_amounts`      | `dict[str, float]`     | Amounts of single expanded compartments, such as {"A_g1": 0.0}, applied after the expansion. The names are checked against the engine when the model is built.                                                                                                                                        |
| `neighbors`                | \`Complete             | Groups                                                                                                                                                                                                                                                                                                |

Source code in `commol/_model/aggregate.py`

```python
class Aggregate(ModelStructure, InitialConditions):
    """
    Amounts per compartment: how much starts where, and who the neighbors of
    a unit are.

    The bins of the process expand over its stratifications into
    compartments. ``bin_amounts`` gives the amount of each bin at the start,
    in the unit of the bin, ``stratification_fractions`` splits it among the
    categories, and ``compartment_amounts`` overrides single compartments.
    Bins of different units, such as a count and a concentration, need no
    total over them.

    Attributes
    ----------
    kind : Literal["aggregate"]
        The model kind.
    variant : str | None
        The variant of the structure (see `ModelStructure`), None for
        `Aggregate` itself.
    bin_amounts : dict[str, float | None]
        The amount of each bin at the start, by bin id, in the unit of the
        bin, such as ``{"A": 990, "B": 10, "C": 1e5}``; finite and not
        negative. None leaves an amount for calibration.
    stratification_fractions : dict[str, dict]
        The fraction of each category, by stratification id, a
        ``dict[str, float]`` such as ``{"group": {"g1": 0.3, "g2": 0.7}}``:
        the split of the amount of every bin it applies to. A stratification
        may instead split each category of a stratification declared before
        it by its own fractions, a ``dict[str, str | dict[str, float]]`` such
        as ``{"type": {"by": "group", "g1": {"t1": 0.25, "t2": 0.75},
        "g2": {"t1": 0.5, "t2": 0.5}}}``: the compartment ``A_g1_t1`` then
        starts with the amount of ``A`` times 0.3 times 0.25. Every
        compartment of ``type`` needs a category of ``group``. A conditional
        stratification splits only the group its conditions select.
    compartment_amounts : dict[str, float]
        Amounts of single expanded compartments, such as ``{"A_g1": 0.0}``,
        applied after the expansion. The names are checked against the engine
        when the model is built.
    neighbors : Complete | Groups | None
        Who the neighbors of a unit are, for the rate variables
        ``neighbors_<bin>`` and ``neighbors``. Required, with no default, when
        a rate reads them.
    """

    kind: Literal["aggregate"] = "aggregate"
    neighbors: _Neighbors | None = Field(
        default=None,
        description="Who the neighbors of a unit are; required when a rate reads them.",
    )

    # ------------------------------------------------------------------
    # The interface of a model kind
    # ------------------------------------------------------------------

    def check_model(self, model: "Model[Self]") -> None:
        """
        Check the structure against the process and the parameters.

        The process needs a bin and a transition, the amounts must name every
        bin, and the engine must check the fractions against the
        stratifications (every stratification and category named, a split by
        category by a stratification declared before it), plan every
        transition, accept every id (no parameter id is an engine variable, a
        bin, a compartment or a subgroup total), resolve the neighbors, find
        every variable a rate or a formula parameter reads, and build the
        initial state.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Raises
        ------
        ValueError
            If one of these checks fails.
        """
        process = model.process
        if not process.bins:
            raise ValueError("At least one bin must be defined.")
        if not process.transitions:
            raise ValueError("At least one transition must be defined.")
        self._check_bin_amounts(process)
        engine_state.transition_flows(process, self)
        engine_state.check_model(model)
        if self.compartment_amounts:
            engine_state.initial_state(process, self)

    def outputs_by_source(self, model: "Model[Self]") -> dict[str, list[str]]:
        """
        Map each bin and accumulator id to the output names it expands into.

        A bin or accumulator produces one output per combination of the
        stratification categories that apply to it. Without stratifications it
        produces a single output named after the id itself.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Returns
        -------
        dict[str, list[str]]
            Bin and accumulator ids, each mapped to its output names in
            declaration order.
        """
        process = model.process
        outputs: dict[str, list[str]] = {
            source_id: []
            for source_id in (
                *(bin_item.id for bin_item in process.bins),
                *(accumulator.id for accumulator in process.accumulators),
            )
        }
        compartments, accumulators = engine_state.expanded_outputs(process)
        for output in (*compartments, *accumulators):
            outputs[output.source].append(output.name)
        return outputs

    def series_label(self, model: "Model[Self]", series_id: str) -> SeriesLabel:
        """
        The name and unit of a bin, for a plot of its series.

        Parameters
        ----------
        model : Model
            The model that holds this structure.
        series_id : str
            A bin id, or any other series id.

        Returns
        -------
        SeriesLabel
            The bin's name and unit, or the id and no unit when no bin has it.
        """
        bin_item = model.process.bin(series_id)
        if bin_item is None:
            return SeriesLabel(series_id, None)
        return SeriesLabel(bin_item.name or series_id, bin_item.unit)

    def uncalibrated_values(self) -> list[str]:
        """
        Ids of the bins whose initial amount needs calibration.

        Returns
        -------
        list[str]
            Bin ids with a None amount.
        """
        return self.get_uncalibrated_bins()

    def check_calibration_values(
        self,
        model: "Model[Self]",
        engine: "EngineProtocol",
        parameters: Sequence["CalibrationParameter"],
        remainder: str | None,
    ) -> None:
        """
        Check that the remainder has the unit of every amount it gives.

        The remainder takes what the amounts set leave of the sum it keeps, so
        a bin, prefix or compartment value must be in the unit of the
        remainder; a category split has no unit.

        Parameters
        ----------
        model : Model
            The model the values are set on.
        engine : EngineProtocol
            The engine built from it.
        parameters : Sequence[CalibrationParameter]
            The values set, with their types and bounds.
        remainder : str | None
            The bin or compartment that takes what the initial conditions
            leave, if any.

        Raises
        ------
        ValueError
            If an amount set is in another unit than the remainder.
        """
        if remainder is None:
            return
        initial_conditions = [
            param for param in parameters if param.parameter_type == "initial_condition"
        ]
        covered, kept = engine_state.initial_condition_coverage(
            engine, initial_conditions, remainder
        )
        bin_of = {c.name: c.bin for c in self.initial_state(model.process)}

        def unit(compartment: str) -> str | None:
            bin_item = model.process.bin(bin_of[compartment])
            return None if bin_item is None else bin_item.unit

        remainder_unit = unit(kept[0])
        for param, compartments in zip(initial_conditions, covered, strict=True):
            for compartment in compartments:
                if unit(compartment) != remainder_unit:
                    raise ValueError(
                        f"The initial-condition remainder '{remainder}' is in "
                        f"{_unit_name(remainder_unit)}, but the calibrated "
                        f"initial condition '{param.id}' sets '{compartment}', "
                        f"in {_unit_name(unit(compartment))}. The remainder "
                        f"keeps a sum of amounts of one unit: name a remainder "
                        f"of the unit of '{param.id}'."
                    )

    def with_calibrated_values(
        self,
        model: "Model[Self]",
        problem: "CalibrationProblem",
        values: Mapping[str, float],
    ) -> Self:
        """
        The structure with calibrated initial conditions written back.

        The written structure rebuilds the initial state the engine
        scored. Bin values update the amounts they declare, in the unit of
        the bin, and category values the splits. Every compartment those do
        not reproduce, such as an exact compartment value or a value that
        overlaps a per-compartment override, is written as a per-compartment
        override.

        Parameters
        ----------
        model : Model
            The model the problem was built on. It is not changed.
        problem : CalibrationProblem
            The problem, for its parameters and remainder.
        values : Mapping[str, float]
            Calibrated value per parameter id.

        Returns
        -------
        Aggregate
            A copy that holds the calibrated initial conditions.

        Raises
        ------
        ValueError
            If the engine cannot resolve an initial-condition id or the
            remainder, or the values leave the remainder below zero.
        """
        from commol._model.calibration_write_back import calibrated_structure

        return calibrated_structure(model, problem, values)

    def report_sections(
        self,
        model: "Model[Self]",
        calibrated_ids: set[str],
        parameter_statistics: "Mapping[str, EntrySummary] | None",
    ) -> ReportSections:
        """
        The ``initial_amounts`` and ``category_splits`` sections of a report.

        See `Model.applied_parameters`.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, EntrySummary] | None
            Statistics of each id from a sample, or None without one.

        Returns
        -------
        ReportSections
            ``{"initial_amounts": ..., "category_splits": ...}``.
        """
        from commol._model.report import category_splits_report, initial_amounts_report

        return {
            "initial_amounts": initial_amounts_report(
                model.process, self, calibrated_ids, parameter_statistics
            ),
            "category_splits": category_splits_report(
                model.process, self, calibrated_ids, parameter_statistics
            ),
        }

    def describe(self, model: "Model[Self]") -> "list[Section]":
        """
        The entities, dynamics, initialization and design concepts.

        ``entities``: the bins, accumulators, stratifications and expanded
        compartments. ``dynamics``: the state graphs, the compact and expanded
        forms of the equations the engine simulates, the impulses and the
        neighbors. ``initialization``: the bin amounts, the category splits,
        the overrides and the initial state the engine builds. ``design_concepts``: the
        interaction of the units, when the structure says who their neighbors
        are.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Returns
        -------
        list[Section]
            The sections of an aggregate model.
        """
        from commol._model.aggregate_description import describe_aggregate

        return describe_aggregate(model)

    def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
        """Check the units of the model; see `Model.check_unit_consistency`."""
        from commol._model.units import UnitChecker

        UnitChecker(model).check_unit_consistency(verbose)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def update_compartment_amounts(
        self, process: Process, amounts: Mapping[str, float]
    ) -> None:
        """
        Upsert per-compartment initial condition overrides.

        A compartment already declared has its amount replaced; a new one is
        appended. The engine validates the result against the expansion of
        the process before it is stored, so a rejected call leaves the
        structure unchanged.

        Parameters
        ----------
        process : Process
            The process of the model, such as ``model.process``.
        amounts : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new
            amount, in the unit of their bin.

        Raises
        ------
        ValueError
            If a name is not an expanded compartment of the model, or an
            amount is not finite or is negative.
        """
        candidate = self.model_copy(deep=True)
        candidate._upsert_compartment_amounts(amounts)
        engine_state.initial_state(process, candidate)
        self.compartment_amounts = candidate.compartment_amounts

    def initial_state(self, process: Process) -> list[InitialCompartment]:
        """
        Expanded compartments and the amount each one starts with.

        The engine builds this state from the bin amounts, the stratification
        fractions and the per-compartment overrides; it is the state a
        simulation starts from.

        Parameters
        ----------
        process : Process
            The process of the model, such as ``model.process``.

        Returns
        -------
        list[InitialCompartment]
            One entry per expanded compartment, in engine order.

        Raises
        ------
        ValueError
            If an amount is not valid or an override names no compartment of
            the structure.
        """
        return engine_state.initial_state(process, self)

    def get_conditioning_categories(
        self, process: Process, stratification_id: str
    ) -> tuple[str, ...]:
        """
        Get the categories a stratification is conditioned on.

        An unconditional stratification subdivides the whole model and has
        no conditioning categories. A conditional one subdivides only the
        subgroup its conditions select, and its fractions are relative to that
        subgroup rather than to the whole model.

        Parameters
        ----------
        process : Process
            The process of the model, such as ``model.process``.
        stratification_id : str
            Id of a stratification of the process.

        Returns
        -------
        tuple[str, ...]
            Ids of the categories selecting the subgroup this stratification
            subdivides, empty for an unconditional stratification. The
            categories of one stratification are a set: the subgroup has one
            of them (`subgroup_total`). A category name another
            stratification also declares is qualified, as in ``group:g1``.

        Raises
        ------
        ValueError
            If the stratification is not found.
        """
        for stratification in process.stratifications:
            if stratification.id == stratification_id:
                return tuple(
                    self.category_id(condition.stratification, category)
                    for condition in (stratification.conditions or [])
                    for category in condition.names()
                )
        raise ValueError(
            f"Stratification '{stratification_id}' not found. Available: "
            f"{sorted(s.id for s in process.stratifications)}"
        )

    def subgroup_total(self, process: Process, categories: Iterable[str] = ()) -> float:
        """
        Get the initial sum of the compartments of a subgroup.

        The sum is taken of the state the engine builds, overrides included,
        over every compartment that has, in each stratification the categories
        name, one of the categories of that stratification: ``["g2", "g3",
        "s1"]`` sums the compartments of ``g2`` or ``g3`` that have ``s1``.
        Passing no categories gives the sum of every compartment. It adds every bin it
        covers, so over bins of different units it has no unit.

        Parameters
        ----------
        process : Process
            The process of the model, such as ``model.process``.
        categories : Iterable[str], optional
            Category ids, bare or qualified (``group:g1``).

        Returns
        -------
        float
            The sum of the compartments of the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found or a bare name is ambiguous.
        """
        sets: dict[str, set[str]] = {}
        for category in categories:
            stratification, name = self.resolve_category(category)
            sets.setdefault(stratification, set()).add(name)
        return sum(
            compartment.amount
            for compartment in self.initial_state(process)
            if all(
                compartment.categories.get(stratification) in names
                for stratification, names in sets.items()
            )
        )

    # ------------------------------------------------------------------
    # Checks against the model
    # ------------------------------------------------------------------

    def _check_bin_amounts(self, process: Process) -> None:
        """Check that the bin amounts name every bin of the process."""
        actual_bins = set(self.bin_amounts)
        expected_bins = {bin_item.id for bin_item in process.bins}
        if actual_bins != expected_bins:
            raise ValueError(
                f"Initial bin amounts keys must exactly match bin ids. "
                f"Missing ids: {sorted(expected_bins - actual_bins)}, "
                f"Extra ids: {sorted(actual_bins - expected_bins)}."
            )
```

### Methods:

#### check_model

```python
check_model(model: Model[Self]) -> None
```

Check the structure against the process and the parameters.

The process needs a bin and a transition, the amounts must name every bin, and the engine must check the fractions against the stratifications (every stratification and category named, a split by category by a stratification declared before it), plan every transition, accept every id (no parameter id is an engine variable, a bin, a compartment or a subgroup total), resolve the neighbors, find every variable a rate or a formula parameter reads, and build the initial state.

Parameters:

| Name    | Type    | Description                          | Default    |
| ------- | ------- | ------------------------------------ | ---------- |
| `model` | `Model` | The model that holds this structure. | *required* |

Raises:

| Type         | Description                   |
| ------------ | ----------------------------- |
| `ValueError` | If one of these checks fails. |

Source code in `commol/_model/aggregate.py`

```python
def check_model(self, model: "Model[Self]") -> None:
    """
    Check the structure against the process and the parameters.

    The process needs a bin and a transition, the amounts must name every
    bin, and the engine must check the fractions against the
    stratifications (every stratification and category named, a split by
    category by a stratification declared before it), plan every
    transition, accept every id (no parameter id is an engine variable, a
    bin, a compartment or a subgroup total), resolve the neighbors, find
    every variable a rate or a formula parameter reads, and build the
    initial state.

    Parameters
    ----------
    model : Model
        The model that holds this structure.

    Raises
    ------
    ValueError
        If one of these checks fails.
    """
    process = model.process
    if not process.bins:
        raise ValueError("At least one bin must be defined.")
    if not process.transitions:
        raise ValueError("At least one transition must be defined.")
    self._check_bin_amounts(process)
    engine_state.transition_flows(process, self)
    engine_state.check_model(model)
    if self.compartment_amounts:
        engine_state.initial_state(process, self)
```

#### outputs_by_source

```python
outputs_by_source(model: Model[Self]) -> dict[str, list[str]]
```

Map each bin and accumulator id to the output names it expands into.

A bin or accumulator produces one output per combination of the stratification categories that apply to it. Without stratifications it produces a single output named after the id itself.

Parameters:

| Name    | Type    | Description                          | Default    |
| ------- | ------- | ------------------------------------ | ---------- |
| `model` | `Model` | The model that holds this structure. | *required* |

Returns:

| Type                   | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| `dict[str, list[str]]` | Bin and accumulator ids, each mapped to its output names in declaration order. |

Source code in `commol/_model/aggregate.py`

```python
def outputs_by_source(self, model: "Model[Self]") -> dict[str, list[str]]:
    """
    Map each bin and accumulator id to the output names it expands into.

    A bin or accumulator produces one output per combination of the
    stratification categories that apply to it. Without stratifications it
    produces a single output named after the id itself.

    Parameters
    ----------
    model : Model
        The model that holds this structure.

    Returns
    -------
    dict[str, list[str]]
        Bin and accumulator ids, each mapped to its output names in
        declaration order.
    """
    process = model.process
    outputs: dict[str, list[str]] = {
        source_id: []
        for source_id in (
            *(bin_item.id for bin_item in process.bins),
            *(accumulator.id for accumulator in process.accumulators),
        )
    }
    compartments, accumulators = engine_state.expanded_outputs(process)
    for output in (*compartments, *accumulators):
        outputs[output.source].append(output.name)
    return outputs
```

#### series_label

```python
series_label(model: Model[Self], series_id: str) -> SeriesLabel
```

The name and unit of a bin, for a plot of its series.

Parameters:

| Name        | Type    | Description                          | Default    |
| ----------- | ------- | ------------------------------------ | ---------- |
| `model`     | `Model` | The model that holds this structure. | *required* |
| `series_id` | `str`   | A bin id, or any other series id.    | *required* |

Returns:

| Type          | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `SeriesLabel` | The bin's name and unit, or the id and no unit when no bin has it. |

Source code in `commol/_model/aggregate.py`

```python
def series_label(self, model: "Model[Self]", series_id: str) -> SeriesLabel:
    """
    The name and unit of a bin, for a plot of its series.

    Parameters
    ----------
    model : Model
        The model that holds this structure.
    series_id : str
        A bin id, or any other series id.

    Returns
    -------
    SeriesLabel
        The bin's name and unit, or the id and no unit when no bin has it.
    """
    bin_item = model.process.bin(series_id)
    if bin_item is None:
        return SeriesLabel(series_id, None)
    return SeriesLabel(bin_item.name or series_id, bin_item.unit)
```

#### uncalibrated_values

```python
uncalibrated_values() -> list[str]
```

Ids of the bins whose initial amount needs calibration.

Returns:

| Type        | Description                 |
| ----------- | --------------------------- |
| `list[str]` | Bin ids with a None amount. |

Source code in `commol/_model/aggregate.py`

```python
def uncalibrated_values(self) -> list[str]:
    """
    Ids of the bins whose initial amount needs calibration.

    Returns
    -------
    list[str]
        Bin ids with a None amount.
    """
    return self.get_uncalibrated_bins()
```

#### check_calibration_values

```python
check_calibration_values(model: Model[Self], engine: EngineProtocol, parameters: Sequence[CalibrationParameter], remainder: str | None) -> None
```

Check that the remainder has the unit of every amount it gives.

The remainder takes what the amounts set leave of the sum it keeps, so a bin, prefix or compartment value must be in the unit of the remainder; a category split has no unit.

Parameters:

| Name         | Type                             | Description                                  | Default                                                                      |
| ------------ | -------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| `model`      | `Model`                          | The model the values are set on.             | *required*                                                                   |
| `engine`     | `EngineProtocol`                 | The engine built from it.                    | *required*                                                                   |
| `parameters` | `Sequence[CalibrationParameter]` | The values set, with their types and bounds. | *required*                                                                   |
| `remainder`  | \`str                            | None\`                                       | The bin or compartment that takes what the initial conditions leave, if any. |

Raises:

| Type         | Description                                             |
| ------------ | ------------------------------------------------------- |
| `ValueError` | If an amount set is in another unit than the remainder. |

Source code in `commol/_model/aggregate.py`

```python
def check_calibration_values(
    self,
    model: "Model[Self]",
    engine: "EngineProtocol",
    parameters: Sequence["CalibrationParameter"],
    remainder: str | None,
) -> None:
    """
    Check that the remainder has the unit of every amount it gives.

    The remainder takes what the amounts set leave of the sum it keeps, so
    a bin, prefix or compartment value must be in the unit of the
    remainder; a category split has no unit.

    Parameters
    ----------
    model : Model
        The model the values are set on.
    engine : EngineProtocol
        The engine built from it.
    parameters : Sequence[CalibrationParameter]
        The values set, with their types and bounds.
    remainder : str | None
        The bin or compartment that takes what the initial conditions
        leave, if any.

    Raises
    ------
    ValueError
        If an amount set is in another unit than the remainder.
    """
    if remainder is None:
        return
    initial_conditions = [
        param for param in parameters if param.parameter_type == "initial_condition"
    ]
    covered, kept = engine_state.initial_condition_coverage(
        engine, initial_conditions, remainder
    )
    bin_of = {c.name: c.bin for c in self.initial_state(model.process)}

    def unit(compartment: str) -> str | None:
        bin_item = model.process.bin(bin_of[compartment])
        return None if bin_item is None else bin_item.unit

    remainder_unit = unit(kept[0])
    for param, compartments in zip(initial_conditions, covered, strict=True):
        for compartment in compartments:
            if unit(compartment) != remainder_unit:
                raise ValueError(
                    f"The initial-condition remainder '{remainder}' is in "
                    f"{_unit_name(remainder_unit)}, but the calibrated "
                    f"initial condition '{param.id}' sets '{compartment}', "
                    f"in {_unit_name(unit(compartment))}. The remainder "
                    f"keeps a sum of amounts of one unit: name a remainder "
                    f"of the unit of '{param.id}'."
                )
```

#### with_calibrated_values

```python
with_calibrated_values(model: Model[Self], problem: CalibrationProblem, values: Mapping[str, float]) -> Self
```

The structure with calibrated initial conditions written back.

The written structure rebuilds the initial state the engine scored. Bin values update the amounts they declare, in the unit of the bin, and category values the splits. Every compartment those do not reproduce, such as an exact compartment value or a value that overlaps a per-compartment override, is written as a per-compartment override.

Parameters:

| Name      | Type                  | Description                                            | Default    |
| --------- | --------------------- | ------------------------------------------------------ | ---------- |
| `model`   | `Model`               | The model the problem was built on. It is not changed. | *required* |
| `problem` | `CalibrationProblem`  | The problem, for its parameters and remainder.         | *required* |
| `values`  | `Mapping[str, float]` | Calibrated value per parameter id.                     | *required* |

Returns:

| Type        | Description                                          |
| ----------- | ---------------------------------------------------- |
| `Aggregate` | A copy that holds the calibrated initial conditions. |

Raises:

| Type         | Description                                                                                                          |
| ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the engine cannot resolve an initial-condition id or the remainder, or the values leave the remainder below zero. |

Source code in `commol/_model/aggregate.py`

```python
def with_calibrated_values(
    self,
    model: "Model[Self]",
    problem: "CalibrationProblem",
    values: Mapping[str, float],
) -> Self:
    """
    The structure with calibrated initial conditions written back.

    The written structure rebuilds the initial state the engine
    scored. Bin values update the amounts they declare, in the unit of
    the bin, and category values the splits. Every compartment those do
    not reproduce, such as an exact compartment value or a value that
    overlaps a per-compartment override, is written as a per-compartment
    override.

    Parameters
    ----------
    model : Model
        The model the problem was built on. It is not changed.
    problem : CalibrationProblem
        The problem, for its parameters and remainder.
    values : Mapping[str, float]
        Calibrated value per parameter id.

    Returns
    -------
    Aggregate
        A copy that holds the calibrated initial conditions.

    Raises
    ------
    ValueError
        If the engine cannot resolve an initial-condition id or the
        remainder, or the values leave the remainder below zero.
    """
    from commol._model.calibration_write_back import calibrated_structure

    return calibrated_structure(model, problem, values)
```

#### report_sections

```python
report_sections(model: Model[Self], calibrated_ids: set[str], parameter_statistics: Mapping[str, EntrySummary] | None) -> ReportSections
```

The `initial_amounts` and `category_splits` sections of a report.

See `Model.applied_parameters`.

Parameters:

| Name                   | Type                         | Description                              | Default                                                   |
| ---------------------- | ---------------------------- | ---------------------------------------- | --------------------------------------------------------- |
| `model`                | `Model`                      | The model that holds the applied values. | *required*                                                |
| `calibrated_ids`       | `set[str]`                   | Ids that calibration set.                | *required*                                                |
| `parameter_statistics` | \`Mapping[str, EntrySummary] | None\`                                   | Statistics of each id from a sample, or None without one. |

Returns:

| Type             | Description                                       |
| ---------------- | ------------------------------------------------- |
| `ReportSections` | {"initial_amounts": ..., "category_splits": ...}. |

Source code in `commol/_model/aggregate.py`

```python
def report_sections(
    self,
    model: "Model[Self]",
    calibrated_ids: set[str],
    parameter_statistics: "Mapping[str, EntrySummary] | None",
) -> ReportSections:
    """
    The ``initial_amounts`` and ``category_splits`` sections of a report.

    See `Model.applied_parameters`.

    Parameters
    ----------
    model : Model
        The model that holds the applied values.
    calibrated_ids : set[str]
        Ids that calibration set.
    parameter_statistics : Mapping[str, EntrySummary] | None
        Statistics of each id from a sample, or None without one.

    Returns
    -------
    ReportSections
        ``{"initial_amounts": ..., "category_splits": ...}``.
    """
    from commol._model.report import category_splits_report, initial_amounts_report

    return {
        "initial_amounts": initial_amounts_report(
            model.process, self, calibrated_ids, parameter_statistics
        ),
        "category_splits": category_splits_report(
            model.process, self, calibrated_ids, parameter_statistics
        ),
    }
```

#### describe

```python
describe(model: Model[Self]) -> list[Section]
```

The entities, dynamics, initialization and design concepts.

`entities`: the bins, accumulators, stratifications and expanded compartments. `dynamics`: the state graphs, the compact and expanded forms of the equations the engine simulates, the impulses and the neighbors. `initialization`: the bin amounts, the category splits, the overrides and the initial state the engine builds. `design_concepts`: the interaction of the units, when the structure says who their neighbors are.

Parameters:

| Name    | Type    | Description                          | Default    |
| ------- | ------- | ------------------------------------ | ---------- |
| `model` | `Model` | The model that holds this structure. | *required* |

Returns:

| Type            | Description                         |
| --------------- | ----------------------------------- |
| `list[Section]` | The sections of an aggregate model. |

Source code in `commol/_model/aggregate.py`

```python
def describe(self, model: "Model[Self]") -> "list[Section]":
    """
    The entities, dynamics, initialization and design concepts.

    ``entities``: the bins, accumulators, stratifications and expanded
    compartments. ``dynamics``: the state graphs, the compact and expanded
    forms of the equations the engine simulates, the impulses and the
    neighbors. ``initialization``: the bin amounts, the category splits,
    the overrides and the initial state the engine builds. ``design_concepts``: the
    interaction of the units, when the structure says who their neighbors
    are.

    Parameters
    ----------
    model : Model
        The model that holds this structure.

    Returns
    -------
    list[Section]
        The sections of an aggregate model.
    """
    from commol._model.aggregate_description import describe_aggregate

    return describe_aggregate(model)
```

#### check_unit_consistency

```python
check_unit_consistency(model: Model[Self], verbose: bool) -> None
```

Check the units of the model; see `Model.check_unit_consistency`.

Source code in `commol/_model/aggregate.py`

```python
def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
    """Check the units of the model; see `Model.check_unit_consistency`."""
    from commol._model.units import UnitChecker

    UnitChecker(model).check_unit_consistency(verbose)
```

#### update_compartment_amounts

```python
update_compartment_amounts(process: Process, amounts: Mapping[str, float]) -> None
```

Upsert per-compartment initial condition overrides.

A compartment already declared has its amount replaced; a new one is appended. The engine validates the result against the expansion of the process before it is stored, so a rejected call leaves the structure unchanged.

Parameters:

| Name      | Type                  | Description                                                                                  | Default    |
| --------- | --------------------- | -------------------------------------------------------------------------------------------- | ---------- |
| `process` | `Process`             | The process of the model, such as model.process.                                             | *required* |
| `amounts` | `Mapping[str, float]` | Dictionary mapping expanded compartment names to their new amount, in the unit of their bin. | *required* |

Raises:

| Type         | Description                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------- |
| `ValueError` | If a name is not an expanded compartment of the model, or an amount is not finite or is negative. |

Source code in `commol/_model/aggregate.py`

```python
def update_compartment_amounts(
    self, process: Process, amounts: Mapping[str, float]
) -> None:
    """
    Upsert per-compartment initial condition overrides.

    A compartment already declared has its amount replaced; a new one is
    appended. The engine validates the result against the expansion of
    the process before it is stored, so a rejected call leaves the
    structure unchanged.

    Parameters
    ----------
    process : Process
        The process of the model, such as ``model.process``.
    amounts : Mapping[str, float]
        Dictionary mapping expanded compartment names to their new
        amount, in the unit of their bin.

    Raises
    ------
    ValueError
        If a name is not an expanded compartment of the model, or an
        amount is not finite or is negative.
    """
    candidate = self.model_copy(deep=True)
    candidate._upsert_compartment_amounts(amounts)
    engine_state.initial_state(process, candidate)
    self.compartment_amounts = candidate.compartment_amounts
```

#### initial_state

```python
initial_state(process: Process) -> list[InitialCompartment]
```

Expanded compartments and the amount each one starts with.

The engine builds this state from the bin amounts, the stratification fractions and the per-compartment overrides; it is the state a simulation starts from.

Parameters:

| Name      | Type      | Description                                      | Default    |
| --------- | --------- | ------------------------------------------------ | ---------- |
| `process` | `Process` | The process of the model, such as model.process. | *required* |

Returns:

| Type                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `list[InitialCompartment]` | One entry per expanded compartment, in engine order. |

Raises:

| Type         | Description                                                                     |
| ------------ | ------------------------------------------------------------------------------- |
| `ValueError` | If an amount is not valid or an override names no compartment of the structure. |

Source code in `commol/_model/aggregate.py`

```python
def initial_state(self, process: Process) -> list[InitialCompartment]:
    """
    Expanded compartments and the amount each one starts with.

    The engine builds this state from the bin amounts, the stratification
    fractions and the per-compartment overrides; it is the state a
    simulation starts from.

    Parameters
    ----------
    process : Process
        The process of the model, such as ``model.process``.

    Returns
    -------
    list[InitialCompartment]
        One entry per expanded compartment, in engine order.

    Raises
    ------
    ValueError
        If an amount is not valid or an override names no compartment of
        the structure.
    """
    return engine_state.initial_state(process, self)
```

#### get_conditioning_categories

```python
get_conditioning_categories(process: Process, stratification_id: str) -> tuple[str, ...]
```

Get the categories a stratification is conditioned on.

An unconditional stratification subdivides the whole model and has no conditioning categories. A conditional one subdivides only the subgroup its conditions select, and its fractions are relative to that subgroup rather than to the whole model.

Parameters:

| Name                | Type      | Description                                      | Default    |
| ------------------- | --------- | ------------------------------------------------ | ---------- |
| `process`           | `Process` | The process of the model, such as model.process. | *required* |
| `stratification_id` | `str`     | Id of a stratification of the process.           | *required* |

Returns:

| Type              | Description                                                                                                                                                                                                                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tuple[str, ...]` | Ids of the categories selecting the subgroup this stratification subdivides, empty for an unconditional stratification. The categories of one stratification are a set: the subgroup has one of them (subgroup_total). A category name another stratification also declares is qualified, as in group:g1. |

Raises:

| Type         | Description                         |
| ------------ | ----------------------------------- |
| `ValueError` | If the stratification is not found. |

Source code in `commol/_model/aggregate.py`

```python
def get_conditioning_categories(
    self, process: Process, stratification_id: str
) -> tuple[str, ...]:
    """
    Get the categories a stratification is conditioned on.

    An unconditional stratification subdivides the whole model and has
    no conditioning categories. A conditional one subdivides only the
    subgroup its conditions select, and its fractions are relative to that
    subgroup rather than to the whole model.

    Parameters
    ----------
    process : Process
        The process of the model, such as ``model.process``.
    stratification_id : str
        Id of a stratification of the process.

    Returns
    -------
    tuple[str, ...]
        Ids of the categories selecting the subgroup this stratification
        subdivides, empty for an unconditional stratification. The
        categories of one stratification are a set: the subgroup has one
        of them (`subgroup_total`). A category name another
        stratification also declares is qualified, as in ``group:g1``.

    Raises
    ------
    ValueError
        If the stratification is not found.
    """
    for stratification in process.stratifications:
        if stratification.id == stratification_id:
            return tuple(
                self.category_id(condition.stratification, category)
                for condition in (stratification.conditions or [])
                for category in condition.names()
            )
    raise ValueError(
        f"Stratification '{stratification_id}' not found. Available: "
        f"{sorted(s.id for s in process.stratifications)}"
    )
```

#### subgroup_total

```python
subgroup_total(process: Process, categories: Iterable[str] = ()) -> float
```

Get the initial sum of the compartments of a subgroup.

The sum is taken of the state the engine builds, overrides included, over every compartment that has, in each stratification the categories name, one of the categories of that stratification: `["g2", "g3", "s1"]` sums the compartments of `g2` or `g3` that have `s1`. Passing no categories gives the sum of every compartment. It adds every bin it covers, so over bins of different units it has no unit.

Parameters:

| Name         | Type            | Description                                      | Default    |
| ------------ | --------------- | ------------------------------------------------ | ---------- |
| `process`    | `Process`       | The process of the model, such as model.process. | *required* |
| `categories` | `Iterable[str]` | Category ids, bare or qualified (group:g1).      | `()`       |

Returns:

| Type    | Description                                            |
| ------- | ------------------------------------------------------ |
| `float` | The sum of the compartments of the subgroup at step 0. |

Raises:

| Type         | Description                                             |
| ------------ | ------------------------------------------------------- |
| `ValueError` | If a category is not found or a bare name is ambiguous. |

Source code in `commol/_model/aggregate.py`

```python
def subgroup_total(self, process: Process, categories: Iterable[str] = ()) -> float:
    """
    Get the initial sum of the compartments of a subgroup.

    The sum is taken of the state the engine builds, overrides included,
    over every compartment that has, in each stratification the categories
    name, one of the categories of that stratification: ``["g2", "g3",
    "s1"]`` sums the compartments of ``g2`` or ``g3`` that have ``s1``.
    Passing no categories gives the sum of every compartment. It adds every bin it
    covers, so over bins of different units it has no unit.

    Parameters
    ----------
    process : Process
        The process of the model, such as ``model.process``.
    categories : Iterable[str], optional
        Category ids, bare or qualified (``group:g1``).

    Returns
    -------
    float
        The sum of the compartments of the subgroup at step 0.

    Raises
    ------
    ValueError
        If a category is not found or a bare name is ambiguous.
    """
    sets: dict[str, set[str]] = {}
    for category in categories:
        stratification, name = self.resolve_category(category)
        sets.setdefault(stratification, set()).add(name)
    return sum(
        compartment.amount
        for compartment in self.initial_state(process)
        if all(
            compartment.categories.get(stratification) in names
            for stratification, names in sets.items()
        )
    )
```

## Complete

Bases: `BaseModel`

Every unit is a neighbor of every other.

`neighbors_<bin>` is the bin total and `neighbors` is `total`, so `k1 * neighbors_B / neighbors` from `A` moves `k1 * A * B / total`.

Attributes:

| Name   | Type                  | Description                       |
| ------ | --------------------- | --------------------------------- |
| `kind` | `Literal['complete']` | The tag of the neighbors in JSON. |

Source code in `commol/_model/neighbors.py`

```python
class Complete(BaseModel):
    """
    Every unit is a neighbor of every other.

    ``neighbors_<bin>`` is the bin total and ``neighbors`` is ``total``, so
    ``k1 * neighbors_B / neighbors`` from ``A`` moves ``k1 * A * B / total``.

    Attributes
    ----------
    kind : Literal["complete"]
        The tag of the neighbors in JSON.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["complete"] = "complete"
```

## Groups

Bases: `BaseModel`

Units are neighbors by the categories of one stratification.

In a compartment of category `g`, `neighbors_<bin>` is `Σ_g' w[g][g'] × (amount of the bin in g')` and `neighbors` is `Σ_g' w[g][g'] × (amount of every bin in g')`.

Attributes:

| Name             | Type                            | Description                                                |
| ---------------- | ------------------------------- | ---------------------------------------------------------- |
| `kind`           | `Literal['groups']`             | The tag of the neighbors in JSON.                          |
| `stratification` | `str`                           | Id of the stratification whose categories group the units. |
| `weights`        | \`dict\[str, dict[str, float]\] | None\`                                                     |

Source code in `commol/_model/neighbors.py`

```python
class Groups(BaseModel):
    """
    Units are neighbors by the categories of one stratification.

    In a compartment of category ``g``, ``neighbors_<bin>`` is
    ``Σ_g' w[g][g'] × (amount of the bin in g')`` and ``neighbors`` is
    ``Σ_g' w[g][g'] × (amount of every bin in g')``.

    Attributes
    ----------
    kind : Literal["groups"]
        The tag of the neighbors in JSON.
    stratification : str
        Id of the stratification whose categories group the units.
    weights : dict[str, dict[str, float]] | None
        ``weights[g][g']``, the weight of category ``g'`` for a unit in ``g``:
        one finite number of at least 0 for every pair of categories. None
        is the identity: a unit sees only its own category.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["groups"] = "groups"
    stratification: str
    weights: dict[str, dict[str, float]] | None = None

    def __init__(
        self,
        stratification: str,
        weights: Mapping[str, Mapping[str, float]] | None = None,
        *,
        kind: Literal["groups"] = "groups",
    ) -> None:
        """
        Group the neighbors of a unit by the categories of a stratification.

        Parameters
        ----------
        stratification : str
            Id of the stratification whose categories group the units.
        weights : Mapping[str, Mapping[str, float]] | None, default=None
            ``weights[g][g']``, the weight of category ``g'`` for a unit in
            ``g``. None is the identity.
        kind : {"groups"}, default="groups"
            The kind of neighbors, as JSON holds it.
        """
        super().__init__(
            stratification=stratification,
            weights=None
            if weights is None
            else {own: dict(row) for own, row in weights.items()},
            kind=kind,
        )
```

### Methods:

#### __init__

```python
__init__(stratification: str, weights: Mapping[str, Mapping[str, float]] | None = None, *, kind: Literal['groups'] = 'groups') -> None
```

Group the neighbors of a unit by the categories of a stratification.

Parameters:

| Name             | Type                                  | Description                                                | Default                                                                          |
| ---------------- | ------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `stratification` | `str`                                 | Id of the stratification whose categories group the units. | *required*                                                                       |
| `weights`        | \`Mapping\[str, Mapping[str, float]\] | None\`                                                     | weights[g][g'], the weight of category g' for a unit in g. None is the identity. |
| `kind`           | `groups`                              | The kind of neighbors, as JSON holds it.                   | `"groups"`                                                                       |

Source code in `commol/_model/neighbors.py`

```python
def __init__(
    self,
    stratification: str,
    weights: Mapping[str, Mapping[str, float]] | None = None,
    *,
    kind: Literal["groups"] = "groups",
) -> None:
    """
    Group the neighbors of a unit by the categories of a stratification.

    Parameters
    ----------
    stratification : str
        Id of the stratification whose categories group the units.
    weights : Mapping[str, Mapping[str, float]] | None, default=None
        ``weights[g][g']``, the weight of category ``g'`` for a unit in
        ``g``. None is the identity.
    kind : {"groups"}, default="groups"
        The kind of neighbors, as JSON holds it.
    """
    super().__init__(
        stratification=stratification,
        weights=None
        if weights is None
        else {own: dict(row) for own, row in weights.items()},
        kind=kind,
    )
```

## InitialCompartment

Bases: `NamedTuple`

One expanded compartment and the amount it starts with.

Attributes:

| Name         | Type             | Description                                                                                                                                             |
| ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`       | `str`            | Expanded compartment name, for example A_g1.                                                                                                            |
| `bin`        | `str`            | Bin the compartment expands from.                                                                                                                       |
| `categories` | `dict[str, str]` | Category applied to the compartment, keyed by stratification id. A conditional stratification whose conditions the compartment does not meet is absent. |
| `amount`     | `float`          | Initial amount, including per-compartment overrides.                                                                                                    |

Source code in `commol/_model/engine_state.py`

```python
class InitialCompartment(NamedTuple):
    """
    One expanded compartment and the amount it starts with.

    Attributes
    ----------
    name : str
        Expanded compartment name, for example ``A_g1``.
    bin : str
        Bin the compartment expands from.
    categories : dict[str, str]
        Category applied to the compartment, keyed by stratification id. A
        conditional stratification whose conditions the compartment does not
        meet is absent.
    amount : float
        Initial amount, including per-compartment overrides.
    """

    name: str
    bin: str
    categories: dict[str, str]
    amount: float
```

## SeriesLabel

Bases: `NamedTuple`

How a plot names a series and its unit.

Attributes:

| Name   | Type  | Description                 |
| ------ | ----- | --------------------------- |
| `name` | `str` | Display name of the series. |
| `unit` | \`str | None\`                      |

Source code in `commol/_model/structure.py`

```python
class SeriesLabel(NamedTuple):
    """
    How a plot names a series and its unit.

    Attributes
    ----------
    name : str
        Display name of the series.
    unit : str | None
        Unit of its values, or None when it has none.
    """

    name: str
    unit: str | None
```

## ReportSections

Bases: `TypedDict`

The sections a kind adds to a report of applied values, by key: an aggregate model adds `initial_amounts` and `category_splits`, and a new kind adds the keys of its sections here.

Source code in `commol/_model/structure.py`

```python
class ReportSections(TypedDict, total=False):
    """
    The sections a kind adds to a report of applied values, by key: an
    aggregate model adds ``initial_amounts`` and ``category_splits``, and a
    new kind adds the keys of its sections here.
    """

    initial_amounts: "dict[str, InitialAmountEntry]"
    category_splits: "dict[str, CategorySplitEntry]"
```

## SeriesValue

Bases: `BaseModel`

An empirical series for a parameter value.

Attributes:

| Name   | Type                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data` | `list[tuple[float, float]]` | (position, value) points on the axis. Need not be pre-sorted; the engine sorts them.                                                                                                                                                                                                                                                                                                                                                                                     |
| `mode` | `str`                       | How the series is read between its points: "pulse" (the value on the step of a point, 0 on every other step; integer positions only), "step_function" (the value of the last point at or before t), or "linear" (linear interpolation in t). The engine decides which positions its model type accepts: a DifferenceEquations or StochasticSteps model only integer steps. The model rejects a position its engine cannot run when it is built or its parameters change. |

Source code in `commol/_model/parameter.py`

```python
class SeriesValue(BaseModel):
    """
    An empirical series for a parameter value.

    Attributes
    ----------
    data : list[tuple[float, float]]
        (position, value) points on the axis. Need not be pre-sorted; the
        engine sorts them.
    mode : str
        How the series is read between its points: ``"pulse"`` (the value on
        the step of a point, 0 on every other step; integer positions only),
        ``"step_function"`` (the value of the last point at or before ``t``),
        or ``"linear"`` (linear interpolation in ``t``). The engine decides
        which positions its model type accepts: a DifferenceEquations or
        StochasticSteps model only integer steps. The model rejects a
        position its engine cannot run when it is built or its parameters
        change.
    """

    model_config = ConfigDict(extra="forbid")

    data: list[tuple[float, float]]
    mode: str = Field(default="pulse", pattern="^(pulse|step_function|linear)$")

    @field_validator("data")
    @classmethod
    def _validate_data(
        cls, value: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Check the points with the rule of the engine (`check_series` of
        `crates/commol-model/src/values.rs`): at least one point, each
        position finite, not negative and once, each value finite.
        """
        commol_rs.model.check_series(value)
        return value
```

## UnitConsistencyError

Bases: `Exception`

Raised when unit consistency check fails.

Source code in `commol/_model/unit_checks.py`

```python
class UnitConsistencyError(Exception):
    """Raised when unit consistency check fails."""

    pass
```

## The parts of a model

The parts that the builder makes, which `Model.process` and `Model.parameters` hold. [Model Structure](https://munqu.github.io/commol/latest/advanced/model-structure/#the-parts-of-a-model) names the method that adds each one.

## Bin

Bases: `BaseModel`

Defines a single bin (base category) of the process of a model.

A bin represents a fundamental category before stratification. The combination of a bin with all stratifications produces the actual compartments.

Attributes:

| Name   | Type  | Description                                     |
| ------ | ----- | ----------------------------------------------- |
| `id`   | `str` | Identifier of the bin.                          |
| `name` | `str` | A descriptive, human-readable name for the bin. |
| `unit` | \`str | None\`                                          |

Source code in `commol/_model/bin.py`

```python
class Bin(BaseModel):
    """
    Defines a single bin (base category) of the process of a model.

    A bin represents a fundamental category before stratification. The combination
    of a bin with all stratifications produces the actual compartments.

    Attributes
    ----------
    id : str
        Identifier of the bin.
    name : str
        A descriptive, human-readable name for the bin.
    unit : str | None
        The unit of measurement for this bin.
        If None, the bin will use the model-level bin_unit when specified.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Identifier of the bin.")
    name: str = Field(..., description="Descriptive, human-readable name for the bin.")
    unit: str | None = Field(
        None,
        description="Unit of measurement for this bin.",
    )

    @override
    def __hash__(self) -> int:
        return hash(self.id)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Bin) and self.id == other.id
```

## Accumulator

Bases: `BaseModel`

A cumulative event counter tracked as output but not as part of the state.

Attributes:

| Name   | Type  | Description                                                                                     |
| ------ | ----- | ----------------------------------------------------------------------------------------------- |
| `id`   | `str` | Identifier of the accumulator, and the name of its output before the stratifications expand it. |
| `name` | `str` | A descriptive, human-readable name for the accumulator.                                         |

Source code in `commol/_model/bin.py`

```python
class Accumulator(BaseModel):
    """
    A cumulative event counter tracked as output but not as part of the state.

    Attributes
    ----------
    id : str
        Identifier of the accumulator, and the name of its output before the
        stratifications expand it.
    name : str
        A descriptive, human-readable name for the accumulator.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Identifier of the accumulator.")
    name: str = Field(..., description="Descriptive, human-readable name.")
```

## Stratification

Bases: `BaseModel`

Defines a categorical subdivision of the compartments.

Attributes:

| Name          | Type                            | Description                                              |
| ------------- | ------------------------------- | -------------------------------------------------------- |
| `id`          | `str`                           | Identifier of the stratification.                        |
| `categories`  | `list[str]`                     | List of the different stratification groups identifiers. |
| `description` | \`str                           | None\`                                                   |
| `conditions`  | \`list[StratificationCondition] | None\`                                                   |

Source code in `commol/_model/stratification.py`

```python
class Stratification(BaseModel):
    """
    Defines a categorical subdivision of the compartments.

    Attributes
    ----------
    id : str
        Identifier of the stratification.
    categories : list[str]
        List of the different stratification groups identifiers.
    description : str | None
        A human-readable description of the stratification.
    conditions : list[StratificationCondition] | None
        When set, this stratification only expands compartments whose
        already-applied stratification categories satisfy ALL conditions.
        Compartments that do not satisfy the conditions are kept without
        appending this stratification's categories.

        May only reference stratifications declared before this one.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default=..., description="Identifier of the stratification.")
    categories: list[str] = Field(
        default=...,
        description="List of the different stratification groups identifiers.",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the stratification."
    )
    conditions: list[StratificationCondition] | None = Field(
        default=None,
        description=(
            "Conditions on previously-declared stratifications that must be "
            "satisfied for this stratification to expand a compartment."
        ),
    )

    @override
    def __hash__(self) -> int:
        return hash(self.id)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Stratification) and self.id == other.id

    @model_validator(mode="after")
    def _validate_categories_length(self) -> Self:
        """
        Enforces that categories are not empty.
        """
        if not self.categories:
            raise ValueError(
                (f"Stratification '{self.id}' must have at least one category.")
            )
        return self

    @model_validator(mode="after")
    def _validate_categories_uniqueness(self) -> Self:
        """
        Enforces that categories are not repeated.
        """
        categories_set = set(self.categories)

        if len(categories_set) != len(self.categories):
            duplicates = [
                item for item in categories_set if self.categories.count(item) > 1
            ]
            raise ValueError(
                (
                    f"Categories for stratification '{self.id}' must not be repeated. "
                    f"Found duplicates: {list(set(duplicates))}."
                )
            )

        return self
```

## CategoryMatrix

Bases: `BaseModel`

A matrix between the categories of a stratification.

`ModelBuilder.add_matrix` adds one to the process.

Attributes:

| Name             | Type                          | Description                                                                                                             |
| ---------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `id`             | `str`                         | The id that matrix_sum reads, and the start of the names of its variables.                                              |
| `stratification` | `str`                         | The id of the stratification whose categories are its rows and its columns.                                             |
| `values`         | `dict[str, dict[str, float]]` | values[i][j], the value of row i and column j: one finite number for each pair of categories of the stratification.     |
| `unit`           | \`str                         | None\`                                                                                                                  |
| `description`    | \`str                         | None\`                                                                                                                  |
| `row_scales`     | `dict[str, str]`              | The id of the parameter that multiplies the row of a category, by category; a category without one keeps its row.       |
| `column_scales`  | `dict[str, str]`              | The id of the parameter that multiplies the column of a category, by category; a category without one keeps its column. |

Source code in `commol/_model/matrix.py`

```python
class CategoryMatrix(BaseModel):
    """
    A matrix between the categories of a stratification.

    ``ModelBuilder.add_matrix`` adds one to the process.

    Attributes
    ----------
    id : str
        The id that ``matrix_sum`` reads, and the start of the names of its
        variables.
    stratification : str
        The id of the stratification whose categories are its rows and its
        columns.
    values : dict[str, dict[str, float]]
        ``values[i][j]``, the value of row ``i`` and column ``j``: one finite
        number for each pair of categories of the stratification.
    unit : str | None
        The unit of its values, which the unit check reads.
    description : str | None
        A human-readable description.
    row_scales : dict[str, str]
        The id of the parameter that multiplies the row of a category, by
        category; a category without one keeps its row.
    column_scales : dict[str, str]
        The id of the parameter that multiplies the column of a category, by
        category; a category without one keeps its column.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default=..., description="Identifier of the matrix.")
    stratification: str = Field(
        default=..., description="Stratification of its rows and columns."
    )
    values: dict[str, dict[str, float]] = Field(
        default=..., description="The value of each row and column, by category."
    )
    unit: str | None = Field(default=None, description="Unit of its values.")
    description: str | None = Field(
        default=None, description="Human-readable description of the matrix."
    )
    row_scales: dict[str, str] = Field(
        default_factory=dict,
        description="The parameter that multiplies the row of each category.",
    )
    column_scales: dict[str, str] = Field(
        default_factory=dict,
        description="The parameter that multiplies the column of each category.",
    )

    def scales(self) -> list[tuple[str, str, str]]:
        """
        Each scale, as ``("row" or "column", category, parameter id)``: the
        rows first, then the columns.
        """
        return [
            (side, category, parameter)
            for side, scales in (
                ("row", self.row_scales),
                ("column", self.column_scales),
            )
            for category, parameter in scales.items()
        ]
```

### Methods:

#### scales

```python
scales() -> list[tuple[str, str, str]]
```

Each scale, as `("row" or "column", category, parameter id)`: the rows first, then the columns.

Source code in `commol/_model/matrix.py`

```python
def scales(self) -> list[tuple[str, str, str]]:
    """
    Each scale, as ``("row" or "column", category, parameter id)``: the
    rows first, then the columns.
    """
    return [
        (side, category, parameter)
        for side, scales in (
            ("row", self.row_scales),
            ("column", self.column_scales),
        )
        for category, parameter in scales.items()
    ]
```

## Parameter

Bases: `BaseModel`

Defines a global model parameter.

Attributes:

| Name          | Type    | Description                      |
| ------------- | ------- | -------------------------------- |
| `id`          | `str`   | The identifier of the parameter. |
| `value`       | \`float | str                              |
| `description` | \`str   | None\`                           |
| `unit`        | \`str   | None\`                           |

Source code in `commol/_model/parameter.py`

```python
class Parameter(BaseModel):
    """
    Defines a global model parameter.

    Attributes
    ----------
    id : str
        The identifier of the parameter.
    value : float | str | SeriesValue | None
        Value of the parameter. Can be:
        - float: A numerical constant value
        - str: A mathematical formula that can reference other parameters,
               special variables (total, total_<category>, step, t, pi, e), or contain
               mathematical expressions
        - SeriesValue: An empirical series evaluated via binary search
        - None: Indicates that the parameter needs to be calibrated before use
    description : str | None
        A human-readable description of the parameter.
    unit : str | None
        The unit of the parameter (e.g., "1/day", "dimensionless", "item").
        If None, the parameter has no unit specified.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default=..., description="Identifier of the parameter.")
    value: float | str | SeriesValue | None = Field(
        default=...,
        description=(
            "Value of the parameter. Can be a float (constant), "
            "str (formula), SeriesValue (empirical series), "
            "or None (requires calibration)."
        ),
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the parameter."
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the parameter (e.g., '1/day', 'dimensionless', 'item').",
    )

    @field_validator("value")
    @classmethod
    def _validate_value(
        cls, value: float | str | SeriesValue | None
    ) -> float | str | SeriesValue | None:
        """Validate the parameter value."""
        if value is None or isinstance(value, SeriesValue):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        return value.strip()

    @model_validator(mode="after")
    def _validate_formula(self) -> Self:
        """Check a formula value as a rate is checked, with the parser of the
        engine."""
        if isinstance(self.value, str):
            try:
                commol_rs.core.MathExpression(self.value).check()
            except ValueError as error:
                raise ValueError(f"Parameter '{self.id}': {error}")
        return self

    def is_calibrated(self) -> bool:
        """
        Check if the parameter has a value (is calibrated).

        Returns
        -------
        bool
            True if the parameter has a value, False if it needs calibration.
        """
        return self.value is not None

    def with_value(
        self, value: float | str | list[tuple[float, float]] | SeriesValue | None
    ) -> "Parameter":
        """
        This parameter with another value.

        Parameters
        ----------
        value : float | str | list[tuple[float, float]] | SeriesValue | None
            The new value, as `Parameter.value` takes it. A list of
            (position, value) points replaces the points of a series and keeps
            its mode; a `SeriesValue` sets both.

        Returns
        -------
        Parameter
            A new parameter with the same id, description and unit.

        Raises
        ------
        ValueError
            If ``value`` is a list and this parameter is not a series, or
            ``value`` is not a valid value; the message names the parameter.
        """
        try:
            if isinstance(value, list):
                if not isinstance(self.value, SeriesValue):
                    raise ValueError(
                        f"Parameter '{self.id}' is not a series, so a list of "
                        "points has no mode: pass a SeriesValue(data=..., mode=...)"
                    )
                value = SeriesValue(data=value, mode=self.value.mode)
            return Parameter(
                id=self.id, value=value, description=self.description, unit=self.unit
            )
        except ValidationError as error:
            reasons = "; ".join(detail["msg"] for detail in error.errors())
            raise ValueError(
                f"Parameter '{self.id}' cannot take {value!r}: {reasons}"
            ) from None
```

### Methods:

#### is_calibrated

```python
is_calibrated() -> bool
```

Check if the parameter has a value (is calibrated).

Returns:

| Type   | Description                                                       |
| ------ | ----------------------------------------------------------------- |
| `bool` | True if the parameter has a value, False if it needs calibration. |

Source code in `commol/_model/parameter.py`

```python
def is_calibrated(self) -> bool:
    """
    Check if the parameter has a value (is calibrated).

    Returns
    -------
    bool
        True if the parameter has a value, False if it needs calibration.
    """
    return self.value is not None
```

#### with_value

```python
with_value(value: float | str | list[tuple[float, float]] | SeriesValue | None) -> Parameter
```

This parameter with another value.

Parameters:

| Name    | Type    | Description | Default                     |
| ------- | ------- | ----------- | --------------------------- |
| `value` | \`float | str         | list\[tuple[float, float]\] |

Returns:

| Type        | Description                                             |
| ----------- | ------------------------------------------------------- |
| `Parameter` | A new parameter with the same id, description and unit. |

Raises:

| Type         | Description                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If value is a list and this parameter is not a series, or value is not a valid value; the message names the parameter. |

Source code in `commol/_model/parameter.py`

```python
def with_value(
    self, value: float | str | list[tuple[float, float]] | SeriesValue | None
) -> "Parameter":
    """
    This parameter with another value.

    Parameters
    ----------
    value : float | str | list[tuple[float, float]] | SeriesValue | None
        The new value, as `Parameter.value` takes it. A list of
        (position, value) points replaces the points of a series and keeps
        its mode; a `SeriesValue` sets both.

    Returns
    -------
    Parameter
        A new parameter with the same id, description and unit.

    Raises
    ------
    ValueError
        If ``value`` is a list and this parameter is not a series, or
        ``value`` is not a valid value; the message names the parameter.
    """
    try:
        if isinstance(value, list):
            if not isinstance(self.value, SeriesValue):
                raise ValueError(
                    f"Parameter '{self.id}' is not a series, so a list of "
                    "points has no mode: pass a SeriesValue(data=..., mode=...)"
                )
            value = SeriesValue(data=value, mode=self.value.mode)
        return Parameter(
            id=self.id, value=value, description=self.description, unit=self.unit
        )
    except ValidationError as error:
        reasons = "; ".join(detail["msg"] for detail in error.errors())
        raise ValueError(
            f"Parameter '{self.id}' cannot take {value!r}: {reasons}"
        ) from None
```

## Transition

Bases: `BaseModel`

Defines a rule for system evolution.

Attributes:

| Name               | Type                   | Description                                                      |
| ------------------ | ---------------------- | ---------------------------------------------------------------- |
| `id`               | `str`                  | Id of the transition.                                            |
| `source`           | `list[str]`            | The origin compartments.                                         |
| `target`           | `list[str]`            | The destination compartments.                                    |
| `accumulators`     | `list[str]`            | Ids of the accumulators that add up the flow of this transition. |
| `rate`             | \`str                  | None\`                                                           |
| `stratified_rates` | \`list[StratifiedRate] | None\`                                                           |
| `per_compartment`  | \`bool                 | None\`                                                           |
| `impulse_at`       | \`list[float]          | None\`                                                           |
| `impulse_every`    | \`tuple[float, float]  | None\`                                                           |

Source code in `commol/_model/transition.py`

```python
class Transition(BaseModel):
    """
    Defines a rule for system evolution.

    Attributes
    ----------
    id : str
        Id of the transition.
    source : list[str]
        The origin compartments.
    target : list[str]
        The destination compartments.
    accumulators : list[str]
        Ids of the accumulators that add up the flow of this transition.
    rate : str | None
        Default mathematical formula, parameter name, or constant value for the flow.
        Used when no stratified rate matches. ``ModelBuilder.add_transition``
        writes a number as its string.

        Operators: +, -, *, /, % (modulo), ^ or ** (power)
        Functions: sin, cos, tan, exp, ln, sqrt, abs, min, max, if, etc.
        Constants: pi, e

        Note: Both ^ and ** are supported for exponentiation (** is converted to ^).

        Examples:
        - "k1" (parameter reference)
        - "0.5" (constant, can also be passed as float 0.5)
        - "k1 * A * B / total" (mathematical formula)
        - "0.3 * sin(2 * pi * t / 365)" (formula that changes along the axis)
        - "2^10" or "2**10" (power: both syntaxes work)
    stratified_rates : list[StratifiedRate] | None
        Stratification-specific rates. Each rate applies to compartments that match
        all specified stratification conditions.
    per_compartment : bool | None
        When True, the id of each source and target bin in a rate reads the
        compartment of each flow; any other bin keeps its total. A transition
        between categories ignores it.
    impulse_at : list[float] | None
        Positions on the axis where the transition is an impulse: at each one,
        the rate is a fraction (proportional) or an amount (absolute) moved at
        once. Between them the transition moves nothing.
    impulse_every : tuple[float, float] | None
        ``(period, offset)``: the transition is an impulse at ``offset``,
        ``offset + period``, ``offset + 2 * period``, and so on. At most one
        of `impulse_at` and `impulse_every` is set.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default=..., description="Id of the transition.")
    source: list[str] = Field(default=..., description="Origin compartments.")
    target: list[str] = Field(default=..., description="Destination compartments.")
    accumulators: list[str] = Field(
        default_factory=list,
        description="Cumulative event counters incremented by this transition flow.",
    )

    rate: str | None = Field(
        None,
        description=(
            "Default rate expression (fallback when no stratified rate matches). "
            "Can be a parameter reference (e.g., 'k1'), a constant (e.g., '0.5'), "
            "or a mathematical expression (e.g., 'k1 * A * B / total')."
        ),
    )

    stratified_rates: list[StratifiedRate] | None = Field(
        default=None, description="List of stratification-specific rates"
    )

    per_compartment: bool | None = Field(
        default=None,
        description=(
            "When True, the id of each source and target bin in a rate reads "
            "the compartment of each flow; any other bin keeps its total. A "
            "transition between categories ignores it."
        ),
    )

    impulse_at: list[float] | None = Field(
        default=None,
        description=(
            "Positions on the axis where the transition is an impulse: at each "
            "one, the rate is a fraction (proportional) or an amount (absolute) "
            "moved at once. Between them the transition moves nothing."
        ),
    )

    impulse_every: tuple[float, float] | None = Field(
        default=None,
        description=(
            "(period, offset): the transition is an impulse at offset, "
            "offset + period, offset + 2 * period, ..."
        ),
    )

    @property
    def is_impulse(self) -> bool:
        """Whether the transition is an impulse rather than a continuous flow."""
        return self.impulse_at is not None or self.impulse_every is not None

    @model_validator(mode="after")
    def _validate_impulse(self) -> Self:
        """
        Check the impulse with the rule of the engine (`check_impulse` of
        `crates/commol-model/src/values.rs`): one of `impulse_at` and
        `impulse_every`, at least one position on the axis, a positive period
        and an offset that is not negative.
        """
        commol_rs.model.check_impulse(self.id, self.impulse_at, self.impulse_every)
        return self

    @field_validator("rate")
    @classmethod
    def _validate_rate(cls, value: str | None) -> str | None:
        """Check the rate, when there is one, with the parser of the engine."""
        if value is None:
            return value
        return _checked_rate(value)
```

### Attributes

#### is_impulse

```python
is_impulse: bool
```

Whether the transition is an impulse rather than a continuous flow.
