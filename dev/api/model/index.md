# Model API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The validated model that `ModelBuilder.build` returns: a header, the process written once, the structure that lays it out (`Aggregate`, with its neighbors `Complete` and `Groups`), and the entries of `Aggregate.initial_state()`. Guide: [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md), [Building Models](https://munqu.github.io/commol/latest/guide/building-models/#the-structure-aggregate), [Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md), [Units](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md). The tree of types is in [Model Structure](https://munqu.github.io/commol/latest/advanced/model-structure/index.md).

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

Source code in `commol/context/model.py`

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

    @field_validator("structure", mode="before")
    @classmethod
    def read_structure_kind(cls, value: object) -> object:
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
        return ModelStructure.of_kind(kind, variant).model_validate(value)

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
            if types[param_id] == CalibrationParameterType.PARAMETER
        }
        # Computed before any mutation, so a rejected value leaves the model
        # unchanged.
        structure = self.structure.with_calibrated_values(self, problem, values)

        if parameters:
            self.update_parameters(parameters)
        self.structure = structure

    def describe(self, layout: Layout = "default") -> ModelDescription:
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
        from commol.context._model.describe import describe_model

        return describe_model(self, layout)

    def axis_units(self) -> AxisUnits:
        """
        The unit of one step of the model, and its engine, to convert values.

        The step unit is ``step_unit`` or, when the model declares none,
        ``step``, the default increment of any axis. The engine is the one of
        the model, so the rates of `AxisUnits` mean what this engine reads.

        Returns
        -------
        AxisUnits
            The converter of the model.

        Examples
        --------
        >>> units = model.axis_units()
        >>> units.steps(2, "week")  # with step_unit="day"
        14.0
        >>> gamma = units.rate_from_mean_span(5, "day")
        """
        return AxisUnits(
            step_unit=self.step_unit or DEFAULT_STEP_UNIT, engine=self.engine.type
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

#### read_structure_kind

```python
read_structure_kind(value: object) -> object
```

Read a structure mapping as the class of its `kind` and `variant`.

Source code in `commol/context/model.py`

```python
@field_validator("structure", mode="before")
@classmethod
def read_structure_kind(cls, value: object) -> object:
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
    return ModelStructure.of_kind(kind, variant).model_validate(value)
```

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

Source code in `commol/context/model.py`

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

Source code in `commol/context/model.py`

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

#### validate_engine_runs_structure

```python
validate_engine_runs_structure() -> Self
```

Ask the engine whether it runs the kind of the structure.

Source code in `commol/context/model.py`

```python
@model_validator(mode="after")
def validate_engine_runs_structure(self) -> Self:
    """Ask the engine whether it runs the kind of the structure."""
    commol_rs.engine.EngineType(self.engine.type).check_kind(self.structure.kind)
    return self
```

#### validate_unique_parameter_ids

```python
validate_unique_parameter_ids() -> Self
```

Validates that parameter IDs are unique.

Source code in `commol/context/model.py`

```python
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
```

#### validate_structure

```python
validate_structure() -> Self
```

Check the structure against the parameters and the engine.

Source code in `commol/context/model.py`

```python
@model_validator(mode="after")
def validate_structure(self) -> Self:
    """Check the structure against the parameters and the engine."""
    self.structure.check_model(self)
    return self
```

#### update_parameters

```python
update_parameters(parameter_values: Mapping[str, float | None]) -> None
```

Update parameter values in the model.

Parameters:

| Name               | Type                  | Description | Default                                               |
| ------------------ | --------------------- | ----------- | ----------------------------------------------------- |
| `parameter_values` | \`Mapping\[str, float | None\]\`    | Dictionary mapping parameter IDs to their new values. |

Raises:

| Type         | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| `ValueError` | If a parameter ID in the dictionary doesn't exist in the model. |

Source code in `commol/context/model.py`

```python
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

Source code in `commol/context/model.py`

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

Source code in `commol/context/model.py`

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
        if types[param_id] == CalibrationParameterType.PARAMETER
    }
    # Computed before any mutation, so a rejected value leaves the model
    # unchanged.
    structure = self.structure.with_calibrated_values(self, problem, values)

    if parameters:
        self.update_parameters(parameters)
    self.structure = structure
```

#### describe

```python
describe(layout: Layout = 'default') -> ModelDescription
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

Source code in `commol/context/model.py`

```python
def describe(self, layout: Layout = "default") -> ModelDescription:
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
    from commol.context._model.describe import describe_model

    return describe_model(self, layout)
```

#### axis_units

```python
axis_units() -> AxisUnits
```

The unit of one step of the model, and its engine, to convert values.

The step unit is `step_unit` or, when the model declares none, `step`, the default increment of any axis. The engine is the one of the model, so the rates of `AxisUnits` mean what this engine reads.

Returns:

| Type        | Description                 |
| ----------- | --------------------------- |
| `AxisUnits` | The converter of the model. |

Examples:

```pycon
>>> units = model.axis_units()
>>> units.steps(2, "week")  # with step_unit="day"
14.0
>>> gamma = units.rate_from_mean_span(5, "day")
```

Source code in `commol/context/model.py`

```python
def axis_units(self) -> AxisUnits:
    """
    The unit of one step of the model, and its engine, to convert values.

    The step unit is ``step_unit`` or, when the model declares none,
    ``step``, the default increment of any axis. The engine is the one of
    the model, so the rates of `AxisUnits` mean what this engine reads.

    Returns
    -------
    AxisUnits
        The converter of the model.

    Examples
    --------
    >>> units = model.axis_units()
    >>> units.steps(2, "week")  # with step_unit="day"
    14.0
    >>> gamma = units.rate_from_mean_span(5, "day")
    """
    return AxisUnits(
        step_unit=self.step_unit or DEFAULT_STEP_UNIT, engine=self.engine.type
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

Source code in `commol/context/model.py`

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

| Name              | Type                   | Description                                                                   |
| ----------------- | ---------------------- | ----------------------------------------------------------------------------- |
| `bins`            | `list[Bin]`            | The states of the process. Stratifications expand each one into compartments. |
| `accumulators`    | `list[Accumulator]`    | Counters of the flow of the transitions that name them.                       |
| `stratifications` | `list[Stratification]` | Categorical subdivisions of the bins, in expansion order.                     |
| `transitions`     | `list[Transition]`     | Flows between bins, or between the categories of one stratification.          |

Source code in `commol/context/process.py`

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
    transitions : list[Transition]
        Flows between bins, or between the categories of one stratification.
    """

    model_config = ConfigDict(extra="forbid")

    bins: list[Bin] = Field(default_factory=list)
    accumulators: list[Accumulator] = Field(default_factory=list)
    stratifications: list[Stratification] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
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
    def validate_stratified_rates(self) -> Self:
        """Validate that stratified rates name stratifications and categories."""
        strat_map = {strat.id: strat for strat in self.stratifications}

        for transition in self.transitions:
            for idx, stratified_rate in enumerate(transition.stratified_rates or []):
                for condition in stratified_rate.conditions:
                    if condition.stratification not in strat_map:
                        raise ValueError(
                            f"In transition '{transition.id}', stratified rate "
                            f"{idx}: Stratification "
                            f"'{condition.stratification}' not found. "
                            f"Available: {list(strat_map.keys())}"
                        )

                    # A condition without a category only redirects the target.
                    strat = strat_map[condition.stratification]
                    if (
                        condition.category is not None
                        and condition.category not in strat.categories
                    ):
                        raise ValueError(
                            f"In transition '{transition.id}', stratified rate "
                            f"{idx}: Category '{condition.category}' not found "
                            f"in stratification '{condition.stratification}'. "
                            f"Available: {strat.categories}"
                        )
        return self

    @model_validator(mode="after")
    def validate_transition_ids(self) -> Self:
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
            True when it has no bin, accumulator, stratification or
            transition.
        """
        return not (
            self.bins or self.accumulators or self.stratifications or self.transitions
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

#### validate_unique_ids

```python
validate_unique_ids() -> Self
```

Validate that bin, stratification and transition ids are unique.

Source code in `commol/context/process.py`

```python
@model_validator(mode="after")
def validate_unique_ids(self) -> Self:
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
```

#### validate_stratified_rates

```python
validate_stratified_rates() -> Self
```

Validate that stratified rates name stratifications and categories.

Source code in `commol/context/process.py`

```python
@model_validator(mode="after")
def validate_stratified_rates(self) -> Self:
    """Validate that stratified rates name stratifications and categories."""
    strat_map = {strat.id: strat for strat in self.stratifications}

    for transition in self.transitions:
        for idx, stratified_rate in enumerate(transition.stratified_rates or []):
            for condition in stratified_rate.conditions:
                if condition.stratification not in strat_map:
                    raise ValueError(
                        f"In transition '{transition.id}', stratified rate "
                        f"{idx}: Stratification "
                        f"'{condition.stratification}' not found. "
                        f"Available: {list(strat_map.keys())}"
                    )

                # A condition without a category only redirects the target.
                strat = strat_map[condition.stratification]
                if (
                    condition.category is not None
                    and condition.category not in strat.categories
                ):
                    raise ValueError(
                        f"In transition '{transition.id}', stratified rate "
                        f"{idx}: Category '{condition.category}' not found "
                        f"in stratification '{condition.stratification}'. "
                        f"Available: {strat.categories}"
                    )
    return self
```

#### validate_transition_ids

```python
validate_transition_ids() -> Self
```

Check that each transition moves between bins, or between the categories of one stratification.

Source code in `commol/context/process.py`

```python
@model_validator(mode="after")
def validate_transition_ids(self) -> Self:
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
```

#### is_empty

```python
is_empty() -> bool
```

Whether the process holds nothing.

Returns:

| Type   | Description                                                         |
| ------ | ------------------------------------------------------------------- |
| `bool` | True when it has no bin, accumulator, stratification or transition. |

Source code in `commol/context/process.py`

```python
def is_empty(self) -> bool:
    """
    Whether the process holds nothing.

    Returns
    -------
    bool
        True when it has no bin, accumulator, stratification or
        transition.
    """
    return not (
        self.bins or self.accumulators or self.stratifications or self.transitions
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

Source code in `commol/context/process.py`

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

Source code in `commol/context/structure.py`

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
    def of_kind(cls, kind: str, variant: str | None = None) -> type["ModelStructure"]:
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

    def calibration_value_types(self) -> frozenset[CalibrationParameterType]:
        """
        Types of value of this kind that calibration may set.

        ``parameter`` and ``factor`` are not here: every kind accepts them.
        The engine's list of kinds declares the others.

        Returns
        -------
        frozenset[CalibrationParameterType]
            For example ``{"initial_condition"}`` for an aggregate model.
        """
        return frozenset(
            CalibrationParameterType(name)
            for name in commol_rs.engine.ModelKind(self.kind).value_types()
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
        parameter_statistics: "Mapping[str, ParameterSummary] | None",  # noqa: ARG002
    ) -> ReportSections:
        """
        Sections of this kind in a report of applied values.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, ParameterSummary] | None
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

Source code in `commol/context/structure.py`

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

#### of_kind

```python
of_kind(kind: str, variant: str | None = None) -> type[ModelStructure]
```

The structure class of a model kind, or of one of its variants.

Parameters:

| Name      | Type  | Description                            | Default                                                     |
| --------- | ----- | -------------------------------------- | ----------------------------------------------------------- |
| `kind`    | `str` | Name of the kind, such as "aggregate". | *required*                                                  |
| `variant` | \`str | None\`                                 | Name of a variant of the kind, or None for the kind itself. |

Returns:

| Type                   | Description                       |
| ---------------------- | --------------------------------- |
| `type[ModelStructure]` | The subclass registered for them. |

Raises:

| Type         | Description                                      |
| ------------ | ------------------------------------------------ |
| `ValueError` | If no structure class has that kind and variant. |

Source code in `commol/context/structure.py`

```python
@classmethod
def of_kind(cls, kind: str, variant: str | None = None) -> type["ModelStructure"]:
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

Source code in `commol/context/structure.py`

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

Source code in `commol/context/structure.py`

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

Source code in `commol/context/structure.py`

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
calibration_value_types() -> frozenset[CalibrationParameterType]
```

Types of value of this kind that calibration may set.

`parameter` and `factor` are not here: every kind accepts them. The engine's list of kinds declares the others.

Returns:

| Type                                  | Description                                               |
| ------------------------------------- | --------------------------------------------------------- |
| `frozenset[CalibrationParameterType]` | For example {"initial_condition"} for an aggregate model. |

Source code in `commol/context/structure.py`

```python
def calibration_value_types(self) -> frozenset[CalibrationParameterType]:
    """
    Types of value of this kind that calibration may set.

    ``parameter`` and ``factor`` are not here: every kind accepts them.
    The engine's list of kinds declares the others.

    Returns
    -------
    frozenset[CalibrationParameterType]
        For example ``{"initial_condition"}`` for an aggregate model.
    """
    return frozenset(
        CalibrationParameterType(name)
        for name in commol_rs.engine.ModelKind(self.kind).value_types()
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

Source code in `commol/context/structure.py`

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

Source code in `commol/context/structure.py`

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

Source code in `commol/context/structure.py`

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
report_sections(model: Model[Self], calibrated_ids: set[str], parameter_statistics: Mapping[str, ParameterSummary] | None) -> ReportSections
```

Sections of this kind in a report of applied values.

Parameters:

| Name                   | Type                             | Description                              | Default                                                   |
| ---------------------- | -------------------------------- | ---------------------------------------- | --------------------------------------------------------- |
| `model`                | `Model`                          | The model that holds the applied values. | *required*                                                |
| `calibrated_ids`       | `set[str]`                       | Ids that calibration set.                | *required*                                                |
| `parameter_statistics` | \`Mapping[str, ParameterSummary] | None\`                                   | Statistics of each id from a sample, or None without one. |

Returns:

| Type             | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `ReportSections` | Each section by its key in the report; none by default. |

Source code in `commol/context/structure.py`

```python
def report_sections(
    self,
    model: "Model[Self]",  # noqa: ARG002
    calibrated_ids: set[str],  # noqa: ARG002
    parameter_statistics: "Mapping[str, ParameterSummary] | None",  # noqa: ARG002
) -> ReportSections:
    """
    Sections of this kind in a report of applied values.

    Parameters
    ----------
    model : Model
        The model that holds the applied values.
    calibrated_ids : set[str]
        Ids that calibration set.
    parameter_statistics : Mapping[str, ParameterSummary] | None
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

Source code in `commol/context/structure.py`

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

Source code in `commol/context/structure.py`

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

The bins of the process expand over its stratifications into compartments. `bin_amounts` gives the amount of each bin at the start, in the unit of the bin, `stratification_fractions` splits it among the categories, and `compartment_amounts` overrides single compartments. Bins of different units, such as people and a concentration, need no total over them.

Attributes:

| Name                       | Type                          | Description                                                                                                                                                                                                                   |
| -------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`                     | `Literal['aggregate']`        | The model kind.                                                                                                                                                                                                               |
| `variant`                  | \`str                         | None\`                                                                                                                                                                                                                        |
| `bin_amounts`              | \`dict\[str, float            | None\]\`                                                                                                                                                                                                                      |
| `stratification_fractions` | `dict[str, dict[str, float]]` | The fraction of each category, by stratification id, such as {"group": {"g1": 0.3, "g2": 0.7}}: the split of the amount of every bin it applies to. A conditional stratification splits only the group its conditions select. |
| `compartment_amounts`      | `dict[str, float]`            | Amounts of single expanded compartments, such as {"A_g1": 0.0}, applied after the expansion. The names are checked against the engine when the model is built.                                                                |
| `neighbors`                | \`Complete                    | Groups                                                                                                                                                                                                                        |

Source code in `commol/context/aggregate.py`

```python
class Aggregate(ModelStructure, InitialConditions):
    """
    Amounts per compartment: how much starts where, and who the neighbors of
    a unit are.

    The bins of the process expand over its stratifications into
    compartments. ``bin_amounts`` gives the amount of each bin at the start,
    in the unit of the bin, ``stratification_fractions`` splits it among the
    categories, and ``compartment_amounts`` overrides single compartments.
    Bins of different units, such as people and a concentration, need no
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
    stratification_fractions : dict[str, dict[str, float]]
        The fraction of each category, by stratification id, such as
        ``{"group": {"g1": 0.3, "g2": 0.7}}``: the split of the amount of
        every bin it applies to. A conditional stratification splits only the
        group its conditions select.
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
    neighbors: Neighbors | None = Field(
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
        bin and the fractions every category of the process, parameter ids
        must not be engine variables or bin ids, every variable of a rate must
        be defined, and the engine must plan every transition, resolve the
        neighbors and build the initial state.

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
        self._check_stratification_fractions(process)
        self._check_parameter_names(model)
        engine_state.transition_flows(process, self)
        engine_state.check_model(model)
        if self.compartment_amounts:
            engine_state.initial_state(process, self)
        self._check_formula_variables(model)

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
            param
            for param in parameters
            if param.parameter_type == CalibrationParameterType.INITIAL_CONDITION
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
        from commol.context._model.calibration_write_back import (
            calibrated_structure,
        )

        return calibrated_structure(model, problem, values)

    def report_sections(
        self,
        model: "Model[Self]",
        calibrated_ids: set[str],
        parameter_statistics: "Mapping[str, ParameterSummary] | None",
    ) -> ReportSections:
        """
        The ``initial_amounts`` and ``category_splits`` sections of a report.

        See :func:`commol.applied_parameters_report`.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, ParameterSummary] | None
            Statistics of each id from a sample, or None without one.

        Returns
        -------
        ReportSections
            ``{"initial_amounts": ..., "category_splits": ...}``.
        """
        from commol.context._model.report import (
            category_splits_report,
            initial_amounts_report,
        )

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
        from commol.context._model.aggregate_description import describe_aggregate

        return describe_aggregate(model)

    def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
        """Check the units of the model; see `Model.check_unit_consistency`."""
        from commol.context._model.units import UnitChecker

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
            subdivides, empty for an unconditional stratification. A category
            name another stratification also declares is qualified, as in
            ``group:g1``.

        Raises
        ------
        ValueError
            If the stratification is not found.
        """
        for stratification in process.stratifications:
            if stratification.id == stratification_id:
                return tuple(
                    self.category_id(condition.stratification, condition.category)
                    for condition in (stratification.conditions or [])
                    if condition.category is not None
                )
        raise ValueError(
            f"Stratification '{stratification_id}' not found. Available: "
            f"{sorted(s.id for s in process.stratifications)}"
        )

    def subgroup_total(self, process: Process, categories: Iterable[str] = ()) -> float:
        """
        Get the initial sum of the compartments of a subgroup.

        The sum is taken of the state the engine builds, overrides included,
        over every compartment that has each of the categories. Passing no
        categories gives the sum of every compartment. It adds every bin it
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
        resolved = [self.resolve_category(c) for c in categories]
        return sum(
            compartment.amount
            for compartment in self.initial_state(process)
            if all(
                compartment.categories.get(stratification) == name
                for stratification, name in resolved
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

    def _check_stratification_fractions(self, process: Process) -> None:
        """Check the stratification fractions against the stratifications."""
        strat_map = {strat.id: strat for strat in process.stratifications}
        actual_strat = set(self.stratification_fractions)
        expected_strat = set(strat_map)
        if actual_strat != expected_strat:
            raise ValueError(
                f"Initial stratification fractions keys must exactly match "
                f"stratification ids. "
                f"Missing ids: {sorted(expected_strat - actual_strat)}, "
                f"Extra ids: {sorted(actual_strat - expected_strat)}."
            )

        for strat_id, fractions in self.stratification_fractions.items():
            strat_instance = strat_map[strat_id]

            expected = set(strat_instance.categories)
            actual = set(fractions)
            if actual != expected:
                raise ValueError(
                    f"Categories for stratification '{strat_id}' must exactly "
                    f"match defined categories in instance '{strat_instance.id}'. "
                    f"Missing categories: {sorted(expected - actual)}, "
                    f"Extra categories: {sorted(actual - expected)}."
                )

            total = sum(fractions.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"Stratification fractions for '{strat_id}' must sum to 1.0, "
                    f"but got {total:.7}."
                )

    def _check_parameter_names(self, model: "Model[Self]") -> None:
        """
        Check that parameter ids are not engine variables or bin ids.

        Expanded compartment names, subgroup totals and the names of the
        neighbors are checked by the engine (``engine_state.check_model``).
        """
        bin_ids = {bin_item.id for bin_item in model.process.bins}
        parameter_ids = {p.id for p in model.parameters}

        special = sorted(parameter_ids & set(RESERVED_VARIABLE_NAMES))
        if special:
            raise ValueError(
                f"Parameter IDs conflict with engine variables: {special}. "
                f"{', '.join(RESERVED_VARIABLE_NAMES)} are reserved. "
                f"Please rename the conflicting parameters."
            )

        conflicting_names = bin_ids & parameter_ids
        if conflicting_names:
            raise ValueError(
                f"Parameter IDs conflict with reserved compartment names: "
                f"{sorted(conflicting_names)}. "
                f"Bin IDs ({sorted(bin_ids)}) are reserved variables. "
                f"Please rename the conflicting parameters."
            )

    def _check_formula_variables(self, model: "Model[Self]") -> None:
        """Check that every variable of a rate expression is defined."""
        parameter_ids = {param.id for param in model.parameters}
        valid_identifiers = parameter_ids | self._process_identifiers(model.process)

        for transition in model.process.transitions:
            rates = [("rate", transition.rate)] if transition.rate else []
            rates += [
                ("stratified_rate", stratified.rate)
                for stratified in transition.stratified_rates or []
            ]
            for context, rate in rates:
                self._check_rate_variables(
                    model.process,
                    rate,
                    transition.id,
                    context,
                    valid_identifiers,
                    parameter_ids,
                )

    def _process_identifiers(self, process: Process) -> set[str]:
        """Identifiers the process, the structure and the engine define."""
        bin_ids = {bin_item.id for bin_item in process.bins}
        category_ids = {
            cat for strat in process.stratifications for cat in strat.categories
        }
        neighbor_terms = {NEIGHBORS_VARIABLE} | {
            f"{NEIGHBORS_PREFIX}{bin_id}" for bin_id in bin_ids
        }
        return (
            bin_ids
            | category_ids
            | set(RESERVED_VARIABLE_NAMES)
            | neighbor_terms
            | set(engine_state.subgroup_total_names(process))
            | self._full_compartment_names(process)
        )

    def _full_compartment_names(self, process: Process) -> set[str]:
        """Returns all full stratified compartment names."""
        if not process.stratifications:
            return set()
        compartments, _ = engine_state.expanded_outputs(process)
        return {output.name for output in compartments}

    def _check_rate_variables(
        self,
        process: Process,
        rate: str,
        transition_id: str,
        context: str,
        valid_identifiers: set[str],
        parameter_ids: set[str],
    ) -> None:
        """Check the variables of one rate expression."""
        variables = get_expression_variables(rate)
        undefined_vars = [var for var in variables if var not in valid_identifiers]
        if not undefined_vars:
            return
        bin_ids = {bin_item.id for bin_item in process.bins}
        message = (
            f"Undefined variables in transition '{transition_id}' "
            f"{context} '{rate}': {', '.join(undefined_vars)}. "
            f"Available parameters: "
            f"{', '.join(sorted(parameter_ids)) if parameter_ids else 'none'}. "
            f"Available bins: "
            f"{', '.join(sorted(bin_ids)) if bin_ids else 'none'}."
        )
        raise ValueError(message)
```

### Methods:

#### check_model

```python
check_model(model: Model[Self]) -> None
```

Check the structure against the process and the parameters.

The process needs a bin and a transition, the amounts must name every bin and the fractions every category of the process, parameter ids must not be engine variables or bin ids, every variable of a rate must be defined, and the engine must plan every transition, resolve the neighbors and build the initial state.

Parameters:

| Name    | Type    | Description                          | Default    |
| ------- | ------- | ------------------------------------ | ---------- |
| `model` | `Model` | The model that holds this structure. | *required* |

Raises:

| Type         | Description                   |
| ------------ | ----------------------------- |
| `ValueError` | If one of these checks fails. |

Source code in `commol/context/aggregate.py`

```python
def check_model(self, model: "Model[Self]") -> None:
    """
    Check the structure against the process and the parameters.

    The process needs a bin and a transition, the amounts must name every
    bin and the fractions every category of the process, parameter ids
    must not be engine variables or bin ids, every variable of a rate must
    be defined, and the engine must plan every transition, resolve the
    neighbors and build the initial state.

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
    self._check_stratification_fractions(process)
    self._check_parameter_names(model)
    engine_state.transition_flows(process, self)
    engine_state.check_model(model)
    if self.compartment_amounts:
        engine_state.initial_state(process, self)
    self._check_formula_variables(model)
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

Source code in `commol/context/aggregate.py`

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

Source code in `commol/context/aggregate.py`

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

Source code in `commol/context/aggregate.py`

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

Source code in `commol/context/aggregate.py`

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
        param
        for param in parameters
        if param.parameter_type == CalibrationParameterType.INITIAL_CONDITION
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

Source code in `commol/context/aggregate.py`

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
    from commol.context._model.calibration_write_back import (
        calibrated_structure,
    )

    return calibrated_structure(model, problem, values)
```

#### report_sections

```python
report_sections(model: Model[Self], calibrated_ids: set[str], parameter_statistics: Mapping[str, ParameterSummary] | None) -> ReportSections
```

The `initial_amounts` and `category_splits` sections of a report.

See :func:`commol.applied_parameters_report`.

Parameters:

| Name                   | Type                             | Description                              | Default                                                   |
| ---------------------- | -------------------------------- | ---------------------------------------- | --------------------------------------------------------- |
| `model`                | `Model`                          | The model that holds the applied values. | *required*                                                |
| `calibrated_ids`       | `set[str]`                       | Ids that calibration set.                | *required*                                                |
| `parameter_statistics` | \`Mapping[str, ParameterSummary] | None\`                                   | Statistics of each id from a sample, or None without one. |

Returns:

| Type             | Description                                       |
| ---------------- | ------------------------------------------------- |
| `ReportSections` | {"initial_amounts": ..., "category_splits": ...}. |

Source code in `commol/context/aggregate.py`

```python
def report_sections(
    self,
    model: "Model[Self]",
    calibrated_ids: set[str],
    parameter_statistics: "Mapping[str, ParameterSummary] | None",
) -> ReportSections:
    """
    The ``initial_amounts`` and ``category_splits`` sections of a report.

    See :func:`commol.applied_parameters_report`.

    Parameters
    ----------
    model : Model
        The model that holds the applied values.
    calibrated_ids : set[str]
        Ids that calibration set.
    parameter_statistics : Mapping[str, ParameterSummary] | None
        Statistics of each id from a sample, or None without one.

    Returns
    -------
    ReportSections
        ``{"initial_amounts": ..., "category_splits": ...}``.
    """
    from commol.context._model.report import (
        category_splits_report,
        initial_amounts_report,
    )

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

Source code in `commol/context/aggregate.py`

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
    from commol.context._model.aggregate_description import describe_aggregate

    return describe_aggregate(model)
```

#### check_unit_consistency

```python
check_unit_consistency(model: Model[Self], verbose: bool) -> None
```

Check the units of the model; see `Model.check_unit_consistency`.

Source code in `commol/context/aggregate.py`

```python
def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
    """Check the units of the model; see `Model.check_unit_consistency`."""
    from commol.context._model.units import UnitChecker

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

Source code in `commol/context/aggregate.py`

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

Source code in `commol/context/aggregate.py`

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

| Type              | Description                                                                                                                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tuple[str, ...]` | Ids of the categories selecting the subgroup this stratification subdivides, empty for an unconditional stratification. A category name another stratification also declares is qualified, as in group:g1. |

Raises:

| Type         | Description                         |
| ------------ | ----------------------------------- |
| `ValueError` | If the stratification is not found. |

Source code in `commol/context/aggregate.py`

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
        subdivides, empty for an unconditional stratification. A category
        name another stratification also declares is qualified, as in
        ``group:g1``.

    Raises
    ------
    ValueError
        If the stratification is not found.
    """
    for stratification in process.stratifications:
        if stratification.id == stratification_id:
            return tuple(
                self.category_id(condition.stratification, condition.category)
                for condition in (stratification.conditions or [])
                if condition.category is not None
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

The sum is taken of the state the engine builds, overrides included, over every compartment that has each of the categories. Passing no categories gives the sum of every compartment. It adds every bin it covers, so over bins of different units it has no unit.

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

Source code in `commol/context/aggregate.py`

```python
def subgroup_total(self, process: Process, categories: Iterable[str] = ()) -> float:
    """
    Get the initial sum of the compartments of a subgroup.

    The sum is taken of the state the engine builds, overrides included,
    over every compartment that has each of the categories. Passing no
    categories gives the sum of every compartment. It adds every bin it
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
    resolved = [self.resolve_category(c) for c in categories]
    return sum(
        compartment.amount
        for compartment in self.initial_state(process)
        if all(
            compartment.categories.get(stratification) == name
            for stratification, name in resolved
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
| `type` | `Literal['complete']` | The tag of the neighbors in JSON. |

Source code in `commol/context/neighbors.py`

```python
class Complete(BaseModel):
    """
    Every unit is a neighbor of every other.

    ``neighbors_<bin>`` is the bin total and ``neighbors`` is ``total``, so
    ``k1 * neighbors_B / neighbors`` from ``A`` moves ``k1 * A * B / total``.

    Attributes
    ----------
    type : Literal["complete"]
        The tag of the neighbors in JSON.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["complete"] = "complete"
```

## Groups

Bases: `BaseModel`

Units are neighbors by the categories of one stratification.

In a compartment of category `g`, `neighbors_<bin>` is `Σ_g' w[g][g'] × (amount of the bin in g')` and `neighbors` is `Σ_g' w[g][g'] × (amount of every bin in g')`.

Attributes:

| Name             | Type                            | Description                                                |
| ---------------- | ------------------------------- | ---------------------------------------------------------- |
| `type`           | `Literal['groups']`             | The tag of the neighbors in JSON.                          |
| `stratification` | `str`                           | Id of the stratification whose categories group the units. |
| `weights`        | \`dict\[str, dict[str, float]\] | None\`                                                     |

Source code in `commol/context/neighbors.py`

```python
class Groups(BaseModel):
    """
    Units are neighbors by the categories of one stratification.

    In a compartment of category ``g``, ``neighbors_<bin>`` is
    ``Σ_g' w[g][g'] × (amount of the bin in g')`` and ``neighbors`` is
    ``Σ_g' w[g][g'] × (amount of every bin in g')``.

    Attributes
    ----------
    type : Literal["groups"]
        The tag of the neighbors in JSON.
    stratification : str
        Id of the stratification whose categories group the units.
    weights : dict[str, dict[str, float]] | None
        ``weights[g][g']``, the weight of category ``g'`` for a unit in ``g``:
        one finite number of at least 0 for every pair of categories. None
        is the identity: a unit sees only its own category.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["groups"] = "groups"
    stratification: str
    weights: dict[str, dict[str, float]] | None = None

    def __init__(
        self,
        stratification: str,
        weights: Mapping[str, Mapping[str, float]] | None = None,
        **data: object,
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
        **data : object
            The other fields, as JSON holds them.
        """
        super().__init__(
            stratification=stratification,
            weights=None
            if weights is None
            else {own: dict(row) for own, row in weights.items()},
            **data,
        )
```

### Methods:

#### __init__

```python
__init__(stratification: str, weights: Mapping[str, Mapping[str, float]] | None = None, **data: object) -> None
```

Group the neighbors of a unit by the categories of a stratification.

Parameters:

| Name             | Type                                  | Description                                                | Default                                                                          |
| ---------------- | ------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `stratification` | `str`                                 | Id of the stratification whose categories group the units. | *required*                                                                       |
| `weights`        | \`Mapping\[str, Mapping[str, float]\] | None\`                                                     | weights[g][g'], the weight of category g' for a unit in g. None is the identity. |
| `**data`         | `object`                              | The other fields, as JSON holds them.                      | `{}`                                                                             |

Source code in `commol/context/neighbors.py`

```python
def __init__(
    self,
    stratification: str,
    weights: Mapping[str, Mapping[str, float]] | None = None,
    **data: object,
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
    **data : object
        The other fields, as JSON holds them.
    """
    super().__init__(
        stratification=stratification,
        weights=None
        if weights is None
        else {own: dict(row) for own, row in weights.items()},
        **data,
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

Source code in `commol/context/_model/engine_state.py`

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

## Descriptions

`Model.describe` returns a `ModelDescription`; its blocks are in `commol.context.description`.

## ModelDescription

Bases: `BaseModel`

A description of a model, as data.

`Model.describe` builds it. `render` turns it into a string, `save` writes it to a file, and a notebook shows its Markdown rendering.

Attributes:

| Name       | Type            | Description                                         |
| ---------- | --------------- | --------------------------------------------------- |
| `title`    | `str`           | The title, the name of the model.                   |
| `layout`   | `str`           | The layout the sections follow: "default" or "odd". |
| `sections` | `list[Section]` | The sections, in order.                             |

Source code in `commol/context/description.py`

```python
class ModelDescription(BaseModel):
    """
    A description of a model, as data.

    `Model.describe` builds it. `render` turns it into a string, `save` writes
    it to a file, and a notebook shows its Markdown rendering.

    Attributes
    ----------
    title : str
        The title, the name of the model.
    layout : str
        The layout the sections follow: ``"default"`` or ``"odd"``.
    sections : list[Section]
        The sections, in order.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    layout: str = "default"
    sections: list[Section]

    def section(self, key: str) -> Section:
        """
        The section with a key.

        Parameters
        ----------
        key : str
            The key of the section, such as ``"dynamics"``.

        Returns
        -------
        Section
            The section.

        Raises
        ------
        KeyError
            If the description has no section with that key; the message
            lists the keys.
        """
        for section in self.sections:
            if section.key == key:
                return section
        raise KeyError(
            f"The description has no section '{key}'. Its sections are "
            f"{[section.key for section in self.sections]}"
        )

    def graphs(self) -> list[StateGraph]:
        """
        Every state graph of the description, in order.

        Returns
        -------
        list[StateGraph]
            The graphs, the expanded ones included.
        """
        return [
            block
            for section in self.sections
            for block in section.blocks
            if isinstance(block, StateGraph)
        ]

    def render(self, format: DescriptionFormat = "text") -> str:
        """
        The description as a string.

        Parameters
        ----------
        format : {"text", "markdown", "latex", "json", "dot"}, default="text"
            ``"text"``: plain text, with the equations in their compact and
            expanded forms. ``"markdown"``: Markdown, with the state graph of
            the model as a Mermaid diagram. ``"latex"``: a LaTeX document.
            ``"json"``: the description as data. ``"dot"``: every state graph
            in Graphviz DOT, the expanded one included.

        Returns
        -------
        str
            The rendering.

        Raises
        ------
        ValueError
            If the format is not one of these.
        """
        from commol.api.description import render

        return render(self, format)

    def save(self, path: str | Path, format: DescriptionFormat | None = None) -> Path:
        """
        Write the description to a file.

        Parameters
        ----------
        path : str | Path
            The file to write.
        format : {"text", "markdown", "latex", "json", "dot"} | None
            The format; None takes it from the suffix of ``path``: ``.txt``,
            ``.md``, ``.tex``, ``.json`` or ``.dot``.

        Returns
        -------
        Path
            The path written.

        Raises
        ------
        ValueError
            If the format is None and the suffix is not one of these, or the
            format is not valid.
        """
        from commol.api.description import save

        return save(self, Path(path), format)

    def _repr_markdown_(self) -> str:
        """The Markdown rendering, which a notebook shows."""
        return self.render("markdown")
```

### Methods:

#### section

```python
section(key: str) -> Section
```

The section with a key.

Parameters:

| Name  | Type  | Description                                 | Default    |
| ----- | ----- | ------------------------------------------- | ---------- |
| `key` | `str` | The key of the section, such as "dynamics". | *required* |

Returns:

| Type      | Description  |
| --------- | ------------ |
| `Section` | The section. |

Raises:

| Type       | Description                                                                  |
| ---------- | ---------------------------------------------------------------------------- |
| `KeyError` | If the description has no section with that key; the message lists the keys. |

Source code in `commol/context/description.py`

```python
def section(self, key: str) -> Section:
    """
    The section with a key.

    Parameters
    ----------
    key : str
        The key of the section, such as ``"dynamics"``.

    Returns
    -------
    Section
        The section.

    Raises
    ------
    KeyError
        If the description has no section with that key; the message
        lists the keys.
    """
    for section in self.sections:
        if section.key == key:
            return section
    raise KeyError(
        f"The description has no section '{key}'. Its sections are "
        f"{[section.key for section in self.sections]}"
    )
```

#### graphs

```python
graphs() -> list[StateGraph]
```

Every state graph of the description, in order.

Returns:

| Type               | Description                             |
| ------------------ | --------------------------------------- |
| `list[StateGraph]` | The graphs, the expanded ones included. |

Source code in `commol/context/description.py`

```python
def graphs(self) -> list[StateGraph]:
    """
    Every state graph of the description, in order.

    Returns
    -------
    list[StateGraph]
        The graphs, the expanded ones included.
    """
    return [
        block
        for section in self.sections
        for block in section.blocks
        if isinstance(block, StateGraph)
    ]
```

#### render

```python
render(format: DescriptionFormat = 'text') -> str
```

The description as a string.

Parameters:

| Name     | Type                                 | Description                                                                                                                                                                                                                                                                                | Default  |
| -------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| `format` | `(text, markdown, latex, json, dot)` | "text": plain text, with the equations in their compact and expanded forms. "markdown": Markdown, with the state graph of the model as a Mermaid diagram. "latex": a LaTeX document. "json": the description as data. "dot": every state graph in Graphviz DOT, the expanded one included. | `"text"` |

Returns:

| Type  | Description    |
| ----- | -------------- |
| `str` | The rendering. |

Raises:

| Type         | Description                        |
| ------------ | ---------------------------------- |
| `ValueError` | If the format is not one of these. |

Source code in `commol/context/description.py`

```python
def render(self, format: DescriptionFormat = "text") -> str:
    """
    The description as a string.

    Parameters
    ----------
    format : {"text", "markdown", "latex", "json", "dot"}, default="text"
        ``"text"``: plain text, with the equations in their compact and
        expanded forms. ``"markdown"``: Markdown, with the state graph of
        the model as a Mermaid diagram. ``"latex"``: a LaTeX document.
        ``"json"``: the description as data. ``"dot"``: every state graph
        in Graphviz DOT, the expanded one included.

    Returns
    -------
    str
        The rendering.

    Raises
    ------
    ValueError
        If the format is not one of these.
    """
    from commol.api.description import render

    return render(self, format)
```

#### save

```python
save(path: str | Path, format: DescriptionFormat | None = None) -> Path
```

Write the description to a file.

Parameters:

| Name     | Type                                 | Description                                                                        | Default            |
| -------- | ------------------------------------ | ---------------------------------------------------------------------------------- | ------------------ |
| `path`   | \`str                                | Path\`                                                                             | The file to write. |
| `format` | `(text, markdown, latex, json, dot)` | The format; None takes it from the suffix of path: .txt, .md, .tex, .json or .dot. | `"text"`           |

Returns:

| Type   | Description       |
| ------ | ----------------- |
| `Path` | The path written. |

Raises:

| Type         | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| `ValueError` | If the format is None and the suffix is not one of these, or the format is not valid. |

Source code in `commol/context/description.py`

```python
def save(self, path: str | Path, format: DescriptionFormat | None = None) -> Path:
    """
    Write the description to a file.

    Parameters
    ----------
    path : str | Path
        The file to write.
    format : {"text", "markdown", "latex", "json", "dot"} | None
        The format; None takes it from the suffix of ``path``: ``.txt``,
        ``.md``, ``.tex``, ``.json`` or ``.dot``.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If the format is None and the suffix is not one of these, or the
        format is not valid.
    """
    from commol.api.description import save

    return save(self, Path(path), format)
```

## Section

Bases: `BaseModel`

One section of a description.

Attributes:

| Name     | Type          | Description                                                                    |
| -------- | ------------- | ------------------------------------------------------------------------------ |
| `key`    | `str`         | The key of the section, such as "dynamics".                                    |
| `title`  | `str`         | The title a rendering shows.                                                   |
| `blocks` | `list[Block]` | The blocks, in order. A section with no blocks is left out of every rendering. |

Source code in `commol/context/description.py`

```python
class Section(BaseModel):
    """
    One section of a description.

    Attributes
    ----------
    key : str
        The key of the section, such as ``"dynamics"``.
    title : str
        The title a rendering shows.
    blocks : list[Block]
        The blocks, in order. A section with no blocks is left out of every
        rendering.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    blocks: list[Block] = Field(default_factory=list)
```

## Text

Bases: `BaseModel`

A paragraph.

Attributes:

| Name      | Type              | Description                   |
| --------- | ----------------- | ----------------------------- |
| `type`    | `Literal['text']` | The tag of the block.         |
| `heading` | \`str             | None\`                        |
| `text`    | `str`             | The paragraph, in plain text. |

Source code in `commol/context/description.py`

```python
class Text(BaseModel):
    """
    A paragraph.

    Attributes
    ----------
    type : Literal["text"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    text : str
        The paragraph, in plain text.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    heading: str | None = None
    text: str
```

## Table

Bases: `BaseModel`

A table: named columns and rows of cells.

Attributes:

| Name      | Type               | Description                              |
| --------- | ------------------ | ---------------------------------------- |
| `type`    | `Literal['table']` | The tag of the block.                    |
| `heading` | \`str              | None\`                                   |
| `columns` | `list[str]`        | The column names.                        |
| `rows`    | `list[list[Cell]]` | The rows, each with one cell per column. |

Source code in `commol/context/description.py`

```python
class Table(BaseModel):
    """
    A table: named columns and rows of cells.

    Attributes
    ----------
    type : Literal["table"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    columns : list[str]
        The column names.
    rows : list[list[Cell]]
        The rows, each with one cell per column.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["table"] = "table"
    heading: str | None = None
    columns: list[str]
    rows: list[list[Cell]]

    @model_validator(mode="after")
    def check_row_lengths(self) -> Self:
        """Each row has one cell per column."""
        for index, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Row {index} of the table has {len(row)} cells; the table "
                    f"has {len(self.columns)} columns: {self.columns}"
                )
        return self
```

### Methods:

#### check_row_lengths

```python
check_row_lengths() -> Self
```

Each row has one cell per column.

Source code in `commol/context/description.py`

```python
@model_validator(mode="after")
def check_row_lengths(self) -> Self:
    """Each row has one cell per column."""
    for index, row in enumerate(self.rows):
        if len(row) != len(self.columns):
            raise ValueError(
                f"Row {index} of the table has {len(row)} cells; the table "
                f"has {len(self.columns)} columns: {self.columns}"
            )
    return self
```

## Equations

Bases: `BaseModel`

A system of equations or of flows.

Attributes:

| Name      | Type                   | Description                        |
| --------- | ---------------------- | ---------------------------------- |
| `type`    | `Literal['equations']` | The tag of the block.              |
| `heading` | \`str                  | None\`                             |
| `title`   | \`str                  | None\`                             |
| `groups`  | `list[EquationGroup]`  | The groups of equations, in order. |

Source code in `commol/context/description.py`

```python
class Equations(BaseModel):
    """
    A system of equations or of flows.

    Attributes
    ----------
    type : Literal["equations"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    title : str | None
        A line that introduces the groups, such as ``Bin Transitions``.
    groups : list[EquationGroup]
        The groups of equations, in order.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["equations"] = "equations"
    heading: str | None = None
    title: str | None = None
    groups: list[EquationGroup]
```

## EquationGroup

Bases: `BaseModel`

Equations under one heading, such as the flows of one transition.

Attributes:

| Name        | Type             | Description                                                                         |
| ----------- | ---------------- | ----------------------------------------------------------------------------------- |
| `heading`   | \`str            | None\`                                                                              |
| `equations` | `list[Equation]` | The equations, in order. A group with a heading and no equations renders as (none). |

Source code in `commol/context/description.py`

```python
class EquationGroup(BaseModel):
    """
    Equations under one heading, such as the flows of one transition.

    Attributes
    ----------
    heading : str | None
        The heading of the group, or None for equations with no heading.
    equations : list[Equation]
        The equations, in order. A group with a heading and no equations
        renders as ``(none)``.
    """

    model_config = ConfigDict(extra="forbid")

    heading: str | None = None
    equations: list[Equation]
```

## Equation

Bases: `BaseModel`

One equation, or one flow, in plain text and in LaTeX.

Attributes:

| Name          | Type                        | Description                                                                                                                               |
| ------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `left`        | `str`                       | The left side in plain text, such as dA/dt or A -> B.                                                                                     |
| `right`       | `str`                       | The right side in plain text.                                                                                                             |
| `left_latex`  | `str`                       | The left side in LaTeX, without math delimiters.                                                                                          |
| `right_latex` | `str`                       | The right side in LaTeX, without math delimiters.                                                                                         |
| `relation`    | `Literal['equals', 'flow']` | "equals": the left side equals the right side. "flow": the left side names where an amount moves (A -> B) and the right side is the flow. |

Source code in `commol/context/description.py`

```python
class Equation(BaseModel):
    """
    One equation, or one flow, in plain text and in LaTeX.

    Attributes
    ----------
    left : str
        The left side in plain text, such as ``dA/dt`` or ``A -> B``.
    right : str
        The right side in plain text.
    left_latex : str
        The left side in LaTeX, without math delimiters.
    right_latex : str
        The right side in LaTeX, without math delimiters.
    relation : Literal["equals", "flow"]
        ``"equals"``: the left side equals the right side. ``"flow"``: the
        left side names where an amount moves (``A -> B``) and the right side
        is the flow.
    """

    model_config = ConfigDict(extra="forbid")

    left: str
    right: str
    left_latex: str
    right_latex: str
    relation: Literal["equals", "flow"] = "equals"
```

## StateGraph

Bases: `BaseModel`

The states of a model and the moves between them.

Attributes:

| Name       | Type                     | Description                                                                                                                                                                    |
| ---------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `type`     | `Literal['state_graph']` | The tag of the block.                                                                                                                                                          |
| `heading`  | \`str                    | None\`                                                                                                                                                                         |
| `name`     | `str`                    | The name of the graph, such as states; a DOT file names it so.                                                                                                                 |
| `nodes`    | `list[GraphNode]`        | The states.                                                                                                                                                                    |
| `edges`    | `list[GraphEdge]`        | The moves between them.                                                                                                                                                        |
| `expanded` | `bool`                   | Whether it is the graph of the expanded compartments. The Markdown rendering draws only a graph that is not expanded: an expanded graph can pass what a Mermaid diagram draws. |

Source code in `commol/context/description.py`

```python
class StateGraph(BaseModel):
    """
    The states of a model and the moves between them.

    Attributes
    ----------
    type : Literal["state_graph"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    name : str
        The name of the graph, such as ``states``; a DOT file names it so.
    nodes : list[GraphNode]
        The states.
    edges : list[GraphEdge]
        The moves between them.
    expanded : bool
        Whether it is the graph of the expanded compartments. The Markdown
        rendering draws only a graph that is not expanded: an expanded graph
        can pass what a Mermaid diagram draws.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["state_graph"] = "state_graph"
    heading: str | None = None
    name: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    expanded: bool = False
```

## GraphNode

Bases: `BaseModel`

A node of a `StateGraph`.

Attributes:

| Name    | Type  | Description                              |
| ------- | ----- | ---------------------------------------- |
| `id`    | `str` | The id of the node, unique in its graph. |
| `label` | `str` | The label a diagram shows.               |
| `group` | \`str | None\`                                   |

Source code in `commol/context/description.py`

```python
class GraphNode(BaseModel):
    """
    A node of a `StateGraph`.

    Attributes
    ----------
    id : str
        The id of the node, unique in its graph.
    label : str
        The label a diagram shows.
    group : str | None
        What the node is, such as ``"bin"`` or a stratification id.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    group: str | None = None
```

## GraphEdge

Bases: `BaseModel`

An edge of a `StateGraph`: an amount that moves from a node to another.

Attributes:

| Name     | Type  | Description                                  |
| -------- | ----- | -------------------------------------------- |
| `source` | \`str | None\`                                       |
| `target` | \`str | None\`                                       |
| `label`  | `str` | The label a diagram shows, such as the rate. |

Source code in `commol/context/description.py`

```python
class GraphEdge(BaseModel):
    """
    An edge of a `StateGraph`: an amount that moves from a node to another.

    Attributes
    ----------
    source : str | None
        The id of the node the amount leaves, or None for an inflow.
    target : str | None
        The id of the node the amount enters, or None for an outflow.
    label : str
        The label a diagram shows, such as the rate.
    """

    model_config = ConfigDict(extra="forbid")

    source: str | None
    target: str | None
    label: str
```

## Matrix

Bases: `BaseModel`

A matrix with labelled rows and columns, such as the weights of groups.

Attributes:

| Name            | Type                | Description                                                  |
| --------------- | ------------------- | ------------------------------------------------------------ |
| `type`          | `Literal['matrix']` | The tag of the block.                                        |
| `heading`       | \`str               | None\`                                                       |
| `row_labels`    | `list[str]`         | The label of each row.                                       |
| `column_labels` | `list[str]`         | The label of each column.                                    |
| `values`        | `list[list[float]]` | One row of values per row label, one value per column label. |

Source code in `commol/context/description.py`

```python
class Matrix(BaseModel):
    """
    A matrix with labelled rows and columns, such as the weights of groups.

    Attributes
    ----------
    type : Literal["matrix"]
        The tag of the block.
    heading : str | None
        A heading above the block.
    row_labels : list[str]
        The label of each row.
    column_labels : list[str]
        The label of each column.
    values : list[list[float]]
        One row of values per row label, one value per column label.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["matrix"] = "matrix"
    heading: str | None = None
    row_labels: list[str]
    column_labels: list[str]
    values: list[list[float]]

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        """One row of values per row label, one value per column label."""
        shape = [len(row) for row in self.values]
        if shape != [len(self.column_labels)] * len(self.row_labels):
            raise ValueError(
                f"A matrix of {len(self.row_labels)} rows and "
                f"{len(self.column_labels)} columns has rows of {shape} values"
            )
        return self
```

### Methods:

#### check_shape

```python
check_shape() -> Self
```

One row of values per row label, one value per column label.

Source code in `commol/context/description.py`

```python
@model_validator(mode="after")
def check_shape(self) -> Self:
    """One row of values per row label, one value per column label."""
    shape = [len(row) for row in self.values]
    if shape != [len(self.column_labels)] * len(self.row_labels):
        raise ValueError(
            f"A matrix of {len(self.row_labels)} rows and "
            f"{len(self.column_labels)} columns has rows of {shape} values"
        )
    return self
```

## UnitConsistencyError

Bases: `Exception`

Raised when unit consistency check fails.

Source code in `commol/utils/equations.py`

```python
class UnitConsistencyError(Exception):
    """Raised when unit consistency check fails."""

    pass
```
