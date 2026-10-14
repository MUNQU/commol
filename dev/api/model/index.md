# Model API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The validated model definition that `ModelBuilder.build` returns, the structure of its kind, and the entries of `CompartmentalStructure.initial_state()`. Guide: [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md), [Units and Equations](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md). The tree of types is in [Model Structure](https://munqu.github.io/commol/latest/advanced/model-structure/index.md).

## Model

Bases: `BaseModel`, `Generic[StructureT]`

A model of any kind: a header, an engine and the structure of its kind.

`Model[CompartmentalStructure]` is a compartmental model, the model `ModelBuilder` builds. A model loaded without a structure class reads the class from the `kind` of its structure.

Attributes:

| Name          | Type              | Description                                                                                                                 |
| ------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `name`        | `str`             | A unique name that identifies the model.                                                                                    |
| `description` | \`str             | None\`                                                                                                                      |
| `version`     | \`str             | None\`                                                                                                                      |
| `parameters`  | `list[Parameter]` | A list of global model parameters.                                                                                          |
| `engine`      | `ModelEngine`     | The engine that runs the model, and its settings.                                                                           |
| `structure`   | `StructureT`      | The part of the model that depends on its kind, such as the bins, stratifications and transitions of a compartmental model. |
| `step_unit`   | \`str             | None\`                                                                                                                      |

Source code in `commol/context/model.py`

```python
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
```

### Methods:

#### read_structure_kind

```python
read_structure_kind(value: object) -> object
```

Read a structure given as a mapping as the class of its `kind`.

Source code in `commol/context/model.py`

```python
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
```

#### from_json

```python
from_json(file_path: str | Path) -> Self
```

Loads a model from a JSON file.

The method reads the specified JSON file, parses its content, and validates it against the Model schema.

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
apply_calibration_parameters(result: CalibrationResult | ProbabilisticCalibrationResult | Mapping[str, float], problem: CalibrationProblem) -> None
```

Write calibrated values back onto the model.

Each value is routed by the `parameter_type` declared for it in `problem`: `parameter` values update model parameters, `factor` values are ignored because they apply to observations rather than to the model, and the structure writes back the values of its kind (`ModelStructure.with_calibrated_values`). A compartmental structure writes `initial_condition` values so that the model rebuilds the initial population the engine scored for them.

Parameters:

| Name      | Type                 | Description                                                        | Default    |
| --------- | -------------------- | ------------------------------------------------------------------ | ---------- |
| `result`  | \`CalibrationResult  | ProbabilisticCalibrationResult                                     | Mapping\`  |
| `problem` | `CalibrationProblem` | The problem the result came from, used to resolve parameter types. | *required* |

Raises:

| Type         | Description                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------- |
| `ValueError` | If an id is not declared in problem, or the structure cannot write back a value of its kind. |

Source code in `commol/context/model.py`

```python
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
```

#### print_equations

```python
print_equations(output_file: str | None = None, format: str = TEXT) -> None
```

Prints the equations of the model in mathematical form.

The structure prints them (`ModelStructure.print_equations`). A compartmental model displays its metadata and the system of equations in both compact (mathematical notation) and expanded (individual equations) forms.

For DifferentialEquations models, displays equations as dX/dt = ... For DifferenceEquations models, displays equations as [X(t+Dt) - X(t)] / Dt = ...

Parameters:

| Name          | Type  | Description                                                                                                                      | Default                                                                                                         |
| ------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str | None\`                                                                                                                           | If provided, writes the equations to this file path instead of printing to console. If None, prints to console. |
| `format`      | `str` | Output format for equations. Must be one of: - "text": Plain text format (default) - "latex": LaTeX mathematical notation format | `"text"`                                                                                                        |

Raises:

| Type                  | Description                                         |
| --------------------- | --------------------------------------------------- |
| `ValueError`          | If format is not "text" or "latex"                  |
| `NotImplementedError` | If the kind of the model has no equations to print. |

Examples:

```pycon
>>> model.print_equations()  # Print to console in text format
>>> model.print_equations(output_file="equations.txt")  # Save text format
>>> model.print_equations(format="latex")  # Print LaTeX to console
>>> model.print_equations(
...     output_file="equations.txt", format="latex"
... )  # Save LaTeX
```

Source code in `commol/context/model.py`

```python
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
```

#### check_unit_consistency

```python
check_unit_consistency(verbose: bool = False) -> None
```

Check unit consistency of all equations in the model.

The structure checks them (`ModelStructure.check_unit_consistency`). For a compartmental model, the engine applies every flow once per step and converts no value between units. So the check infers one step unit from the parameter units: the one unit of time (`day`, `week`, `step`, …) that they use, or `step` when they use none. Every flow must then be exactly `<bin unit>/<step unit>`, such as `person/day`.

Parameters:

| Name      | Type   | Description                                                      | Default |
| --------- | ------ | ---------------------------------------------------------------- | ------- |
| `verbose` | `bool` | If True, prints a success message when all units are consistent. | `False` |

Raises:

| Type                   | Description                                                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `UnitConsistencyError` | If a constant parameter or a bin has no unit, if the parameters use more than one unit of time (1/day and 1/week), or if a flow does not have the unit <bin unit>/<step unit>. |
| `ValueError`           | If the model type doesn't support unit checking.                                                                                                                               |
| `NotImplementedError`  | If the kind of the model has no unit check.                                                                                                                                    |

Notes

- Predefined variables (N, N_young, etc.) have the bin unit
- Time step variables (t, step) are dimensionless
- Mathematical constants (pi, e) are dimensionless

Source code in `commol/context/model.py`

```python
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
```

## ModelStructure

Bases: `BaseModel`, `ABC`

The part of a model that depends on its kind.

Each kind subclasses it and sets `kind` to a `Literal` default, its name in the engine's list of kinds. A subclass that does not implement an abstract method cannot be made: the error names the method.

Attributes:

| Name   | Type  | Description                                               |
| ------ | ----- | --------------------------------------------------------- |
| `kind` | `str` | Name of the model kind, the tag of the structure in JSON. |

Source code in `commol/context/structure.py`

```python
class ModelStructure(BaseModel, ABC):
    """
    The part of a model that depends on its kind.

    Each kind subclasses it and sets ``kind`` to a ``Literal`` default, its
    name in the engine's list of kinds. A subclass that does not implement an
    abstract method cannot be made: the error names the method.

    Attributes
    ----------
    kind : str
        Name of the model kind, the tag of the structure in JSON.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str

    _kinds: ClassVar[dict[str, type["ModelStructure"]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Register a concrete subclass under its ``kind``."""
        super().__pydantic_init_subclass__(**kwargs)
        kind = cls.model_fields["kind"].default
        if inspect.isabstract(cls) or not isinstance(kind, str):
            return
        known = ModelStructure._kinds.get(kind)
        if known is not None and known.__qualname__ != cls.__qualname__:
            raise TypeError(
                f"Model kind '{kind}' is already the kind of {known.__qualname__}"
            )
        ModelStructure._kinds[kind] = cls

    @classmethod
    def of_kind(cls, kind: str) -> type["ModelStructure"]:
        """
        The structure class of a model kind.

        Parameters
        ----------
        kind : str
            Name of the kind, such as ``"compartmental"``.

        Returns
        -------
        type[ModelStructure]
            The subclass registered for it.

        Raises
        ------
        ValueError
            If no structure class has that kind.
        """
        structure = cls._kinds.get(kind)
        if structure is None:
            raise ValueError(
                f"Unknown model kind '{kind}'. Known kinds: {sorted(cls._kinds)}"
            )
        return structure

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
    def outputs_by_source(self) -> dict[str, list[str]]:
        """
        The output names of the model, grouped by the id they come from.

        Reports and plots name a group of outputs by this id and sum them.

        Returns
        -------
        dict[str, list[str]]
            Each id mapped to its output names, in engine order.
        """

    def series_label(self, series_id: str) -> SeriesLabel:
        """
        How a plot names a series and its unit.

        Parameters
        ----------
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
            For example ``{"initial_condition"}`` for a compartmental model.
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
        parameter_statistics: "Mapping[str, ParameterSetStatistics] | None",  # noqa: ARG002
    ) -> ReportSections:
        """
        Sections of this kind in a report of applied values.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, ParameterSetStatistics] | None
            Ensemble spread per id, or None without an ensemble.

        Returns
        -------
        ReportSections
            Each section by its key in the report; none by default.
        """
        return {}

    def print_equations(
        self, model: "Model[Self]", output_file: str | None, format: str
    ) -> None:
        """
        Print the equations of the model.

        Raises
        ------
        NotImplementedError
            If the kind has no equations to print.
        """
        raise NotImplementedError(f"A {self.kind} model has no equations to print")

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

Register a concrete subclass under its `kind`.

Source code in `commol/context/structure.py`

```python
@classmethod
def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
    """Register a concrete subclass under its ``kind``."""
    super().__pydantic_init_subclass__(**kwargs)
    kind = cls.model_fields["kind"].default
    if inspect.isabstract(cls) or not isinstance(kind, str):
        return
    known = ModelStructure._kinds.get(kind)
    if known is not None and known.__qualname__ != cls.__qualname__:
        raise TypeError(
            f"Model kind '{kind}' is already the kind of {known.__qualname__}"
        )
    ModelStructure._kinds[kind] = cls
```

#### of_kind

```python
of_kind(kind: str) -> type[ModelStructure]
```

The structure class of a model kind.

Parameters:

| Name   | Type  | Description                                | Default    |
| ------ | ----- | ------------------------------------------ | ---------- |
| `kind` | `str` | Name of the kind, such as "compartmental". | *required* |

Returns:

| Type                   | Description                     |
| ---------------------- | ------------------------------- |
| `type[ModelStructure]` | The subclass registered for it. |

Raises:

| Type         | Description                          |
| ------------ | ------------------------------------ |
| `ValueError` | If no structure class has that kind. |

Source code in `commol/context/structure.py`

```python
@classmethod
def of_kind(cls, kind: str) -> type["ModelStructure"]:
    """
    The structure class of a model kind.

    Parameters
    ----------
    kind : str
        Name of the kind, such as ``"compartmental"``.

    Returns
    -------
    type[ModelStructure]
        The subclass registered for it.

    Raises
    ------
    ValueError
        If no structure class has that kind.
    """
    structure = cls._kinds.get(kind)
    if structure is None:
        raise ValueError(
            f"Unknown model kind '{kind}'. Known kinds: {sorted(cls._kinds)}"
        )
    return structure
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
outputs_by_source() -> dict[str, list[str]]
```

The output names of the model, grouped by the id they come from.

Reports and plots name a group of outputs by this id and sum them.

Returns:

| Type                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `dict[str, list[str]]` | Each id mapped to its output names, in engine order. |

Source code in `commol/context/structure.py`

```python
@abstractmethod
def outputs_by_source(self) -> dict[str, list[str]]:
    """
    The output names of the model, grouped by the id they come from.

    Reports and plots name a group of outputs by this id and sum them.

    Returns
    -------
    dict[str, list[str]]
        Each id mapped to its output names, in engine order.
    """
```

#### series_label

```python
series_label(series_id: str) -> SeriesLabel
```

How a plot names a series and its unit.

Parameters:

| Name        | Type  | Description                                   | Default    |
| ----------- | ----- | --------------------------------------------- | ---------- |
| `series_id` | `str` | An output name or an id of outputs_by_source. | *required* |

Returns:

| Type          | Description                                              |
| ------------- | -------------------------------------------------------- |
| `SeriesLabel` | The id itself and no unit, unless the kind knows better. |

Source code in `commol/context/structure.py`

```python
def series_label(self, series_id: str) -> SeriesLabel:
    """
    How a plot names a series and its unit.

    Parameters
    ----------
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

| Type                                  | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| `frozenset[CalibrationParameterType]` | For example {"initial_condition"} for a compartmental model. |

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
        For example ``{"initial_condition"}`` for a compartmental model.
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
report_sections(model: Model[Self], calibrated_ids: set[str], parameter_statistics: Mapping[str, ParameterSetStatistics] | None) -> ReportSections
```

Sections of this kind in a report of applied values.

Parameters:

| Name                   | Type                                   | Description                              | Default                                              |
| ---------------------- | -------------------------------------- | ---------------------------------------- | ---------------------------------------------------- |
| `model`                | `Model`                                | The model that holds the applied values. | *required*                                           |
| `calibrated_ids`       | `set[str]`                             | Ids that calibration set.                | *required*                                           |
| `parameter_statistics` | \`Mapping[str, ParameterSetStatistics] | None\`                                   | Ensemble spread per id, or None without an ensemble. |

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
    parameter_statistics: "Mapping[str, ParameterSetStatistics] | None",  # noqa: ARG002
) -> ReportSections:
    """
    Sections of this kind in a report of applied values.

    Parameters
    ----------
    model : Model
        The model that holds the applied values.
    calibrated_ids : set[str]
        Ids that calibration set.
    parameter_statistics : Mapping[str, ParameterSetStatistics] | None
        Ensemble spread per id, or None without an ensemble.

    Returns
    -------
    ReportSections
        Each section by its key in the report; none by default.
    """
    return {}
```

#### print_equations

```python
print_equations(model: Model[Self], output_file: str | None, format: str) -> None
```

Print the equations of the model.

Raises:

| Type                  | Description                            |
| --------------------- | -------------------------------------- |
| `NotImplementedError` | If the kind has no equations to print. |

Source code in `commol/context/structure.py`

```python
def print_equations(
    self, model: "Model[Self]", output_file: str | None, format: str
) -> None:
    """
    Print the equations of the model.

    Raises
    ------
    NotImplementedError
        If the kind has no equations to print.
    """
    raise NotImplementedError(f"A {self.kind} model has no equations to print")
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

## CompartmentalStructure

Bases: `ModelStructure`

Bins, stratifications, transitions and initial conditions of a model.

Attributes:

| Name                 | Type                       | Description                                                                 |
| -------------------- | -------------------------- | --------------------------------------------------------------------------- |
| `kind`               | `Literal['compartmental']` | The model kind.                                                             |
| `bins`               | `list[Bin]`                | The states of the model. Stratifications expand each one into compartments. |
| `accumulators`       | `list[Accumulator]`        | Counters of the flow of the transitions that name them.                     |
| `stratifications`    | `list[Stratification]`     | Categorical subdivisions of the bins, in expansion order.                   |
| `transitions`        | `list[Transition]`         | Flows between bins, or between the categories of one stratification.        |
| `initial_conditions` | `InitialConditions`        | Population size and the fractions the compartments start with.              |

Source code in `commol/context/compartmental.py`

```python
class CompartmentalStructure(ModelStructure):
    """
    Bins, stratifications, transitions and initial conditions of a model.

    Attributes
    ----------
    kind : Literal["compartmental"]
        The model kind.
    bins : list[Bin]
        The states of the model. Stratifications expand each one into
        compartments.
    accumulators : list[Accumulator]
        Counters of the flow of the transitions that name them.
    stratifications : list[Stratification]
        Categorical subdivisions of the bins, in expansion order.
    transitions : list[Transition]
        Flows between bins, or between the categories of one stratification.
    initial_conditions : InitialConditions
        Population size and the fractions the compartments start with.
    """

    kind: Literal["compartmental"] = "compartmental"

    bins: list[Bin]
    accumulators: list[Accumulator] = Field(default_factory=list)
    stratifications: list[Stratification]
    transitions: list[Transition]
    initial_conditions: InitialConditions

    @field_validator("bins")
    @classmethod
    def validate_bins_not_empty(cls, v: list[Bin]) -> list[Bin]:
        """Require at least one bin."""
        if not v:
            raise ValueError("At least one bin must be defined.")
        return v

    @field_validator("transitions")
    @classmethod
    def validate_transitions_not_empty(cls, v: list[Transition]) -> list[Transition]:
        """Require at least one transition."""
        if not v:
            raise ValueError("At least one transition must be defined.")
        return v

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
    def validate_bin_initial_conditions(self) -> Self:
        """Validate the bin fractions against the bins."""
        bin_fractions = {
            bf.bin: bf.fraction for bf in self.initial_conditions.bin_fractions
        }
        actual_bins = set(bin_fractions)
        expected_bins = {bin_item.id for bin_item in self.bins}
        if actual_bins != expected_bins:
            raise ValueError(
                f"Initial bin fractions keys must exactly match bin ids. "
                f"Missing ids: {expected_bins - actual_bins}, "
                f"Extra ids: {actual_bins - expected_bins}."
            )

        # The sum is checked once every fraction has a value; InitialConditions
        # checks the uncalibrated case.
        calibrated = [f for f in bin_fractions.values() if f is not None]
        if len(calibrated) == len(bin_fractions):
            total = sum(calibrated)
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(f"Bin fractions must sum to 1.0, but got {total:.7f}.")
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
    def validate_stratification_initial_conditions(self) -> Self:
        """Validate the stratification fractions against the stratifications."""
        strat_map = {strat.id: strat for strat in self.stratifications}
        actual_strat = {
            sf.stratification for sf in self.initial_conditions.stratification_fractions
        }
        expected_strat = set(strat_map)
        if actual_strat != expected_strat:
            raise ValueError(
                f"Initial stratification fractions keys must exactly match "
                f"stratification ids. Missing ids: {expected_strat - actual_strat}, "
                f"Extra ids: {actual_strat - expected_strat}."
            )

        for strat_fractions in self.initial_conditions.stratification_fractions:
            strat_id = strat_fractions.stratification
            strat_instance = strat_map[strat_id]
            fractions = {sf.category: sf.fraction for sf in strat_fractions.fractions}

            expected = set(strat_instance.categories)
            actual = set(fractions)
            if actual != expected:
                raise ValueError(
                    f"Categories for stratification '{strat_id}' must exactly "
                    f"match defined categories in instance '{strat_instance.id}'. "
                    f"Missing categories: {expected - actual}, "
                    f"Extra categories: {actual - expected}."
                )

            total = sum(fractions.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"Stratification fractions for '{strat_id}' must sum to 1.0, "
                    f"but got {total:.7}."
                )
        return self

    # ------------------------------------------------------------------
    # The interface of a model kind
    # ------------------------------------------------------------------

    def check_model(self, model: "Model[Self]") -> None:
        """
        Check the structure against the parameters of its model.

        Parameter ids must not be engine variables or bin ids, every variable
        of a rate must be defined, transitions must move between bins or
        between the categories of one stratification, and the engine must
        plan every transition and build the initial state.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Raises
        ------
        ValueError
            If one of these checks fails.
        """
        self._check_parameter_names(model)
        engine_state.transition_flows(self)
        engine_state.check_names(model)
        if self.initial_conditions.compartment_fractions:
            engine_state.initial_state(self)
        self._check_formula_variables(model)
        self._check_transition_ids()

    def outputs_by_source(self) -> dict[str, list[str]]:
        """
        Map each bin and accumulator id to the output names it expands into.

        A bin or accumulator produces one output per combination of the
        stratification categories that apply to it. Without stratifications it
        produces a single output named after the id itself.

        Returns
        -------
        dict[str, list[str]]
            Bin and accumulator ids, each mapped to its output names in
            declaration order.
        """
        outputs: dict[str, list[str]] = {
            source_id: []
            for source_id in (
                *(bin_item.id for bin_item in self.bins),
                *(accumulator.id for accumulator in self.accumulators),
            )
        }
        compartments, accumulators = engine_state.expanded_outputs(self)
        for output in (*compartments, *accumulators):
            outputs[output.source].append(output.name)
        return outputs

    def series_label(self, series_id: str) -> SeriesLabel:
        """
        The name and unit of a bin, for a plot of its series.

        Parameters
        ----------
        series_id : str
            A bin id, or any other series id.

        Returns
        -------
        SeriesLabel
            The bin's name and unit, or the id and no unit when no bin has it.
        """
        bin_item = next((b for b in self.bins if b.id == series_id), None)
        if bin_item is None:
            return SeriesLabel(series_id, None)
        return SeriesLabel(bin_item.name or series_id, bin_item.unit)

    def uncalibrated_values(self) -> list[str]:
        """
        Ids of the bins whose initial fraction needs calibration.

        Returns
        -------
        list[str]
            Bin ids with a None fraction.
        """
        return self.initial_conditions.get_uncalibrated_bins()

    def with_calibrated_values(
        self,
        model: "Model[Self]",
        problem: "CalibrationProblem",
        values: Mapping[str, float],
    ) -> Self:
        """
        The structure with calibrated initial conditions written back.

        The written structure rebuilds the initial population the engine
        scored. Bin and category values update the fractions they declare.
        Every compartment those fractions do not reproduce, such as an exact
        compartment value or a value that overlaps a per-compartment override,
        is written as a per-compartment override.

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
        CompartmentalStructure
            A copy that holds the calibrated initial conditions.

        Raises
        ------
        ValueError
            If the engine cannot resolve an initial-condition id or the
            remainder, or the initial population exceeds the population size.
        """
        from commol.context._model.calibration_write_back import (
            calibrated_initial_conditions,
        )

        initial_conditions = calibrated_initial_conditions(model, problem, values)
        return self.model_copy(update={"initial_conditions": initial_conditions})

    def report_sections(
        self,
        model: "Model[Self]",  # noqa: ARG002
        calibrated_ids: set[str],
        parameter_statistics: "Mapping[str, ParameterSetStatistics] | None",
    ) -> ReportSections:
        """
        The ``initial_conditions`` section of a report of applied values.

        See :func:`commol.applied_parameters_report`.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, ParameterSetStatistics] | None
            Ensemble spread per id, or None without an ensemble.

        Returns
        -------
        ReportSections
            ``{"initial_conditions": ...}``.
        """
        from commol.context._model.report import initial_conditions_report

        return {
            "initial_conditions": initial_conditions_report(
                self, calibrated_ids, parameter_statistics
            )
        }

    def print_equations(
        self, model: "Model[Self]", output_file: str | None, format: str
    ) -> None:
        """Print the equations of the model; see `Model.print_equations`."""
        from commol.context._model.equations import EquationPrinter

        EquationPrinter(model).print_equations(output_file, format)

    def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
        """Check the units of the model; see `Model.check_unit_consistency`."""
        from commol.context._model.units import UnitChecker

        UnitChecker(model).check_unit_consistency(verbose)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def update_initial_conditions(
        self, bin_fractions: Mapping[str, float | None]
    ) -> None:
        """
        Update initial condition fractions for specified bins.

        Parameters
        ----------
        bin_fractions : Mapping[str, float | None]
            Dictionary mapping bin IDs to their new fraction values.

        Raises
        ------
        ValueError
            If a bin ID in the dictionary doesn't exist in the model.
        """
        self.initial_conditions.update_bin_fractions(bin_fractions)

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update initial stratification fractions for specified categories.

        Within each stratification, at most one category may be omitted; the
        omitted category receives the remaining fraction.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping category names to their new fraction values.

        Raises
        ------
        ValueError
            If a category doesn't exist in the model, a value lies outside
            [0.0, 1.0], a stratification has more than one category omitted, or
            the resulting fractions do not sum to 1.0.
        """
        self.initial_conditions.update_stratification_fractions(fractions)

    def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment initial condition overrides.

        A compartment already declared has its fraction replaced; a new one is
        appended. The engine validates the result before it is stored, so a
        rejected call leaves the structure unchanged.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new fraction
            of the total population.

        Raises
        ------
        ValueError
            If a name is not an expanded compartment of the model, a value lies
            outside ``[0.0, 1.0]``, or the resulting overrides make the initial
            population exceed ``population_size``.
        """
        candidate = self.initial_conditions.model_copy(deep=True)
        candidate.update_compartment_fractions(fractions)
        engine_state.initial_state(
            self.model_copy(update={"initial_conditions": candidate})
        )
        self.initial_conditions = candidate

    def initial_state(self) -> list[InitialCompartment]:
        """
        Expanded compartments and the population each one starts with.

        The engine builds this state from the bin fractions, the stratification
        fractions and the per-compartment overrides; it is the population a
        simulation starts from.

        Returns
        -------
        list[InitialCompartment]
            One entry per expanded compartment, in engine order.

        Raises
        ------
        ValueError
            If the per-compartment overrides are not valid for the structure.
        """
        return engine_state.initial_state(self)

    def get_conditioning_categories(self, stratification_id: str) -> tuple[str, ...]:
        """
        Get the categories a stratification is conditioned on.

        An unconditional stratification subdivides the whole population and has
        no conditioning categories. A conditional one subdivides only the
        subgroup its conditions select, and its fractions are relative to that
        subgroup rather than to the whole population.

        Parameters
        ----------
        stratification_id : str
            Id of a stratification of the model.

        Returns
        -------
        tuple[str, ...]
            Ids of the categories selecting the subgroup this stratification
            subdivides, empty for an unconditional stratification. A category
            name another stratification also declares is qualified, as in
            ``sex:female``.

        Raises
        ------
        ValueError
            If the stratification is not found.
        """
        for stratification in self.stratifications:
            if stratification.id == stratification_id:
                return tuple(
                    self.initial_conditions.category_id(
                        condition.stratification, condition.category
                    )
                    for condition in (stratification.conditions or [])
                    if condition.category is not None
                )
        raise ValueError(
            f"Stratification '{stratification_id}' not found. Available: "
            f"{sorted(s.id for s in self.stratifications)}"
        )

    def subgroup_population(self, categories: Iterable[str] = ()) -> float:
        """
        Get the initial head count of a subgroup.

        See
        :meth:`~commol.context.initial_conditions.InitialConditions.subgroup_population`.

        Per-compartment overrides are folded into the sum for every compartment
        the category chain selects, so the head count matches the population
        the engine builds. Without overrides the declared fractions are
        multiplied.

        Parameters
        ----------
        categories : Iterable[str], optional
            Category ids, bare or qualified (``sex:female``), forming a chain
            from the whole population down to the subgroup, outermost first.

        Returns
        -------
        float
            Number of individuals in the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found or a bare name is ambiguous.
        """
        selected = tuple(categories)
        initial_conditions = self.initial_conditions
        if not initial_conditions.compartment_fractions:
            return initial_conditions.subgroup_population(selected)
        resolved = [initial_conditions.resolve_category(c) for c in selected]
        return sum(
            compartment.population
            for compartment in self.initial_state()
            if all(
                compartment.categories.get(stratification) == name
                for stratification, name in resolved
            )
        )

    # ------------------------------------------------------------------
    # Checks against the model
    # ------------------------------------------------------------------

    def _check_parameter_names(self, model: "Model[Self]") -> None:
        """
        Check that parameter ids are not engine variables or bin ids.

        Expanded compartment names and subpopulation totals are checked by the
        engine (``engine_state.check_names``).
        """
        bin_ids = {bin_item.id for bin_item in self.bins}
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
        valid_identifiers = parameter_ids | self._structure_identifiers()

        for transition in self.transitions:
            rates = [("rate", transition.rate)] if transition.rate else []
            rates += [
                ("stratified_rate", stratified.rate)
                for stratified in transition.stratified_rates or []
            ]
            for context, rate in rates:
                self._check_rate_variables(
                    rate, transition.id, context, valid_identifiers, parameter_ids
                )

    def _structure_identifiers(self) -> set[str]:
        """Identifiers the structure and the engine define for rates."""
        category_ids = {
            cat for strat in self.stratifications for cat in strat.categories
        }
        return (
            {bin_item.id for bin_item in self.bins}
            | category_ids
            | set(RESERVED_VARIABLE_NAMES)
            | self._subpopulation_n_vars()
            | self._full_compartment_names()
            | self._bin_subpopulation_vars()
        )

    def _subpopulation_n_vars(self) -> set[str]:
        """Generates all possible N_{category...} variable names."""
        if not self.stratifications:
            return set()

        subpopulation_n_vars: set[str] = set()
        category_groups = [s.categories for s in self.stratifications]
        for combo_tuple in product(*category_groups):
            for i in range(1, len(combo_tuple) + 1):
                for subset in combinations(combo_tuple, i):
                    subpopulation_n_vars.add(f"N_{'_'.join(subset)}")
        return subpopulation_n_vars

    def _full_compartment_names(self) -> set[str]:
        """Returns all full stratified compartment names."""
        if not self.stratifications:
            return set()
        compartments, _ = engine_state.expanded_outputs(self)
        return {output.name for output in compartments}

    def _bin_subpopulation_vars(self) -> set[str]:
        """Generates partial bin-stratification sum variable names.

        For bins and stratifications, generates the combinations. These represent
        partial sums over one or more stratification dimensions. Requires 2+
        stratifications (with 1, partial sums duplicate existing names).
        """
        if len(self.stratifications) < 2:
            return set()

        bin_strat_vars: set[str] = set()
        bin_ids = [bin_item.id for bin_item in self.bins]
        category_groups = [s.categories for s in self.stratifications]

        for combo_tuple in product(*category_groups):
            num_cats = len(combo_tuple)
            full_mask = (1 << num_cats) - 1
            for subset_mask in range(1, full_mask):
                subset = [
                    cat for k, cat in enumerate(combo_tuple) if (subset_mask >> k) & 1
                ]
                suffix = "_".join(subset)
                for bin_id in bin_ids:
                    bin_strat_vars.add(f"{bin_id}_{suffix}")

        return bin_strat_vars

    def _check_rate_variables(
        self,
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
        bin_ids = {bin_item.id for bin_item in self.bins}
        message = (
            f"Undefined variables in transition '{transition_id}' "
            f"{context} '{rate}': {', '.join(undefined_vars)}. "
            f"Available parameters: "
            f"{', '.join(sorted(parameter_ids)) if parameter_ids else 'none'}. "
            f"Available bins: "
            f"{', '.join(sorted(bin_ids)) if bin_ids else 'none'}."
        )
        if any(var.startswith("N_") for var in undefined_vars):
            n_vars = sorted(self._subpopulation_n_vars())
            message += (
                " Subpopulation names follow the declaration order of the "
                "stratifications. Available subpopulations: "
                f"{', '.join(n_vars) if n_vars else 'none'}."
            )
        raise ValueError(message)

    def _check_transition_ids(self) -> None:
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
                        f"{', '.join(parent_stratification_ids)}. A single "
                        f"transition must only move between categories belonging "
                        f"to the same parent stratification."
                    )
```

### Methods:

#### validate_bins_not_empty

```python
validate_bins_not_empty(v: list[Bin]) -> list[Bin]
```

Require at least one bin.

Source code in `commol/context/compartmental.py`

```python
@field_validator("bins")
@classmethod
def validate_bins_not_empty(cls, v: list[Bin]) -> list[Bin]:
    """Require at least one bin."""
    if not v:
        raise ValueError("At least one bin must be defined.")
    return v
```

#### validate_transitions_not_empty

```python
validate_transitions_not_empty(v: list[Transition]) -> list[Transition]
```

Require at least one transition.

Source code in `commol/context/compartmental.py`

```python
@field_validator("transitions")
@classmethod
def validate_transitions_not_empty(cls, v: list[Transition]) -> list[Transition]:
    """Require at least one transition."""
    if not v:
        raise ValueError("At least one transition must be defined.")
    return v
```

#### validate_unique_ids

```python
validate_unique_ids() -> Self
```

Validate that bin, stratification and transition ids are unique.

Source code in `commol/context/compartmental.py`

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

#### validate_bin_initial_conditions

```python
validate_bin_initial_conditions() -> Self
```

Validate the bin fractions against the bins.

Source code in `commol/context/compartmental.py`

```python
@model_validator(mode="after")
def validate_bin_initial_conditions(self) -> Self:
    """Validate the bin fractions against the bins."""
    bin_fractions = {
        bf.bin: bf.fraction for bf in self.initial_conditions.bin_fractions
    }
    actual_bins = set(bin_fractions)
    expected_bins = {bin_item.id for bin_item in self.bins}
    if actual_bins != expected_bins:
        raise ValueError(
            f"Initial bin fractions keys must exactly match bin ids. "
            f"Missing ids: {expected_bins - actual_bins}, "
            f"Extra ids: {actual_bins - expected_bins}."
        )

    # The sum is checked once every fraction has a value; InitialConditions
    # checks the uncalibrated case.
    calibrated = [f for f in bin_fractions.values() if f is not None]
    if len(calibrated) == len(bin_fractions):
        total = sum(calibrated)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Bin fractions must sum to 1.0, but got {total:.7f}.")
    return self
```

#### validate_stratified_rates

```python
validate_stratified_rates() -> Self
```

Validate that stratified rates name stratifications and categories.

Source code in `commol/context/compartmental.py`

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

#### validate_stratification_initial_conditions

```python
validate_stratification_initial_conditions() -> Self
```

Validate the stratification fractions against the stratifications.

Source code in `commol/context/compartmental.py`

```python
@model_validator(mode="after")
def validate_stratification_initial_conditions(self) -> Self:
    """Validate the stratification fractions against the stratifications."""
    strat_map = {strat.id: strat for strat in self.stratifications}
    actual_strat = {
        sf.stratification for sf in self.initial_conditions.stratification_fractions
    }
    expected_strat = set(strat_map)
    if actual_strat != expected_strat:
        raise ValueError(
            f"Initial stratification fractions keys must exactly match "
            f"stratification ids. Missing ids: {expected_strat - actual_strat}, "
            f"Extra ids: {actual_strat - expected_strat}."
        )

    for strat_fractions in self.initial_conditions.stratification_fractions:
        strat_id = strat_fractions.stratification
        strat_instance = strat_map[strat_id]
        fractions = {sf.category: sf.fraction for sf in strat_fractions.fractions}

        expected = set(strat_instance.categories)
        actual = set(fractions)
        if actual != expected:
            raise ValueError(
                f"Categories for stratification '{strat_id}' must exactly "
                f"match defined categories in instance '{strat_instance.id}'. "
                f"Missing categories: {expected - actual}, "
                f"Extra categories: {actual - expected}."
            )

        total = sum(fractions.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Stratification fractions for '{strat_id}' must sum to 1.0, "
                f"but got {total:.7}."
            )
    return self
```

#### check_model

```python
check_model(model: Model[Self]) -> None
```

Check the structure against the parameters of its model.

Parameter ids must not be engine variables or bin ids, every variable of a rate must be defined, transitions must move between bins or between the categories of one stratification, and the engine must plan every transition and build the initial state.

Parameters:

| Name    | Type    | Description                          | Default    |
| ------- | ------- | ------------------------------------ | ---------- |
| `model` | `Model` | The model that holds this structure. | *required* |

Raises:

| Type         | Description                   |
| ------------ | ----------------------------- |
| `ValueError` | If one of these checks fails. |

Source code in `commol/context/compartmental.py`

```python
def check_model(self, model: "Model[Self]") -> None:
    """
    Check the structure against the parameters of its model.

    Parameter ids must not be engine variables or bin ids, every variable
    of a rate must be defined, transitions must move between bins or
    between the categories of one stratification, and the engine must
    plan every transition and build the initial state.

    Parameters
    ----------
    model : Model
        The model that holds this structure.

    Raises
    ------
    ValueError
        If one of these checks fails.
    """
    self._check_parameter_names(model)
    engine_state.transition_flows(self)
    engine_state.check_names(model)
    if self.initial_conditions.compartment_fractions:
        engine_state.initial_state(self)
    self._check_formula_variables(model)
    self._check_transition_ids()
```

#### outputs_by_source

```python
outputs_by_source() -> dict[str, list[str]]
```

Map each bin and accumulator id to the output names it expands into.

A bin or accumulator produces one output per combination of the stratification categories that apply to it. Without stratifications it produces a single output named after the id itself.

Returns:

| Type                   | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| `dict[str, list[str]]` | Bin and accumulator ids, each mapped to its output names in declaration order. |

Source code in `commol/context/compartmental.py`

```python
def outputs_by_source(self) -> dict[str, list[str]]:
    """
    Map each bin and accumulator id to the output names it expands into.

    A bin or accumulator produces one output per combination of the
    stratification categories that apply to it. Without stratifications it
    produces a single output named after the id itself.

    Returns
    -------
    dict[str, list[str]]
        Bin and accumulator ids, each mapped to its output names in
        declaration order.
    """
    outputs: dict[str, list[str]] = {
        source_id: []
        for source_id in (
            *(bin_item.id for bin_item in self.bins),
            *(accumulator.id for accumulator in self.accumulators),
        )
    }
    compartments, accumulators = engine_state.expanded_outputs(self)
    for output in (*compartments, *accumulators):
        outputs[output.source].append(output.name)
    return outputs
```

#### series_label

```python
series_label(series_id: str) -> SeriesLabel
```

The name and unit of a bin, for a plot of its series.

Parameters:

| Name        | Type  | Description                       | Default    |
| ----------- | ----- | --------------------------------- | ---------- |
| `series_id` | `str` | A bin id, or any other series id. | *required* |

Returns:

| Type          | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `SeriesLabel` | The bin's name and unit, or the id and no unit when no bin has it. |

Source code in `commol/context/compartmental.py`

```python
def series_label(self, series_id: str) -> SeriesLabel:
    """
    The name and unit of a bin, for a plot of its series.

    Parameters
    ----------
    series_id : str
        A bin id, or any other series id.

    Returns
    -------
    SeriesLabel
        The bin's name and unit, or the id and no unit when no bin has it.
    """
    bin_item = next((b for b in self.bins if b.id == series_id), None)
    if bin_item is None:
        return SeriesLabel(series_id, None)
    return SeriesLabel(bin_item.name or series_id, bin_item.unit)
```

#### uncalibrated_values

```python
uncalibrated_values() -> list[str]
```

Ids of the bins whose initial fraction needs calibration.

Returns:

| Type        | Description                   |
| ----------- | ----------------------------- |
| `list[str]` | Bin ids with a None fraction. |

Source code in `commol/context/compartmental.py`

```python
def uncalibrated_values(self) -> list[str]:
    """
    Ids of the bins whose initial fraction needs calibration.

    Returns
    -------
    list[str]
        Bin ids with a None fraction.
    """
    return self.initial_conditions.get_uncalibrated_bins()
```

#### with_calibrated_values

```python
with_calibrated_values(model: Model[Self], problem: CalibrationProblem, values: Mapping[str, float]) -> Self
```

The structure with calibrated initial conditions written back.

The written structure rebuilds the initial population the engine scored. Bin and category values update the fractions they declare. Every compartment those fractions do not reproduce, such as an exact compartment value or a value that overlaps a per-compartment override, is written as a per-compartment override.

Parameters:

| Name      | Type                  | Description                                            | Default    |
| --------- | --------------------- | ------------------------------------------------------ | ---------- |
| `model`   | `Model`               | The model the problem was built on. It is not changed. | *required* |
| `problem` | `CalibrationProblem`  | The problem, for its parameters and remainder.         | *required* |
| `values`  | `Mapping[str, float]` | Calibrated value per parameter id.                     | *required* |

Returns:

| Type                     | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| `CompartmentalStructure` | A copy that holds the calibrated initial conditions. |

Raises:

| Type         | Description                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the engine cannot resolve an initial-condition id or the remainder, or the initial population exceeds the population size. |

Source code in `commol/context/compartmental.py`

```python
def with_calibrated_values(
    self,
    model: "Model[Self]",
    problem: "CalibrationProblem",
    values: Mapping[str, float],
) -> Self:
    """
    The structure with calibrated initial conditions written back.

    The written structure rebuilds the initial population the engine
    scored. Bin and category values update the fractions they declare.
    Every compartment those fractions do not reproduce, such as an exact
    compartment value or a value that overlaps a per-compartment override,
    is written as a per-compartment override.

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
    CompartmentalStructure
        A copy that holds the calibrated initial conditions.

    Raises
    ------
    ValueError
        If the engine cannot resolve an initial-condition id or the
        remainder, or the initial population exceeds the population size.
    """
    from commol.context._model.calibration_write_back import (
        calibrated_initial_conditions,
    )

    initial_conditions = calibrated_initial_conditions(model, problem, values)
    return self.model_copy(update={"initial_conditions": initial_conditions})
```

#### report_sections

```python
report_sections(model: Model[Self], calibrated_ids: set[str], parameter_statistics: Mapping[str, ParameterSetStatistics] | None) -> ReportSections
```

The `initial_conditions` section of a report of applied values.

See :func:`commol.applied_parameters_report`.

Parameters:

| Name                   | Type                                   | Description                              | Default                                              |
| ---------------------- | -------------------------------------- | ---------------------------------------- | ---------------------------------------------------- |
| `model`                | `Model`                                | The model that holds the applied values. | *required*                                           |
| `calibrated_ids`       | `set[str]`                             | Ids that calibration set.                | *required*                                           |
| `parameter_statistics` | \`Mapping[str, ParameterSetStatistics] | None\`                                   | Ensemble spread per id, or None without an ensemble. |

Returns:

| Type             | Description                  |
| ---------------- | ---------------------------- |
| `ReportSections` | {"initial_conditions": ...}. |

Source code in `commol/context/compartmental.py`

```python
def report_sections(
    self,
    model: "Model[Self]",  # noqa: ARG002
    calibrated_ids: set[str],
    parameter_statistics: "Mapping[str, ParameterSetStatistics] | None",
) -> ReportSections:
    """
    The ``initial_conditions`` section of a report of applied values.

    See :func:`commol.applied_parameters_report`.

    Parameters
    ----------
    model : Model
        The model that holds the applied values.
    calibrated_ids : set[str]
        Ids that calibration set.
    parameter_statistics : Mapping[str, ParameterSetStatistics] | None
        Ensemble spread per id, or None without an ensemble.

    Returns
    -------
    ReportSections
        ``{"initial_conditions": ...}``.
    """
    from commol.context._model.report import initial_conditions_report

    return {
        "initial_conditions": initial_conditions_report(
            self, calibrated_ids, parameter_statistics
        )
    }
```

#### print_equations

```python
print_equations(model: Model[Self], output_file: str | None, format: str) -> None
```

Print the equations of the model; see `Model.print_equations`.

Source code in `commol/context/compartmental.py`

```python
def print_equations(
    self, model: "Model[Self]", output_file: str | None, format: str
) -> None:
    """Print the equations of the model; see `Model.print_equations`."""
    from commol.context._model.equations import EquationPrinter

    EquationPrinter(model).print_equations(output_file, format)
```

#### check_unit_consistency

```python
check_unit_consistency(model: Model[Self], verbose: bool) -> None
```

Check the units of the model; see `Model.check_unit_consistency`.

Source code in `commol/context/compartmental.py`

```python
def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
    """Check the units of the model; see `Model.check_unit_consistency`."""
    from commol.context._model.units import UnitChecker

    UnitChecker(model).check_unit_consistency(verbose)
```

#### update_initial_conditions

```python
update_initial_conditions(bin_fractions: Mapping[str, float | None]) -> None
```

Update initial condition fractions for specified bins.

Parameters:

| Name            | Type                  | Description | Default                                                  |
| --------------- | --------------------- | ----------- | -------------------------------------------------------- |
| `bin_fractions` | \`Mapping\[str, float | None\]\`    | Dictionary mapping bin IDs to their new fraction values. |

Raises:

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `ValueError` | If a bin ID in the dictionary doesn't exist in the model. |

Source code in `commol/context/compartmental.py`

```python
def update_initial_conditions(
    self, bin_fractions: Mapping[str, float | None]
) -> None:
    """
    Update initial condition fractions for specified bins.

    Parameters
    ----------
    bin_fractions : Mapping[str, float | None]
        Dictionary mapping bin IDs to their new fraction values.

    Raises
    ------
    ValueError
        If a bin ID in the dictionary doesn't exist in the model.
    """
    self.initial_conditions.update_bin_fractions(bin_fractions)
```

#### update_stratification_fractions

```python
update_stratification_fractions(fractions: Mapping[str, float]) -> None
```

Update initial stratification fractions for specified categories.

Within each stratification, at most one category may be omitted; the omitted category receives the remaining fraction.

Parameters:

| Name        | Type                  | Description                                                     | Default    |
| ----------- | --------------------- | --------------------------------------------------------------- | ---------- |
| `fractions` | `Mapping[str, float]` | Dictionary mapping category names to their new fraction values. | *required* |

Raises:

| Type         | Description                                                                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a category doesn't exist in the model, a value lies outside [0.0, 1.0], a stratification has more than one category omitted, or the resulting fractions do not sum to 1.0. |

Source code in `commol/context/compartmental.py`

```python
def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
    """
    Update initial stratification fractions for specified categories.

    Within each stratification, at most one category may be omitted; the
    omitted category receives the remaining fraction.

    Parameters
    ----------
    fractions : Mapping[str, float]
        Dictionary mapping category names to their new fraction values.

    Raises
    ------
    ValueError
        If a category doesn't exist in the model, a value lies outside
        [0.0, 1.0], a stratification has more than one category omitted, or
        the resulting fractions do not sum to 1.0.
    """
    self.initial_conditions.update_stratification_fractions(fractions)
```

#### update_compartment_fractions

```python
update_compartment_fractions(fractions: Mapping[str, float]) -> None
```

Upsert per-compartment initial condition overrides.

A compartment already declared has its fraction replaced; a new one is appended. The engine validates the result before it is stored, so a rejected call leaves the structure unchanged.

Parameters:

| Name        | Type                  | Description                                                                                  | Default    |
| ----------- | --------------------- | -------------------------------------------------------------------------------------------- | ---------- |
| `fractions` | `Mapping[str, float]` | Dictionary mapping expanded compartment names to their new fraction of the total population. | *required* |

Raises:

| Type         | Description                                                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a name is not an expanded compartment of the model, a value lies outside [0.0, 1.0], or the resulting overrides make the initial population exceed population_size. |

Source code in `commol/context/compartmental.py`

```python
def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
    """
    Upsert per-compartment initial condition overrides.

    A compartment already declared has its fraction replaced; a new one is
    appended. The engine validates the result before it is stored, so a
    rejected call leaves the structure unchanged.

    Parameters
    ----------
    fractions : Mapping[str, float]
        Dictionary mapping expanded compartment names to their new fraction
        of the total population.

    Raises
    ------
    ValueError
        If a name is not an expanded compartment of the model, a value lies
        outside ``[0.0, 1.0]``, or the resulting overrides make the initial
        population exceed ``population_size``.
    """
    candidate = self.initial_conditions.model_copy(deep=True)
    candidate.update_compartment_fractions(fractions)
    engine_state.initial_state(
        self.model_copy(update={"initial_conditions": candidate})
    )
    self.initial_conditions = candidate
```

#### initial_state

```python
initial_state() -> list[InitialCompartment]
```

Expanded compartments and the population each one starts with.

The engine builds this state from the bin fractions, the stratification fractions and the per-compartment overrides; it is the population a simulation starts from.

Returns:

| Type                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `list[InitialCompartment]` | One entry per expanded compartment, in engine order. |

Raises:

| Type         | Description                                                       |
| ------------ | ----------------------------------------------------------------- |
| `ValueError` | If the per-compartment overrides are not valid for the structure. |

Source code in `commol/context/compartmental.py`

```python
def initial_state(self) -> list[InitialCompartment]:
    """
    Expanded compartments and the population each one starts with.

    The engine builds this state from the bin fractions, the stratification
    fractions and the per-compartment overrides; it is the population a
    simulation starts from.

    Returns
    -------
    list[InitialCompartment]
        One entry per expanded compartment, in engine order.

    Raises
    ------
    ValueError
        If the per-compartment overrides are not valid for the structure.
    """
    return engine_state.initial_state(self)
```

#### get_conditioning_categories

```python
get_conditioning_categories(stratification_id: str) -> tuple[str, ...]
```

Get the categories a stratification is conditioned on.

An unconditional stratification subdivides the whole population and has no conditioning categories. A conditional one subdivides only the subgroup its conditions select, and its fractions are relative to that subgroup rather than to the whole population.

Parameters:

| Name                | Type  | Description                          | Default    |
| ------------------- | ----- | ------------------------------------ | ---------- |
| `stratification_id` | `str` | Id of a stratification of the model. | *required* |

Returns:

| Type              | Description                                                                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tuple[str, ...]` | Ids of the categories selecting the subgroup this stratification subdivides, empty for an unconditional stratification. A category name another stratification also declares is qualified, as in sex:female. |

Raises:

| Type         | Description                         |
| ------------ | ----------------------------------- |
| `ValueError` | If the stratification is not found. |

Source code in `commol/context/compartmental.py`

```python
def get_conditioning_categories(self, stratification_id: str) -> tuple[str, ...]:
    """
    Get the categories a stratification is conditioned on.

    An unconditional stratification subdivides the whole population and has
    no conditioning categories. A conditional one subdivides only the
    subgroup its conditions select, and its fractions are relative to that
    subgroup rather than to the whole population.

    Parameters
    ----------
    stratification_id : str
        Id of a stratification of the model.

    Returns
    -------
    tuple[str, ...]
        Ids of the categories selecting the subgroup this stratification
        subdivides, empty for an unconditional stratification. A category
        name another stratification also declares is qualified, as in
        ``sex:female``.

    Raises
    ------
    ValueError
        If the stratification is not found.
    """
    for stratification in self.stratifications:
        if stratification.id == stratification_id:
            return tuple(
                self.initial_conditions.category_id(
                    condition.stratification, condition.category
                )
                for condition in (stratification.conditions or [])
                if condition.category is not None
            )
    raise ValueError(
        f"Stratification '{stratification_id}' not found. Available: "
        f"{sorted(s.id for s in self.stratifications)}"
    )
```

#### subgroup_population

```python
subgroup_population(categories: Iterable[str] = ()) -> float
```

Get the initial head count of a subgroup.

See :meth:`~commol.context.initial_conditions.InitialConditions.subgroup_population`.

Per-compartment overrides are folded into the sum for every compartment the category chain selects, so the head count matches the population the engine builds. Without overrides the declared fractions are multiplied.

Parameters:

| Name         | Type            | Description                                                                                                                    | Default |
| ------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `categories` | `Iterable[str]` | Category ids, bare or qualified (sex:female), forming a chain from the whole population down to the subgroup, outermost first. | `()`    |

Returns:

| Type    | Description                                      |
| ------- | ------------------------------------------------ |
| `float` | Number of individuals in the subgroup at step 0. |

Raises:

| Type         | Description                                             |
| ------------ | ------------------------------------------------------- |
| `ValueError` | If a category is not found or a bare name is ambiguous. |

Source code in `commol/context/compartmental.py`

```python
def subgroup_population(self, categories: Iterable[str] = ()) -> float:
    """
    Get the initial head count of a subgroup.

    See
    :meth:`~commol.context.initial_conditions.InitialConditions.subgroup_population`.

    Per-compartment overrides are folded into the sum for every compartment
    the category chain selects, so the head count matches the population
    the engine builds. Without overrides the declared fractions are
    multiplied.

    Parameters
    ----------
    categories : Iterable[str], optional
        Category ids, bare or qualified (``sex:female``), forming a chain
        from the whole population down to the subgroup, outermost first.

    Returns
    -------
    float
        Number of individuals in the subgroup at step 0.

    Raises
    ------
    ValueError
        If a category is not found or a bare name is ambiguous.
    """
    selected = tuple(categories)
    initial_conditions = self.initial_conditions
    if not initial_conditions.compartment_fractions:
        return initial_conditions.subgroup_population(selected)
    resolved = [initial_conditions.resolve_category(c) for c in selected]
    return sum(
        compartment.population
        for compartment in self.initial_state()
        if all(
            compartment.categories.get(stratification) == name
            for stratification, name in resolved
        )
    )
```

## InitialCompartment

Bases: `NamedTuple`

One expanded compartment and the population it starts with.

Attributes:

| Name         | Type             | Description                                                                                                                                             |
| ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`       | `str`            | Expanded compartment name, for example IH_y60.                                                                                                          |
| `bin`        | `str`            | Bin the compartment expands from.                                                                                                                       |
| `categories` | `dict[str, str]` | Category applied to the compartment, keyed by stratification id. A conditional stratification whose conditions the compartment does not meet is absent. |
| `population` | `float`          | Initial head count, including per-compartment overrides.                                                                                                |

Source code in `commol/context/_model/engine_state.py`

```python
class InitialCompartment(NamedTuple):
    """
    One expanded compartment and the population it starts with.

    Attributes
    ----------
    name : str
        Expanded compartment name, for example ``IH_y60``.
    bin : str
        Bin the compartment expands from.
    categories : dict[str, str]
        Category applied to the compartment, keyed by stratification id. A
        conditional stratification whose conditions the compartment does not
        meet is absent.
    population : float
        Initial head count, including per-compartment overrides.
    """

    name: str
    bin: str
    categories: dict[str, str]
    population: float
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
