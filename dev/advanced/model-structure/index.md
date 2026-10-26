# Model Structure

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

A `Model` is a tree of Pydantic models. `ModelBuilder` creates it, `Model.to_json` writes it, and the Rust engine reads the same tree from JSON (`crates/commol-core/src/types/`). Most users never build these types by hand; read this page to understand model files or to work on Commol.

A model has a header, a **process** written once, a **structure** that lays the process out, and an **engine** that runs it. The process (bins, accumulators, stratifications, transitions) is the same for every kind of model; the structure changes between kinds and is tagged in JSON by `kind`, and by `variant` for a narrower form of a structure. The one kind is `aggregate`: amounts per compartment.

```text
Model                      name, description, version, step_unit, origin
├── parameters: [Parameter]                (a value may be a SeriesValue)
├── engine: ModelEngine
│   ├── type: "DifferenceEquations", "DifferentialEquations",
│   │         "StochasticEvents" or "StochasticSteps"
│   └── settings: ODESolver | StochasticEventsSettings | StochasticStepsSettings | None
│                                           (settings of that engine; None: its defaults)
├── process: Process
│   ├── bins: [Bin]
│   ├── accumulators: [Accumulator]
│   ├── stratifications: [Stratification]
│   │   └── conditions: [StratificationCondition]
│   └── transitions: [Transition]          (impulse_at, impulse_every for impulses)
│       └── stratified_rates: [StratifiedRate]
│           └── conditions: [StratificationCondition]
└── structure: Aggregate                   (kind: "aggregate", variant: None)
    ├── bin_amounts: {bin: amount | None}          (in the unit of the bin)
    ├── stratification_fractions: {stratification: {category: fraction}}
    ├── compartment_amounts: {compartment: amount}
    └── neighbors: Complete | Groups | None
```

In JSON the structure is an object with its `kind`, its `variant` and the fields of its class:

```json
{
  "name": "Model",
  "parameters": [],
  "engine": {"type": "DifferenceEquations", "settings": null},
  "process": {"bins": [], "accumulators": [], "stratifications": [], "transitions": []},
  "structure": {
    "kind": "aggregate",
    "variant": null,
    "bin_amounts": {"A": 990.0, "B": 10.0},
    "stratification_fractions": {},
    "compartment_amounts": {},
    "neighbors": null
  },
  "step_unit": null,
  "origin": null
}
```

`Model`, `Process`, `ModelStructure`, `Aggregate`, `Complete` and `Groups` are documented in [Model API](https://munqu.github.io/commol/latest/api/model/index.md). The engine must run the kind of the structure: a model whose engine runs another kind is rejected when it is made, with an error that names the engine, both kinds and the engines of the kind of the model. A variant keeps the kind, the fields and the engines of its structure, and loads back as its own class ([Design decisions](https://munqu.github.io/commol/latest/codebase/decisions/model-kinds/#a-model-is-a-header-a-process-a-structure-and-an-engine)).

## Bins and accumulators

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

Source code in `commol/context/bin.py`

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

Source code in `commol/context/bin.py`

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

## Stratifications

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

Source code in `commol/context/stratification.py`

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
    def validate_categories_length(self) -> Self:
        """
        Enforces that categories are not empty.
        """
        if not self.categories:
            raise ValueError(
                (f"Stratification '{self.id}' must have at least one category.")
            )
        return self

    @model_validator(mode="after")
    def validate_categories_uniqueness(self) -> Self:
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

    @model_validator(mode="after")
    def validate_conditions_have_categories(self) -> Self:
        """
        Enforces that stratification expansion conditions filter by category.
        """
        if self.conditions is None:
            return self

        for condition in self.conditions:
            if condition.category is None:
                raise ValueError(
                    (
                        f"Condition for stratification '{self.id}' referencing "
                        f"'{condition.stratification}' must include a category. "
                        "Category-less conditions are only valid for transition "
                        "target overrides."
                    )
                )

        return self
```

### Methods:

#### validate_categories_length

```python
validate_categories_length() -> Self
```

Enforces that categories are not empty.

Source code in `commol/context/stratification.py`

```python
@model_validator(mode="after")
def validate_categories_length(self) -> Self:
    """
    Enforces that categories are not empty.
    """
    if not self.categories:
        raise ValueError(
            (f"Stratification '{self.id}' must have at least one category.")
        )
    return self
```

#### validate_categories_uniqueness

```python
validate_categories_uniqueness() -> Self
```

Enforces that categories are not repeated.

Source code in `commol/context/stratification.py`

```python
@model_validator(mode="after")
def validate_categories_uniqueness(self) -> Self:
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

#### validate_conditions_have_categories

```python
validate_conditions_have_categories() -> Self
```

Enforces that stratification expansion conditions filter by category.

Source code in `commol/context/stratification.py`

```python
@model_validator(mode="after")
def validate_conditions_have_categories(self) -> Self:
    """
    Enforces that stratification expansion conditions filter by category.
    """
    if self.conditions is None:
        return self

    for condition in self.conditions:
        if condition.category is None:
            raise ValueError(
                (
                    f"Condition for stratification '{self.id}' referencing "
                    f"'{condition.stratification}' must include a category. "
                    "Category-less conditions are only valid for transition "
                    "target overrides."
                )
            )

    return self
```

## StratificationCondition

Bases: `BaseModel`

Specifies a category within a stratification for rate matching.

When `to` is set, the matched category in the source compartment is replaced with the `to` category in the target compartment name. This enables cross-category transitions within the same bin (e.g., from `g1` to `g2`).

Attributes:

| Name             | Type  | Description                                                |
| ---------------- | ----- | ---------------------------------------------------------- |
| `stratification` | `str` | The ID of the stratification (e.g., "group", "type")       |
| `category`       | `str` | The category within that stratification (e.g., "g1", "t1") |
| `to`             | \`str | None\`                                                     |

Source code in `commol/context/stratification_condition.py`

```python
class StratificationCondition(BaseModel):
    """
    Specifies a category within a stratification for rate matching.

    When ``to`` is set, the matched category in the source compartment is
    replaced with the ``to`` category in the target compartment name. This
    enables cross-category transitions within the same bin (e.g., from ``g1``
    to ``g2``).

    Attributes
    ----------
    stratification : str
        The ID of the stratification (e.g., "group", "type")
    category : str
        The category within that stratification (e.g., "g1", "t1")
    to : str | None
        Target category override. When set, the target compartment will use
        this category instead of the source compartment's category for this
        stratification.
    """

    model_config = ConfigDict(extra="forbid")

    stratification: str = Field(default=..., description="ID of the stratification")
    category: str | None = Field(
        default=None,
        description=(
            "Category within the stratification. When None, the condition acts as a "
            "target-only override: it does not filter source compartments but still "
            "contributes a 'to' category to the computed target compartment name."
        ),
    )
    to: str | None = Field(
        default=None,
        description=(
            "Target category override. When set, the target compartment uses "
            "this category instead of the source's for this stratification."
        ),
    )
```

## Parameters

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

Source code in `commol/context/parameter.py`

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
    def validate_value(
        cls, value: float | str | SeriesValue | None
    ) -> float | str | SeriesValue | None:
        """Validate the parameter value."""
        if value is None or isinstance(value, SeriesValue):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        return value.strip()

    @model_validator(mode="after")
    def validate_formula(self) -> Self:
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

    def with_value(self, value: ParameterUpdate) -> "Parameter":
        """
        This parameter with another value.

        Parameters
        ----------
        value : ParameterUpdate
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

#### validate_value

```python
validate_value(value: float | str | SeriesValue | None) -> float | str | SeriesValue | None
```

Validate the parameter value.

Source code in `commol/context/parameter.py`

```python
@field_validator("value")
@classmethod
def validate_value(
    cls, value: float | str | SeriesValue | None
) -> float | str | SeriesValue | None:
    """Validate the parameter value."""
    if value is None or isinstance(value, SeriesValue):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value.strip()
```

#### validate_formula

```python
validate_formula() -> Self
```

Check a formula value as a rate is checked, with the parser of the engine.

Source code in `commol/context/parameter.py`

```python
@model_validator(mode="after")
def validate_formula(self) -> Self:
    """Check a formula value as a rate is checked, with the parser of the
    engine."""
    if isinstance(self.value, str):
        try:
            commol_rs.core.MathExpression(self.value).check()
        except ValueError as error:
            raise ValueError(f"Parameter '{self.id}': {error}")
    return self
```

#### is_calibrated

```python
is_calibrated() -> bool
```

Check if the parameter has a value (is calibrated).

Returns:

| Type   | Description                                                       |
| ------ | ----------------------------------------------------------------- |
| `bool` | True if the parameter has a value, False if it needs calibration. |

Source code in `commol/context/parameter.py`

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
with_value(value: ParameterUpdate) -> Parameter
```

This parameter with another value.

Parameters:

| Name    | Type              | Description                                                                                                                                                 | Default    |
| ------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `value` | `ParameterUpdate` | The new value, as Parameter.value takes it. A list of (position, value) points replaces the points of a series and keeps its mode; a SeriesValue sets both. | *required* |

Returns:

| Type        | Description                                             |
| ----------- | ------------------------------------------------------- |
| `Parameter` | A new parameter with the same id, description and unit. |

Raises:

| Type         | Description                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If value is a list and this parameter is not a series, or value is not a valid value; the message names the parameter. |

Source code in `commol/context/parameter.py`

```python
def with_value(self, value: ParameterUpdate) -> "Parameter":
    """
    This parameter with another value.

    Parameters
    ----------
    value : ParameterUpdate
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

The value of a series parameter, `SeriesValue`, is documented in [Model API](https://munqu.github.io/commol/latest/api/model/index.md).

## Transitions

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

Source code in `commol/context/transition.py`

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
    def validate_impulse(self) -> Self:
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
    def validate_rate(cls, value: str | None) -> str | None:
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

### Methods:

#### validate_impulse

```python
validate_impulse() -> Self
```

Check the impulse with the rule of the engine (`check_impulse` of `crates/commol-model/src/values.rs`): one of `impulse_at` and `impulse_every`, at least one position on the axis, a positive period and an offset that is not negative.

Source code in `commol/context/transition.py`

```python
@model_validator(mode="after")
def validate_impulse(self) -> Self:
    """
    Check the impulse with the rule of the engine (`check_impulse` of
    `crates/commol-model/src/values.rs`): one of `impulse_at` and
    `impulse_every`, at least one position on the axis, a positive period
    and an offset that is not negative.
    """
    commol_rs.model.check_impulse(self.id, self.impulse_at, self.impulse_every)
    return self
```

#### validate_rate

```python
validate_rate(value: str | None) -> str | None
```

Check the rate, when there is one, with the parser of the engine.

Source code in `commol/context/transition.py`

```python
@field_validator("rate")
@classmethod
def validate_rate(cls, value: str | None) -> str | None:
    """Check the rate, when there is one, with the parser of the engine."""
    if value is None:
        return value
    return _checked_rate(value)
```

## StratifiedRate

Bases: `BaseModel`

Defines a rate for specific stratification categories.

Attributes:

| Name         | Type                            | Description                                                |
| ------------ | ------------------------------- | ---------------------------------------------------------- |
| `conditions` | `list[StratificationCondition]` | List of stratification-category pairs that must match      |
| `rate`       | `str`                           | Rate expression for compartments matching these conditions |
| `absolute`   | \`bool                          | None\`                                                     |

Source code in `commol/context/transition.py`

```python
class StratifiedRate(BaseModel):
    """
    Defines a rate for specific stratification categories.

    Attributes
    ----------
    conditions : list[StratificationCondition]
        List of stratification-category pairs that must match
    rate : str
        Rate expression for compartments matching these conditions
    absolute : bool | None
        True: the flow is the rate, not multiplied by the source amount.
        False: the flow is the rate times the source amount. None: the rate
        is absolute when it reads a compartment, a bin total or a
        ``total_…`` variable.
    """

    model_config = ConfigDict(extra="forbid")

    conditions: list[StratificationCondition] = Field(
        default=..., description="Stratification conditions that must match"
    )
    rate: str = Field(
        default=..., description="Rate expression for matching compartments"
    )
    absolute: bool | None = Field(
        default=None,
        description=(
            "When True, the flow is treated as absolute (not multiplied by the "
            "source amount). When None, inferred from whether the rate references "
            "any compartment variable."
        ),
    )

    @field_validator("conditions")
    @classmethod
    def validate_not_empty(
        cls, v: list[StratificationCondition]
    ) -> list[StratificationCondition]:
        """Ensure at least one stratification condition is specified."""
        if not v:
            raise ValueError(
                "conditions must contain at least one stratification condition"
            )
        return v

    @model_validator(mode="after")
    def validate_unique_stratifications(self) -> Self:
        """Ensure no duplicate stratification IDs."""
        strat_ids = [sc.stratification for sc in self.conditions]
        if len(strat_ids) != len(set(strat_ids)):
            duplicates = [s for s in strat_ids if strat_ids.count(s) > 1]
            raise ValueError(
                (
                    f"Duplicate stratifications found: {list(set(duplicates))}. "
                    f"Each stratification can only be specified once per rate."
                )
            )
        return self

    @field_validator("rate")
    @classmethod
    def validate_rate(cls, value: str) -> str:
        """Check the rate with the parser of the engine."""
        return _checked_rate(value)
```

### Methods:

#### validate_not_empty

```python
validate_not_empty(v: list[StratificationCondition]) -> list[StratificationCondition]
```

Ensure at least one stratification condition is specified.

Source code in `commol/context/transition.py`

```python
@field_validator("conditions")
@classmethod
def validate_not_empty(
    cls, v: list[StratificationCondition]
) -> list[StratificationCondition]:
    """Ensure at least one stratification condition is specified."""
    if not v:
        raise ValueError(
            "conditions must contain at least one stratification condition"
        )
    return v
```

#### validate_unique_stratifications

```python
validate_unique_stratifications() -> Self
```

Ensure no duplicate stratification IDs.

Source code in `commol/context/transition.py`

```python
@model_validator(mode="after")
def validate_unique_stratifications(self) -> Self:
    """Ensure no duplicate stratification IDs."""
    strat_ids = [sc.stratification for sc in self.conditions]
    if len(strat_ids) != len(set(strat_ids)):
        duplicates = [s for s in strat_ids if strat_ids.count(s) > 1]
        raise ValueError(
            (
                f"Duplicate stratifications found: {list(set(duplicates))}. "
                f"Each stratification can only be specified once per rate."
            )
        )
    return self
```

#### validate_rate

```python
validate_rate(value: str) -> str
```

Check the rate with the parser of the engine.

Source code in `commol/context/transition.py`

```python
@field_validator("rate")
@classmethod
def validate_rate(cls, value: str) -> str:
    """Check the rate with the parser of the engine."""
    return _checked_rate(value)
```

## Engine

The engine's list of engines (`crates/commol-core/src/types/engine.rs`) validates the settings: settings of another engine raise an error that names both. `ODESolver` is documented in [Simulation API](https://munqu.github.io/commol/latest/api/simulation/index.md).

## ModelEngine

Bases: `BaseModel`

The engine that runs a model, and its settings.

Attributes:

| Name       | Type             | Description                                                                                                                                                                                     |
| ---------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`     | `str`            | Name of an engine of the engine's list, such as "DifferenceEquations" (discrete time) or "DifferentialEquations" (continuous time). The engine must run the kind of the structure of the model. |
| `settings` | \`EngineSettings | None\`                                                                                                                                                                                          |

Source code in `commol/context/engine.py`

```python
class ModelEngine(BaseModel):
    """
    The engine that runs a model, and its settings.

    Attributes
    ----------
    type : str
        Name of an engine of the engine's list, such as
        ``"DifferenceEquations"`` (discrete time) or
        ``"DifferentialEquations"`` (continuous time). The engine must run the
        kind of the structure of the model.
    settings : EngineSettings | None
        Settings of that engine; None uses its defaults.
        ``DifferentialEquations`` takes `ODESolver`, ``StochasticEvents``
        takes `StochasticEventsSettings` and ``StochasticSteps`` takes
        `StochasticStepsSettings`.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Name of the engine.")
    settings: EngineSettings | None = Field(
        default=None, description="Settings of the engine; None uses its defaults."
    )

    @field_validator("settings", mode="before")
    @classmethod
    def settings_of_the_engine(
        cls,
        value: EngineSettings | Mapping[str, JsonValue] | None,
        info: ValidationInfo,
    ) -> EngineSettings | Mapping[str, JsonValue] | None:
        """Read settings given as a mapping with the class of their engine."""
        settings_class = _SETTINGS.get(info.data.get("type", ""))
        if isinstance(value, Mapping) and settings_class is not None:
            return settings_class.model_validate(value)
        return value

    @model_validator(mode="after")
    def check_settings(self) -> Self:
        """Ask the engine's list whether it has the engine and the settings."""
        settings_json = None if self.settings is None else engine_json(self.settings)
        commol_rs.engine.EngineType(engine_json(self.type)).check_settings(
            settings_json
        )
        return self
```

### Methods:

#### settings_of_the_engine

```python
settings_of_the_engine(value: EngineSettings | Mapping[str, JsonValue] | None, info: ValidationInfo) -> EngineSettings | Mapping[str, JsonValue] | None
```

Read settings given as a mapping with the class of their engine.

Source code in `commol/context/engine.py`

```python
@field_validator("settings", mode="before")
@classmethod
def settings_of_the_engine(
    cls,
    value: EngineSettings | Mapping[str, JsonValue] | None,
    info: ValidationInfo,
) -> EngineSettings | Mapping[str, JsonValue] | None:
    """Read settings given as a mapping with the class of their engine."""
    settings_class = _SETTINGS.get(info.data.get("type", ""))
    if isinstance(value, Mapping) and settings_class is not None:
        return settings_class.model_validate(value)
    return value
```

#### check_settings

```python
check_settings() -> Self
```

Ask the engine's list whether it has the engine and the settings.

Source code in `commol/context/engine.py`

```python
@model_validator(mode="after")
def check_settings(self) -> Self:
    """Ask the engine's list whether it has the engine and the settings."""
    settings_json = None if self.settings is None else engine_json(self.settings)
    commol_rs.engine.EngineType(engine_json(self.type)).check_settings(
        settings_json
    )
    return self
```

## Initial conditions

`Aggregate` holds the fields and methods of `InitialConditions`, and checks them against the process when the model is built.

## InitialConditions

Bases: `BaseModel`

How much starts where: the amount of each bin and the split of it.

Every amount is in the unit of its bin, so bins of different units, such as a count and a concentration, start without a total over them. `Aggregate` holds these fields and their methods; the checks against the bins and the stratifications of the process are in `Aggregate.check_model`.

Attributes:

| Name                       | Type                          | Description                                                                                                                                                                                                                                                   |
| -------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bin_amounts`              | \`dict\[str, float            | None\]\`                                                                                                                                                                                                                                                      |
| `stratification_fractions` | `dict[str, dict[str, float]]` | The fraction of each category, by stratification id and category: the split of the amount of every bin the stratification applies to; each from 0 to 1, and a sum of 1 within 1e-6. A conditional stratification splits only the group its conditions select. |
| `compartment_amounts`      | `dict[str, float]`            | Amounts of single expanded compartments, by compartment name, applied after the expansion.                                                                                                                                                                    |

Source code in `commol/context/initial_conditions.py`

```python
class InitialConditions(BaseModel):
    """
    How much starts where: the amount of each bin and the split of it.

    Every amount is in the unit of its bin, so bins of different units, such
    as a count and a concentration, start without a total over them.
    `Aggregate` holds these fields and their methods; the checks against the
    bins and the stratifications of the process are in
    `Aggregate.check_model`.

    Attributes
    ----------
    bin_amounts : dict[str, float | None]
        The amount of each bin at the start, by bin id, in the unit of the
        bin; finite and not negative. None leaves an amount for calibration.
    stratification_fractions : dict[str, dict[str, float]]
        The fraction of each category, by stratification id and category: the
        split of the amount of every bin the stratification applies to; each
        from 0 to 1, and a sum of 1 within 1e-6. A conditional stratification
        splits only the group its conditions select.
    compartment_amounts : dict[str, float]
        Amounts of single expanded compartments, by compartment name, applied
        after the expansion.
    """

    model_config = ConfigDict(extra="forbid")

    bin_amounts: dict[str, float | None] = Field(
        ...,
        description=(
            "The amount of each bin, by bin id, in the unit of the bin. None "
            "leaves an amount for calibration."
        ),
    )
    stratification_fractions: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="The fraction of each category, by stratification id.",
    )
    compartment_amounts: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Amounts of single expanded compartments, applied after the expansion."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def reject_fractions_of_a_total(cls, data: JsonValue) -> JsonValue:
        """Name the field of amounts that replaces a field of fractions.

        A total over bins of different units has no meaning, so the initial
        state is stated as amounts, and a structure with ``total``,
        ``bin_fractions`` or ``compartment_fractions`` is rejected with the
        field to use.
        """
        if isinstance(data, Mapping):
            stated = [field for field in _FRACTION_FIELDS if field in data]
            if stated:
                replacements = sorted({_FRACTION_FIELDS[field] for field in stated})
                raise ValueError(
                    f"An aggregate structure states the amount of each bin in "
                    f"the unit of the bin, not fractions of a total: replace "
                    f"{', '.join(stated)} with {' and '.join(replacements)}, "
                    f'such as bin_amounts={{"A": 990, "B": 10, "C": 0}}.'
                )
        return data

    @field_validator("bin_amounts")
    @classmethod
    def validate_bin_amounts(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        """Check every bin amount with the rule of the engine."""
        commol_rs.model.check_bin_amounts(list(value.items()))
        return value

    @field_validator("stratification_fractions")
    @classmethod
    def validate_stratification_fractions(
        cls, value: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Check the fractions of every stratification with the rule of the
        engine."""
        _check_fractions(value)
        return value

    @field_validator("compartment_amounts")
    @classmethod
    def validate_compartment_amounts(cls, value: dict[str, float]) -> dict[str, float]:
        """Check every compartment amount with the rule of the engine."""
        commol_rs.model.check_compartment_amounts(list(value.items()))
        return value

    def get_uncalibrated_bins(self) -> list[str]:
        """
        Get list of bin IDs that have uncalibrated amounts (value = None).

        Returns
        -------
        list[str]
            List of bin IDs with uncalibrated initial conditions.
        """
        return [bin_id for bin_id, amount in self.bin_amounts.items() if amount is None]

    def get_categories_with_fractions(self) -> set[str]:
        """
        Get the categories that have a declared initial fraction.

        Returns
        -------
        set[str]
            Category names collected from ``stratification_fractions``. A
            category declared on a stratification but absent from the initial
            conditions is not included.
        """
        return {
            category
            for categories in self.stratification_fractions.values()
            for category in categories
        }

    def resolve_category(self, category: str) -> tuple[str, str]:
        """
        Resolve a category id to its stratification and its bare name.

        A qualified id (``group:g1``) names the stratification. A bare name
        must be declared by exactly one stratification.

        Parameters
        ----------
        category : str
            A bare or qualified category id.

        Returns
        -------
        tuple[str, str]
            The stratification id and the bare category name.

        Raises
        ------
        ValueError
            If the category is not found or a bare name is ambiguous.
        """
        stratification_id, separator, name = category.rpartition(CATEGORY_SEPARATOR)
        owners = [
            stratification
            for stratification, categories in self.stratification_fractions.items()
            if (not separator or stratification == stratification_id)
            and name in categories
        ]
        if len(owners) > 1:
            qualified = " or ".join(
                f"'{owner}{CATEGORY_SEPARATOR}{name}'" for owner in owners
            )
            raise ValueError(
                f"Category '{category}' is ambiguous: it is declared by "
                f"stratifications {owners}. Qualify it as {qualified}."
            )
        if not owners:
            raise ValueError(
                f"Category '{category}' not found in initial conditions. "
                f"Available categories: {sorted(self.get_categories_with_fractions())}"
            )
        return owners[0], name

    def category_id(self, stratification: str, category: str) -> str:
        """
        Shortest id that names a category without ambiguity.

        Parameters
        ----------
        stratification : str
            Id of the stratification that declares the category.
        category : str
            Bare category name.

        Returns
        -------
        str
            The bare name, or ``stratification:category`` when another
            stratification declares a category of the same name.
        """
        declaring = sum(
            category in categories
            for categories in self.stratification_fractions.values()
        )
        if declaring > 1:
            return f"{stratification}{CATEGORY_SEPARATOR}{category}"
        return category

    def get_category_fraction(self, category: str) -> float:
        """
        Get the fraction of a category within the group it subdivides.

        Parameters
        ----------
        category : str
            A category id, bare or qualified (``group:g1``).

        Returns
        -------
        float
            The declared fraction, relative to the group the category's
            stratification applies to rather than to the whole model.

        Raises
        ------
        ValueError
            If the category is not found or a bare name is ambiguous.
        """
        stratification, name = self.resolve_category(category)
        return self.stratification_fractions[stratification][name]

    def _updated_fractions(
        self, stratification: str, named: Mapping[str, float]
    ) -> dict[str, float]:
        """The fractions of a single stratification after an update that
        names ``named``; the one category left unnamed takes the rest."""
        categories = self.stratification_fractions[stratification]
        unnamed = [category for category in categories if category not in named]
        if len(unnamed) > 1:
            raise ValueError(
                f"Cannot update stratification '{stratification}': "
                f"categories {sorted(unnamed)} were left unnamed. At most one "
                f"category may be omitted, so that it can take the remaining "
                f"fraction."
            )
        remainder = max(0.0, 1.0 - sum(named.values()))
        return {category: named.get(category, remainder) for category in categories}

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update stratification category fractions by category name.

        Categories belonging to several stratifications may be given in one
        call. A category name declared by more than one stratification must be
        qualified with its stratification, as in ``group:g1``. Within each
        stratification, at most one category may be omitted; the omitted
        category receives the rest of 1, or 0 when the given values pass 1.
        The fractions of every stratification are checked with the rule of the
        engine before any of them changes.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping category names, bare or qualified, to their new
            fraction values.

        Raises
        ------
        ValueError
            If a category is not found or is ambiguous, or a stratification
            has more than one category omitted; with the message of the
            engine, if a fraction is not from 0 to 1 or the fractions of a
            stratification do not sum to 1 within 1e-6.
        """
        by_stratification: dict[str, dict[str, float]] = {}
        for category, value in fractions.items():
            stratification, name = self.resolve_category(category)
            by_stratification.setdefault(stratification, {})[name] = value

        updated = {
            stratification: self._updated_fractions(stratification, named)
            for stratification, named in by_stratification.items()
        }
        _check_fractions(updated)
        for stratification, categories in updated.items():
            self.stratification_fractions[stratification].update(categories)

    def _upsert_compartment_amounts(self, amounts: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_amounts`` has its
        amount replaced; a new one is appended. Compartment names depend on
        the expansion of the process, so
        :meth:`~commol.context.aggregate.Aggregate.update_compartment_amounts`
        validates them against the engine, not here.

        Raises
        ------
        ValueError
            With the message of the engine, if an amount is not finite or is
            negative.
        """
        commol_rs.model.check_compartment_amounts(list(amounts.items()))
        self.compartment_amounts.update(amounts)

    def update_bin_amounts(self, amounts: Mapping[str, float | None]) -> None:
        """
        Update bin amounts by bin ID.

        Every amount is checked before any bin changes, so a rejected call
        leaves the initial conditions unchanged. The bins of one unit keep no
        sum: to move an amount between bins, update every bin involved.

        Parameters
        ----------
        amounts : Mapping[str, float | None]
            Dictionary mapping bin IDs to their new amount, in the unit of the
            bin. None marks a bin that needs calibration.

        Raises
        ------
        ValueError
            If a bin ID is not found in the initial conditions, or, with the
            message of the engine, an amount is not finite or is negative.
        """
        unknown = sorted(set(amounts) - set(self.bin_amounts))
        if unknown:
            raise ValueError(
                f"Bin(s) {unknown} not found in initial conditions. "
                f"Available bins: {list(self.bin_amounts)}"
            )
        commol_rs.model.check_bin_amounts(list(amounts.items()))
        self.bin_amounts.update(amounts)
```

### Methods:

#### reject_fractions_of_a_total

```python
reject_fractions_of_a_total(data: JsonValue) -> JsonValue
```

Name the field of amounts that replaces a field of fractions.

A total over bins of different units has no meaning, so the initial state is stated as amounts, and a structure with `total`, `bin_fractions` or `compartment_fractions` is rejected with the field to use.

Source code in `commol/context/initial_conditions.py`

```python
@model_validator(mode="before")
@classmethod
def reject_fractions_of_a_total(cls, data: JsonValue) -> JsonValue:
    """Name the field of amounts that replaces a field of fractions.

    A total over bins of different units has no meaning, so the initial
    state is stated as amounts, and a structure with ``total``,
    ``bin_fractions`` or ``compartment_fractions`` is rejected with the
    field to use.
    """
    if isinstance(data, Mapping):
        stated = [field for field in _FRACTION_FIELDS if field in data]
        if stated:
            replacements = sorted({_FRACTION_FIELDS[field] for field in stated})
            raise ValueError(
                f"An aggregate structure states the amount of each bin in "
                f"the unit of the bin, not fractions of a total: replace "
                f"{', '.join(stated)} with {' and '.join(replacements)}, "
                f'such as bin_amounts={{"A": 990, "B": 10, "C": 0}}.'
            )
    return data
```

#### validate_bin_amounts

```python
validate_bin_amounts(value: dict[str, float | None]) -> dict[str, float | None]
```

Check every bin amount with the rule of the engine.

Source code in `commol/context/initial_conditions.py`

```python
@field_validator("bin_amounts")
@classmethod
def validate_bin_amounts(
    cls, value: dict[str, float | None]
) -> dict[str, float | None]:
    """Check every bin amount with the rule of the engine."""
    commol_rs.model.check_bin_amounts(list(value.items()))
    return value
```

#### validate_stratification_fractions

```python
validate_stratification_fractions(value: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]
```

Check the fractions of every stratification with the rule of the engine.

Source code in `commol/context/initial_conditions.py`

```python
@field_validator("stratification_fractions")
@classmethod
def validate_stratification_fractions(
    cls, value: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Check the fractions of every stratification with the rule of the
    engine."""
    _check_fractions(value)
    return value
```

#### validate_compartment_amounts

```python
validate_compartment_amounts(value: dict[str, float]) -> dict[str, float]
```

Check every compartment amount with the rule of the engine.

Source code in `commol/context/initial_conditions.py`

```python
@field_validator("compartment_amounts")
@classmethod
def validate_compartment_amounts(cls, value: dict[str, float]) -> dict[str, float]:
    """Check every compartment amount with the rule of the engine."""
    commol_rs.model.check_compartment_amounts(list(value.items()))
    return value
```

#### get_uncalibrated_bins

```python
get_uncalibrated_bins() -> list[str]
```

Get list of bin IDs that have uncalibrated amounts (value = None).

Returns:

| Type        | Description                                           |
| ----------- | ----------------------------------------------------- |
| `list[str]` | List of bin IDs with uncalibrated initial conditions. |

Source code in `commol/context/initial_conditions.py`

```python
def get_uncalibrated_bins(self) -> list[str]:
    """
    Get list of bin IDs that have uncalibrated amounts (value = None).

    Returns
    -------
    list[str]
        List of bin IDs with uncalibrated initial conditions.
    """
    return [bin_id for bin_id, amount in self.bin_amounts.items() if amount is None]
```

#### get_categories_with_fractions

```python
get_categories_with_fractions() -> set[str]
```

Get the categories that have a declared initial fraction.

Returns:

| Type       | Description                                                                                                                                             |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `set[str]` | Category names collected from stratification_fractions. A category declared on a stratification but absent from the initial conditions is not included. |

Source code in `commol/context/initial_conditions.py`

```python
def get_categories_with_fractions(self) -> set[str]:
    """
    Get the categories that have a declared initial fraction.

    Returns
    -------
    set[str]
        Category names collected from ``stratification_fractions``. A
        category declared on a stratification but absent from the initial
        conditions is not included.
    """
    return {
        category
        for categories in self.stratification_fractions.values()
        for category in categories
    }
```

#### resolve_category

```python
resolve_category(category: str) -> tuple[str, str]
```

Resolve a category id to its stratification and its bare name.

A qualified id (`group:g1`) names the stratification. A bare name must be declared by exactly one stratification.

Parameters:

| Name       | Type  | Description                      | Default    |
| ---------- | ----- | -------------------------------- | ---------- |
| `category` | `str` | A bare or qualified category id. | *required* |

Returns:

| Type              | Description                                       |
| ----------------- | ------------------------------------------------- |
| `tuple[str, str]` | The stratification id and the bare category name. |

Raises:

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `ValueError` | If the category is not found or a bare name is ambiguous. |

Source code in `commol/context/initial_conditions.py`

```python
def resolve_category(self, category: str) -> tuple[str, str]:
    """
    Resolve a category id to its stratification and its bare name.

    A qualified id (``group:g1``) names the stratification. A bare name
    must be declared by exactly one stratification.

    Parameters
    ----------
    category : str
        A bare or qualified category id.

    Returns
    -------
    tuple[str, str]
        The stratification id and the bare category name.

    Raises
    ------
    ValueError
        If the category is not found or a bare name is ambiguous.
    """
    stratification_id, separator, name = category.rpartition(CATEGORY_SEPARATOR)
    owners = [
        stratification
        for stratification, categories in self.stratification_fractions.items()
        if (not separator or stratification == stratification_id)
        and name in categories
    ]
    if len(owners) > 1:
        qualified = " or ".join(
            f"'{owner}{CATEGORY_SEPARATOR}{name}'" for owner in owners
        )
        raise ValueError(
            f"Category '{category}' is ambiguous: it is declared by "
            f"stratifications {owners}. Qualify it as {qualified}."
        )
    if not owners:
        raise ValueError(
            f"Category '{category}' not found in initial conditions. "
            f"Available categories: {sorted(self.get_categories_with_fractions())}"
        )
    return owners[0], name
```

#### category_id

```python
category_id(stratification: str, category: str) -> str
```

Shortest id that names a category without ambiguity.

Parameters:

| Name             | Type  | Description                                          | Default    |
| ---------------- | ----- | ---------------------------------------------------- | ---------- |
| `stratification` | `str` | Id of the stratification that declares the category. | *required* |
| `category`       | `str` | Bare category name.                                  | *required* |

Returns:

| Type  | Description                                                                                                 |
| ----- | ----------------------------------------------------------------------------------------------------------- |
| `str` | The bare name, or stratification:category when another stratification declares a category of the same name. |

Source code in `commol/context/initial_conditions.py`

```python
def category_id(self, stratification: str, category: str) -> str:
    """
    Shortest id that names a category without ambiguity.

    Parameters
    ----------
    stratification : str
        Id of the stratification that declares the category.
    category : str
        Bare category name.

    Returns
    -------
    str
        The bare name, or ``stratification:category`` when another
        stratification declares a category of the same name.
    """
    declaring = sum(
        category in categories
        for categories in self.stratification_fractions.values()
    )
    if declaring > 1:
        return f"{stratification}{CATEGORY_SEPARATOR}{category}"
    return category
```

#### get_category_fraction

```python
get_category_fraction(category: str) -> float
```

Get the fraction of a category within the group it subdivides.

Parameters:

| Name       | Type  | Description                                  | Default    |
| ---------- | ----- | -------------------------------------------- | ---------- |
| `category` | `str` | A category id, bare or qualified (group:g1). | *required* |

Returns:

| Type    | Description                                                                                                           |
| ------- | --------------------------------------------------------------------------------------------------------------------- |
| `float` | The declared fraction, relative to the group the category's stratification applies to rather than to the whole model. |

Raises:

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `ValueError` | If the category is not found or a bare name is ambiguous. |

Source code in `commol/context/initial_conditions.py`

```python
def get_category_fraction(self, category: str) -> float:
    """
    Get the fraction of a category within the group it subdivides.

    Parameters
    ----------
    category : str
        A category id, bare or qualified (``group:g1``).

    Returns
    -------
    float
        The declared fraction, relative to the group the category's
        stratification applies to rather than to the whole model.

    Raises
    ------
    ValueError
        If the category is not found or a bare name is ambiguous.
    """
    stratification, name = self.resolve_category(category)
    return self.stratification_fractions[stratification][name]
```

#### update_stratification_fractions

```python
update_stratification_fractions(fractions: Mapping[str, float]) -> None
```

Update stratification category fractions by category name.

Categories belonging to several stratifications may be given in one call. A category name declared by more than one stratification must be qualified with its stratification, as in `group:g1`. Within each stratification, at most one category may be omitted; the omitted category receives the rest of 1, or 0 when the given values pass 1. The fractions of every stratification are checked with the rule of the engine before any of them changes.

Parameters:

| Name        | Type                  | Description                                                                         | Default    |
| ----------- | --------------------- | ----------------------------------------------------------------------------------- | ---------- |
| `fractions` | `Mapping[str, float]` | Dictionary mapping category names, bare or qualified, to their new fraction values. | *required* |

Raises:

| Type         | Description                                                                                                                                                                                                                            |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a category is not found or is ambiguous, or a stratification has more than one category omitted; with the message of the engine, if a fraction is not from 0 to 1 or the fractions of a stratification do not sum to 1 within 1e-6. |

Source code in `commol/context/initial_conditions.py`

```python
def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
    """
    Update stratification category fractions by category name.

    Categories belonging to several stratifications may be given in one
    call. A category name declared by more than one stratification must be
    qualified with its stratification, as in ``group:g1``. Within each
    stratification, at most one category may be omitted; the omitted
    category receives the rest of 1, or 0 when the given values pass 1.
    The fractions of every stratification are checked with the rule of the
    engine before any of them changes.

    Parameters
    ----------
    fractions : Mapping[str, float]
        Dictionary mapping category names, bare or qualified, to their new
        fraction values.

    Raises
    ------
    ValueError
        If a category is not found or is ambiguous, or a stratification
        has more than one category omitted; with the message of the
        engine, if a fraction is not from 0 to 1 or the fractions of a
        stratification do not sum to 1 within 1e-6.
    """
    by_stratification: dict[str, dict[str, float]] = {}
    for category, value in fractions.items():
        stratification, name = self.resolve_category(category)
        by_stratification.setdefault(stratification, {})[name] = value

    updated = {
        stratification: self._updated_fractions(stratification, named)
        for stratification, named in by_stratification.items()
    }
    _check_fractions(updated)
    for stratification, categories in updated.items():
        self.stratification_fractions[stratification].update(categories)
```

#### update_bin_amounts

```python
update_bin_amounts(amounts: Mapping[str, float | None]) -> None
```

Update bin amounts by bin ID.

Every amount is checked before any bin changes, so a rejected call leaves the initial conditions unchanged. The bins of one unit keep no sum: to move an amount between bins, update every bin involved.

Parameters:

| Name      | Type                  | Description | Default                                                                                                          |
| --------- | --------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `amounts` | \`Mapping\[str, float | None\]\`    | Dictionary mapping bin IDs to their new amount, in the unit of the bin. None marks a bin that needs calibration. |

Raises:

| Type         | Description                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a bin ID is not found in the initial conditions, or, with the message of the engine, an amount is not finite or is negative. |

Source code in `commol/context/initial_conditions.py`

```python
def update_bin_amounts(self, amounts: Mapping[str, float | None]) -> None:
    """
    Update bin amounts by bin ID.

    Every amount is checked before any bin changes, so a rejected call
    leaves the initial conditions unchanged. The bins of one unit keep no
    sum: to move an amount between bins, update every bin involved.

    Parameters
    ----------
    amounts : Mapping[str, float | None]
        Dictionary mapping bin IDs to their new amount, in the unit of the
        bin. None marks a bin that needs calibration.

    Raises
    ------
    ValueError
        If a bin ID is not found in the initial conditions, or, with the
        message of the engine, an amount is not finite or is negative.
    """
    unknown = sorted(set(amounts) - set(self.bin_amounts))
    if unknown:
        raise ValueError(
            f"Bin(s) {unknown} not found in initial conditions. "
            f"Available bins: {list(self.bin_amounts)}"
        )
    commol_rs.model.check_bin_amounts(list(amounts.items()))
    self.bin_amounts.update(amounts)
```
