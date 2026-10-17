# Model Structure

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

A `Model` is a tree of Pydantic models. `ModelBuilder` creates it, `Model.to_json` writes it, and the Rust engine reads the same tree from JSON (`crates/commol-core/src/types/`). Most users never build these types by hand; read this page to understand model files or to work on Commol.

A model has a header, a **process** written once, a **structure** that lays the process out, and an **engine** that runs it. The process (bins, accumulators, stratifications, transitions) is the same for every kind of model; the structure changes between kinds and is tagged in JSON by `kind`, and by `variant` for a narrower form of a structure. The kind today is `aggregate`: amounts per compartment.

```text
Model                      name, description, version, step_unit
├── parameters: [Parameter]                (a value may be a SeriesValue)
├── engine: ModelEngine
│   ├── type: "DifferenceEquations", "DifferentialEquations",
│   │         "StochasticEvents" or "StochasticSteps"
│   └── settings: OdeSolver | StochasticEventsSettings | StochasticStepsSettings | None
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
    ├── total: float
    ├── bin_fractions: {bin: fraction | None}
    ├── stratification_fractions: {stratification: {category: fraction}}
    ├── compartment_fractions: {compartment: fraction}
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
    "total": 1000.0,
    "bin_fractions": {"A": 0.99, "B": 0.01},
    "stratification_fractions": {},
    "compartment_fractions": {},
    "neighbors": null
  },
  "step_unit": null
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

Source code in `commol/context/bin.py`

```python
class Accumulator(BaseModel):
    """A cumulative event counter tracked as output but not as part of the state."""

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
        if not value.strip():
            raise ValueError("Formula cannot be empty")
        return value.strip()

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
    if not value.strip():
        raise ValueError("Formula cannot be empty")
    return value.strip()
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

## SeriesValue

Bases: `BaseModel`

An empirical series for a parameter value.

Attributes:

| Name   | Type                        | Description                                                                                                                                                                                                                                                                                                                                             |
| ------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data` | `list[tuple[float, float]]` | (position, value) points on the axis. Need not be pre-sorted; the engine sorts them.                                                                                                                                                                                                                                                                    |
| `mode` | `str`                       | How the series is read between its points: "pulse" (the value on the step of a point, 0 on every other step; integer positions only), "step_function" (the value of the last point at or before t), or "linear" (linear interpolation in t). The engine decides which positions its model type accepts: a DifferenceEquations model only integer steps. |

Source code in `commol/context/parameter.py`

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
        which positions its model type accepts: a DifferenceEquations model
        only integer steps.
    """

    model_config = ConfigDict(extra="forbid")

    data: list[tuple[float, float]]
    mode: str = Field(default="pulse", pattern="^(pulse|step_function|linear)$")

    @field_validator("data")
    @classmethod
    def validate_data(
        cls, value: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Validate series points before they reach the Rust engine."""
        if not value:
            raise ValueError("Series data must contain at least one point")

        positions: list[float] = []
        for position, point_value in value:
            if not (math.isfinite(position) and position >= 0):
                raise ValueError("Series positions must be finite and non-negative")
            if not math.isfinite(point_value):
                raise ValueError("Series values must be finite")
            positions.append(position)

        if len(positions) != len(set(positions)):
            raise ValueError("Series positions must be unique")

        return value
```

### Methods:

#### validate_data

```python
validate_data(value: list[tuple[float, float]]) -> list[tuple[float, float]]
```

Validate series points before they reach the Rust engine.

Source code in `commol/context/parameter.py`

```python
@field_validator("data")
@classmethod
def validate_data(
    cls, value: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Validate series points before they reach the Rust engine."""
    if not value:
        raise ValueError("Series data must contain at least one point")

    positions: list[float] = []
    for position, point_value in value:
        if not (math.isfinite(position) and position >= 0):
            raise ValueError("Series positions must be finite and non-negative")
        if not math.isfinite(point_value):
            raise ValueError("Series values must be finite")
        positions.append(position)

    if len(positions) != len(set(positions)):
        raise ValueError("Series positions must be unique")

    return value
```

## Transitions

## Transition

Bases: `BaseModel`

Defines a rule for system evolution.

Attributes:

| Name               | Type                   | Description                   |
| ------------------ | ---------------------- | ----------------------------- |
| `id`               | `str`                  | Id of the transition.         |
| `source`           | `list[str]`            | The origin compartments.      |
| `target`           | `list[str]`            | The destination compartments. |
| `rate`             | \`str                  | None\`                        |
| `stratified_rates` | \`list[StratifiedRate] | None\`                        |

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
    rate : str | None
        Default mathematical formula, parameter name, or constant value for the flow.
        Used when no stratified rate matches. Numeric values are automatically
        converted to strings during validation.

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
            "or a mathematical expression (e.g., 'k1 * A * B / total'). "
            "Numeric values are automatically converted to strings during validation."
        ),
    )

    stratified_rates: list[StratifiedRate] | None = Field(
        default=None, description="List of stratification-specific rates"
    )

    per_compartment: bool | None = Field(
        default=None,
        description=(
            "When True, base compartment names in the rate expression "
            "are replaced with the specific stratified compartment name "
            "for each expanded transition flow."
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
        """Check the impulse schedule: one kind, valid positions and period."""
        if self.impulse_at is not None and self.impulse_every is not None:
            raise ValueError(
                f"Transition '{self.id}': set impulse_at or impulse_every, not both."
            )
        if self.impulse_at is not None:
            if not self.impulse_at:
                raise ValueError(
                    f"Transition '{self.id}': impulse_at needs at least one position."
                )
            bad = [p for p in self.impulse_at if not math.isfinite(p) or p < 0]
            if bad:
                raise ValueError(
                    f"Transition '{self.id}': impulse positions must be finite "
                    f"and not negative, got {bad}."
                )
        if self.impulse_every is not None:
            period, offset = self.impulse_every
            if not (math.isfinite(period) and period > 0):
                raise ValueError(
                    f"Transition '{self.id}': the impulse period must be "
                    f"positive, got {period}."
                )
            if not (math.isfinite(offset) and offset >= 0):
                raise ValueError(
                    f"Transition '{self.id}': the impulse offset must be finite "
                    f"and not negative, got {offset}."
                )
        return self

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, value: str | None) -> str | None:
        """
        Convert numeric rates to strings and perform security and syntax validation.
        """
        if value is None:
            return value
        try:
            validate_expression_security(value)
            commol_rs.core.MathExpression(value).validate()
        except ValueError as e:
            raise ValueError(f"Validation failed for rate '{value}': {e}")
        return value
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

Check the impulse schedule: one kind, valid positions and period.

Source code in `commol/context/transition.py`

```python
@model_validator(mode="after")
def validate_impulse(self) -> Self:
    """Check the impulse schedule: one kind, valid positions and period."""
    if self.impulse_at is not None and self.impulse_every is not None:
        raise ValueError(
            f"Transition '{self.id}': set impulse_at or impulse_every, not both."
        )
    if self.impulse_at is not None:
        if not self.impulse_at:
            raise ValueError(
                f"Transition '{self.id}': impulse_at needs at least one position."
            )
        bad = [p for p in self.impulse_at if not math.isfinite(p) or p < 0]
        if bad:
            raise ValueError(
                f"Transition '{self.id}': impulse positions must be finite "
                f"and not negative, got {bad}."
            )
    if self.impulse_every is not None:
        period, offset = self.impulse_every
        if not (math.isfinite(period) and period > 0):
            raise ValueError(
                f"Transition '{self.id}': the impulse period must be "
                f"positive, got {period}."
            )
        if not (math.isfinite(offset) and offset >= 0):
            raise ValueError(
                f"Transition '{self.id}': the impulse offset must be finite "
                f"and not negative, got {offset}."
            )
    return self
```

#### validate_rate

```python
validate_rate(value: str | None) -> str | None
```

Convert numeric rates to strings and perform security and syntax validation.

Source code in `commol/context/transition.py`

```python
@field_validator("rate", mode="before")
@classmethod
def validate_rate(cls, value: str | None) -> str | None:
    """
    Convert numeric rates to strings and perform security and syntax validation.
    """
    if value is None:
        return value
    try:
        validate_expression_security(value)
        commol_rs.core.MathExpression(value).validate()
    except ValueError as e:
        raise ValueError(f"Validation failed for rate '{value}': {e}")
    return value
```

## StratifiedRate

Bases: `BaseModel`

Defines a rate for specific stratification categories.

Attributes:

| Name         | Type                            | Description                                                |
| ------------ | ------------------------------- | ---------------------------------------------------------- |
| `conditions` | `list[StratificationCondition]` | List of stratification-category pairs that must match      |
| `rate`       | `str`                           | Rate expression for compartments matching these conditions |

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

    @field_validator("rate", mode="before")
    @classmethod
    def validate_rate(cls, value: str) -> str:
        """Perform security and syntax validation on the rate expression."""
        try:
            validate_expression_security(value)
            commol_rs.core.MathExpression(value).validate()
        except ValueError as e:
            raise ValueError(f"Validation failed for rate '{value}': {e}")
        return value
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

Perform security and syntax validation on the rate expression.

Source code in `commol/context/transition.py`

```python
@field_validator("rate", mode="before")
@classmethod
def validate_rate(cls, value: str) -> str:
    """Perform security and syntax validation on the rate expression."""
    try:
        validate_expression_security(value)
        commol_rs.core.MathExpression(value).validate()
    except ValueError as e:
        raise ValueError(f"Validation failed for rate '{value}': {e}")
    return value
```

## Engine

The engine's list of engines (`crates/commol-core/src/types/engine.rs`) validates the settings: settings of another engine raise an error that names both. `OdeSolver` is documented in [Simulation API](https://munqu.github.io/commol/latest/api/simulation/index.md).

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
        ``DifferentialEquations`` takes `OdeSolver`, ``StochasticEvents``
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
    def settings_of_the_engine(cls, value: Any, info: ValidationInfo) -> Any:
        """Read settings given as a mapping with the class of their engine."""
        settings_class = _SETTINGS.get(info.data.get("type", ""))
        if isinstance(value, dict) and settings_class is not None:
            return settings_class.model_validate(value)
        return value

    @model_validator(mode="after")
    def check_settings(self) -> Self:
        """Ask the engine's list whether it has the engine and the settings."""
        settings_json = (
            None if self.settings is None else self.settings.model_dump_json()
        )
        commol_rs.engine.EngineType(self.type).check_settings(settings_json)
        return self
```

### Methods:

#### settings_of_the_engine

```python
settings_of_the_engine(value: Any, info: ValidationInfo) -> Any
```

Read settings given as a mapping with the class of their engine.

Source code in `commol/context/engine.py`

```python
@field_validator("settings", mode="before")
@classmethod
def settings_of_the_engine(cls, value: Any, info: ValidationInfo) -> Any:
    """Read settings given as a mapping with the class of their engine."""
    settings_class = _SETTINGS.get(info.data.get("type", ""))
    if isinstance(value, dict) and settings_class is not None:
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
    settings_json = (
        None if self.settings is None else self.settings.model_dump_json()
    )
    commol_rs.engine.EngineType(self.type).check_settings(settings_json)
    return self
```

## Initial conditions

`Aggregate` holds the fields and methods of `InitialConditions`, and checks them against the process when the model is built.

## InitialConditions

Bases: `BaseModel`

How much starts where: the total and the fractions of it.

`Aggregate` holds these fields and their methods; the checks against the bins and the stratifications of the process are in `Aggregate.check_model`.

Attributes:

| Name                       | Type                          | Description                                                                                                                                 |
| -------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `total`                    | `float`                       | The sum of every compartment at the start, finite and greater than 0. It can be a count, a mass or a concentration.                         |
| `bin_fractions`            | \`dict\[str, float            | None\]\`                                                                                                                                    |
| `stratification_fractions` | `dict[str, dict[str, float]]` | The fraction of each category, by stratification id and category. A conditional stratification splits only the group its conditions select. |
| `compartment_fractions`    | `dict[str, float]`            | Fractions of the total in single expanded compartments, by compartment name, applied after the expansion.                                   |

Source code in `commol/context/initial_conditions.py`

```python
class InitialConditions(BaseModel):
    """
    How much starts where: the total and the fractions of it.

    `Aggregate` holds these fields and their methods; the checks against the
    bins and the stratifications of the process are in
    `Aggregate.check_model`.

    Attributes
    ----------
    total : float
        The sum of every compartment at the start, finite and greater than 0.
        It can be a count, a mass or a concentration.
    bin_fractions : dict[str, float | None]
        The fraction of the total in each bin, by bin id. None leaves a
        fraction for calibration.
    stratification_fractions : dict[str, dict[str, float]]
        The fraction of each category, by stratification id and category.
        A conditional stratification splits only the group its conditions
        select.
    compartment_fractions : dict[str, float]
        Fractions of the total in single expanded compartments, by compartment
        name, applied after the expansion.
    """

    model_config = ConfigDict(extra="forbid")

    total: float = Field(
        ...,
        description="The sum of every compartment at the start, finite and > 0.",
    )
    bin_fractions: dict[str, float | None] = Field(
        ...,
        description=(
            "The fraction of the total in each bin, by bin id. None leaves a "
            "fraction for calibration."
        ),
    )
    stratification_fractions: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="The fraction of each category, by stratification id.",
    )
    compartment_fractions: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Fractions of the total in single expanded compartments, applied "
            "after the expansion."
        ),
    )

    @field_validator("total")
    @classmethod
    def validate_total(cls, value: float) -> float:
        """Enforce a finite total greater than 0.

        Every fraction of a zero total is zero, and the report divides by the
        total, so a zero, negative or non-finite total is rejected as a domain
        error.
        """
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"total must be finite and greater than 0, got {value}.")
        return value

    @field_validator("compartment_fractions")
    @classmethod
    def validate_compartment_fraction_range(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        """Enforce that every compartment fraction lies within ``[0.0, 1.0]``."""
        for compartment, fraction in value.items():
            _check_fraction(f"compartment '{compartment}'", fraction)
        return value

    @staticmethod
    def _validate_bin_fraction_total(candidate: Mapping[str, float | None]) -> None:
        """
        Enforce the bin-fraction conservation rule on a complete mapping.

        Uncalibrated (``None``) fractions floor the calibrated total strictly
        below 1.0 so they can be filled later. When every bin is calibrated,
        the fractions must sum to 1.0, because they are shares of a single
        total. Construction and updates apply the same rule, so an update
        never produces a definition that cannot be loaded again.

        Parameters
        ----------
        candidate : Mapping[str, float | None]
            Fraction of every bin.

        Raises
        ------
        ValueError
            If the fractions break the rule.
        """
        calibrated = [value for value in candidate.values() if value is not None]
        if not calibrated:
            return
        total = sum(calibrated)
        uncalibrated = sum(1 for value in candidate.values() if value is None)

        if uncalibrated > 0:
            if total >= 1.0:
                raise ValueError(
                    f"Calibrated bin fractions sum to {total:.4f}, but must be "
                    f"LESS than 1.0 to leave room for {uncalibrated} "
                    f"uncalibrated fraction(s). Calibrated fractions: "
                    f"{[item for item in candidate.items() if item[1] is not None]}"
                )
        elif not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(
                f"Bin fractions must sum to 1.0, got {total:.4f}. "
                f"Fractions: {list(candidate.values())}"
            )

    @model_validator(mode="after")
    def validate_calibrated_fractions_sum_to_one(self) -> Self:
        """
        Validate that bin fractions sum appropriately.

        Rules:
        - If some fractions are None (uncalibrated): calibrated ones must sum to LESS
            than 1.0
        - If all fractions are calibrated (no None): must sum to exactly 1.0
        - If all fractions are None: skip validation (will be set before simulation)
        """
        self._validate_bin_fraction_total(self.bin_fractions)
        return self

    def get_uncalibrated_bins(self) -> list[str]:
        """
        Get list of bin IDs that have uncalibrated fractions (value = None).

        Returns
        -------
        list[str]
            List of bin IDs with uncalibrated initial conditions.
        """
        return [
            bin_id
            for bin_id, fraction in self.bin_fractions.items()
            if fraction is None
        ]

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

    def _fraction_subgroup_total(self, categories: Iterable[str] = ()) -> float:
        """
        Get the initial total of a subgroup from the fractions alone.

        Category fractions are relative to the group their stratification
        applies to, so the fractions of a nested chain of categories multiply.
        Passing no categories gives the total. Per-compartment overrides are
        not included; `Aggregate.subgroup_total` includes them.
        """
        total = self.total
        for category in categories:
            total *= self.get_category_fraction(category)
        return total

    def _apply_stratification_update(
        self, stratification: str, named: Mapping[str, float]
    ) -> None:
        """Write the fractions of a single stratification."""
        categories = self.stratification_fractions[stratification]
        unnamed = [category for category in categories if category not in named]
        total = sum(named.values())

        if len(unnamed) > 1:
            raise ValueError(
                f"Cannot update stratification '{stratification}': "
                f"categories {sorted(unnamed)} were left unnamed. At most one "
                f"category may be omitted, so that it can take the remaining "
                f"fraction."
            )
        if not unnamed and not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Fractions for stratification '{stratification}' "
                f"must sum to 1.0, got {total:.6f}. Values: {dict(named)}"
            )
        if unnamed and total > 1.0 + 1e-6:
            raise ValueError(
                f"Fractions for stratification '{stratification}' "
                f"sum to {total:.6f}, which would leave a negative fraction for "
                f"'{unnamed[0]}'. Values: {dict(named)}"
            )

        remainder = max(0.0, 1.0 - total)
        for category in categories:
            categories[category] = named.get(category, remainder)

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update stratification category fractions by category name.

        Categories belonging to several stratifications may be given in one
        call. A category name declared by more than one stratification must be
        qualified with its stratification, as in ``group:g1``. Within each
        stratification, at most one category may be omitted; the omitted
        category receives the remaining fraction so that the stratification
        sums to 1.0. When every category is given, the values
        must already sum to 1.0.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping category names, bare or qualified, to their new
            fraction values. Each value must lie between 0.0 and 1.0.

        Raises
        ------
        ValueError
            If a category is not found or is ambiguous, a value lies outside
            [0.0, 1.0], a stratification has more than one category omitted,
            fully specified values do not sum to 1.0, or the given values
            exceed 1.0.
        """
        for category, value in fractions.items():
            _check_fraction(f"category '{category}'", value)

        by_stratification: dict[str, dict[str, float]] = {}
        for category, value in fractions.items():
            stratification, name = self.resolve_category(category)
            by_stratification.setdefault(stratification, {})[name] = value

        for stratification, named in by_stratification.items():
            self._apply_stratification_update(stratification, named)

    def _upsert_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_fractions`` has its
        fraction replaced; a new one is appended. Compartment names and the
        sum of the amounts depend on the expansion of the process, so
        :meth:`~commol.context.aggregate.Aggregate.update_compartment_fractions`
        validates them against the engine, not here.

        Raises
        ------
        ValueError
            If a value lies outside ``[0.0, 1.0]``.
        """
        for compartment, value in fractions.items():
            _check_fraction(f"compartment '{compartment}'", value)
        self.compartment_fractions.update(fractions)

    def update_bin_fractions(self, fractions: Mapping[str, float | None]) -> None:
        """
        Update bin fractions by bin ID.

        The candidate fractions are validated before any bin is mutated, so a
        rejected call leaves the initial conditions unchanged. The conservation
        rule of :meth:`validate_calibrated_fractions_sum_to_one` is re-applied
        to the resulting mapping: the bin fractions must sum to 1.0, while
        uncalibrated (``None``) bins reserve the remainder. To move an amount
        between bins, update every bin involved in one call.

        Parameters
        ----------
        fractions : Mapping[str, float | None]
            Dictionary mapping bin IDs to their new fraction values.
            None values indicate bins that need calibration.

        Raises
        ------
        ValueError
            If a bin ID is not found in the initial conditions, or the
            resulting fractions violate the conservation rule.
        """
        unknown = sorted(set(fractions) - set(self.bin_fractions))
        if unknown:
            raise ValueError(
                f"Bin(s) {unknown} not found in initial conditions. "
                f"Available bins: {list(self.bin_fractions)}"
            )

        candidate = {**self.bin_fractions, **fractions}
        self._validate_bin_fraction_total(candidate)
        self.bin_fractions.update(fractions)
```

### Methods:

#### validate_total

```python
validate_total(value: float) -> float
```

Enforce a finite total greater than 0.

Every fraction of a zero total is zero, and the report divides by the total, so a zero, negative or non-finite total is rejected as a domain error.

Source code in `commol/context/initial_conditions.py`

```python
@field_validator("total")
@classmethod
def validate_total(cls, value: float) -> float:
    """Enforce a finite total greater than 0.

    Every fraction of a zero total is zero, and the report divides by the
    total, so a zero, negative or non-finite total is rejected as a domain
    error.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"total must be finite and greater than 0, got {value}.")
    return value
```

#### validate_compartment_fraction_range

```python
validate_compartment_fraction_range(value: dict[str, float]) -> dict[str, float]
```

Enforce that every compartment fraction lies within `[0.0, 1.0]`.

Source code in `commol/context/initial_conditions.py`

```python
@field_validator("compartment_fractions")
@classmethod
def validate_compartment_fraction_range(
    cls, value: dict[str, float]
) -> dict[str, float]:
    """Enforce that every compartment fraction lies within ``[0.0, 1.0]``."""
    for compartment, fraction in value.items():
        _check_fraction(f"compartment '{compartment}'", fraction)
    return value
```

#### validate_calibrated_fractions_sum_to_one

```python
validate_calibrated_fractions_sum_to_one() -> Self
```

Validate that bin fractions sum appropriately.

Rules:

- If some fractions are None (uncalibrated): calibrated ones must sum to LESS than 1.0
- If all fractions are calibrated (no None): must sum to exactly 1.0
- If all fractions are None: skip validation (will be set before simulation)

Source code in `commol/context/initial_conditions.py`

```python
@model_validator(mode="after")
def validate_calibrated_fractions_sum_to_one(self) -> Self:
    """
    Validate that bin fractions sum appropriately.

    Rules:
    - If some fractions are None (uncalibrated): calibrated ones must sum to LESS
        than 1.0
    - If all fractions are calibrated (no None): must sum to exactly 1.0
    - If all fractions are None: skip validation (will be set before simulation)
    """
    self._validate_bin_fraction_total(self.bin_fractions)
    return self
```

#### get_uncalibrated_bins

```python
get_uncalibrated_bins() -> list[str]
```

Get list of bin IDs that have uncalibrated fractions (value = None).

Returns:

| Type        | Description                                           |
| ----------- | ----------------------------------------------------- |
| `list[str]` | List of bin IDs with uncalibrated initial conditions. |

Source code in `commol/context/initial_conditions.py`

```python
def get_uncalibrated_bins(self) -> list[str]:
    """
    Get list of bin IDs that have uncalibrated fractions (value = None).

    Returns
    -------
    list[str]
        List of bin IDs with uncalibrated initial conditions.
    """
    return [
        bin_id
        for bin_id, fraction in self.bin_fractions.items()
        if fraction is None
    ]
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

Categories belonging to several stratifications may be given in one call. A category name declared by more than one stratification must be qualified with its stratification, as in `group:g1`. Within each stratification, at most one category may be omitted; the omitted category receives the remaining fraction so that the stratification sums to 1.0. When every category is given, the values must already sum to 1.0.

Parameters:

| Name        | Type                  | Description                                                                                                                  | Default    |
| ----------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `fractions` | `Mapping[str, float]` | Dictionary mapping category names, bare or qualified, to their new fraction values. Each value must lie between 0.0 and 1.0. | *required* |

Raises:

| Type         | Description                                                                                                                                                                                                 |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a category is not found or is ambiguous, a value lies outside [0.0, 1.0], a stratification has more than one category omitted, fully specified values do not sum to 1.0, or the given values exceed 1.0. |

Source code in `commol/context/initial_conditions.py`

```python
def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
    """
    Update stratification category fractions by category name.

    Categories belonging to several stratifications may be given in one
    call. A category name declared by more than one stratification must be
    qualified with its stratification, as in ``group:g1``. Within each
    stratification, at most one category may be omitted; the omitted
    category receives the remaining fraction so that the stratification
    sums to 1.0. When every category is given, the values
    must already sum to 1.0.

    Parameters
    ----------
    fractions : Mapping[str, float]
        Dictionary mapping category names, bare or qualified, to their new
        fraction values. Each value must lie between 0.0 and 1.0.

    Raises
    ------
    ValueError
        If a category is not found or is ambiguous, a value lies outside
        [0.0, 1.0], a stratification has more than one category omitted,
        fully specified values do not sum to 1.0, or the given values
        exceed 1.0.
    """
    for category, value in fractions.items():
        _check_fraction(f"category '{category}'", value)

    by_stratification: dict[str, dict[str, float]] = {}
    for category, value in fractions.items():
        stratification, name = self.resolve_category(category)
        by_stratification.setdefault(stratification, {})[name] = value

    for stratification, named in by_stratification.items():
        self._apply_stratification_update(stratification, named)
```

#### update_bin_fractions

```python
update_bin_fractions(fractions: Mapping[str, float | None]) -> None
```

Update bin fractions by bin ID.

The candidate fractions are validated before any bin is mutated, so a rejected call leaves the initial conditions unchanged. The conservation rule of :meth:`validate_calibrated_fractions_sum_to_one` is re-applied to the resulting mapping: the bin fractions must sum to 1.0, while uncalibrated (`None`) bins reserve the remainder. To move an amount between bins, update every bin involved in one call.

Parameters:

| Name        | Type                  | Description | Default                                                                                                   |
| ----------- | --------------------- | ----------- | --------------------------------------------------------------------------------------------------------- |
| `fractions` | \`Mapping\[str, float | None\]\`    | Dictionary mapping bin IDs to their new fraction values. None values indicate bins that need calibration. |

Raises:

| Type         | Description                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a bin ID is not found in the initial conditions, or the resulting fractions violate the conservation rule. |

Source code in `commol/context/initial_conditions.py`

```python
def update_bin_fractions(self, fractions: Mapping[str, float | None]) -> None:
    """
    Update bin fractions by bin ID.

    The candidate fractions are validated before any bin is mutated, so a
    rejected call leaves the initial conditions unchanged. The conservation
    rule of :meth:`validate_calibrated_fractions_sum_to_one` is re-applied
    to the resulting mapping: the bin fractions must sum to 1.0, while
    uncalibrated (``None``) bins reserve the remainder. To move an amount
    between bins, update every bin involved in one call.

    Parameters
    ----------
    fractions : Mapping[str, float | None]
        Dictionary mapping bin IDs to their new fraction values.
        None values indicate bins that need calibration.

    Raises
    ------
    ValueError
        If a bin ID is not found in the initial conditions, or the
        resulting fractions violate the conservation rule.
    """
    unknown = sorted(set(fractions) - set(self.bin_fractions))
    if unknown:
        raise ValueError(
            f"Bin(s) {unknown} not found in initial conditions. "
            f"Available bins: {list(self.bin_fractions)}"
        )

    candidate = {**self.bin_fractions, **fractions}
    self._validate_bin_fraction_total(candidate)
    self.bin_fractions.update(fractions)
```
