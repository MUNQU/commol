# Model Structure

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

A `Model` is a tree of Pydantic models. `ModelBuilder` creates it, `Model.to_json` writes it, and the Rust engine reads the same structure from JSON (`crates/commol-core/src/types/`). Most users never build these types by hand; read this page to understand model files or to work on Commol.

```text
Model                      name, description, version, step_unit
├── population: Population
│   ├── bins: [Bin]
│   ├── accumulators: [Accumulator]
│   ├── stratifications: [Stratification]
│   │   └── conditions: [StratificationCondition]
│   ├── transitions: [Transition]          (the engine reads dynamics.transitions)
│   └── initial_conditions: InitialConditions
│       ├── bin_fractions: [BinFraction]
│       ├── stratification_fractions: [StratificationFractions]
│       │   └── fractions: [StratificationFraction]
│       └── compartment_fractions: [CompartmentFraction]
├── parameters: [Parameter]                (a value may be a TimeSeriesValue)
└── dynamics: Dynamics
    ├── typology: "DifferenceEquations" or "DifferentialEquations"
    ├── solver: OdeSolver | None            (differential models only)
    └── transitions: [Transition]          (impulse_at, impulse_every for impulses)
        └── stratified_rates: [StratifiedRate]
            └── conditions: [StratificationCondition]
```

`Model` itself is documented in [Model API](https://munqu.github.io/commol/latest/api/model/index.md).

## Population

## Population

Bases: `BaseModel`

Defines the compartments, stratifications, and initial conditions of the population.

Attributes:

| Name                 | Type                   | Description                                              |
| -------------------- | ---------------------- | -------------------------------------------------------- |
| `bins`               | `list[Bin]`            | A list of compartments or states that make up the model. |
| `stratifications`    | `list[Stratification]` | A list of categorical subdivisions of the population.    |
| `initial_conditions` | `Initialization`       | Initial state of the subpopulations and stratifications. |

Source code in `commol/context/population.py`

```python
class Population(BaseModel):
    """
    Defines the compartments, stratifications, and initial conditions of the population.

    Attributes
    ----------
    bins : list[Bin]
        A list of compartments or states that make up the model.
    stratifications : list[Stratification]
        A list of categorical subdivisions of the population.
    initial_conditions: Initialization
        Initial state of the subpopulations and stratifications.
    """

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

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        """
        Validates that bin and stratification IDs are unique.
        """
        bin_ids = [ds.id for ds in self.bins]
        if len(bin_ids) != len(set(bin_ids)):
            duplicates = [item for item in set(bin_ids) if bin_ids.count(item) > 1]
            raise ValueError(f"Duplicate bin IDs found: {duplicates}")

        stratification_ids = [s.id for s in self.stratifications]
        if len(stratification_ids) != len(set(stratification_ids)):
            duplicates = [
                item
                for item in set(stratification_ids)
                if stratification_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate stratification IDs found: {duplicates}")

        return self

    @model_validator(mode="after")
    def validate_bin_initial_conditions(self) -> Self:
        """
        Validates initial conditions against the defined model bins.
        """
        initial_conditions = self.initial_conditions

        bins_map = {bin_item.id: bin_item for bin_item in self.bins}

        bin_fractions_dict = {
            bf.bin: bf.fraction for bf in initial_conditions.bin_fractions
        }

        actual_bins = set(bin_fractions_dict.keys())
        expected_bins = set(bins_map.keys())

        if actual_bins != expected_bins:
            missing = expected_bins - actual_bins
            extra = actual_bins - expected_bins
            raise ValueError(
                (
                    f"Initial bin fractions keys must exactly match "
                    f"bin ids. Missing ids: {missing}, Extra ids: {extra}."
                )
            )

        # Only validate sum if all fractions are calibrated (not None)
        # The InitialConditions validator handles the None case
        if all(frac is not None for frac in bin_fractions_dict.values()):
            # Filter out None values before summing (type checker requirement)
            calibrated_fractions = [
                f for f in bin_fractions_dict.values() if f is not None
            ]
            bins_sum_fractions = sum(calibrated_fractions)
            if not math.isclose(bins_sum_fractions, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"Bin fractions must sum to 1.0, but got {bins_sum_fractions:.7f}."
                )

        return self

    @model_validator(mode="after")
    def validate_stratified_rates(self) -> Self:
        """
        Validates that stratified rates reference existing stratifications and
        categories.
        """
        strat_map = {strat.id: strat for strat in self.stratifications}

        for transition in self.transitions:
            if transition.stratified_rates:
                for idx, stratified_rate in enumerate(transition.stratified_rates):
                    for condition in stratified_rate.conditions:
                        # Validate stratification exists
                        if condition.stratification not in strat_map:
                            raise ValueError(
                                (
                                    f"In transition '{transition.id}', stratified rate "
                                    f"{idx}: Stratification "
                                    f"'{condition.stratification}' not found. "
                                    f"Available: {list(strat_map.keys())}"
                                )
                            )

                        # Validate category exists (None = target-only override, skip)
                        strat = strat_map[condition.stratification]
                        if (
                            condition.category is not None
                            and condition.category not in strat.categories
                        ):
                            raise ValueError(
                                (
                                    f"In transition '{transition.id}', stratified rate "
                                    f"{idx}: Category '{condition.category}' not found "
                                    f"in stratification '{condition.stratification}'. "
                                    f"Available: {strat.categories}"
                                )
                            )

        return self

    @model_validator(mode="after")
    def validate_stratification_initial_conditions(self) -> Self:
        """
        Validates initial conditions against the defined model Stratification.
        """
        initial_conditions = self.initial_conditions

        strat_map = {strat.id: strat for strat in self.stratifications}

        actual_strat = {
            sf.stratification for sf in initial_conditions.stratification_fractions
        }
        expected_strat = set(strat_map.keys())

        if actual_strat != expected_strat:
            missing = expected_strat - actual_strat
            extra = actual_strat - expected_strat
            raise ValueError(
                (
                    f"Initial stratification fractions keys must exactly match "
                    f"stratification ids. Missing ids: {missing}, Extra ids: {extra}."
                )
            )

        for strat_fractions in initial_conditions.stratification_fractions:
            strat_id = strat_fractions.stratification
            strat_instance = strat_map[strat_id]

            fractions_dict = {
                sf.category: sf.fraction for sf in strat_fractions.fractions
            }

            categories_expected = set(strat_instance.categories)
            categories_actual = set(fractions_dict.keys())

            if categories_actual != categories_expected:
                missing = categories_expected - categories_actual
                extra = categories_actual - categories_expected
                raise ValueError(
                    (
                        f"Categories for stratification '{strat_id}' must exactly "
                        f"match defined categories in instance '{strat_instance.id}'. "
                        f"Missing categories: {missing}, Extra categories: {extra}."
                    )
                )

            strat_sum_fractions = sum(fractions_dict.values())
            if not math.isclose(strat_sum_fractions, 1.0, abs_tol=1e-6):
                raise ValueError(
                    (
                        f"Stratification fractions for '{strat_id}' must sum to 1.0, "
                        f"but got {strat_sum_fractions:.7}."
                    )
                )

        return self
```

### Methods:

#### validate_bins_not_empty

```python
validate_bins_not_empty(v: list[Bin]) -> list[Bin]
```

Require at least one bin.

Source code in `commol/context/population.py`

```python
@field_validator("bins")
@classmethod
def validate_bins_not_empty(cls, v: list[Bin]) -> list[Bin]:
    """Require at least one bin."""
    if not v:
        raise ValueError("At least one bin must be defined.")
    return v
```

#### validate_unique_ids

```python
validate_unique_ids() -> Self
```

Validates that bin and stratification IDs are unique.

Source code in `commol/context/population.py`

```python
@model_validator(mode="after")
def validate_unique_ids(self) -> Self:
    """
    Validates that bin and stratification IDs are unique.
    """
    bin_ids = [ds.id for ds in self.bins]
    if len(bin_ids) != len(set(bin_ids)):
        duplicates = [item for item in set(bin_ids) if bin_ids.count(item) > 1]
        raise ValueError(f"Duplicate bin IDs found: {duplicates}")

    stratification_ids = [s.id for s in self.stratifications]
    if len(stratification_ids) != len(set(stratification_ids)):
        duplicates = [
            item
            for item in set(stratification_ids)
            if stratification_ids.count(item) > 1
        ]
        raise ValueError(f"Duplicate stratification IDs found: {duplicates}")

    return self
```

#### validate_bin_initial_conditions

```python
validate_bin_initial_conditions() -> Self
```

Validates initial conditions against the defined model bins.

Source code in `commol/context/population.py`

```python
@model_validator(mode="after")
def validate_bin_initial_conditions(self) -> Self:
    """
    Validates initial conditions against the defined model bins.
    """
    initial_conditions = self.initial_conditions

    bins_map = {bin_item.id: bin_item for bin_item in self.bins}

    bin_fractions_dict = {
        bf.bin: bf.fraction for bf in initial_conditions.bin_fractions
    }

    actual_bins = set(bin_fractions_dict.keys())
    expected_bins = set(bins_map.keys())

    if actual_bins != expected_bins:
        missing = expected_bins - actual_bins
        extra = actual_bins - expected_bins
        raise ValueError(
            (
                f"Initial bin fractions keys must exactly match "
                f"bin ids. Missing ids: {missing}, Extra ids: {extra}."
            )
        )

    # Only validate sum if all fractions are calibrated (not None)
    # The InitialConditions validator handles the None case
    if all(frac is not None for frac in bin_fractions_dict.values()):
        # Filter out None values before summing (type checker requirement)
        calibrated_fractions = [
            f for f in bin_fractions_dict.values() if f is not None
        ]
        bins_sum_fractions = sum(calibrated_fractions)
        if not math.isclose(bins_sum_fractions, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Bin fractions must sum to 1.0, but got {bins_sum_fractions:.7f}."
            )

    return self
```

#### validate_stratified_rates

```python
validate_stratified_rates() -> Self
```

Validates that stratified rates reference existing stratifications and categories.

Source code in `commol/context/population.py`

```python
@model_validator(mode="after")
def validate_stratified_rates(self) -> Self:
    """
    Validates that stratified rates reference existing stratifications and
    categories.
    """
    strat_map = {strat.id: strat for strat in self.stratifications}

    for transition in self.transitions:
        if transition.stratified_rates:
            for idx, stratified_rate in enumerate(transition.stratified_rates):
                for condition in stratified_rate.conditions:
                    # Validate stratification exists
                    if condition.stratification not in strat_map:
                        raise ValueError(
                            (
                                f"In transition '{transition.id}', stratified rate "
                                f"{idx}: Stratification "
                                f"'{condition.stratification}' not found. "
                                f"Available: {list(strat_map.keys())}"
                            )
                        )

                    # Validate category exists (None = target-only override, skip)
                    strat = strat_map[condition.stratification]
                    if (
                        condition.category is not None
                        and condition.category not in strat.categories
                    ):
                        raise ValueError(
                            (
                                f"In transition '{transition.id}', stratified rate "
                                f"{idx}: Category '{condition.category}' not found "
                                f"in stratification '{condition.stratification}'. "
                                f"Available: {strat.categories}"
                            )
                        )

    return self
```

#### validate_stratification_initial_conditions

```python
validate_stratification_initial_conditions() -> Self
```

Validates initial conditions against the defined model Stratification.

Source code in `commol/context/population.py`

```python
@model_validator(mode="after")
def validate_stratification_initial_conditions(self) -> Self:
    """
    Validates initial conditions against the defined model Stratification.
    """
    initial_conditions = self.initial_conditions

    strat_map = {strat.id: strat for strat in self.stratifications}

    actual_strat = {
        sf.stratification for sf in initial_conditions.stratification_fractions
    }
    expected_strat = set(strat_map.keys())

    if actual_strat != expected_strat:
        missing = expected_strat - actual_strat
        extra = actual_strat - expected_strat
        raise ValueError(
            (
                f"Initial stratification fractions keys must exactly match "
                f"stratification ids. Missing ids: {missing}, Extra ids: {extra}."
            )
        )

    for strat_fractions in initial_conditions.stratification_fractions:
        strat_id = strat_fractions.stratification
        strat_instance = strat_map[strat_id]

        fractions_dict = {
            sf.category: sf.fraction for sf in strat_fractions.fractions
        }

        categories_expected = set(strat_instance.categories)
        categories_actual = set(fractions_dict.keys())

        if categories_actual != categories_expected:
            missing = categories_expected - categories_actual
            extra = categories_actual - categories_expected
            raise ValueError(
                (
                    f"Categories for stratification '{strat_id}' must exactly "
                    f"match defined categories in instance '{strat_instance.id}'. "
                    f"Missing categories: {missing}, Extra categories: {extra}."
                )
            )

        strat_sum_fractions = sum(fractions_dict.values())
        if not math.isclose(strat_sum_fractions, 1.0, abs_tol=1e-6):
            raise ValueError(
                (
                    f"Stratification fractions for '{strat_id}' must sum to 1.0, "
                    f"but got {strat_sum_fractions:.7}."
                )
            )

    return self
```

## Bin

Bases: `BaseModel`

Defines a single bin (base category) in the compartmental model.

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
    Defines a single bin (base category) in the compartmental model.

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

A cumulative event counter tracked as output but not as population state.

Source code in `commol/context/bin.py`

```python
class Accumulator(BaseModel):
    """A cumulative event counter tracked as output but not as population state."""

    id: str = Field(..., description="Identifier of the accumulator.")
    name: str = Field(..., description="Descriptive, human-readable name.")
```

## Stratifications

## Stratification

Bases: `BaseModel`

Defines a categorical subdivision of the population.

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
    Defines a categorical subdivision of the population.

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

When `to` is set, the matched category in the source compartment is replaced with the `to` category in the target compartment name. This enables cross-category transitions within the same bin (e.g., aging).

Attributes:

| Name             | Type  | Description                                                      |
| ---------------- | ----- | ---------------------------------------------------------------- |
| `stratification` | `str` | The ID of the stratification (e.g., "age", "location")           |
| `category`       | `str` | The category within that stratification (e.g., "young", "urban") |
| `to`             | \`str | None\`                                                           |

Source code in `commol/context/stratification_condition.py`

```python
class StratificationCondition(BaseModel):
    """
    Specifies a category within a stratification for rate matching.

    When ``to`` is set, the matched category in the source compartment is
    replaced with the ``to`` category in the target compartment name. This
    enables cross-category transitions within the same bin (e.g., aging).

    Attributes
    ----------
    stratification : str
        The ID of the stratification (e.g., "age", "location")
    category : str
        The category within that stratification (e.g., "young", "urban")
    to : str | None
        Target category override. When set, the target compartment will use
        this category instead of the source compartment's category for this
        stratification.
    """

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
    value : float | str | TimeSeriesValue | None
        Value of the parameter. Can be:
        - float: A numerical constant value
        - str: A mathematical formula that can reference other parameters,
               special variables (N, N_category, step/t, pi, e), or contain
               mathematical expressions
        - TimeSeriesValue: An empirical time series evaluated via binary search
        - None: Indicates that the parameter needs to be calibrated before use
    description : str | None
        A human-readable description of the parameter.
    unit : str | None
        The unit of the parameter (e.g., "1/day", "dimensionless", "person").
        If None, the parameter has no unit specified.
    """

    id: str = Field(default=..., description="Identifier of the parameter.")
    value: float | str | TimeSeriesValue | None = Field(
        default=...,
        description=(
            "Value of the parameter. Can be a float (constant), "
            "str (formula), TimeSeriesValue (empirical series), "
            "or None (requires calibration)."
        ),
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the parameter."
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the parameter (e.g., '1/day', 'dimensionless', 'person').",
    )

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, value: float | str | TimeSeriesValue | None
    ) -> float | str | TimeSeriesValue | None:
        """Validate the parameter value."""
        if value is None or isinstance(value, TimeSeriesValue):
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
validate_value(value: float | str | TimeSeriesValue | None) -> float | str | TimeSeriesValue | None
```

Validate the parameter value.

Source code in `commol/context/parameter.py`

```python
@field_validator("value")
@classmethod
def validate_value(
    cls, value: float | str | TimeSeriesValue | None
) -> float | str | TimeSeriesValue | None:
    """Validate the parameter value."""
    if value is None or isinstance(value, TimeSeriesValue):
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

## TimeSeriesValue

Bases: `BaseModel`

An empirical time series for a parameter value.

Attributes:

| Name   | Type                      | Description                                                                                                                  |
| ------ | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `data` | `list[tuple[int, float]]` | (step, value) pairs. Need not be pre-sorted; the engine sorts them.                                                          |
| `mode` | `str`                     | Interpolation mode: "pulse" (non-zero only at listed steps), "step_function" (zero-order hold), or "linear" (interpolation). |

Source code in `commol/context/parameter.py`

```python
class TimeSeriesValue(BaseModel):
    """
    An empirical time series for a parameter value.

    Attributes
    ----------
    data : list[tuple[int, float]]
        (step, value) pairs. Need not be pre-sorted; the engine sorts them.
    mode : str
        Interpolation mode: ``"pulse"`` (non-zero only at listed steps),
        ``"step_function"`` (zero-order hold), or ``"linear"`` (interpolation).
    """

    data: list[tuple[int, float]]
    mode: str = Field(default="pulse", pattern="^(pulse|step_function|linear)$")

    @field_validator("data")
    @classmethod
    def validate_data(cls, value: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """Validate time-series points before they reach the Rust engine."""
        if not value:
            raise ValueError("Time-series data must contain at least one point")

        steps: list[int] = []
        for step, amount in value:
            if step < 0:
                raise ValueError("Time-series steps must be non-negative")
            if not math.isfinite(amount):
                raise ValueError("Time-series values must be finite")
            steps.append(step)

        if len(steps) != len(set(steps)):
            raise ValueError("Time-series steps must be unique")

        return value
```

### Methods:

#### validate_data

```python
validate_data(value: list[tuple[int, float]]) -> list[tuple[int, float]]
```

Validate time-series points before they reach the Rust engine.

Source code in `commol/context/parameter.py`

```python
@field_validator("data")
@classmethod
def validate_data(cls, value: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Validate time-series points before they reach the Rust engine."""
    if not value:
        raise ValueError("Time-series data must contain at least one point")

    steps: list[int] = []
    for step, amount in value:
        if step < 0:
            raise ValueError("Time-series steps must be non-negative")
        if not math.isfinite(amount):
            raise ValueError("Time-series values must be finite")
        steps.append(step)

    if len(steps) != len(set(steps)):
        raise ValueError("Time-series steps must be unique")

    return value
```

## Dynamics

## Dynamics

Bases: `BaseModel`

Defines how the system evolves.

Attributes:

| Name          | Type                                                      | Description                        |
| ------------- | --------------------------------------------------------- | ---------------------------------- |
| `typology`    | `Literal['DifferenceEquations', 'DifferentialEquations']` | The type of model.                 |
| `transitions` | `List[Transition]`                                        | A list of rules for state changes. |
| `solver`      | \`OdeSolver                                               | None\`                             |

Source code in `commol/context/dynamics.py`

```python
class Dynamics(BaseModel):
    """
    Defines how the system evolves.

    Attributes
    ----------
    typology : Literal["DifferenceEquations", "DifferentialEquations"]
        The type of model.
    transitions : List[Transition]
        A list of rules for state changes.
    solver : OdeSolver | None
        Solver settings of a differential-equations model; None uses the
        defaults of `OdeSolver`. A difference-equations model has none.
    """

    typology: Literal[
        ModelTypes.DIFFERENCE_EQUATIONS, ModelTypes.DIFFERENTIAL_EQUATIONS
    ]
    transitions: list[Transition]
    solver: OdeSolver | None = Field(
        default=None,
        description="Solver settings of a differential-equations model.",
    )

    @model_validator(mode="after")
    def validate_typology_settings(self) -> Self:
        """
        A solver needs a differential model; a difference model steps only at
        integers, so its impulse positions must be integers.
        """
        if self.typology == ModelTypes.DIFFERENCE_EQUATIONS:
            if self.solver is not None:
                raise ValueError(
                    "Solver settings apply only to a DifferentialEquations model; "
                    "a DifferenceEquations model has no solver."
                )
            for transition in self.transitions:
                bad = [p for p in transition.impulse_positions() if p != int(p)]
                if bad:
                    raise ValueError(
                        f"Transition '{transition.id}': impulse position {bad[0]} "
                        f"is not an integer step; a DifferenceEquations model "
                        f"steps only at integers."
                    )
        return self

    @field_validator("transitions")
    @classmethod
    def validate_transitions_not_empty(cls, v: list[Transition]) -> list[Transition]:
        """Require at least one transition."""
        if not v:
            raise ValueError("At least one transition must be defined.")
        return v

    @model_validator(mode="after")
    def validate_unique_transition_ids(self) -> Self:
        """
        Validates that transition IDs are unique.
        """
        transition_ids = [t.id for t in self.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            duplicates = [
                item for item in set(transition_ids) if transition_ids.count(item) > 1
            ]
            raise ValueError(f"Duplicate transition IDs found: {duplicates}")
        return self
```

### Methods:

#### validate_typology_settings

```python
validate_typology_settings() -> Self
```

A solver needs a differential model; a difference model steps only at integers, so its impulse positions must be integers.

Source code in `commol/context/dynamics.py`

```python
@model_validator(mode="after")
def validate_typology_settings(self) -> Self:
    """
    A solver needs a differential model; a difference model steps only at
    integers, so its impulse positions must be integers.
    """
    if self.typology == ModelTypes.DIFFERENCE_EQUATIONS:
        if self.solver is not None:
            raise ValueError(
                "Solver settings apply only to a DifferentialEquations model; "
                "a DifferenceEquations model has no solver."
            )
        for transition in self.transitions:
            bad = [p for p in transition.impulse_positions() if p != int(p)]
            if bad:
                raise ValueError(
                    f"Transition '{transition.id}': impulse position {bad[0]} "
                    f"is not an integer step; a DifferenceEquations model "
                    f"steps only at integers."
                )
    return self
```

#### validate_transitions_not_empty

```python
validate_transitions_not_empty(v: list[Transition]) -> list[Transition]
```

Require at least one transition.

Source code in `commol/context/dynamics.py`

```python
@field_validator("transitions")
@classmethod
def validate_transitions_not_empty(cls, v: list[Transition]) -> list[Transition]:
    """Require at least one transition."""
    if not v:
        raise ValueError("At least one transition must be defined.")
    return v
```

#### validate_unique_transition_ids

```python
validate_unique_transition_ids() -> Self
```

Validates that transition IDs are unique.

Source code in `commol/context/dynamics.py`

```python
@model_validator(mode="after")
def validate_unique_transition_ids(self) -> Self:
    """
    Validates that transition IDs are unique.
    """
    transition_ids = [t.id for t in self.transitions]
    if len(transition_ids) != len(set(transition_ids)):
        duplicates = [
            item for item in set(transition_ids) if transition_ids.count(item) > 1
        ]
        raise ValueError(f"Duplicate transition IDs found: {duplicates}")
    return self
```

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

Source code in `commol/context/dynamics.py`

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
        - "beta" (parameter reference)
        - "0.5" (constant, can also be passed as float 0.5)
        - "beta * S * I / N" (mathematical formula)
        - "0.3 * sin(2 * pi * t / 365)" (time-dependent formula)
        - "2^10" or "2**10" (power: both syntaxes work)
    stratified_rates : list[StratifiedRate] | None
        Stratification-specific rates. Each rate applies to compartments that match
        all specified stratification conditions.
    """

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
            "Can be a parameter reference (e.g., 'beta'), a constant (e.g., '0.5'), "
            "or a mathematical expression (e.g., 'beta * S * I / N'). "
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
            "one, the rate is a fraction (per-capita) or an amount (absolute) "
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

    def impulse_positions(self) -> list[float]:
        """
        The positions of `impulse_at`, or `[offset, period]` of `impulse_every`.

        These are the values that must be integers in a difference model.
        """
        if self.impulse_at is not None:
            return list(self.impulse_at)
        if self.impulse_every is not None:
            period, offset = self.impulse_every
            return [offset, period]
        return []

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

#### impulse_positions

```python
impulse_positions() -> list[float]
```

The positions of `impulse_at`, or `[offset, period]` of `impulse_every`.

These are the values that must be integers in a difference model.

Source code in `commol/context/dynamics.py`

```python
def impulse_positions(self) -> list[float]:
    """
    The positions of `impulse_at`, or `[offset, period]` of `impulse_every`.

    These are the values that must be integers in a difference model.
    """
    if self.impulse_at is not None:
        return list(self.impulse_at)
    if self.impulse_every is not None:
        period, offset = self.impulse_every
        return [offset, period]
    return []
```

#### validate_impulse

```python
validate_impulse() -> Self
```

Check the impulse schedule: one kind, valid positions and period.

Source code in `commol/context/dynamics.py`

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

Source code in `commol/context/dynamics.py`

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

`OdeSolver` is documented in [Simulation API](https://munqu.github.io/commol/latest/api/simulation/index.md).

## StratifiedRate

Bases: `BaseModel`

Defines a rate for specific stratification categories.

Attributes:

| Name         | Type                            | Description                                                |
| ------------ | ------------------------------- | ---------------------------------------------------------- |
| `conditions` | `list[StratificationCondition]` | List of stratification-category pairs that must match      |
| `rate`       | `str`                           | Rate expression for compartments matching these conditions |

Source code in `commol/context/dynamics.py`

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

    conditions: list[StratificationCondition] = Field(
        default=..., description="Stratification conditions that must match"
    )
    rate: str = Field(
        default=..., description="Rate expression for matching compartments"
    )
    absolute: bool | None = Field(
        default=None,
        description=(
            "When True, the flow is treated as absolute (not multiplied by source "
            "population). When None, inferred from whether the rate references "
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

Source code in `commol/context/dynamics.py`

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

Source code in `commol/context/dynamics.py`

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

Source code in `commol/context/dynamics.py`

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

## Initial conditions

## InitialConditions

Bases: `BaseModel`

Initial conditions for a simulation.

Attributes:

| Name                       | Type                                        | Description                                                                                                                                                               |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `population_size`          | `int`                                       | Population size.                                                                                                                                                          |
| `bin_fractions`            | `list[BinFraction]`                         | List of bin fractions. Each item contains a bin id and its initial fractional size. Fractions can be None if they need calibration.                                       |
| `stratification_fractions` | `(list[StratificationFractions], optional)` | List of stratification fractions. Each item contains a stratification id and its category fractions.                                                                      |
| `compartment_fractions`    | `(list[CompartmentFraction], optional)`     | List of per-compartment overrides applied after the uniform expansion. Each item contains an expanded compartment name and the fraction of the total population it holds. |

Source code in `commol/context/initial_conditions.py`

```python
class InitialConditions(BaseModel):
    """
    Initial conditions for a simulation.

    Attributes
    ----------
    population_size : int
        Population size.
    bin_fractions : list[BinFraction]
        List of bin fractions. Each item contains a bin id and
        its initial fractional size. Fractions can be None if they need calibration.
    stratification_fractions : list[StratificationFractions], optional
        List of stratification fractions. Each item contains a stratification id and
        its category fractions.
    compartment_fractions : list[CompartmentFraction], optional
        List of per-compartment overrides applied after the uniform expansion. Each
        item contains an expanded compartment name and the fraction of the total
        population it holds.
    """

    population_size: int = Field(..., description="Population size.")
    bin_fractions: list[BinFraction] = Field(
        default=...,
        description=(
            "List of bin fractions. Each item contains a bin id and its initial "
            "fractional size. Fractions can be None if they need calibration."
        ),
    )
    stratification_fractions: list[StratificationFractions] = Field(
        default_factory=list,
        description=(
            "List of stratification fractions. Each item contains a stratification id "
            "and its category fractions."
        ),
    )
    compartment_fractions: list[CompartmentFraction] = Field(
        default_factory=list,
        description=(
            "List of per-compartment overrides applied after the uniform expansion. "
            "Each item contains an expanded compartment name and the fraction of the "
            "total population it holds."
        ),
    )

    @field_validator("population_size")
    @classmethod
    def validate_population_size(cls, value: int) -> int:
        """Enforce a strictly positive population size.

        A zero population size would make every fraction-to-head-count
        conversion a division by zero, so it is rejected as a domain error
        rather than surfacing later as ``ZeroDivisionError``.
        """
        if value <= 0:
            raise ValueError(f"population_size must be greater than 0, got {value}.")
        return value

    @model_validator(mode="after")
    def validate_compartment_fractions_unique(self) -> Self:
        """Validate that each compartment appears at most once."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for compartment_fraction in self.compartment_fractions:
            name = compartment_fraction.compartment
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate compartment fractions found: {sorted(duplicates)}."
            )
        return self

    @staticmethod
    def _validate_bin_fraction_total(candidate: Mapping[str, float | None]) -> None:
        """
        Enforce the bin-fraction conservation rule on a complete mapping.

        Uncalibrated (``None``) fractions floor the calibrated total strictly
        below 1.0 so they can be filled later. When every bin is calibrated,
        the fractions must sum to 1.0, because they are shares of a single
        population. Construction and updates apply the same rule, so an update
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
        self._validate_bin_fraction_total(
            {bf.bin: bf.fraction for bf in self.bin_fractions}
        )
        return self

    def get_uncalibrated_bins(self) -> list[str]:
        """
        Get list of bin IDs that have uncalibrated fractions (value = None).

        Returns
        -------
        list[str]
            List of bin IDs with uncalibrated initial conditions.
        """
        return [bf.bin for bf in self.bin_fractions if bf.fraction is None]

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
            fraction.category
            for stratification in self.stratification_fractions
            for fraction in stratification.fractions
        }

    def resolve_category(self, category: str) -> tuple[str, str]:
        """
        Resolve a category id to its stratification and its bare name.

        A qualified id (``sex:female``) names the stratification. A bare name
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
        owner, name = self._owning_stratification(category)
        return owner.stratification, name

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
            any(fraction.category == category for fraction in item.fractions)
            for item in self.stratification_fractions
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
            A category id, bare or qualified (``sex:female``).

        Returns
        -------
        float
            The declared fraction, relative to the group the category's
            stratification applies to rather than to the whole population.

        Raises
        ------
        ValueError
            If the category is not found or a bare name is ambiguous.
        """
        owner, name = self._owning_stratification(category)
        return next(
            fraction.fraction
            for fraction in owner.fractions
            if fraction.category == name
        )

    def subgroup_population(self, categories: Iterable[str] = ()) -> float:
        """
        Get the initial head count of a subgroup.

        Category fractions are relative to the group their stratification
        applies to, so the fractions of a nested chain of categories multiply.
        Passing no categories gives the whole population.

        Parameters
        ----------
        categories : Iterable[str], optional
            Category ids, bare or qualified, forming a chain from the whole
            population down to the subgroup, outermost first.

        Returns
        -------
        float
            Number of individuals in the subgroup at step 0.

        Raises
        ------
        ValueError
            If a category is not found or a bare name is ambiguous.
        """
        population = float(self.population_size)
        for category in categories:
            population *= self.get_category_fraction(category)
        return population

    def _owning_stratification(
        self, category: str
    ) -> tuple[StratificationFractions, str]:
        """
        Return the stratification that declares a category, and its bare name.

        A qualified id (``sex:female``) names the stratification. A bare name
        must be declared by exactly one stratification.
        """
        stratification_id, separator, name = category.rpartition(CATEGORY_SEPARATOR)
        owners = [
            stratification
            for stratification in self.stratification_fractions
            if (not separator or stratification.stratification == stratification_id)
            and any(fraction.category == name for fraction in stratification.fractions)
        ]
        if len(owners) > 1:
            qualified = " or ".join(
                f"'{owner.stratification}{CATEGORY_SEPARATOR}{name}'"
                for owner in owners
            )
            raise ValueError(
                f"Category '{category}' is ambiguous: it is declared by "
                f"stratifications {[owner.stratification for owner in owners]}. "
                f"Qualify it as {qualified}."
            )
        if not owners:
            raise ValueError(
                f"Category '{category}' not found in initial conditions. "
                f"Available categories: {sorted(self.get_categories_with_fractions())}"
            )
        return owners[0], name

    def _apply_stratification_update(
        self,
        stratification: StratificationFractions,
        named: Mapping[str, float],
    ) -> None:
        """Write the fractions of a single stratification."""
        unnamed = [
            fraction.category
            for fraction in stratification.fractions
            if fraction.category not in named
        ]
        total = sum(named.values())

        if len(unnamed) > 1:
            raise ValueError(
                f"Cannot update stratification '{stratification.stratification}': "
                f"categories {sorted(unnamed)} were left unnamed. At most one "
                f"category may be omitted, so that it can take the remaining "
                f"fraction."
            )
        if not unnamed and not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Fractions for stratification '{stratification.stratification}' "
                f"must sum to 1.0, got {total:.6f}. Values: {dict(named)}"
            )
        if unnamed and total > 1.0 + 1e-6:
            raise ValueError(
                f"Fractions for stratification '{stratification.stratification}' "
                f"sum to {total:.6f}, which would leave a negative fraction for "
                f"'{unnamed[0]}'. Values: {dict(named)}"
            )

        remainder = max(0.0, 1.0 - total)
        for fraction in stratification.fractions:
            fraction.fraction = named.get(fraction.category, remainder)

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update stratification category fractions by category name.

        Categories belonging to several stratifications may be given in one
        call. A category name declared by more than one stratification must be
        qualified with its stratification, as in ``sex:female``. Within each
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
            [0.0, 1.0], a stratification has more than one category omitted, fully
            specified
            values do not sum to 1.0, or the given values exceed 1.0.
        """
        for category, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Fraction for category '{category}' must be between 0.0 and "
                    f"1.0, got {value}."
                )

        by_stratification: dict[str, dict[str, float]] = {}
        for category, value in fractions.items():
            owner, name = self._owning_stratification(category)
            by_stratification.setdefault(owner.stratification, {})[name] = value

        for stratification in self.stratification_fractions:
            named = by_stratification.get(stratification.stratification)
            if named is not None:
                self._apply_stratification_update(stratification, named)

    def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_fractions`` has its
        fraction replaced; a new one is appended. Compartment names and the
        total population depend on the model expansion, so they are validated
        by :meth:`~commol.context.model.Model.update_compartment_fractions`
        against the engine, not here.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Dictionary mapping expanded compartment names to their new
            fraction of the total population. Each value must lie between
            0.0 and 1.0.

        Raises
        ------
        ValueError
            If a value lies outside ``[0.0, 1.0]``.
        """
        updated = {
            compartment: CompartmentFraction(compartment=compartment, fraction=value)
            for compartment, value in fractions.items()
        }
        existing = {override.compartment for override in self.compartment_fractions}
        self.compartment_fractions = [
            updated.get(override.compartment, override)
            for override in self.compartment_fractions
        ] + [
            override
            for compartment, override in updated.items()
            if compartment not in existing
        ]

    def update_bin_fractions(self, fractions: Mapping[str, float | None]) -> None:
        """
        Update bin fractions by bin ID.

        The candidate fractions are validated before any bin is mutated, so a
        rejected call leaves the initial conditions unchanged. The conservation
        rule of :meth:`validate_calibrated_fractions_sum_to_one` is re-applied
        to the resulting mapping: the bin fractions must sum to 1.0, while
        uncalibrated (``None``) bins reserve the remainder. To move population
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
        known = {bf.bin for bf in self.bin_fractions}
        unknown = sorted(set(fractions) - known)
        if unknown:
            raise ValueError(
                f"Bin(s) {unknown} not found in initial conditions. "
                f"Available bins: {[bf.bin for bf in self.bin_fractions]}"
            )

        candidate = {bf.bin: bf.fraction for bf in self.bin_fractions}
        candidate.update(fractions)
        self._validate_bin_fraction_total(candidate)

        for bin_id, fraction in fractions.items():
            for bf in self.bin_fractions:
                if bf.bin == bin_id:
                    bf.fraction = fraction
                    break
```

### Methods:

#### validate_population_size

```python
validate_population_size(value: int) -> int
```

Enforce a strictly positive population size.

A zero population size would make every fraction-to-head-count conversion a division by zero, so it is rejected as a domain error rather than surfacing later as `ZeroDivisionError`.

Source code in `commol/context/initial_conditions.py`

```python
@field_validator("population_size")
@classmethod
def validate_population_size(cls, value: int) -> int:
    """Enforce a strictly positive population size.

    A zero population size would make every fraction-to-head-count
    conversion a division by zero, so it is rejected as a domain error
    rather than surfacing later as ``ZeroDivisionError``.
    """
    if value <= 0:
        raise ValueError(f"population_size must be greater than 0, got {value}.")
    return value
```

#### validate_compartment_fractions_unique

```python
validate_compartment_fractions_unique() -> Self
```

Validate that each compartment appears at most once.

Source code in `commol/context/initial_conditions.py`

```python
@model_validator(mode="after")
def validate_compartment_fractions_unique(self) -> Self:
    """Validate that each compartment appears at most once."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for compartment_fraction in self.compartment_fractions:
        name = compartment_fraction.compartment
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"Duplicate compartment fractions found: {sorted(duplicates)}."
        )
    return self
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
    self._validate_bin_fraction_total(
        {bf.bin: bf.fraction for bf in self.bin_fractions}
    )
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
    return [bf.bin for bf in self.bin_fractions if bf.fraction is None]
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
        fraction.category
        for stratification in self.stratification_fractions
        for fraction in stratification.fractions
    }
```

#### resolve_category

```python
resolve_category(category: str) -> tuple[str, str]
```

Resolve a category id to its stratification and its bare name.

A qualified id (`sex:female`) names the stratification. A bare name must be declared by exactly one stratification.

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

    A qualified id (``sex:female``) names the stratification. A bare name
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
    owner, name = self._owning_stratification(category)
    return owner.stratification, name
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
        any(fraction.category == category for fraction in item.fractions)
        for item in self.stratification_fractions
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

| Name       | Type  | Description                                    | Default    |
| ---------- | ----- | ---------------------------------------------- | ---------- |
| `category` | `str` | A category id, bare or qualified (sex:female). | *required* |

Returns:

| Type    | Description                                                                                                                |
| ------- | -------------------------------------------------------------------------------------------------------------------------- |
| `float` | The declared fraction, relative to the group the category's stratification applies to rather than to the whole population. |

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
        A category id, bare or qualified (``sex:female``).

    Returns
    -------
    float
        The declared fraction, relative to the group the category's
        stratification applies to rather than to the whole population.

    Raises
    ------
    ValueError
        If the category is not found or a bare name is ambiguous.
    """
    owner, name = self._owning_stratification(category)
    return next(
        fraction.fraction
        for fraction in owner.fractions
        if fraction.category == name
    )
```

#### subgroup_population

```python
subgroup_population(categories: Iterable[str] = ()) -> float
```

Get the initial head count of a subgroup.

Category fractions are relative to the group their stratification applies to, so the fractions of a nested chain of categories multiply. Passing no categories gives the whole population.

Parameters:

| Name         | Type            | Description                                                                                                       | Default |
| ------------ | --------------- | ----------------------------------------------------------------------------------------------------------------- | ------- |
| `categories` | `Iterable[str]` | Category ids, bare or qualified, forming a chain from the whole population down to the subgroup, outermost first. | `()`    |

Returns:

| Type    | Description                                      |
| ------- | ------------------------------------------------ |
| `float` | Number of individuals in the subgroup at step 0. |

Raises:

| Type         | Description                                             |
| ------------ | ------------------------------------------------------- |
| `ValueError` | If a category is not found or a bare name is ambiguous. |

Source code in `commol/context/initial_conditions.py`

```python
def subgroup_population(self, categories: Iterable[str] = ()) -> float:
    """
    Get the initial head count of a subgroup.

    Category fractions are relative to the group their stratification
    applies to, so the fractions of a nested chain of categories multiply.
    Passing no categories gives the whole population.

    Parameters
    ----------
    categories : Iterable[str], optional
        Category ids, bare or qualified, forming a chain from the whole
        population down to the subgroup, outermost first.

    Returns
    -------
    float
        Number of individuals in the subgroup at step 0.

    Raises
    ------
    ValueError
        If a category is not found or a bare name is ambiguous.
    """
    population = float(self.population_size)
    for category in categories:
        population *= self.get_category_fraction(category)
    return population
```

#### update_stratification_fractions

```python
update_stratification_fractions(fractions: Mapping[str, float]) -> None
```

Update stratification category fractions by category name.

Categories belonging to several stratifications may be given in one call. A category name declared by more than one stratification must be qualified with its stratification, as in `sex:female`. Within each stratification, at most one category may be omitted; the omitted category receives the remaining fraction so that the stratification sums to 1.0. When every category is given, the values must already sum to 1.0.

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
    qualified with its stratification, as in ``sex:female``. Within each
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
        [0.0, 1.0], a stratification has more than one category omitted, fully
        specified
        values do not sum to 1.0, or the given values exceed 1.0.
    """
    for category, value in fractions.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Fraction for category '{category}' must be between 0.0 and "
                f"1.0, got {value}."
            )

    by_stratification: dict[str, dict[str, float]] = {}
    for category, value in fractions.items():
        owner, name = self._owning_stratification(category)
        by_stratification.setdefault(owner.stratification, {})[name] = value

    for stratification in self.stratification_fractions:
        named = by_stratification.get(stratification.stratification)
        if named is not None:
            self._apply_stratification_update(stratification, named)
```

#### update_compartment_fractions

```python
update_compartment_fractions(fractions: Mapping[str, float]) -> None
```

Upsert per-compartment overrides by expanded compartment name.

A compartment already present in `compartment_fractions` has its fraction replaced; a new one is appended. Compartment names and the total population depend on the model expansion, so they are validated by :meth:`~commol.context.model.Model.update_compartment_fractions` against the engine, not here.

Parameters:

| Name        | Type                  | Description                                                                                                                           | Default    |
| ----------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `fractions` | `Mapping[str, float]` | Dictionary mapping expanded compartment names to their new fraction of the total population. Each value must lie between 0.0 and 1.0. | *required* |

Raises:

| Type         | Description                         |
| ------------ | ----------------------------------- |
| `ValueError` | If a value lies outside [0.0, 1.0]. |

Source code in `commol/context/initial_conditions.py`

```python
def update_compartment_fractions(self, fractions: Mapping[str, float]) -> None:
    """
    Upsert per-compartment overrides by expanded compartment name.

    A compartment already present in ``compartment_fractions`` has its
    fraction replaced; a new one is appended. Compartment names and the
    total population depend on the model expansion, so they are validated
    by :meth:`~commol.context.model.Model.update_compartment_fractions`
    against the engine, not here.

    Parameters
    ----------
    fractions : Mapping[str, float]
        Dictionary mapping expanded compartment names to their new
        fraction of the total population. Each value must lie between
        0.0 and 1.0.

    Raises
    ------
    ValueError
        If a value lies outside ``[0.0, 1.0]``.
    """
    updated = {
        compartment: CompartmentFraction(compartment=compartment, fraction=value)
        for compartment, value in fractions.items()
    }
    existing = {override.compartment for override in self.compartment_fractions}
    self.compartment_fractions = [
        updated.get(override.compartment, override)
        for override in self.compartment_fractions
    ] + [
        override
        for compartment, override in updated.items()
        if compartment not in existing
    ]
```

#### update_bin_fractions

```python
update_bin_fractions(fractions: Mapping[str, float | None]) -> None
```

Update bin fractions by bin ID.

The candidate fractions are validated before any bin is mutated, so a rejected call leaves the initial conditions unchanged. The conservation rule of :meth:`validate_calibrated_fractions_sum_to_one` is re-applied to the resulting mapping: the bin fractions must sum to 1.0, while uncalibrated (`None`) bins reserve the remainder. To move population between bins, update every bin involved in one call.

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
    uncalibrated (``None``) bins reserve the remainder. To move population
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
    known = {bf.bin for bf in self.bin_fractions}
    unknown = sorted(set(fractions) - known)
    if unknown:
        raise ValueError(
            f"Bin(s) {unknown} not found in initial conditions. "
            f"Available bins: {[bf.bin for bf in self.bin_fractions]}"
        )

    candidate = {bf.bin: bf.fraction for bf in self.bin_fractions}
    candidate.update(fractions)
    self._validate_bin_fraction_total(candidate)

    for bin_id, fraction in fractions.items():
        for bf in self.bin_fractions:
            if bf.bin == bin_id:
                bf.fraction = fraction
                break
```

## BinFraction

Bases: `BaseModel`

Fraction for a single bin.

Attributes:

| Name       | Type    | Description |
| ---------- | ------- | ----------- |
| `bin`      | `str`   | The bin id. |
| `fraction` | \`float | None\`      |

Source code in `commol/context/initial_conditions.py`

```python
class BinFraction(BaseModel):
    """
    Fraction for a single bin.

    Attributes
    ----------
    bin : str
        The bin id.
    fraction : float | None
        The fractional size of this bin. Set to None if this initial condition
        needs to be calibrated.
    """

    bin: str = Field(default=..., description="The bin id.")
    fraction: float | None = Field(
        default=...,
        description=(
            "The fractional size of this bin. "
            "Set to None if this initial condition needs to be calibrated."
        ),
    )

    def is_calibrated(self) -> bool:
        """Check if this bin fraction has a calibrated value (not None)."""
        return self.fraction is not None
```

### Methods:

#### is_calibrated

```python
is_calibrated() -> bool
```

Check if this bin fraction has a calibrated value (not None).

Source code in `commol/context/initial_conditions.py`

```python
def is_calibrated(self) -> bool:
    """Check if this bin fraction has a calibrated value (not None)."""
    return self.fraction is not None
```

## StratificationFractions

Bases: `BaseModel`

Fractions for a stratification.

Attributes:

| Name             | Type                           | Description                                         |
| ---------------- | ------------------------------ | --------------------------------------------------- |
| `stratification` | `str`                          | The stratification id.                              |
| `fractions`      | `list[StratificationFraction]` | List of category fractions for this stratification. |

Source code in `commol/context/initial_conditions.py`

```python
class StratificationFractions(BaseModel):
    """
    Fractions for a stratification.

    Attributes
    ----------
    stratification : str
        The stratification id.
    fractions : list[StratificationFraction]
        List of category fractions for this stratification.
    """

    stratification: str = Field(..., description="The stratification id.")
    fractions: list[StratificationFraction] = Field(
        default=..., description="List of category fractions for this stratification."
    )
```

## StratificationFraction

Bases: `BaseModel`

Fractions for a single stratification category.

Attributes:

| Name       | Type    | Description                           |
| ---------- | ------- | ------------------------------------- |
| `category` | `str`   | The stratification category name.     |
| `fraction` | `float` | The fractional size of this category. |

Source code in `commol/context/initial_conditions.py`

```python
class StratificationFraction(BaseModel):
    """
    Fractions for a single stratification category.

    Attributes
    ----------
    category : str
        The stratification category name.
    fraction : float
        The fractional size of this category.
    """

    category: str = Field(default=..., description="The stratification category name.")
    fraction: float = Field(
        default=..., description="The fractional size of this category."
    )
```

## CompartmentFraction

Bases: `BaseModel`

Fraction for a single expanded compartment.

Attributes:

| Name          | Type    | Description                                                                    |
| ------------- | ------- | ------------------------------------------------------------------------------ |
| `compartment` | `str`   | The expanded compartment name, for example IH_y60 or S_oe60_v_c.               |
| `fraction`    | `float` | The fraction of the total population in this compartment, between 0.0 and 1.0. |

Source code in `commol/context/initial_conditions.py`

```python
class CompartmentFraction(BaseModel):
    """
    Fraction for a single expanded compartment.

    Attributes
    ----------
    compartment : str
        The expanded compartment name, for example ``IH_y60`` or ``S_oe60_v_c``.
    fraction : float
        The fraction of the total population in this compartment,
        between 0.0 and 1.0.
    """

    compartment: str = Field(default=..., description="The expanded compartment name.")
    fraction: float = Field(
        default=...,
        description="The fraction of the total population in this compartment.",
    )

    @field_validator("fraction")
    @classmethod
    def validate_fraction_range(cls, value: float) -> float:
        """Enforce that the fraction lies within ``[0.0, 1.0]``."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Fraction must be between 0.0 and 1.0, got {value}.")
        return value
```

### Methods:

#### validate_fraction_range

```python
validate_fraction_range(value: float) -> float
```

Enforce that the fraction lies within `[0.0, 1.0]`.

Source code in `commol/context/initial_conditions.py`

```python
@field_validator("fraction")
@classmethod
def validate_fraction_range(cls, value: float) -> float:
    """Enforce that the fraction lies within ``[0.0, 1.0]``."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Fraction must be between 0.0 and 1.0, got {value}.")
    return value
```
