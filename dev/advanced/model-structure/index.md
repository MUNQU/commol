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
│   ├── matrices: [CategoryMatrix]         (values: {row: {column: value}}, row_scales, column_scales)
│   └── transitions: [Transition]          (impulse_at, impulse_every for impulses)
│       └── stratified_rates: [StratifiedRate]
│           └── conditions: [StratificationCondition]
└── structure: Aggregate                   (kind: "aggregate", variant: None)
    ├── bin_amounts: {bin: amount | None}          (in the unit of the bin)
    ├── stratification_fractions: {stratification: {category: fraction}
    │                                  | {"by": stratification, category: {category: fraction}}}
    ├── compartment_amounts: {compartment: amount}
    └── neighbors: Complete | Groups | None
```

In JSON the structure is an object with its `kind`, its `variant` and the fields of its class:

```json
{
  "name": "Model",
  "parameters": [],
  "engine": {"type": "DifferenceEquations", "settings": null},
  "process": {"bins": [], "accumulators": [], "stratifications": [], "matrices": [], "transitions": []},
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

## The parts of a model

`Model.process` holds the parts that the builder makes, and `Model.parameters` holds the parameters. The root `commol` exports the class of each part, so a typed function that reads a part names its class. The [Model API](https://munqu.github.io/commol/latest/api/model/#the-parts-of-a-model) documents them.

| Part                 | Added by                                                                                                                                                           | Read with                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| `Bin`, `Accumulator` | `ModelBuilder.add_bin`, `ModelBuilder.add_accumulator`                                                                                                             | `Process.bins`, `Process.accumulators` |
| `Stratification`     | `ModelBuilder.add_stratification`, with a `StratificationCondition` ([ModelBuilder API](https://munqu.github.io/commol/latest/api/model-builder/index.md))         | `Process.stratifications`              |
| `CategoryMatrix`     | `ModelBuilder.add_matrix`; a rate reads it through `matrix_sum` ([Matrix sums](https://munqu.github.io/commol/latest/guide/mathematical-expressions/#matrix-sums)) | `Process.matrices`                     |
| `Parameter`          | `ModelBuilder.add_parameter`; a series parameter holds a `SeriesValue` ([Model API](https://munqu.github.io/commol/latest/api/model/index.md))                     | `Model.parameters`                     |
| `Transition`         | `ModelBuilder.add_transition`, with a `StratifiedRate` ([ModelBuilder API](https://munqu.github.io/commol/latest/api/model-builder/index.md))                      | `Process.transitions`                  |

## Engine

The engine's list of engines (`crates/commol-core/src/types/engine.rs`) validates the settings: settings of another engine raise an error that names both. `Model.engine` is a `ModelEngine`, which the [Engines API](https://munqu.github.io/commol/latest/api/engines/index.md) documents with `ODESolver`.

## Initial conditions

`Aggregate` holds the fields and methods of `InitialConditions`, and checks them against the process when the model is built.

## InitialConditions

Bases: `BaseModel`

How much starts where: the amount of each bin and the split of it.

Every amount is in the unit of its bin, so bins of different units, such as a count and a concentration, start without a total over them. `Aggregate` holds these fields and their methods; the checks against the bins and the stratifications of the process are in `Aggregate.check_model`.

Attributes:

| Name                       | Type               | Description                                                                                                                                                                                                                        |
| -------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bin_amounts`              | \`dict\[str, float | None\]\`                                                                                                                                                                                                                           |
| `stratification_fractions` | `dict[str, dict]`  | The split of each stratification, by stratification id: a dict[str, float], the fraction of each category, such as {"t1": 0.4, "t2": 0.6}, the split of the amount of every bin the stratification applies to; or a dict\[str, str |
| `compartment_amounts`      | `dict[str, float]` | Amounts of single expanded compartments, by compartment name, applied after the expansion.                                                                                                                                         |

Source code in `commol/_model/initial_conditions.py`

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
    stratification_fractions : dict[str, dict]
        The split of each stratification, by stratification id: a
        ``dict[str, float]``, the fraction of each category, such as
        ``{"t1": 0.4, "t2": 0.6}``, the split of the amount of every bin the
        stratification applies to; or a
        ``dict[str, str | dict[str, float]]``, a split per category of a
        stratification declared before it, such as
        ``{"by": "group", "g1": {"t1": 0.25, "t2": 0.75}, "g2": {"t1": 0.5,
        "t2": 0.5}}``. Each fraction is from 0 to 1, and each split sums to 1
        within 1e-6. A conditional stratification splits only the group its
        conditions select.
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
    stratification_fractions: dict[str, _Fractions] = Field(
        default_factory=dict,
        description=(
            "The fraction of each category, by stratification id, or a split "
            "per category of an earlier stratification under 'by'."
        ),
    )
    compartment_amounts: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Amounts of single expanded compartments, applied after the expansion."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_fractions_of_a_total(cls, data: JsonValue) -> JsonValue:
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
    def _validate_bin_amounts(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        """Check every bin amount with the rule of the engine."""
        commol_rs.model.check_bin_amounts(list(value.items()))
        return value

    @field_validator("stratification_fractions")
    @classmethod
    def _validate_stratification_fractions(
        cls, value: dict[str, _Fractions]
    ) -> dict[str, _Fractions]:
        """Check the fractions of every stratification with the rule of the
        engine."""
        _check_fractions(value)
        return value

    @field_validator("compartment_amounts")
    @classmethod
    def _validate_compartment_amounts(cls, value: dict[str, float]) -> dict[str, float]:
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
            Category names collected from ``stratification_fractions``, the
            categories of each split of a split by category included. A
            category declared on a stratification but absent from the initial
            conditions is not included.
        """
        return {
            category
            for fractions in self.stratification_fractions.values()
            for category in split_categories(fractions)
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
            for stratification, fractions in self.stratification_fractions.items()
            if (not separator or stratification == stratification_id)
            and name in split_categories(fractions)
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
            category in split_categories(fractions)
            for fractions in self.stratification_fractions.values()
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
            If the category is not found, a bare name is ambiguous, or the
            fractions of its stratification differ by the category of an
            earlier one.
        """
        stratification, name = self.resolve_category(category)
        fractions = self.stratification_fractions[stratification]
        by = split_by(fractions)
        if by is not None:
            raise ValueError(
                f"The fractions of stratification '{stratification}' differ by "
                f"the category of '{by}': read them in stratification_fractions."
            )
        return category_splits(fractions)[0][1][name]

    def _updated_fractions(
        self, stratification: str, named: Mapping[str, float]
    ) -> _Fractions:
        """The fractions of a single stratification after an update that
        names ``named``: in each split, the named categories take their
        values and the one category left unnamed takes the rest."""
        fractions = self.stratification_fractions[stratification]
        splits = {
            of: _updated_split(stratification, split, named)
            for of, split in category_splits(fractions)
        }
        by = split_by(fractions)
        if by is None:
            return splits[None]
        return {FRACTIONS_BY: by, **{str(of): split for of, split in splits.items()}}

    def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
        """
        Update stratification category fractions by category name.

        Categories belonging to several stratifications may be given in one
        call. A category name declared by more than one stratification must be
        qualified with its stratification, as in ``group:g1``. Within each
        stratification, at most one category may be omitted; the omitted
        category receives the rest of 1, or 0 when the given values pass 1.
        A stratification whose fractions differ by the category of an earlier
        one takes the values in every split, as a calibrated category does.
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
            engine, if a fraction is not from 0 to 1 or a split does not sum
            to 1 within 1e-6.
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
        self.stratification_fractions.update(updated)

    def _upsert_compartment_amounts(self, amounts: Mapping[str, float]) -> None:
        """
        Upsert per-compartment overrides by expanded compartment name.

        A compartment already present in ``compartment_amounts`` has its
        amount replaced; a new one is appended. Compartment names depend on
        the expansion of the process, so
        :meth:`~commol._model.aggregate.Aggregate.update_compartment_amounts`
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

#### get_uncalibrated_bins

```python
get_uncalibrated_bins() -> list[str]
```

Get list of bin IDs that have uncalibrated amounts (value = None).

Returns:

| Type        | Description                                           |
| ----------- | ----------------------------------------------------- |
| `list[str]` | List of bin IDs with uncalibrated initial conditions. |

Source code in `commol/_model/initial_conditions.py`

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

| Type       | Description                                                                                                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `set[str]` | Category names collected from stratification_fractions, the categories of each split of a split by category included. A category declared on a stratification but absent from the initial conditions is not included. |

Source code in `commol/_model/initial_conditions.py`

```python
def get_categories_with_fractions(self) -> set[str]:
    """
    Get the categories that have a declared initial fraction.

    Returns
    -------
    set[str]
        Category names collected from ``stratification_fractions``, the
        categories of each split of a split by category included. A
        category declared on a stratification but absent from the initial
        conditions is not included.
    """
    return {
        category
        for fractions in self.stratification_fractions.values()
        for category in split_categories(fractions)
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

Source code in `commol/_model/initial_conditions.py`

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
        for stratification, fractions in self.stratification_fractions.items()
        if (not separator or stratification == stratification_id)
        and name in split_categories(fractions)
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

Source code in `commol/_model/initial_conditions.py`

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
        category in split_categories(fractions)
        for fractions in self.stratification_fractions.values()
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

| Type         | Description                                                                                                                              |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the category is not found, a bare name is ambiguous, or the fractions of its stratification differ by the category of an earlier one. |

Source code in `commol/_model/initial_conditions.py`

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
        If the category is not found, a bare name is ambiguous, or the
        fractions of its stratification differ by the category of an
        earlier one.
    """
    stratification, name = self.resolve_category(category)
    fractions = self.stratification_fractions[stratification]
    by = split_by(fractions)
    if by is not None:
        raise ValueError(
            f"The fractions of stratification '{stratification}' differ by "
            f"the category of '{by}': read them in stratification_fractions."
        )
    return category_splits(fractions)[0][1][name]
```

#### update_stratification_fractions

```python
update_stratification_fractions(fractions: Mapping[str, float]) -> None
```

Update stratification category fractions by category name.

Categories belonging to several stratifications may be given in one call. A category name declared by more than one stratification must be qualified with its stratification, as in `group:g1`. Within each stratification, at most one category may be omitted; the omitted category receives the rest of 1, or 0 when the given values pass 1. A stratification whose fractions differ by the category of an earlier one takes the values in every split, as a calibrated category does. The fractions of every stratification are checked with the rule of the engine before any of them changes.

Parameters:

| Name        | Type                  | Description                                                                         | Default    |
| ----------- | --------------------- | ----------------------------------------------------------------------------------- | ---------- |
| `fractions` | `Mapping[str, float]` | Dictionary mapping category names, bare or qualified, to their new fraction values. | *required* |

Raises:

| Type         | Description                                                                                                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a category is not found or is ambiguous, or a stratification has more than one category omitted; with the message of the engine, if a fraction is not from 0 to 1 or a split does not sum to 1 within 1e-6. |

Source code in `commol/_model/initial_conditions.py`

```python
def update_stratification_fractions(self, fractions: Mapping[str, float]) -> None:
    """
    Update stratification category fractions by category name.

    Categories belonging to several stratifications may be given in one
    call. A category name declared by more than one stratification must be
    qualified with its stratification, as in ``group:g1``. Within each
    stratification, at most one category may be omitted; the omitted
    category receives the rest of 1, or 0 when the given values pass 1.
    A stratification whose fractions differ by the category of an earlier
    one takes the values in every split, as a calibrated category does.
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
        engine, if a fraction is not from 0 to 1 or a split does not sum
        to 1 within 1e-6.
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
    self.stratification_fractions.update(updated)
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

Source code in `commol/_model/initial_conditions.py`

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
