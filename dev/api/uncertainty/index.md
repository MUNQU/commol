# Uncertainty API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

What every uncertainty method takes and returns: the configuration base, the weighted parameter sample, the predictive and its observation noise, the resampling and conformal methods, the scores and the one quantile. Guide: [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md).

## InferenceMethod

Bases: `BaseModel`, `Generic[ResultT]`

The configuration of a method that reads a calibration problem, whose result is `ResultT`.

A method subclasses one of its kinds, `UncertaintyMethod` or `IdentifiabilityMethod`, with its result type, and registers its runner (`commol.api.inference.registry.register`).

Source code in `commol/context/inference/method.py`

```python
class InferenceMethod(BaseModel, Generic[ResultT]):
    """
    The configuration of a method that reads a calibration problem, whose
    result is `ResultT`.

    A method subclasses one of its kinds, `UncertaintyMethod` or
    `IdentifiabilityMethod`, with its result type, and registers its runner
    (`commol.api.inference.registry.register`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
```

## UncertaintyMethod

Bases: `InferenceMethod[ResultT]`, `Generic[ResultT]`

The configuration of an uncertainty method, whose result is `ResultT`.

A method subclasses it with its result type, for example `class BayesianBootstrap(UncertaintyMethod[BootstrapResult])`, and registers its runner (`commol.api.inference.registry.register`). `Calibrator.uncertainty(method)` returns that result type, so a static type checker infers it from the configuration's type.

Source code in `commol/context/inference/method.py`

```python
class UncertaintyMethod(InferenceMethod[ResultT], Generic[ResultT]):
    """
    The configuration of an uncertainty method, whose result is `ResultT`.

    A method subclasses it with its result type, for example
    ``class BayesianBootstrap(UncertaintyMethod[BootstrapResult])``, and
    registers its runner (`commol.api.inference.registry.register`).
    ``Calibrator.uncertainty(method)`` returns that result type, so a static
    type checker infers it from the configuration's type.
    """
```

## ParameterSample

Bases: `BaseModel`

Draws of the parameters of a calibration problem, each with a weight.

A method that has draws returns them in a sample: a bootstrap, a sequential Monte Carlo run, a Markov chain. Plots, reports, the predictive and the scores read a sample, never the result of one method. Every statistic is weighted and computed in Rust; with equal weights it is the unweighted statistic.

A sample has no single set of parameters: to run one draw, take it from `draws`.

Attributes:

| Name          | Type                        | Description                                                                                                           |
| ------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `names`       | `list[str]`                 | One name per column, in the order of the problem's parameters.                                                        |
| `bounds`      | `list[tuple[float, float]]` | The bounds of each parameter; every value lies inside them.                                                           |
| `values`      | `list[list[float]]`         | One row per draw, one column per name.                                                                                |
| `log_weights` | `list[float]`               | The natural logarithm of each draw's weight, normalized so that the weights sum to 1. A draw of weight zero has -inf. |
| `groups`      | \`list[int]                 | None\`                                                                                                                |
| `method`      | `SampleMethod`              | What made the sample.                                                                                                 |

Raises:

| Type         | Description                                                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If a value is not finite or outside its bounds, a log weight is NaN or +inf, every log weight is -inf, a name repeats, or the shapes differ. |

Source code in `commol/context/inference/sample.py`

```python
class ParameterSample(BaseModel):
    """
    Draws of the parameters of a calibration problem, each with a weight.

    A method that has draws returns them in a sample: a bootstrap, a
    sequential Monte Carlo run, a Markov chain. Plots, reports, the
    predictive and the scores read a sample, never the result of one method.
    Every statistic is weighted and computed in Rust; with equal weights it
    is the unweighted statistic.

    A sample has no single set of parameters: to run one draw, take it from
    `draws`.

    Attributes
    ----------
    names : list[str]
        One name per column, in the order of the problem's parameters.
    bounds : list[tuple[float, float]]
        The bounds of each parameter; every value lies inside them.
    values : list[list[float]]
        One row per draw, one column per name.
    log_weights : list[float]
        The natural logarithm of each draw's weight, normalized so that the
        weights sum to 1. A draw of weight zero has ``-inf``.
    groups : list[int] | None
        The group of each draw (a chain, a replicate or a wave), or None.
    method : SampleMethod
        What made the sample.

    Raises
    ------
    ValueError
        If a value is not finite or outside its bounds, a log weight is NaN or
        ``+inf``, every log weight is ``-inf``, a name repeats, or the shapes
        differ.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    names: list[str] = Field(description="One name per column.")
    bounds: list[tuple[float, float]] = Field(description="Bounds of each parameter.")
    values: list[list[float]] = Field(description="One row per draw.")
    log_weights: list[float] = Field(description="Normalized log weight of each draw.")
    groups: list[int] | None = Field(default=None, description="Group of each draw.")
    method: SampleMethod = Field(description="What made the sample.")

    _rust: "ParameterSampleProtocol" = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def normalize_log_weights(cls, data: object) -> object:
        """Check the invariants in Rust and normalize the log weights."""
        if not isinstance(data, Mapping):
            return data
        rust = commol_rs.inference.ParameterSample(
            list(data.get("names", [])),
            [tuple(bound) for bound in data.get("bounds", [])],
            [list(row) for row in data.get("values", [])],
            [float(w) for w in data.get("log_weights", [])],
            data.get("groups"),
        )
        return {**data, "log_weights": rust.log_weights}

    @model_validator(mode="after")
    def build_rust_sample(self) -> Self:
        """Keep the Rust sample that computes the statistics."""
        self._rust = commol_rs.inference.ParameterSample(
            self.names, self.bounds, self.values, self.log_weights, self.groups
        )
        return self

    def __eq__(self, other: object) -> bool:
        """Two samples are equal when their fields are."""
        if not isinstance(other, ParameterSample):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def equally_weighted(
        cls,
        names: list[str],
        bounds: list[tuple[float, float]],
        values: list[list[float]],
        method: SampleMethod,
        groups: list[int] | None = None,
    ) -> Self:
        """
        A sample whose draws have equal weights.

        Parameters
        ----------
        names : list[str]
            One name per column.
        bounds : list[tuple[float, float]]
            The bounds of each parameter.
        values : list[list[float]]
            One row per draw.
        method : SampleMethod
            What made the sample.
        groups : list[int] | None, optional
            The group of each draw.

        Returns
        -------
        ParameterSample
            The sample.
        """
        return cls(
            names=names,
            bounds=bounds,
            values=values,
            log_weights=[0.0] * len(values),
            groups=groups,
            method=method,
        )

    @property
    def rust_sample(self) -> "ParameterSampleProtocol":
        """The sample in Rust, for the runners of methods and the predictive."""
        return self._rust

    @property
    def weights(self) -> list[float]:
        """The weight of each draw; they sum to 1."""
        return self._rust.weights

    def __len__(self) -> int:
        """The number of draws."""
        return len(self.values)

    def draws(self) -> Iterator[dict[str, float]]:
        """
        Every draw, as a mapping of parameter name to value.

        Yields
        ------
        dict[str, float]
            One draw, in the order of `values`.
        """
        for row in self.values:
            yield dict(zip(self.names, row, strict=True))

    def mean(self) -> dict[str, float]:
        """
        The weighted mean of each parameter.

        Returns
        -------
        dict[str, float]
            One mean per name.
        """
        return dict(zip(self.names, self._rust.mean(), strict=True))

    def covariance(self) -> list[list[float]]:
        """
        The weighted covariance matrix, in the order of `names`.

        With weights ``w`` that sum to 1 it is
        ``Σ wᵢ(xᵢ − x̄)(yᵢ − ȳ) / (1 − Σ wᵢ²)``: with equal weights, the
        sample covariance. It is NaN for a sample of one effective draw.

        Returns
        -------
        list[list[float]]
            One row per name.
        """
        return self._rust.covariance()

    def correlation(self) -> list[list[float]]:
        """
        The weighted correlation matrix, in the order of `names`.

        Returns
        -------
        list[list[float]]
            One row per name; NaN for a parameter with no spread.
        """
        return self._rust.correlation()

    def quantiles(self, levels: list[float]) -> dict[str, list[float]]:
        """
        The weighted quantiles of each parameter.

        The quantile is the inverse of the weighted empirical distribution,
        interpolated between the midpoints of the cumulative weights; with
        equal weights it is NumPy's ``method="hazen"``.

        Parameters
        ----------
        levels : list[float]
            Quantile levels, each between 0 and 1.

        Returns
        -------
        dict[str, list[float]]
            One quantile per level for each name.

        Raises
        ------
        ValueError
            If a level is not between 0 and 1.
        """
        rows = self._rust.quantiles([float(level) for level in levels])
        return {
            name: [row[column] for row in rows]
            for column, name in enumerate(self.names)
        }

    def interval(self, confidence_level: float) -> dict[str, tuple[float, float]]:
        """
        The central interval of each parameter.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1, such as 0.95.

        Returns
        -------
        dict[str, tuple[float, float]]
            ``(lower, upper)`` for each name: the quantiles
            ``(1 − c)/2`` and ``(1 + c)/2``.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        lower, upper = self._rust.interval(confidence_level)
        return {
            name: (low, high)
            for name, low, high in zip(self.names, lower, upper, strict=True)
        }

    def effective_size(self) -> float:
        """
        Kish's effective sample size: ``1 / Σ wᵢ²``.

        Returns
        -------
        float
            Between 1 and the number of draws; equal weights give the number
            of draws.
        """
        return self._rust.effective_size()

    def resample(self, count: int, seed: int) -> "ParameterSample":
        """
        Draw `count` draws with replacement, each with probability its weight.

        Parameters
        ----------
        count : int
            Number of draws.
        seed : int
            Seed of the generator.

        Returns
        -------
        ParameterSample
            A sample of equal weights.
        """
        indices = self._rust.resample(count, seed)
        return ParameterSample.equally_weighted(
            self.names,
            self.bounds,
            [self.values[i] for i in indices],
            SampleMethod(
                name="resample",
                settings={"of": self.method.name, "count": count, "seed": seed},
            ),
        )

    def summary(self, confidence_level: float) -> dict[str, ParameterSummary]:
        """
        Weighted statistics of each parameter, for a report.

        Parameters
        ----------
        confidence_level : float
            Confidence level of the interval, strictly between 0 and 1.

        Returns
        -------
        dict[str, ParameterSummary]
            One summary per name.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        lower, upper = self._rust.interval(confidence_level)
        quantiles = self.quantiles([0.0, 0.5, 1.0])
        mean = self.mean()
        covariance = self.covariance()
        return {
            name: ParameterSummary(
                mean=mean[name],
                median=quantiles[name][1],
                std=covariance[index][index] ** 0.5,
                lower=lower[index],
                upper=upper[index],
                min=quantiles[name][0],
                max=quantiles[name][2],
            )
            for index, name in enumerate(self.names)
        }
```

### Attributes

#### rust_sample

```python
rust_sample: ParameterSampleProtocol
```

The sample in Rust, for the runners of methods and the predictive.

#### weights

```python
weights: list[float]
```

The weight of each draw; they sum to 1.

### Methods:

#### normalize_log_weights

```python
normalize_log_weights(data: object) -> object
```

Check the invariants in Rust and normalize the log weights.

Source code in `commol/context/inference/sample.py`

```python
@model_validator(mode="before")
@classmethod
def normalize_log_weights(cls, data: object) -> object:
    """Check the invariants in Rust and normalize the log weights."""
    if not isinstance(data, Mapping):
        return data
    rust = commol_rs.inference.ParameterSample(
        list(data.get("names", [])),
        [tuple(bound) for bound in data.get("bounds", [])],
        [list(row) for row in data.get("values", [])],
        [float(w) for w in data.get("log_weights", [])],
        data.get("groups"),
    )
    return {**data, "log_weights": rust.log_weights}
```

#### build_rust_sample

```python
build_rust_sample() -> Self
```

Keep the Rust sample that computes the statistics.

Source code in `commol/context/inference/sample.py`

```python
@model_validator(mode="after")
def build_rust_sample(self) -> Self:
    """Keep the Rust sample that computes the statistics."""
    self._rust = commol_rs.inference.ParameterSample(
        self.names, self.bounds, self.values, self.log_weights, self.groups
    )
    return self
```

#### __eq__

```python
__eq__(other: object) -> bool
```

Two samples are equal when their fields are.

Source code in `commol/context/inference/sample.py`

```python
def __eq__(self, other: object) -> bool:
    """Two samples are equal when their fields are."""
    if not isinstance(other, ParameterSample):
        return NotImplemented
    return self.model_dump() == other.model_dump()
```

#### equally_weighted

```python
equally_weighted(names: list[str], bounds: list[tuple[float, float]], values: list[list[float]], method: SampleMethod, groups: list[int] | None = None) -> Self
```

A sample whose draws have equal weights.

Parameters:

| Name     | Type                        | Description                   | Default                 |
| -------- | --------------------------- | ----------------------------- | ----------------------- |
| `names`  | `list[str]`                 | One name per column.          | *required*              |
| `bounds` | `list[tuple[float, float]]` | The bounds of each parameter. | *required*              |
| `values` | `list[list[float]]`         | One row per draw.             | *required*              |
| `method` | `SampleMethod`              | What made the sample.         | *required*              |
| `groups` | \`list[int]                 | None\`                        | The group of each draw. |

Returns:

| Type              | Description |
| ----------------- | ----------- |
| `ParameterSample` | The sample. |

Source code in `commol/context/inference/sample.py`

```python
@classmethod
def equally_weighted(
    cls,
    names: list[str],
    bounds: list[tuple[float, float]],
    values: list[list[float]],
    method: SampleMethod,
    groups: list[int] | None = None,
) -> Self:
    """
    A sample whose draws have equal weights.

    Parameters
    ----------
    names : list[str]
        One name per column.
    bounds : list[tuple[float, float]]
        The bounds of each parameter.
    values : list[list[float]]
        One row per draw.
    method : SampleMethod
        What made the sample.
    groups : list[int] | None, optional
        The group of each draw.

    Returns
    -------
    ParameterSample
        The sample.
    """
    return cls(
        names=names,
        bounds=bounds,
        values=values,
        log_weights=[0.0] * len(values),
        groups=groups,
        method=method,
    )
```

#### __len__

```python
__len__() -> int
```

The number of draws.

Source code in `commol/context/inference/sample.py`

```python
def __len__(self) -> int:
    """The number of draws."""
    return len(self.values)
```

#### draws

```python
draws() -> Iterator[dict[str, float]]
```

Every draw, as a mapping of parameter name to value.

Yields:

| Type               | Description                       |
| ------------------ | --------------------------------- |
| `dict[str, float]` | One draw, in the order of values. |

Source code in `commol/context/inference/sample.py`

```python
def draws(self) -> Iterator[dict[str, float]]:
    """
    Every draw, as a mapping of parameter name to value.

    Yields
    ------
    dict[str, float]
        One draw, in the order of `values`.
    """
    for row in self.values:
        yield dict(zip(self.names, row, strict=True))
```

#### mean

```python
mean() -> dict[str, float]
```

The weighted mean of each parameter.

Returns:

| Type               | Description        |
| ------------------ | ------------------ |
| `dict[str, float]` | One mean per name. |

Source code in `commol/context/inference/sample.py`

```python
def mean(self) -> dict[str, float]:
    """
    The weighted mean of each parameter.

    Returns
    -------
    dict[str, float]
        One mean per name.
    """
    return dict(zip(self.names, self._rust.mean(), strict=True))
```

#### covariance

```python
covariance() -> list[list[float]]
```

The weighted covariance matrix, in the order of `names`.

With weights `w` that sum to 1 it is `Σ wᵢ(xᵢ − x̄)(yᵢ − ȳ) / (1 − Σ wᵢ²)`: with equal weights, the sample covariance. It is NaN for a sample of one effective draw.

Returns:

| Type                | Description       |
| ------------------- | ----------------- |
| `list[list[float]]` | One row per name. |

Source code in `commol/context/inference/sample.py`

```python
def covariance(self) -> list[list[float]]:
    """
    The weighted covariance matrix, in the order of `names`.

    With weights ``w`` that sum to 1 it is
    ``Σ wᵢ(xᵢ − x̄)(yᵢ − ȳ) / (1 − Σ wᵢ²)``: with equal weights, the
    sample covariance. It is NaN for a sample of one effective draw.

    Returns
    -------
    list[list[float]]
        One row per name.
    """
    return self._rust.covariance()
```

#### correlation

```python
correlation() -> list[list[float]]
```

The weighted correlation matrix, in the order of `names`.

Returns:

| Type                | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `list[list[float]]` | One row per name; NaN for a parameter with no spread. |

Source code in `commol/context/inference/sample.py`

```python
def correlation(self) -> list[list[float]]:
    """
    The weighted correlation matrix, in the order of `names`.

    Returns
    -------
    list[list[float]]
        One row per name; NaN for a parameter with no spread.
    """
    return self._rust.correlation()
```

#### quantiles

```python
quantiles(levels: list[float]) -> dict[str, list[float]]
```

The weighted quantiles of each parameter.

The quantile is the inverse of the weighted empirical distribution, interpolated between the midpoints of the cumulative weights; with equal weights it is NumPy's `method="hazen"`.

Parameters:

| Name     | Type          | Description                            | Default    |
| -------- | ------------- | -------------------------------------- | ---------- |
| `levels` | `list[float]` | Quantile levels, each between 0 and 1. | *required* |

Returns:

| Type                     | Description                           |
| ------------------------ | ------------------------------------- |
| `dict[str, list[float]]` | One quantile per level for each name. |

Raises:

| Type         | Description                        |
| ------------ | ---------------------------------- |
| `ValueError` | If a level is not between 0 and 1. |

Source code in `commol/context/inference/sample.py`

```python
def quantiles(self, levels: list[float]) -> dict[str, list[float]]:
    """
    The weighted quantiles of each parameter.

    The quantile is the inverse of the weighted empirical distribution,
    interpolated between the midpoints of the cumulative weights; with
    equal weights it is NumPy's ``method="hazen"``.

    Parameters
    ----------
    levels : list[float]
        Quantile levels, each between 0 and 1.

    Returns
    -------
    dict[str, list[float]]
        One quantile per level for each name.

    Raises
    ------
    ValueError
        If a level is not between 0 and 1.
    """
    rows = self._rust.quantiles([float(level) for level in levels])
    return {
        name: [row[column] for row in rows]
        for column, name in enumerate(self.names)
    }
```

#### interval

```python
interval(confidence_level: float) -> dict[str, tuple[float, float]]
```

The central interval of each parameter.

Parameters:

| Name               | Type    | Description                             | Default    |
| ------------------ | ------- | --------------------------------------- | ---------- |
| `confidence_level` | `float` | Strictly between 0 and 1, such as 0.95. | *required* |

Returns:

| Type                             | Description                                                          |
| -------------------------------- | -------------------------------------------------------------------- |
| `dict[str, tuple[float, float]]` | (lower, upper) for each name: the quantiles (1 − c)/2 and (1 + c)/2. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the confidence level is not strictly between 0 and 1. |

Source code in `commol/context/inference/sample.py`

```python
def interval(self, confidence_level: float) -> dict[str, tuple[float, float]]:
    """
    The central interval of each parameter.

    Parameters
    ----------
    confidence_level : float
        Strictly between 0 and 1, such as 0.95.

    Returns
    -------
    dict[str, tuple[float, float]]
        ``(lower, upper)`` for each name: the quantiles
        ``(1 − c)/2`` and ``(1 + c)/2``.

    Raises
    ------
    ValueError
        If the confidence level is not strictly between 0 and 1.
    """
    lower, upper = self._rust.interval(confidence_level)
    return {
        name: (low, high)
        for name, low, high in zip(self.names, lower, upper, strict=True)
    }
```

#### effective_size

```python
effective_size() -> float
```

Kish's effective sample size: `1 / Σ wᵢ²`.

Returns:

| Type    | Description                                                                |
| ------- | -------------------------------------------------------------------------- |
| `float` | Between 1 and the number of draws; equal weights give the number of draws. |

Source code in `commol/context/inference/sample.py`

```python
def effective_size(self) -> float:
    """
    Kish's effective sample size: ``1 / Σ wᵢ²``.

    Returns
    -------
    float
        Between 1 and the number of draws; equal weights give the number
        of draws.
    """
    return self._rust.effective_size()
```

#### resample

```python
resample(count: int, seed: int) -> ParameterSample
```

Draw `count` draws with replacement, each with probability its weight.

Parameters:

| Name    | Type  | Description            | Default    |
| ------- | ----- | ---------------------- | ---------- |
| `count` | `int` | Number of draws.       | *required* |
| `seed`  | `int` | Seed of the generator. | *required* |

Returns:

| Type              | Description                |
| ----------------- | -------------------------- |
| `ParameterSample` | A sample of equal weights. |

Source code in `commol/context/inference/sample.py`

```python
def resample(self, count: int, seed: int) -> "ParameterSample":
    """
    Draw `count` draws with replacement, each with probability its weight.

    Parameters
    ----------
    count : int
        Number of draws.
    seed : int
        Seed of the generator.

    Returns
    -------
    ParameterSample
        A sample of equal weights.
    """
    indices = self._rust.resample(count, seed)
    return ParameterSample.equally_weighted(
        self.names,
        self.bounds,
        [self.values[i] for i in indices],
        SampleMethod(
            name="resample",
            settings={"of": self.method.name, "count": count, "seed": seed},
        ),
    )
```

#### summary

```python
summary(confidence_level: float) -> dict[str, ParameterSummary]
```

Weighted statistics of each parameter, for a report.

Parameters:

| Name               | Type    | Description                                                 | Default    |
| ------------------ | ------- | ----------------------------------------------------------- | ---------- |
| `confidence_level` | `float` | Confidence level of the interval, strictly between 0 and 1. | *required* |

Returns:

| Type                          | Description           |
| ----------------------------- | --------------------- |
| `dict[str, ParameterSummary]` | One summary per name. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the confidence level is not strictly between 0 and 1. |

Source code in `commol/context/inference/sample.py`

```python
def summary(self, confidence_level: float) -> dict[str, ParameterSummary]:
    """
    Weighted statistics of each parameter, for a report.

    Parameters
    ----------
    confidence_level : float
        Confidence level of the interval, strictly between 0 and 1.

    Returns
    -------
    dict[str, ParameterSummary]
        One summary per name.

    Raises
    ------
    ValueError
        If the confidence level is not strictly between 0 and 1.
    """
    lower, upper = self._rust.interval(confidence_level)
    quantiles = self.quantiles([0.0, 0.5, 1.0])
    mean = self.mean()
    covariance = self.covariance()
    return {
        name: ParameterSummary(
            mean=mean[name],
            median=quantiles[name][1],
            std=covariance[index][index] ** 0.5,
            lower=lower[index],
            upper=upper[index],
            min=quantiles[name][0],
            max=quantiles[name][2],
        )
        for index, name in enumerate(self.names)
    }
```

## SampleMethod

Bases: `BaseModel`

What made a sample: the name of the method and its settings.

Attributes:

| Name       | Type                   | Description                                           |
| ---------- | ---------------------- | ----------------------------------------------------- |
| `name`     | `str`                  | The name of the method, such as "bayesian_bootstrap". |
| `settings` | `dict[str, JsonValue]` | The settings the method ran with.                     |

Source code in `commol/context/inference/sample.py`

```python
class SampleMethod(BaseModel):
    """
    What made a sample: the name of the method and its settings.

    Attributes
    ----------
    name : str
        The name of the method, such as ``"bayesian_bootstrap"``.
    settings : dict[str, JsonValue]
        The settings the method ran with.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="The name of the method.")
    settings: dict[str, JsonValue] = Field(
        default_factory=dict, description="The settings the method ran with."
    )
```

## ParameterSummary

Bases: `BaseModel`

Weighted statistics of one parameter of a sample.

Attributes:

| Name     | Type    | Description                                                    |
| -------- | ------- | -------------------------------------------------------------- |
| `mean`   | `float` | The weighted mean.                                             |
| `median` | `float` | The weighted median.                                           |
| `std`    | `float` | The weighted standard deviation (ParameterSample.covariance).  |
| `lower`  | `float` | The lower end of the central interval of the confidence level. |
| `upper`  | `float` | The upper end of the central interval of the confidence level. |
| `min`    | `float` | The smallest value of positive weight.                         |
| `max`    | `float` | The largest value of positive weight.                          |

Source code in `commol/context/inference/sample.py`

```python
class ParameterSummary(BaseModel):
    """
    Weighted statistics of one parameter of a sample.

    Attributes
    ----------
    mean : float
        The weighted mean.
    median : float
        The weighted median.
    std : float
        The weighted standard deviation (`ParameterSample.covariance`).
    lower : float
        The lower end of the central interval of the confidence level.
    upper : float
        The upper end of the central interval of the confidence level.
    min : float
        The smallest value of positive weight.
    max : float
        The largest value of positive weight.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    median: float
    std: float
    lower: float
    upper: float
    min: float
    max: float
```

## Predictive

The run of every draw of a parameter sample.

The draws run in parallel in Rust, through the forward model of the calibration problem: with a stochastic engine every replicate of a draw is a draw of the predictive, with an equal share of its weight. A band reduces the draws position by position with their weights; without observation noise it is a band of the model, not a prediction interval.

Source code in `commol/api/inference/predictive.py`

```python
class Predictive:
    """
    The run of every draw of a parameter sample.

    The draws run in parallel in Rust, through the forward model of the
    calibration problem: with a stochastic engine every replicate of a draw is
    a draw of the predictive, with an equal share of its weight. A band
    reduces the draws position by position with their weights; without
    observation noise it is a band of the model, not a prediction interval.
    """

    def __init__(
        self,
        calibrator: "Calibrator",
        sample: ParameterSample,
        positions: Sequence[float],
    ):
        """
        Run every draw of `sample` at `positions`.

        Parameters
        ----------
        calibrator : Calibrator
            The calibration problem the sample is a sample of.
        sample : ParameterSample
            Its parameters, in the problem's order.
        positions : Sequence[float]
            Positions to report at, ascending.

        Raises
        ------
        ValueError
            If the sample's parameters are not the problem's.
        InfeasibleStateError
            If a draw leaves the values its model allows.
        """
        self.calibrator: "Calibrator" = calibrator
        self.sample: ParameterSample = sample
        self.positions: list[float] = [float(position) for position in positions]
        self.output_names: list[str] = list(calibrator.simulation.engine.output_names)
        self._rust = calibrator.rust_problem.predictive(
            sample.rust_sample, self.positions
        )

    @property
    def weights(self) -> list[float]:
        """The weight of each draw of the predictive; they sum to 1."""
        return self._rust.weights

    def runs(self) -> list[SimulationResults]:
        """
        The trajectory of every draw, in the order of `weights`.

        Returns
        -------
        list[SimulationResults]
            One run per draw, at `positions`.
        """
        return [
            SimulationResults.from_rows(self.positions, self.output_names, rows)
            for rows in self._rust.trajectories
        ]

    def band(self, confidence_level: float) -> PredictiveBand:
        """
        The band of every output.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        PredictiveBand
            The weighted mean, median and central interval of each output at
            each position.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        mean, median, lower, upper = self._rust.band(confidence_level)

        def by_output(table: list[list[float]]) -> dict[str, list[float]]:
            return {
                name: [row[index] for row in table]
                for index, name in enumerate(self.output_names)
            }

        return PredictiveBand(
            confidence_level=confidence_level,
            positions=self.positions,
            mean=by_output(mean),
            median=by_output(median),
            lower=by_output(lower),
            upper=by_output(upper),
        )

    def series_band(
        self,
        outputs: Sequence[str],
        confidence_level: float,
        cumulative: bool = False,
    ) -> SeriesBand:
        """
        The band of the sum of some outputs, or of its running total.

        Each draw is reduced to the series first, then the series are reduced
        across the draws.

        Parameters
        ----------
        outputs : Sequence[str]
            The outputs summed.
        confidence_level : float
            Strictly between 0 and 1.
        cumulative : bool, optional
            Whether the series is the running total along the positions.

        Returns
        -------
        SeriesBand
            One value per position.

        Raises
        ------
        ValueError
            If an output is unknown or the confidence level is out of range.
        """
        unknown = [name for name in outputs if name not in self.output_names]
        if unknown:
            raise ValueError(
                f"Unknown outputs {unknown}. Outputs: {self.output_names}."
            )
        indices = [self.output_names.index(name) for name in outputs]
        mean, median, lower, upper = self._rust.series_band(
            indices, cumulative, confidence_level
        )
        return SeriesBand(
            confidence_level=confidence_level,
            positions=self.positions,
            mean=mean,
            median=median,
            lower=lower,
            upper=upper,
        )

    def at_observations(
        self,
        observed_data: Sequence[ObservedDataPoint],
        noise: ObservationNoise,
    ) -> ObservationDraws:
        """
        Every draw's value of each observation, with observation noise.

        The observations need not be the ones the problem was fitted to: held
        out data are read the same way, through the observation operator.
        The noise is drawn from the calibrator's seed.

        Parameters
        ----------
        observed_data : Sequence[ObservedDataPoint]
            The observations.
        noise : NoNoise | ResidualNoise
            What an observation shows around a draw's prediction, with no
            default: ``NoNoise()`` gives a band of the model,
            ``ResidualNoise(...)`` a prediction interval.

        Returns
        -------
        ObservationDraws
            Each draw's value of each observation.

        Raises
        ------
        ValueError
            If an observation names an unknown output or factor, or a series
            the residual noise has no residuals of.
        InfeasibleStateError
            If a draw leaves the values its model allows.
        """
        residual_noise = None
        if isinstance(noise, ResidualNoise):
            residual_noise = (
                [
                    noise.fit.best_parameters[name]
                    for name in self.calibrator.parameter_names
                ],
                noise.block_length,
                noise.residual_scale,
            )
        draws = self.calibrator.rust_problem.observation_draws(
            self.sample.rust_sample,
            engine_state.rust_observed_data(list(observed_data)),
            self.calibrator.seed,
            residual_noise=residual_noise,
        )
        return ObservationDraws(observed_data, draws)
```

### Attributes

#### weights

```python
weights: list[float]
```

The weight of each draw of the predictive; they sum to 1.

### Methods:

#### __init__

```python
__init__(calibrator: Calibrator, sample: ParameterSample, positions: Sequence[float])
```

Run every draw of `sample` at `positions`.

Parameters:

| Name         | Type              | Description                                        | Default    |
| ------------ | ----------------- | -------------------------------------------------- | ---------- |
| `calibrator` | `Calibrator`      | The calibration problem the sample is a sample of. | *required* |
| `sample`     | `ParameterSample` | Its parameters, in the problem's order.            | *required* |
| `positions`  | `Sequence[float]` | Positions to report at, ascending.                 | *required* |

Raises:

| Type                   | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `ValueError`           | If the sample's parameters are not the problem's. |
| `InfeasibleStateError` | If a draw leaves the values its model allows.     |

Source code in `commol/api/inference/predictive.py`

```python
def __init__(
    self,
    calibrator: "Calibrator",
    sample: ParameterSample,
    positions: Sequence[float],
):
    """
    Run every draw of `sample` at `positions`.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem the sample is a sample of.
    sample : ParameterSample
        Its parameters, in the problem's order.
    positions : Sequence[float]
        Positions to report at, ascending.

    Raises
    ------
    ValueError
        If the sample's parameters are not the problem's.
    InfeasibleStateError
        If a draw leaves the values its model allows.
    """
    self.calibrator: "Calibrator" = calibrator
    self.sample: ParameterSample = sample
    self.positions: list[float] = [float(position) for position in positions]
    self.output_names: list[str] = list(calibrator.simulation.engine.output_names)
    self._rust = calibrator.rust_problem.predictive(
        sample.rust_sample, self.positions
    )
```

#### runs

```python
runs() -> list[SimulationResults]
```

The trajectory of every draw, in the order of `weights`.

Returns:

| Type                      | Description                     |
| ------------------------- | ------------------------------- |
| `list[SimulationResults]` | One run per draw, at positions. |

Source code in `commol/api/inference/predictive.py`

```python
def runs(self) -> list[SimulationResults]:
    """
    The trajectory of every draw, in the order of `weights`.

    Returns
    -------
    list[SimulationResults]
        One run per draw, at `positions`.
    """
    return [
        SimulationResults.from_rows(self.positions, self.output_names, rows)
        for rows in self._rust.trajectories
    ]
```

#### band

```python
band(confidence_level: float) -> PredictiveBand
```

The band of every output.

Parameters:

| Name               | Type    | Description               | Default    |
| ------------------ | ------- | ------------------------- | ---------- |
| `confidence_level` | `float` | Strictly between 0 and 1. | *required* |

Returns:

| Type             | Description                                                                     |
| ---------------- | ------------------------------------------------------------------------------- |
| `PredictiveBand` | The weighted mean, median and central interval of each output at each position. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the confidence level is not strictly between 0 and 1. |

Source code in `commol/api/inference/predictive.py`

```python
def band(self, confidence_level: float) -> PredictiveBand:
    """
    The band of every output.

    Parameters
    ----------
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    PredictiveBand
        The weighted mean, median and central interval of each output at
        each position.

    Raises
    ------
    ValueError
        If the confidence level is not strictly between 0 and 1.
    """
    mean, median, lower, upper = self._rust.band(confidence_level)

    def by_output(table: list[list[float]]) -> dict[str, list[float]]:
        return {
            name: [row[index] for row in table]
            for index, name in enumerate(self.output_names)
        }

    return PredictiveBand(
        confidence_level=confidence_level,
        positions=self.positions,
        mean=by_output(mean),
        median=by_output(median),
        lower=by_output(lower),
        upper=by_output(upper),
    )
```

#### series_band

```python
series_band(outputs: Sequence[str], confidence_level: float, cumulative: bool = False) -> SeriesBand
```

The band of the sum of some outputs, or of its running total.

Each draw is reduced to the series first, then the series are reduced across the draws.

Parameters:

| Name               | Type            | Description                                                  | Default    |
| ------------------ | --------------- | ------------------------------------------------------------ | ---------- |
| `outputs`          | `Sequence[str]` | The outputs summed.                                          | *required* |
| `confidence_level` | `float`         | Strictly between 0 and 1.                                    | *required* |
| `cumulative`       | `bool`          | Whether the series is the running total along the positions. | `False`    |

Returns:

| Type         | Description             |
| ------------ | ----------------------- |
| `SeriesBand` | One value per position. |

Raises:

| Type         | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| `ValueError` | If an output is unknown or the confidence level is out of range. |

Source code in `commol/api/inference/predictive.py`

```python
def series_band(
    self,
    outputs: Sequence[str],
    confidence_level: float,
    cumulative: bool = False,
) -> SeriesBand:
    """
    The band of the sum of some outputs, or of its running total.

    Each draw is reduced to the series first, then the series are reduced
    across the draws.

    Parameters
    ----------
    outputs : Sequence[str]
        The outputs summed.
    confidence_level : float
        Strictly between 0 and 1.
    cumulative : bool, optional
        Whether the series is the running total along the positions.

    Returns
    -------
    SeriesBand
        One value per position.

    Raises
    ------
    ValueError
        If an output is unknown or the confidence level is out of range.
    """
    unknown = [name for name in outputs if name not in self.output_names]
    if unknown:
        raise ValueError(
            f"Unknown outputs {unknown}. Outputs: {self.output_names}."
        )
    indices = [self.output_names.index(name) for name in outputs]
    mean, median, lower, upper = self._rust.series_band(
        indices, cumulative, confidence_level
    )
    return SeriesBand(
        confidence_level=confidence_level,
        positions=self.positions,
        mean=mean,
        median=median,
        lower=lower,
        upper=upper,
    )
```

#### at_observations

```python
at_observations(observed_data: Sequence[ObservedDataPoint], noise: ObservationNoise) -> ObservationDraws
```

Every draw's value of each observation, with observation noise.

The observations need not be the ones the problem was fitted to: held out data are read the same way, through the observation operator. The noise is drawn from the calibrator's seed.

Parameters:

| Name            | Type                          | Description       | Default                                                                                                                                               |
| --------------- | ----------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observed_data` | `Sequence[ObservedDataPoint]` | The observations. | *required*                                                                                                                                            |
| `noise`         | \`NoNoise                     | ResidualNoise\`   | What an observation shows around a draw's prediction, with no default: NoNoise() gives a band of the model, ResidualNoise(...) a prediction interval. |

Returns:

| Type               | Description                            |
| ------------------ | -------------------------------------- |
| `ObservationDraws` | Each draw's value of each observation. |

Raises:

| Type                   | Description                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| `ValueError`           | If an observation names an unknown output or factor, or a series the residual noise has no residuals of. |
| `InfeasibleStateError` | If a draw leaves the values its model allows.                                                            |

Source code in `commol/api/inference/predictive.py`

```python
def at_observations(
    self,
    observed_data: Sequence[ObservedDataPoint],
    noise: ObservationNoise,
) -> ObservationDraws:
    """
    Every draw's value of each observation, with observation noise.

    The observations need not be the ones the problem was fitted to: held
    out data are read the same way, through the observation operator.
    The noise is drawn from the calibrator's seed.

    Parameters
    ----------
    observed_data : Sequence[ObservedDataPoint]
        The observations.
    noise : NoNoise | ResidualNoise
        What an observation shows around a draw's prediction, with no
        default: ``NoNoise()`` gives a band of the model,
        ``ResidualNoise(...)`` a prediction interval.

    Returns
    -------
    ObservationDraws
        Each draw's value of each observation.

    Raises
    ------
    ValueError
        If an observation names an unknown output or factor, or a series
        the residual noise has no residuals of.
    InfeasibleStateError
        If a draw leaves the values its model allows.
    """
    residual_noise = None
    if isinstance(noise, ResidualNoise):
        residual_noise = (
            [
                noise.fit.best_parameters[name]
                for name in self.calibrator.parameter_names
            ],
            noise.block_length,
            noise.residual_scale,
        )
    draws = self.calibrator.rust_problem.observation_draws(
        self.sample.rust_sample,
        engine_state.rust_observed_data(list(observed_data)),
        self.calibrator.seed,
        residual_noise=residual_noise,
    )
    return ObservationDraws(observed_data, draws)
```

## PredictiveBand

Bases: `BaseModel`

The band of every output of a predictive.

Attributes:

| Name                         | Type                     | Description                                                                           |
| ---------------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| `confidence_level`           | `float`                  | The level of the central interval.                                                    |
| `positions`                  | `list[float]`            | Where the values are.                                                                 |
| `mean, median, lower, upper` | `dict[str, list[float]]` | For each output, one value per position, reduced across the draws with their weights. |

Source code in `commol/api/inference/predictive.py`

```python
class PredictiveBand(BaseModel):
    """
    The band of every output of a predictive.

    Attributes
    ----------
    confidence_level : float
        The level of the central interval.
    positions : list[float]
        Where the values are.
    mean, median, lower, upper : dict[str, list[float]]
        For each output, one value per position, reduced across the draws
        with their weights.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_level: float
    positions: list[float]
    mean: dict[str, list[float]] = Field(description="Weighted mean")
    median: dict[str, list[float]] = Field(description="Weighted median")
    lower: dict[str, list[float]] = Field(description="Lower quantile")
    upper: dict[str, list[float]] = Field(description="Upper quantile")
```

## SeriesBand

Bases: `BaseModel`

The band of one series: four values at each of its positions.

Attributes:

| Name               | Type          | Description                         |
| ------------------ | ------------- | ----------------------------------- |
| `confidence_level` | `float`       | The level of the central interval.  |
| `positions`        | `list[float]` | Where the values are.               |
| `mean`             | `list[float]` | The weighted mean across the draws. |
| `median`           | `list[float]` | The weighted median.                |
| `lower`            | `list[float]` | The quantile (1 − c)/2.             |
| `upper`            | `list[float]` | The quantile (1 + c)/2.             |

Source code in `commol/api/inference/predictive.py`

```python
class SeriesBand(BaseModel):
    """
    The band of one series: four values at each of its positions.

    Attributes
    ----------
    confidence_level : float
        The level of the central interval.
    positions : list[float]
        Where the values are.
    mean : list[float]
        The weighted mean across the draws.
    median : list[float]
        The weighted median.
    lower : list[float]
        The quantile ``(1 − c)/2``.
    upper : list[float]
        The quantile ``(1 + c)/2``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_level: float
    positions: list[float]
    mean: list[float]
    median: list[float]
    lower: list[float]
    upper: list[float]
```

## ObservationDraws

Every draw's value of a set of observations.

The values are read through the observation operator of the calibration problem (sums of outputs, windows, factors), the one the loss uses. Without observation noise their interval is a band of the model, not a prediction interval.

Source code in `commol/api/inference/predictive.py`

```python
class ObservationDraws:
    """
    Every draw's value of a set of observations.

    The values are read through the observation operator of the calibration
    problem (sums of outputs, windows, factors), the one the loss uses. Without
    observation noise their interval is a band of the model, not a prediction
    interval.
    """

    def __init__(
        self,
        observations: Sequence[ObservedDataPoint],
        draws: "ObservationDrawsProtocol",
    ):
        """
        Wrap the draws of `observations`.

        Parameters
        ----------
        observations : Sequence[ObservedDataPoint]
            The observations, in the order of each draw's values.
        draws : ObservationDrawsProtocol
            The draws, computed in Rust.
        """
        self.observations: list[ObservedDataPoint] = list(observations)
        self.rust_draws: "ObservationDrawsProtocol" = draws

    @property
    def values(self) -> list[list[float]]:
        """Each draw's value of each observation, ``[draw][observation]``."""
        return self.rust_draws.values

    @property
    def weights(self) -> list[float]:
        """The weight of each draw; they sum to 1."""
        return self.rust_draws.weights

    @property
    def observed(self) -> list[float]:
        """The observed value of each observation."""
        return [point.value for point in self.observations]

    def band(self, confidence_level: float) -> SeriesBand:
        """
        The band of each observation across the draws.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        SeriesBand
            One value per observation, at its position.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        mean, median, lower, upper = self.rust_draws.band(confidence_level)
        return SeriesBand(
            confidence_level=confidence_level,
            positions=[point.at for point in self.observations],
            mean=mean,
            median=median,
            lower=lower,
            upper=upper,
        )
```

### Attributes

#### values

```python
values: list[list[float]]
```

Each draw's value of each observation, `[draw][observation]`.

#### weights

```python
weights: list[float]
```

The weight of each draw; they sum to 1.

#### observed

```python
observed: list[float]
```

The observed value of each observation.

### Methods:

#### __init__

```python
__init__(observations: Sequence[ObservedDataPoint], draws: ObservationDrawsProtocol)
```

Wrap the draws of `observations`.

Parameters:

| Name           | Type                          | Description                                           | Default    |
| -------------- | ----------------------------- | ----------------------------------------------------- | ---------- |
| `observations` | `Sequence[ObservedDataPoint]` | The observations, in the order of each draw's values. | *required* |
| `draws`        | `ObservationDrawsProtocol`    | The draws, computed in Rust.                          | *required* |

Source code in `commol/api/inference/predictive.py`

```python
def __init__(
    self,
    observations: Sequence[ObservedDataPoint],
    draws: "ObservationDrawsProtocol",
):
    """
    Wrap the draws of `observations`.

    Parameters
    ----------
    observations : Sequence[ObservedDataPoint]
        The observations, in the order of each draw's values.
    draws : ObservationDrawsProtocol
        The draws, computed in Rust.
    """
    self.observations: list[ObservedDataPoint] = list(observations)
    self.rust_draws: "ObservationDrawsProtocol" = draws
```

#### band

```python
band(confidence_level: float) -> SeriesBand
```

The band of each observation across the draws.

Parameters:

| Name               | Type    | Description               | Default    |
| ------------------ | ------- | ------------------------- | ---------- |
| `confidence_level` | `float` | Strictly between 0 and 1. | *required* |

Returns:

| Type         | Description                                 |
| ------------ | ------------------------------------------- |
| `SeriesBand` | One value per observation, at its position. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the confidence level is not strictly between 0 and 1. |

Source code in `commol/api/inference/predictive.py`

```python
def band(self, confidence_level: float) -> SeriesBand:
    """
    The band of each observation across the draws.

    Parameters
    ----------
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    SeriesBand
        One value per observation, at its position.

    Raises
    ------
    ValueError
        If the confidence level is not strictly between 0 and 1.
    """
    mean, median, lower, upper = self.rust_draws.band(confidence_level)
    return SeriesBand(
        confidence_level=confidence_level,
        positions=[point.at for point in self.observations],
        mean=mean,
        median=median,
        lower=lower,
        upper=upper,
    )
```

## NoNoise

Bases: `BaseModel`

No observation noise: an observation shows the prediction of each draw.

The band of a predictive with `NoNoise` is a band of the model. It says where the model curve lies, not where the next observation falls, so its coverage of data is below its level.

Source code in `commol/context/inference/noise.py`

```python
class NoNoise(BaseModel):
    """
    No observation noise: an observation shows the prediction of each draw.

    The band of a predictive with `NoNoise` is a band of the model. It says
    where the model curve lies, not where the next observation falls, so its
    coverage of data is below its level.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
```

## ResidualNoise

Bases: `BaseModel`

Observation noise made of the residuals of a fit, with no noise model.

Each draw adds moving blocks of `block_length` positions of residuals of the fit, drawn with replacement across every series at once, to its predictions, on `residual_scale`. The band is then a prediction interval. It serves the series the fit has residuals of, at any position: the residual of a series at a drawn position is the one at the nearest position where the series was observed.

Attributes:

| Name             | Type                         | Description                                                                                                                                                            |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit`            | `CalibrationResult`          | The fit whose residuals are added.                                                                                                                                     |
| `block_length`   | `int`                        | Positions per block, at least 1 and at most the number of distinct positions of the problem's observations. A bootstrap reports the length it used in its diagnostics. |
| `residual_scale` | `{additive, multiplicative}` | "additive" adds r = y − ŷ; "multiplicative" multiplies by exp(r) with r = log(y / ŷ) and needs positive observed and fitted values.                                    |

Source code in `commol/context/inference/noise.py`

```python
class ResidualNoise(BaseModel):
    """
    Observation noise made of the residuals of a fit, with no noise model.

    Each draw adds moving blocks of `block_length` positions of residuals of
    the fit, drawn with replacement across every series at once, to its
    predictions, on `residual_scale`. The band is then a prediction interval.
    It serves the series the fit has residuals of, at any position: the
    residual of a series at a drawn position is the one at the nearest
    position where the series was observed.

    Attributes
    ----------
    fit : CalibrationResult
        The fit whose residuals are added.
    block_length : int
        Positions per block, at least 1 and at most the number of distinct
        positions of the problem's observations. A bootstrap reports the
        length it used in its diagnostics.
    residual_scale : {"additive", "multiplicative"}
        ``"additive"`` adds ``r = y − ŷ``; ``"multiplicative"`` multiplies by
        ``exp(r)`` with ``r = log(y / ŷ)`` and needs positive observed and
        fitted values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult = Field(description="The fit whose residuals are added.")
    block_length: int = Field(ge=1, description="Positions per block.")
    residual_scale: Literal["additive", "multiplicative"] = Field(
        description="The scale of the residuals."
    )
```

## Resampling

Guide: [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/index.md).

## BayesianBootstrap

Bases: `UncertaintyMethod[BootstrapResult]`

The Bayesian (weighted) bootstrap by blocks.

Each replicate draws one Dirichlet(1, …, 1) weight per block, scaled to mean 1, multiplies the weight of every observation of the block by it, and fits. With a loss it is the loss-likelihood bootstrap of Lyddon, Holmes and Walker (2019); with one position per block, the Bayesian bootstrap of Rubin (1981) and the weighted likelihood bootstrap of Newton and Raftery (1994). The observed values never change.

Attributes:

| Name           | Type  | Description                                                                                                   |
| -------------- | ----- | ------------------------------------------------------------------------------------------------------------- |
| `replicates`   | `int` | Number of replicates, at least 1.                                                                             |
| `block_length` | \`int | auto\`                                                                                                        |
| `starts`       | `int` | Starts of the fit and of each replicate (default 1): the fit, then rows of a Latin hypercube over the bounds. |

Source code in `commol/context/inference/resampling.py`

```python
class BayesianBootstrap(UncertaintyMethod[BootstrapResult]):
    """
    The Bayesian (weighted) bootstrap by blocks.

    Each replicate draws one Dirichlet(1, …, 1) weight per block, scaled to
    mean 1, multiplies the weight of every observation of the block by it,
    and fits. With a loss it is the loss-likelihood bootstrap of Lyddon,
    Holmes and Walker (2019); with one position per block, the Bayesian
    bootstrap of Rubin (1981) and the weighted likelihood bootstrap of Newton
    and Raftery (1994). The observed values never change.

    Attributes
    ----------
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block.
    starts : int
        Starts of the fit and of each replicate (default 1): the fit, then
        rows of a Latin hypercube over the bounds.
    """

    replicates: int = Field(ge=1, description="Number of replicates.")
    block_length: BlockLength = Field(description="Positions per block.")
    starts: int = Field(default=1, ge=1, description="Starts per fit.")
```

## ResidualBootstrap

Bases: `UncertaintyMethod[BootstrapResult]`

The residual block bootstrap.

Each replicate joins blocks of residuals of the fit, drawn with replacement across every series at once, adds them to the fitted values on `residual_scale`, and fits the synthetic data.

Attributes:

| Name             | Type                         | Description                                                                                                                                                                              |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `replicates`     | `int`                        | Number of replicates, at least 1.                                                                                                                                                        |
| `block_length`   | \`int                        | auto\`                                                                                                                                                                                   |
| `residual_scale` | `{additive, multiplicative}` | "additive": r = y − ŷ and ŷ + r\*. "multiplicative": r = log(y / ŷ) and ŷ · exp(r\*); it needs positive observed and fitted values and keeps synthetic values positive.                  |
| `scheme`         | `{moving, stationary}`       | "moving": blocks of block_length consecutive positions (Künsch 1989). "stationary": blocks of geometric length with mean block_length, wrapped around the end (Politis and Romano 1994). |
| `starts`         | `int`                        | Starts of the fit and of each replicate (default 1).                                                                                                                                     |

Source code in `commol/context/inference/resampling.py`

```python
class ResidualBootstrap(UncertaintyMethod[BootstrapResult]):
    """
    The residual block bootstrap.

    Each replicate joins blocks of residuals of the fit, drawn with
    replacement across every series at once, adds them to the fitted values
    on `residual_scale`, and fits the synthetic data.

    Attributes
    ----------
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block; with ``scheme="stationary"`` the mean length.
    residual_scale : {"additive", "multiplicative"}
        ``"additive"``: ``r = y − ŷ`` and ``ŷ + r*``. ``"multiplicative"``:
        ``r = log(y / ŷ)`` and ``ŷ · exp(r*)``; it needs positive observed and
        fitted values and keeps synthetic values positive.
    scheme : {"moving", "stationary"}
        ``"moving"``: blocks of `block_length` consecutive positions (Künsch
        1989). ``"stationary"``: blocks of geometric length with mean
        `block_length`, wrapped around the end (Politis and Romano 1994).
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    replicates: int = Field(ge=1, description="Number of replicates.")
    block_length: BlockLength = Field(description="Positions per block.")
    residual_scale: Literal["additive", "multiplicative"] = Field(
        description="The scale of the residuals."
    )
    scheme: Literal["moving", "stationary"] = Field(
        description="How blocks of residuals are drawn."
    )
    starts: int = Field(default=1, ge=1, description="Starts per fit.")
```

## WildBootstrap

Bases: `UncertaintyMethod[BootstrapResult]`

The wild block bootstrap.

Each replicate multiplies the additive residuals of each block by one draw of `multipliers` and fits `ŷ + r · v`. Every residual stays at its own position, so it holds under noise whose size changes along the series.

Attributes:

| Name           | Type                   | Description                                                                                                                                                      |
| -------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `replicates`   | `int`                  | Number of replicates, at least 1.                                                                                                                                |
| `block_length` | \`int                  | auto\`                                                                                                                                                           |
| `multipliers`  | `{rademacher, mammen}` | "rademacher": ±1 with probability 1/2. "mammen": two values with mean 0, variance 1 and third moment 1 (Mammen 1993), which keeps the skewness of the residuals. |
| `starts`       | `int`                  | Starts of the fit and of each replicate (default 1).                                                                                                             |

Source code in `commol/context/inference/resampling.py`

```python
class WildBootstrap(UncertaintyMethod[BootstrapResult]):
    """
    The wild block bootstrap.

    Each replicate multiplies the additive residuals of each block by one
    draw of `multipliers` and fits ``ŷ + r · v``. Every residual stays at its
    own position, so it holds under noise whose size changes along the
    series.

    Attributes
    ----------
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block.
    multipliers : {"rademacher", "mammen"}
        ``"rademacher"``: ±1 with probability 1/2. ``"mammen"``: two values
        with mean 0, variance 1 and third moment 1 (Mammen 1993), which keeps
        the skewness of the residuals.
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    replicates: int = Field(ge=1, description="Number of replicates.")
    block_length: BlockLength = Field(description="Positions per block.")
    multipliers: Literal["rademacher", "mammen"] = Field(
        description="The distribution of the multipliers."
    )
    starts: int = Field(default=1, ge=1, description="Starts per fit.")
```

## BlockJackknife

Bases: `UncertaintyMethod[JackknifeResult]`

The block jackknife (Künsch 1989): one fit per block left out.

It gives a standard error and a bias of each parameter, and the influence of each block, with as many fits as blocks. A block is left out by giving its observations weight 0.

Attributes:

| Name           | Type  | Description                                              |
| -------------- | ----- | -------------------------------------------------------- |
| `block_length` | \`int | auto\`                                                   |
| `starts`       | `int` | Starts of the fit and of each leave-out fit (default 1). |

Source code in `commol/context/inference/resampling.py`

```python
class BlockJackknife(UncertaintyMethod[JackknifeResult]):
    """
    The block jackknife (Künsch 1989): one fit per block left out.

    It gives a standard error and a bias of each parameter, and the influence
    of each block, with as many fits as blocks. A block is left out by giving
    its observations weight 0.

    Attributes
    ----------
    block_length : int | "auto"
        Positions per block; it must leave at least two blocks.
    starts : int
        Starts of the fit and of each leave-out fit (default 1).
    """

    block_length: BlockLength = Field(description="Positions per block.")
    starts: int = Field(default=1, ge=1, description="Starts per fit.")
```

## BootstrapResult

Bases: `BaseModel`

The result of a bootstrap.

Attributes:

| Name          | Type                    | Description                                                                                         |
| ------------- | ----------------------- | --------------------------------------------------------------------------------------------------- |
| `fit`         | `CalibrationResult`     | The fit every replicate starts from: Calibrator.run(starts).                                        |
| `sample`      | `ParameterSample`       | One draw per replicate that did not fail, with equal weights; the group of a draw is its replicate. |
| `diagnostics` | `ResamplingDiagnostics` | The block length, and the replicates that failed, did not converge or made values below zero.       |

Source code in `commol/context/inference/resampling.py`

```python
class BootstrapResult(BaseModel):
    """
    The result of a bootstrap.

    Attributes
    ----------
    fit : CalibrationResult
        The fit every replicate starts from: ``Calibrator.run(starts)``.
    sample : ParameterSample
        One draw per replicate that did not fail, with equal weights; the
        group of a draw is its replicate.
    diagnostics : ResamplingDiagnostics
        The block length, and the replicates that failed, did not converge or
        made values below zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    sample: ParameterSample
    diagnostics: ResamplingDiagnostics

    def interval_error(self, confidence_level: float) -> dict[str, BoundError]:
        """
        The Monte Carlo error of the central interval of each parameter.

        More replicates make it smaller; when it is wide against the interval,
        run more.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        dict[str, BoundError]
            The error of each bound, by parameter.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"A confidence level is strictly between 0 and 1, got "
                f"{confidence_level}."
            )
        low, high = (1.0 - confidence_level) / 2.0, (1.0 + confidence_level) / 2.0
        errors: dict[str, BoundError] = {}
        for index, name in enumerate(self.sample.names):
            column = [row[index] for row in self.sample.values]
            errors[name] = BoundError(
                lower=commol_rs.inference.quantile_error(column, low),
                upper=commol_rs.inference.quantile_error(column, high),
            )
        return errors
```

### Methods:

#### interval_error

```python
interval_error(confidence_level: float) -> dict[str, BoundError]
```

The Monte Carlo error of the central interval of each parameter.

More replicates make it smaller; when it is wide against the interval, run more.

Parameters:

| Name               | Type    | Description               | Default    |
| ------------------ | ------- | ------------------------- | ---------- |
| `confidence_level` | `float` | Strictly between 0 and 1. | *required* |

Returns:

| Type                    | Description                            |
| ----------------------- | -------------------------------------- |
| `dict[str, BoundError]` | The error of each bound, by parameter. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the confidence level is not strictly between 0 and 1. |

Source code in `commol/context/inference/resampling.py`

```python
def interval_error(self, confidence_level: float) -> dict[str, BoundError]:
    """
    The Monte Carlo error of the central interval of each parameter.

    More replicates make it smaller; when it is wide against the interval,
    run more.

    Parameters
    ----------
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    dict[str, BoundError]
        The error of each bound, by parameter.

    Raises
    ------
    ValueError
        If the confidence level is not strictly between 0 and 1.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"A confidence level is strictly between 0 and 1, got "
            f"{confidence_level}."
        )
    low, high = (1.0 - confidence_level) / 2.0, (1.0 + confidence_level) / 2.0
    errors: dict[str, BoundError] = {}
    for index, name in enumerate(self.sample.names):
        column = [row[index] for row in self.sample.values]
        errors[name] = BoundError(
            lower=commol_rs.inference.quantile_error(column, low),
            upper=commol_rs.inference.quantile_error(column, high),
        )
    return errors
```

## JackknifeResult

Bases: `BaseModel`

The result of a block jackknife.

With `B` fits `θ̂₍₋ᵦ₎` that leave one block out, around the fit `θ̂`, and their mean `θ̄`: the bias is `(B − 1)(θ̄ − θ̂)`, the standard error `√((B − 1)/B · Σ (θ̂₍₋ᵦ₎ − θ̄)²)`, and the pseudo-value of block `b` is `B θ̂ − (B − 1) θ̂₍₋ᵦ₎`. The pseudo-values are not draws: their percentiles are not an interval.

Attributes:

| Name             | Type                        | Description                                                                                                                                           |
| ---------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit`            | `CalibrationResult`         | The fit with every observation.                                                                                                                       |
| `estimate`       | `dict[str, float]`          | The bias-corrected estimate of each parameter.                                                                                                        |
| `bias`           | `dict[str, float]`          | The jackknife bias of each parameter.                                                                                                                 |
| `standard_error` | `dict[str, float]`          | The jackknife standard error of each parameter.                                                                                                       |
| `blocks`         | `list[tuple[float, float]]` | The first and last position of each block that was left out and fitted, in order.                                                                     |
| `pseudo_values`  | `dict[str, list[float]]`    | The pseudo-values of each parameter, one per block.                                                                                                   |
| `influence`      | `dict[str, list[float]]`    | How far leaving each block out moves each parameter, in standard errors: (θ̂₍₋ᵦ₎ − θ̂) / se, one per block. NaN or infinite when a standard error is 0. |
| `diagnostics`    | `ResamplingDiagnostics`     | The block length and the fits that failed or did not converge.                                                                                        |

Source code in `commol/context/inference/resampling.py`

```python
class JackknifeResult(BaseModel):
    """
    The result of a block jackknife.

    With ``B`` fits ``θ̂₍₋ᵦ₎`` that leave one block out, around the fit
    ``θ̂``, and their mean ``θ̄``: the bias is ``(B − 1)(θ̄ − θ̂)``, the
    standard error ``√((B − 1)/B · Σ (θ̂₍₋ᵦ₎ − θ̄)²)``, and the pseudo-value of
    block ``b`` is ``B θ̂ − (B − 1) θ̂₍₋ᵦ₎``. The pseudo-values are not draws:
    their percentiles are not an interval.

    Attributes
    ----------
    fit : CalibrationResult
        The fit with every observation.
    estimate : dict[str, float]
        The bias-corrected estimate of each parameter.
    bias : dict[str, float]
        The jackknife bias of each parameter.
    standard_error : dict[str, float]
        The jackknife standard error of each parameter.
    blocks : list[tuple[float, float]]
        The first and last position of each block that was left out and
        fitted, in order.
    pseudo_values : dict[str, list[float]]
        The pseudo-values of each parameter, one per block.
    influence : dict[str, list[float]]
        How far leaving each block out moves each parameter, in standard
        errors: ``(θ̂₍₋ᵦ₎ − θ̂) / se``, one per block. NaN or infinite when a
        standard error is 0.
    diagnostics : ResamplingDiagnostics
        The block length and the fits that failed or did not converge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    fit: CalibrationResult
    estimate: dict[str, float]
    bias: dict[str, float]
    standard_error: dict[str, float]
    blocks: list[tuple[float, float]]
    pseudo_values: dict[str, list[float]]
    influence: dict[str, list[float]]
    diagnostics: ResamplingDiagnostics
```

## ResamplingDiagnostics

Bases: `BaseModel`

What happened to the replicates of a resampling method.

Attributes:

| Name                     | Type               | Description                                                                                                                       |
| ------------------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `block_length`           | `int`              | The block length the method ran with.                                                                                             |
| `block_length_estimates` | \`dict[str, float] | None\`                                                                                                                            |
| `failed`                 | `list[str]`        | Each replicate that failed, with its reason. A failed replicate is not in the sample.                                             |
| `not_converged`          | `int`              | Replicates kept although their optimizer did not converge.                                                                        |
| `negative_values`        | `int`              | Synthetic observed values below zero, over every replicate. They are kept; many of them point to residual_scale="multiplicative". |

Source code in `commol/context/inference/resampling.py`

```python
class ResamplingDiagnostics(BaseModel):
    """
    What happened to the replicates of a resampling method.

    Attributes
    ----------
    block_length : int
        The block length the method ran with.
    block_length_estimates : dict[str, float] | None
        With ``block_length="auto"``, the estimate of each series, by label;
        otherwise None.
    failed : list[str]
        Each replicate that failed, with its reason. A failed replicate is not
        in the sample.
    not_converged : int
        Replicates kept although their optimizer did not converge.
    negative_values : int
        Synthetic observed values below zero, over every replicate. They are
        kept; many of them point to ``residual_scale="multiplicative"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_length: int = Field(description="The block length used.")
    block_length_estimates: dict[str, float] | None = Field(
        description="The estimate of each series with the automatic length."
    )
    failed: list[str] = Field(description="Each replicate that failed.")
    not_converged: int = Field(description="Replicates that did not converge.")
    negative_values: int = Field(description="Synthetic values below zero.")
```

## BoundError

Bases: `BaseModel`

The Monte Carlo error of the bounds of an interval: for each bound, an order-statistic interval that holds the bound of infinitely many replicates with about 95 % probability.

Attributes:

| Name    | Type                  | Description                            |
| ------- | --------------------- | -------------------------------------- |
| `lower` | `tuple[float, float]` | The error interval of the lower bound. |
| `upper` | `tuple[float, float]` | The error interval of the upper bound. |

Source code in `commol/context/inference/resampling.py`

```python
class BoundError(BaseModel):
    """
    The Monte Carlo error of the bounds of an interval: for each bound, an
    order-statistic interval that holds the bound of infinitely many
    replicates with about 95 % probability.

    Attributes
    ----------
    lower : tuple[float, float]
        The error interval of the lower bound.
    upper : tuple[float, float]
        The error interval of the upper bound.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: tuple[float, float]
    upper: tuple[float, float]
```

## Sequential Monte Carlo

Guide: [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/index.md).

## Priors

Bases: `BaseModel`

One prior per calibrated parameter.

A sampling method takes its priors with no default: what is known about each parameter before the data is a modelling choice. Give a mapping of parameter to distribution, or `uniform_on_bounds` for `Uniform()` on every parameter.

Attributes:

| Name            | Type                      | Description |
| --------------- | ------------------------- | ----------- |
| `distributions` | \`dict[str, Distribution] | None\`      |

Source code in `commol/context/inference/distributions.py`

```python
class Priors(BaseModel):
    """
    One prior per calibrated parameter.

    A sampling method takes its priors with no default: what is known about
    each parameter before the data is a modelling choice. Give a mapping of
    parameter to distribution, or `uniform_on_bounds` for ``Uniform()`` on
    every parameter.

    Attributes
    ----------
    distributions : dict[str, Distribution] | None
        The prior of each parameter; None gives ``Uniform()`` to every
        parameter (`uniform_on_bounds`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    distributions: dict[str, Distribution] | None = Field(
        description="The prior of each parameter, or None for uniform on the bounds."
    )

    @classmethod
    def uniform_on_bounds(cls) -> Self:
        """
        ``Uniform()`` on the bounds of every parameter.

        Returns
        -------
        Priors
            The priors; every parameter needs finite bounds.
        """
        return cls(distributions=None)

    def rust_priors(self) -> list[tuple[str, str, list[float]]] | None:
        """The priors as the Rust runners read them; None is uniform."""
        if self.distributions is None:
            return None
        return [
            distribution.rust_prior(name)
            for name, distribution in self.distributions.items()
        ]
```

### Methods:

#### uniform_on_bounds

```python
uniform_on_bounds() -> Self
```

`Uniform()` on the bounds of every parameter.

Returns:

| Type     | Description                                      |
| -------- | ------------------------------------------------ |
| `Priors` | The priors; every parameter needs finite bounds. |

Source code in `commol/context/inference/distributions.py`

```python
@classmethod
def uniform_on_bounds(cls) -> Self:
    """
    ``Uniform()`` on the bounds of every parameter.

    Returns
    -------
    Priors
        The priors; every parameter needs finite bounds.
    """
    return cls(distributions=None)
```

#### rust_priors

```python
rust_priors() -> list[tuple[str, str, list[float]]] | None
```

The priors as the Rust runners read them; None is uniform.

Source code in `commol/context/inference/distributions.py`

```python
def rust_priors(self) -> list[tuple[str, str, list[float]]] | None:
    """The priors as the Rust runners read them; None is uniform."""
    if self.distributions is None:
        return None
    return [
        distribution.rust_prior(name)
        for name, distribution in self.distributions.items()
    ]
```

## Uniform

Bases: `DistributionBase`

The uniform distribution on the bounds; they must be finite.

Source code in `commol/context/inference/distributions.py`

```python
class Uniform(DistributionBase):
    """The uniform distribution on the bounds; they must be finite."""

    rust_kind: ClassVar[str] = "uniform"
    kind: Literal["uniform"] = "uniform"
```

## LogUniform

Bases: `DistributionBase`

The log-uniform distribution on the bounds: `log x` is uniform, the density is proportional to `1/x`. The bounds must be finite and the lower one above 0.

Source code in `commol/context/inference/distributions.py`

```python
class LogUniform(DistributionBase):
    """
    The log-uniform distribution on the bounds: ``log x`` is uniform, the
    density is proportional to ``1/x``. The bounds must be finite and the
    lower one above 0.
    """

    rust_kind: ClassVar[str] = "log_uniform"
    kind: Literal["log_uniform"] = "log_uniform"
```

## Beta

Bases: `DistributionBase`

The beta distribution of shapes `a` and `b`, scaled to the bounds; they must be finite.

Attributes:

| Name | Type    | Description                |
| ---- | ------- | -------------------------- |
| `a`  | `float` | The first shape, above 0.  |
| `b`  | `float` | The second shape, above 0. |

Source code in `commol/context/inference/distributions.py`

```python
class Beta(DistributionBase):
    """
    The beta distribution of shapes `a` and `b`, scaled to the bounds; they
    must be finite.

    Attributes
    ----------
    a : float
        The first shape, above 0.
    b : float
        The second shape, above 0.
    """

    rust_kind: ClassVar[str] = "beta"
    kind: Literal["beta"] = "beta"
    a: float = Field(gt=0.0, allow_inf_nan=False, description="The first shape.")
    b: float = Field(gt=0.0, allow_inf_nan=False, description="The second shape.")

    def rust_parameters(self) -> list[float]:
        """The shapes `a` and `b`."""
        return [self.a, self.b]
```

### Methods:

#### rust_parameters

```python
rust_parameters() -> list[float]
```

The shapes `a` and `b`.

Source code in `commol/context/inference/distributions.py`

```python
def rust_parameters(self) -> list[float]:
    """The shapes `a` and `b`."""
    return [self.a, self.b]
```

## Normal

Bases: `DistributionBase`

The normal distribution, truncated to the bounds.

Attributes:

| Name   | Type    | Description                                        |
| ------ | ------- | -------------------------------------------------- |
| `mean` | `float` | The mean before truncation.                        |
| `sd`   | `float` | The standard deviation before truncation, above 0. |

Source code in `commol/context/inference/distributions.py`

```python
class Normal(DistributionBase):
    """
    The normal distribution, truncated to the bounds.

    Attributes
    ----------
    mean : float
        The mean before truncation.
    sd : float
        The standard deviation before truncation, above 0.
    """

    rust_kind: ClassVar[str] = "normal"
    kind: Literal["normal"] = "normal"
    mean: float = Field(allow_inf_nan=False, description="The mean.")
    sd: float = Field(
        gt=0.0, allow_inf_nan=False, description="The standard deviation."
    )

    def rust_parameters(self) -> list[float]:
        """The mean and the standard deviation."""
        return [self.mean, self.sd]
```

### Methods:

#### rust_parameters

```python
rust_parameters() -> list[float]
```

The mean and the standard deviation.

Source code in `commol/context/inference/distributions.py`

```python
def rust_parameters(self) -> list[float]:
    """The mean and the standard deviation."""
    return [self.mean, self.sd]
```

## LogNormal

Bases: `DistributionBase`

The log-normal distribution, truncated to the bounds and to `(0, ∞)`: `log x` is normal. The upper bound must be above 0.

Attributes:

| Name    | Type    | Description                               |
| ------- | ------- | ----------------------------------------- |
| `mu`    | `float` | The mean of log x.                        |
| `sigma` | `float` | The standard deviation of log x, above 0. |

Source code in `commol/context/inference/distributions.py`

```python
class LogNormal(DistributionBase):
    """
    The log-normal distribution, truncated to the bounds and to ``(0, ∞)``:
    ``log x`` is normal. The upper bound must be above 0.

    Attributes
    ----------
    mu : float
        The mean of ``log x``.
    sigma : float
        The standard deviation of ``log x``, above 0.
    """

    rust_kind: ClassVar[str] = "log_normal"
    kind: Literal["log_normal"] = "log_normal"
    mu: float = Field(allow_inf_nan=False, description="The mean of log x.")
    sigma: float = Field(
        gt=0.0, allow_inf_nan=False, description="The standard deviation of log x."
    )

    def rust_parameters(self) -> list[float]:
        """`mu` and `sigma`."""
        return [self.mu, self.sigma]
```

### Methods:

#### rust_parameters

```python
rust_parameters() -> list[float]
```

`mu` and `sigma`.

Source code in `commol/context/inference/distributions.py`

```python
def rust_parameters(self) -> list[float]:
    """`mu` and `sigma`."""
    return [self.mu, self.sigma]
```

## Gamma

Bases: `DistributionBase`

The gamma distribution of shape `shape` and rate `rate` (mean `shape / rate`), truncated to the bounds and to `(0, ∞)`. The upper bound must be above 0.

Attributes:

| Name    | Type    | Description         |
| ------- | ------- | ------------------- |
| `shape` | `float` | The shape, above 0. |
| `rate`  | `float` | The rate, above 0.  |

Source code in `commol/context/inference/distributions.py`

```python
class Gamma(DistributionBase):
    """
    The gamma distribution of shape `shape` and rate `rate` (mean
    ``shape / rate``), truncated to the bounds and to ``(0, ∞)``. The upper
    bound must be above 0.

    Attributes
    ----------
    shape : float
        The shape, above 0.
    rate : float
        The rate, above 0.
    """

    rust_kind: ClassVar[str] = "gamma"
    kind: Literal["gamma"] = "gamma"
    shape: float = Field(gt=0.0, allow_inf_nan=False, description="The shape.")
    rate: float = Field(gt=0.0, allow_inf_nan=False, description="The rate.")

    def rust_parameters(self) -> list[float]:
        """The shape and the rate."""
        return [self.shape, self.rate]
```

### Methods:

#### rust_parameters

```python
rust_parameters() -> list[float]
```

The shape and the rate.

Source code in `commol/context/inference/distributions.py`

```python
def rust_parameters(self) -> list[float]:
    """The shape and the rate."""
    return [self.shape, self.rate]
```

## GibbsPosterior

Bases: `WithPriors`, `UncertaintyMethod[GibbsResult]`

The Gibbs posterior `π(θ) ∝ prior(θ) · exp(−η · loss(θ))` of the problem's loss, with its weights and normalization, sampled by sequential Monte Carlo from the priors (temperature 0) to the posterior (temperature 1).

`η` sets how much the data weigh against the priors; the loss has no scale of its own, so `η` has no default.

Attributes:

| Name                 | Type     | Description                                                                                                                                                   |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `priors`             | \`Priors | dict[str, Distribution]\`                                                                                                                                     |
| `eta`                | \`float  | InformationMatching                                                                                                                                           |
| `particles`          | `int`    | Number of particles, at least 2.                                                                                                                              |
| `target_cess`        | `float`  | The conditional effective sample size of each step, as a share of the particles, strictly between 0 and 1: the next temperature is the highest that keeps it. |
| `resample_threshold` | `float`  | Resample when the effective size falls below this share of the particles, in (0, 1\].                                                                         |
| `stay_probability`   | `float`  | The probability that a particle does not move in a step, strictly between 0 and 1: each step moves every particle ⌈log c / log(1 − a)⌉ times.                 |

Source code in `commol/context/inference/smc.py`

```python
class GibbsPosterior(WithPriors, UncertaintyMethod[GibbsResult]):
    """
    The Gibbs posterior ``π(θ) ∝ prior(θ) · exp(−η · loss(θ))`` of the
    problem's loss, with its weights and normalization, sampled by sequential
    Monte Carlo from the priors (temperature 0) to the posterior
    (temperature 1).

    `η` sets how much the data weigh against the priors; the loss has no
    scale of its own, so `η` has no default.

    Attributes
    ----------
    priors : Priors | dict[str, Distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    eta : float | InformationMatching | BootstrapCoverage
        A number above 0, ``EtaCalibration.information_matching()`` or
        ``EtaCalibration.bootstrap_coverage(...)``.
    particles : int
        Number of particles, at least 2.
    target_cess : float
        The conditional effective sample size of each step, as a share of
        the particles, strictly between 0 and 1: the next temperature is the
        highest that keeps it.
    resample_threshold : float
        Resample when the effective size falls below this share of the
        particles, in ``(0, 1]``.
    stay_probability : float
        The probability that a particle does not move in a step, strictly
        between 0 and 1: each step moves every particle
        ``⌈log c / log(1 − a)⌉`` times.
    """

    eta: Eta = Field(description="The scale of the loss, or its calibration.")
    particles: int = Field(ge=2, description="Number of particles.")
    target_cess: float = Field(gt=0.0, lt=1.0, description="Share kept per step.")
    resample_threshold: float = Field(
        gt=0.0, le=1.0, description="Resample below this share."
    )
    stay_probability: float = Field(
        gt=0.0, lt=1.0, description="Probability a particle does not move."
    )

    @model_validator(mode="before")
    @classmethod
    def require_eta(cls, data: object) -> object:
        """Name the choices of `eta` when it is missing."""
        _require(data, "eta", f"give {ETA_CHOICES}")
        if isinstance(data, Mapping):
            eta = data["eta"]
            if isinstance(eta, int | float) and not eta > 0:
                raise ValueError(f"eta must be {ETA_CHOICES}; got {eta}.")
        return data
```

### Methods:

#### require_eta

```python
require_eta(data: object) -> object
```

Name the choices of `eta` when it is missing.

Source code in `commol/context/inference/smc.py`

```python
@model_validator(mode="before")
@classmethod
def require_eta(cls, data: object) -> object:
    """Name the choices of `eta` when it is missing."""
    _require(data, "eta", f"give {ETA_CHOICES}")
    if isinstance(data, Mapping):
        eta = data["eta"]
        if isinstance(eta, int | float) and not eta > 0:
            raise ValueError(f"eta must be {ETA_CHOICES}; got {eta}.")
    return data
```

## EtaCalibration

Bases: `BaseModel`

A rule that chooses the scale `η` of a Gibbs posterior from the data.

Use `information_matching` or `bootstrap_coverage`.

Source code in `commol/context/inference/smc.py`

```python
class EtaCalibration(BaseModel):
    """
    A rule that chooses the scale `η` of a Gibbs posterior from the data.

    Use `information_matching` or `bootstrap_coverage`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def information_matching(cls) -> "InformationMatching":
        """
        Match the curvature of the loss to the variance of its terms at the
        fit (Holmes and Walker 2017).

        It needs ``loss_function="sse"`` and a deterministic engine, and is
        cheap and local.

        Returns
        -------
        InformationMatching
            The calibration.
        """
        return InformationMatching()

    @classmethod
    def bootstrap_coverage(
        cls, replicates: int, block_length: BlockLength, confidence_level: float
    ) -> "BootstrapCoverage":
        """
        Choose `η` so that the posterior intervals of Bayesian-bootstrap data
        sets cover the fit at their level (Syring and Martin 2019).

        It works with any loss and engine, and costs one sampler per
        replicate.

        Parameters
        ----------
        replicates : int
            Number of Bayesian-bootstrap data sets, at least 1.
        block_length : int | "auto"
            Positions per block of the bootstrap.
        confidence_level : float
            The level of the intervals, strictly between 0 and 1.

        Returns
        -------
        BootstrapCoverage
            The calibration.
        """
        return BootstrapCoverage(
            replicates=replicates,
            block_length=block_length,
            confidence_level=confidence_level,
        )
```

### Methods:

#### information_matching

```python
information_matching() -> InformationMatching
```

Match the curvature of the loss to the variance of its terms at the fit (Holmes and Walker 2017).

It needs `loss_function="sse"` and a deterministic engine, and is cheap and local.

Returns:

| Type                  | Description      |
| --------------------- | ---------------- |
| `InformationMatching` | The calibration. |

Source code in `commol/context/inference/smc.py`

```python
@classmethod
def information_matching(cls) -> "InformationMatching":
    """
    Match the curvature of the loss to the variance of its terms at the
    fit (Holmes and Walker 2017).

    It needs ``loss_function="sse"`` and a deterministic engine, and is
    cheap and local.

    Returns
    -------
    InformationMatching
        The calibration.
    """
    return InformationMatching()
```

#### bootstrap_coverage

```python
bootstrap_coverage(replicates: int, block_length: BlockLength, confidence_level: float) -> BootstrapCoverage
```

Choose `η` so that the posterior intervals of Bayesian-bootstrap data sets cover the fit at their level (Syring and Martin 2019).

It works with any loss and engine, and costs one sampler per replicate.

Parameters:

| Name               | Type    | Description                                           | Default                               |
| ------------------ | ------- | ----------------------------------------------------- | ------------------------------------- |
| `replicates`       | `int`   | Number of Bayesian-bootstrap data sets, at least 1.   | *required*                            |
| `block_length`     | \`int   | auto\`                                                | Positions per block of the bootstrap. |
| `confidence_level` | `float` | The level of the intervals, strictly between 0 and 1. | *required*                            |

Returns:

| Type                | Description      |
| ------------------- | ---------------- |
| `BootstrapCoverage` | The calibration. |

Source code in `commol/context/inference/smc.py`

```python
@classmethod
def bootstrap_coverage(
    cls, replicates: int, block_length: BlockLength, confidence_level: float
) -> "BootstrapCoverage":
    """
    Choose `η` so that the posterior intervals of Bayesian-bootstrap data
    sets cover the fit at their level (Syring and Martin 2019).

    It works with any loss and engine, and costs one sampler per
    replicate.

    Parameters
    ----------
    replicates : int
        Number of Bayesian-bootstrap data sets, at least 1.
    block_length : int | "auto"
        Positions per block of the bootstrap.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.

    Returns
    -------
    BootstrapCoverage
        The calibration.
    """
    return BootstrapCoverage(
        replicates=replicates,
        block_length=block_length,
        confidence_level=confidence_level,
    )
```

## InformationMatching

Bases: `EtaCalibration`

`η = tr(J I⁻¹ J) / tr(J)` at the fit, with `J` the curvature of the loss terms and `I` their variance (Holmes and Walker 2017; Lyddon, Holmes and Walker 2019). For the sum of squares of a model with Gaussian noise of variance `σ²` it estimates `1/(2σ²)`. It needs `loss_function="sse"` and a deterministic engine.

Source code in `commol/context/inference/smc.py`

```python
class InformationMatching(EtaCalibration):
    """
    `η = tr(J I⁻¹ J) / tr(J)` at the fit, with ``J`` the curvature of the
    loss terms and ``I`` their variance (Holmes and Walker 2017; Lyddon,
    Holmes and Walker 2019). For the sum of squares of a model with Gaussian
    noise of variance ``σ²`` it estimates ``1/(2σ²)``. It needs
    ``loss_function="sse"`` and a deterministic engine.
    """

    kind: Literal["information_matching"] = "information_matching"
```

## BootstrapCoverage

Bases: `EtaCalibration`

The `η` at which the central intervals of `confidence_level` of the Gibbs posteriors of Bayesian-bootstrap data sets hold the fit at that rate, over the replicates and the parameters (Syring and Martin 2019).

Attributes:

| Name               | Type    | Description                                           |
| ------------------ | ------- | ----------------------------------------------------- |
| `replicates`       | `int`   | Number of Bayesian-bootstrap data sets, at least 1.   |
| `block_length`     | \`int   | auto\`                                                |
| `confidence_level` | `float` | The level of the intervals, strictly between 0 and 1. |

Source code in `commol/context/inference/smc.py`

```python
class BootstrapCoverage(EtaCalibration):
    """
    The `η` at which the central intervals of `confidence_level` of the Gibbs
    posteriors of Bayesian-bootstrap data sets hold the fit at that rate,
    over the replicates and the parameters (Syring and Martin 2019).

    Attributes
    ----------
    replicates : int
        Number of Bayesian-bootstrap data sets, at least 1.
    block_length : int | "auto"
        Positions per block of the bootstrap.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.
    """

    kind: Literal["bootstrap_coverage"] = "bootstrap_coverage"
    replicates: int = Field(ge=1, description="Number of bootstrap data sets.")
    block_length: BlockLength = Field(description="Positions per block.")
    confidence_level: float = Field(
        gt=0.0, lt=1.0, description="The level of the intervals."
    )
```

## AbcSmc

Bases: `WithPriors`, `UncertaintyMethod[SmcResult]`

Approximate Bayesian computation by sequential Monte Carlo (Del Moral, Doucet and Jasra 2012): the tolerance `ε` falls step by step, and a particle is inside when the distance of its run is at most `ε`.

Each next tolerance keeps the weighted share `target_cess` of the particles: with weights of 0 or 1, the weighted quantile of their distances. With a stochastic engine each distance reads one fresh replicate, which the particle keeps until a move replaces it. The run stops when the first move of a step accepts less than `min_acceptance`, and keeps the sample of the tolerance before.

Attributes:

| Name               | Type           | Description                                                                         |
| ------------------ | -------------- | ----------------------------------------------------------------------------------- |
| `priors`           | \`Priors       | dict[str, Distribution]\`                                                           |
| `distance`         | \`LossDistance | SeriesDistance\`                                                                    |
| `particles`        | `int`          | Number of particles, at least 2.                                                    |
| `target_cess`      | `float`        | The weighted share of the particles each tolerance keeps, strictly between 0 and 1. |
| `min_acceptance`   | `float`        | The acceptance rate below which the run stops, strictly between 0 and 1.            |
| `stay_probability` | `float`        | The probability that a particle does not move in a step, strictly between 0 and 1.  |

Source code in `commol/context/inference/smc.py`

```python
class AbcSmc(WithPriors, UncertaintyMethod[SmcResult]):
    """
    Approximate Bayesian computation by sequential Monte Carlo (Del Moral,
    Doucet and Jasra 2012): the tolerance ``ε`` falls step by step, and a
    particle is inside when the distance of its run is at most ``ε``.

    Each next tolerance keeps the weighted share `target_cess` of the
    particles: with weights of 0 or 1, the weighted quantile of their
    distances. With a stochastic engine each distance reads one fresh
    replicate, which the particle keeps until a move replaces it. The run
    stops when the first move of a step accepts less than `min_acceptance`,
    and keeps the sample of the tolerance before.

    Attributes
    ----------
    priors : Priors | dict[str, Distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    distance : LossDistance | SeriesDistance
        How far a run is from the observations.
    particles : int
        Number of particles, at least 2.
    target_cess : float
        The weighted share of the particles each tolerance keeps, strictly
        between 0 and 1.
    min_acceptance : float
        The acceptance rate below which the run stops, strictly between 0
        and 1.
    stay_probability : float
        The probability that a particle does not move in a step, strictly
        between 0 and 1.
    """

    distance: AbcDistance = Field(description="The distance.")
    particles: int = Field(ge=2, description="Number of particles.")
    target_cess: float = Field(gt=0.0, lt=1.0, description="Share kept per step.")
    min_acceptance: float = Field(
        gt=0.0, lt=1.0, description="Acceptance rate below which the run stops."
    )
    stay_probability: float = Field(
        gt=0.0, lt=1.0, description="Probability a particle does not move."
    )
```

## AbcRejection

Bases: `WithPriors`, `UncertaintyMethod[SmcResult]`

ABC rejection: `draws` draws of the priors, of which the closest share `quantile` is kept, with equal weights. It is one step of `AbcSmc` with no move. With a stochastic engine each distance reads one fresh replicate.

Attributes:

| Name       | Type           | Description                                            |
| ---------- | -------------- | ------------------------------------------------------ |
| `priors`   | \`Priors       | dict[str, Distribution]\`                              |
| `distance` | \`LossDistance | SeriesDistance\`                                       |
| `draws`    | `int`          | Number of draws of the priors, at least 2.             |
| `quantile` | `float`        | The share of the draws kept, strictly between 0 and 1. |

Source code in `commol/context/inference/smc.py`

```python
class AbcRejection(WithPriors, UncertaintyMethod[SmcResult]):
    """
    ABC rejection: `draws` draws of the priors, of which the closest share
    `quantile` is kept, with equal weights. It is one step of `AbcSmc` with
    no move. With a stochastic engine each distance reads one fresh
    replicate.

    Attributes
    ----------
    priors : Priors | dict[str, Distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    distance : LossDistance | SeriesDistance
        How far a run is from the observations.
    draws : int
        Number of draws of the priors, at least 2.
    quantile : float
        The share of the draws kept, strictly between 0 and 1.
    """

    distance: AbcDistance = Field(description="The distance.")
    draws: int = Field(ge=2, description="Number of draws.")
    quantile: float = Field(gt=0.0, lt=1.0, description="The share of draws kept.")
```

## LossDistance

Bases: `BaseModel`

The problem's loss as the distance of ABC: `loss_function` with the observation weights and `normalize_observations`. With a stochastic engine it reads one fresh replicate per evaluation, so the problem's `replicates` and `replicate_summary` do not apply.

Source code in `commol/context/inference/smc.py`

```python
class LossDistance(BaseModel):
    """
    The problem's loss as the distance of ABC: ``loss_function`` with the
    observation weights and ``normalize_observations``. With a stochastic
    engine it reads one fresh replicate per evaluation, so the problem's
    ``replicates`` and ``replicate_summary`` do not apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["loss"] = "loss"
```

## SeriesDistance

Bases: `BaseModel`

The largest distance over the series: `metric` of each series on its residuals divided by the root mean square of its observed values, so series in different units compare on one scale.

Attributes:

| Name     | Type          | Description                                                                               |
| -------- | ------------- | ----------------------------------------------------------------------------------------- |
| `metric` | `{rmse, mae}` | The weighted root mean square or mean absolute value of the scaled residuals of a series. |

Source code in `commol/context/inference/smc.py`

```python
class SeriesDistance(BaseModel):
    """
    The largest distance over the series: `metric` of each series on its
    residuals divided by the root mean square of its observed values, so
    series in different units compare on one scale.

    Attributes
    ----------
    metric : {"rmse", "mae"}
        The weighted root mean square or mean absolute value of the scaled
        residuals of a series.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["series"] = "series"
    metric: Literal["rmse", "mae"] = Field(description="The metric of a series.")
```

## SmcResult

Bases: `BaseModel`

The result of a sequential Monte Carlo method.

Attributes:

| Name           | Type              | Description                                                                                                                                                                               |
| -------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample`       | `ParameterSample` | The weighted particles at the last level.                                                                                                                                                 |
| `levels`       | `list[float]`     | The level of each step: the temperature of a Gibbs posterior, from above 0 to 1, or the tolerance of ABC, falling.                                                                        |
| `kept`         | \`list\[float     | None\]\`                                                                                                                                                                                  |
| `log_evidence` | `float`           | The log of the product of the mean incremental weights: for a Gibbs posterior, the log of ∫ prior(θ) exp(−η loss(θ)) dθ; for ABC, the log of the prior probability of the last tolerance. |
| `diagnostics`  | `SmcDiagnostics`  | The effective sizes, acceptance rates and move counts of each step.                                                                                                                       |

Source code in `commol/context/inference/smc.py`

```python
class SmcResult(BaseModel):
    """
    The result of a sequential Monte Carlo method.

    Attributes
    ----------
    sample : ParameterSample
        The weighted particles at the last level.
    levels : list[float]
        The level of each step: the temperature of a Gibbs posterior, from
        above 0 to 1, or the tolerance of ABC, falling.
    kept : list[float | None]
        The value each particle of the sample keeps, in its order: the loss
        of a Gibbs posterior, or the distance of ABC; None for a particle
        outside the support.
    log_evidence : float
        The log of the product of the mean incremental weights: for a Gibbs
        posterior, the log of ``∫ prior(θ) exp(−η loss(θ)) dθ``; for ABC, the
        log of the prior probability of the last tolerance.
    diagnostics : SmcDiagnostics
        The effective sizes, acceptance rates and move counts of each step.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    sample: ParameterSample
    levels: list[float]
    kept: list[float | None]
    log_evidence: float
    diagnostics: SmcDiagnostics
```

## GibbsResult

Bases: `SmcResult`

The result of a Gibbs posterior.

Attributes:

| Name       | Type                | Description                   |
| ---------- | ------------------- | ----------------------------- |
| `eta`      | `float`             | The η the posterior ran with. |
| `fit`      | \`CalibrationResult | None\`                        |
| `coverage` | \`float             | None\`                        |

Source code in `commol/context/inference/smc.py`

```python
class GibbsResult(SmcResult):
    """
    The result of a Gibbs posterior.

    Attributes
    ----------
    eta : float
        The `η` the posterior ran with.
    fit : CalibrationResult | None
        The fit a calibration of `η` read; None when `η` is a number.
    coverage : float | None
        With `BootstrapCoverage`, the coverage of the fit by the intervals of
        the bootstrap posteriors at `η`; otherwise None.
    """

    eta: float
    fit: CalibrationResult | None
    coverage: float | None
```

## SmcDiagnostics

Bases: `BaseModel`

What happened at each step of a sequential Monte Carlo run.

Attributes:

| Name               | Type          | Description                                                                                                                  |
| ------------------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `effective_sizes`  | `list[float]` | Kish's effective size of the weights after the reweighting of each step. A step that keeps few particles has a small one.    |
| `resampled`        | `list[bool]`  | Whether each step resampled.                                                                                                 |
| `acceptance_rates` | `list[float]` | The share of the proposals of the first move of each step that were accepted; for ABC rejection, the share of draws kept.    |
| `move_counts`      | `list[int]`   | How many times each particle moved in each step: ⌈log c / log(1 − a)⌉ with c the stay probability and a the acceptance rate. |
| `evaluations`      | `int`         | The number of model runs.                                                                                                    |
| `stop_reason`      | \`str         | None\`                                                                                                                       |
| `notes`            | `list[str]`   | What the reader of the sample should know, such as settings of the problem that do not apply.                                |

Source code in `commol/context/inference/smc.py`

```python
class SmcDiagnostics(BaseModel):
    """
    What happened at each step of a sequential Monte Carlo run.

    Attributes
    ----------
    effective_sizes : list[float]
        Kish's effective size of the weights after the reweighting of each
        step. A step that keeps few particles has a small one.
    resampled : list[bool]
        Whether each step resampled.
    acceptance_rates : list[float]
        The share of the proposals of the first move of each step that were
        accepted; for ABC rejection, the share of draws kept.
    move_counts : list[int]
        How many times each particle moved in each step:
        ``⌈log c / log(1 − a)⌉`` with ``c`` the stay probability and ``a``
        the acceptance rate.
    evaluations : int
        The number of model runs.
    stop_reason : str | None
        Why the run stopped before its last level, or None.
    notes : list[str]
        What the reader of the sample should know, such as settings of the
        problem that do not apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    effective_sizes: list[float]
    resampled: list[bool]
    acceptance_rates: list[float]
    move_counts: list[int]
    evaluations: int
    stop_reason: str | None
    notes: list[str]
```

## Conformal prediction

Guide: [Conformal Prediction Intervals](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/index.md).

## ConformalMethod

Bases: `UncertaintyMethod[ConformalResult]`

What every conformal method takes: a score, the pooling of its quantiles, and a level, none of them with a default.

Attributes:

| Name    | Type                             | Description                                                                                                                                    |
| ------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `score` | `{absolute, normalized, spread}` | The nonconformity score. "normalized" needs by="all": per series its factor is one constant, so its intervals would equal those of "absolute". |
| `by`    | `{series, all}`                  | Whether each series gets its own quantile, or all series share one.                                                                            |
| `level` | `float`                          | The level of the intervals, strictly between 0 and 1.                                                                                          |

Source code in `commol/context/inference/conformal.py`

```python
class ConformalMethod(UncertaintyMethod[ConformalResult]):
    """
    What every conformal method takes: a score, the pooling of its
    quantiles, and a level, none of them with a default.

    Attributes
    ----------
    score : {"absolute", "normalized", "spread"}
        The nonconformity score. ``"normalized"`` needs ``by="all"``: per
        series its factor is one constant, so its intervals would equal those
        of ``"absolute"``.
    by : {"series", "all"}
        Whether each series gets its own quantile, or all series share one.
    level : float
        The level of the intervals, strictly between 0 and 1.
    """

    score: ConformalScore = Field(description="The nonconformity score.")
    by: Pooling = Field(description="Which scores each quantile reads.")
    level: float = Field(gt=0.0, lt=1.0, description="The level of the intervals.")

    @model_validator(mode="before")
    @classmethod
    def check_score_and_by(cls, data: object) -> object:
        """Check, in Rust, that `score` and `by` are given and fit together."""
        if isinstance(data, Mapping):

            def name(key: str) -> str | None:
                value = data.get(key)
                return None if value is None else str(value)

            commol_rs.inference.check_conformal_settings(name("score"), name("by"))
        return data
```

### Methods:

#### check_score_and_by

```python
check_score_and_by(data: object) -> object
```

Check, in Rust, that `score` and `by` are given and fit together.

Source code in `commol/context/inference/conformal.py`

```python
@model_validator(mode="before")
@classmethod
def check_score_and_by(cls, data: object) -> object:
    """Check, in Rust, that `score` and `by` are given and fit together."""
    if isinstance(data, Mapping):

        def name(key: str) -> str | None:
            value = data.get(key)
            return None if value is None else str(value)

        commol_rs.inference.check_conformal_settings(name("score"), name("by"))
    return data
```

## SplitConformal

Bases: `ConformalMethod`

Split conformal prediction.

It fits the problem's observations and scores `calibration_data`, observations held out of the fit. When the scores of the held-out and of new observations are exchangeable, each interval covers a new observation with probability at least `level`.

Attributes:

| Name               | Type                      | Description                                                   |
| ------------------ | ------------------------- | ------------------------------------------------------------- |
| `calibration_data` | `list[ObservedDataPoint]` | Held-out observations; each of their series gets intervals.   |
| `positions`        | `list[float]`             | Where the intervals are, for each series of calibration_data. |
| `sample`           | \`ParameterSample         | None\`                                                        |
| `starts`           | `int`                     | Starts of the fit (default 1).                                |

Source code in `commol/context/inference/conformal.py`

```python
class SplitConformal(ConformalMethod):
    """
    Split conformal prediction.

    It fits the problem's observations and scores `calibration_data`,
    observations held out of the fit. When the scores of the held-out and of
    new observations are exchangeable, each interval covers a new
    observation with probability at least `level`.

    Attributes
    ----------
    calibration_data : list[ObservedDataPoint]
        Held-out observations; each of their series gets intervals.
    positions : list[float]
        Where the intervals are, for each series of `calibration_data`.
    sample : ParameterSample | None
        None: the prediction is the fit. A sample: the prediction is the
        weighted median of its predictive, and its weighted standard
        deviation is the spread of ``score="spread"``.
    starts : int
        Starts of the fit (default 1).
    """

    calibration_data: list[ObservedDataPoint] = Field(
        min_length=1, description="Held-out observations."
    )
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    sample: ParameterSample | None = Field(
        default=None, description="The sample whose predictive predicts."
    )
    starts: int = Field(default=1, ge=1, description="Starts of the fit.")

    @model_validator(mode="after")
    def check_spread(self) -> Self:
        """Reject ``score="spread"`` without a sample."""
        _no_spread_without_sample(self)
        return self
```

### Methods:

#### check_spread

```python
check_spread() -> Self
```

Reject `score="spread"` without a sample.

Source code in `commol/context/inference/conformal.py`

```python
@model_validator(mode="after")
def check_spread(self) -> Self:
    """Reject ``score="spread"`` without a sample."""
    _no_spread_without_sample(self)
    return self
```

## RollingConformal

Bases: `ConformalMethod`

Rolling-origin conformal prediction: intervals ahead of the data, one quantile per horizon.

Each origin fits the observations at or before it and scores the later ones. Horizon `k` is the `k`-th observation of a series after the origin: a count, not a distance on the axis.

Attributes:

| Name        | Type          | Description                                                                  |
| ----------- | ------------- | ---------------------------------------------------------------------------- |
| `origins`   | `list[float]` | The positions of the origins.                                                |
| `positions` | `list[float]` | Where the intervals are, all after the data; the k-th position is horizon k. |
| `starts`    | `int`         | Starts of each fit (default 1).                                              |

Source code in `commol/context/inference/conformal.py`

```python
class RollingConformal(ConformalMethod):
    """
    Rolling-origin conformal prediction: intervals ahead of the data, one
    quantile per horizon.

    Each origin fits the observations at or before it and scores the later
    ones. Horizon ``k`` is the ``k``-th observation of a series after the
    origin: a count, not a distance on the axis.

    Attributes
    ----------
    origins : list[float]
        The positions of the origins.
    positions : list[float]
        Where the intervals are, all after the data; the ``k``-th position
        is horizon ``k``.
    starts : int
        Starts of each fit (default 1).
    """

    origins: list[float] = Field(min_length=1, description="The origins.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, ge=1, description="Starts of each fit.")

    @model_validator(mode="after")
    def check_spread(self) -> Self:
        """Reject ``score="spread"``: one fit per origin has no spread."""
        if self.score == "spread":
            raise ValueError(
                "RollingConformal fits once per origin, so its predictions have no "
                'spread for score="spread"; use "absolute" or "normalized"'
            )
        return self
```

### Methods:

#### check_spread

```python
check_spread() -> Self
```

Reject `score="spread"`: one fit per origin has no spread.

Source code in `commol/context/inference/conformal.py`

```python
@model_validator(mode="after")
def check_spread(self) -> Self:
    """Reject ``score="spread"``: one fit per origin has no spread."""
    if self.score == "spread":
        raise ValueError(
            "RollingConformal fits once per origin, so its predictions have no "
            'spread for score="spread"; use "absolute" or "normalized"'
        )
    return self
```

## JackknifePlus

Bases: `ConformalMethod`

Jackknife+ (Barber et al. 2021) with blocks: one refit per block left out, and intervals from the left-out residuals and predictions.

With exchangeable blocks it covers at least `1 − 2α` when each block holds one observation, and the CV+ bound with `K` blocks otherwise.

Attributes:

| Name           | Type          | Description                                              |
| -------------- | ------------- | -------------------------------------------------------- |
| `block_length` | \`int         | auto\`                                                   |
| `positions`    | `list[float]` | Where the intervals are, for each series of the problem. |
| `starts`       | `int`         | Starts of the fit and of each refit (default 1).         |

Source code in `commol/context/inference/conformal.py`

```python
class JackknifePlus(ConformalMethod):
    """
    Jackknife+ (Barber et al. 2021) with blocks: one refit per block left
    out, and intervals from the left-out residuals and predictions.

    With exchangeable blocks it covers at least ``1 − 2α`` when each block
    holds one observation, and the CV+ bound with ``K`` blocks otherwise.

    Attributes
    ----------
    block_length : int | "auto"
        Positions per block; it must leave at least two blocks.
    positions : list[float]
        Where the intervals are, for each series of the problem.
    starts : int
        Starts of the fit and of each refit (default 1).
    """

    block_length: BlockLength = Field(description="Positions per block.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, ge=1, description="Starts per fit.")
```

## EnbPI

Bases: `ConformalMethod`

Ensemble batch prediction intervals (Xu and Xie 2021).

Each replicate draws blocks with replacement and fits. The replicates that left out the block of an observation give its leave-one-out prediction and score; their mean over every observation is the point prediction.

Attributes:

| Name           | Type          | Description                                              |
| -------------- | ------------- | -------------------------------------------------------- |
| `replicates`   | `int`         | Number of replicates, at least 2.                        |
| `block_length` | \`int         | auto\`                                                   |
| `positions`    | `list[float]` | Where the intervals are, for each series of the problem. |
| `starts`       | `int`         | Starts of the fit and of each replicate (default 1).     |

Source code in `commol/context/inference/conformal.py`

```python
class EnbPI(ConformalMethod):
    """
    Ensemble batch prediction intervals (Xu and Xie 2021).

    Each replicate draws blocks with replacement and fits. The replicates
    that left out the block of an observation give its leave-one-out
    prediction and score; their mean over every observation is the point
    prediction.

    Attributes
    ----------
    replicates : int
        Number of replicates, at least 2.
    block_length : int | "auto"
        Positions per block.
    positions : list[float]
        Where the intervals are, for each series of the problem.
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    replicates: int = Field(ge=2, description="Number of replicates.")
    block_length: BlockLength = Field(description="Positions per block.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, ge=1, description="Starts per fit.")
```

## AdaptiveConformal

Bases: `ConformalMethod`

Adaptive conformal inference (Gibbs and Candès 2021).

It fits the problem's observations and walks `stream` in order of position. The interval of each point is the quantile, at the working level `1 − α_t`, of the scores of the earlier points of the stream; then the working level moves by `α_{t+1} = α_t + γ (α − err_t)`. The long-run share of misses tends to `α`, with no assumption on the data.

Attributes:

| Name     | Type                      | Description                                  |
| -------- | ------------------------- | -------------------------------------------- |
| `step`   | `float`                   | The step γ of the working level, positive.   |
| `stream` | `list[ObservedDataPoint]` | The observations that arrive after the data. |
| `sample` | \`ParameterSample         | None\`                                       |
| `starts` | `int`                     | Starts of the fit (default 1).               |

Source code in `commol/context/inference/conformal.py`

```python
class AdaptiveConformal(ConformalMethod):
    """
    Adaptive conformal inference (Gibbs and Candès 2021).

    It fits the problem's observations and walks `stream` in order of
    position. The interval of each point is the quantile, at the working
    level ``1 − α_t``, of the scores of the earlier points of the stream;
    then the working level moves by ``α_{t+1} = α_t + γ (α − err_t)``. The
    long-run share of misses tends to ``α``, with no assumption on the data.

    Attributes
    ----------
    step : float
        The step ``γ`` of the working level, positive.
    stream : list[ObservedDataPoint]
        The observations that arrive after the data.
    sample : ParameterSample | None
        As in `SplitConformal`.
    starts : int
        Starts of the fit (default 1).
    """

    step: float = Field(gt=0.0, description="The step of the working level.")
    stream: list[ObservedDataPoint] = Field(
        min_length=1, description="The observations that arrive."
    )
    sample: ParameterSample | None = Field(
        default=None, description="The sample whose predictive predicts."
    )
    starts: int = Field(default=1, ge=1, description="Starts of the fit.")

    @model_validator(mode="after")
    def check_spread(self) -> Self:
        """Reject ``score="spread"`` without a sample."""
        _no_spread_without_sample(self)
        return self
```

### Methods:

#### check_spread

```python
check_spread() -> Self
```

Reject `score="spread"` without a sample.

Source code in `commol/context/inference/conformal.py`

```python
@model_validator(mode="after")
def check_spread(self) -> Self:
    """Reject ``score="spread"`` without a sample."""
    _no_spread_without_sample(self)
    return self
```

## ConformalResult

Bases: `BaseModel`

The result of a conformal method.

Attributes:

| Name        | Type                     | Description                                                             |
| ----------- | ------------------------ | ----------------------------------------------------------------------- |
| `intervals` | `PredictionIntervals`    | The intervals at the positions the method was given.                    |
| `scores`    | `dict[str, list[float]]` | The scores of each series, by label, in the order the method took them. |
| `guarantee` | `str`                    | The coverage the method guarantees, and what the guarantee assumes.     |

Source code in `commol/context/inference/conformal.py`

```python
class ConformalResult(BaseModel):
    """
    The result of a conformal method.

    Attributes
    ----------
    intervals : PredictionIntervals
        The intervals at the positions the method was given.
    scores : dict[str, list[float]]
        The scores of each series, by label, in the order the method took
        them.
    guarantee : str
        The coverage the method guarantees, and what the guarantee assumes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    intervals: PredictionIntervals
    scores: dict[str, list[float]]
    guarantee: str
```

## PredictionIntervals

Bases: `BaseModel`

Prediction intervals for observations: where new observations of each series fall, at a level.

Attributes:

| Name        | Type                         | Description                                                                                                                                    |
| ----------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `level`     | `float`                      | The level of the intervals, 1 − α.                                                                                                             |
| `series`    | `dict[str, SeriesIntervals]` | The intervals of each series, by label.                                                                                                        |
| `unbounded` | `dict[str, list[float]]`     | The positions of each series whose interval is unbounded, because its quantile had fewer than 1/α − 1 scores. A series with none is not a key. |

Source code in `commol/context/inference/conformal.py`

```python
class PredictionIntervals(BaseModel):
    """
    Prediction intervals for observations: where new observations of each
    series fall, at a level.

    Attributes
    ----------
    level : float
        The level of the intervals, ``1 − α``.
    series : dict[str, SeriesIntervals]
        The intervals of each series, by label.
    unbounded : dict[str, list[float]]
        The positions of each series whose interval is unbounded, because
        its quantile had fewer than ``1/α − 1`` scores. A series with none
        is not a key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    level: float
    series: dict[str, SeriesIntervals]
    unbounded: dict[str, list[float]]
```

## SeriesIntervals

Bases: `BaseModel`

The prediction intervals of one series.

Attributes:

| Name         | Type          | Description                                          |
| ------------ | ------------- | ---------------------------------------------------- |
| `positions`  | `list[float]` | Where the intervals are.                             |
| `prediction` | `list[float]` | The point prediction at each position.               |
| `lower`      | `list[float]` | The lower end of each interval; -inf when unbounded. |
| `upper`      | `list[float]` | The upper end of each interval; inf when unbounded.  |

Source code in `commol/context/inference/conformal.py`

```python
class SeriesIntervals(BaseModel):
    """
    The prediction intervals of one series.

    Attributes
    ----------
    positions : list[float]
        Where the intervals are.
    prediction : list[float]
        The point prediction at each position.
    lower : list[float]
        The lower end of each interval; ``-inf`` when unbounded.
    upper : list[float]
        The upper end of each interval; ``inf`` when unbounded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")

    positions: list[float]
    prediction: list[float]
    lower: list[float]
    upper: list[float]
```

## Scores and intervals

## scores

Scores of a predictive against observations.

Every score is computed in Rust from each draw's value of the observations (`Predictive.at_observations`), per observation and per series. Which observations to score, for example the ones held out of the calibration, and the observation noise are arguments with no default: a band of the model (`NoNoise`) compared with data covers less than its level.

### Classes

#### Score

Bases: `BaseModel`

One score of a predictive at a set of observations.

Attributes:

| Name              | Type               | Description                                                                      |
| ----------------- | ------------------ | -------------------------------------------------------------------------------- |
| `per_observation` | `list[float]`      | The score of each observation, in the order given.                               |
| `per_series`      | `dict[str, float]` | The mean score of each series (observation label), in order of first appearance. |
| `mean`            | `float`            | The mean over every observation.                                                 |

Source code in `commol/api/inference/scores.py`

```python
class Score(BaseModel):
    """
    One score of a predictive at a set of observations.

    Attributes
    ----------
    per_observation : list[float]
        The score of each observation, in the order given.
    per_series : dict[str, float]
        The mean score of each series (observation label), in order of first
        appearance.
    mean : float
        The mean over every observation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    per_observation: list[float] = Field(description="Score of each observation")
    per_series: dict[str, float] = Field(description="Mean score of each series")
    mean: float = Field(description="Mean over every observation")
```

### Functions:

#### coverage

```python
coverage(predictive: Predictive, observations: Sequence[ObservedDataPoint], noise: ObservationNoise, confidence_level: float) -> Score
```

The empirical coverage of the central interval of `confidence_level`.

Parameters:

| Name               | Type                          | Description                | Default                                                          |
| ------------------ | ----------------------------- | -------------------------- | ---------------------------------------------------------------- |
| `predictive`       | `Predictive`                  | The predictive.            | *required*                                                       |
| `observations`     | `Sequence[ObservedDataPoint]` | The observations to score. | *required*                                                       |
| `noise`            | \`NoNoise                     | ResidualNoise\`            | The observation noise of the draws (Predictive.at_observations). |
| `confidence_level` | `float`                       | Strictly between 0 and 1.  | *required*                                                       |

Returns:

| Type    | Description                                                                                                                             |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `Score` | 1 for each observation inside its interval, 0 outside; the means are the coverage. A calibrated interval covers about confidence_level. |

Source code in `commol/api/inference/scores.py`

```python
def coverage(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: ObservationNoise,
    confidence_level: float,
) -> Score:
    """
    The empirical coverage of the central interval of `confidence_level`.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise
        The observation noise of the draws (`Predictive.at_observations`).
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    Score
        1 for each observation inside its interval, 0 outside; the means are
        the coverage. A calibrated interval covers about `confidence_level`.
    """
    draws = predictive.at_observations(observations, noise)
    return _score(draws, draws.rust_draws.coverage(draws.observed, confidence_level))
```

#### interval_score

```python
interval_score(predictive: Predictive, observations: Sequence[ObservedDataPoint], noise: ObservationNoise, confidence_level: float) -> Score
```

The interval score: `(u − l) + (2/α)(l − y)₊ + (2/α)(y − u)₊`.

Parameters:

| Name               | Type                          | Description                                     | Default                                                          |
| ------------------ | ----------------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| `predictive`       | `Predictive`                  | The predictive.                                 | *required*                                                       |
| `observations`     | `Sequence[ObservedDataPoint]` | The observations to score.                      | *required*                                                       |
| `noise`            | \`NoNoise                     | ResidualNoise\`                                 | The observation noise of the draws (Predictive.at_observations). |
| `confidence_level` | `float`                       | The level c of the central interval; α = 1 − c. | *required*                                                       |

Returns:

| Type    | Description      |
| ------- | ---------------- |
| `Score` | Lower is better. |

Source code in `commol/api/inference/scores.py`

```python
def interval_score(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: ObservationNoise,
    confidence_level: float,
) -> Score:
    """
    The interval score: ``(u − l) + (2/α)(l − y)₊ + (2/α)(y − u)₊``.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise
        The observation noise of the draws (`Predictive.at_observations`).
    confidence_level : float
        The level ``c`` of the central interval; ``α = 1 − c``.

    Returns
    -------
    Score
        Lower is better.
    """
    draws = predictive.at_observations(observations, noise)
    return _score(
        draws, draws.rust_draws.interval_score(draws.observed, confidence_level)
    )
```

#### weighted_interval_score

```python
weighted_interval_score(predictive: Predictive, observations: Sequence[ObservedDataPoint], noise: ObservationNoise, confidence_levels: Sequence[float]) -> Score
```

The weighted interval score of Bracher et al. (2021).

`(½|y − m| + Σₖ (αₖ/2) IS_αₖ) / (K + ½)`, with `m` the median and `αₖ = 1 − cₖ` for each confidence level `cₖ`.

Parameters:

| Name                | Type                          | Description                          | Default                                                          |
| ------------------- | ----------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| `predictive`        | `Predictive`                  | The predictive.                      | *required*                                                       |
| `observations`      | `Sequence[ObservedDataPoint]` | The observations to score.           | *required*                                                       |
| `noise`             | \`NoNoise                     | ResidualNoise\`                      | The observation noise of the draws (Predictive.at_observations). |
| `confidence_levels` | `Sequence[float]`             | The levels of the central intervals. | *required*                                                       |

Returns:

| Type    | Description      |
| ------- | ---------------- |
| `Score` | Lower is better. |

References

Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating epidemic forecasts in an interval format. PLOS Computational Biology, 17(2), e1008618. https://doi.org/10.1371/journal.pcbi.1008618

Source code in `commol/api/inference/scores.py`

```python
def weighted_interval_score(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: ObservationNoise,
    confidence_levels: Sequence[float],
) -> Score:
    """
    The weighted interval score of Bracher et al. (2021).

    ``(½|y − m| + Σₖ (αₖ/2) IS_αₖ) / (K + ½)``, with ``m`` the median and
    ``αₖ = 1 − cₖ`` for each confidence level ``cₖ``.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise
        The observation noise of the draws (`Predictive.at_observations`).
    confidence_levels : Sequence[float]
        The levels of the central intervals.

    Returns
    -------
    Score
        Lower is better.

    References
    ----------
    Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating
    epidemic forecasts in an interval format. PLOS Computational Biology,
    17(2), e1008618. https://doi.org/10.1371/journal.pcbi.1008618
    """
    draws = predictive.at_observations(observations, noise)
    return _score(
        draws,
        draws.rust_draws.weighted_interval_score(
            draws.observed, [float(c) for c in confidence_levels]
        ),
    )
```

#### crps

```python
crps(predictive: Predictive, observations: Sequence[ObservedDataPoint], noise: ObservationNoise) -> Score
```

The continuous ranked probability score of the weighted draws.

`Σ wᵢ|xᵢ − y| − ½ Σᵢⱼ wᵢwⱼ|xᵢ − xⱼ|`, the function the CRPS loss of a stochastic calibration uses.

Parameters:

| Name           | Type                          | Description                | Default                                                          |
| -------------- | ----------------------------- | -------------------------- | ---------------------------------------------------------------- |
| `predictive`   | `Predictive`                  | The predictive.            | *required*                                                       |
| `observations` | `Sequence[ObservedDataPoint]` | The observations to score. | *required*                                                       |
| `noise`        | \`NoNoise                     | ResidualNoise\`            | The observation noise of the draws (Predictive.at_observations). |

Returns:

| Type    | Description                                       |
| ------- | ------------------------------------------------- |
| `Score` | Lower is better; in the unit of the observations. |

Source code in `commol/api/inference/scores.py`

```python
def crps(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: ObservationNoise,
) -> Score:
    """
    The continuous ranked probability score of the weighted draws.

    ``Σ wᵢ|xᵢ − y| − ½ Σᵢⱼ wᵢwⱼ|xᵢ − xⱼ|``, the function the CRPS loss of a
    stochastic calibration uses.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise
        The observation noise of the draws (`Predictive.at_observations`).

    Returns
    -------
    Score
        Lower is better; in the unit of the observations.
    """
    draws = predictive.at_observations(observations, noise)
    return _score(draws, draws.rust_draws.crps(draws.observed))
```

#### pit

```python
pit(predictive: Predictive, observations: Sequence[ObservedDataPoint], noise: ObservationNoise, seed: int) -> Score
```

The probability integral transform of each observation.

`F(y)` of the weighted draws, randomized between `F(y⁻)` and `F(y)` where draws tie with `y`, so a discrete predictive gives uniform values too.

Parameters:

| Name           | Type                          | Description                        | Default                                                          |
| -------------- | ----------------------------- | ---------------------------------- | ---------------------------------------------------------------- |
| `predictive`   | `Predictive`                  | The predictive.                    | *required*                                                       |
| `observations` | `Sequence[ObservedDataPoint]` | The observations to score.         | *required*                                                       |
| `noise`        | \`NoNoise                     | ResidualNoise\`                    | The observation noise of the draws (Predictive.at_observations). |
| `seed`         | `int`                         | Seed of the randomization of ties. | *required*                                                       |

Returns:

| Type    | Description                                                   |
| ------- | ------------------------------------------------------------- |
| `Score` | Values in [0, 1]; calibrated predictions give uniform values. |

Source code in `commol/api/inference/scores.py`

```python
def pit(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: ObservationNoise,
    seed: int,
) -> Score:
    """
    The probability integral transform of each observation.

    ``F(y)`` of the weighted draws, randomized between ``F(y⁻)`` and ``F(y)``
    where draws tie with ``y``, so a discrete predictive gives uniform values
    too.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise
        The observation noise of the draws (`Predictive.at_observations`).
    seed : int
        Seed of the randomization of ties.

    Returns
    -------
    Score
        Values in ``[0, 1]``; calibrated predictions give uniform values.
    """
    draws = predictive.at_observations(observations, noise)
    return _score(draws, draws.rust_draws.pit(draws.observed, seed))
```

## weighted_quantiles

```python
weighted_quantiles(values: Sequence[float], levels: Sequence[float], weights: Sequence[float] | None = None) -> list[float]
```

Quantiles of weighted values, by the one convention of commol.

The quantile is the inverse of the weighted empirical distribution, interpolated between the midpoints of the cumulative weights. With equal weights it equals `numpy.percentile(..., method="hazen")`. A weight of zero removes a value.

Parameters:

| Name      | Type              | Description                            | Default                                                                    |
| --------- | ----------------- | -------------------------------------- | -------------------------------------------------------------------------- |
| `values`  | `Sequence[float]` | The values.                            | *required*                                                                 |
| `levels`  | `Sequence[float]` | Quantile levels, each between 0 and 1. | *required*                                                                 |
| `weights` | \`Sequence[float] | None\`                                 | One non-negative weight per value, of any scale. None gives equal weights. |

Returns:

| Type          | Description             |
| ------------- | ----------------------- |
| `list[float]` | One quantile per level. |

Raises:

| Type         | Description                                                               |
| ------------ | ------------------------------------------------------------------------- |
| `ValueError` | If the lengths differ, no weight is positive or a level is not in [0, 1]. |

Source code in `commol/api/inference/intervals.py`

```python
def weighted_quantiles(
    values: Sequence[float],
    levels: Sequence[float],
    weights: Sequence[float] | None = None,
) -> list[float]:
    """
    Quantiles of weighted values, by the one convention of commol.

    The quantile is the inverse of the weighted empirical distribution,
    interpolated between the midpoints of the cumulative weights. With equal
    weights it equals ``numpy.percentile(..., method="hazen")``. A weight of
    zero removes a value.

    Parameters
    ----------
    values : Sequence[float]
        The values.
    levels : Sequence[float]
        Quantile levels, each between 0 and 1.
    weights : Sequence[float] | None, optional
        One non-negative weight per value, of any scale. None gives equal
        weights.

    Returns
    -------
    list[float]
        One quantile per level.

    Raises
    ------
    ValueError
        If the lengths differ, no weight is positive or a level is not in
        ``[0, 1]``.
    """
    values = [float(value) for value in values]
    if weights is None:
        weights = [1.0] * len(values)
    return commol_rs.inference.weighted_quantiles(
        values, [float(w) for w in weights], [float(level) for level in levels]
    )
```

## member_statistics

```python
member_statistics(per_member: Sequence[float], confidence_level: float, weights: Sequence[float] | None = None) -> dict[str, float]
```

```python
member_statistics(per_member: Sequence[Sequence[float]], confidence_level: float, weights: Sequence[float] | None = None) -> dict[str, list[float]]
```

```python
member_statistics(per_member: NDArray[float64], confidence_level: float, weights: Sequence[float] | None = None) -> dict[str, float | list[float]]
```

```python
member_statistics(per_member: EnsembleValues, confidence_level: float, weights: Sequence[float] | None = None) -> dict[str, float] | dict[str, list[float]] | dict[str, float | list[float]]
```

Spread of one quantity across the members of an ensemble.

Each member must already be reduced to the quantity of interest, so that the reported interval is a quantile of member values (:func:`weighted_quantiles`).

Parameters:

| Name               | Type              | Description                                                          | Default                                                                                              |
| ------------------ | ----------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `per_member`       | \`Sequence[float] | Sequence\[Sequence[float]\]                                          | NDArray\`                                                                                            |
| `confidence_level` | `float`           | Confidence level of the reported interval, strictly between 0 and 1. | *required*                                                                                           |
| `weights`          | \`Sequence[float] | None\`                                                               | One weight per member, such as the weights of a :class:~commol.Predictive. None gives equal weights. |

Returns:

| Type               | Description                |
| ------------------ | -------------------------- |
| \`dict[str, float] | dict\[str, list[float]\]\` |

Raises:

| Type         | Description                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------------- |
| `ValueError` | If confidence_level is out of range, per_member is empty, or the weights do not match the members. |

Source code in `commol/api/inference/intervals.py`

```python
def member_statistics(
    per_member: EnsembleValues,
    confidence_level: float,
    weights: Sequence[float] | None = None,
) -> dict[str, float] | dict[str, list[float]] | dict[str, float | list[float]]:
    """
    Spread of one quantity across the members of an ensemble.

    Each member must already be reduced to the quantity of interest, so that
    the reported interval is a quantile of member values
    (:func:`weighted_quantiles`).

    Parameters
    ----------
    per_member : Sequence[float] | Sequence[Sequence[float]] | NDArray
        One entry per ensemble member. Entries may be scalars, in which case
        every returned statistic is a scalar, or equal-length series, in which
        case every returned statistic is a series reduced across members at
        each position. A numpy array is accepted, but its rank is not visible
        to a type checker; pass a list of lists for a precise return type.
    confidence_level : float
        Confidence level of the reported interval, strictly between 0 and 1.
    weights : Sequence[float] | None, optional
        One weight per member, such as the weights of a
        :class:`~commol.Predictive`. None gives equal weights.

    Returns
    -------
    dict[str, float] | dict[str, list[float]]
        Keys ``mean``, ``median``, ``ci_lower``, ``ci_upper``, ``min`` and
        ``max``. Scalar members give scalar statistics, series members give
        series statistics. ``min`` and ``max`` are over the members of
        positive weight.

    Raises
    ------
    ValueError
        If ``confidence_level`` is out of range, ``per_member`` is empty, or
        the weights do not match the members.
    """
    lower_point, upper_point = ci_percentiles(confidence_level)
    values = np.asarray(per_member, dtype=float)
    if values.size == 0:
        raise ValueError("per_member must hold at least one ensemble member.")
    member_weights = (
        [1.0] * values.shape[0] if weights is None else [float(w) for w in weights]
    )
    if len(member_weights) != values.shape[0]:
        raise ValueError(
            f"One weight per member: {values.shape[0]} members, "
            f"{len(member_weights)} weights."
        )
    levels = [0.0, lower_point / 100.0, 0.5, upper_point / 100.0, 1.0]
    columns = values.reshape(values.shape[0], -1).T
    table = [
        weighted_quantiles(column.tolist(), levels, member_weights)
        for column in columns
    ]
    means = [
        commol_rs.inference.weighted_mean(column.tolist(), member_weights)
        for column in columns
    ]

    def shaped(per_column: list[float]) -> float | list[float]:
        return per_column[0] if values.ndim == 1 else per_column

    return {
        "mean": shaped(means),
        "median": shaped([row[2] for row in table]),
        "ci_lower": shaped([row[1] for row in table]),
        "ci_upper": shaped([row[3] for row in table]),
        "min": shaped([row[0] for row in table]),
        "max": shaped([row[4] for row in table]),
    }
```

## ci_percentiles

```python
ci_percentiles(confidence_level: float) -> tuple[float, float]
```

Lower and upper percentile points of a two-sided confidence interval.

Parameters:

| Name               | Type    | Description                                                 | Default    |
| ------------------ | ------- | ----------------------------------------------------------- | ---------- |
| `confidence_level` | `float` | Confidence level of the interval, strictly between 0 and 1. | *required* |

Returns:

| Type                  | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `tuple[float, float]` | Lower and upper percentile points, on the 0-100 scale. |

Raises:

| Type         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `ValueError` | If confidence_level is not strictly between 0 and 1. |

Source code in `commol/api/inference/intervals.py`

```python
def ci_percentiles(confidence_level: float) -> tuple[float, float]:
    """
    Lower and upper percentile points of a two-sided confidence interval.

    Parameters
    ----------
    confidence_level : float
        Confidence level of the interval, strictly between 0 and 1.

    Returns
    -------
    tuple[float, float]
        Lower and upper percentile points, on the 0-100 scale.

    Raises
    ------
    ValueError
        If ``confidence_level`` is not strictly between 0 and 1.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, got "
            f"{confidence_level}. Note this is a fraction, not a percentage: "
            f"use 0.95 for a 95% interval."
        )
    return (
        (1.0 - confidence_level) / 2.0 * 100.0,
        (1.0 + confidence_level) / 2.0 * 100.0,
    )
```

## registry

The runners of inference methods, looked up by configuration type.

A method registers its runner with `register`; `Calibrator.uncertainty` and `Calibrator.identifiability` find it with `run_method`. The entry points know no method.

### Classes

### Functions:

#### register

```python
register(method_type: type[MethodT], runner: Callable[[Calibrator, MethodT], BaseModel]) -> None
```

Register the runner of an inference method.

Parameters:

| Name          | Type                                                 | Description                                             | Default    |
| ------------- | ---------------------------------------------------- | ------------------------------------------------------- | ---------- |
| `method_type` | `type[InferenceMethod]`                              | The configuration type of the method.                   | *required* |
| `runner`      | `Callable[[Calibrator, InferenceMethod], BaseModel]` | Runs the method on a calibrator and returns its result. | *required* |

Raises:

| Type         | Description                       |
| ------------ | --------------------------------- |
| `ValueError` | If the type already has a runner. |

Source code in `commol/api/inference/registry.py`

```python
def register(
    method_type: type[MethodT],
    runner: Callable[["Calibrator", MethodT], BaseModel],
) -> None:
    """
    Register the runner of an inference method.

    Parameters
    ----------
    method_type : type[InferenceMethod]
        The configuration type of the method.
    runner : Callable[[Calibrator, InferenceMethod], BaseModel]
        Runs the method on a calibrator and returns its result.

    Raises
    ------
    ValueError
        If the type already has a runner.
    """
    if method_type in _RUNNERS:
        raise ValueError(f"{method_type.__name__} already has a runner.")
    _RUNNERS[method_type] = cast("Runner", runner)
```

#### registered

```python
registered() -> list[str]
```

The names of the configuration types that have a runner.

Returns:

| Type        | Description   |
| ----------- | ------------- |
| `list[str]` | Sorted names. |

Source code in `commol/api/inference/registry.py`

```python
def registered() -> list[str]:
    """
    The names of the configuration types that have a runner.

    Returns
    -------
    list[str]
        Sorted names.
    """
    return sorted(method_type.__name__ for method_type in _RUNNERS)
```

#### run_method

```python
run_method(calibrator: Calibrator, method: InferenceMethod[ResultT], kind: type[InferenceMethod[BaseModel]]) -> ResultT
```

Run `method` on `calibrator` with the runner of its type.

Parameters:

| Name         | Type                       | Description                                                                           | Default    |
| ------------ | -------------------------- | ------------------------------------------------------------------------------------- | ---------- |
| `calibrator` | `Calibrator`               | The calibration problem.                                                              | *required* |
| `method`     | `InferenceMethod[ResultT]` | The configuration of the method.                                                      | *required* |
| `kind`       | `type[InferenceMethod]`    | The kind of method the entry point takes: UncertaintyMethod or IdentifiabilityMethod. | *required* |

Returns:

| Type      | Description          |
| --------- | -------------------- |
| `ResultT` | The method's result. |

Raises:

| Type         | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `TypeError`  | If method is not of kind; the message names its entry point. |
| `ValueError` | If no runner is registered for the configuration's type.     |

Source code in `commol/api/inference/registry.py`

```python
def run_method(
    calibrator: "Calibrator",
    method: InferenceMethod[ResultT],
    kind: type[InferenceMethod[BaseModel]],
) -> ResultT:
    """
    Run `method` on `calibrator` with the runner of its type.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : InferenceMethod[ResultT]
        The configuration of the method.
    kind : type[InferenceMethod]
        The kind of method the entry point takes: `UncertaintyMethod` or
        `IdentifiabilityMethod`.

    Returns
    -------
    ResultT
        The method's result.

    Raises
    ------
    TypeError
        If `method` is not of `kind`; the message names its entry point.
    ValueError
        If no runner is registered for the configuration's type.
    """
    if not isinstance(method, kind):
        right = next(
            (
                name
                for other, name in _ENTRY_POINTS.items()
                if isinstance(method, other)
            ),
            None,
        )
        hint = f" Run it with {right}." if right else ""
        raise TypeError(f"{type(method).__name__} is not a {kind.__name__}.{hint}")
    runner = _RUNNERS.get(type(method))
    if runner is None:
        raise ValueError(
            f"No method is registered for {type(method).__name__}. "
            f"Registered: {registered()}."
        )
    return cast("ResultT", runner(calibrator, method))
```
