# Uncertainty API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

What every uncertainty method takes and returns: the configuration base, the weighted parameter sample, the predictive, the scores and the one quantile. Guide: [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md).

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
        self, observed_data: Sequence[ObservedDataPoint]
    ) -> ObservationDraws:
        """
        Every draw's value of each observation.

        The observations need not be the ones the problem was fitted to: held
        out data are read the same way, through the observation operator.

        Parameters
        ----------
        observed_data : Sequence[ObservedDataPoint]
            The observations.

        Returns
        -------
        ObservationDraws
            Each draw's value of each observation.

        Raises
        ------
        ValueError
            If an observation names an unknown output or factor.
        InfeasibleStateError
            If a draw leaves the values its model allows.
        """
        draws = self.calibrator.rust_problem.observation_draws(
            self.sample.rust_sample,
            engine_state.rust_observed_data(list(observed_data)),
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
at_observations(observed_data: Sequence[ObservedDataPoint]) -> ObservationDraws
```

Every draw's value of each observation.

The observations need not be the ones the problem was fitted to: held out data are read the same way, through the observation operator.

Parameters:

| Name            | Type                          | Description       | Default    |
| --------------- | ----------------------------- | ----------------- | ---------- |
| `observed_data` | `Sequence[ObservedDataPoint]` | The observations. | *required* |

Returns:

| Type               | Description                            |
| ------------------ | -------------------------------------- |
| `ObservationDraws` | Each draw's value of each observation. |

Raises:

| Type                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `ValueError`           | If an observation names an unknown output or factor. |
| `InfeasibleStateError` | If a draw leaves the values its model allows.        |

Source code in `commol/api/inference/predictive.py`

```python
def at_observations(
    self, observed_data: Sequence[ObservedDataPoint]
) -> ObservationDraws:
    """
    Every draw's value of each observation.

    The observations need not be the ones the problem was fitted to: held
    out data are read the same way, through the observation operator.

    Parameters
    ----------
    observed_data : Sequence[ObservedDataPoint]
        The observations.

    Returns
    -------
    ObservationDraws
        Each draw's value of each observation.

    Raises
    ------
    ValueError
        If an observation names an unknown output or factor.
    InfeasibleStateError
        If a draw leaves the values its model allows.
    """
    draws = self.calibrator.rust_problem.observation_draws(
        self.sample.rust_sample,
        engine_state.rust_observed_data(list(observed_data)),
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

## scores

Scores of a predictive against observations.

Every score is computed in Rust from each draw's value of the observations (`Predictive.at_observations`), per observation and per series. Which observations to score, for example the ones held out of the calibration, is an argument with no default.

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
coverage(predictive: Predictive, observations: Sequence[ObservedDataPoint], confidence_level: float) -> Score
```

The empirical coverage of the central interval of `confidence_level`.

Parameters:

| Name               | Type                          | Description                | Default    |
| ------------------ | ----------------------------- | -------------------------- | ---------- |
| `predictive`       | `Predictive`                  | The predictive.            | *required* |
| `observations`     | `Sequence[ObservedDataPoint]` | The observations to score. | *required* |
| `confidence_level` | `float`                       | Strictly between 0 and 1.  | *required* |

Returns:

| Type    | Description                                                                                                                             |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `Score` | 1 for each observation inside its interval, 0 outside; the means are the coverage. A calibrated interval covers about confidence_level. |

Source code in `commol/api/inference/scores.py`

```python
def coverage(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
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
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    Score
        1 for each observation inside its interval, 0 outside; the means are
        the coverage. A calibrated interval covers about `confidence_level`.
    """
    draws = predictive.at_observations(observations)
    return _score(draws, draws.rust_draws.coverage(draws.observed, confidence_level))
```

#### interval_score

```python
interval_score(predictive: Predictive, observations: Sequence[ObservedDataPoint], confidence_level: float) -> Score
```

The interval score: `(u − l) + (2/α)(l − y)₊ + (2/α)(y − u)₊`.

Parameters:

| Name               | Type                          | Description                                     | Default    |
| ------------------ | ----------------------------- | ----------------------------------------------- | ---------- |
| `predictive`       | `Predictive`                  | The predictive.                                 | *required* |
| `observations`     | `Sequence[ObservedDataPoint]` | The observations to score.                      | *required* |
| `confidence_level` | `float`                       | The level c of the central interval; α = 1 − c. | *required* |

Returns:

| Type    | Description      |
| ------- | ---------------- |
| `Score` | Lower is better. |

Source code in `commol/api/inference/scores.py`

```python
def interval_score(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
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
    confidence_level : float
        The level ``c`` of the central interval; ``α = 1 − c``.

    Returns
    -------
    Score
        Lower is better.
    """
    draws = predictive.at_observations(observations)
    return _score(
        draws, draws.rust_draws.interval_score(draws.observed, confidence_level)
    )
```

#### weighted_interval_score

```python
weighted_interval_score(predictive: Predictive, observations: Sequence[ObservedDataPoint], confidence_levels: Sequence[float]) -> Score
```

The weighted interval score of Bracher et al. (2021).

`(½|y − m| + Σₖ (αₖ/2) IS_αₖ) / (K + ½)`, with `m` the median and `αₖ = 1 − cₖ` for each confidence level `cₖ`.

Parameters:

| Name                | Type                          | Description                          | Default    |
| ------------------- | ----------------------------- | ------------------------------------ | ---------- |
| `predictive`        | `Predictive`                  | The predictive.                      | *required* |
| `observations`      | `Sequence[ObservedDataPoint]` | The observations to score.           | *required* |
| `confidence_levels` | `Sequence[float]`             | The levels of the central intervals. | *required* |

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
    draws = predictive.at_observations(observations)
    return _score(
        draws,
        draws.rust_draws.weighted_interval_score(
            draws.observed, [float(c) for c in confidence_levels]
        ),
    )
```

#### crps

```python
crps(predictive: Predictive, observations: Sequence[ObservedDataPoint]) -> Score
```

The continuous ranked probability score of the weighted draws.

`Σ wᵢ|xᵢ − y| − ½ Σᵢⱼ wᵢwⱼ|xᵢ − xⱼ|`, the function the CRPS loss of a stochastic calibration uses.

Parameters:

| Name           | Type                          | Description                | Default    |
| -------------- | ----------------------------- | -------------------------- | ---------- |
| `predictive`   | `Predictive`                  | The predictive.            | *required* |
| `observations` | `Sequence[ObservedDataPoint]` | The observations to score. | *required* |

Returns:

| Type    | Description                                       |
| ------- | ------------------------------------------------- |
| `Score` | Lower is better; in the unit of the observations. |

Source code in `commol/api/inference/scores.py`

```python
def crps(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
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

    Returns
    -------
    Score
        Lower is better; in the unit of the observations.
    """
    draws = predictive.at_observations(observations)
    return _score(draws, draws.rust_draws.crps(draws.observed))
```

#### pit

```python
pit(predictive: Predictive, observations: Sequence[ObservedDataPoint], seed: int) -> Score
```

The probability integral transform of each observation.

`F(y)` of the weighted draws, randomized between `F(y⁻)` and `F(y)` where draws tie with `y`, so a discrete predictive gives uniform values too.

Parameters:

| Name           | Type                          | Description                        | Default    |
| -------------- | ----------------------------- | ---------------------------------- | ---------- |
| `predictive`   | `Predictive`                  | The predictive.                    | *required* |
| `observations` | `Sequence[ObservedDataPoint]` | The observations to score.         | *required* |
| `seed`         | `int`                         | Seed of the randomization of ties. | *required* |

Returns:

| Type    | Description                                                   |
| ------- | ------------------------------------------------------------- |
| `Score` | Values in [0, 1]; calibrated predictions give uniform values. |

Source code in `commol/api/inference/scores.py`

```python
def pit(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
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
    seed : int
        Seed of the randomization of ties.

    Returns
    -------
    Score
        Values in ``[0, 1]``; calibrated predictions give uniform values.
    """
    draws = predictive.at_observations(observations)
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

| Type         | Description                                     |
| ------------ | ----------------------------------------------- |
| `ValueError` | If the lengths differ or no weight is positive. |

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
        If the lengths differ or no weight is positive.
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
