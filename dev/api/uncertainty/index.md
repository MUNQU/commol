# Uncertainty API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.uncertainty`: what every uncertainty method returns. The weighted parameter sample, its predictive and its figures, the scenarios, and the one quantile. Guide: [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md).

## ParameterSample

Bases: `ResultModel`

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

Source code in `commol/uncertainty/sample.py`

```python
class ParameterSample(ResultModel):
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

    model_config = ConfigDict(extra="forbid", frozen=True)

    names: list[str] = Field(description="One name per column.")
    bounds: list[tuple[float, float]] = Field(description="Bounds of each parameter.")
    values: list[list[float]] = Field(description="One row per draw.")
    log_weights: list[float] = Field(description="Normalized log weight of each draw.")
    groups: list[int] | None = Field(default=None, description="Group of each draw.")
    method: SampleMethod = Field(description="What made the sample.")

    _rust: "ParameterSampleProtocol" = PrivateAttr()

    @field_validator("log_weights")
    @classmethod
    def _normalize_log_weights(
        cls, log_weights: list[float], info: ValidationInfo
    ) -> list[float]:
        """
        Check the invariants in Rust and normalize the log weights; with a
        field before them that is not valid, pydantic reports that field.
        """
        data = info.data
        if not {"names", "bounds", "values"} <= data.keys():
            return log_weights
        checked = commol_rs.inference.ParameterSample(
            data["names"], data["bounds"], data["values"], log_weights, None
        )
        return checked.log_weights

    @model_validator(mode="after")
    def _build_engine_sample(self) -> Self:
        """Keep the sample of the engine, which computes the statistics and
        which the engine reads as the handle of this sample."""
        self._rust = commol_rs.inference.ParameterSample(
            self.names, self.bounds, self.values, self.log_weights, self.groups
        )
        return self

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

    @property
    def effective_size(self) -> float:
        """
        Kish's effective sample size, ``1 / Σ wᵢ²``: between 1 and the number
        of draws; equal weights give the number of draws.
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

#### weights

```python
weights: list[float]
```

The weight of each draw; they sum to 1.

#### effective_size

```python
effective_size: float
```

Kish's effective sample size, `1 / Σ wᵢ²`: between 1 and the number of draws; equal weights give the number of draws.

### Methods:

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Source code in `commol/uncertainty/sample.py`

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

Bases: `ResultModel`

What made a sample: the name of the method and its settings.

Attributes:

| Name       | Type                   | Description                                           |
| ---------- | ---------------------- | ----------------------------------------------------- |
| `name`     | `str`                  | The name of the method, such as "bayesian_bootstrap". |
| `settings` | `dict[str, JsonValue]` | The settings the method ran with.                     |

Source code in `commol/uncertainty/sample.py`

```python
class SampleMethod(ResultModel):
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

Bases: `ResultModel`

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

Source code in `commol/uncertainty/sample.py`

```python
class ParameterSummary(ResultModel):
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

The draws run in parallel in Rust, through the forward model of a calibration problem or of a sensitivity analysis: with a stochastic engine every replicate of a draw is a draw of the predictive, with an equal share of its weight. A band reduces the draws position by position with their weights; without observation noise it is a band of the model, not a prediction interval.

Source code in `commol/uncertainty/predictive.py`

```python
class Predictive:
    """
    The run of every draw of a parameter sample.

    The draws run in parallel in Rust, through the forward model of a
    calibration problem or of a sensitivity analysis: with a stochastic engine
    every replicate of a draw is a draw of the predictive, with an equal share
    of its weight. A band
    reduces the draws position by position with their weights; without
    observation noise it is a band of the model, not a prediction interval.
    """

    def __init__(
        self,
        source: PredictiveSource,
        sample: ParameterSample,
        positions: Sequence[float],
    ) -> None:
        """
        Run every draw of `sample` at `positions`.

        Parameters
        ----------
        source : PredictiveSource
            The calibration problem the sample is a sample of (`Calibrator`),
            or the sensitivity analysis of its inputs (`SensitivityAnalysis`).
        sample : ParameterSample
            Its parameters, in the order of the problem or the analysis.
        positions : Sequence[float]
            Positions to report at, sorted and unique.

        Raises
        ------
        ValueError
            Before any draw runs: if the sample's parameters are not the
            source's, or the positions are not sorted and unique or the engine
            does not report at one of them (a difference engine reports only
            at integer steps).
        InfeasibleStateError
            If a draw leaves the values its model allows. The message names
            the parameter set, then says what the run says (the compartment,
            the step and the flows), or what the initial amounts exceed.
        """
        self.source: PredictiveSource = source
        self.sample: ParameterSample = sample
        self.positions: list[float] = [float(position) for position in positions]
        self.output_names: list[str] = list(source.simulation.engine.output_names)
        self._rust = source._predictive(sample, self.positions)

    @property
    def weights(self) -> list[float]:
        """The weight of each draw of the predictive; they sum to 1."""
        return self._rust.weights

    @property
    def draw_seeds(self) -> list[int]:
        """
        The seed of each draw of `sample`, in its order.

        With a stochastic engine, replicate ``i`` of draw ``j`` runs with the
        seed replicate ``i`` of ``Simulation.replicates(count, seed=draw_seeds[j])``
        takes, so those replicates of the model at that draw give its runs.
        """
        return self._rust.draw_seeds

    @property
    def plot(self) -> PredictivePlot:
        """
        The figures of the draws: ``plot.series()`` and ``plot.cumulative()``,
        the weighted median and the band of each series.
        """
        return PredictivePlot(self)

    def runs(self) -> list[SimulationResults]:
        """
        The trajectory of every draw, in the order of `weights`.

        Returns
        -------
        list[SimulationResults]
            One run per draw, at `positions`, with the model of the source.
        """
        model = self.source.simulation.model_definition
        return [
            SimulationResults.from_rows(
                self.positions, self.output_names, rows, model=model
            )
            for rows in self._rust.trajectories
        ]

    def quantities(
        self, quantities: "Sequence[QuantityBase]"
    ) -> list[dict[str, float]]:
        """
        The quantities of interest of every draw, in the order of `weights`.

        Each run is reduced on its own, as `SimulationResults.quantities`
        reduces one, with the Rust code of `Propagation`. The predictive runs
        at `positions`, so they must hold ``quantity_schedule(quantities)``.

        Parameters
        ----------
        quantities : Sequence[quantity]
            The quantities; a `Trajectory` gives one scalar per position.

        Returns
        -------
        list[dict[str, float]]
            One dict per draw of the predictive, with one value per label of
            the quantities.

        Raises
        ------
        ValueError
            If there is no quantity, two scalars have one label, a quantity
            names an output the model does not have, or `positions` lack a
            position a quantity reads; the message lists every such position
            with its quantity.
        """
        return quantities_of_runs(self.runs(), self.positions, quantities)

    def report(
        self,
        confidence_level: float,
        positions: Sequence[float],
        window: float,
        windows: int,
        observed_data: Iterable[ObservedDataPoint] = (),
        accumulators: Iterable[str] = (),
        prediction_intervals: "PredictionIntervalsData | None" = None,
    ) -> "SimulationReport":
        """
        Report the runs of the draws as `SimulationResults.report` reports one.

        Every number becomes a ``{mean, median, ci_lower, ci_upper, min,
        max}`` block, reduced draw by draw with the weights of the draws; the
        layout is the one of the report of a run. Each leaf comes from the
        draws only, so the report takes no point estimate. The outputs of an
        accumulator come from the model of the source.

        Parameters
        ----------
        confidence_level : float
            Confidence level of the reported intervals, strictly between 0
            and 1.
        positions : Sequence[float]
            Positions the series and cumulative accumulators are sampled at;
            each must be one of `positions` of the predictive.
        window : float
            Length of one window on the axis.
        windows : int
            Number of windows on the observed and windowed axes.
        observed_data : Iterable[ObservedDataPoint], optional
            The calibration targets to report alongside the draws.
        accumulators : Iterable[str], optional
            Accumulator ids to report.
        prediction_intervals : PredictionIntervalsData | None, optional
            Prediction intervals of a conformal method
            (``ConformalResult.intervals``), reported as they are.

        Returns
        -------
        SimulationReport
            Keys ``observed``, ``series`` and ``accumulators``, then
            ``prediction_intervals`` when they are given.

        Raises
        ------
        ValueError
            If the confidence level is out of range, or the predictive has no
            value at a position the report reads.
        KeyError
            If an accumulator is not a bin or accumulator of the model.
        """
        from commol._simulation.reports import _ReportRuns, _simulation_report

        return _simulation_report(
            _ReportRuns(
                model=self.source.simulation.model_definition,
                runs=self.runs(),
                confidence_level=confidence_level,
                weights=self.weights,
            ),
            positions,
            window,
            windows,
            observed_data,
            accumulators,
            prediction_intervals,
        )

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
        names = self.output_names
        return PredictiveBand(
            confidence_level=confidence_level,
            positions=self.positions,
            mean=by_output(names, mean),
            median=by_output(names, median),
            lower=by_output(names, lower),
            upper=by_output(names, upper),
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
        noise: _ObservationNoise,
    ) -> ObservationDraws:
        """
        Every draw's value of each observation, with observation noise.

        The observations need not be the ones the problem was fitted to: held
        out data are read the same way, through the observation operator,
        and so are the observations of the model of an analysis. The noise
        is drawn from the calibrator's seed.

        Parameters
        ----------
        observed_data : Sequence[ObservedDataPoint]
            The observations.
        noise : NoNoise | ResidualNoise | DistributionNoise
            What an observation shows around a draw's prediction, with no
            default: ``NoNoise()`` gives a band of the model,
            ``ResidualNoise(...)`` and ``DistributionNoise()`` a prediction
            interval. Both need a calibrator: residuals come from a fit, and
            observation models from a problem.

        Returns
        -------
        ObservationDraws
            Each draw's value of each observation.

        Raises
        ------
        ValueError
            If an observation names an unknown output or factor, a series the
            residual noise has no residuals of, or a series with no
            observation model; `DistributionNoise` comes with a problem that
            has a loss; or the noise is not `NoNoise()` and the source is an
            analysis.
        InfeasibleStateError
            If a draw leaves the values its model allows. The message names
            the parameter set, then says what the run says (the compartment,
            the step and the flows), or what the initial amounts exceed.
        """
        draws = self.source._observation_draws(self.sample, list(observed_data), noise)
        return ObservationDraws(observed_data, draws)
```

### Attributes

#### weights

```python
weights: list[float]
```

The weight of each draw of the predictive; they sum to 1.

#### draw_seeds

```python
draw_seeds: list[int]
```

The seed of each draw of `sample`, in its order.

With a stochastic engine, replicate `i` of draw `j` runs with the seed replicate `i` of `Simulation.replicates(count, seed=draw_seeds[j])` takes, so those replicates of the model at that draw give its runs.

#### plot

```python
plot: PredictivePlot
```

The figures of the draws: `plot.series()` and `plot.cumulative()`, the weighted median and the band of each series.

### Methods:

#### __init__

```python
__init__(source: PredictiveSource, sample: ParameterSample, positions: Sequence[float]) -> None
```

Run every draw of `sample` at `positions`.

Parameters:

| Name        | Type               | Description                                                                                                                      | Default    |
| ----------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `source`    | `PredictiveSource` | The calibration problem the sample is a sample of (Calibrator), or the sensitivity analysis of its inputs (SensitivityAnalysis). | *required* |
| `sample`    | `ParameterSample`  | Its parameters, in the order of the problem or the analysis.                                                                     | *required* |
| `positions` | `Sequence[float]`  | Positions to report at, sorted and unique.                                                                                       | *required* |

Raises:

| Type                   | Description                                                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`           | Before any draw runs: if the sample's parameters are not the source's, or the positions are not sorted and unique or the engine does not report at one of them (a difference engine reports only at integer steps). |
| `InfeasibleStateError` | If a draw leaves the values its model allows. The message names the parameter set, then says what the run says (the compartment, the step and the flows), or what the initial amounts exceed.                       |

Source code in `commol/uncertainty/predictive.py`

```python
def __init__(
    self,
    source: PredictiveSource,
    sample: ParameterSample,
    positions: Sequence[float],
) -> None:
    """
    Run every draw of `sample` at `positions`.

    Parameters
    ----------
    source : PredictiveSource
        The calibration problem the sample is a sample of (`Calibrator`),
        or the sensitivity analysis of its inputs (`SensitivityAnalysis`).
    sample : ParameterSample
        Its parameters, in the order of the problem or the analysis.
    positions : Sequence[float]
        Positions to report at, sorted and unique.

    Raises
    ------
    ValueError
        Before any draw runs: if the sample's parameters are not the
        source's, or the positions are not sorted and unique or the engine
        does not report at one of them (a difference engine reports only
        at integer steps).
    InfeasibleStateError
        If a draw leaves the values its model allows. The message names
        the parameter set, then says what the run says (the compartment,
        the step and the flows), or what the initial amounts exceed.
    """
    self.source: PredictiveSource = source
    self.sample: ParameterSample = sample
    self.positions: list[float] = [float(position) for position in positions]
    self.output_names: list[str] = list(source.simulation.engine.output_names)
    self._rust = source._predictive(sample, self.positions)
```

#### runs

```python
runs() -> list[SimulationResults]
```

The trajectory of every draw, in the order of `weights`.

Returns:

| Type                      | Description                                                   |
| ------------------------- | ------------------------------------------------------------- |
| `list[SimulationResults]` | One run per draw, at positions, with the model of the source. |

Source code in `commol/uncertainty/predictive.py`

```python
def runs(self) -> list[SimulationResults]:
    """
    The trajectory of every draw, in the order of `weights`.

    Returns
    -------
    list[SimulationResults]
        One run per draw, at `positions`, with the model of the source.
    """
    model = self.source.simulation.model_definition
    return [
        SimulationResults.from_rows(
            self.positions, self.output_names, rows, model=model
        )
        for rows in self._rust.trajectories
    ]
```

#### quantities

```python
quantities(quantities: Sequence[QuantityBase]) -> list[dict[str, float]]
```

The quantities of interest of every draw, in the order of `weights`.

Each run is reduced on its own, as `SimulationResults.quantities` reduces one, with the Rust code of `Propagation`. The predictive runs at `positions`, so they must hold `quantity_schedule(quantities)`.

Parameters:

| Name         | Type                 | Description                                                 | Default    |
| ------------ | -------------------- | ----------------------------------------------------------- | ---------- |
| `quantities` | `Sequence[quantity]` | The quantities; a Trajectory gives one scalar per position. | *required* |

Returns:

| Type                     | Description                                                                      |
| ------------------------ | -------------------------------------------------------------------------------- |
| `list[dict[str, float]]` | One dict per draw of the predictive, with one value per label of the quantities. |

Raises:

| Type         | Description                                                                                                                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If there is no quantity, two scalars have one label, a quantity names an output the model does not have, or positions lack a position a quantity reads; the message lists every such position with its quantity. |

Source code in `commol/uncertainty/predictive.py`

```python
def quantities(
    self, quantities: "Sequence[QuantityBase]"
) -> list[dict[str, float]]:
    """
    The quantities of interest of every draw, in the order of `weights`.

    Each run is reduced on its own, as `SimulationResults.quantities`
    reduces one, with the Rust code of `Propagation`. The predictive runs
    at `positions`, so they must hold ``quantity_schedule(quantities)``.

    Parameters
    ----------
    quantities : Sequence[quantity]
        The quantities; a `Trajectory` gives one scalar per position.

    Returns
    -------
    list[dict[str, float]]
        One dict per draw of the predictive, with one value per label of
        the quantities.

    Raises
    ------
    ValueError
        If there is no quantity, two scalars have one label, a quantity
        names an output the model does not have, or `positions` lack a
        position a quantity reads; the message lists every such position
        with its quantity.
    """
    return quantities_of_runs(self.runs(), self.positions, quantities)
```

#### report

```python
report(confidence_level: float, positions: Sequence[float], window: float, windows: int, observed_data: Iterable[ObservedDataPoint] = (), accumulators: Iterable[str] = (), prediction_intervals: PredictionIntervalsData | None = None) -> SimulationReport
```

Report the runs of the draws as `SimulationResults.report` reports one.

Every number becomes a `{mean, median, ci_lower, ci_upper, min, max}` block, reduced draw by draw with the weights of the draws; the layout is the one of the report of a run. Each leaf comes from the draws only, so the report takes no point estimate. The outputs of an accumulator come from the model of the source.

Parameters:

| Name                   | Type                          | Description                                                                                                       | Default                                                                                       |
| ---------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `confidence_level`     | `float`                       | Confidence level of the reported intervals, strictly between 0 and 1.                                             | *required*                                                                                    |
| `positions`            | `Sequence[float]`             | Positions the series and cumulative accumulators are sampled at; each must be one of positions of the predictive. | *required*                                                                                    |
| `window`               | `float`                       | Length of one window on the axis.                                                                                 | *required*                                                                                    |
| `windows`              | `int`                         | Number of windows on the observed and windowed axes.                                                              | *required*                                                                                    |
| `observed_data`        | `Iterable[ObservedDataPoint]` | The calibration targets to report alongside the draws.                                                            | `()`                                                                                          |
| `accumulators`         | `Iterable[str]`               | Accumulator ids to report.                                                                                        | `()`                                                                                          |
| `prediction_intervals` | \`PredictionIntervalsData     | None\`                                                                                                            | Prediction intervals of a conformal method (ConformalResult.intervals), reported as they are. |

Returns:

| Type               | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `SimulationReport` | Keys observed, series and accumulators, then prediction_intervals when they are given. |

Raises:

| Type         | Description                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the confidence level is out of range, or the predictive has no value at a position the report reads. |
| `KeyError`   | If an accumulator is not a bin or accumulator of the model.                                             |

Source code in `commol/uncertainty/predictive.py`

```python
def report(
    self,
    confidence_level: float,
    positions: Sequence[float],
    window: float,
    windows: int,
    observed_data: Iterable[ObservedDataPoint] = (),
    accumulators: Iterable[str] = (),
    prediction_intervals: "PredictionIntervalsData | None" = None,
) -> "SimulationReport":
    """
    Report the runs of the draws as `SimulationResults.report` reports one.

    Every number becomes a ``{mean, median, ci_lower, ci_upper, min,
    max}`` block, reduced draw by draw with the weights of the draws; the
    layout is the one of the report of a run. Each leaf comes from the
    draws only, so the report takes no point estimate. The outputs of an
    accumulator come from the model of the source.

    Parameters
    ----------
    confidence_level : float
        Confidence level of the reported intervals, strictly between 0
        and 1.
    positions : Sequence[float]
        Positions the series and cumulative accumulators are sampled at;
        each must be one of `positions` of the predictive.
    window : float
        Length of one window on the axis.
    windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the draws.
    accumulators : Iterable[str], optional
        Accumulator ids to report.
    prediction_intervals : PredictionIntervalsData | None, optional
        Prediction intervals of a conformal method
        (``ConformalResult.intervals``), reported as they are.

    Returns
    -------
    SimulationReport
        Keys ``observed``, ``series`` and ``accumulators``, then
        ``prediction_intervals`` when they are given.

    Raises
    ------
    ValueError
        If the confidence level is out of range, or the predictive has no
        value at a position the report reads.
    KeyError
        If an accumulator is not a bin or accumulator of the model.
    """
    from commol._simulation.reports import _ReportRuns, _simulation_report

    return _simulation_report(
        _ReportRuns(
            model=self.source.simulation.model_definition,
            runs=self.runs(),
            confidence_level=confidence_level,
            weights=self.weights,
        ),
        positions,
        window,
        windows,
        observed_data,
        accumulators,
        prediction_intervals,
    )
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

Source code in `commol/uncertainty/predictive.py`

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
    names = self.output_names
    return PredictiveBand(
        confidence_level=confidence_level,
        positions=self.positions,
        mean=by_output(names, mean),
        median=by_output(names, median),
        lower=by_output(names, lower),
        upper=by_output(names, upper),
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

Source code in `commol/uncertainty/predictive.py`

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
at_observations(observed_data: Sequence[ObservedDataPoint], noise: _ObservationNoise) -> ObservationDraws
```

Every draw's value of each observation, with observation noise.

The observations need not be the ones the problem was fitted to: held out data are read the same way, through the observation operator, and so are the observations of the model of an analysis. The noise is drawn from the calibrator's seed.

Parameters:

| Name            | Type                          | Description       | Default             |
| --------------- | ----------------------------- | ----------------- | ------------------- |
| `observed_data` | `Sequence[ObservedDataPoint]` | The observations. | *required*          |
| `noise`         | \`NoNoise                     | ResidualNoise     | DistributionNoise\` |

Returns:

| Type               | Description                            |
| ------------------ | -------------------------------------- |
| `ObservationDraws` | Each draw's value of each observation. |

Raises:

| Type                   | Description                                                                                                                                                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`           | If an observation names an unknown output or factor, a series the residual noise has no residuals of, or a series with no observation model; DistributionNoise comes with a problem that has a loss; or the noise is not NoNoise() and the source is an analysis. |
| `InfeasibleStateError` | If a draw leaves the values its model allows. The message names the parameter set, then says what the run says (the compartment, the step and the flows), or what the initial amounts exceed.                                                                     |

Source code in `commol/uncertainty/predictive.py`

```python
def at_observations(
    self,
    observed_data: Sequence[ObservedDataPoint],
    noise: _ObservationNoise,
) -> ObservationDraws:
    """
    Every draw's value of each observation, with observation noise.

    The observations need not be the ones the problem was fitted to: held
    out data are read the same way, through the observation operator,
    and so are the observations of the model of an analysis. The noise
    is drawn from the calibrator's seed.

    Parameters
    ----------
    observed_data : Sequence[ObservedDataPoint]
        The observations.
    noise : NoNoise | ResidualNoise | DistributionNoise
        What an observation shows around a draw's prediction, with no
        default: ``NoNoise()`` gives a band of the model,
        ``ResidualNoise(...)`` and ``DistributionNoise()`` a prediction
        interval. Both need a calibrator: residuals come from a fit, and
        observation models from a problem.

    Returns
    -------
    ObservationDraws
        Each draw's value of each observation.

    Raises
    ------
    ValueError
        If an observation names an unknown output or factor, a series the
        residual noise has no residuals of, or a series with no
        observation model; `DistributionNoise` comes with a problem that
        has a loss; or the noise is not `NoNoise()` and the source is an
        analysis.
    InfeasibleStateError
        If a draw leaves the values its model allows. The message names
        the parameter set, then says what the run says (the compartment,
        the step and the flows), or what the initial amounts exceed.
    """
    draws = self.source._observation_draws(self.sample, list(observed_data), noise)
    return ObservationDraws(observed_data, draws)
```

## PredictiveBand

Bases: `ResultModel`

The band of every output of a predictive.

Attributes:

| Name                         | Type                     | Description                                                                           |
| ---------------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| `confidence_level`           | `float`                  | The level of the central interval.                                                    |
| `positions`                  | `list[float]`            | Where the values are.                                                                 |
| `mean, median, lower, upper` | `dict[str, list[float]]` | For each output, one value per position, reduced across the draws with their weights. |

Source code in `commol/uncertainty/predictive.py`

```python
class PredictiveBand(ResultModel):
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

Bases: `ResultModel`

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

Source code in `commol/uncertainty/predictive.py`

```python
class SeriesBand(ResultModel):
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

The values are read through the observation operator of the calibration problem (sums of outputs, windows, scales, factors), the one the loss uses. Without observation noise their interval is a band of the model, not a prediction interval.

Source code in `commol/uncertainty/predictive.py`

```python
class ObservationDraws:
    """
    Every draw's value of a set of observations.

    The values are read through the observation operator of the calibration
    problem (sums of outputs, windows, scales, factors), the one the loss
    uses. Without observation noise their interval is a band of the model,
    not a prediction interval.
    """

    def __init__(
        self,
        observations: Sequence[ObservedDataPoint],
        draws: "ObservationDrawsProtocol",
    ) -> None:
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
        self._draws: "ObservationDrawsProtocol" = draws

    @property
    def values(self) -> list[list[float]]:
        """Each draw's value of each observation, ``[draw][observation]``."""
        return self._draws.values

    @property
    def weights(self) -> list[float]:
        """The weight of each draw; they sum to 1."""
        return self._draws.weights

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
        mean, median, lower, upper = self._draws.band(confidence_level)
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
__init__(observations: Sequence[ObservedDataPoint], draws: ObservationDrawsProtocol) -> None
```

Wrap the draws of `observations`.

Parameters:

| Name           | Type                          | Description                                           | Default    |
| -------------- | ----------------------------- | ----------------------------------------------------- | ---------- |
| `observations` | `Sequence[ObservedDataPoint]` | The observations, in the order of each draw's values. | *required* |
| `draws`        | `ObservationDrawsProtocol`    | The draws, computed in Rust.                          | *required* |

Source code in `commol/uncertainty/predictive.py`

```python
def __init__(
    self,
    observations: Sequence[ObservedDataPoint],
    draws: "ObservationDrawsProtocol",
) -> None:
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
    self._draws: "ObservationDrawsProtocol" = draws
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

Source code in `commol/uncertainty/predictive.py`

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
    mean, median, lower, upper = self._draws.band(confidence_level)
    return SeriesBand(
        confidence_level=confidence_level,
        positions=[point.at for point in self.observations],
        mean=mean,
        median=median,
        lower=lower,
        upper=upper,
    )
```

## PredictivePlot

The figures of a predictive, read from `predictive.plot`.

Each panel shows the weighted median of the draws and their band, with no other run; the model of the source names the series.

Source code in `commol/uncertainty/plots.py`

```python
class PredictivePlot:
    """
    The figures of a predictive, read from ``predictive.plot``.

    Each panel shows the weighted median of the draws and their band, with no
    other run; the model of the source names the series.
    """

    def __init__(self, predictive: "Predictive") -> None:
        """
        The figures of `predictive`.

        Parameters
        ----------
        predictive : Predictive
            The predictive.
        """
        self._predictive = predictive

    def series(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        prediction_intervals: "PredictionIntervalsData | None" = None,
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the weighted median of the draws of each series and their band,
        one panel per series.

        Each draw is read through the observation operator of the problem
        (the sums, windows, scales and factors of the observations of the
        series, each at its own position when they do not share one operator
        with no window), then the draws are reduced with their weights.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, each drawn as a point on the panel of its label;
            they set the operator each draw is read through.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        series : list[str] | None
            What to draw, as in `ResultsPlot.series`.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `ResultsPlot.series`.
        tick_every : float | None
            The distance between labelled positions, as in
            `ResultsPlot.series`.
        x_label : str
            The label of the x axis.
        prediction_intervals : PredictionIntervalsData | None
            Prediction intervals of a conformal method, as in
            `ResultsPlot.series`.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the median.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If an observation names an unknown output or factor, the
            confidence level is out of range, or ``axes`` has no panel of a
            series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return predictive_series(
            self._predictive,
            output_file=output_file,
            observed_data=observed_data,
            config=config,
            series=series,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            prediction_intervals=prediction_intervals,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )

    def cumulative(
        self,
        output_file: str | None = None,
        observed_data: "list[ObservedDataPoint] | None" = None,
        config: "PlotConfig | None" = None,
        series: list[str] | None = None,
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw the weighted median and the band of the running total of each
        series across the draws, one panel per series.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        observed_data : list[ObservedDataPoint] | None
            Observations, drawn as their running total on the panel of their
            label. Each observation is divided by its scale and by the
            weighted median of its factor in the sample.
        config : PlotConfig | None
            Layout, size and style. If None, the defaults.
        series : list[str] | None
            What to draw, as in `ResultsPlot.series`.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.series`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the line of the median.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If an observation names a factor the problem does not have, the
            confidence level is out of range, or ``axes`` has no panel of a
            series.
        KeyError
            If an entry of ``series`` is neither an output nor a bin or
            accumulator id of the model.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return predictive_cumulative(
            self._predictive,
            output_file=output_file,
            observed_data=observed_data,
            config=config,
            series=series,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )

    def observations(
        self,
        observed_data: "list[ObservedDataPoint]",
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        show_legend: bool = True,
        position_label: Callable[[float], str] | None = None,
        tick_every: float | None = None,
        x_label: str = "Step",
        confidence_level: float = 0.95,
        axes: "Figure | None" = None,
        **kwargs: str | int | float | bool | None,
    ) -> "Figure":
        """
        Draw each observed series with the weighted median of the draws and
        their band, read through its observation operator, one panel per
        label.

        The values are those of ``at_observations(observed_data, NoNoise())``:
        each draw's value of each observation through its own operator (the
        sum of its outputs, its window, its scale and the draw's value of its
        factor),
        the values the loss compares. A value with a window is drawn as a
        step over its window with a marker at ``at``, and its band as a
        rectangle over the same window; a value with no window is a marker,
        joined only as a guide. The band is a band of the model, with no
        observation noise, not a prediction interval, and its legend says so.

        Parameters
        ----------
        observed_data : list[ObservedDataPoint]
            The observations, as in `ResultsPlot.observations`.
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        show_legend : bool
            Whether each panel shows its legend.
        position_label : Callable[[float], str] | None
            Turns a position into a tick label, as in `ResultsPlot.series`.
        tick_every : float | None
            The distance between labelled positions, as in
            `ResultsPlot.series`.
        x_label : str
            The label of the x axis.
        confidence_level : float
            The level of the band, strictly between 0 and 1.
        axes : Figure | None
            A figure of commol to draw on, as in `ResultsPlot.observations`.
        **kwargs : str | int | float | bool | None
            Passed to ``Axes.plot`` for the steps and the guide of the median.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If there are no observations, an observation names an unknown
            output or factor, the confidence level is out of range, or
            ``axes`` has no panel of a label.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        return predictive_observations(
            self._predictive,
            observed_data,
            output_file=output_file,
            config=config,
            show_legend=show_legend,
            position_label=position_label,
            tick_every=tick_every,
            x_label=x_label,
            confidence_level=confidence_level,
            axes=axes,
            **kwargs,
        )

    def _panel_model(self) -> "Model | None":
        """The model of the source of the predictive."""
        return self._predictive.source.simulation.model_definition

    def _panel_outputs(self) -> PanelOutputs:
        """The outputs of the predictive."""
        return PanelOutputs(self._predictive.output_names, self._panel_model())

    def _series_panels(
        self,
        confidence_level: float,
        observed_data: "list[ObservedDataPoint]",
        series: list[str],
    ) -> list[Panel]:
        """The panels of `series`, as `series` draws them."""
        return predictive_panels(
            self._predictive, confidence_level, observed_data, series
        )

    def _window_trace(
        self,
        outputs: PanelOutputs,
        series_id: str,
        confidence_level: float,
        window: float,
    ) -> tuple[Trace, Band | None]:
        """The median gain of a series over each complete window, and its band."""
        return predictive_window(
            self._predictive, outputs, series_id, confidence_level, window
        )

    def _observation_panels(
        self, confidence_level: float, observed_data: "list[ObservedDataPoint]"
    ) -> list[ObservationPanel]:
        """The panels of the observed series, as `observations` draws them."""
        return predictive_observation_panels(
            self._predictive, confidence_level, observed_data
        )
```

### Methods:

#### __init__

```python
__init__(predictive: Predictive) -> None
```

The figures of `predictive`.

Parameters:

| Name         | Type         | Description     | Default    |
| ------------ | ------------ | --------------- | ---------- |
| `predictive` | `Predictive` | The predictive. | *required* |

Source code in `commol/uncertainty/plots.py`

```python
def __init__(self, predictive: "Predictive") -> None:
    """
    The figures of `predictive`.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    """
    self._predictive = predictive
```

#### series

```python
series(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, config: PlotConfig | None = None, series: list[str] | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', prediction_intervals: PredictionIntervalsData | None = None, confidence_level: float = 0.95, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the weighted median of the draws of each series and their band, one panel per series.

Each draw is read through the observation operator of the problem (the sums, windows, scales and factors of the observations of the series, each at its own position when they do not share one operator with no window), then the draws are reduced with their weights.

Parameters:

| Name                   | Type                       | Description                                      | Default                                                                                                         |
| ---------------------- | -------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `output_file`          | \`str                      | None\`                                           | Path to save the figure to. If None, the figure is only returned.                                               |
| `observed_data`        | \`list[ObservedDataPoint]  | None\`                                           | Observations, each drawn as a point on the panel of its label; they set the operator each draw is read through. |
| `config`               | \`PlotConfig               | None\`                                           | Layout and style. If None, the defaults.                                                                        |
| `series`               | \`list[str]                | None\`                                           | What to draw, as in ResultsPlot.series.                                                                         |
| `show_legend`          | `bool`                     | Whether each panel shows its legend.             | `True`                                                                                                          |
| `position_label`       | \`Callable\[[float], str\] | None\`                                           | Turns a position into a tick label, as in ResultsPlot.series.                                                   |
| `tick_every`           | \`float                    | None\`                                           | The distance between labelled positions, as in ResultsPlot.series.                                              |
| `x_label`              | `str`                      | The label of the x axis.                         | `'Step'`                                                                                                        |
| `prediction_intervals` | \`PredictionIntervalsData  | None\`                                           | Prediction intervals of a conformal method, as in ResultsPlot.series.                                           |
| `confidence_level`     | `float`                    | The level of the band, strictly between 0 and 1. | `0.95`                                                                                                          |
| `axes`                 | \`Figure                   | None\`                                           | A figure of commol to draw on, as in ResultsPlot.series.                                                        |
| `**kwargs`             | \`str                      | int                                              | float                                                                                                           |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If an observation names an unknown output or factor, the confidence level is out of range, or axes has no panel of a series. |
| `KeyError`    | If an entry of series is neither an output nor a bin or accumulator id of the model.                                         |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                 |

Source code in `commol/uncertainty/plots.py`

```python
def series(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    prediction_intervals: "PredictionIntervalsData | None" = None,
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the weighted median of the draws of each series and their band,
    one panel per series.

    Each draw is read through the observation operator of the problem
    (the sums, windows, scales and factors of the observations of the
    series, each at its own position when they do not share one operator
    with no window), then the draws are reduced with their weights.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, each drawn as a point on the panel of its label;
        they set the operator each draw is read through.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    series : list[str] | None
        What to draw, as in `ResultsPlot.series`.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position into a tick label, as in `ResultsPlot.series`.
    tick_every : float | None
        The distance between labelled positions, as in
        `ResultsPlot.series`.
    x_label : str
        The label of the x axis.
    prediction_intervals : PredictionIntervalsData | None
        Prediction intervals of a conformal method, as in
        `ResultsPlot.series`.
    confidence_level : float
        The level of the band, strictly between 0 and 1.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the median.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If an observation names an unknown output or factor, the
        confidence level is out of range, or ``axes`` has no panel of a
        series.
    KeyError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id of the model.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return predictive_series(
        self._predictive,
        output_file=output_file,
        observed_data=observed_data,
        config=config,
        series=series,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        prediction_intervals=prediction_intervals,
        confidence_level=confidence_level,
        axes=axes,
        **kwargs,
    )
```

#### cumulative

```python
cumulative(output_file: str | None = None, observed_data: list[ObservedDataPoint] | None = None, config: PlotConfig | None = None, series: list[str] | None = None, confidence_level: float = 0.95, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw the weighted median and the band of the running total of each series across the draws, one panel per series.

Parameters:

| Name               | Type                      | Description                                      | Default                                                                                                                                                                  |
| ------------------ | ------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `output_file`      | \`str                     | None\`                                           | Path to save the figure to. If None, the figure is only returned.                                                                                                        |
| `observed_data`    | \`list[ObservedDataPoint] | None\`                                           | Observations, drawn as their running total on the panel of their label. Each observation is divided by its scale and by the weighted median of its factor in the sample. |
| `config`           | \`PlotConfig              | None\`                                           | Layout, size and style. If None, the defaults.                                                                                                                           |
| `series`           | \`list[str]               | None\`                                           | What to draw, as in ResultsPlot.series.                                                                                                                                  |
| `confidence_level` | `float`                   | The level of the band, strictly between 0 and 1. | `0.95`                                                                                                                                                                   |
| `axes`             | \`Figure                  | None\`                                           | A figure of commol to draw on, as in ResultsPlot.series.                                                                                                                 |
| `**kwargs`         | \`str                     | int                                              | float                                                                                                                                                                    |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                         |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`  | If an observation names a factor the problem does not have, the confidence level is out of range, or axes has no panel of a series. |
| `KeyError`    | If an entry of series is neither an output nor a bin or accumulator id of the model.                                                |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                        |

Source code in `commol/uncertainty/plots.py`

```python
def cumulative(
    self,
    output_file: str | None = None,
    observed_data: "list[ObservedDataPoint] | None" = None,
    config: "PlotConfig | None" = None,
    series: list[str] | None = None,
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw the weighted median and the band of the running total of each
    series across the draws, one panel per series.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    observed_data : list[ObservedDataPoint] | None
        Observations, drawn as their running total on the panel of their
        label. Each observation is divided by its scale and by the
        weighted median of its factor in the sample.
    config : PlotConfig | None
        Layout, size and style. If None, the defaults.
    series : list[str] | None
        What to draw, as in `ResultsPlot.series`.
    confidence_level : float
        The level of the band, strictly between 0 and 1.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.series`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the line of the median.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If an observation names a factor the problem does not have, the
        confidence level is out of range, or ``axes`` has no panel of a
        series.
    KeyError
        If an entry of ``series`` is neither an output nor a bin or
        accumulator id of the model.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return predictive_cumulative(
        self._predictive,
        output_file=output_file,
        observed_data=observed_data,
        config=config,
        series=series,
        confidence_level=confidence_level,
        axes=axes,
        **kwargs,
    )
```

#### observations

```python
observations(observed_data: list[ObservedDataPoint], output_file: str | None = None, config: PlotConfig | None = None, show_legend: bool = True, position_label: Callable[[float], str] | None = None, tick_every: float | None = None, x_label: str = 'Step', confidence_level: float = 0.95, axes: Figure | None = None, **kwargs: str | int | float | bool | None) -> Figure
```

Draw each observed series with the weighted median of the draws and their band, read through its observation operator, one panel per label.

The values are those of `at_observations(observed_data, NoNoise())`: each draw's value of each observation through its own operator (the sum of its outputs, its window, its scale and the draw's value of its factor), the values the loss compares. A value with a window is drawn as a step over its window with a marker at `at`, and its band as a rectangle over the same window; a value with no window is a marker, joined only as a guide. The band is a band of the model, with no observation noise, not a prediction interval, and its legend says so.

Parameters:

| Name               | Type                       | Description                                       | Default                                                            |
| ------------------ | -------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| `observed_data`    | `list[ObservedDataPoint]`  | The observations, as in ResultsPlot.observations. | *required*                                                         |
| `output_file`      | \`str                      | None\`                                            | Path to save the figure to. If None, the figure is only returned.  |
| `config`           | \`PlotConfig               | None\`                                            | Layout and style. If None, the defaults.                           |
| `show_legend`      | `bool`                     | Whether each panel shows its legend.              | `True`                                                             |
| `position_label`   | \`Callable\[[float], str\] | None\`                                            | Turns a position into a tick label, as in ResultsPlot.series.      |
| `tick_every`       | \`float                    | None\`                                            | The distance between labelled positions, as in ResultsPlot.series. |
| `x_label`          | `str`                      | The label of the x axis.                          | `'Step'`                                                           |
| `confidence_level` | `float`                    | The level of the band, strictly between 0 and 1.  | `0.95`                                                             |
| `axes`             | \`Figure                   | None\`                                            | A figure of commol to draw on, as in ResultsPlot.observations.     |
| `**kwargs`         | \`str                      | int                                               | float                                                              |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ValueError`  | If there are no observations, an observation names an unknown output or factor, the confidence level is out of range, or axes has no panel of a label. |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]').                                                                                           |

Source code in `commol/uncertainty/plots.py`

```python
def observations(
    self,
    observed_data: "list[ObservedDataPoint]",
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    show_legend: bool = True,
    position_label: Callable[[float], str] | None = None,
    tick_every: float | None = None,
    x_label: str = "Step",
    confidence_level: float = 0.95,
    axes: "Figure | None" = None,
    **kwargs: str | int | float | bool | None,
) -> "Figure":
    """
    Draw each observed series with the weighted median of the draws and
    their band, read through its observation operator, one panel per
    label.

    The values are those of ``at_observations(observed_data, NoNoise())``:
    each draw's value of each observation through its own operator (the
    sum of its outputs, its window, its scale and the draw's value of its
    factor),
    the values the loss compares. A value with a window is drawn as a
    step over its window with a marker at ``at``, and its band as a
    rectangle over the same window; a value with no window is a marker,
    joined only as a guide. The band is a band of the model, with no
    observation noise, not a prediction interval, and its legend says so.

    Parameters
    ----------
    observed_data : list[ObservedDataPoint]
        The observations, as in `ResultsPlot.observations`.
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    show_legend : bool
        Whether each panel shows its legend.
    position_label : Callable[[float], str] | None
        Turns a position into a tick label, as in `ResultsPlot.series`.
    tick_every : float | None
        The distance between labelled positions, as in
        `ResultsPlot.series`.
    x_label : str
        The label of the x axis.
    confidence_level : float
        The level of the band, strictly between 0 and 1.
    axes : Figure | None
        A figure of commol to draw on, as in `ResultsPlot.observations`.
    **kwargs : str | int | float | bool | None
        Passed to ``Axes.plot`` for the steps and the guide of the median.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If there are no observations, an observation names an unknown
        output or factor, the confidence level is out of range, or
        ``axes`` has no panel of a label.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    return predictive_observations(
        self._predictive,
        observed_data,
        output_file=output_file,
        config=config,
        show_legend=show_legend,
        position_label=position_label,
        tick_every=tick_every,
        x_label=x_label,
        confidence_level=confidence_level,
        axes=axes,
        **kwargs,
    )
```

## Scenarios

Bases: `Mapping[str, Predictive]`

Every draw of a sample run under each scenario, with the draws paired.

A scenario is a set of parameter values that override those of the model of the source; the source and its model stay unchanged. Each scenario holds the `Predictive` of the sample on a copy of the model with its values, built from the source with the seed of the source (`with_model`). So run `k` of every scenario is the same draw, and with a stochastic engine it runs with the same seed: common random numbers, whose noise cancels in a difference.

`Scenarios` is a mapping from the name of a scenario to its `Predictive`, so `plot_compared(scenarios)` draws them on the same panels.

Source code in `commol/uncertainty/scenarios.py`

```python
class Scenarios(Mapping[str, Predictive]):
    """
    Every draw of a sample run under each scenario, with the draws paired.

    A scenario is a set of parameter values that override those of the model
    of the source; the source and its model stay unchanged. Each scenario
    holds the `Predictive` of the sample on a copy of the model with its
    values, built from the source with the seed of the source
    (`with_model`). So run ``k`` of every scenario is the same draw, and with
    a stochastic engine it runs with the same seed: common random numbers,
    whose noise cancels in a difference.

    `Scenarios` is a mapping from the name of a scenario to its `Predictive`,
    so ``plot_compared(scenarios)`` draws them on the same panels.
    """

    def __init__(
        self,
        source: PredictiveSource,
        sample: ParameterSample,
        scenarios: Mapping[
            str,
            Mapping[str, float | str | list[tuple[float, float]] | SeriesValue | None],
        ],
        positions: Sequence[float],
    ) -> None:
        """
        Run every draw of `sample` under each scenario, at `positions`.

        Parameters
        ----------
        source : PredictiveSource
            The calibration problem the sample is a sample of (`Calibrator`),
            or the sensitivity analysis of its inputs (`SensitivityAnalysis`).
        sample : ParameterSample
            Its parameters, in the order of the problem or the analysis.
        scenarios : Mapping[str, Mapping[str, value of a parameter]]
            Each scenario by name: the value of each parameter it overrides,
            any value `Model.update_parameters` takes, such as a number, a
            formula or the points of a series. An empty mapping keeps the
            model as it is.
        positions : Sequence[float]
            Positions to report at, sorted and unique. A comparison reads the
            runs here, so they hold ``quantity_schedule(quantities)``.

        Raises
        ------
        ValueError
            Before any draw runs: if there is no scenario, a scenario sets a
            parameter of the sample or one the model does not have (the
            message names the scenario and the parameter), or a value does
            not fit the model. Then as `Predictive` raises.
        InfeasibleStateError
            If a draw leaves the values its model allows, as `Predictive`
            raises.
        """
        if not scenarios:
            raise ValueError("Give at least one scenario.")
        model = source.simulation.model_definition
        models = {}
        for name, overrides in scenarios.items():
            held = [parameter for parameter in overrides if parameter in sample.names]
            if held:
                raise ValueError(
                    f"Scenario '{name}' sets {held}, which the sample holds: each "
                    "draw sets them. A scenario overrides only parameters the "
                    "sample does not hold."
                )
            scenario_model = model.model_copy(deep=True)
            try:
                scenario_model.update_parameters(overrides)
            except ValueError as error:
                raise ValueError(f"Scenario '{name}': {error}") from error
            models[name] = scenario_model
        self.sample: ParameterSample = sample
        self.positions: list[float] = [float(position) for position in positions]
        self._predictives: dict[str, Predictive] = {
            name: Predictive(source.with_model(scenario_model), sample, positions)
            for name, scenario_model in models.items()
        }

    def __getitem__(self, name: str) -> Predictive:
        """
        The predictive of one scenario.

        Parameters
        ----------
        name : str
            The name of the scenario.

        Returns
        -------
        Predictive
            The runs of every draw under the scenario.

        Raises
        ------
        KeyError
            If there is no scenario of that name; the message lists the names.
        """
        if name not in self._predictives:
            raise KeyError(
                f"No scenario '{name}'. Scenarios: {list(self._predictives)}."
            )
        return self._predictives[name]

    def __iter__(self) -> Iterator[str]:
        """The names of the scenarios, in the order they were given."""
        return iter(self._predictives)

    def __len__(self) -> int:
        """The number of scenarios."""
        return len(self._predictives)

    @property
    def weights(self) -> list[float]:
        """The weight of each run, the same in every scenario; they sum to 1."""
        return next(iter(self._predictives.values())).weights

    def summary(
        self,
        name: str,
        quantities: "Sequence[QuantityBase]",
        confidence_level: float,
    ) -> list[ScenarioSummary]:
        """
        The weighted distribution of each quantity across the runs of one
        scenario.

        Parameters
        ----------
        name : str
            The scenario.
        quantities : Sequence[quantity]
            The quantities; a `Trajectory` gives one scalar per position.
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        list[ScenarioSummary]
            One summary per label of the quantities, in their order. With one
            run, the value of that run and no interval.

        Raises
        ------
        KeyError
            If there is no scenario of that name.
        ValueError
            If the confidence level is not strictly between 0 and 1, a
            quantity does not fit the runs (as `Predictive.quantities`
            raises), or it is NaN in a run.
        """
        return _summaries(
            self[name].quantities(quantities), self.weights, confidence_level
        )

    def compare(
        self,
        name: str,
        baseline: str,
        quantities: "Sequence[QuantityBase]",
        confidence_level: float,
    ) -> list[ScenarioSummary]:
        """
        The weighted distribution of the difference of each quantity between
        two scenarios, taken run by run.

        The difference ``name − baseline`` is taken within each run, where
        both scenarios have the same draw and seed, and then reduced across
        the runs with their weights. So its interval is the interval of the
        paired differences, not the difference of two intervals.

        Parameters
        ----------
        name : str
            The scenario compared.
        baseline : str
            The scenario subtracted.
        quantities : Sequence[quantity]
            The quantities; a `Trajectory` gives one scalar per position, each
            compared at its position.
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        list[ScenarioSummary]
            One summary per label of the quantities, in their order. With one
            run, the difference of that run and no interval.

        Raises
        ------
        KeyError
            If there is no scenario of either name.
        ValueError
            If the confidence level is not strictly between 0 and 1, a
            quantity does not fit the runs (as `Predictive.quantities`
            raises), or it is NaN in a run of either scenario.
        """
        compared = self[name].quantities(quantities)
        subtracted = self[baseline].quantities(quantities)
        differences = [
            {label: value - other[label] for label, value in run.items()}
            for run, other in zip(compared, subtracted, strict=True)
        ]
        return _summaries(differences, self.weights, confidence_level)
```

### Attributes

#### weights

```python
weights: list[float]
```

The weight of each run, the same in every scenario; they sum to 1.

### Methods:

#### __init__

```python
__init__(source: PredictiveSource, sample: ParameterSample, scenarios: Mapping[str, Mapping[str, float | str | list[tuple[float, float]] | SeriesValue | None]], positions: Sequence[float]) -> None
```

Run every draw of `sample` under each scenario, at `positions`.

Parameters:

| Name        | Type                                               | Description                                                                                                                                                                                                 | Default    |
| ----------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `source`    | `PredictiveSource`                                 | The calibration problem the sample is a sample of (Calibrator), or the sensitivity analysis of its inputs (SensitivityAnalysis).                                                                            | *required* |
| `sample`    | `ParameterSample`                                  | Its parameters, in the order of the problem or the analysis.                                                                                                                                                | *required* |
| `scenarios` | `Mapping[str, Mapping[str, value of a parameter]]` | Each scenario by name: the value of each parameter it overrides, any value Model.update_parameters takes, such as a number, a formula or the points of a series. An empty mapping keeps the model as it is. | *required* |
| `positions` | `Sequence[float]`                                  | Positions to report at, sorted and unique. A comparison reads the runs here, so they hold quantity_schedule(quantities).                                                                                    | *required* |

Raises:

| Type                   | Description                                                                                                                                                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`           | Before any draw runs: if there is no scenario, a scenario sets a parameter of the sample or one the model does not have (the message names the scenario and the parameter), or a value does not fit the model. Then as Predictive raises. |
| `InfeasibleStateError` | If a draw leaves the values its model allows, as Predictive raises.                                                                                                                                                                       |

Source code in `commol/uncertainty/scenarios.py`

```python
def __init__(
    self,
    source: PredictiveSource,
    sample: ParameterSample,
    scenarios: Mapping[
        str,
        Mapping[str, float | str | list[tuple[float, float]] | SeriesValue | None],
    ],
    positions: Sequence[float],
) -> None:
    """
    Run every draw of `sample` under each scenario, at `positions`.

    Parameters
    ----------
    source : PredictiveSource
        The calibration problem the sample is a sample of (`Calibrator`),
        or the sensitivity analysis of its inputs (`SensitivityAnalysis`).
    sample : ParameterSample
        Its parameters, in the order of the problem or the analysis.
    scenarios : Mapping[str, Mapping[str, value of a parameter]]
        Each scenario by name: the value of each parameter it overrides,
        any value `Model.update_parameters` takes, such as a number, a
        formula or the points of a series. An empty mapping keeps the
        model as it is.
    positions : Sequence[float]
        Positions to report at, sorted and unique. A comparison reads the
        runs here, so they hold ``quantity_schedule(quantities)``.

    Raises
    ------
    ValueError
        Before any draw runs: if there is no scenario, a scenario sets a
        parameter of the sample or one the model does not have (the
        message names the scenario and the parameter), or a value does
        not fit the model. Then as `Predictive` raises.
    InfeasibleStateError
        If a draw leaves the values its model allows, as `Predictive`
        raises.
    """
    if not scenarios:
        raise ValueError("Give at least one scenario.")
    model = source.simulation.model_definition
    models = {}
    for name, overrides in scenarios.items():
        held = [parameter for parameter in overrides if parameter in sample.names]
        if held:
            raise ValueError(
                f"Scenario '{name}' sets {held}, which the sample holds: each "
                "draw sets them. A scenario overrides only parameters the "
                "sample does not hold."
            )
        scenario_model = model.model_copy(deep=True)
        try:
            scenario_model.update_parameters(overrides)
        except ValueError as error:
            raise ValueError(f"Scenario '{name}': {error}") from error
        models[name] = scenario_model
    self.sample: ParameterSample = sample
    self.positions: list[float] = [float(position) for position in positions]
    self._predictives: dict[str, Predictive] = {
        name: Predictive(source.with_model(scenario_model), sample, positions)
        for name, scenario_model in models.items()
    }
```

#### __getitem__

```python
__getitem__(name: str) -> Predictive
```

The predictive of one scenario.

Parameters:

| Name   | Type  | Description               | Default    |
| ------ | ----- | ------------------------- | ---------- |
| `name` | `str` | The name of the scenario. | *required* |

Returns:

| Type         | Description                                |
| ------------ | ------------------------------------------ |
| `Predictive` | The runs of every draw under the scenario. |

Raises:

| Type       | Description                                                        |
| ---------- | ------------------------------------------------------------------ |
| `KeyError` | If there is no scenario of that name; the message lists the names. |

Source code in `commol/uncertainty/scenarios.py`

```python
def __getitem__(self, name: str) -> Predictive:
    """
    The predictive of one scenario.

    Parameters
    ----------
    name : str
        The name of the scenario.

    Returns
    -------
    Predictive
        The runs of every draw under the scenario.

    Raises
    ------
    KeyError
        If there is no scenario of that name; the message lists the names.
    """
    if name not in self._predictives:
        raise KeyError(
            f"No scenario '{name}'. Scenarios: {list(self._predictives)}."
        )
    return self._predictives[name]
```

#### __iter__

```python
__iter__() -> Iterator[str]
```

The names of the scenarios, in the order they were given.

Source code in `commol/uncertainty/scenarios.py`

```python
def __iter__(self) -> Iterator[str]:
    """The names of the scenarios, in the order they were given."""
    return iter(self._predictives)
```

#### __len__

```python
__len__() -> int
```

The number of scenarios.

Source code in `commol/uncertainty/scenarios.py`

```python
def __len__(self) -> int:
    """The number of scenarios."""
    return len(self._predictives)
```

#### summary

```python
summary(name: str, quantities: Sequence[QuantityBase], confidence_level: float) -> list[ScenarioSummary]
```

The weighted distribution of each quantity across the runs of one scenario.

Parameters:

| Name               | Type                 | Description                                                 | Default    |
| ------------------ | -------------------- | ----------------------------------------------------------- | ---------- |
| `name`             | `str`                | The scenario.                                               | *required* |
| `quantities`       | `Sequence[quantity]` | The quantities; a Trajectory gives one scalar per position. | *required* |
| `confidence_level` | `float`              | Strictly between 0 and 1.                                   | *required* |

Returns:

| Type                    | Description                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| `list[ScenarioSummary]` | One summary per label of the quantities, in their order. With one run, the value of that run and no interval. |

Raises:

| Type         | Description                                                                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KeyError`   | If there is no scenario of that name.                                                                                                               |
| `ValueError` | If the confidence level is not strictly between 0 and 1, a quantity does not fit the runs (as Predictive.quantities raises), or it is NaN in a run. |

Source code in `commol/uncertainty/scenarios.py`

```python
def summary(
    self,
    name: str,
    quantities: "Sequence[QuantityBase]",
    confidence_level: float,
) -> list[ScenarioSummary]:
    """
    The weighted distribution of each quantity across the runs of one
    scenario.

    Parameters
    ----------
    name : str
        The scenario.
    quantities : Sequence[quantity]
        The quantities; a `Trajectory` gives one scalar per position.
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    list[ScenarioSummary]
        One summary per label of the quantities, in their order. With one
        run, the value of that run and no interval.

    Raises
    ------
    KeyError
        If there is no scenario of that name.
    ValueError
        If the confidence level is not strictly between 0 and 1, a
        quantity does not fit the runs (as `Predictive.quantities`
        raises), or it is NaN in a run.
    """
    return _summaries(
        self[name].quantities(quantities), self.weights, confidence_level
    )
```

#### compare

```python
compare(name: str, baseline: str, quantities: Sequence[QuantityBase], confidence_level: float) -> list[ScenarioSummary]
```

The weighted distribution of the difference of each quantity between two scenarios, taken run by run.

The difference `name − baseline` is taken within each run, where both scenarios have the same draw and seed, and then reduced across the runs with their weights. So its interval is the interval of the paired differences, not the difference of two intervals.

Parameters:

| Name               | Type                 | Description                                                                                | Default    |
| ------------------ | -------------------- | ------------------------------------------------------------------------------------------ | ---------- |
| `name`             | `str`                | The scenario compared.                                                                     | *required* |
| `baseline`         | `str`                | The scenario subtracted.                                                                   | *required* |
| `quantities`       | `Sequence[quantity]` | The quantities; a Trajectory gives one scalar per position, each compared at its position. | *required* |
| `confidence_level` | `float`              | Strictly between 0 and 1.                                                                  | *required* |

Returns:

| Type                    | Description                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `list[ScenarioSummary]` | One summary per label of the quantities, in their order. With one run, the difference of that run and no interval. |

Raises:

| Type         | Description                                                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KeyError`   | If there is no scenario of either name.                                                                                                                                |
| `ValueError` | If the confidence level is not strictly between 0 and 1, a quantity does not fit the runs (as Predictive.quantities raises), or it is NaN in a run of either scenario. |

Source code in `commol/uncertainty/scenarios.py`

```python
def compare(
    self,
    name: str,
    baseline: str,
    quantities: "Sequence[QuantityBase]",
    confidence_level: float,
) -> list[ScenarioSummary]:
    """
    The weighted distribution of the difference of each quantity between
    two scenarios, taken run by run.

    The difference ``name − baseline`` is taken within each run, where
    both scenarios have the same draw and seed, and then reduced across
    the runs with their weights. So its interval is the interval of the
    paired differences, not the difference of two intervals.

    Parameters
    ----------
    name : str
        The scenario compared.
    baseline : str
        The scenario subtracted.
    quantities : Sequence[quantity]
        The quantities; a `Trajectory` gives one scalar per position, each
        compared at its position.
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    list[ScenarioSummary]
        One summary per label of the quantities, in their order. With one
        run, the difference of that run and no interval.

    Raises
    ------
    KeyError
        If there is no scenario of either name.
    ValueError
        If the confidence level is not strictly between 0 and 1, a
        quantity does not fit the runs (as `Predictive.quantities`
        raises), or it is NaN in a run of either scenario.
    """
    compared = self[name].quantities(quantities)
    subtracted = self[baseline].quantities(quantities)
    differences = [
        {label: value - other[label] for label, value in run.items()}
        for run, other in zip(compared, subtracted, strict=True)
    ]
    return _summaries(differences, self.weights, confidence_level)
```

## ScenarioSummary

Bases: `ResultModel`

The weighted distribution of one quantity across the runs of a scenario, or of its difference between two scenarios.

Attributes:

| Name               | Type    | Description                                   |
| ------------------ | ------- | --------------------------------------------- |
| `label`            | `str`   | The quantity, as expand_quantities labels it. |
| `confidence_level` | `float` | The level of the central interval.            |
| `mean`             | `float` | The weighted mean.                            |
| `median`           | `float` | The weighted median.                          |
| `lower`            | \`float | None\`                                        |
| `upper`            | \`float | None\`                                        |

Source code in `commol/uncertainty/scenarios.py`

```python
class ScenarioSummary(ResultModel):
    """
    The weighted distribution of one quantity across the runs of a scenario,
    or of its difference between two scenarios.

    Attributes
    ----------
    label : str
        The quantity, as `expand_quantities` labels it.
    confidence_level : float
        The level of the central interval.
    mean : float
        The weighted mean.
    median : float
        The weighted median.
    lower : float | None
        The quantile ``(1 − c)/2``; None when there is one run.
    upper : float | None
        The quantile ``(1 + c)/2``; None when there is one run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    confidence_level: float
    mean: float
    median: float
    lower: float | None
    upper: float | None
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

Source code in `commol/uncertainty/intervals.py`

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
    return quantiles(values, levels, weights)
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
member_statistics(per_member: Sequence[float] | Sequence[Sequence[float]] | NDArray[float64], confidence_level: float, weights: Sequence[float] | None = None) -> dict[str, float] | dict[str, list[float]] | dict[str, float | list[float]]
```

Spread of one quantity across the members of an ensemble.

Each member must already be reduced to the quantity of interest, so that the reported interval is a quantile of member values (:func:`weighted_quantiles`).

Parameters:

| Name               | Type              | Description                                                          | Default                                                                                                          |
| ------------------ | ----------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `per_member`       | \`Sequence[float] | Sequence\[Sequence[float]\]                                          | NDArray\`                                                                                                        |
| `confidence_level` | `float`           | Confidence level of the reported interval, strictly between 0 and 1. | *required*                                                                                                       |
| `weights`          | \`Sequence[float] | None\`                                                               | One weight per member, such as the weights of a :class:~commol.uncertainty.Predictive. None gives equal weights. |

Returns:

| Type               | Description                |
| ------------------ | -------------------------- |
| \`dict[str, float] | dict\[str, list[float]\]\` |

Raises:

| Type         | Description                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------------- |
| `ValueError` | If confidence_level is out of range, per_member is empty, or the weights do not match the members. |

Source code in `commol/uncertainty/intervals.py`

```python
def member_statistics(
    per_member: Sequence[float] | Sequence[Sequence[float]] | NDArray[np.float64],
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
        :class:`~commol.uncertainty.Predictive`. None gives equal weights.

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
    return spread(per_member, confidence_level, weights)
```
