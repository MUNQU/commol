# Conformal API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.conformal`: the conformal prediction methods and their result. Guide: [Conformal Prediction Intervals](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/index.md).

## ConformalMethod

Bases: `UncertaintyMethod[ConformalResult]`

What every conformal method takes: a score, the pooling of its quantiles, and a confidence level, none of them with a default.

Attributes:

| Name               | Type                             | Description                                                                                                                                    |
| ------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `score`            | `{absolute, normalized, spread}` | The nonconformity score. "normalized" needs by="all": per series its factor is one constant, so its intervals would equal those of "absolute". |
| `by`               | `{series, all}`                  | Whether each series gets its own quantile, or all series share one.                                                                            |
| `confidence_level` | `float`                          | The level of the intervals, strictly between 0 and 1.                                                                                          |

Source code in `commol/conformal/methods.py`

```python
class ConformalMethod(UncertaintyMethod[ConformalResult]):
    """
    What every conformal method takes: a score, the pooling of its
    quantiles, and a confidence level, none of them with a default.

    Attributes
    ----------
    score : {"absolute", "normalized", "spread"}
        The nonconformity score. ``"normalized"`` needs ``by="all"``: per
        series its factor is one constant, so its intervals would equal those
        of ``"absolute"``.
    by : {"series", "all"}
        Whether each series gets its own quantile, or all series share one.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.
    """

    score: Literal["absolute", "normalized", "spread"] = Field(
        description="The nonconformity score."
    )
    by: Literal["series", "all"] = Field(
        description="Which scores each quantile reads."
    )
    confidence_level: float = Field(description="The level of the intervals.")

    @model_validator(mode="before")
    @classmethod
    def _check_score_and_by(cls, data: JsonValue) -> JsonValue:
        """Check, in Rust, that `score` and `by` are given and fit together."""
        if isinstance(data, Mapping):
            given = {
                key: data[key] for key in ("score", "by") if data.get(key) is not None
            }
            commol_rs.inference.check_conformal_settings(engine_json(given))
        return data
```

## SplitConformal

Bases: `ConformalMethod`, `RangeChecked`

Split conformal prediction.

It fits the problem's observations and scores `calibration_data`, observations held out of the fit. When the scores of the held-out and of new observations are exchangeable, each interval covers a new observation with probability at least `confidence_level`.

It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

Attributes:

| Name               | Type                      | Description                                                   |
| ------------------ | ------------------------- | ------------------------------------------------------------- |
| `calibration_data` | `list[ObservedDataPoint]` | Held-out observations; each of their series gets intervals.   |
| `positions`        | `list[float]`             | Where the intervals are, for each series of calibration_data. |
| `sample`           | \`ParameterSample         | None\`                                                        |
| `starts`           | `int`                     | Starts of the fit (default 1).                                |

Source code in `commol/conformal/methods.py`

```python
class SplitConformal(ConformalMethod, RangeChecked):
    """
    Split conformal prediction.

    It fits the problem's observations and scores `calibration_data`,
    observations held out of the fit. When the scores of the held-out and of
    new observations are exchangeable, each interval covers a new
    observation with probability at least `confidence_level`.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

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

    _runner = staticmethod(run_split_conformal)

    _engine_handles: ClassVar[tuple[str, ...]] = ("sample",)

    calibration_data: list[ObservedDataPoint] = Field(
        min_length=1, description="Held-out observations."
    )
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    sample: ParameterSample | None = Field(
        default=None, description="The sample whose predictive predicts."
    )
    starts: int = Field(default=1, description="Starts of the fit.")

    @model_validator(mode="after")
    def _check_spread(self) -> Self:
        """Reject ``score="spread"`` without a sample."""
        _no_spread_without_sample(self)
        return self
```

## RollingConformal

Bases: `ConformalMethod`, `RangeChecked`

Rolling-origin conformal prediction: intervals ahead of the data, one quantile per horizon.

Each origin fits the observations at or before it and scores the later ones. Horizon `k` is the `k`-th observation of a series after the origin: a count, not a distance on the axis.

It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

Attributes:

| Name        | Type          | Description                                                                  |
| ----------- | ------------- | ---------------------------------------------------------------------------- |
| `origins`   | `list[float]` | The positions of the origins.                                                |
| `positions` | `list[float]` | Where the intervals are, all after the data; the k-th position is horizon k. |
| `starts`    | `int`         | Starts of each fit (default 1).                                              |

Source code in `commol/conformal/methods.py`

```python
class RollingConformal(ConformalMethod, RangeChecked):
    """
    Rolling-origin conformal prediction: intervals ahead of the data, one
    quantile per horizon.

    Each origin fits the observations at or before it and scores the later
    ones. Horizon ``k`` is the ``k``-th observation of a series after the
    origin: a count, not a distance on the axis.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

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

    _runner = staticmethod(run_rolling_conformal)

    origins: list[float] = Field(min_length=1, description="The origins.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, description="Starts of each fit.")

    @model_validator(mode="after")
    def _check_spread(self) -> Self:
        """Reject ``score="spread"``: one fit per origin has no spread."""
        if self.score == "spread":
            raise ValueError(
                "RollingConformal fits once per origin, so its predictions have no "
                'spread for score="spread"; use "absolute" or "normalized"'
            )
        return self
```

## JackknifePlus

Bases: `ConformalMethod`, `RangeChecked`

Jackknife+ (Barber et al. 2021) with blocks: one refit per block left out, and intervals from the left-out residuals and predictions.

With exchangeable blocks it covers at least `1 − 2α` when each block holds one observation, and the CV+ bound with `K` blocks otherwise.

It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

Attributes:

| Name           | Type          | Description                                              |
| -------------- | ------------- | -------------------------------------------------------- |
| `block_length` | \`int         | auto\`                                                   |
| `positions`    | `list[float]` | Where the intervals are, for each series of the problem. |
| `starts`       | `int`         | Starts of the fit and of each refit (default 1).         |

Source code in `commol/conformal/methods.py`

```python
class JackknifePlus(ConformalMethod, RangeChecked):
    """
    Jackknife+ (Barber et al. 2021) with blocks: one refit per block left
    out, and intervals from the left-out residuals and predictions.

    With exchangeable blocks it covers at least ``1 − 2α`` when each block
    holds one observation, and the CV+ bound with ``K`` blocks otherwise.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

    Attributes
    ----------
    block_length : int | "auto"
        Positions per block; it must leave at least two blocks.
    positions : list[float]
        Where the intervals are, for each series of the problem.
    starts : int
        Starts of the fit and of each refit (default 1).
    """

    _runner = staticmethod(run_jackknife_plus)

    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, description="Starts per fit.")
```

## EnbPI

Bases: `ConformalMethod`, `RangeChecked`

Ensemble batch prediction intervals (Xu and Xie 2021).

Each replicate draws blocks with replacement and fits. The replicates that left out the block of an observation give its leave-one-out prediction and score; their mean over every observation is the point prediction.

It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

Attributes:

| Name           | Type          | Description                                              |
| -------------- | ------------- | -------------------------------------------------------- |
| `replicates`   | `int`         | Number of replicates, at least 2.                        |
| `block_length` | \`int         | auto\`                                                   |
| `positions`    | `list[float]` | Where the intervals are, for each series of the problem. |
| `starts`       | `int`         | Starts of the fit and of each replicate (default 1).     |

Source code in `commol/conformal/methods.py`

```python
class EnbPI(ConformalMethod, RangeChecked):
    """
    Ensemble batch prediction intervals (Xu and Xie 2021).

    Each replicate draws blocks with replacement and fits. The replicates
    that left out the block of an observation give its leave-one-out
    prediction and score; their mean over every observation is the point
    prediction.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

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

    _runner = staticmethod(run_enbpi)

    replicates: int = Field(description="Number of replicates.")
    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, description="Starts per fit.")
```

## AdaptiveConformal

Bases: `ConformalMethod`, `RangeChecked`

Adaptive conformal inference (Gibbs and Candès 2021).

It fits the problem's observations and walks `stream` in order of position. The interval of each point is the quantile, at the working level `1 − α_t`, of the scores of the earlier points of the stream; then the working level moves by `α_{t+1} = α_t + γ (α − err_t)`. The long-run share of misses tends to `α`, with no assumption on the data.

It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

Attributes:

| Name     | Type                      | Description                                  |
| -------- | ------------------------- | -------------------------------------------- |
| `step`   | `float`                   | The step γ of the working level, positive.   |
| `stream` | `list[ObservedDataPoint]` | The observations that arrive after the data. |
| `sample` | \`ParameterSample         | None\`                                       |
| `starts` | `int`                     | Starts of the fit (default 1).               |

Source code in `commol/conformal/methods.py`

```python
class AdaptiveConformal(ConformalMethod, RangeChecked):
    """
    Adaptive conformal inference (Gibbs and Candès 2021).

    It fits the problem's observations and walks `stream` in order of
    position. The interval of each point is the quantile, at the working
    level ``1 − α_t``, of the scores of the earlier points of the stream;
    then the working level moves by ``α_{t+1} = α_t + γ (α − err_t)``. The
    long-run share of misses tends to ``α``, with no assumption on the data.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

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

    _runner = staticmethod(run_adaptive_conformal)

    step: float = Field(description="The step of the working level.")
    stream: list[ObservedDataPoint] = Field(
        min_length=1, description="The observations that arrive."
    )
    _engine_handles: ClassVar[tuple[str, ...]] = ("sample",)

    sample: ParameterSample | None = Field(
        default=None, description="The sample whose predictive predicts."
    )
    starts: int = Field(default=1, description="Starts of the fit.")

    @model_validator(mode="after")
    def _check_spread(self) -> Self:
        """Reject ``score="spread"`` without a sample."""
        _no_spread_without_sample(self)
        return self
```

## ConformalResult

Bases: `ResultModel`

The result of a conformal method.

Attributes:

| Name                     | Type                     | Description                                                                                                                                                                                                                                                                 |
| ------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `intervals`              | `PredictionIntervals`    | The intervals at the positions the method was given.                                                                                                                                                                                                                        |
| `scores`                 | `dict[str, list[float]]` | The scores of each series, by label, in the order the method took them.                                                                                                                                                                                                     |
| `guarantee`              | `str`                    | The coverage the method guarantees, and what the guarantee assumes.                                                                                                                                                                                                         |
| `failed`                 | `list[str]`              | Each refit that failed, which gives no score: "Origin <index> failed: <reason>" for RollingConformal, "Block <index> failed: <reason>" for JackknifePlus, "Replicate <index> failed: <reason>" for EnbPI; also logged as a warning. Empty for a method that does not refit. |
| `not_converged`          | `int`                    | Refits kept although their optimizer did not converge; 0 for a method that does not refit.                                                                                                                                                                                  |
| `block_length`           | \`int                    | None\`                                                                                                                                                                                                                                                                      |
| `block_length_estimates` | \`dict[str, float]       | None\`                                                                                                                                                                                                                                                                      |

Source code in `commol/conformal/results.py`

```python
class ConformalResult(ResultModel):
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
    failed : list[str]
        Each refit that failed, which gives no score: ``"Origin <index>
        failed: <reason>"`` for `RollingConformal`, ``"Block <index> failed:
        <reason>"`` for `JackknifePlus`, ``"Replicate <index> failed:
        <reason>"`` for `EnbPI`; also logged as a warning. Empty for a method
        that does not refit.
    not_converged : int
        Refits kept although their optimizer did not converge; 0 for a
        method that does not refit.
    block_length : int | None
        With `JackknifePlus` and `EnbPI`, the positions per block they ran
        with; None for a method without blocks.
    block_length_estimates : dict[str, float] | None
        With ``block_length="auto"``, the estimate of each series, by label;
        otherwise None.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intervals: PredictionIntervals
    scores: dict[str, list[float]]
    guarantee: str
    failed: list[str] = Field(description="Each refit that failed.")
    not_converged: int = Field(description="Refits that did not converge.")
    block_length: int | None = Field(description="The block length used.")
    block_length_estimates: dict[str, float] | None = Field(
        description="The estimate of each series with the automatic length."
    )
```

## PredictionIntervals

Bases: `ResultModel`

Prediction intervals for observations: where new observations of each series fall, at a confidence level.

Attributes:

| Name               | Type                         | Description                                                                                                                                    |
| ------------------ | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `confidence_level` | `float`                      | The level of the intervals, 1 − α.                                                                                                             |
| `series`           | `dict[str, SeriesIntervals]` | The intervals of each series, by label.                                                                                                        |
| `unbounded`        | `dict[str, list[float]]`     | The positions of each series whose interval is unbounded, because its quantile had fewer than 1/α − 1 scores. A series with none is not a key. |

Source code in `commol/conformal/results.py`

```python
class PredictionIntervals(ResultModel):
    """
    Prediction intervals for observations: where new observations of each
    series fall, at a confidence level.

    Attributes
    ----------
    confidence_level : float
        The level of the intervals, ``1 − α``.
    series : dict[str, SeriesIntervals]
        The intervals of each series, by label.
    unbounded : dict[str, list[float]]
        The positions of each series whose interval is unbounded, because
        its quantile had fewer than ``1/α − 1`` scores. A series with none
        is not a key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_level: float
    series: dict[str, SeriesIntervals]
    unbounded: dict[str, list[float]]
```

## SeriesIntervals

Bases: `ResultModel`

The prediction intervals of one series.

Attributes:

| Name         | Type          | Description                                          |
| ------------ | ------------- | ---------------------------------------------------- |
| `positions`  | `list[float]` | Where the intervals are.                             |
| `prediction` | `list[float]` | The point prediction at each position.               |
| `lower`      | `list[float]` | The lower end of each interval; -inf when unbounded. |
| `upper`      | `list[float]` | The upper end of each interval; inf when unbounded.  |

Source code in `commol/conformal/results.py`

```python
class SeriesIntervals(ResultModel):
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

    model_config = ConfigDict(extra="forbid", frozen=True)

    positions: list[float]
    prediction: list[float]
    lower: list[float]
    upper: list[float]
```
