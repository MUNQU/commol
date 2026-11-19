# Dependence API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.dependence`: the dependence of the values of each series, of any values and of the residuals of a predictive, with the references of its tests and its result. Guide: [Dependence](https://munqu.github.io/commol/latest/guide/uncertainty/dependence/index.md).

## series_dependence

```python
series_dependence(values: Sequence[float], observations: Sequence[ObservedDataPoint], max_lag: int | Literal['auto'], replicates: int, confidence_level: float, seed: int) -> SeriesDependence
```

The dependence of one value per observation, series by series.

The values of each label, in position order, are read on the grid of the smallest spacing of the series, and each lag over the pairs of values that exist at that distance (Parzen 1963; Stoffer and Toloi 1992). For each series: the autocorrelation up to `max_lag`; the robust portmanteau `Q̃ = Σ_k (Σ e_t e_{t−k})² / Σ e_t² e_{t−k}²` of the values centered per series (Dalla, Giraitis and Phillips 2022), which holds under heteroscedasticity, over the `h` lags of 4 pairs or more with a product other than 0, with its p-value against `χ²(h)`; and the integrated autocorrelation time `τ` by the initial monotone sequence of Geyer (1992), with its band under independence from `replicates` sets of the centered values with signs ±1 (a wild band). For the series together: their correlations at lag 0, and over the standardized series `z` the factors `Var(Σz) / ΣVar(z)` and `τ` of `Σz`, whose product `LRV(Σz) / ΣVar(z)` is the factor by which the dependence multiplies the information, with its p-value against the wild sets.

Parameters:

| Name               | Type                          | Description                                                                                                                            | Default                                                                                                               |
| ------------------ | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `values`           | `Sequence[float]`             | One finite value per observation, in the order of observations: a quantile residual of held-out data, an additive residual, any score. | *required*                                                                                                            |
| `observations`     | `Sequence[ObservedDataPoint]` | The observations the values belong to.                                                                                                 | *required*                                                                                                            |
| `max_lag`          | \`int                         | auto\`                                                                                                                                 | The largest lag, in steps of the grid of each series; "auto" is min(10, n / 5), at least 1, for a series of n values. |
| `replicates`       | `int`                         | The wild sets of the band and of the test of the factor, at least 1.                                                                   | *required*                                                                                                            |
| `confidence_level` | `float`                       | The level of the band of τ.                                                                                                            | *required*                                                                                                            |
| `seed`             | `int`                         | The seed of the signs, such as Calibrator.seed.                                                                                        | *required*                                                                                                            |

Returns:

| Type               | Description                    |
| ------------------ | ------------------------------ |
| `SeriesDependence` | The statistics, with one draw. |

Raises:

| Type         | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| `TypeError`  | If values holds lists: one set of values only.                                        |
| `ValueError` | If a setting is outside its range, or values is not one finite value per observation. |

References

Dalla, V., Giraitis, L., & Phillips, P. C. B. (2022). Robust tests for white noise and cross-correlation. Econometric Theory, 38(5), 913–941.

Geyer, C. J. (1992). Practical Markov chain Monte Carlo. Statistical Science, 7(4), 473–483.

Source code in `commol/dependence/analysis.py`

```python
def series_dependence(
    values: Sequence[float],
    observations: Sequence[ObservedDataPoint],
    max_lag: int | Literal["auto"],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> SeriesDependence:
    """
    The dependence of one value per observation, series by series.

    The values of each label, in position order, are read on the grid of the
    smallest spacing of the series, and each lag over the pairs of values
    that exist at that distance (Parzen 1963; Stoffer and Toloi 1992). For
    each series: the autocorrelation up to `max_lag`; the robust portmanteau
    ``Q̃ = Σ_k (Σ e_t e_{t−k})² / Σ e_t² e_{t−k}²`` of the values centered
    per series (Dalla, Giraitis and Phillips 2022), which holds under
    heteroscedasticity, over the ``h`` lags of 4 pairs or more with a
    product other than 0, with its p-value against ``χ²(h)``; and the
    integrated autocorrelation time ``τ`` by the initial monotone sequence
    of Geyer (1992), with its band under independence from `replicates`
    sets of the centered values with signs ±1 (a wild band). For the series
    together: their correlations at lag 0, and over the standardized series
    ``z`` the factors ``Var(Σz) / ΣVar(z)`` and ``τ`` of ``Σz``, whose product
    ``LRV(Σz) / ΣVar(z)`` is the factor by which the dependence multiplies the
    information, with its p-value against the wild sets.

    Parameters
    ----------
    values : Sequence[float]
        One finite value per observation, in the order of `observations`: a
        quantile residual of held-out data, an additive residual, any score.
    observations : Sequence[ObservedDataPoint]
        The observations the values belong to.
    max_lag : int | "auto"
        The largest lag, in steps of the grid of each series; ``"auto"`` is
        ``min(10, n / 5)``, at least 1, for a series of ``n`` values.
    replicates : int
        The wild sets of the band and of the test of the factor, at least 1.
    confidence_level : float
        The level of the band of ``τ``.
    seed : int
        The seed of the signs, such as ``Calibrator.seed``.

    Returns
    -------
    SeriesDependence
        The statistics, with one draw.

    Raises
    ------
    TypeError
        If `values` holds lists: one set of values only.
    ValueError
        If a setting is outside its range, or `values` is not one finite
        value per observation.

    References
    ----------
    Dalla, V., Giraitis, L., & Phillips, P. C. B. (2022). Robust tests for
    white noise and cross-correlation. Econometric Theory, 38(5), 913–941.

    Geyer, C. J. (1992). Practical Markov chain Monte Carlo. Statistical
    Science, 7(4), 473–483.
    """
    if any(isinstance(value, Sequence) for value in values):
        raise TypeError(
            "series_dependence reads one set of values, one per observation; for "
            "the residuals of several draws or randomizations, use "
            "residual_dependence"
        )
    raw = commol_rs.inference.series_dependence(
        engine_json(list(observations)),
        list(values),
        max_lag,
        replicates,
        confidence_level,
        seed,
    )
    settings = DependenceSettings(
        max_lag=max_lag,
        confidence_level=confidence_level,
        seed=seed,
        replicates=replicates,
    )
    return SeriesDependence._of(raw, settings)
```

## residual_dependence

```python
residual_dependence(predictive: Predictive, observations: Sequence[ObservedDataPoint], noise: _ObservationNoise, max_lag: int | Literal['auto'], randomizations: int, reference: AtParameters | Refitted, confidence_level: float) -> SeriesDependence
```

The dependence of the residuals of a predictive, series by series.

With `DistributionNoise`, the residuals of each draw are its randomized quantile residuals (Dunn and Smyth 1996), standard normal and independent for independent data under the right model; a count model randomizes them, so the statistics are averaged over `randomizations` randomizations, common to every draw, and the result does not depend on one of them. With `NoNoise`, for a loss or a stochastic engine, they are the additive residuals `y − ŷ`, `ŷ` the prediction the loss scores, and `randomizations` is 1. The statistics are those of `series_dependence`, averaged over the randomizations and, with their weights, over the draws (`τ` as the ratio of the mean long-run variance to the mean variance).

With one draw, the tests and the band of `τ` come from `reference`: `Refitted` holds at a fit, `AtParameters` at the parameters the data were drawn from. With many draws, the result gives the weighted statistics, and the median and the weighted share below the level of the p-values of the draws, with no test of its own: test a fit with `Calibrator.point_sample(fit)`. Every random number comes from the seed of the source of the predictive, so the same predictive gives the same result.

Parameters:

| Name               | Type                          | Description                                                                                          | Default                                                                                                               |
| ------------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `predictive`       | `Predictive`                  | The predictive of a Calibrator.                                                                      | *required*                                                                                                            |
| `observations`     | `Sequence[ObservedDataPoint]` | The observations whose residuals are read: the problem's, for Refitted, or held-out data.            | *required*                                                                                                            |
| `noise`            | \`NoNoise                     | DistributionNoise\`                                                                                  | DistributionNoise() for the quantile residuals of observation models, NoNoise() for the additive residuals of a loss. |
| `max_lag`          | \`int                         | auto\`                                                                                               | The largest lag, as in series_dependence.                                                                             |
| `randomizations`   | `int`                         | The randomizations of the quantile residuals, at least 1; 1 with NoNoise.                            | *required*                                                                                                            |
| `reference`        | \`AtParameters                | Refitted\`                                                                                           | The reference of the tests and the band of τ.                                                                         |
| `confidence_level` | `float`                       | The level of the band of τ; a p-value of a draw is below the level at or below 1 − confidence_level. | *required*                                                                                                            |

Returns:

| Type               | Description                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| `SeriesDependence` | The statistics and their tests, with failed and not_converged of the refits. |

Raises:

| Type                   | Description                                                                                                                                                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TypeError`            | If noise is ResidualNoise.                                                                                                                                                                                                                                                                        |
| `ValueError`           | If a setting is outside its range, the predictive is of a sensitivity analysis, quantile residuals have a loss or an observed value outside the support of its model, additive residuals have more than one randomization, or Refitted reads many draws or observations other than the problem's. |
| `RuntimeError`         | If every refit fails.                                                                                                                                                                                                                                                                             |
| `InfeasibleStateError` | If a draw leaves the values its model allows.                                                                                                                                                                                                                                                     |

References

Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals. Journal of Computational and Graphical Statistics, 5(3), 236–244.

Source code in `commol/dependence/analysis.py`

```python
def residual_dependence(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: _ObservationNoise,
    max_lag: int | Literal["auto"],
    randomizations: int,
    reference: AtParameters | Refitted,
    confidence_level: float,
) -> SeriesDependence:
    """
    The dependence of the residuals of a predictive, series by series.

    With `DistributionNoise`, the residuals of each draw are its randomized
    quantile residuals (Dunn and Smyth 1996), standard normal and
    independent for independent data under the right model; a count model
    randomizes them, so the statistics are averaged over `randomizations`
    randomizations, common to every draw, and the result does not depend on
    one of them. With `NoNoise`, for a loss or a stochastic engine, they are
    the additive residuals ``y − ŷ``, ``ŷ`` the prediction the loss scores,
    and `randomizations` is 1. The statistics are those of
    `series_dependence`, averaged over the randomizations and, with their
    weights, over the draws (``τ`` as the ratio of the mean long-run variance
    to the mean variance).

    With one draw, the tests and the band of ``τ`` come from `reference`:
    `Refitted` holds at a fit, `AtParameters` at the parameters the data
    were drawn from. With many draws, the result gives the weighted
    statistics, and the median and the weighted share below the level of
    the p-values of the draws, with no test of its own: test a fit with
    ``Calibrator.point_sample(fit)``. Every random number comes from the seed
    of the source of the predictive, so the same predictive gives the same
    result.

    Parameters
    ----------
    predictive : Predictive
        The predictive of a `Calibrator`.
    observations : Sequence[ObservedDataPoint]
        The observations whose residuals are read: the problem's, for
        `Refitted`, or held-out data.
    noise : NoNoise | DistributionNoise
        `DistributionNoise()` for the quantile residuals of observation
        models, `NoNoise()` for the additive residuals of a loss.
    max_lag : int | "auto"
        The largest lag, as in `series_dependence`.
    randomizations : int
        The randomizations of the quantile residuals, at least 1; 1 with
        `NoNoise`.
    reference : AtParameters | Refitted
        The reference of the tests and the band of ``τ``.
    confidence_level : float
        The level of the band of ``τ``; a p-value of a draw is below the
        level at or below ``1 − confidence_level``.

    Returns
    -------
    SeriesDependence
        The statistics and their tests, with ``failed`` and ``not_converged``
        of the refits.

    Raises
    ------
    TypeError
        If `noise` is `ResidualNoise`.
    ValueError
        If a setting is outside its range, the predictive is of a
        sensitivity analysis, quantile residuals have a loss or an observed
        value outside the support of its model, additive residuals have more
        than one randomization, or `Refitted` reads many draws or
        observations other than the problem's.
    RuntimeError
        If every refit fails.
    InfeasibleStateError
        If a draw leaves the values its model allows.

    References
    ----------
    Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals.
    Journal of Computational and Graphical Statistics, 5(3), 236–244.
    """
    raw = predictive.source._residual_dependence(
        predictive.sample,
        list(observations),
        noise,
        max_lag,
        randomizations,
        engine_json(reference),
        confidence_level,
    )
    for entry in raw.failed:
        logger.warning(entry)
    settings = DependenceSettings(
        max_lag=max_lag,
        confidence_level=confidence_level,
        seed=predictive.source.seed,
        noise=noise if isinstance(noise, (NoNoise, DistributionNoise)) else None,
        randomizations=randomizations,
        reference=reference,
    )
    return SeriesDependence._of(raw, settings)
```

## AtParameters

Bases: `RangeChecked`

A reference at the parameters of each draw, with no refit.

With `DistributionNoise`, `replicates` data sets drawn from the observation models at the predictions of the draw give the law of the integrated statistics: the p-values of `Q̃` and of the information factor, and the band of `τ`. No model runs again. With `NoNoise`, `Q̃` is read against `χ²(h)` and the band comes from `replicates` sets of the residuals with signs ±1. A fit leaves its residuals correlated with each other, the more so the more parameters it has, so with the data of the fit this reference rejects more often than its level (8 % at 5 % with 3 parameters and 24 to 26 % with 10, at 35 positions); `Refitted` holds there. It is the reference of held-out data.

Attributes:

| Name         | Type            | Description                                 |
| ------------ | --------------- | ------------------------------------------- |
| `kind`       | `at_parameters` | The reference, which the engine reads.      |
| `replicates` | `int`           | The data sets of the reference, at least 1. |

Source code in `commol/dependence/references.py`

```python
class AtParameters(RangeChecked):
    """
    A reference at the parameters of each draw, with no refit.

    With `DistributionNoise`, `replicates` data sets drawn from the
    observation models at the predictions of the draw give the law of the
    integrated statistics: the p-values of ``Q̃`` and of the information
    factor, and the band of ``τ``. No model runs again. With `NoNoise`,
    ``Q̃`` is read against ``χ²(h)`` and the band comes from `replicates`
    sets of the residuals with signs ±1. A fit leaves its residuals
    correlated with each other, the more so the more parameters it has, so
    with the data of the fit this reference rejects more often than its
    level (8 % at 5 % with 3 parameters and 24 to 26 % with 10, at 35
    positions); `Refitted` holds there. It is the reference of held-out data.

    Attributes
    ----------
    kind : "at_parameters"
        The reference, which the engine reads.
    replicates : int
        The data sets of the reference, at least 1.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["at_parameters"] = Field(
        default="at_parameters", description="The reference."
    )
    replicates: int = Field(description="The data sets of the reference.")
```

## Refitted

Bases: `RangeChecked`

A reference of data sets drawn at the fit and fitted again.

Each of `replicates` data sets is drawn from the observation models at the fit (`DistributionNoise`), or is the fit plus its residuals with signs ±1 (`NoNoise`), and is fitted again by the optimizer of the problem from `starts` starts. The statistics of the residuals of the refits are the law of the statistics of the fit, so the test holds at a fit whatever its number of parameters: the p-value of `Q̃` and of the information factor is `(1 + #{Q_b ≥ Q}) / (B + 1)`. It needs the sample of one point and the observations of the problem; the refits run in parallel and log a record of each.

Attributes:

| Name         | Type       | Description                                                                                |
| ------------ | ---------- | ------------------------------------------------------------------------------------------ |
| `kind`       | `refitted` | The reference, which the engine reads.                                                     |
| `replicates` | `int`      | The refits, at least 1.                                                                    |
| `starts`     | `int`      | Starts of each refit (default 1): the fit, then rows of a Latin hypercube over the bounds. |

Source code in `commol/dependence/references.py`

```python
class Refitted(RangeChecked):
    """
    A reference of data sets drawn at the fit and fitted again.

    Each of `replicates` data sets is drawn from the observation models at
    the fit (`DistributionNoise`), or is the fit plus its residuals with
    signs ±1 (`NoNoise`), and is fitted again by the optimizer of the
    problem from `starts` starts. The statistics of the residuals of the
    refits are the law of the statistics of the fit, so the test holds at a
    fit whatever its number of parameters: the p-value of ``Q̃`` and of the
    information factor is ``(1 + #{Q_b ≥ Q}) / (B + 1)``. It needs the sample
    of one point and the observations of the problem; the refits run in
    parallel and log a record of each.

    Attributes
    ----------
    kind : "refitted"
        The reference, which the engine reads.
    replicates : int
        The refits, at least 1.
    starts : int
        Starts of each refit (default 1): the fit, then rows of a Latin
        hypercube over the bounds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["refitted"] = Field(default="refitted", description="The reference.")
    replicates: int = Field(description="The refits.")
    starts: int = Field(default=1, description="Starts per refit.")
```

## SeriesDependence

Bases: `ResultModel`

Whether the values of each series are dependent, and how strongly.

Attributes:

| Name            | Type                      | Description                                                                                                                                  |
| --------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `series`        | `list[SeriesStatistics]`  | Each series, in order of first appearance of its label.                                                                                      |
| `correlations`  | `list[SeriesCorrelation]` | Each pair of series with statistics.                                                                                                         |
| `problem`       | `ProblemDependence`       | The series together.                                                                                                                         |
| `settings`      | `DependenceSettings`      | The settings of the analysis.                                                                                                                |
| `draws`         | `int`                     | The draws of the sample; 1 for values. With many, the statistics are the weighted means of the draws, and the result has no test of its own. |
| `failed`        | `list[str]`               | Each refit of a Refitted reference that failed (also logged).                                                                                |
| `not_converged` | `int`                     | The refits kept although their optimizer did not converge.                                                                                   |

Source code in `commol/dependence/results.py`

```python
class SeriesDependence(ResultModel):
    """
    Whether the values of each series are dependent, and how strongly.

    Attributes
    ----------
    series : list[SeriesStatistics]
        Each series, in order of first appearance of its label.
    correlations : list[SeriesCorrelation]
        Each pair of series with statistics.
    problem : ProblemDependence
        The series together.
    settings : DependenceSettings
        The settings of the analysis.
    draws : int
        The draws of the sample; 1 for values. With many, the statistics are
        the weighted means of the draws, and the result has no test of its
        own.
    failed : list[str]
        Each refit of a `Refitted` reference that failed (also logged).
    not_converged : int
        The refits kept although their optimizer did not converge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    series: list[SeriesStatistics]
    correlations: list[SeriesCorrelation]
    problem: ProblemDependence
    settings: DependenceSettings
    draws: int
    failed: list[str]
    not_converged: int

    @classmethod
    def _of(
        cls, raw: "SeriesDependenceProtocol", settings: DependenceSettings
    ) -> "SeriesDependence":
        return cls(
            series=[SeriesStatistics._of(series) for series in raw.series],
            correlations=[
                SeriesCorrelation(first=first, second=second, correlation=value)
                for first, second, value in raw.correlations
            ],
            problem=ProblemDependence(
                variance_factor=raw.variance_factor,
                tau=raw.tau,
                tau_reason=raw.tau_reason,
                information_factor=raw.information_factor,
                p_value=raw.p_value,
                draw_p_values=_draw_p_values(raw.draw_p_values),
            ),
            settings=settings,
            draws=raw.draws,
            failed=raw.failed,
            not_converged=raw.not_converged,
        )
```

## SeriesStatistics

Bases: `ResultModel`

The dependence of the values of one series.

The series reads its values in position order, on the grid of its smallest spacing; each lag pairs the values that exist at that distance.

Attributes:

| Name                 | Type                    | Description                                                                                                                   |
| -------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `label`              | `str`                   | The label of the series.                                                                                                      |
| `values`             | `int`                   | The values it reads.                                                                                                          |
| `reason`             | \`str                   | None\`                                                                                                                        |
| `max_lag`            | `int`                   | The largest lag, in steps of the grid.                                                                                        |
| `correlations`       | \`list\[float           | None\]\`                                                                                                                      |
| `pairs`              | `list[int]`             | The pairs of values each lag reads.                                                                                           |
| `few_pairs`          | `list[tuple[int, int]]` | Each lag of fewer than 4 pairs, as (lag, pairs): it keeps its correlation and is left out of Q̃ and of its degrees of freedom. |
| `portmanteau`        | \`float                 | None\`                                                                                                                        |
| `portmanteau_reason` | \`str                   | None\`                                                                                                                        |
| `degrees_of_freedom` | `int`                   | The lags in the sum of Q̃: those of 4 pairs or more with a product other than 0.                                               |
| `p_value`            | \`float                 | None\`                                                                                                                        |
| `draw_p_values`      | \`DrawPValues           | None\`                                                                                                                        |
| `tau`                | \`float                 | None\`                                                                                                                        |
| `tau_reason`         | \`str                   | None\`                                                                                                                        |
| `tau_band`           | \`tuple[float, float]   | None\`                                                                                                                        |
| `cut_lag`            | \`int                   | None\`                                                                                                                        |

Source code in `commol/dependence/results.py`

```python
class SeriesStatistics(ResultModel):
    """
    The dependence of the values of one series.

    The series reads its values in position order, on the
    grid of its smallest spacing; each lag pairs the values that exist at
    that distance.

    Attributes
    ----------
    label : str
        The label of the series.
    values : int
        The values it reads.
    reason : str | None
        Why it has no statistics: two observations at one position, a
        position off its grid, or a grid of more than 10 points per value.
    max_lag : int
        The largest lag, in steps of the grid.
    correlations : list[float | None]
        The autocorrelation at lags ``1 … max_lag``, each over its pairs;
        None at a lag with no pair.
    pairs : list[int]
        The pairs of values each lag reads.
    few_pairs : list[tuple[int, int]]
        Each lag of fewer than 4 pairs, as ``(lag, pairs)``: it keeps its
        correlation and is left out of ``Q̃`` and of its degrees of freedom.
    portmanteau : float | None
        The robust portmanteau ``Q̃`` (Dalla, Giraitis and Phillips 2022).
    portmanteau_reason : str | None
        Why there is no ``Q̃``: the series has no statistics, every lag
        reads fewer than 4 pairs, or no lag of 4 pairs or more has a
        product other than 0, as of constant values.
    degrees_of_freedom : int
        The lags in the sum of ``Q̃``: those of 4 pairs or more with a
        product other than 0.
    p_value : float | None
        The p-value of ``Q̃`` under the reference, with one draw.
    draw_p_values : DrawPValues | None
        The spread of the p-values of the draws, with many.
    tau : float | None
        The integrated autocorrelation time, by the initial monotone
        sequence of Geyer (1992).
    tau_reason : str | None
        Why ``τ`` is undefined: fewer than 4 values, constant values, or an
        estimate not above 0, which reads as below its band.
    tau_band : tuple[float, float] | None
        The central interval of ``τ`` under independence at the confidence
        level, with one draw; ``-inf`` where the replicates have no ``τ``,
        and None when none has one.
    cut_lag : int | None
        The largest lag in the sum of ``τ``; over several sets, the largest
        of theirs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    values: int
    reason: str | None
    max_lag: int
    correlations: list[float | None]
    pairs: list[int]
    few_pairs: list[tuple[int, int]]
    portmanteau: float | None
    portmanteau_reason: str | None
    degrees_of_freedom: int
    p_value: float | None
    draw_p_values: DrawPValues | None
    tau: float | None
    tau_reason: str | None
    tau_band: tuple[float, float] | None
    cut_lag: int | None

    @classmethod
    def _of(cls, raw: "SeriesStatisticsProtocol") -> "SeriesStatistics":
        return cls(
            label=raw.label,
            values=raw.values,
            reason=raw.reason,
            max_lag=raw.max_lag,
            correlations=raw.correlations,
            pairs=raw.pairs,
            few_pairs=raw.few_pairs,
            portmanteau=raw.portmanteau,
            portmanteau_reason=raw.portmanteau_reason,
            degrees_of_freedom=raw.degrees_of_freedom,
            p_value=raw.p_value,
            draw_p_values=_draw_p_values(raw.draw_p_values),
            tau=raw.tau,
            tau_reason=raw.tau_reason,
            tau_band=raw.tau_band,
            cut_lag=raw.cut_lag,
        )
```

## SeriesCorrelation

Bases: `ResultModel`

The correlation at lag 0 of two series, over the positions they share.

Attributes:

| Name          | Type    | Description                     |
| ------------- | ------- | ------------------------------- |
| `first`       | `str`   | The label of the first series.  |
| `second`      | `str`   | The label of the second series. |
| `correlation` | \`float | None\`                          |

Source code in `commol/dependence/results.py`

```python
class SeriesCorrelation(ResultModel):
    """
    The correlation at lag 0 of two series, over the positions they share.

    Attributes
    ----------
    first : str
        The label of the first series.
    second : str
        The label of the second series.
    correlation : float | None
        None with fewer than two shared positions or a constant series.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    first: str
    second: str
    correlation: float | None
```

## ProblemDependence

Bases: `ResultModel`

The dependence of the series together, over the standardized series `z`: how many times the observations count their information.

Attributes:

| Name                 | Type          | Description |
| -------------------- | ------------- | ----------- |
| `variance_factor`    | \`float       | None\`      |
| `tau`                | \`float       | None\`      |
| `tau_reason`         | \`str         | None\`      |
| `information_factor` | \`float       | None\`      |
| `p_value`            | \`float       | None\`      |
| `draw_p_values`      | \`DrawPValues | None\`      |

Source code in `commol/dependence/results.py`

```python
class ProblemDependence(ResultModel):
    """
    The dependence of the series together, over the standardized series
    ``z``: how many times the observations count their information.

    Attributes
    ----------
    variance_factor : float | None
        ``Var(Σz) / ΣVar(z)``, ``Σz`` summed at each position of any
        series: the dependence across series at one position.
    tau : float | None
        ``τ`` of ``Σz``: the dependence of the sum along the positions.
    tau_reason : str | None
        Why ``τ`` of ``Σz`` is undefined.
    information_factor : float | None
        Their product, ``LRV(Σz) / ΣVar(z)``: 1 for independent
        observations, 2 when each counts its information twice.
    p_value : float | None
        The p-value of the information factor under the reference, with one
        draw.
    draw_p_values : DrawPValues | None
        The spread of the p-values of the draws, with many.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    variance_factor: float | None
    tau: float | None
    tau_reason: str | None
    information_factor: float | None
    p_value: float | None
    draw_p_values: DrawPValues | None
```

## DrawPValues

Bases: `ResultModel`

The spread of the p-values of the draws of a sample, each against its own reference: no test of the sample.

Attributes:

| Name          | Type    | Description                                          |
| ------------- | ------- | ---------------------------------------------------- |
| `median`      | `float` | The weighted median.                                 |
| `share_below` | `float` | The weighted share at or below 1 − confidence_level. |

Source code in `commol/dependence/results.py`

```python
class DrawPValues(ResultModel):
    """
    The spread of the p-values of the draws of a sample, each against its
    own reference: no test of the sample.

    Attributes
    ----------
    median : float
        The weighted median.
    share_below : float
        The weighted share at or below ``1 − confidence_level``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    median: float = Field(description="The weighted median.")
    share_below: float = Field(description="The weighted share below the level.")
```

## DependenceSettings

Bases: `ResultModel`

The settings of a dependence analysis.

Attributes:

| Name               | Type           | Description                     |
| ------------------ | -------------- | ------------------------------- |
| `max_lag`          | \`int          | auto\`                          |
| `confidence_level` | `float`        | The level of the band of τ.     |
| `seed`             | `int`          | The seed of its random numbers. |
| `replicates`       | \`int          | None\`                          |
| `noise`            | \`NoNoise      | DistributionNoise               |
| `randomizations`   | \`int          | None\`                          |
| `reference`        | \`AtParameters | Refitted                        |

Source code in `commol/dependence/results.py`

```python
class DependenceSettings(ResultModel):
    """
    The settings of a dependence analysis.

    Attributes
    ----------
    max_lag : int | "auto"
        The largest lag of each series.
    confidence_level : float
        The level of the band of ``τ``.
    seed : int
        The seed of its random numbers.
    replicates : int | None
        The wild sets of `series_dependence`.
    noise : NoNoise | DistributionNoise | None
        The residuals of `residual_dependence`.
    randomizations : int | None
        The randomizations of `residual_dependence`.
    reference : AtParameters | Refitted | None
        The reference of `residual_dependence`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_lag: int | Literal["auto"]
    confidence_level: float
    seed: int
    replicates: int | None = None
    noise: NoNoise | DistributionNoise | None = Field(
        default=None, discriminator="kind"
    )
    randomizations: int | None = None
    reference: AtParameters | Refitted | None = Field(
        default=None, discriminator="kind"
    )
```
