# Reports API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Plain-dictionary reports, each a method of the object whose data it describes. Guide: [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md).

## applied_parameters

```python
applied_parameters(values: CalibrationResult | Mapping[str, float], problem: CalibrationProblem | None = None, sample: SummarizedSample | None = None, confidence_level: float | None = None) -> AppliedParametersReport
```

Report every parameter of the model, and the values of its kind.

Call it on the model **after** `apply_calibration_parameters`. Each entry carries the value the model holds and whether calibration set it, so entries the calibration left alone appear alongside the fitted ones. The structure of the model adds the sections of its kind (`ModelStructure.report_sections`); an aggregate model adds `initial_amounts` and `category_splits`.

Parameters that are not a single number, such as a formula or a series, are omitted: they are inputs rather than fitted values. `factor` and `noise` parameters have no place in the model, so they are reported from `problem` when it is given.

`initial_amounts` gives each bin and each overridden compartment of an aggregate model its `amount` the engine builds and the `unit` of its bin, so bins of different units are reported side by side; no row is a share of a total. A bin left uncalibrated keeps a `None` amount as its "needs calibration" marker even when overrides touch some of its compartments. `category_splits` gives each category its `fraction` of the group it splits, and that `group`: the categories a conditional stratification is conditioned on, empty when it splits every bin. A stratification whose fractions differ by the category of an earlier one gives a row per split, under the key `t1 of g1`, with that category in its `group`.

With a `sample` each entry gains a `spread` holding the weighted statistics of its draws (`ParameterSample.summary`). Every entry is in the unit it was calibrated in: a parameter in its own, an amount in the unit of its bin, and a category split as a fraction of its group.

Parameters:

| Name               | Type                 | Description | Default                                                                                     |
| ------------------ | -------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| `values`           | \`CalibrationResult  | Mapping\`   | The calibration result, or a mapping of parameter id to value such as one draw of a sample. |
| `problem`          | \`CalibrationProblem | None\`      | The problem the values came from, used to report factor and noise parameters.               |
| `sample`           | \`ParameterSample    | None\`      | Draws of the parameters, such as the sample of an uncertainty method.                       |
| `confidence_level` | \`float              | None\`      | Confidence level of the intervals; required with a sample.                                  |

Returns:

| Type                      | Description                                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AppliedParametersReport` | Key parameters and the sections of the kind (initial_amounts and category_splits for an aggregate model), preceded by confidence_level when there is one. |

Raises:

| Type         | Description                                      |
| ------------ | ------------------------------------------------ |
| `ValueError` | If a sample is given without a confidence level. |

Source code in `commol/_model/model.py`

```python
def applied_parameters(
    self,
    values: CalibrationResult | Mapping[str, float],
    problem: CalibrationProblem | None = None,
    sample: "SummarizedSample | None" = None,
    confidence_level: float | None = None,
) -> "AppliedParametersReport":
    """
    Report every parameter of the model, and the values of its kind.

    Call it on the model **after** `apply_calibration_parameters`. Each
    entry carries the value the model holds and whether calibration set
    it, so entries the calibration left alone appear alongside the fitted
    ones. The structure of the model adds the sections of its kind
    (`ModelStructure.report_sections`); an aggregate model adds
    ``initial_amounts`` and ``category_splits``.

    Parameters that are not a single number, such as a formula or a
    series, are omitted: they are inputs rather than fitted values.
    ``factor`` and ``noise`` parameters have no place in the model, so
    they are reported from `problem` when it is given.

    ``initial_amounts`` gives each bin and each overridden compartment of
    an aggregate model its ``amount`` the engine builds and the ``unit``
    of its bin, so bins of different units are reported side by side; no
    row is a share of a total. A bin left uncalibrated keeps a ``None``
    amount as its "needs calibration" marker even when overrides touch
    some of its compartments. ``category_splits`` gives each category its
    ``fraction`` of the group it splits, and that ``group``: the
    categories a conditional stratification is conditioned on, empty when
    it splits every bin. A stratification whose fractions differ by the
    category of an earlier one gives a row per split, under the key
    ``t1 of g1``, with that category in its ``group``.

    With a `sample` each entry gains a ``spread`` holding the weighted
    statistics of its draws (`ParameterSample.summary`). Every entry is in
    the unit it was calibrated in: a parameter in its own, an amount in
    the unit of its bin, and a category split as a fraction of its group.

    Parameters
    ----------
    values : CalibrationResult | Mapping
        The calibration result, or a mapping of parameter id to value such
        as one draw of a sample.
    problem : CalibrationProblem | None, optional
        The problem the values came from, used to report factor and noise
        parameters.
    sample : ParameterSample | None, optional
        Draws of the parameters, such as the sample of an uncertainty
        method.
    confidence_level : float | None, optional
        Confidence level of the intervals; required with a sample.

    Returns
    -------
    AppliedParametersReport
        Key ``parameters`` and the sections of the kind
        (``initial_amounts`` and ``category_splits`` for an aggregate
        model), preceded by ``confidence_level`` when there is one.

    Raises
    ------
    ValueError
        If a sample is given without a confidence level.
    """
    return applied_parameters_report(
        self, values, problem, sample, confidence_level
    )
```

## report

```python
report(positions: Sequence[float], window: float, windows: int, observed_data: Iterable[ObservedDataPoint] = (), accumulators: Iterable[str] = (), prediction_intervals: PredictionIntervalsData | None = None) -> SimulationReport
```

Report the observed data, compartment series and accumulators of the run.

The blocks share one window axis, so the observed values, the windowed accumulator increments and the sampled series can be read against each other. Add whatever envelope you need around them, such as a run label, axis labels or summary figures in the terms of your domain.

Parameters:

| Name                   | Type                          | Description                                                                                              | Default                                                                                       |
| ---------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `positions`            | `Sequence[float]`             | Positions the series and cumulative accumulators are sampled at; each must be a position of the results. | *required*                                                                                    |
| `window`               | `float`                       | Length of one window on the axis.                                                                        | *required*                                                                                    |
| `windows`              | `int`                         | Number of windows on the observed and windowed axes.                                                     | *required*                                                                                    |
| `observed_data`        | `Iterable[ObservedDataPoint]` | The calibration targets to report alongside the run.                                                     | `()`                                                                                          |
| `accumulators`         | `Iterable[str]`               | Accumulator ids to report; their outputs come from model.                                                | `()`                                                                                          |
| `prediction_intervals` | \`PredictionIntervalsData     | None\`                                                                                                   | Prediction intervals of a conformal method (ConformalResult.intervals), reported as they are. |

Returns:

| Type               | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `SimulationReport` | Keys observed, series and accumulators, then prediction_intervals when they are given. |

Raises:

| Type         | Description                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If the results have no value at a position the report reads, or they have no model and an accumulator is reported. |
| `KeyError`   | If an accumulator is not a bin or accumulator of the model.                                                        |

Source code in `commol/_simulation/results.py`

```python
def report(
    self,
    positions: Sequence[float],
    window: float,
    windows: int,
    observed_data: "Iterable[ObservedDataPoint]" = (),
    accumulators: Iterable[str] = (),
    prediction_intervals: "PredictionIntervalsData | None" = None,
) -> "SimulationReport":
    """
    Report the observed data, compartment series and accumulators of the run.

    The blocks share one window axis, so the observed values, the windowed
    accumulator increments and the sampled series can be read against
    each other. Add whatever envelope you need around them, such as a run
    label, axis labels or summary figures in the terms of your domain.

    Parameters
    ----------
    positions : Sequence[float]
        Positions the series and cumulative accumulators are sampled at;
        each must be a position of the results.
    window : float
        Length of one window on the axis.
    windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the run.
    accumulators : Iterable[str], optional
        Accumulator ids to report; their outputs come from `model`.
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
        If the results have no value at a position the report reads, or
        they have no `model` and an accumulator is reported.
    KeyError
        If an accumulator is not a bin or accumulator of the model.
    """
    from commol._simulation.reports import _ReportRuns, _simulation_report

    return _simulation_report(
        _ReportRuns(model=self.model, runs=[self]),
        positions,
        window,
        windows,
        observed_data,
        accumulators,
        prediction_intervals,
    )
```

## report

```python
report(confidence_level: float, positions: Sequence[float], window: float, windows: int, observed_data: Iterable[ObservedDataPoint] = (), accumulators: Iterable[str] = (), prediction_intervals: PredictionIntervalsData | None = None) -> SimulationReport
```

Report the replicates as `SimulationResults.report` reports one run.

Every number becomes a `{mean, median, ci_lower, ci_upper, min, max}` block, reduced replicate by replicate; the layout is the one of the report of a run.

Parameters:

| Name                   | Type                          | Description                                                                                                 | Default                                                                                       |
| ---------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `confidence_level`     | `float`                       | Confidence level of the reported intervals, strictly between 0 and 1.                                       | *required*                                                                                    |
| `positions`            | `Sequence[float]`             | Positions the series and cumulative accumulators are sampled at; each must be a position of the replicates. | *required*                                                                                    |
| `window`               | `float`                       | Length of one window on the axis.                                                                           | *required*                                                                                    |
| `windows`              | `int`                         | Number of windows on the observed and windowed axes.                                                        | *required*                                                                                    |
| `observed_data`        | `Iterable[ObservedDataPoint]` | The calibration targets to report alongside the replicates.                                                 | `()`                                                                                          |
| `accumulators`         | `Iterable[str]`               | Accumulator ids to report; their outputs come from model.                                                   | `()`                                                                                          |
| `prediction_intervals` | \`PredictionIntervalsData     | None\`                                                                                                      | Prediction intervals of a conformal method (ConformalResult.intervals), reported as they are. |

Returns:

| Type               | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `SimulationReport` | Keys observed, series and accumulators, then prediction_intervals when they are given. |

Raises:

| Type         | Description                                                                                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If there are no replicates, the confidence level is out of range, the replicates have no value at a position the report reads, or they have no model and an accumulator is reported. |
| `KeyError`   | If an accumulator is not a bin or accumulator of the model.                                                                                                                          |

Source code in `commol/_simulation/results.py`

```python
def report(
    self,
    confidence_level: float,
    positions: Sequence[float],
    window: float,
    windows: int,
    observed_data: "Iterable[ObservedDataPoint]" = (),
    accumulators: Iterable[str] = (),
    prediction_intervals: "PredictionIntervalsData | None" = None,
) -> "SimulationReport":
    """
    Report the replicates as `SimulationResults.report` reports one run.

    Every number becomes a ``{mean, median, ci_lower, ci_upper, min,
    max}`` block, reduced replicate by replicate; the layout is the one of
    the report of a run.

    Parameters
    ----------
    confidence_level : float
        Confidence level of the reported intervals, strictly between 0
        and 1.
    positions : Sequence[float]
        Positions the series and cumulative accumulators are sampled at;
        each must be a position of the replicates.
    window : float
        Length of one window on the axis.
    windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the replicates.
    accumulators : Iterable[str], optional
        Accumulator ids to report; their outputs come from `model`.
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
        If there are no replicates, the confidence level is out of range,
        the replicates have no value at a position the report reads, or
        they have no `model` and an accumulator is reported.
    KeyError
        If an accumulator is not a bin or accumulator of the model.
    """
    from commol._simulation.reports import _ReportRuns, _simulation_report

    if not self:
        raise ValueError("There are no replicates.")
    return _simulation_report(
        _ReportRuns(
            model=self.model,
            runs=list(self),
            confidence_level=confidence_level,
        ),
        positions,
        window,
        windows,
        observed_data,
        accumulators,
        prediction_intervals,
    )
```

## report

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

## report

```python
report() -> IdentifiabilityReportData
```

Report the identifiability of each parameter as plain data.

Returns:

| Type                        | Description                                                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `IdentifiabilityReportData` | parameters: each parameter, in problem order, with its structural and practical verdicts; sections: the evidence of each method that ran. |

Source code in `commol/identifiability/results.py`

```python
def report(self) -> IdentifiabilityReportData:
    """
    Report the identifiability of each parameter as plain data.

    Returns
    -------
    IdentifiabilityReportData
        ``parameters``: each parameter, in problem order, with its
        structural and practical verdicts; ``sections``: the evidence of
        each method that ran.
    """
    return identifiability_report(self)
```

## The types of the reports

Each report is a `TypedDict`: a plain dictionary whose keys a type checker knows. The root `commol` exports the types of `Model.applied_parameters`, of `Aggregate.report_sections` and of a simulation report; the [Identifiability API](https://munqu.github.io/commol/latest/api/identifiability/index.md) holds the types of the identifiability report.

## AppliedParametersReport

Bases: `ReportSections`

Every parameter of a calibrated model, and the sections of its kind (`ReportSections`, which `ModelStructure.report_sections` gives).

Source code in `commol/_model/applied_parameters.py`

```python
class AppliedParametersReport(ReportSections):
    """
    Every parameter of a calibrated model, and the sections of its kind
    (`ReportSections`, which `ModelStructure.report_sections` gives).
    """

    confidence_level: NotRequired[float]
    parameters: dict[str, ParameterEntry]
```

## ParameterEntry

Bases: `TypedDict`

Reported value of one parameter.

Source code in `commol/_model/applied_parameters.py`

```python
class ParameterEntry(TypedDict):
    """Reported value of one parameter."""

    value: float
    calibrated: bool
    spread: NotRequired[EntrySpread]
```

## EntrySpread

Bases: `TypedDict`

Weighted spread of one reported entry across the draws of a sample: the block under the key `spread` of the entry.

Source code in `commol/_model/report_entry.py`

```python
class EntrySpread(TypedDict):
    """Weighted spread of one reported entry across the draws of a sample: the
    block under the key `spread` of the entry."""

    mean: float
    median: float
    ci_lower: float
    ci_upper: float
    min: float
    max: float
    std: float
```

## InitialAmountEntry

Bases: `TypedDict`

Reported initial amount of one bin or one overridden compartment.

Source code in `commol/_model/report.py`

```python
class InitialAmountEntry(TypedDict):
    """Reported initial amount of one bin or one overridden compartment."""

    amount: float | None
    unit: str | None
    calibrated: bool
    spread: NotRequired[EntrySpread]
```

## CategorySplitEntry

Bases: `TypedDict`

Reported split of one category within the group it splits.

Source code in `commol/_model/report.py`

```python
class CategorySplitEntry(TypedDict):
    """Reported split of one category within the group it splits."""

    group: list[str]
    fraction: float
    calibrated: bool
    spread: NotRequired[EntrySpread]
```

## SimulationReport

Bases: `TypedDict`

Observed data, compartment series and accumulators of a run, and the prediction intervals when the report has them.

Source code in `commol/_simulation/reports.py`

```python
class SimulationReport(TypedDict):
    """
    Observed data, compartment series and accumulators of a run, and the
    prediction intervals when the report has them.
    """

    observed: dict[str, ObservedSeries]
    series: dict[str, _ReportLeaf]
    accumulators: dict[str, AccumulatorReport]
    prediction_intervals: NotRequired[PredictionIntervalsReport]
```

## ObservedSeries

Bases: `TypedDict`

One observed series on the reported window axis.

Source code in `commol/_simulation/reports.py`

```python
class ObservedSeries(TypedDict):
    """One observed series on the reported window axis."""

    values: list[float | None]
    weights: NotRequired[list[float | None]]
    scales: NotRequired[list[float | None]]
    outputs: NotRequired[list[str]]
    factor_id: NotRequired[str]
```

## AccumulatorReport

Bases: `TypedDict`

Both readings of one accumulator.

Source code in `commol/_simulation/reports.py`

```python
class AccumulatorReport(TypedDict):
    """Both readings of one accumulator."""

    total: _ReportLeaf
    windowed: WindowedAccumulator
    cumulative: WindowedAccumulator
```

## WindowedAccumulator

Bases: `TypedDict`

One reading of an accumulator, as a total and per output.

Source code in `commol/_simulation/reports.py`

```python
class WindowedAccumulator(TypedDict):
    """One reading of an accumulator, as a total and per output."""

    total: _ReportLeaf
    by_output: dict[str, _ReportLeaf]
```

## PredictionIntervalsReport

Bases: `TypedDict`

Prediction intervals at a confidence level, by series, and the unbounded ones.

Source code in `commol/_simulation/reports.py`

```python
class PredictionIntervalsReport(TypedDict):
    """Prediction intervals at a confidence level, by series, and the unbounded
    ones."""

    confidence_level: float
    series: dict[str, SeriesIntervalsReport]
    unbounded: dict[str, list[float]]
```

## SeriesIntervalsReport

Bases: `TypedDict`

The prediction intervals of one series.

Source code in `commol/_simulation/reports.py`

```python
class SeriesIntervalsReport(TypedDict):
    """The prediction intervals of one series."""

    positions: list[float]
    prediction: list[float]
    lower: list[float]
    upper: list[float]
```
