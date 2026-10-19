# Reports API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Plain-dictionary reports of a calibrated model and of a run. Guide: [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md).

## applied_parameters_report

```python
applied_parameters_report(model: Model, result: CalibrationResult | Mapping[str, float], problem: CalibrationProblem | None = None, sample: ParameterSample | None = None, confidence_level: float | None = None) -> AppliedParametersReport
```

Report every parameter of a calibrated model, and the values of its kind.

Each entry carries the value the model holds and whether calibration set it, so entries the calibration left alone appear alongside the fitted ones. The structure of the model adds the sections of its kind (`ModelStructure.report_sections`); an aggregate model adds `initial_amounts` and `category_splits`.

Parameters that are not a single number, such as a formula or a series, are omitted: they are inputs rather than fitted values. `factor` parameters have no place in the model, so they are reported from `problem` when it is given.

`initial_amounts` gives each bin and each overridden compartment of an aggregate model its `amount` the engine builds and the `unit` of its bin, so bins of different units are reported side by side; no row is a share of a total. A bin left uncalibrated keeps a `None` amount as its "needs calibration" marker even when overrides touch some of its compartments. `category_splits` gives each category its `fraction` of the group it splits, and that `group`: the categories a conditional stratification is conditioned on, empty when it splits every bin.

With a `sample` each entry gains an `interval` holding the weighted statistics of its draws (`ParameterSample.summary`). Every entry is in the unit it was calibrated in: a parameter in its own, an amount in the unit of its bin, and a category split as a fraction of its group.

Parameters:

| Name               | Type                 | Description                            | Default                                                                                     |
| ------------------ | -------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------- |
| `model`            | `Model`              | The model carrying the applied values. | *required*                                                                                  |
| `result`           | \`CalibrationResult  | Mapping\`                              | The calibration result, or a mapping of parameter id to value such as one draw of a sample. |
| `problem`          | \`CalibrationProblem | None\`                                 | The problem the result came from, used to report factor parameters.                         |
| `sample`           | \`ParameterSample    | None\`                                 | Draws of the parameters, such as the sample of an uncertainty method.                       |
| `confidence_level` | \`float              | None\`                                 | Confidence level of the intervals; required with a sample.                                  |

Returns:

| Type                      | Description                                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AppliedParametersReport` | Key parameters and the sections of the kind (initial_amounts and category_splits for an aggregate model), preceded by confidence_level when there is one. |

Raises:

| Type         | Description                                      |
| ------------ | ------------------------------------------------ |
| `ValueError` | If a sample is given without a confidence level. |

Source code in `commol/api/reporting.py`

```python
def applied_parameters_report(
    model: Model,
    result: CalibrationResult | Mapping[str, float],
    problem: CalibrationProblem | None = None,
    sample: ParameterSample | None = None,
    confidence_level: float | None = None,
) -> AppliedParametersReport:
    """
    Report every parameter of a calibrated model, and the values of its kind.

    Each entry carries the value the model holds and whether calibration set
    it, so entries the calibration left alone appear alongside the fitted ones.
    The structure of the model adds the sections of its kind
    (`ModelStructure.report_sections`); an aggregate model adds
    ``initial_amounts`` and ``category_splits``.

    Parameters that are not a single number, such as a formula or a
    series, are omitted: they are inputs rather than fitted values.
    ``factor`` parameters have no place in the model, so they are reported from
    `problem` when it is given.

    ``initial_amounts`` gives each bin and each overridden compartment of an
    aggregate model its ``amount`` the engine builds and the ``unit`` of its
    bin, so bins of different units are reported side by side; no row is a
    share of a total. A bin left uncalibrated keeps a ``None`` amount as its
    "needs calibration" marker even when overrides touch some of its
    compartments. ``category_splits`` gives each category its ``fraction`` of
    the group it splits, and that ``group``: the categories a conditional
    stratification is conditioned on, empty when it splits every bin.

    With a `sample` each entry gains an ``interval`` holding the weighted
    statistics of its draws (`ParameterSample.summary`). Every entry is in
    the unit it was calibrated in: a parameter in its own, an amount in the
    unit of its bin, and a category split as a fraction of its group.

    Parameters
    ----------
    model : Model
        The model carrying the applied values.
    result : CalibrationResult | Mapping
        The calibration result, or a mapping of parameter id to value such as
        one draw of a sample.
    problem : CalibrationProblem | None, optional
        The problem the result came from, used to report factor parameters.
    sample : ParameterSample | None, optional
        Draws of the parameters, such as the sample of an uncertainty method.
    confidence_level : float | None, optional
        Confidence level of the intervals; required with a sample.

    Returns
    -------
    AppliedParametersReport
        Key ``parameters`` and the sections of the kind (``initial_amounts``
        and ``category_splits`` for an aggregate model), preceded by
        ``confidence_level`` when there is one.

    Raises
    ------
    ValueError
        If a sample is given without a confidence level.
    """
    calibrated = calibrated_values(result)
    parameter_statistics = _sample_statistics(sample, confidence_level)
    calibrated_ids = set(calibrated)

    parameters: dict[str, ParameterEntry] = {
        parameter.id: _parameter_entry(
            parameter.id, parameter.value, calibrated_ids, parameter_statistics
        )
        for parameter in model.parameters
        if isinstance(parameter.value, (int, float))
    }
    for factor_id in _factor_ids(problem):
        if factor_id in calibrated:
            parameters[factor_id] = _parameter_entry(
                factor_id, calibrated[factor_id], calibrated_ids, parameter_statistics
            )

    report: dict[str, object] = {}
    # `confidence_level` leads the report, so it is written first.
    if confidence_level is not None:
        report["confidence_level"] = confidence_level
    report["parameters"] = parameters
    report.update(
        model.structure.report_sections(model, calibrated_ids, parameter_statistics)
    )
    return cast(AppliedParametersReport, report)
```

## simulation_report

```python
simulation_report(simulation: Simulation, results: SimulationResults, at: Sequence[float], window: float, num_windows: int, observed_data: Iterable[ObservedDataPoint] = (), accumulators: Iterable[str] = (), ensemble_runs: Sequence[SimulationResults] | Predictive | None = None, confidence_level: float | None = None, prediction_intervals: PredictionIntervals | None = None) -> SimulationReport
```

Report the observed data, compartment series and accumulators of a run.

The three blocks share one window axis, so the observed values, the windowed accumulator increments and the sampled series can be read against each other. With `ensemble_runs` every leaf becomes a `{mean, median, ci_lower, ci_upper, min, max}` block computed across members, and the layout is otherwise unchanged.

Callers add whatever envelope they need around these blocks, such as a run label, axis labels or domain-specific summary figures.

Parameters:

| Name                   | Type                          | Description                                                      | Default                                                                                       |
| ---------------------- | ----------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `simulation`           | `Simulation`                  | The simulation the results came from.                            | *required*                                                                                    |
| `results`              | `SimulationResults`           | Results of a run, with their positions.                          | *required*                                                                                    |
| `at`                   | `Sequence[float]`             | Positions the series and cumulative accumulators are sampled at. | *required*                                                                                    |
| `window`               | `float`                       | Length of one window on the axis.                                | *required*                                                                                    |
| `num_windows`          | `int`                         | Number of windows on the observed and windowed axes.             | *required*                                                                                    |
| `observed_data`        | `Iterable[ObservedDataPoint]` | The calibration targets to report alongside the run.             | `()`                                                                                          |
| `accumulators`         | `Iterable[str]`               | Accumulator ids to report.                                       | `()`                                                                                          |
| `ensemble_runs`        | \`Sequence[SimulationResults] | Predictive                                                       | None\`                                                                                        |
| `confidence_level`     | \`float                       | None\`                                                           | Confidence level of the reported intervals.                                                   |
| `prediction_intervals` | \`PredictionIntervals         | None\`                                                           | Prediction intervals of a conformal method (ConformalResult.intervals), reported as they are. |

Returns:

| Type               | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `SimulationReport` | Keys observed, series and accumulators, then prediction_intervals when they are given. |

Raises:

| Type         | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `ValueError` | If the results have no value at a position the report reads. |

Source code in `commol/api/reporting.py`

```python
def simulation_report(
    simulation: Simulation,
    results: SimulationResults,
    at: Sequence[float],
    window: float,
    num_windows: int,
    observed_data: Iterable[ObservedDataPoint] = (),
    accumulators: Iterable[str] = (),
    ensemble_runs: "Sequence[SimulationResults] | Predictive | None" = None,
    confidence_level: float | None = None,
    prediction_intervals: PredictionIntervals | None = None,
) -> SimulationReport:
    """
    Report the observed data, compartment series and accumulators of a run.

    The three blocks share one window axis, so the observed values, the
    windowed accumulator increments and the sampled series can be read against
    each other. With `ensemble_runs` every leaf becomes a
    ``{mean, median, ci_lower, ci_upper, min, max}`` block computed across
    members, and the layout is otherwise unchanged.

    Callers add whatever envelope they need around these blocks, such as a run
    label, axis labels or domain-specific summary figures.

    Parameters
    ----------
    simulation : Simulation
        The simulation the results came from.
    results : SimulationResults
        Results of a run, with their positions.
    at : Sequence[float]
        Positions the series and cumulative accumulators are sampled at.
    window : float
        Length of one window on the axis.
    num_windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the run.
    accumulators : Iterable[str], optional
        Accumulator ids to report.
    ensemble_runs : Sequence[SimulationResults] | Predictive | None, optional
        One results mapping per member, such as the replicates of a stochastic
        engine, or a `Predictive`, whose draws carry their weights.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.
    prediction_intervals : PredictionIntervals | None, optional
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
        If the results have no value at a position the report reads.
    """
    accumulator_ids = list(accumulators)
    report: SimulationReport = {
        "observed": observed_report(observed_data, window, num_windows),
        "series": series_report(
            simulation,
            results,
            at,
            accumulator_ids,
            ensemble_runs,
            confidence_level,
        ),
        "accumulators": accumulators_report(
            simulation,
            results,
            accumulator_ids,
            at,
            window,
            num_windows,
            ensemble_runs,
            confidence_level,
        ),
    }
    if prediction_intervals is not None:
        report["prediction_intervals"] = cast(
            "PredictionIntervalsReport", prediction_intervals.model_dump()
        )
    return report
```

## identifiability_report

```python
identifiability_report(report: IdentifiabilityReport) -> IdentifiabilityReportData
```

Report the identifiability of each parameter as plain data.

Parameters:

| Name     | Type                    | Description                                                                  | Default    |
| -------- | ----------------------- | ---------------------------------------------------------------------------- | ---------- |
| `report` | `IdentifiabilityReport` | The result of Calibrator.identifiability, or of IdentifiabilityReport.merge. | *required* |

Returns:

| Type                        | Description                                                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `IdentifiabilityReportData` | parameters: each parameter, in problem order, with its structural and practical verdicts; sections: the evidence of each method that ran. |

Source code in `commol/api/reporting.py`

```python
def identifiability_report(report: IdentifiabilityReport) -> IdentifiabilityReportData:
    """
    Report the identifiability of each parameter as plain data.

    Parameters
    ----------
    report : IdentifiabilityReport
        The result of ``Calibrator.identifiability``, or of
        `IdentifiabilityReport.merge`.

    Returns
    -------
    IdentifiabilityReportData
        ``parameters``: each parameter, in problem order, with its structural
        and practical verdicts; ``sections``: the evidence of each method that
        ran.
    """
    parameters: dict[str, VerdictsEntry] = {}
    for name in report.parameters:
        verdicts = report.verdicts[name]
        structural = verdicts.structural
        practical = verdicts.practical
        parameters[name] = {
            "structural": None
            if structural is None
            else {
                "verdict": structural.verdict,
                "reasons": list(structural.reasons),
                "combinations": list(structural.combinations),
            },
            "practical": None
            if practical is None
            else {"verdict": practical.verdict, "reasons": list(practical.reasons)},
        }
    sections: dict[str, list[dict[str, object]]] = {
        "structural": [section.model_dump() for section in report.structural],
        "collinearity": [section.model_dump() for section in report.collinearity],
        "sample": [section.model_dump() for section in report.sample],
        "starts": [section.model_dump() for section in report.starts],
    }
    return {"parameters": parameters, "sections": sections}
```
