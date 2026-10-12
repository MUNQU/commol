# Reports API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Plain-dictionary reports of a calibrated model and of a run. Guide: [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md).

## applied_parameters_report

```python
applied_parameters_report(model: Model, result: CalibrationResult | ProbabilisticCalibrationResult | Mapping[str, float], problem: CalibrationProblem | None = None, parameter_statistics: Mapping[str, ParameterSetStatistics] | None = None, confidence_level: float | None = None) -> AppliedParametersReport
```

Report every parameter and initial condition of a calibrated model.

Each entry carries the value the model holds and whether calibration set it, so entries the calibration left alone appear alongside the fitted ones.

Parameters that are not a single number, such as a formula or a time-series, are omitted: they are inputs rather than fitted values. `scale` parameters have no place in the model, so they are reported from `problem` when it is given.

Initial conditions carry the effective head `value` the engine builds from the uniform expansion plus the declared compartment overrides, its `fraction` as that head count over the declared population size, and the `group` it belongs to: `population` for bins and unconditional stratifications, and the conditioning categories for a conditional one. `N` is the declared population size, with fraction 1.0. A bin left uncalibrated keeps a `None` fraction and `None` value as its "needs calibration" marker even when overrides touch some of its compartments.

For a probabilistic result each entry gains an `interval` holding the ensemble spread. A parameter is in the unit it was calibrated in. An initial-condition entry is in the fraction the row reports: a conditional stratification's category is calibrated as a group fraction but reported over the whole population, so its interval is rescaled by the group's share of the population and brackets the reported fraction.

Parameters:

| Name                   | Type                                   | Description                            | Default                                                                                                             |
| ---------------------- | -------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `model`                | `Model`                                | The model carrying the applied values. | *required*                                                                                                          |
| `result`               | \`CalibrationResult                    | ProbabilisticCalibrationResult         | Mapping\`                                                                                                           |
| `problem`              | \`CalibrationProblem                   | None\`                                 | The problem the result came from, used to report scale parameters.                                                  |
| `parameter_statistics` | \`Mapping[str, ParameterSetStatistics] | None\`                                 | Ensemble spread per entry. Taken from result when it is a probabilistic result and this is not given.               |
| `confidence_level`     | \`float                                | None\`                                 | Confidence level the intervals describe. Taken from result when it is a probabilistic result and this is not given. |

Returns:

| Type                      | Description                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------- |
| `AppliedParametersReport` | Keys parameters and initial_conditions, preceded by confidence_level when there is one. |

Source code in `commol/api/reporting.py`

```python
def applied_parameters_report(
    model: Model,
    result: CalibrationResult | ProbabilisticCalibrationResult | Mapping[str, float],
    problem: CalibrationProblem | None = None,
    parameter_statistics: Mapping[str, ParameterSetStatistics] | None = None,
    confidence_level: float | None = None,
) -> AppliedParametersReport:
    """
    Report every parameter and initial condition of a calibrated model.

    Each entry carries the value the model holds and whether calibration set
    it, so entries the calibration left alone appear alongside the fitted ones.

    Parameters that are not a single number, such as a formula or a
    time-series, are omitted: they are inputs rather than fitted values.
    ``scale`` parameters have no place in the model, so they are reported from
    `problem` when it is given.

    Initial conditions carry the effective head ``value`` the engine builds
    from the uniform expansion plus the declared compartment overrides, its
    ``fraction`` as that head count over the declared population size, and the
    ``group`` it belongs to: ``population`` for bins and unconditional
    stratifications, and the conditioning categories for a conditional one.
    ``N`` is the declared population size, with fraction 1.0. A bin left
    uncalibrated keeps a ``None`` fraction and ``None`` value as its "needs
    calibration" marker even when overrides touch some of its compartments.

    For a probabilistic result each entry gains an ``interval`` holding the
    ensemble spread. A parameter is in the unit it was calibrated in. An
    initial-condition entry is in the fraction the row reports: a conditional
    stratification's category is calibrated as a group fraction but reported
    over the whole population, so its interval is rescaled by the group's share
    of the population and brackets the reported fraction.

    Parameters
    ----------
    model : Model
        The model carrying the applied values.
    result : CalibrationResult | ProbabilisticCalibrationResult | Mapping
        The calibration outcome, or a mapping of parameter id to value such as
        one ensemble member's parameters.
    problem : CalibrationProblem | None, optional
        The problem the result came from, used to report scale parameters.
    parameter_statistics : Mapping[str, ParameterSetStatistics] | None, optional
        Ensemble spread per entry. Taken from `result` when it is a
        probabilistic result and this is not given.
    confidence_level : float | None, optional
        Confidence level the intervals describe. Taken from `result` when it is
        a probabilistic result and this is not given.

    Returns
    -------
    AppliedParametersReport
        Keys ``parameters`` and ``initial_conditions``, preceded by
        ``confidence_level`` when there is one.
    """
    calibrated = calibrated_values(result)
    parameter_statistics, confidence_level = _ensemble_details(
        result, parameter_statistics, confidence_level
    )
    calibrated_ids = set(calibrated)

    parameters: dict[str, ParameterEntry] = {
        parameter.id: _parameter_entry(
            parameter.id, parameter.value, calibrated_ids, parameter_statistics
        )
        for parameter in model.parameters
        if isinstance(parameter.value, (int, float))
    }
    for scale_id in _scale_ids(problem):
        if scale_id in calibrated:
            parameters[scale_id] = _parameter_entry(
                scale_id, calibrated[scale_id], calibrated_ids, parameter_statistics
            )

    initial_conditions = _initial_conditions_report(
        model, calibrated_ids, parameter_statistics
    )
    if confidence_level is None:
        return {
            "parameters": parameters,
            "initial_conditions": initial_conditions,
        }
    # `confidence_level` leads the report, so it is written first.
    return {
        "confidence_level": confidence_level,
        "parameters": parameters,
        "initial_conditions": initial_conditions,
    }
```

## simulation_report

```python
simulation_report(simulation: Simulation, results: Mapping[str, Sequence[float]], steps: Sequence[int], window_steps: int, num_windows: int, observed_data: Iterable[ObservedDataPoint] = (), accumulators: Iterable[str] = (), ensemble_runs: Sequence[Mapping[str, Sequence[float]]] | None = None, confidence_level: float | None = None) -> SimulationReport
```

Report the observed data, compartment series and accumulators of a run.

The three blocks share one window axis, so the observed values, the windowed accumulator increments and the sampled series can be read against each other. With `ensemble_runs` every leaf becomes a `{mean, median, ci_lower, ci_upper, min, max}` block computed across members, and the layout is otherwise unchanged.

Callers add whatever envelope they need around these blocks, such as a run label, axis labels or domain-specific summary figures.

Parameters:

| Name               | Type                                          | Description                                                  | Default                                     |
| ------------------ | --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------- |
| `simulation`       | `Simulation`                                  | The simulation the results came from.                        | *required*                                  |
| `results`          | `Mapping[str, Sequence[float]]`               | Results in dict_of_lists form.                               | *required*                                  |
| `steps`            | `Sequence[int]`                               | Steps the series and cumulative accumulators are sampled at. | *required*                                  |
| `window_steps`     | `int`                                         | Length of one window, in steps.                              | *required*                                  |
| `num_windows`      | `int`                                         | Number of windows on the observed and windowed axes.         | *required*                                  |
| `observed_data`    | `Iterable[ObservedDataPoint]`                 | The calibration targets to report alongside the run.         | `()`                                        |
| `accumulators`     | `Iterable[str]`                               | Accumulator ids to report.                                   | `()`                                        |
| `ensemble_runs`    | \`Sequence\[Mapping\[str, Sequence[float]\]\] | None\`                                                       | One results mapping per ensemble member.    |
| `confidence_level` | \`float                                       | None\`                                                       | Confidence level of the reported intervals. |

Returns:

| Type               | Description                             |
| ------------------ | --------------------------------------- |
| `SimulationReport` | Keys observed, series and accumulators. |

Source code in `commol/api/reporting.py`

```python
def simulation_report(
    simulation: Simulation,
    results: Mapping[str, Sequence[float]],
    steps: Sequence[int],
    window_steps: int,
    num_windows: int,
    observed_data: Iterable[ObservedDataPoint] = (),
    accumulators: Iterable[str] = (),
    ensemble_runs: Sequence[Mapping[str, Sequence[float]]] | None = None,
    confidence_level: float | None = None,
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
    results : Mapping[str, Sequence[float]]
        Results in `dict_of_lists` form.
    steps : Sequence[int]
        Steps the series and cumulative accumulators are sampled at.
    window_steps : int
        Length of one window, in steps.
    num_windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the run.
    accumulators : Iterable[str], optional
        Accumulator ids to report.
    ensemble_runs : Sequence[Mapping[str, Sequence[float]]] | None, optional
        One results mapping per ensemble member.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.

    Returns
    -------
    SimulationReport
        Keys ``observed``, ``series`` and ``accumulators``.
    """
    accumulator_ids = list(accumulators)
    return {
        "observed": observed_report(observed_data, window_steps, num_windows),
        "series": series_report(
            simulation,
            results,
            steps,
            accumulator_ids,
            ensemble_runs,
            confidence_level,
        ),
        "accumulators": accumulators_report(
            simulation,
            results,
            accumulator_ids,
            steps,
            window_steps,
            num_windows,
            ensemble_runs,
            confidence_level,
        ),
    }
```
