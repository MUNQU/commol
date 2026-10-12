# Reports

[Back to Model Calibration](calibration.md)

Two functions turn a calibrated model and a run into plain dictionaries, ready
for JSON: `applied_parameters_report` (what the model holds and what calibration
set) and `simulation_report` (what a run produced, next to the observations).
Both work for a single calibration and for an ensemble, with the same layout.

## Quick path

```python
import json

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
    applied_parameters_report,
    simulation_report,
    windowed_totals,
)

def model(k1: float | None):
    return (
        ModelBuilder(name="Reported")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="k2", value=0.1)
        .add_accumulator(id="cum_ab", name="Total moved from A to B")
        .add_transition(
            id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N", accumulators=["cum_ab"]
        )
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 0.99},
                {"bin": "B", "fraction": 0.01},
                {"bin": "C", "fraction": 0.0},
            ],
        )
        .build(typology="DifferenceEquations")
    )

steps = [7, 14, 21, 28]
truth = Simulation(model(0.4)).run(num_steps=28)
observed_data = [
    ObservedDataPoint(step=step, compartment="cum_ab", value=count, window_steps=7)
    for step, count in zip(steps, windowed_totals(truth["cum_ab"], 7, steps))
]
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0)
    ],
    optimization_config=NelderMeadConfig(max_iterations=500),
)

calibrated = model(None)
result = Calibrator(Simulation(calibrated), problem).run()
calibrated.apply_calibration_parameters(result, problem)

applied = applied_parameters_report(calibrated, result, problem)
print(json.dumps(applied["parameters"]["k2"]))  # {"value": 0.1, "calibrated": false}

simulation = Simulation(calibrated)
results = simulation.run(num_steps=28)
run_report = simulation_report(
    simulation,
    results,
    steps=steps,
    window_steps=7,
    num_windows=4,
    observed_data=observed_data,
    accumulators=["cum_ab"],
)
print(sorted(run_report))  # ['accumulators', 'observed', 'series']
```

## `applied_parameters_report`

`applied_parameters_report(model, result, problem=None, parameter_statistics=None,
confidence_level=None)` lists every number the calibrated model holds, and
whether calibration set it:

```json
{
  "parameters": {
    "k1": {"value": 0.4, "calibrated": true},
    "k2": {"value": 0.1, "calibrated": false}
  },
  "initial_conditions": {
    "N": {"group": "population", "fraction": 1.0, "value": 1000.0, "calibrated": false},
    "A": {"group": "population", "fraction": 0.99, "value": 990.0, "calibrated": false},
    "B": {"group": "population", "fraction": 0.01, "value": 10.0, "calibrated": false},
    "C": {"group": "population", "fraction": 0.0, "value": 0.0, "calibrated": false}
  }
}
```

| Argument | Meaning |
|---|---|
| `model` | The model **after** `apply_calibration_parameters`. |
| `result` | A `CalibrationResult`, a `ProbabilisticCalibrationResult`, or an `{id: value}` mapping. |
| `problem` | Optional. Adds the `scale` parameters, which the model does not hold. |
| `parameter_statistics`, `confidence_level` | Optional. Taken from a probabilistic `result` when not given. |

| Entry | Rule |
|---|---|
| Parameters | Only numeric parameters: formulas and time series are inputs, not fitted values. |
| `N` | The declared `population_size`, with fraction 1.0. |
| Initial-condition `value` | The head count the engine builds, overrides included. |
| Initial-condition `fraction` | That head count over `population_size`, so every row shares one denominator. |
| `group` | `"population"`, or the conditioning categories for a conditional stratification. |
| An uncalibrated bin (`None` fraction) | `fraction` and `value` are `null`. |
| A probabilistic result | Each entry gains an `interval` block, and the report starts with `confidence_level`. |

The key order of each entry is fixed (`value`, `calibrated`, then `interval`;
`group`, `fraction`, `value`, `calibrated`, then `interval`), so saved JSON files
stay comparable.

## `simulation_report`

`simulation_report(simulation, results, steps, window_steps, num_windows,
observed_data=(), accumulators=(), ensemble_runs=None, confidence_level=None)`
reports a run on one axis of windows:

| Block | Content |
|---|---|
| `observed` | For each observation label: `values`, padded to `num_windows`, so it lines up with `windowed` index for index |
| `series` | Each compartment output sampled at `steps` |
| `accumulators` | For each accumulator: `total` (its final value), `windowed` (the gain in each window, as the loss compares it) and `cumulative` (sampled at `steps`); each reading as a `total` and `by_output` |

| Argument | Meaning |
|---|---|
| `steps` | The steps where series and cumulative readings are sampled. |
| `window_steps`, `num_windows` | The window axis of `observed` and `windowed`. |
| `observed_data` | The observations to report next to the run. |
| `accumulators` | The accumulator ids to report. |
| `ensemble_runs`, `confidence_level` | One results mapping per ensemble member. Every number becomes a `{mean, median, ci_lower, ci_upper, min, max}` block; the layout does not change. |

The report holds only these blocks. Add your own envelope around them, such as
a run label or summary figures in your domain's terms:

```python
document = {"kind": "calibration", "steps": steps, **run_report}
print(json.dumps(document)[:30])
```

## Checklist

- [ ] The model passed to `applied_parameters_report` already has the calibrated values.
- [ ] `problem` is passed when there are `scale` parameters.
- [ ] `steps`, `window_steps` and `num_windows` match the observations.

## Next steps

| To learn | Read |
|---|---|
| Ensembles and intervals | [Probabilistic Calibration](probabilistic-calibration.md) |
| Signatures | [API reference: Reports](../api/reports.md) |
