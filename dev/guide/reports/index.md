# Reports

[Back to Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)

Two functions turn a calibrated model and a run into plain dictionaries, ready for JSON: `applied_parameters_report` (what the model holds and what calibration set) and `simulation_report` (what a run produced, next to the observations). Both work for a single calibration and for a sample of the parameters, with the same layout.

## Quick path

```python
import json

from commol import (
    Aggregate,
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
            id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total", accumulators=["cum_ab"]
        )
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .build(
            Aggregate(
                total=1000,
                bin_fractions={"A": 0.99, "B": 0.01, "C": 0.0},
            ),
            engine="DifferenceEquations",
        )
    )

ends = [7, 14, 21, 28]
truth = Simulation(model(0.4)).run(num_steps=28)
observed_data = [
    ObservedDataPoint(at=end, output="cum_ab", value=count, window=7)
    for end, count in zip(ends, windowed_totals(truth["cum_ab"], truth.positions, 7, ends))
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
    at=ends,
    window=7,
    num_windows=4,
    observed_data=observed_data,
    accumulators=["cum_ab"],
)
print(sorted(run_report))  # ['accumulators', 'observed', 'series']
```

## `applied_parameters_report`

`applied_parameters_report(model, result, problem=None, parameter_statistics=None, confidence_level=None)` lists every number the calibrated model holds, and whether calibration set it. The `parameters` section is the same for every kind of model; the structure of the model adds the sections of its kind (`ModelStructure.report_sections`). An aggregate model adds `initial_conditions`:

```json
{
  "parameters": {
    "k1": {"value": 0.4, "calibrated": true},
    "k2": {"value": 0.1, "calibrated": false}
  },
  "initial_conditions": {
    "total": {"group": "total", "fraction": 1.0, "value": 1000.0, "calibrated": false},
    "A": {"group": "total", "fraction": 0.99, "value": 990.0, "calibrated": false},
    "B": {"group": "total", "fraction": 0.01, "value": 10.0, "calibrated": false},
    "C": {"group": "total", "fraction": 0.0, "value": 0.0, "calibrated": false}
  }
}
```

| Argument                     | Meaning                                                                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                      | The model **after** `apply_calibration_parameters`.                                                                                     |
| `result`                     | A `CalibrationResult`, or an `{id: value}` mapping such as one of `sample.draws()`.                                                     |
| `problem`                    | Optional. Adds the `factor` parameters, which the model does not hold.                                                                  |
| `sample`, `confidence_level` | Optional. A `ParameterSample`: each entry gains its weighted statistics (`ParameterSample.summary`). A sample needs a confidence level. |

| Entry                                 | Rule                                                                                                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Parameters                            | Only numeric parameters: formulas and series are inputs, not fitted values.                                                                                        |
| `total`                               | The declared `total`, with fraction 1.0.                                                                                                                           |
| Initial-condition `value`             | The amount the engine builds, overrides included.                                                                                                                  |
| Initial-condition `fraction`          | That amount over `total`, so every row shares one denominator.                                                                                                     |
| `group`                               | `"total"`, or the conditioning categories for a conditional stratification.                                                                                        |
| An uncalibrated bin (`None` fraction) | `fraction` and `value` are `null`.                                                                                                                                 |
| A `sample`                            | Each entry gains an `interval` block (`mean`, `median`, `ci_lower`, `ci_upper`, `min`, `max`, `std`, all weighted), and the report starts with `confidence_level`. |

The key order of each entry is fixed (`value`, `calibrated`, then `interval`; `group`, `fraction`, `value`, `calibrated`, then `interval`), so saved JSON files stay comparable.

## `simulation_report`

`simulation_report(simulation, results, at, window, num_windows, observed_data=(), accumulators=(), ensemble_runs=None, confidence_level=None)` reports a run on one axis of windows. It reads each value at a position of the results (`results.positions`), so it works for any model type: a differential model reports at non-integer positions once the results hold them ([Reporting at any position](https://munqu.github.io/commol/latest/guide/simulations/#reporting-at-any-position)).

| Block          | Content                                                                                                                                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observed`     | For each observation label: `values`, padded to `num_windows`, so it lines up with `windowed` index for index. An observation goes to the window that holds its position (`window_index`). `weights`, `outputs` and `factor_id` appear when a series uses them. |
| `series`       | Each compartment output sampled at `at`                                                                                                                                                                                                                         |
| `accumulators` | For each accumulator: `total` (its final value), `windowed` (the gain in each window that ends within the results, as the loss compares it) and `cumulative` (sampled at `at`); each reading as a `total` and `by_output`                                       |

| Argument                            | Meaning                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results`                           | `SimulationResults` of `run` or `run_at`.                                                                                                                                                                                                                                                                                                                                 |
| `at`                                | The positions where series and cumulative readings are sampled; each must be a position of the results (`ValueError` otherwise).                                                                                                                                                                                                                                          |
| `window`, `num_windows`             | The window axis of `observed` and `windowed`.                                                                                                                                                                                                                                                                                                                             |
| `observed_data`                     | The observations to report next to the run.                                                                                                                                                                                                                                                                                                                               |
| `accumulators`                      | The accumulator ids to report.                                                                                                                                                                                                                                                                                                                                            |
| `ensemble_runs`, `confidence_level` | The `SimulationReplicates` of a stochastic engine, one `SimulationResults` per member, or a [`Predictive`](https://munqu.github.io/commol/latest/guide/uncertainty/#the-predictive), whose draws carry their weights. Every number becomes a `{mean, median, ci_lower, ci_upper, min, max}` block, reduced member by member with the weights; the layout does not change. |

The report holds only these blocks. Add your own envelope around them, such as a run label or summary figures in your domain's terms:

```python
document = {"kind": "calibration", "at": ends, **run_report}
print(json.dumps(document)[:30])
```

## Checklist

- [ ] The model passed to `applied_parameters_report` already has the calibrated values.
- [ ] `problem` is passed when there are `factor` parameters.
- [ ] `at`, `window` and `num_windows` match the observations, and the results hold every position the report reads.

## Next steps

| To learn                           | Read                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| Samples, predictives and intervals | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)      |
| Signatures                         | [API reference: Reports](https://munqu.github.io/commol/latest/api/reports/index.md) |
