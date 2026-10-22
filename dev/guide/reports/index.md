# Reports

[Back to Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)

A report turns the data of an object into plain dictionaries, ready for JSON. Each report is a method of the object whose data it describes:

| Method                                                    | Reports                                                                                                                                              |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Model.applied_parameters(values, problem=None, …)`       | What the model holds and what calibration set                                                                                                        |
| `SimulationResults.report(positions, window, windows, …)` | What a run produced, next to the observations                                                                                                        |
| `SimulationReplicates.report(confidence_level, …)`        | The same, with each number reduced across the replicates of a stochastic engine                                                                      |
| `Predictive.report(confidence_level, …)`                  | The same, with each number reduced across the draws of a sample, with their weights                                                                  |
| `IdentifiabilityReport.report()`                          | The verdicts of an identifiability analysis ([Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/#the-report)) |

A single run and an ensemble give the same layout.

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
                bin_amounts={"A": 990, "B": 10, "C": 0},
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
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=500),
)

calibrated = model(None)
result = Calibrator(Simulation(calibrated), problem).run()
calibrated.apply_calibration_parameters(result, problem)

applied = calibrated.applied_parameters(result, problem)
print(json.dumps(applied["parameters"]["k2"]))  # {"value": 0.1, "calibrated": false}

results = Simulation(calibrated).run(num_steps=28)
run_report = results.report(
    positions=ends,
    window=7,
    windows=4,
    observed_data=observed_data,
    accumulators=["cum_ab"],
)
print(sorted(run_report))  # ['accumulators', 'observed', 'series']
```

## The applied parameters of a model

`Model.applied_parameters(values, problem=None, sample=None, confidence_level=None)` lists every number the calibrated model holds, and whether calibration set it. The `parameters` section is the same for every kind of model; the structure of the model adds the sections of its kind (`ModelStructure.report_sections`). An aggregate model adds `initial_amounts` and `category_splits`:

```json
{
  "parameters": {
    "k1": {"value": 0.4, "calibrated": true},
    "k2": {"value": 0.1, "calibrated": false}
  },
  "initial_amounts": {
    "A": {"amount": 990.0, "unit": "item", "calibrated": false},
    "B": {"amount": 10.0, "unit": "item", "calibrated": true},
    "C": {"amount": 100000.0, "unit": "mg/l", "calibrated": false}
  },
  "category_splits": {
    "g1": {"group": [], "fraction": 0.5, "calibrated": false},
    "g2": {"group": [], "fraction": 0.5, "calibrated": false}
  }
}
```

| Argument                     | Meaning                                                                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| The model                    | Call the method on the model **after** `apply_calibration_parameters`.                                                                  |
| `values`                     | A `CalibrationResult`, or an `{id: value}` mapping such as one of `sample.draws()`.                                                     |
| `problem`                    | Optional. Adds the `factor` parameters, which the model does not hold.                                                                  |
| `sample`, `confidence_level` | Optional. A `ParameterSample`: each entry gains its weighted statistics (`ParameterSample.summary`). A sample needs a confidence level. |

| Entry                               | Rule                                                                                                                                                                                                      |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Parameters                          | Only numeric parameters: formulas and series are inputs, not fitted values.                                                                                                                               |
| `initial_amounts`                   | One row per bin and per overridden compartment: the `amount` the engine builds, overrides included, and the `unit` of its bin. No row is a share of a total, so bins of different units sit side by side. |
| `category_splits`                   | One row per category: its `fraction` of the group it splits. A category has no amount, because it can split bins of different units.                                                                      |
| `group`                             | The conditioning categories of a conditional stratification, or `[]` when it splits every bin.                                                                                                            |
| An uncalibrated bin (`None` amount) | `amount` is `null`.                                                                                                                                                                                       |
| A `sample`                          | Each entry gains an `interval` block (`mean`, `median`, `ci_lower`, `ci_upper`, `min`, `max`, `std`, all weighted), and the report starts with `confidence_level`.                                        |

The key order of each entry is fixed (`value`, `calibrated`, then `interval`; `amount`, `unit`, `calibrated`, then `interval`; `group`, `fraction`, `calibrated`, then `interval`), so saved JSON files stay comparable. An interval is in the unit of its entry: an amount in the unit of its bin, a split as a fraction of its group.

## The report of a run

`SimulationResults.report(positions, window, windows, observed_data=(), accumulators=(), prediction_intervals=None)` reports a run on one axis of windows. It reads each value at a position of the results (`results.positions`), so it works for any model type: a differential model reports at non-integer positions once the results hold them ([Reporting at any position](https://munqu.github.io/commol/latest/guide/simulations/#reporting-at-any-position)).

| Block          | Content                                                                                                                                                                                                                                                                           |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observed`     | For each observation label: `values`, padded to `windows`, so it lines up with `windowed` index for index. An observation goes to the window that holds its position (the rule of `AxisUnits.window_index`). `weights`, `outputs` and `factor_id` appear when a series uses them. |
| `series`       | Each compartment output sampled at `positions`                                                                                                                                                                                                                                    |
| `accumulators` | For each accumulator: `total` (its final value), `windowed` (the gain in each window that ends within the results, as the loss compares it) and `cumulative` (sampled at `positions`); each reading as a `total` and `by_output`                                                  |

| Argument               | Meaning                                                                                                                                                                                                                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `positions`            | The positions where series and cumulative readings are sampled; each must be a position of the results (`ValueError` otherwise).                                                                                                                                                                                              |
| `window`, `windows`    | The window axis of `observed` and `windowed`: the length of one window and how many.                                                                                                                                                                                                                                          |
| `observed_data`        | The observations to report next to the run.                                                                                                                                                                                                                                                                                   |
| `accumulators`         | The accumulator ids to report. The results read the outputs of each one from their model (`results.model`, which `Simulation.run`, `run_at` and `replicates` set); results built by hand have no model, and reporting an accumulator raises `ValueError` that names it.                                                       |
| `prediction_intervals` | The `PredictionIntervals` of a [conformal method](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/index.md) (`ConformalResult.intervals`). The report gains a last block, `prediction_intervals`: its `confidence_level`, `series` (by label: `positions`, `prediction`, `lower`, `upper`) and `unbounded`. |

The report holds only these blocks, and `prediction_intervals` when it is given. Add your own envelope around them, such as a run label or summary figures in your domain's terms:

```python
document = {"kind": "calibration", "at": ends, **run_report}
print(json.dumps(document)[:30])
```

The results hold a reference to their model, not a copy, and a report reads only its structure (the outputs of each bin and accumulator), never the values of its parameters, so a later `update_parameters` does not change the report of a run. Results with their model still pickle.

## The report of an ensemble

`SimulationReplicates.report(confidence_level, positions, window, windows, observed_data=(), accumulators=(), prediction_intervals=None)` reports the replicates of a stochastic engine, and `Predictive.report` with the same arguments reports the draws of a [`Predictive`](https://munqu.github.io/commol/latest/guide/uncertainty/#the-predictive). The layout is the one of the report of a run; every number becomes a `{mean, median, ci_lower, ci_upper, min, max}` block, computed per member and then reduced across the members with their weights (equal for replicates). A predictive takes no point estimate: each number comes from the draws only.

```python
from commol import ParameterSample, Predictive, SampleMethod

sample = ParameterSample.equally_weighted(
    ["k1"], [(0.0, 1.0)], [[0.35], [0.4], [0.45]], SampleMethod(name="draws")
)
predictive = Predictive(Calibrator(Simulation(model(None)), problem), sample, range(29))
band_report = predictive.report(
    0.9, positions=ends, window=7, windows=4, accumulators=["cum_ab"]
)
print(list(band_report["series"]["A"]))  # ['mean', 'median', 'ci_lower', 'ci_upper', 'min', 'max']
```

## The report of an identifiability analysis

`IdentifiabilityReport.report()` gives the result of `Calibrator.identifiability`, of `ProfileResult.identifiability()`, or of `IdentifiabilityReport.merge`, as `parameters` (each parameter, in problem order, with its `structural` and `practical` verdicts, their reasons and combinations, or `None`) and `sections` (the evidence of each method that ran, by section name).

## Checklist

- [ ] The model reports `applied_parameters` after `apply_calibration_parameters`.
- [ ] `problem` is passed when there are `factor` parameters.
- [ ] `positions`, `window` and `windows` match the observations, and the results hold every position the report reads.

## Next steps

| To learn                           | Read                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| Samples, predictives and intervals | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)      |
| Signatures                         | [API reference: Reports](https://munqu.github.io/commol/latest/api/reports/index.md) |
