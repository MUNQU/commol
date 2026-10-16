# Observations

[Back to Model Calibration](calibration.md)

An `ObservedDataPoint` says what the model must reproduce: an output at a
position on the axis. This page shows how to observe several outputs, sums of
outputs, partial observations (factors), counts per period (windows) and
positions between integers, and how to weigh them.

## Quick path

```python
from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)

def model(k1: float | None):
    return (
        ModelBuilder(name="Observed")
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

truth = Simulation(model(0.4)).run(num_steps=70)

def calibrate_k1(observed_data, **problem_options) -> float:
    problem = CalibrationProblem(
        observed_data=observed_data,
        parameters=[
            CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0)
        ],
        optimization_config=NelderMeadConfig(max_iterations=2000),
        **problem_options,
    )
    result = Calibrator(Simulation(model(None)), problem).run()
    return round(result.best_parameters["k1"], 3)

# Observe B and C at the same steps
observed = [
    ObservedDataPoint(at=step, output=output, value=truth[output][step])
    for step in range(10, 51, 10)
    for output in ("B", "C")
]
print(calibrate_k1(observed))  # 0.4
```

Each observation compares the model output named in `output` with `value` at
the position `at`. Any output works: a compartment (`B`, `B_g1`) or an
accumulator (`cum_ab`). The observation's `label` names its series in reports,
plots and the per-series normalization; it defaults to `output`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `at` | `float` | required | The position on the axis. The engine decides which positions its model type accepts. |
| `output` | `str \| None` | `None` | The observed output. Give `output` or `outputs`, not both. |
| `outputs` | `list[str] \| None` | `None` | Outputs summed before the comparison; needs a `label`. |
| `label` | `str` | `output` | The name of the observed series. |
| `value` | `float` | required | The observed value, not negative. |
| `weight` | `float` | `1.0` | How much the observation counts in the loss. |
| `factor_id` | `str \| None` | `None` | A `factor` unknown that multiplies the prediction. |
| `window` | `float \| None` | `None` | Compares with the gain over `(at − window, at]` instead of the value. |

An unknown field, such as a misspelt name, raises `ValidationError` and names
it.

## Sums of outputs

When one measurement covers several outputs, list them in `outputs`. They are
summed before the comparison; `label` names the sum for plots and reports:

```python
b_and_c = [
    ObservedDataPoint(
        at=step,
        label="B_plus_C",
        outputs=["B", "C"],
        value=truth["B"][step] + truth["C"][step],
    )
    for step in range(10, 51, 10)
]
print(calibrate_k1(b_and_c))  # 0.4
```

`outputs` may name compartments and accumulators, and works with `factor_id`
and `window`. It must not be empty.

## Weights and normalization

| Option | Where | Effect |
|---|---|---|
| `weight=2.0` | `ObservedDataPoint` | The observation counts twice as much in every loss: its squared (or absolute) residual is multiplied by 2. See [Losses](calibration.md#losses). |
| `normalize_observations=True` | `CalibrationProblem` | Divides each residual by the RMS of its series, so a series of thousands does not drown a series of tens. |

```python
weighted = [
    ObservedDataPoint(at=10, output="B", value=truth["B"][10], weight=2.0),
    ObservedDataPoint(at=30, output="B", value=truth["B"][30], weight=1.0),
    ObservedDataPoint(at=30, output="cum_ab", value=truth["cum_ab"][30], weight=1.0),
]
print(calibrate_k1(weighted, loss_function="sse", normalize_observations=True))  # 0.4
```

## Partial observations: factors

When the data sees only part of an output, for example a fixed share of it, a
`factor` unknown multiplies the prediction before the comparison. Link
observations to it with `factor_id`:

```python
seen = 0.3
partial = [
    ObservedDataPoint(at=step, output="B", value=seen * truth["B"][step], factor_id="share_seen")
    for step in range(5, 51, 5)
]

problem = CalibrationProblem(
    observed_data=partial,
    parameters=[
        CalibrationParameter(id="share_seen", parameter_type="factor", min_bound=0.01, max_bound=1.0),
    ],
    optimization_config=NelderMeadConfig(max_iterations=2000),
)
scaled_result = Calibrator(Simulation(model(0.4)), problem).run()
print(round(scaled_result.best_parameters["share_seen"], 3))  # 0.3
```

| Fact | Detail |
|---|---|
| Several factors | Each observation names its own `factor_id`, so outputs can have different shares. |
| Not part of the model | `apply_calibration_parameters` skips `factor` values. |
| Plots | Pass `calibration_result=` to `SimulationPlotter.plot_series`; it applies the calibrated factors to the comparison. |
| Identifiability | A rate and a share can explain the same data. Fit them together only when the data can tell them apart: [Identifiability](uncertainty/identifiability.md) checks it. |

## Counts per period: windows

Data often counts what happened **in** each period, such as the amount that
entered a bin in each interval of 7 steps. Observe a cumulative output (an accumulator) with `window`, and the
model's prediction becomes the gain over the window:

```text
predicted(at) = output(at) - output(at - window)
```

```python
from commol import window_ends, windowed_totals

ends = window_ends(window=7, end=70)
counts = windowed_totals(truth["cum_ab"], truth.positions, 7, ends)

windowed = [
    ObservedDataPoint(at=end, output="cum_ab", value=count, window=7)
    for end, count in zip(ends, counts)
]
print(calibrate_k1(windowed))  # 0.4
```

| Rule | Detail |
|---|---|
| `window <= at` | Otherwise the observation is rejected when it is created. |
| Any output | Usually an accumulator, but any compartment works. |
| Mixed windows | Observations may use different windows, and positions need not be multiples of the window. |
| Combinations | Works with `factor_id` and `outputs`. |

`windowed_totals` (module function) and `Simulation.windowed_totals` apply the
same reduction outside calibration, so a report shows exactly what the loss
compared. See [Windowed totals](simulations.md#windowed-totals).

## Positions between integers

A `DifferentialEquations` model runs in continuous time, so an observation may
be at any position, such as `2.5`, and a window may cover any length. The
calibration reads the engine at those positions
(`SimulationEngine::run_at_positions_into_buffer`). The engine decides which
positions its model type accepts (`SimulationEngine::check_position`): a
`DifferenceEquations` model reports only at integer steps, so `Calibrator`
raises `ValueError` for any other position, and the engine's message names the
observation, or the constraint, that holds it (`check_positions` in
`crates/commol-calibration/src/calibration_problem.rs`).

```python
def ode_model(k1: float | None):
    return (
        ModelBuilder(name="Observed in continuous time")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_parameter(id="k1", value=k1)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
        .build(
            Aggregate(
                total=1000,
                bin_fractions={"A": 1.0, "B": 0.0},
            ),
            engine="DifferentialEquations",
        )
    )

positions = [0.5, 1.75, 3.25, 6.5]
observed_b = Simulation(ode_model(0.2)).run_at(positions)["B"]
problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(at=at, output="B", value=value)
        for at, value in zip(positions, observed_b)
    ],
    parameters=[CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0)],
    optimization_config=NelderMeadConfig(max_iterations=500),
)
print(round(Calibrator(Simulation(ode_model(None)), problem).run().best_parameters["k1"], 3))  # 0.2
```

## Stratified models

Observe expanded outputs by name, such as `B_g1`, or sum them with
`outputs`. `Simulation.outputs_for("B")` lists the names.

## Checklist

- [ ] Every `output` (and every entry of `outputs`) is an output name; a sum has a `label`.
- [ ] Weights say how much each observation counts, in every loss.
- [ ] Every `factor_id` has a `CalibrationParameter` with `parameter_type="factor"`.
- [ ] Windowed observations use a cumulative output and `window <= at`.

## Next steps

| To learn | Read |
|---|---|
| Tuning the optimizer | [Optimizers](calibration-optimizers.md) |
| Reports of what was compared | [Reports](reports.md) |
