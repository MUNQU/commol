# Observations

[Back to Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)

An `ObservedDataPoint` says what the model must reproduce: an output at a step. This page shows how to observe several outputs, sums of outputs, partial observations (scale factors), counts per period (windows) and steps between integers, and how to weigh them.

## Quick path

```python
from commol import (
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
    ObservedDataPoint(step=step, compartment=output, value=truth[output][step])
    for step in range(10, 51, 10)
    for output in ("B", "C")
]
print(calibrate_k1(observed))  # 0.4
```

Each observation compares the model output named in `compartment` with `value`. Any output works: a compartment (`B`, `B_g1`) or an accumulator (`cum_ab`).

## Sums of outputs

When one measurement covers several outputs, list them in `compartments`. They are summed before the comparison; `compartment` becomes a label for plots and reports:

```python
b_and_c = [
    ObservedDataPoint(
        step=step,
        compartment="B_plus_C",
        compartments=["B", "C"],
        value=truth["B"][step] + truth["C"][step],
    )
    for step in range(10, 51, 10)
]
print(calibrate_k1(b_and_c))  # 0.4
```

`compartments` may name compartments and accumulators, and works with `scale_id` and `window_steps`. It must not be empty.

## Weights and normalization

| Option                        | Where                | Effect                                                                                                                                                                                    |
| ----------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `weight=2.0`                  | `ObservedDataPoint`  | The observation counts twice as much in every loss: its squared (or absolute) residual is multiplied by 2. See [Losses](https://munqu.github.io/commol/latest/guide/calibration/#losses). |
| `normalize_observations=True` | `CalibrationProblem` | Divides each residual by the RMS of its series, so a series of thousands does not drown a series of tens.                                                                                 |

```python
weighted = [
    ObservedDataPoint(step=10, compartment="B", value=truth["B"][10], weight=2.0),
    ObservedDataPoint(step=30, compartment="B", value=truth["B"][30], weight=1.0),
    ObservedDataPoint(step=30, compartment="cum_ab", value=truth["cum_ab"][30], weight=1.0),
]
print(calibrate_k1(weighted, loss_function="sse", normalize_observations=True))  # 0.4
```

## Partial observations: scale factors

When the data sees only part of an output, for example a fixed share of it, a `scale` unknown multiplies the prediction before the comparison. Link observations to it with `scale_id`:

```python
seen = 0.3
partial = [
    ObservedDataPoint(step=step, compartment="B", value=seen * truth["B"][step], scale_id="share_seen")
    for step in range(5, 51, 5)
]

problem = CalibrationProblem(
    observed_data=partial,
    parameters=[
        CalibrationParameter(id="share_seen", parameter_type="scale", min_bound=0.01, max_bound=1.0),
    ],
    optimization_config=NelderMeadConfig(max_iterations=2000),
)
scaled_result = Calibrator(Simulation(model(0.4)), problem).run()
print(round(scaled_result.best_parameters["share_seen"], 3))  # 0.3
```

| Fact                  | Detail                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Several scale factors | Each observation names its own `scale_id`, so outputs can have different shares.                                    |
| Not part of the model | `apply_calibration_parameters` skips `scale` values.                                                                |
| Plots                 | Pass `calibration_result=` to `SimulationPlotter.plot_series`; it applies the calibrated factors to the comparison. |
| Identifiability       | A rate and a share can explain the same data. Fit them together only when the data can tell them apart.             |

## Counts per period: windows

Data often counts what happened **in** each period, such as new arrivals per week. Observe a cumulative output (an accumulator) with `window_steps`, and the model's prediction becomes the gain over the window:

```text
predicted(step) = output(step) - output(step - window_steps)
```

```python
from commol import windowed_totals

weekly_steps = list(range(7, 71, 7))
weekly_counts = windowed_totals(truth["cum_ab"], 7, weekly_steps)

weekly = [
    ObservedDataPoint(step=step, compartment="cum_ab", value=count, window_steps=7)
    for step, count in zip(weekly_steps, weekly_counts)
]
print(calibrate_k1(weekly))  # 0.4
```

| Rule                   | Detail                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| `window_steps <= step` | Otherwise the observation is rejected when it is created.                                     |
| Any output             | Usually an accumulator, but any compartment works.                                            |
| Mixed windows          | Observations may use different `window_steps`, and steps need not be multiples of the window. |
| Combinations           | Works with `scale_id` and `compartments`.                                                     |

`windowed_totals` (module function) and `Simulation.windowed_totals` apply the same reduction outside calibration, so a report shows exactly what the loss compared. See [Windowed totals](https://munqu.github.io/commol/latest/guide/simulations/#windowed-totals).

## Steps between integers

A `DifferentialEquations` model runs in continuous time, so an observation may be at any step, such as `2.5`, and a window may cover any length. The calibration reads the engine at those positions (`SimulationEngine::run_at_positions_into_buffer`). A `DifferenceEquations` model reports only at integer steps: `Calibrator` raises `ValueError`, naming the observation or the constraint, for a step that is not an integer (`check_steps_for_typology` in `py-commol/commol/context/calibration.py`).

```python
def ode_model(k1: float | None):
    return (
        ModelBuilder(name="Observed in continuous time")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_parameter(id="k1", value=k1)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
        )
        .build(typology="DifferentialEquations")
    )

steps = [0.5, 1.75, 3.25, 6.5]
observed_b = Simulation(ode_model(0.2)).run_at(steps)["B"]
problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(step=step, compartment="B", value=value)
        for step, value in zip(steps, observed_b)
    ],
    parameters=[CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0)],
    optimization_config=NelderMeadConfig(max_iterations=500),
)
print(round(Calibrator(Simulation(ode_model(None)), problem).run().best_parameters["k1"], 3))  # 0.2
```

A step that is a whole number is stored as an `int`, so `step=3.0` and `step=3` are the same observation.

## Stratified models

Observe expanded outputs by name, such as `B_g1`, or sum them with `compartments`. `Simulation.outputs_for("B")` lists the names.

## Checklist

- [ ] Every `compartment` (and every entry of `compartments`) is an output name.
- [ ] Weights say how much each observation counts, in every loss.
- [ ] Every `scale_id` has a `CalibrationParameter` with `parameter_type="scale"`.
- [ ] Windowed observations use a cumulative output and `window_steps <= step`.

## Next steps

| To learn                     | Read                                                                                      |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| Tuning the optimizer         | [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/index.md) |
| Reports of what was compared | [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md)                   |
