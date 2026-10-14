# Running Simulations

[Back to Home](https://munqu.github.io/commol/latest/index.md)

`Simulation` runs a built `Model` in the Rust engine and returns its outputs: one series per compartment and per accumulator, with the positions on the axis they were read at. This page covers running a model, reading and aggregating its outputs, running it in continuous time as differential equations, and plotting them.

## Quick path

```python
from commol import ModelBuilder, Simulation

model = (
    ModelBuilder(name="Basic Model")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_accumulator(id="cum_ab", name="Total moved from A to B")
    .add_transition(
        id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N", accumulators=["cum_ab"]
    )
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.9},
            {"bin": "B", "fraction": 0.1},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(engine="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)

print(simulation.simulation_outputs)  # ['A', 'B', 'C', 'cum_ab']
print(f"C at step 100: {results['C'][-1]:.0f}")
```

## `Simulation`

| Member                                           | Returns                | What it does                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------ | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Simulation(model)`                              | —                      | Builds the engine for the model. A model with uncalibrated parameters can be built, but not run.                                                                                                                                                                                                                                                                             |
| `run(num_steps)`                                 | `SimulationResults`    | Runs from the initial state, with the model as it is now, and reports at the positions `0, 1, …, num_steps`. Raises `ValueError` if a parameter is still `None`.                                                                                                                                                                                                             |
| `run_at(at)`                                     | `SimulationResults`    | Runs from the initial state and reports at the given sorted positions only. The engine decides which positions its model type accepts: a `DifferentialEquations` model any non-negative position, such as `0.5`; a `DifferenceEquations` model only integer steps. Otherwise `ValueError`, with the engine's message. See [Differential equations](#differential-equations). |
| `simulation_outputs`                             | `list[str]`            | The output names, in engine order: compartments first, then accumulators.                                                                                                                                                                                                                                                                                                    |
| `outputs_for(id)`                                | `list[str]`            | The outputs of one bin or accumulator.                                                                                                                                                                                                                                                                                                                                       |
| `group_outputs(ids)`                             | `dict[str, list[str]]` | The outputs of several bins or accumulators, keyed by id.                                                                                                                                                                                                                                                                                                                    |
| `total_series(results, ids)`                     | `list[float]`          | The sum of all outputs of the given ids, at each position of the results.                                                                                                                                                                                                                                                                                                    |
| `windowed_totals(results, ids, window, at=None)` | `list[float]`          | What the given accumulators gained in each window.                                                                                                                                                                                                                                                                                                                           |
| `engine`                                         | engine                 | The underlying Rust engine, for the model as it is now: the engine `model.engine.type` names (`engine.engine_type`). `engine.step()` advances it one step; `engine.reset()` returns it to the initial state.                                                                                                                                                                 |

Every `run` starts again from the initial state, so two calls with the same model give the same result (`Simulation.run` resets the engine first). To continue a run step by step, use the engine:

```python
first = simulation.run(num_steps=10)
again = simulation.run(num_steps=10)
print(again["A"] == first["A"])  # True: it started again

simulation.engine.step()  # step 11, from where the last run stopped
print(simulation.engine.population[0] < first["A"][-1])  # True: A keeps falling
```

The engine follows the model. When the model changed after the `Simulation` was created, for example with `update_parameters` or `apply_calibration_parameters`, `run` and `engine` rebuild the engine from the model as it is now (`Simulation._sync_engine` compares `model_dump_json()` with the JSON the engine was built from):

```python
model.update_parameters({"k1": 0.0})
frozen = simulation.run(num_steps=10)
print(frozen["A"][10] == frozen["A"][0])  # True: k1 = 0 stops A → B
model.update_parameters({"k1": 0.3})
```

## Results

`run` and `run_at` return `SimulationResults` (`py-commol/commol/api/results.py`): a `dict` from output name to one value per position, so `results["A"]`, `json.dumps(results)` and `pandas.DataFrame(results)` work as for any mapping. The positions the values were read at are in `results.positions`.

| Member              | Shape                                                                 | Use for                                         |
| ------------------- | --------------------------------------------------------------------- | ----------------------------------------------- |
| `results[name]`     | `[value at positions[0], …]`                                          | Reading one output, plotting                    |
| `results.positions` | `[0.0, 1.0, …, n]` for `run(n)`; the requested positions for `run_at` | Reading a value at a position, windows, reports |
| `results.rows()`    | `[[every output at positions[0]], …]`, in `simulation_outputs` order  | Matrix operations, export                       |

`run(num_steps)` holds `num_steps + 1` positions: position 0 is the initial state.

```python
rows = results.rows()

print(results.positions[:3])  # [0.0, 1.0, 2.0]
print(len(rows))              # 101
print(len(rows[0]))           # 4: A, B, C and cum_ab
print([round(value) for value in rows[0]])  # [900, 100, 0, 0]
```

## Accumulators

An accumulator output starts at 0 and only grows: it is the total that its transitions have moved since step 0. Here `cum_ab` is everything that has moved from `A` to `B`, so `A` plus `cum_ab` stays at the initial `A`:

```python
print(round(results["A"][-1] + results["cum_ab"][-1]))  # 900
```

## Stratified outputs

With stratifications, every bin and accumulator has one output per combination of categories, named `{id}_{category}_{category}…` in the order the stratifications were declared.

```python
stratified = (
    ModelBuilder(name="Stratified")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_stratification(id="type", categories=["t1", "t2"])
    .add_accumulator(id="cum_ab", name="Total moved from A to B")
    .add_parameter(id="k", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k", accumulators=["cum_ab"])
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.5},
                    {"category": "g2", "fraction": 0.5},
                ],
            },
            {
                "stratification": "type",
                "fractions": [
                    {"category": "t1", "fraction": 0.5},
                    {"category": "t2", "fraction": 0.5},
                ],
            },
        ],
    )
    .build(engine="DifferenceEquations")
)

stratified_simulation = Simulation(stratified)
stratified_results = stratified_simulation.run(num_steps=50)

print(stratified_simulation.outputs_for("B"))
# ['B_g1_t1', 'B_g1_t2', 'B_g2_t1', 'B_g2_t2']
print(stratified_simulation.group_outputs(["A", "cum_ab"])["cum_ab"])
# ['cum_ab_g1_t1', 'cum_ab_g1_t2', 'cum_ab_g2_t1', 'cum_ab_g2_t2']
```

### Totals by bin

`total_series` sums every output of the ids you give, at each position:

```python
total_b = stratified_simulation.total_series(stratified_results, ["B"])
total_ab = stratified_simulation.total_series(stratified_results, ["A", "B"])
print(round(total_ab[-1]))  # 1000: A and B hold the whole population
```

Do not select outputs by name prefix

`[k for k in results if k.startswith("B_")]` also matches an accumulator named `B_events`, and, with conditional stratifications, outputs may have fewer suffixes than their siblings. `outputs_for`, `group_outputs` and `total_series` use the model structure, so neither case can go wrong.

### Totals by category

To select compartments by category, use their labels from `CompartmentalStructure.initial_state()`: each entry has `name`, `bin` and `categories`.

```python
in_g1 = [
    compartment.name
    for compartment in stratified.structure.initial_state()
    if compartment.categories.get("group") == "g1"
]
print(in_g1)  # ['A_g1_t1', 'A_g1_t2', 'B_g1_t1', 'B_g1_t2']

total_g1 = [
    sum(stratified_results[name][step] for name in in_g1)
    for step in range(len(stratified_results[in_g1[0]]))
]
print(round(total_g1[-1]))  # 500: g1 started with half of the population
```

## Windowed totals

`windowed_totals` turns a cumulative output into the amount it gained in each window of length `window` on the axis. Window `w` covers `(w · window, (w + 1) · window]`, and each value is the output at the end of the window minus the output at its start. Both must be positions of the results.

```python
per_week = stratified_simulation.windowed_totals(
    stratified_results, ["cum_ab"], window=7
)
print(len(per_week))  # 7: the complete windows of a 50-step run
```

| Argument | Meaning                                                                                                                                                                           |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ids`    | Accumulator ids; their outputs are summed first.                                                                                                                                  |
| `window` | The window length on the axis. Must be positive.                                                                                                                                  |
| `at`     | The positions where windows close. By default, every complete window of the results (`window_ends`). Pass observation positions to get the values a calibration compares against. |

The window grid and the same reduction work on any series with the module-level functions of `py-commol/commol/api/windows.py`:

| Function                                              | Returns                                                                                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `window_start(i, window)`, `window_end(i, window)`    | The positions that bound window `i`: `i · window` and `(i + 1) · window`                                                            |
| `window_index(at, window)`                            | The window that holds a position, `max(ceil(at / window) − 1, 0)`; the inverse of `window_end`. Reports place observations with it. |
| `window_ends(window, end)`                            | Every complete window end up to `end`                                                                                               |
| `windowed_totals(series, positions, window, at=None)` | The gain of a series over each window                                                                                               |

```python
from commol import window_end, window_ends, window_index, windowed_totals

print(window_ends(window=7, end=30))                       # [7, 14, 21, 28]
print(window_end(0, 7), window_index(7, 7), window_index(7.5, 7))  # 7 0 1
print(windowed_totals([0, 1, 3, 6, 10, 15, 21, 28], range(8), 3))  # [6, 15]
```

A window need not be a whole number of steps: see [Reporting at any position](#reporting-at-any-position).

## Differential equations

A model built with `engine="DifferentialEquations"` runs in continuous time. The engine (`crates/commol-differential/`) integrates `dX/dt = Σ flows(t, X)` with the ODE solver [diffsol](https://github.com/martinjrobins/diffsol). The flows are the same as in a difference model: the same transitions, rates, accumulators, outputs, calibration, reports and plots. Only the way the engine advances changes.

```python
import math

from commol import OdeSolver

ode_model = (
    ModelBuilder(name="Decay")
    .add_bin(id="A", name="A")
    .add_bin(id="B", name="B")
    .add_parameter(id="k1", value=0.1)
    .add_accumulator(id="moved", name="Moved from A to B")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1", accumulators=["moved"])
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
    )
    .build(
        engine="DifferentialEquations",
        settings=OdeSolver(method="tsit45", rtol=1e-8, atol=1e-8),
    )
)

ode = Simulation(ode_model)
at = ode.run_at([0, 0.5, 2.25])
print(round(at["A"][2], 3), round(1000 * math.exp(-0.1 * 2.25), 3))  # 798.516 798.516
print(round(ode.run(num_steps=10)["moved"][10], 1))  # 632.1: the integral of the flow
```

### Solver settings

`OdeSolver` holds the settings of the `DifferentialEquations` engine. They are part of the model (`Model.engine.settings`), so a saved model reproduces its runs. Pass them where you choose the engine, `build(engine="DifferentialEquations", settings=...)`, so one builder can build the model for each engine with its own settings; without them the defaults apply. The `DifferenceEquations` engine takes no settings: the engine rejects settings of another engine with an error that names both (`ValidationError`).

| Field    | Default    | Meaning                                                                                                                                                     |
| -------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `method` | `"tsit45"` | `"tsit45"`: explicit Runge–Kutta 5(4), for non-stiff models. `"bdf"`: backward differentiation formulas, for stiff models (rates of very different speeds). |
| `rtol`   | `1e-6`     | Relative tolerance of each state component.                                                                                                                 |
| `atol`   | `1e-6`     | Absolute tolerance of each state component, in the unit of the bins.                                                                                        |

`bdf` builds its Jacobian from a finite-difference Jacobian-vector product. It uses a dense matrix below `SPARSE_THRESHOLD` state components and a sparse one from there (`crates/commol-differential/src/solve.rs`).

### Time inside a step

On `[n, n + 1)` a rate sees `step = n` and the position `t`, which moves continuously. A rate that reads `step`, a `"pulse"` series or a pulse pattern (`AxisPattern.pulse`, `pulses`, `periodic`) is constant on its unit interval, and the engine restarts the integration at each integer where such a rate can jump. A per-capita pulse `p` on step `n` acts as the constant rate `p` for one unit: it moves `1 − e^(−p)` of its source, not `p` as in a difference model. To move a fraction at once, use an [impulse](https://munqu.github.io/commol/latest/guide/transitions/#impulses).

A `"step_function"` or `"linear"` [series](https://munqu.github.io/commol/latest/guide/building-models/#series-parameters) and the patterns `AxisPattern.interval` and `linear_ramp` read `t`, so they change at any position, and the engine restarts the integration at their points and bounds. In a difference model `t` equals `step`, so they give the same results as a rate that reads `step`.

A rate can also jump where `t` reaches a value that it compares `t` with, such as `if(t >= t_on, beta, 0)`. The engine restarts the integration there too, so the solver cannot step over a short window. This works when the value reads only numbers, `pi`, `e` and parameters with a constant or calibrated value, and `t` is one side of the comparison: write `t >= 10`, not `t - 10 >= 0`. A jump that depends on the state, such as `if(t >= A, ...)`, is not found.

`TimeScale.rate_from_mean_duration` and `rate_from_probability` give per-step probabilities for difference models. A differential model needs the continuous rate; see [Steps and physical time](https://munqu.github.io/commol/latest/guide/core-concepts/#steps-and-physical-time).

```python
from commol import AxisPattern

pulse_model = (
    ModelBuilder(name="Pulse")
    .add_bin(id="A", name="A")
    .add_bin(id="B", name="B")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate=AxisPattern.pulse(at=2, amount=0.2))
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
    )
)
ode_pulse = Simulation(pulse_model.build(engine="DifferentialEquations")).run(num_steps=3)
print(round(ode_pulse["A"][2] - ode_pulse["A"][3], 1))  # 181.3 = 1000 (1 − e^−0.2)
```

### Reporting at any position

`run_at(at)` reports at sorted positions on the axis, from the dense output of the solver. Observations (`ObservedDataPoint.at`), windows (`ObservedDataPoint.window`) and constraint positions (`CalibrationConstraint.at`) may lie between integers for a differential model: the calibration reads the engine at those positions ([Observations](https://munqu.github.io/commol/latest/guide/calibration-observations/#positions-between-integers)). A difference model reports only at integer steps; the engine rejects any other position and names the object that holds it.

Windows, reports and plots read a value at a position of the results, so a window of `2.5` works once the results hold its ends:

```python
ends = window_ends(window=2.5, end=10)    # [2.5, 5.0, 7.5, 10.0]
ode_windows = ode.run_at([0.0, *ends])
gained = ode.windowed_totals(ode_windows, ["moved"], window=2.5)
print([round(value, 1) for value in gained])  # [221.2, 172.3, 134.2, 104.5]
```

## Plotting

`SimulationPlotter(simulation, results)` draws one subplot per output with seaborn, and returns the matplotlib `Figure`:

```python
from commol import PlotConfig, SimulationPlotter

plotter = SimulationPlotter(simulation, results)  # the results of the first run
plotter.plot_series(output_file="series.png")
plotter.plot_cumulative(output_file="cumulative.png")

plotter.plot_series(
    output_file="custom.png",
    config=PlotConfig(figsize=(16, 10), dpi=150, style="whitegrid", palette="Set2", context="talk"),
    series=["B", "C"],
    x_label="Week",
    tick_every=10,
    position_label=lambda at: f"w{int(at) // 7}",
    linewidth=2.5,
    alpha=0.8,
)
```

| Argument                                  | Default                                        | Meaning                                                                                                                                                                                                                                                                                              |
| ----------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_file`                             | `None`                                         | Where to save the figure. `None` only returns it.                                                                                                                                                                                                                                                    |
| `series`                                  | `None` (every output)                          | What to plot, one subplot each: an output name (`B_g1`) or a bin or accumulator id (`B`), which plots the sum of its outputs (`Simulation.outputs_for`). An exact output name wins. An unknown entry raises `ValueError`.                                                                            |
| `observed_data`                           | `None`                                         | `ObservedDataPoint`s to draw on the subplot of their `label`, at their positions (a differential model may observe between integers). An observation with a `window` draws the windowed line, which reads the results at every window end and start (`ValueError` when the results do not hold one). |
| `calibration_result`                      | `None`                                         | With a probabilistic result, draws the median and the confidence band; factor parameters are applied to the observations.                                                                                                                                                                            |
| `config`                                  | `PlotConfig()`                                 | Figure size, dpi, layout and seaborn style.                                                                                                                                                                                                                                                          |
| `show_legend`                             | `True`                                         | `plot_series` only.                                                                                                                                                                                                                                                                                  |
| `x_label`, `tick_every`, `position_label` | `"Step"`, every plotted position, the position | `plot_series` only. Label the x axis, for example in weeks. The x axis shows `results.positions`.                                                                                                                                                                                                    |
| other keyword arguments                   | —                                              | Passed to `seaborn.lineplot`, such as `linewidth` or `alpha`.                                                                                                                                                                                                                                        |

| `PlotConfig` field | Default            | Values                                                      |
| ------------------ | ------------------ | ----------------------------------------------------------- |
| `figsize`          | `(12, 8)`          | Inches (width, height)                                      |
| `dpi`              | `100`              | Resolution                                                  |
| `layout`           | `None` (automatic) | `(rows, columns)`                                           |
| `style`            | `None`             | `"darkgrid"`, `"whitegrid"`, `"dark"`, `"white"`, `"ticks"` |
| `palette`          | `None`             | A seaborn palette or matplotlib colormap, such as `"Set2"`  |
| `context`          | `None`             | `"paper"`, `"notebook"`, `"talk"`, `"poster"`               |

To compare with observations:

```python
from commol import ObservedDataPoint

observed = [
    ObservedDataPoint(at=10, output="B", value=45.2),
    ObservedDataPoint(at=20, output="B", value=78.5),
]
plotter.plot_series(output_file="with_data.png", observed_data=observed)
```

## Next steps

| To learn                | Read                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| Fitting a model to data | [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)      |
| Complete models         | [Examples](https://munqu.github.io/commol/latest/guide/examples/index.md)                  |
| Signatures              | [API reference: Simulation](https://munqu.github.io/commol/latest/api/simulation/index.md) |
