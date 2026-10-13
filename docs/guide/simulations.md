# Running Simulations

[Back to Home](../index.md)

`Simulation` runs a built `Model` in the Rust engine and returns its outputs:
one series per compartment and per accumulator. This page covers running a
model, reading and aggregating its outputs, and plotting them.

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
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)

print(simulation.simulation_outputs)  # ['A', 'B', 'C', 'cum_ab']
print(f"C at step 100: {results['C'][-1]:.0f}")
```

## `Simulation`

| Member | Returns | What it does |
|---|---|---|
| `Simulation(model)` | — | Builds the engine for the model. A model with uncalibrated parameters can be built, but not run. |
| `run(num_steps, output_format="dict_of_lists")` | results | Runs `num_steps` steps from the engine's current state; see below. Raises `ValueError` if a parameter is still `None`. |
| `simulation_outputs` | `list[str]` | The output names, in engine order: compartments first, then accumulators. |
| `outputs_for(id)` | `list[str]` | The outputs of one bin or accumulator. |
| `group_outputs(ids)` | `dict[str, list[str]]` | The outputs of several bins or accumulators, keyed by id. |
| `total_series(results, ids)` | `list[float]` | The sum of all outputs of the given ids, at each step. |
| `windowed_totals(results, ids, window_steps, at_steps=None)` | `list[float]` | What the given accumulators gained in each window. |
| `engine` | engine | The underlying Rust engine. `engine.reset()` returns it to the initial state. |

!!! warning "A second `run` continues the first one"
    The engine keeps its state between calls: a second `run` starts where the
    first one stopped, and its step 0 is the previous last step. To start
    again from the initial state, call `simulation.engine.reset()` or create a
    new `Simulation`.

```python
simulation.engine.reset()
first = simulation.run(num_steps=10)
continued = simulation.run(num_steps=10)
print(continued["A"][0] == first["A"][-1])  # True: it continued

simulation.engine.reset()
again = simulation.run(num_steps=10)
print(again["A"] == first["A"])  # True: it started again
```

## Output formats

| `output_format` | Shape | Use for |
|---|---|---|
| `"dict_of_lists"` (default) | `{output name: [value at step 0, …, value at step n]}` | Reading one output, plotting |
| `"list_of_lists"` | `[[every output at step 0], …, [every output at step n]]`, in `simulation_outputs` order | Matrix operations, export |

Both contain `num_steps + 1` values per output: step 0 is the initial state.

```python
simulation.engine.reset()
rows = simulation.run(num_steps=100, output_format="list_of_lists")

print(len(rows))         # 101
print(len(rows[0]))      # 4: A, B, C and cum_ab
print([round(value) for value in rows[0]])  # [900, 100, 0, 0]
```

## Accumulators

An accumulator output starts at 0 and only grows: it is the total that its
transitions have moved since step 0. Here `cum_ab` is everything that has moved
from `A` to `B`, so `A` plus `cum_ab` stays at the initial `A`:

```python
print(round(results["A"][-1] + results["cum_ab"][-1]))  # 900
```

## Stratified outputs

With stratifications, every bin and accumulator has one output per
combination of categories, named `{id}_{category}_{category}…` in the order the
stratifications were declared.

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
    .build(typology="DifferenceEquations")
)

stratified_simulation = Simulation(stratified)
stratified_results = stratified_simulation.run(num_steps=50)

print(stratified_simulation.outputs_for("B"))
# ['B_g1_t1', 'B_g1_t2', 'B_g2_t1', 'B_g2_t2']
print(stratified_simulation.group_outputs(["A", "cum_ab"])["cum_ab"])
# ['cum_ab_g1_t1', 'cum_ab_g1_t2', 'cum_ab_g2_t1', 'cum_ab_g2_t2']
```

### Totals by bin

`total_series` sums every output of the ids you give, at each step:

```python
total_b = stratified_simulation.total_series(stratified_results, ["B"])
total_ab = stratified_simulation.total_series(stratified_results, ["A", "B"])
print(round(total_ab[-1]))  # 1000: A and B hold the whole population
```

!!! warning "Do not select outputs by name prefix"
    `[k for k in results if k.startswith("B_")]` also matches an accumulator
    named `B_events`, and, with conditional stratifications, outputs may have
    fewer suffixes than their siblings. `outputs_for`, `group_outputs` and
    `total_series` use the model structure, so neither case can go wrong.

### Totals by category

To select compartments by category, use their labels from
`Model.initial_state()`: each entry has `name`, `bin` and `categories`.

```python
in_g1 = [
    compartment.name
    for compartment in stratified.initial_state()
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

`windowed_totals` turns a cumulative output into the amount it gained in each
window of `window_steps` steps. A window covers the steps `(end − window, end]`.

```python
per_week = stratified_simulation.windowed_totals(
    stratified_results, ["cum_ab"], window_steps=7
)
print(len(per_week))  # 7: the complete windows of a 50-step run
```

| Argument | Meaning |
|---|---|
| `ids` | Accumulator ids; their outputs are summed first. |
| `window_steps` | The window length, in steps. Must be positive. |
| `at_steps` | The steps where windows close. By default, every complete window of the run. Pass observation steps to get the values a calibration compares against. |

The same reduction works on any series with the module-level functions:

```python
from commol import window_end_steps, windowed_totals

print(window_end_steps(window_steps=7, num_steps=30))        # [7, 14, 21, 28]
print(windowed_totals([0, 1, 3, 6, 10, 15, 21, 28], 3))       # [6, 15]
```

## Plotting

`SimulationPlotter(simulation, results)` draws one subplot per output with
seaborn, and returns the matplotlib `Figure`:

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
    step_to_label=lambda step: f"w{step // 7}",
    linewidth=2.5,
    alpha=0.8,
)
```

| Argument | Default | Meaning |
|---|---|---|
| `output_file` | `None` | Where to save the figure. `None` only returns it. |
| `series` | `None` (every output) | What to plot, one subplot each: an output name (`B_g1`) or a bin or accumulator id (`B`), which plots the sum of its outputs (`Simulation.outputs_for`). An exact output name wins. An unknown entry raises `ValueError`. |
| `observed_data` | `None` | `ObservedDataPoint`s to draw on the matching subplots. |
| `calibration_result` | `None` | With a probabilistic result, draws the median and the confidence band; scale parameters are applied to the observations. |
| `config` | `PlotConfig()` | Figure size, dpi, layout and seaborn style. |
| `show_legend` | `True` | `plot_series` only. |
| `x_label`, `tick_every`, `step_to_label` | `"Step"`, every step, the step | `plot_series` only. Label the x axis, for example in weeks. |
| other keyword arguments | — | Passed to `seaborn.lineplot`, such as `linewidth` or `alpha`. |

| `PlotConfig` field | Default | Values |
|---|---|---|
| `figsize` | `(12, 8)` | Inches (width, height) |
| `dpi` | `100` | Resolution |
| `layout` | `None` (automatic) | `(rows, columns)` |
| `style` | `None` | `"darkgrid"`, `"whitegrid"`, `"dark"`, `"white"`, `"ticks"` |
| `palette` | `None` | A seaborn palette or matplotlib colormap, such as `"Set2"` |
| `context` | `None` | `"paper"`, `"notebook"`, `"talk"`, `"poster"` |

To compare with observations:

```python
from commol import ObservedDataPoint

observed = [
    ObservedDataPoint(step=10, compartment="B", value=45.2),
    ObservedDataPoint(step=20, compartment="B", value=78.5),
]
plotter.plot_series(output_file="with_data.png", observed_data=observed)
```

## Next steps

| To learn | Read |
|---|---|
| Fitting a model to data | [Model Calibration](calibration.md) |
| Complete models | [Examples](examples.md) |
| Signatures | [API reference: Simulation](../api/simulation.md) |
