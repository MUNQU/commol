# Running Simulations

[Back to Home](../index.md)

`Simulation` runs a built `Model` in the Rust engine and returns its outputs:
one series per compartment and per accumulator, with the positions on the axis
they were read at. This page covers running a model, reading and aggregating
its outputs, running it in continuous time as differential equations, and
plotting them.

## Quick path

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Basic Model")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_accumulator(id="cum_ab", name="Total moved from A to B")
    .add_transition(
        id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total", accumulators=["cum_ab"]
    )
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .build(
        Aggregate(
            bin_amounts={"A": 900, "B": 100, "C": 0},
        ),
        engine="DifferenceEquations",
    )
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
| `run(num_steps, *, seed=None)` | `SimulationResults` | Runs from the initial state, with the model as it is now, and reports at the positions `0, 1, …, num_steps`. Raises `ValueError` if a parameter or a bin amount is still `None`, with the engine's message that names each of them (`CompiledModel::check_values` in `crates/commol-model/src/system.rs`), `InfeasibleStateError` if a difference step leaves an amount below zero ([Amounts below zero](transitions.md#amounts-below-zero)), and `RuntimeError` that names the flow and the position if a rate is not a number. |
| `run_at(at, *, seed=None)` | `SimulationResults` | Runs from the initial state and reports at the given sorted positions only. The engine decides which positions its model type accepts: a `DifferentialEquations` or `StochasticEvents` model any non-negative position, such as `0.5`; a `DifferenceEquations` or `StochasticSteps` model only integer steps. Otherwise `ValueError`, with the engine's message. See [Differential equations](#differential-equations). |
| `replicates(count, *, seed=None)` | `ReplicateRuns` | For a stochastic engine: an object whose `run` and `run_at` return `SimulationReplicates`, `count` runs with one seed each. See [Randomness, seeds and replicates](#randomness-seeds-and-replicates). |
| `is_stochastic` | `bool` | Whether the engine draws random numbers. |
| `simulation_outputs` | `list[str]` | The output names, in engine order: compartments first, then accumulators. |
| `outputs_for(id)` | `list[str]` | The outputs of one bin or accumulator. |
| `group_outputs(ids)` | `dict[str, list[str]]` | The outputs of several bins or accumulators, keyed by id. |
| `total_series(results, ids)` | `list[float]` | The sum of the given outputs at each position of the results. A bin or accumulator id stands for all its outputs; an output name stands for itself. |
| `engine` | engine | The underlying Rust engine, for the model as it is now: the engine `model.engine.type` names (`engine.engine_type`). `engine.step()` advances it one step; `engine.reset()` returns it to the initial state. |

Every `run` starts again from the initial state, so two calls with the same
model give the same result (`Simulation.run` resets the engine first). To
continue a run step by step, use the engine:

```python
first = simulation.run(num_steps=10)
again = simulation.run(num_steps=10)
print(again["A"] == first["A"])  # True: it started again

simulation.engine.step()  # step 11, from where the last run stopped
print(simulation.engine.state[0] < first["A"][-1])  # True: A keeps falling
```

The engine follows the model. When the model changed after the
`Simulation` was created, for example with `update_parameters` or
`apply_calibration_parameters`, `run` and `engine` rebuild the engine from the
model as it is now (`Simulation._sync_engine` compares the JSON of the model
with the JSON the engine was built from):

```python
model.update_parameters({"k1": 0.0})
frozen = simulation.run(num_steps=10)
print(frozen["A"][10] == frozen["A"][0])  # True: k1 = 0 stops A → B
model.update_parameters({"k1": 0.3})
```

## Results

`run` and `run_at` return `SimulationResults` (`py-commol/commol/_simulation/results.py`):
a `dict` from output name to one value per position, so `results["A"]`,
`json.dumps(results)` and `pandas.DataFrame(results)` work as for any mapping.
The positions the values were read at are in `results.positions`.

| Member | Shape | Use for |
|---|---|---|
| `results[name]` | `[value at positions[0], …]` | Reading one output, plotting |
| `results.positions` | `[0.0, 1.0, …, n]` for `run(n)`; the requested positions for `run_at` | Reading a value at a position, windows, reports |
| `results.rows()` | `[[every output at positions[0]], …]`, in `simulation_outputs` order | Matrix operations, export |
| `results.model` | The `Model` of the run, which `run` and `run_at` set; `None` for results built by hand | The outputs of each bin and accumulator, read by reports |
| `results.windowed_totals(ids, window, at=None)` | `[gain in each window]`: what the given accumulators, or outputs of them, gained in each window ([Windowed totals](#windowed-totals)) | Comparing a run with counts per period |
| `results.sample(ids, at)` | `[sum at each position of at]`: the given outputs, such as a bin at the end of each window ([Values at positions](#values-at-positions)) | Reading a bin where an observation reads it |
| `results.report(positions, window, windows, …)` | The report of the run, as plain data ([Reports](reports.md#the-report-of-a-run)) | JSON export next to the observations |
| `results.quantities(quantities)` | `{label: value}`, the quantities of interest of the run, as a propagation reduces each run ([The quantities of one run](uncertainty/sensitivity.md#the-quantities-of-one-run)) | The numbers a run is judged on, such as the point estimate of a fit |
| `results.plot.series(…)`, `results.plot.cumulative(…)`, `results.plot.observations(…)` | A matplotlib `Figure` ([Plotting](#plotting)) | Figures of the run |

`run(num_steps)` holds `num_steps + 1` positions: position 0 is the initial
state.

```python
rows = results.rows()

print(results.positions[:3])  # [0.0, 1.0, 2.0]
print(len(rows))              # 101
print(len(rows[0]))           # 4: A, B, C and cum_ab
print([round(value) for value in rows[0]])  # [900, 100, 0, 0]
```

Two results are equal (`==`) when their values, `positions`, `seed` and
`model` are equal, a NaN equal to a NaN. A plain `dict` is not a result, so it
is not equal to one even with the same values: compare `dict(results)` to read
the values only. `SimulationReplicates` compare their runs, their seed, their
positions and their model in the same way. Results can change, so they have no
hash ([Comparing results](uncertainty/index.md#comparing-results)).

```python
print(results == dict(results), dict(results) == dict(results))  # False True
```

## Randomness, seeds and replicates

A stochastic engine draws random numbers, so its runs depend on a seed; a
deterministic engine, such as `DifferenceEquations` and
`DifferentialEquations`, gives the same run for every seed
(`EngineType.is_stochastic` in `crates/commol-core/src/types/engine.rs`).

| Call | Deterministic engine | Stochastic engine |
|---|---|---|
| `run(n, seed=s)`, `run_at(at, seed=s)` | Ignores the seed; `results.seed` is `None` | One run from seed `s`; `results.seed` is `s` |
| `run(n)` without a seed | `results.seed` is `None` | The engine draws a seed and keeps it in `results.seed`, so the run can be repeated |
| `replicates(count, seed=s).run(…)` | Raises `ValueError` that names the engine: every replicate would be equal | `count` runs, in parallel in the engine, each with its own seed; without `seed`, the engine draws one and keeps it in `replicates.seed` |

```python
print(simulation.is_stochastic)                # False: a difference engine
print(simulation.run(num_steps=5, seed=1).seed)  # None: the run depends on no seed
try:
    simulation.replicates(10)
except ValueError as error:
    print(error)  # The DifferenceEquations engine is deterministic: ...
```

Replicate `i` takes the seed derived from `seed` and `i`, the SplitMix64 of
the SplitMix64 of `seed` plus `i` (`replicate_seed` in
`crates/commol-random/src/generator.rs`), so two seeds share no replicate, a
replicate gives the same run however many replicates run, and
`simulation.run(…, seed=replicates[i].seed)` repeats it. `SimulationReplicates`
(`py-commol/commol/_simulation/results.py`) is a list of `SimulationResults`:

| Member | Returns |
|---|---|
| `replicates[i]`, `len(replicates)` | One run, with its `seed`; the number of runs |
| `seed`, `seeds`, `positions`, `model` | The seed the replicate seeds derive from, the seed of each replicate, the positions, the `Model` of the replicates |
| `mean()` | `SimulationResults` of the mean of every output at each position |
| `quantiles(levels)` | `{level: SimulationResults}`, at each position, by [the one quantile](../codebase/decisions/results.md#one-confidence-interval-convention) (NumPy's `method="hazen"`) |
| `member_statistics(output, confidence_level)` | `mean`, `median`, `ci_lower`, `ci_upper`, `min`, `max` of one output at each position, reduced replicate by replicate with [the one quantile](../codebase/decisions/results.md#one-confidence-interval-convention) |
| `report(confidence_level, positions, window, windows, …)` | The report of a run, with each number reduced replicate by replicate ([Reports](reports.md#the-report-of-an-ensemble)) |
| `quantities(quantities)` | One `{label: value}` per replicate, in seed order, with no mean ([The quantities of one run](uncertainty/sensitivity.md#the-quantities-of-one-run)) |
| `plot.series(…)`, `plot.cumulative(…)`, `plot.observations(…)` | The figure of the mean and of a band of the replicates, the figure of the running total of the mean, and the figure of each observed series ([Plotting](#plotting)) |

```python
stochastic = Simulation(
    ModelBuilder(name="Stochastic Model")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k")
    .build(Aggregate(bin_amounts={"A": 1000, "B": 0}), engine="StochasticEvents")
)
replicates = stochastic.replicates(200, seed=42).run(num_steps=50)
mean = replicates.mean()
band = replicates.member_statistics("B", confidence_level=0.9)
replicates.plot.series()  # the mean and a 95 % band
print(replicates.seeds[:2] == [run.seed for run in replicates[:2]])  # True
```

The stochastic engines and what they draw are in
[Stochastic Simulation](stochastic.md).

The model and the seed of a run repeat it on the same platform with the same
commol version:

| Where | The same seed gives |
|---|---|
| The same machine and commol version | The same run |
| Another platform, the same commol version | The same random numbers (`Xoshiro256PlusPlus` and samplers written in commol, `crates/commol-random/`), but the rate functions (`exp`, `sin`, `log`, …) come from the system's math library, which can differ in the last bit between operating systems. The runs have the same distribution, and a draw seldom differs |
| Another commol version | A run that may change; the changelog says so |

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
    .build(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0},
            stratification_fractions={
                "group": {"g1": 0.5, "g2": 0.5},
                "type": {"t1": 0.5, "t2": 0.5},
            },
        ),
        engine="DifferenceEquations",
    )
)

stratified_simulation = Simulation(stratified)
stratified_results = stratified_simulation.run(num_steps=50)

print(stratified_simulation.outputs_for("B"))
# ['B_g1_t1', 'B_g1_t2', 'B_g2_t1', 'B_g2_t2']
print(stratified_simulation.group_outputs(["A", "cum_ab"])["cum_ab"])
# ['cum_ab_g1_t1', 'cum_ab_g1_t2', 'cum_ab_g2_t1', 'cum_ab_g2_t2']
```

### Totals by bin

`total_series` sums every output of the bin and accumulator ids you give, at
each position:

```python
total_b = stratified_simulation.total_series(stratified_results, ["B"])
total_ab = stratified_simulation.total_series(stratified_results, ["A", "B"])
print(round(total_ab[-1]))  # 1000: A and B hold the whole total
```

!!! warning "Do not select outputs by name prefix"
    `[k for k in results if k.startswith("B_")]` also matches an accumulator
    named `B_events`, and, with conditional stratifications, outputs may have
    fewer suffixes than their siblings. `outputs_for`, `group_outputs` and
    `total_series` use the model structure, so neither case can go wrong.

### Totals by category

`total_series` also takes output names, and each one stands for itself. To
select compartments by category, use their labels from
`model.structure.initial_state(model.process)`: each entry has `name`, `bin`
and `categories`.

```python
in_g1 = [
    compartment.name
    for compartment in stratified.structure.initial_state(stratified.process)
    if compartment.categories.get("group") == "g1"
]
print(in_g1)  # ['A_g1_t1', 'A_g1_t2', 'B_g1_t1', 'B_g1_t2']

total_g1 = stratified_simulation.total_series(stratified_results, in_g1)
print(round(total_g1[-1]))  # 500: g1 started with half of every bin
```

## Windowed totals

`results.windowed_totals` turns a cumulative output into the amount it gained
in each window of length `window` on the axis, the windows of the
[window grid](core-concepts.md#units-of-the-axis). Each value is the output at
the end of the window minus the output at its start. Both must be positions of
the results.

```python
per_week = stratified_results.windowed_totals(["cum_ab"], window=7)
print(len(per_week))  # 7: the complete windows of a 50-step run
```

| Argument | Meaning |
|---|---|
| `ids` | Accumulator ids, each for all its outputs, and output names, each for itself, such as some strata of an accumulator. Their outputs are summed first. An id that is neither raises `KeyError`, with the bins, accumulators and outputs of the model. Results without a model (`results.model` is `None`) read output names only. |
| `window` | The window length on the axis: a finite number greater than 0. |
| `at` | The positions where windows close. By default, every complete window of the results. Pass observation positions to get the values a calibration compares against. |

The axis of a model gives the window grid of a width in a unit
(`Model.axis_units()`; [Units of the axis](core-concepts.md#units-of-the-axis)):

```python
from commol import AxisUnits

daily = AxisUnits(step_unit="day")
print(daily.window_ends(1, "week", 30))  # [7.0, 14.0, 21.0, 28.0]
print(daily.window_end(0, 1, "week"), daily.window_index(7, 1, "week"))  # 7.0 0
```

A window of the grid is a whole number of steps. The `window` of
`windowed_totals` need not be: see
[Reporting at any position](#reporting-at-any-position). The engine computes
every window that results, reports and figures read, and checks its length
(`window_ends` and the other functions of
`crates/commol-calibration/src/observation.rs`), so a calibration and a report
read the same windows.

## Values at positions

A bin holds an amount at a position, so it is read at the position, not over
the window before it. `results.sample` reads the given bins and accumulators at
each position of `at`; the ids are the ones of `total_series`, and their
outputs are summed first. Next to `windowed_totals`, it gives the amount in a bin at the
end of each window and what an accumulator gained over that window:

```python
ends = stratified.axis_units().window_ends(7, "step", 21)  # [7.0, 14.0, 21.0]

in_a = stratified_results.sample(["A"], ends)
gained = stratified_results.windowed_totals(["cum_ab"], 7, ends)
print([round(value) for value in in_a])    # [478, 229, 109]
print([round(value) for value in gained])  # [522, 250, 119]
```

Both read a position of the results within the tolerance of the axis, and
raise `ValueError` that names a position the results do not hold.

Each position of `at` must be a position of the results, within a relative
`1e-9`; otherwise `ValueError` asks you to run the simulation there with
`run_at`.

## Differential equations

A model built with `engine="DifferentialEquations"` runs in continuous time.
The engine (`crates/commol-differential/`) integrates
`dX/dt = Σ flows(t, X)` with the ODE solver
[diffsol](https://github.com/martinjrobins/diffsol). The flows are the same as
in a difference model: the same transitions, rates, accumulators, outputs,
calibration, reports and plots. Only the way the engine advances changes.

```python
import math

from commol.engines import ODESolver

ode_model = (
    ModelBuilder(name="Decay")
    .add_bin(id="A", name="A")
    .add_bin(id="B", name="B")
    .add_parameter(id="k1", value=0.1)
    .add_accumulator(id="moved", name="Moved from A to B")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1", accumulators=["moved"])
    .build(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0},
        ),
        engine="DifferentialEquations",
        settings=ODESolver(method="tsit45", rtol=1e-8, atol=1e-8),
    )
)

ode = Simulation(ode_model)
at = ode.run_at([0, 0.5, 2.25])
print(round(at["A"][2], 3), round(1000 * math.exp(-0.1 * 2.25), 3))  # 798.516 798.516
print(round(ode.run(num_steps=10)["moved"][10], 1))  # 632.1: the integral of the flow
```

### Solver settings

`ODESolver` holds the settings of the `DifferentialEquations` engine. They
are part of the model (`Model.engine.settings`), so a saved model reproduces
its runs. Pass them where you choose the engine,
`build(engine="DifferentialEquations", settings=...)`, so one builder can
build the model for each engine with its own settings; without them the
defaults apply. The `DifferenceEquations` engine takes no settings: the
engine rejects settings of another engine with an error that names both
(`ValidationError`).

| Field | Default | Meaning |
|---|---|---|
| `method` | `"tsit45"` | `"tsit45"`: explicit Runge–Kutta 5(4), for non-stiff models. `"bdf"`: backward differentiation formulas, for stiff models (rates of very different speeds). |
| `rtol` | `1e-6` | Relative tolerance of each state component. |
| `atol` | `1e-6` | Absolute tolerance of each state component, in the unit of the bins. |

The defaults are the engine's (`commol_rs.engine.ODE_DEFAULT_RTOL` and
`ODE_DEFAULT_ATOL`), and the engine checks the settings when they are built:
`ODESolver(rtol=0.0)` raises `ValidationError` with the message of the
engine, `rtol is a finite number greater than 0, got 0`.

`bdf` builds its Jacobian from a finite-difference Jacobian-vector product. It
uses a dense matrix below `SPARSE_THRESHOLD` state components and a sparse one
from there (`crates/commol-differential/src/solve.rs`).

A run stops with `RuntimeError` when the solver cannot take a step, or when
a step leaves the state not finite, as when the solution blows up. The
message names the last position the solver reached, with either method.

### Time inside a step

On `[n, n + 1)` a rate sees `step = n` and the position `t`, which moves
continuously. A rate that reads `step`, a `"pulse"` series or a pulse pattern
(`AxisPattern.pulse`, `pulses`, `periodic`) is constant on its unit interval,
and the engine restarts the integration at each integer where such a rate can
jump. A proportional pulse `p` on step `n` acts as the constant rate `p` for one
unit: it moves `1 − e^(−p)` of its source, not `p` as in a difference model. To
move a fraction at once, use an [impulse](transitions.md#impulses).

A `"step_function"` or `"linear"` [series](building-models.md#series-parameters)
and the patterns `AxisPattern.interval` and `linear_ramp` read `t`, so they
change at any position, and the engine restarts the integration at their
points and bounds. In a difference model `t` equals `step`, so they give the
same results as a rate that reads `step`.

A rate can also jump where `t` reaches a value that it compares `t` with, such
as `if(t >= t_on, k1, 0)`. The engine restarts the integration there too, so
the solver cannot step over a short window. This works when the value reads
only numbers, `pi`, `e` and parameters with a constant or calibrated value, and
`t` is one side of the comparison: write `t >= 10`, not `t - 10 >= 0`. A jump
that depends on the state, such as `if(t >= A, ...)`, is not found.

A rate of a differential model is continuous, not a fraction per step; see
[What a rate means](core-concepts.md#what-a-rate-means).
`Model.axis_units()` gives the rate that the engine of a model reads from a mean
span or a probability over a period; see
[Units of the axis](core-concepts.md#units-of-the-axis).

```python
from commol import AxisPattern

pulse_model = (
    ModelBuilder(name="Pulse")
    .add_bin(id="A", name="A")
    .add_bin(id="B", name="B")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate=AxisPattern.pulse(at=2, value=0.2))
)
all_in_a = Aggregate(bin_amounts={"A": 1000, "B": 0})
ode_pulse = Simulation(
    pulse_model.build(all_in_a, engine="DifferentialEquations")
).run(num_steps=3)
print(round(ode_pulse["A"][2] - ode_pulse["A"][3], 1))  # 181.3 = 1000 (1 − e^−0.2)
```

### Reporting at any position

`run_at(at)` reports at sorted positions on the axis, from the dense output of
the solver. [Observations](calibration-observations.md) (`ObservedDataPoint.at`), windows
(`ObservedDataPoint.window`) and constraint positions
(`CalibrationConstraint.at`) may lie between integers for a differential
model: the calibration reads the engine at those positions
([Observations](calibration-observations.md#positions-between-integers)). A
difference model reports only at integer steps; the engine rejects any other
position and names the object that holds it.

Windows, reports and plots read a value at a position of the results, so a
window of `2.5` works once the results hold its ends:

```python
ends = [2.5, 5.0, 7.5, 10.0]
ode_windows = ode.run_at([0.0, *ends])
gained = ode_windows.windowed_totals(["moved"], window=2.5)
print([round(value, 1) for value in gained])  # [221.2, 172.3, 134.2, 104.5]
```

## Plotting

A figure is read from the results it draws. It needs matplotlib, the optional
extra `commol[plot]`:

```bash
pip install 'commol[plot]'
```

`import commol` loads no plotting library. Without matplotlib, a figure raises
`ImportError` with this install line.

| Results | Figures |
|---|---|
| `SimulationResults` | `results.plot.series(…)`: each series along the axis. `results.plot.cumulative(…)`: the running total of each series. `results.plot.observations(…)`: each [observed series](#a-panel-per-observed-series), read through its observation operator. |
| `SimulationReplicates` | `replicates.plot.series(…)`: the mean and a band of `confidence_level` (default 0.95), each replicate reduced to the series first. `replicates.plot.cumulative(…)`: the running total of the mean. `replicates.plot.observations(…)`: the mean and the band of each observed series. |
| [`Predictive`](uncertainty/index.md#the-predictive) | `predictive.plot.series(…)` and `predictive.plot.cumulative(…)`: the weighted median of the draws and their band of `confidence_level`, with no other run. `predictive.plot.observations(…)`: the median and the band of each observed series. |
| The results of [sensitivity methods](uncertainty/sensitivity.md#plots) and of [profiles](uncertainty/profiles.md#plots) | `result.plot(…)` |
| Several runs, replicates or predictives | `plot_compared({label: result}, …)`: each result on the same panels, one per series or, with `panels="observations"`, one per observed series ([compare results](#compare-results-on-the-same-panels)). |

A figure draws one panel per series and returns the matplotlib `Figure`. It
reads the name and the unit of each series from the model of the results
(`results.model`). Results built by hand have no model: their figures draw
only their outputs, and a bin or accumulator id raises `ValueError`.

```python
from commol.plotting import PlotConfig

results.plot.series(output_file="series.png")  # the results of the first run
results.plot.cumulative(output_file="cumulative.png")

results.plot.series(
    output_file="custom.png",
    config=PlotConfig(figsize=(16, 10), dpi=150, style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-talk"]),
    series=["B", "C"],
    x_label="Week",
    tick_every=10,
    position_label=lambda at: f"w{int(at) // 7}",
    linewidth=2.5,
    alpha=0.8,
)
```

| Argument | Default | Meaning |
|---|---|---|
| `output_file` | `None` | Where to save the figure. `None` only returns it. |
| `series` | `None` (every output) | What to draw, one panel each: an output name (`B_g1`) or a bin or accumulator id (`B`), which draws the sum of its outputs (`ModelStructure.outputs_by_source`). An exact output name wins. An unknown entry raises `ValueError`. |
| `observed_data` | `None` | [`ObservedDataPoint`](calibration-observations.md)s to draw on the panel of their `label`, at their positions (a differential model may observe between integers). When the observations of a series share one sum of outputs, one factor and one `scale` and have no window, the model line covers every position of the results. Otherwise the line joins the model value of each observation at its position, read through its own outputs, `window`, `scale` and factor as the loss reads it (`ValueError` when the results have no value at a position or a window start). A predictive reads each draw through the same operator. |
| `calibration_result` | `None` | Runs and replicates only. A [`CalibrationResult`](calibration.md#results) gives the values of the [factors](calibration-observations.md#partial-observations-factors) that multiply the model; without it, each factor counts as 1, with one warning for each factor. The `scale` of each observation applies in both cases. A predictive reads the factors of each draw. |
| `confidence_level` | `0.95` | Replicates and predictives only. The level of the band. |
| `prediction_intervals` | `None` | `plot.series` only. The `PredictionIntervals` of a [conformal method](uncertainty/conformal.md): the panel of each of their series, by label, shows the band from `lower` to `upper` and the point prediction, dashed. An unbounded interval is left out of the band. |
| `axes` | `None` | A `Figure` of commol to draw on ([several results on one figure](#several-results-on-one-figure)). |
| `config` | `PlotConfig()` | Figure size, dpi, layout and matplotlib style. |
| `show_legend` | `True` | `plot.series` only. |
| `x_label`, `tick_every`, `position_label` | `"Step"`, every plotted position, the position | `plot.series` only. Label the x axis, for example in weeks. The x axis shows `results.positions`. With no `position_label`, a model with an [origin](units-and-equations.md#dates-on-the-axis) labels each tick with its date (`2020-01-08`, with the time when it is not midnight): at the plotted positions `tick_every` apart, or at the ticks matplotlib places when `tick_every` is `None`. |
| other keyword arguments | — | Passed to `Axes.plot` for the line of the model, such as `linewidth` or `alpha`. |

| `PlotConfig` field | Default | Values |
|---|---|---|
| `figsize` | `(12, 8)` | Inches (width, height) of a figure of running totals. The other figures get 4.5 by 3.5 inches per panel. |
| `dpi` | `100` | Resolution |
| `layout` | `None` (automatic) | `(rows, columns)` |
| `style` | `None` | A [matplotlib style](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html) or a list of styles, applied in order: a name of `matplotlib.style.available`, such as `"ggplot"` or `"seaborn-v0_8-whitegrid"`, or the path of a style file |

`PlotConfig.presentation()`, `publication()` and `notebook()` give a size, a
dpi and styles for each use.

The style applies to the figure of one call only. Each figure draws in a copy
of `matplotlib.rcParams` (`_scoped_style` in
`py-commol/commol/plotting/figures.py`), so the figures that you draw with
matplotlib after a figure of commol keep your own style.

To compare with observations:

```python
from commol import ObservedDataPoint

observed = [
    ObservedDataPoint(at=10, output="B", value=45.2),
    ObservedDataPoint(at=20, output="B", value=78.5),
]
results.plot.series(output_file="with_data.png", observed_data=observed)
```

### A panel per observed series

`plot.observations(observed_data, …)` checks a model against its data: one
panel per `label` of the observations, in the order of its first observation.
The model is read only at the observations, through the observation operator
of each one (the sum of its `outputs`, its `window`, its `scale` and its factor): the
values the loss compares. An observed series that sums some outputs, such as
`B + C`, has its own panel, which no series of `plot.series` gives.

```python
observed = [
    ObservedDataPoint(at=at, output="cum_ab", value=value, window=7)
    for at, value in [(7, 120.0), (14, 160.0), (21, 190.0)]
] + [
    ObservedDataPoint(at=at, outputs=["B", "C"], label="B_and_C", value=value)
    for at, value in [(10, 300.0), (20, 520.0)]
] + [
    ObservedDataPoint(at=at, output="B", value=value)
    for at, value in [(10, 240.0), (20, 380.0)]
]

figure = results.plot.observations(observed, output_file="observations.png")
print([ax.get_label() for ax in figure.axes if ax.get_visible()])  # ['cum_ab', 'B_and_C', 'B']

results.plot.series(series=["B"], axes=figure)  # the run along the whole axis
```

| Drawn | How |
|---|---|
| A value with a window | A total over `(at - window, at]`: a step over its window, with a marker at `at`. Each point has its own window. |
| A value with no window | A marker at `at`. A dotted line joins these markers only as a guide. |
| The band of replicates or of a predictive | A rectangle over the window of each value with a window, and a band filled between the values with no window. Its legend says that it is a band of the model, with no observation noise, not a prediction interval. |
| The observed values | Red points. |

| Results | Model | Band |
|---|---|---|
| `SimulationResults` | The run | None |
| `SimulationReplicates` | The mean of the replicates | The interval of `confidence_level` across the replicates, each read through the operator of each observation first |
| `Predictive` | The weighted median of the draws | The band of `predictive.at_observations(observed_data, NoNoise())`, each draw with its own values of the factors |

`calibration_result`, for runs and replicates, gives the values of the
factors. The other arguments (`output_file`, `config`, `show_legend`,
`x_label`, `tick_every`, `position_label`, `confidence_level`, `axes`) are
those of `plot.series`. The other keyword arguments go to `Axes.plot` for the
steps and the guide of the model. A label that names a series, such as `B`,
also has the panel of that series: `plot.series(series=["B"], axes=figure)`
draws the run along the whole axis, after the data too. An empty
`observed_data`, or results with no value at the position of an observation
or at the start of its window, raise `ValueError`.

### Several results on one figure

Each panel holds the id of its series as the label of its `Axes`
(`ax.get_label()`). Give the `Figure` of an earlier figure as `axes=`, and a
figure draws each of its series on the panel of that series. A series with no
panel raises `ValueError` that names it. `config` then sets only the style:
the size, dpi and layout are those of the given figure.

```python
faster_model = model.model_copy(deep=True)
faster_model.update_parameters({"k1": 0.5})
faster = Simulation(faster_model).run(num_steps=100)

figure = results.plot.series(series=["B", "C"])
faster.plot.series(series=["B", "C"], axes=figure, linestyle="--")
print([ax.get_label() for ax in figure.axes if ax.get_visible()])  # ['B', 'C']
```

### Compare results on the same panels

`plot_compared({label: result}, …)` draws several results of the same model on
the same panels and returns the `Figure`. A value is a `SimulationResults`, a
`SimulationReplicates` or a [`Predictive`](uncertainty/index.md#the-predictive).
It is a function, not a method: a comparison describes no single result.

```python
from commol.plotting import plot_compared

figure = plot_compared({"k1 = 0.3": results, "k1 = 0.5": faster}, series=["B", "C"])
print([line.get_label() for line in figure.axes[0].lines])  # ['k1 = 0.3', 'k1 = 0.5']
```

| What | How it is drawn |
|---|---|
| Each result | One line per panel, labelled with its key, in the same color in every panel: `C0` of the matplotlib style for the first key, `C1` for the second, and so on. |
| Replicates | Their mean and the band of `confidence_level` of the replicates, in the color of their key. |
| A predictive | The weighted median of its draws and their band of `confidence_level`, in the color of their key. The legend of a band starts with the key: `fit: 95% of the draws`. |
| `observed_data` | Drawn once, as points on the panel of their label. Without `window`, they also set how each line is read, as in `plot.series`. |
| `window` | Each line is the gain of the series over each complete window, at the end of the window, as `SimulationResults.windowed_totals` reads it. A band is the band of the gains of the replicates or of the draws. |

The first result names the series and gives the title and the unit of each
panel. Every result has the outputs of the first; otherwise `plot_compared`
raises `ValueError` that names the key of the result and the outputs that
differ. `axes=` draws on the panels of an earlier figure, and `series`,
`config`, `show_legend`, `x_label`, `tick_every`, `position_label`,
`output_file` and other keyword arguments (for `Axes.plot`, on every line)
work as in `plot.series`.

```python
weekly = plot_compared({"k1 = 0.3": results, "k1 = 0.5": faster}, series=["cum_ab"], window=7)
print([float(at) for at in weekly.axes[0].lines[0].get_xdata()][:3])  # [7.0, 14.0, 21.0]
```

`panels="observations"` draws one panel per `label` of `observed_data`, as
[`plot.observations`](#a-panel-per-observed-series) draws one result: each
result is read only at the observations, through the observation operator of
each one (the sum of its `outputs`, its `window`, its `scale` and its factor). Each result
has the steps, markers and band of `plot.observations` in the color of its
key, and the observations are drawn once. The default, `panels="series"`,
draws one panel per entry of `series`.

```python
checked = plot_compared(
    {"k1 = 0.3": results, "k1 = 0.5": faster}, observed_data=observed, panels="observations"
)
print([ax.get_label() for ax in checked.axes if ax.get_visible()])  # ['cum_ab', 'B_and_C', 'B']
```

With `panels="observations"`, `series` and `window` raise `ValueError`: the
labels of `observed_data` set the panels, and each observation sets its own
window. An empty `observed_data`, or a result with no value at the position of
an observation or at the start of its window, also raises `ValueError`.
`plot_compared` takes no `calibration_result`: a run or replicates read an
observation with a factor as if the factor were 1, and log one warning for
each factor. A predictive reads the factors of each draw.

## Next steps

| To learn | Read |
|---|---|
| Runs at random, with whole units | [Stochastic Simulation](stochastic.md) |
| Fitting a model to data | [Model Calibration](calibration.md) |
| Complete models | [Examples](examples.md) |
| Signatures | [API reference: Simulation](../api/simulation.md) |
