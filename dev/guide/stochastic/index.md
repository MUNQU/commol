# Stochastic Simulation

[Back to Home](https://munqu.github.io/commol/latest/index.md)

A stochastic engine runs the process of a model as whole units that move at random, with no change to the process: the rates of the model become the probabilities or the hazards of its events. Use it when the variability between runs, an extinction or the behaviour of small amounts matters. This page covers the two stochastic engines of an aggregate model, how a rate becomes an event, whole units, the methods and their cost, and calibration with replicates.

## Quick path

```python
from commol import Aggregate, ModelBuilder, Simulation

builder = (
    ModelBuilder(name="Stochastic Model")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
)
structure = Aggregate(bin_amounts={"A": 999, "B": 1, "C": 0})

model = builder.build(structure, engine="StochasticEvents")
simulation = Simulation(model)

one_run = simulation.run(num_steps=100, seed=1)
print(one_run["C"][-1] == int(one_run["C"][-1]))  # True: whole units

replicates = simulation.replicates(100, seed=2).run(num_steps=100)
print(len(replicates))  # 100 runs, one seed each
died_out = sum(run["C"][-1] < 10 for run in replicates)
print(died_out)  # about a third of the runs: the one unit of B leaves before it spreads
```

The same `builder` and `structure` build the deterministic model with `engine="DifferentialEquations"`: only the engine changes.

## When to use it

| You need                                                            | Engine                                           |
| ------------------------------------------------------------------- | ------------------------------------------------ |
| The mean behaviour of a large amount                                | `DifferentialEquations` or `DifferenceEquations` |
| How much runs vary, and the spread of an output                     | `StochasticEvents` or `StochasticSteps`          |
| The chance that an amount reaches zero and stays there (extinction) | `StochasticEvents` or `StochasticSteps`          |
| Small amounts, where a unit more or less changes the outcome        | `StochasticEvents` or `StochasticSteps`          |

## The two engines

| Engine             | Time           | Method                                                                               | Rate meaning                                | Positions                         | Crate               |
| ------------------ | -------------- | ------------------------------------------------------------------------------------ | ------------------------------------------- | --------------------------------- | ------------------- |
| `StochasticEvents` | Continuous     | Gillespie's direct method (the stochastic simulation algorithm, SSA), or tau-leaping | `continuous`: a hazard per unit of the axis | Any finite, non-negative position | `commol-stochastic` |
| `StochasticSteps`  | Discrete steps | The chain binomial                                                                   | `per_step`: a probability per step          | Integer steps                     | `commol-stochastic` |

Their mean-field limits are the deterministic engines: the expected change of the state of `StochasticEvents` is the right-hand side of `DifferentialEquations`, and the expected step of `StochasticSteps` is the step of `DifferenceEquations` while no cap acts (see below). A rate keeps its meaning: a `per_step` rate of a difference model is the probability of the chain binomial, and a `continuous` rate of a differential model is the hazard of the events ([What a rate means](https://munqu.github.io/commol/latest/guide/core-concepts/#what-a-rate-means)).

## How a rate becomes an event

Each flow of the model is one kind of event. An event moves one unit: it takes one from each source of the flow, and adds one to each target and to each accumulator that counts the flow (`CompiledModel::flow_changes` in `crates/commol-model/src/system.rs`).

| Rate                    | `StochasticEvents` (propensity)                                       | `StochasticSteps` (draw of a step)                           |
| ----------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------ |
| Proportional, `k`       | Each unit of the source leaves at hazard `k`: propensity `k × source` | Each unit leaves with probability `k`: `Binomial(source, k)` |
| Absolute, `r`           | Events at rate `r`                                                    | `Poisson(r)`, capped by what the sources still hold          |
| No source (inflow), `r` | Events at rate `r`                                                    | `Poisson(r)`                                                 |

- `StochasticEvents`: a flow has no events while a source holds fewer units than an event takes. A negative propensity is an error that names the flow.
- `StochasticSteps`: every draw reads the state at the start of the step. The units of one compartment leave by its proportional flows in one multinomial draw (each flow with its probability, staying with `1 − Σ k`), so a unit never leaves twice and an amount never goes below zero. The absolute flows then draw in the order of the model, each capped by what its sources still hold. A negative rate, a proportional rate above 1, or a sum above 1 of the proportional rates out of one compartment raises `InfeasibleStateError`, which names the compartment, the step and the flows (`StochasticSteps::step` in `crates/commol-stochastic/src/steps.rs`).

```python
from commol import InfeasibleStateError

too_likely = (
    ModelBuilder(name="Too likely")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.7)
    .add_parameter(id="k2", value=0.5)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .add_transition(id="t_ac", source=["A"], target=["C"], rate="k2")
    .build(structure, engine="StochasticSteps")
)
try:
    Simulation(too_likely).run(num_steps=5, seed=1)
except InfeasibleStateError as error:
    print(error)  # At step 0, the proportional rates out of compartment 'A': flows 't_ab' 0.7, 't_ac' 0.5 sum to 1.2, above 1. ...
```

`Model.describe()` shows the events in its `dynamics` section: one row per flow with its change of the state and its propensity or its draw ([Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md)):

```python
print(model.describe().render("markdown").split("### Events")[1].split("\n\n")[1])
# | Transition | Change | Propensity |
# |---|---|---|
# | t_ab | A -1, B +1 | k1 * A * B / total |
# | t_bc | B -1, C +1 | k2 * B |
```

## Whole units

A stochastic engine moves whole units, so it starts from whole amounts (`check_model` in `crates/commol-model/src/builder.rs`):

| Initial amounts                                                                      | What happens                                                                                             |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| The amount of a bin is not a whole number                                            | `build` raises `ValueError` and names the engine and the bin                                             |
| Every amount is whole                                                                | Nothing more is needed                                                                                   |
| The amount of a compartment is not whole (a bin amount times its category fractions) | `build` raises `ValueError`, unless the settings of the engine name a `rounding`; the message lists both |

There is no default rounding: with small amounts the rounding decides which compartments start with a unit, so it is a modelling choice. With half a unit in each of ten groups, five groups start with one unit and the others empty.

| `rounding`            | Start of each run                                                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"largest_remainder"` | The same whole amounts in every run: in each bin, each amount rounded down, and the units left over to the largest remainders, so they sum to the amount of the bin |
| `"random"`            | In each run, the amount of each bin split by a multinomial draw with the fractions of the model, so the mean start equals the amounts of the model                  |

Both round each bin on its own: no unit goes from one bin to another, such as from a bin of a concentration to a bin of a count.

```python
from commol import StochasticEventsSettings

groups = (
    ModelBuilder(name="Groups")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=[f"g{i}" for i in range(10)])
    .add_parameter(id="k1", value=0.3)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * B / total")
)
seeded = Aggregate(
    bin_amounts={"A": 995, "B": 5},
    stratification_fractions={"group": {f"g{i}": 0.1 for i in range(10)}},
)
try:
    groups.build(seeded, engine="StochasticEvents")
except ValueError as error:
    print("largest_remainder" in str(error), "random" in str(error))  # True True

grouped = groups.build(
    seeded,
    engine="StochasticEvents",
    settings=StochasticEventsSettings(rounding="random"),
)
start = Simulation(grouped).run_at([0.0], seed=3)
print(sum(start[f"B_g{i}"][0] for i in range(10)) + sum(start[f"A_g{i}"][0] for i in range(10)))
# 1000.0
```

Initial conditions that calibration sets go through the same rounding. With no rounding, a calibrated value that is not whole is an error that lists the two options.

## Seeds and replicates

A run is reproducible from its model and its seed: `simulation.run(…, seed=s)` gives one run, and `simulation.replicates(count, seed=s).run(…)` gives `count` runs with one seed each, in parallel. A run does not depend on the positions it reports at: the output at a position is the state after the last event before it. See [Randomness, seeds and replicates](https://munqu.github.io/commol/latest/guide/simulations/#randomness-seeds-and-replicates) for the seeds, `SimulationReplicates` and its statistics, plots of the mean and a band, and reports. The draws of a sample that the uncertainty methods run follow a tree of these seeds ([The predictive](https://munqu.github.io/commol/latest/guide/uncertainty/#the-predictive)).

```python
again = simulation.run(num_steps=100, seed=1)
print(again == one_run)  # True
print(replicates.member_statistics("C", confidence_level=0.9)["ci_lower"][-1] >= 0)  # True
```

## Methods and their cost

`StochasticEvents` takes `StochasticEventsSettings`, and `StochasticSteps` takes `StochasticStepsSettings` (only `rounding`):

| Setting     | Default    | Meaning                                                                                                                      |
| ----------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `method`    | `"direct"` | `"direct"`: one event at a time, exact. `"tau_leaping"`: a Poisson number of events of each flow in a leap, an approximation |
| `tau`       | `None`     | The length of a leap of `"tau_leaping"`; `None` chooses each leap as Cao, Gillespie and Petzold (2006) do                    |
| `epsilon`   | `0.03`     | How much a propensity may change in a chosen leap, relative to its value                                                     |
| `tolerance` | `1e-8`     | Relative tolerance of the time of an event when a rate reads `t`                                                             |
| `rounding`  | `None`     | See [Whole units](#whole-units)                                                                                              |

- **Cost of the direct method.** The number of events grows with the total and with the rates: every unit that moves is one event. For a large total, use `"tau_leaping"` or a deterministic engine.
- **Tau-leaping.** A chosen leap is as long as the propensities allow while they change by less than `epsilon`; a leap that would hold fewer than ten events is one exact event instead, and a leap that leaves an amount below zero is halved and drawn again. A fixed `tau` is first-order accurate: a smaller `tau` is closer to the direct method.
- **Rates that read `t`.** Between events the state is fixed, but a rate that reads `t`, such as `k * (1 + sin(t))`, still changes. The direct method then integrates the total propensity from the last event until it reaches an exponential draw (a time change, Anderson 2007), within `tolerance`. A model whose rates do not read `t` pays nothing for it.
- **Breakpoints.** The engine stops at every position where a rate can jump (the impulses, the values a rate compares `t` with, the points of a series read at `t`, and every integer when a rate reads `step`), so no wait and no leap crosses one (`CompiledModel::breakpoints`).
- **Impulses.** At its positions, before the events there, a proportional impulse moves a binomial draw of its source (its rate is the probability that one unit moves, between 0 and 1); an absolute impulse moves its rate, a whole number of units. Each moves at most what its sources still hold.
- **A new seed between two steps.** `simulation.engine.set_seed(seed)` acts from the current position: the state there stays, and every event after it comes from the new seed. A wait that was pending is drawn again from the position, and a pending leap keeps only its events up to the position.

```python
leaping = builder.build(
    structure,
    engine="StochasticEvents",
    settings=StochasticEventsSettings(method="tau_leaping"),
)
leaped = Simulation(leaping).replicates(100, seed=2).run(num_steps=100)
print(abs(leaped.mean()["C"][-1] - replicates.mean()["C"][-1]) < 150)  # True
```

## Calibration with replicates

A calibration of a stochastic model runs `replicates` runs per evaluation with common random numbers, and compares the observations with the summary of the replicates you choose: `"mean"`, `"median"`, `Quantile(q)` or `"crps"` ([Stochastic engines](https://munqu.github.io/commol/latest/guide/calibration/#stochastic-engines)).

```python
from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
)

mean = replicates.mean()
observed = [
    ObservedDataPoint(at=at, output="C", value=mean["C"][at]) for at in (40, 70, 100)
]
unknown = builder.build(structure, engine="StochasticEvents")
unknown.update_parameters({"k2": None})
problem = CalibrationProblem(
    observed_data=observed,
    parameters=[
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.01, max_bound=0.5)
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=100),
    replicates=50,
    replicate_summary="mean",
    seed=7,
)
result = Calibrator(Simulation(unknown), problem).run()
print(0.05 < result.best_parameters["k2"] < 0.2)  # True: near 0.1
```

## Checklist

- [ ] The engine is `StochasticEvents` (continuous time) or `StochasticSteps` (per-step probabilities), as the rates of the model mean.
- [ ] The amount of each bin is whole, and the settings name a `rounding` when the amount of a compartment is not.
- [ ] Every run has a seed, or keeps the one commol drew (`results.seed`).
- [ ] A large total uses `"tau_leaping"` or a deterministic engine.
- [ ] A calibration sets `replicates`, `replicate_summary` and `seed`.

## Next steps

| To learn                             | Read                                                                                                            |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Seeds, replicates, plots and reports | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/#randomness-seeds-and-replicates) |
| Calibration of a stochastic model    | [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/#stochastic-engines)                |
| Signatures                           | [API reference: Simulation](https://munqu.github.io/commol/latest/api/simulation/index.md)                      |
