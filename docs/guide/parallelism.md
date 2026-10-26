# Threads and Parallelism

[Back to Home](../index.md)

commol runs its independent work in parallel in Rust: the starts of a
calibration, the replicates of a bootstrap, the particles of a swarm, the
draws of a predictive. This page is for a user who wants to choose how many
threads that work uses, and which levels of it run in parallel. By default,
commol uses one thread per core and every level in parallel; a seed gives the
same result whatever you choose.

## Quick path

```python
import commol.parallelism

commol.parallelism.set_threads(4)  # every parallel method from here on, in this thread

with commol.parallelism.threads(2):  # only inside the block
    print(commol.parallelism.get_threads(), commol.parallelism.get_levels())  # 2 all

with commol.parallelism.threads(levels="outer"):  # the particles of a swarm in order
    print(commol.parallelism.get_threads(), commol.parallelism.get_levels())  # 4 outer
```

| Name | What it does |
|---|---|
| `commol.parallelism.set_threads(n, levels=None)` | Sets the number of threads, at least 1, and optionally the levels, until the next call. |
| `commol.parallelism.threads(n=None, levels=None)` | A context manager: the setting holds inside the block and the previous one comes back after it, also on an error. `None` keeps the current value. |
| `commol.parallelism.get_threads()` | The number of threads: the one set, or the number of cores. |
| `commol.parallelism.get_levels()` | `"all"` or `"outer"`: the levels set, or `"all"`. |

A number below 1, or `levels` other than `"all"` and `"outer"`, raises
`ValueError`.

## What runs in parallel

| Method | What runs in parallel |
|---|---|
| `Calibrator.run(starts=n)` | The starts |
| A particle swarm (`ParticleSwarmConfig`) | The particles of an iteration |
| The bootstraps, the block jackknife | The replicates |
| `Predictive`, `Scenarios`, scores | The draws of the sample |
| `GibbsPosterior`, `BayesianPosterior`, `ABCSMC`, `ABCRejection` | The particles or the draws |
| `ProfileLikelihood`, `PredictionProfile` | Each side of each quantity |
| The conformal methods | The fits of their folds or blocks |
| The sensitivity methods | The runs of the design |
| The local approximations, the identifiability methods | The runs of the finite differences, the points |
| `Simulation.replicates(count)` | The replicates of a stochastic engine |

One run of a model is never split: each thread runs whole runs.

## The number of threads

Every parallel method runs in a pool of `n` threads that commol builds from
the setting, and uses at most those threads, nested levels included. The
setting is a `contextvars` variable: it holds in the Python thread, or the
`asyncio` task, that sets it, and another thread starts with the default.

`commol.parallelism.threads` limits one block, for example when another library shares
the machine:

```python
from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.resampling import BayesianBootstrap

def model(k1: float | None):
    return (
        ModelBuilder(name="Threads")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_parameter(id="k1", value=k1)
        .add_transition(id="ab", source=["A"], target=["B"], rate="k1 * A")
        .build(Aggregate(bin_amounts={"A": 1000, "B": 0}), engine="DifferenceEquations")
    )

run = Simulation(model(0.2)).run(20)
problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(at=t, output="B", value=run["B"][t]) for t in range(2, 21, 2)
    ],
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0)
    ],
    loss_function="sse",
    optimization_config=ParticleSwarmConfig(particles=10, max_iterations=30),
    seed=3,
)
calibrator = Calibrator(Simulation(model(None)), problem)

with commol.parallelism.threads(2):
    fit = calibrator.run(starts=4)
    bootstrap = calibrator.uncertainty(BayesianBootstrap(replicates=8, block_length=1))
print(round(fit.best_parameters["k1"], 3))  # 0.2
```

## The levels

Some methods nest one parallel level in another: the starts of a calibration
each run a swarm, and so do the replicates of a bootstrap. `levels` chooses
which of them run in parallel:

| `levels` | The outer level (starts, replicates, draws) | The particles of a swarm |
|---|---|---|
| `"all"` (default) | In parallel | In parallel |
| `"outer"` | In parallel | One after the other |

With `"all"`, both levels share the one pool, and a thread with no work takes
work from another one, so a calibration with one start uses every thread and
a run with fewer starts than threads fills the gaps. `"outer"` keeps the
particles of a swarm in order.

The levels change no result: a seed gives the same result with any number of
threads and either level, bit for bit, the history of
`Calibrator.run_with_history` included.

```python
with commol.parallelism.threads(4, levels="outer"):
    in_order = calibrator.run_with_history()
with commol.parallelism.threads(4, levels="all"):
    in_parallel = calibrator.run_with_history()
print(in_order == in_parallel)  # True
```

## `RAYON_NUM_THREADS`

commol does not read the environment variable `RAYON_NUM_THREADS` of the
`rayon` library, and does not use the global thread pool of `rayon`. Use
`commol.parallelism.set_threads` instead. Another library of the process that uses
`rayon` keeps its own setting.

## Troubleshooting

| Symptom | Try |
|---|---|
| commol takes every core from other work | `commol.parallelism.set_threads(n)` with fewer threads |
| A setting does not hold in a thread you started | Set it in that thread: each Python thread has its own |
| A calibration with one start uses one core | Check `commol.parallelism.get_levels()`: `"outer"` runs the particles in order |

## Next steps

| To learn | Read |
|---|---|
| The optimizers and their starts | [Optimizers](calibration-optimizers.md) |
| The uncertainty methods | [Uncertainty](uncertainty/index.md) |
| Complete models, with a fit on two threads | [Examples](examples.md) |
