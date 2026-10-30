# Optimizers

[Back to Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)

The class of `optimization_config` chooses the optimizer: `NelderMeadConfig` for a fast local search, `ParticleSwarmConfig` for a global one. Both run in Rust on the [argmin](https://github.com/MUNQU/argmin) library, and both keep every unknown inside its bounds.

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
    ParticleSwarmConfig,
    Simulation,
)

def model(k1: float | None, k2: float | None):
    return (
        ModelBuilder(name="Optimizers")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="k2", value=k2)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .build(
            Aggregate(
                bin_amounts={"A": 990, "B": 10, "C": 0},
            ),
            engine="DifferenceEquations",
        )
    )

truth = Simulation(model(0.4, 0.1)).run(num_steps=50)
observed_data = [
    ObservedDataPoint(at=step, output="B", value=truth["B"][step])
    for step in range(5, 51, 5)
]

def calibrate(optimization_config) -> dict[str, float]:
    problem = CalibrationProblem(
        observed_data=observed_data,
        parameters=[
            CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
            CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
        ],
        loss_function="sse",
        optimization_config=optimization_config,
        seed=7,
    )
    result = Calibrator(Simulation(model(None, None)), problem).run()
    return {name: round(value, 2) for name, value in sorted(result.best_parameters.items())}

print(calibrate(NelderMeadConfig(max_iterations=2000)))                    # {'k1': 0.4, 'k2': 0.1}
print(calibrate(ParticleSwarmConfig(particles=30, max_iterations=300)))  # {'k1': 0.4, 'k2': 0.1}
```

## Choosing

| Optimizer      | Strength                                       | Weakness                                      | Use when                                 |
| -------------- | ---------------------------------------------- | --------------------------------------------- | ---------------------------------------- |
| Nelder-Mead    | Few evaluations; precise near the answer       | Local: can stop at a bound or a local minimum | Few unknowns, a good `initial_guess`     |
| Particle swarm | Searches the whole box; robust to local minima | Many evaluations                              | Several unknowns, no good starting point |

To search the whole box and still end precise, give a swarm a [polish](#polish-with-nelder-mead): each start then ends with Nelder-Mead from the best point of the swarm.

## Nelder-Mead

```python
nelder_mead = NelderMeadConfig(
    max_iterations=1000,
    sd_tolerance=1e-6,
    initial_simplex_size=0.1,
    alpha=1.0,
    gamma=2.0,
    rho=0.5,
    sigma=0.5,
    verbose=False,
    header_interval=100,
)
print(calibrate(nelder_mead))  # {'k1': 0.4, 'k2': 0.1}
```

| Field                            | Default                  | Meaning                                                                                                                                  |
| -------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `max_iterations`                 | `1000`                   | The iteration limit.                                                                                                                     |
| `sd_tolerance`                   | `1e-6`                   | Stops when the spread of the simplex values falls below it.                                                                              |
| `initial_simplex_size`           | `0.1`                    | The size of the first simplex: each vertex moves one unknown by this fraction of its bound width, towards the farther bound. In (0, 1\]. |
| `alpha`, `gamma`, `rho`, `sigma` | `None` (argmin defaults) | Reflection, expansion, contraction and shrink coefficients.                                                                              |
| `verbose`                        | `False`                  | Writes a progress table to `sys.stdout` (a notebook shows it). It shows one start, so it needs `run(starts=1)`.                          |
| `header_interval`                | `100`                    | With `verbose`, repeats the table header every this many iterations.                                                                     |

The search starts at `initial_guess`, or at the midpoint of the bounds; a start of 0 works. Nelder-Mead moves in an unbounded coordinate `u`, and the value it scores is `lo + (hi − lo)·(sin u + 1)/2` (the transform of MINUIT and lmfit), so every value it tries is inside the bounds (`SineTransform` of the argmin fork). An optimum exactly at a bound is reached only asymptotically, so it can take more iterations. Nelder-Mead needs finite bounds: a run with an infinite bound fails with the message `Nelder-Mead needs finite bounds`.

## `converged` and `termination_reason`

| `termination_reason`          | `converged` | Meaning                                                                                                                  |
| ----------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------ |
| `Terminated(SolverConverged)` | `True`      | Nelder-Mead: the spread of the simplex values fell below `sd_tolerance`. Particle swarm: a [stop rule](#stop-rules) held |
| `Terminated(MaxItersReached)` | `False`     | The run used all `max_iterations`                                                                                        |
| Any other reason              | `False`     | The run stopped for another reason that argmin reports                                                                   |

`converged` comes from `converged` in `crates/commol-calibration/src/optimization.rs`. A particle swarm without a [stop rule](#stop-rules) or a [polish](#polish-with-nelder-mead) always stops at `max_iterations`, so it reports `converged=False`; judge it by `final_loss`. A polished swarm reports the convergence of Nelder-Mead.

## Particle swarm

```python
swarm = ParticleSwarmConfig(
    particles=40,
    max_iterations=300,
    initialization="latin_hypercube",
    inertia="chaotic",
    inertia_min=0.4,
    inertia_max=0.9,
    acceleration="time_varying",
    cognitive_initial=2.5,
    cognitive_final=0.5,
    social_initial=0.5,
    social_final=2.5,
    velocity_clamp_factor=0.2,
    velocity_mutation_threshold=0.005,
    mutation="gaussian",
    mutation_scale=0.1,
    mutation_probability=0.05,
    mutation_application="global_best",
)
print(calibrate(swarm))  # {'k1': 0.4, 'k2': 0.1}
```

Each strategy of the swarm is a name, and its numbers are fields of their own. A strategy left as `None` is that of canonical PSO. A number set without its strategy, or with another strategy, is an error, so a setting is never ignored: `ParticleSwarmConfig(inertia_factor=0.5)` fails with `inertia_factor is read only with another inertia`.

| Field                                                                    | Default     | Meaning                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `particles`                                                              | `20`        | The swarm size. More particles search more widely and cost more evaluations.                                                                                                                                                                 |
| `max_iterations`                                                         | `1000`      | The iteration limit.                                                                                                                                                                                                                         |
| `initialization`                                                         | `"uniform"` | Where particles start: `"uniform"`, `"latin_hypercube"` (spread evenly) or `"opposition_based"` (each random point with its mirror).                                                                                                         |
| `verbose`                                                                | `False`     | Writes progress to `sys.stdout`; needs `run(starts=1)`.                                                                                                                                                                                      |
| `header_interval`                                                        | `100`       | With `verbose`, repeats the table header every this many iterations.                                                                                                                                                                         |
| `inertia`                                                                | `None`      | `"constant"` uses `inertia_factor`; `"chaotic"` moves between `inertia_min` and `inertia_max`. `None` is the weight of canonical PSO, 1/(2 ln 2) ≈ 0.72135.                                                                                  |
| `inertia_factor`                                                         | `None`      | With `inertia="constant"`: the weight, above 0. `None` is the weight of canonical PSO.                                                                                                                                                       |
| `inertia_min`, `inertia_max`                                             | `None`      | With `inertia="chaotic"`, both needed: the range of the weight, `inertia_min < inertia_max`.                                                                                                                                                 |
| `acceleration`                                                           | `None`      | `"constant"` uses `cognitive` and `social`; `"time_varying"` moves each pull from its initial to its final value, from explore (high cognitive) to exploit (high social). `None` is the coefficients of canonical PSO, 0.5 + ln 2 ≈ 1.19315. |
| `cognitive`, `social`                                                    | `None`      | With `acceleration="constant"`: the pulls towards each particle's best and the swarm's best. `None` is the coefficient of canonical PSO.                                                                                                     |
| `cognitive_initial`, `cognitive_final`, `social_initial`, `social_final` | `None`      | With `acceleration="time_varying"`, all four needed.                                                                                                                                                                                         |
| `mutation`                                                               | `None`      | Random jumps to escape local minima: `"gaussian"` or `"cauchy"` (larger jumps). `None` mutates no particle.                                                                                                                                  |
| `mutation_scale`, `mutation_probability`, `mutation_application`         | `None`      | With a mutation, all three needed: the size of a jump, the probability per iteration, and the particles it moves: `"global_best"`, `"all_particles"` or `"below_average"`.                                                                   |
| `velocity_clamp_factor`                                                  | `None`      | Limits the speed to a share of the search box (0.1–0.2 is typical).                                                                                                                                                                          |
| `velocity_mutation_threshold`                                            | `None`      | Restarts the velocity of particles slower than it (0.001–0.01).                                                                                                                                                                              |
| `improvement_window`, `improvement_tol_abs`, `improvement_tol_rel`       | `None`      | The improvement [stop rule](#stop-rules): all three or none.                                                                                                                                                                                 |
| `radius_tol`                                                             | `None`      | The radius [stop rule](#stop-rules).                                                                                                                                                                                                         |
| `polish`                                                                 | `None`      | A `NelderMeadConfig`: each start ends with Nelder-Mead from the best point of the swarm ([Polish with Nelder-Mead](#polish-with-nelder-mead)).                                                                                               |

### Stop rules

A swarm stops at `max_iterations` unless a stop rule holds first. The two rules come from the argmin fork; both are off until you set them, and neither has a default value. A run that stops by a rule reports `converged=True` and `termination_reason` `Terminated(SolverConverged)`.

| Rule        | Stops when                                                                                                                                                      | Fields           |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Improvement | The best loss improved by at most \`improvement_tol_abs + improvement_tol_rel·                                                                                  | best loss        |
| Radius      | Every particle is within `radius_tol` of the best position, as a fraction of the size of the search box (the length of its diagonal) (Zielinski and Laur, 2007) | `radius_tol` ≥ 0 |

```python
stopping = ParticleSwarmConfig(
    particles=30,
    max_iterations=2000,
    improvement_window=30,
    improvement_tol_abs=1e-12,
    improvement_tol_rel=1e-9,
)
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    loss_function="sse",
    optimization_config=stopping,
    seed=7,
)
result = Calibrator(Simulation(model(None, None)), problem).run()
print(result.converged, result.iterations < 2000)  # True True
```

A swarm can stall for some iterations and then improve again, so a short `improvement_window` can stop it early. Mutation and velocity mutation scatter the particles again, and unknowns that the data cannot tell apart keep them spread, so with them the radius can stay above `radius_tol`. Neither rule draws random numbers: with the same seed, a run with a rule is the run without it up to the iteration where it stops. In a [profile](https://munqu.github.io/commol/latest/guide/uncertainty/profiles/index.md), the improvement rule also sets the tolerance of a swarm.

### Polish with Nelder-Mead

A swarm stops before it converges; Nelder-Mead from its end converges. With `polish=NelderMeadConfig(...)`, each start runs the swarm from its point, then Nelder-Mead from the best point of the swarm of that start, on the threads of the start (`run` in `crates/commol-calibration/src/optimization.rs`).

```python
polished = ParticleSwarmConfig(
    particles=20,
    max_iterations=30,
    polish=NelderMeadConfig(max_iterations=2000, sd_tolerance=1e-10),
)
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    loss_function="sse",
    optimization_config=polished,
    seed=7,
)
result = Calibrator(Simulation(model(None, None)), problem).run(starts=4)
print(all(start.converged for start in result.starts))  # True
print(result.termination_reason)  # Terminated(SolverConverged)
print({name: round(value, 4) for name, value in result.best_parameters.items()})  # {'k1': 0.4, 'k2': 0.1}
```

| What                              | With a polish                                                                                                                             |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `StartSummary`                    | `start` is the start point of the swarm; `end`, `loss` and `converged` are those of Nelder-Mead                                           |
| `iterations`                      | The iterations of the swarm and of Nelder-Mead of the best start                                                                          |
| `converged`, `termination_reason` | Those of Nelder-Mead                                                                                                                      |
| Uncertainty methods               | Every fit of the problem polishes: the refits of a bootstrap, the subproblems of a profile, the starts that `StartsIdentifiability` reads |
| `verbose`, `header_interval`      | Those of the swarm write the table of the swarm; those of the polish, the table of Nelder-Mead                                            |

Each start polishes its own end, so the starts stay apart, and [`StartsIdentifiability`](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/#starts) reads the spread of the polished ends. A swarm polishes only with Nelder-Mead.

## Several starts

`Calibrator.run(starts=n)` runs the optimizer from `n` start points in parallel and returns the best result. Start 0 is the initial guess of each parameter (its `initial_guess`, or the midpoint of its bounds). Start `i > 0` is a row of a Latin hypercube over the bounds, drawn from the problem's `seed`: with `n − 1` rows, each parameter has one start in each of `n − 1` equal strata of its bounds, so the starts cover the parameter space (`start_points` in `crates/commol-calibration/src/starts.rs`). A particle swarm draws its own population; each start takes its own seed, also drawn from the problem's `seed`, so start 0 of `run(starts=n)` is the result of `run()`, with or without `verbose`.

A particle swarm also scores the particles of each iteration in parallel, in the order of the particles, so the result does not depend on the number of threads. `commol.parallelism.threads` chooses the threads and whether the particles run in parallel: see [Threads and Parallelism](https://munqu.github.io/commol/latest/guide/parallelism/index.md).

```python
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=2000),
    seed=7,
)
result = Calibrator(Simulation(model(None, None)), problem).run(starts=4)
print(len(result.starts), result.failed)  # 4 []
print(result.starts[0].start)  # {'k1': 0.5, 'k2': 0.5}
print({name: round(value, 2) for name, value in result.best_parameters.items()})  # {'k1': 0.4, 'k2': 0.1}
```

| `CalibrationResult` field | Meaning                                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `starts`                  | One `StartSummary` per start that finished, in order: `start`, `end`, `loss`, `converged`                                 |
| `failed`                  | Each start that failed, `"Start <index> failed: <reason>"`; each is also logged as a warning (`logging`, logger `commol`) |

The result is the start of least loss, the first of them on a tie; if every start fails, `run` raises a `RuntimeError` with the reason of each start. Starts that end at distinct points with equal losses are a sign that the data do not identify the parameters; `StartsIdentifiability` names the parameters that differ (see [Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/#starts)).

## Reproducibility

Set `CalibrationProblem(seed=...)`: the starts, a particle swarm and the replicates of a stochastic engine then make the same random choices on every run. Without a seed, the engine draws one when the calibrator is built; `Calibrator.seed` gives it and the calibrator logs it. The result keeps it in `CalibrationResult.seed`, and the sample of each uncertainty method in `sample.method.settings["seed"]`: set it as `problem.seed` to repeat the run, also from a result read back from JSON.

```python
again = problem.model_copy(update={"seed": result.seed})
print(Calibrator(Simulation(model(None, None)), again).run(starts=4) == result)  # True
```

## History of evaluations

`Calibrator.run_with_history()` optimizes from the initial guess and returns a `CalibrationHistory`: the `result`, and every evaluation of a feasible point the optimizer made (`CalibrationEvaluation`: its `parameters` and `loss`), in order. A point the objective penalizes is not kept.

## Troubleshooting

| Symptom                            | Try                                                                                                                                                                                                         |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A result sits at a bound           | Check the bounds, give a better `initial_guess`, or use particle swarm                                                                                                                                      |
| `converged=False`                  | Check `termination_reason`: with `MaxItersReached`, raise `max_iterations`; a particle swarm needs a [stop rule](#stop-rules) or a [polish](#polish-with-nelder-mead) to converge                           |
| Different results on each run      | Set `seed`; use more particles or iterations                                                                                                                                                                |
| A local minimum                    | `run(starts=n)` from several points                                                                                                                                                                         |
| A good loss but implausible values | Unknowns that the data cannot tell apart; fix one, or add observations                                                                                                                                      |
| Slow                               | Fewer observation steps or particles; Nelder-Mead from a good start; check `commol.parallelism.get_threads()` ([Threads and Parallelism](https://munqu.github.io/commol/latest/guide/parallelism/index.md)) |

## Next steps

| To learn                          | Read                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| Relations the result must respect | [Constraints](https://munqu.github.io/commol/latest/guide/calibration-constraints/index.md) |
| Uncertainty                       | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)             |
