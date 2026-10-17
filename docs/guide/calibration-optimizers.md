# Optimizers

[Back to Model Calibration](calibration.md)

The class of `optimization_config` chooses the optimizer: `NelderMeadConfig` for
a fast local search, `ParticleSwarmConfig` for a global one. Both run in Rust on
the [argmin](https://github.com/MUNQU/argmin) library, and both keep every
unknown inside its bounds.

## Choosing

| Optimizer | Strength | Weakness | Use when |
|---|---|---|---|
| Nelder-Mead | Few evaluations; precise near the answer | Local: can stop at a bound or a local minimum | Few unknowns, a good `initial_guess` |
| Particle swarm | Searches the whole box; robust to local minima | Many evaluations | Several unknowns, no good starting point |

A common pattern: run particle swarm first, then Nelder-Mead from its result
with `initial_guess`.

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
                total=1000,
                bin_fractions={"A": 0.99, "B": 0.01, "C": 0.0},
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
        optimization_config=optimization_config,
        seed=7,
    )
    result = Calibrator(Simulation(model(None, None)), problem).run()
    return {name: round(value, 2) for name, value in sorted(result.best_parameters.items())}

print(calibrate(NelderMeadConfig(max_iterations=2000)))                    # {'k1': 0.4, 'k2': 0.1}
print(calibrate(ParticleSwarmConfig(num_particles=30, max_iterations=300)))  # {'k1': 0.4, 'k2': 0.1}
```

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

| Field | Default | Meaning |
|---|---|---|
| `max_iterations` | `1000` | The iteration limit. |
| `sd_tolerance` | `1e-6` | Stops when the spread of the simplex values falls below it. |
| `initial_simplex_size` | `0.1` | The size of the first simplex: each vertex moves one unknown by this fraction of its bound width, towards the farther bound. In (0, 1]. |
| `alpha`, `gamma`, `rho`, `sigma` | `None` (argmin defaults) | Reflection, expansion, contraction and shrink coefficients. |
| `verbose` | `False` | Writes a progress table to `sys.stdout` (a notebook shows it). It shows one start, so it needs `run(starts=1)`. |
| `header_interval` | `100` | With `verbose`, repeats the table header every this many iterations. |

The search starts at `initial_guess`, or at the midpoint of the bounds; a start
of 0 works. Nelder-Mead searches an unbounded coordinate `u` and scores
`x = lo + (hi − lo)·(sin u + 1)/2` (the transform of MINUIT and lmfit), so every
value it tries is inside the bounds (`SineTransform` in
`crates/commol-calibration/src/optimization.rs`). An optimum exactly at a bound
is reached only asymptotically, so it can take more iterations.

## `converged` and `termination_reason`

| `termination_reason` | `converged` | Meaning |
|---|---|---|
| `Terminated(SolverConverged)` | `True` | Nelder-Mead: the spread of the simplex values fell below `sd_tolerance` |
| `Terminated(TargetCostReached)` | `True` | Particle swarm: the loss reached `target_cost` |
| `Terminated(MaxItersReached)` | `False` | The run used all `max_iterations` |
| Any other reason | `False` | The run stopped for another reason, such as a timeout |

`converged` comes from `converged` in `optimization.rs`. A particle swarm
without `target_cost` always stops at `max_iterations`, so it reports
`converged=False`; judge it by `final_loss`.

## Particle swarm

```python
swarm = (
    ParticleSwarmConfig(num_particles=40, max_iterations=300, initialization="latin_hypercube")
    .inertia("chaotic", w_min=0.4, w_max=0.9)
    .acceleration("time_varying", c1_initial=2.5, c1_final=0.5, c2_initial=0.5, c2_final=2.5)
    .velocity(clamp_factor=0.2, mutation_threshold=0.005)
    .mutation("gaussian", scale=0.1, probability=0.05, application="global_best")
)
print(calibrate(swarm))  # {'k1': 0.4, 'k2': 0.1}
```

| Field | Default | Meaning |
|---|---|---|
| `num_particles` | `20` | The swarm size. More particles search more widely and cost more evaluations. |
| `max_iterations` | `1000` | The iteration limit. |
| `initialization` | `"uniform"` | Where particles start: `"uniform"`, `"latin_hypercube"` (spread evenly) or `"opposition_based"` (each random point with its mirror). |
| `verbose` | `False` | Writes progress to `sys.stdout`; needs `run(starts=1)`. |

Each method returns the configuration, so they chain:

| Method | Arguments | Effect |
|---|---|---|
| `.inertia("constant", factor=0.721)` | `factor` | A fixed inertia weight. |
| `.inertia("chaotic", w_min=…, w_max=…)` | `w_min < w_max` | An inertia weight that varies chaotically between the two. |
| `.acceleration("constant", cognitive=1.193, social=1.193)` | `cognitive`, `social` | Fixed pulls towards each particle's best and the swarm's best. |
| `.acceleration("time_varying", c1_initial=…, c1_final=…, c2_initial=…, c2_final=…)` | four factors | Pulls that change from explore (high `c1`) to exploit (high `c2`). |
| `.velocity(clamp_factor=None, mutation_threshold=None)` | fractions | `clamp_factor` limits speed to a share of the search box (0.1–0.2 is typical); `mutation_threshold` restarts particles slower than it (0.001–0.01). |
| `.mutation(strategy, *, scale, probability, application)` | see right | Random jumps to escape local minima: `strategy` `"gaussian"` or `"cauchy"` (larger jumps), `probability` per iteration, `application` `"global_best"`, `"all_particles"` or `"below_average"`. |

## Several starts

`Calibrator.run(starts=n)` runs the optimizer from `n` start points in
parallel and returns the best result. Start 0 is the initial guess of each
parameter (its `initial_guess`, or the midpoint of its bounds). Start `i > 0`
is a row of a Latin hypercube over the bounds, drawn from the problem's
`seed`: with `n − 1` rows, each parameter has one start in each of `n − 1`
equal strata of its bounds, so the starts cover the parameter space
(`start_points` in `crates/commol-calibration/src/starts.rs`). A particle
swarm draws its own population; its start `i > 0` takes its own seed.

```python
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    optimization_config=NelderMeadConfig(max_iterations=2000),
    seed=7,
)
result = Calibrator(Simulation(model(None, None)), problem).run(starts=4)
print(len(result.starts), result.failed_runs)  # 4 []
print(result.starts[0].start)  # {'k1': 0.5, 'k2': 0.5}
print({name: round(value, 2) for name, value in result.best_parameters.items()})  # {'k1': 0.4, 'k2': 0.1}
```

| `CalibrationResult` field | Meaning |
|---|---|
| `starts` | One `StartSummary` per start that finished, in order: `start`, `end`, `loss`, `converged` |
| `failed_runs` | Why each start that failed did, with its number; each is also logged as a warning (`logging`, logger `commol.api.calibrator`) |

The result is the start of least loss; if every start fails, `run` raises a
`RuntimeError` with every reason. Starts that end at distinct points with equal
losses are a sign that the data do not identify the parameters;
`StartsIdentifiability` names the parameters that differ (see
[Identifiability](uncertainty/identifiability.md#starts)).

## Reproducibility

Set `CalibrationProblem(seed=...)`: the starts, a particle swarm and the
replicates of a stochastic engine then make the same random choices on every
run. Without a seed, `Calibrator.seed` draws one from the operating system and
logs it, so the run can be repeated.

## History of evaluations

`Calibrator.run_with_history()` optimizes from the initial guess and returns a
`CalibrationHistory`: the `result`, and every evaluation of a feasible point
the optimizer made (`CalibrationEvaluation`: its `parameters` and `loss`), in
order. A point the objective penalizes is not kept.

## Troubleshooting

| Symptom | Try |
|---|---|
| A result sits at a bound | Check the bounds, give a better `initial_guess`, or use particle swarm |
| `converged=False` | Check `termination_reason`: with `MaxItersReached`, raise `max_iterations` |
| Different results on each run | Set `seed`; use more particles or iterations |
| A local minimum | `run(starts=n)` from several points |
| A good loss but implausible values | Unknowns that the data cannot tell apart; fix one, or add observations |
| Slow | Fewer observation steps or particles; Nelder-Mead from a good start |

## Next steps

| To learn | Read |
|---|---|
| Relations the result must respect | [Constraints](calibration-constraints.md) |
| Uncertainty | [Uncertainty](uncertainty/index.md) |
