# Model Calibration

[Back to Home](../index.md)

Calibration finds the parameter values (and, if you want, initial conditions
and observation scale factors) that make a model reproduce observed data. You
describe the observations and the unknowns in a `CalibrationProblem`, and a
`Calibrator` runs an optimizer in the Rust engine.

| Topic | Page |
|---|---|
| The workflow, the problem, losses and results | This page |
| Calibrating initial conditions | [Calibrating Initial Conditions](calibration-initial-conditions.md) |
| Several outputs, aggregates, weights, scale factors, windowed data | [Observations](calibration-observations.md) |
| Nelder-Mead and particle swarm settings | [Optimizers](calibration-optimizers.md) |
| Relations the result must respect | [Constraints](calibration-constraints.md) |
| Reports of a calibrated model and run | [Reports](reports.md) |
| Samples, predictive bands and scores | [Uncertainty](uncertainty/index.md) |

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

def basic_model(k1: float | None, k2: float | None, engine="DifferenceEquations"):
    return (
        ModelBuilder(name="Basic Model")
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
            engine=engine,
        )
    )

# Observations generated from k1 = 0.4, k2 = 0.1
truth = Simulation(basic_model(0.4, 0.1)).run(num_steps=50)
observed_data = [
    ObservedDataPoint(at=step, output="B", value=truth["B"][step])
    for step in range(5, 51, 5)
]

# 1. A model whose unknowns are None
model = basic_model(None, None)

# 2. The problem: observations, unknowns, loss and optimizer
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=2000),
    seed=42,
)

# 3. Calibrate
result = Calibrator(Simulation(model), problem).run()
print({name: round(value, 3) for name, value in sorted(result.best_parameters.items())})
# {'k1': 0.4, 'k2': 0.1}

# 4. Write the values back, and simulate
model.apply_calibration_parameters(result, problem)
calibrated = Simulation(model).run(num_steps=100)
```

A `Simulation` follows its model: after `apply_calibration_parameters` or
`update_parameters`, its next `run` (and its `engine`) uses the new values,
also when it was created while a parameter was `None`. See
[Simulations](simulations.md#simulation).

## The problem

| `CalibrationProblem` field | Type | Default | Meaning |
|---|---|---|---|
| `observed_data` | `list[ObservedDataPoint]` | required | What the model must reproduce. See [Observations](calibration-observations.md). |
| `parameters` | `list[CalibrationParameter]` | required | The unknowns and their bounds. |
| `loss_function` | `str` | `"sse"` | How the misfit is measured; see [Losses](#losses). |
| `optimization_config` | `NelderMeadConfig` or `ParticleSwarmConfig` | required | The optimizer and its settings. See [Optimizers](calibration-optimizers.md). |
| `constraints` | `list[CalibrationConstraint]` | `[]` | Relations the result must respect. See [Constraints](calibration-constraints.md). |
| `normalize_observations` | `bool` | `False` | Divides each residual by the RMS of its observation series, so series of very different size weigh the same. |
| `initial_condition_remainder` | `str \| None` | `None` | The compartment or bin that absorbs the difference when initial conditions are calibrated. See [Calibrating Initial Conditions](calibration-initial-conditions.md). |
| `seed` | `int \| None` | `None` | Makes every random choice reproducible. |

### Observations

| `ObservedDataPoint` field | Type | Default | Meaning |
|---|---|---|---|
| `at` | `float` | required | The position of the observation on the axis. A difference model accepts integer steps only. |
| `output` | `str \| None` | `None` | The output it observes: a compartment or accumulator name, such as `B` or `B_g1`. Give `output` or `outputs`. |
| `outputs` | `list[str] \| None` | `None` | Outputs to sum before the comparison; needs a `label`. |
| `label` | `str` | `output` | The name of the observed series in reports, plots and the normalization. |
| `value` | `float` | required | The observed value. |
| `weight` | `float` | `1.0` | How much the observation counts in every loss; see [Losses](#losses). |
| `factor_id` | `str \| None` | `None` | A `factor` parameter that multiplies the prediction. |
| `window` | `float \| None` | `None` | Compares with the gain over `(at − window, at]` instead of the value. |

### Unknowns

| `CalibrationParameter` field | Type | Default | Meaning |
|---|---|---|---|
| `id` | `str` | required | What to calibrate; depends on `parameter_type`. |
| `parameter_type` | `str` | required | `"parameter"`, `"initial_condition"` or `"factor"`. |
| `min_bound`, `max_bound` | `float` | required | The search interval. |
| `initial_guess` | `float \| None` | `None` | Where the optimizer starts; the midpoint of the bounds if `None`. |

| `parameter_type` | `id` names | The value is |
|---|---|---|
| `"parameter"` | A model parameter, usually declared with `value=None` | The parameter value |
| `"initial_condition"` | A bin, a category, or an expanded compartment | An amount in the unit of its bin, or the split of a category; see [Calibrating Initial Conditions](calibration-initial-conditions.md) |
| `"factor"` | Any new id, used by observations through `factor_id` | A factor on the predictions; not part of the model |

`"parameter"` and `"factor"` work with every kind of model. The other types are
values of the model kind, and the kind declares which it accepts
(`model.structure.calibration_value_types()`): an aggregate model accepts
`"initial_condition"`. A type the kind does not accept raises a `ValueError`
that names the kind when you create the `Calibrator`.

## Losses

| `loss_function` | Formula (residual `r = observed − predicted`, weight `w`) | Use when |
|---|---|---|
| `"sse"` (default) | `Σ w·r²` | The usual choice |
| `"rmse"` | `√(Σ w·r² / Σ w)` | You want the error in the data's unit |
| `"mae"` | `Σ w·|r| / Σ w` | The data has outliers |

!!! note "How weights count"
    Every loss uses a weight the same way: `weight=2.0` makes an observation
    count twice as much as one with weight 1.0. With every weight at its
    default of 1.0, the losses are the plain sum of squares, root mean square
    and mean absolute error. `LossConfig` in
    `crates/commol-calibration/src/types.rs` computes them; the discrepancy
    (`crates/commol-calibration/src/discrepancy.rs`) applies them.

With `normalize_observations=True`, each residual is first divided by the RMS
of the observations of its series (its `label`).

A run the model does not allow gets a penalty instead of a loss: a run with
values that are not finite, or a difference step that leaves an amount below
zero (`InfeasibleStateError`, see
[Amounts below zero](transitions.md#amounts-below-zero)). The penalty is much
larger than any fit, and the run is not recorded in the history, so the
optimizer moves away from it and goes on. Any other error of a run, such as a
rate that cannot be evaluated or that is not a number, stops the calibration
with its message
(`OptimizationObjective` in `crates/commol-calibration/src/objective.rs`, the
only place that turns an infeasible point into a penalty).

## Stochastic engines

With a stochastic engine, one run per evaluation would make the loss change
from one evaluation to the next for the same parameters, and Nelder-Mead and
particle swarm would chase the noise. So each evaluation runs `replicates`
replicates with **common random numbers**: the same replicate seeds at every
evaluation, derived from `seed`. The loss is then a deterministic function of
the parameters (`CalibrationProblem::with_replicates` in
`crates/commol-calibration/src/calibration_problem.rs`).

You choose what each observation is compared with; there is no default,
because what to fit is a modelling choice:

| `replicate_summary` | Compared with each observation | Loss |
|---|---|---|
| `"mean"` | The mean of the replicates | `loss_function` |
| `"median"` | The median | `loss_function` |
| `Quantile(q)` | The quantile `q` | `loss_function` |
| `"crps"` | The empirical distribution of the replicates | The continuous ranked probability score, `E|X − y| − ½ E|X − X'|`, a proper scoring rule; its loss is the weighted mean score, whatever `loss_function` is |

```python
stochastic = basic_model(None, 0.1, engine="StochasticSteps")
runs = Simulation(basic_model(0.4, 0.1)).run(num_steps=50)
problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(at=step, output="B", value=runs["B"][step])
        for step in range(5, 51, 5)
    ],
    parameters=[CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0)],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=200),
    replicates=50,
    replicate_summary="crps",  # or "mean", "median", Quantile(0.9)
    seed=7,
)
result = Calibrator(Simulation(stochastic), problem).run()
print(round(result.best_parameters["k1"], 1))  # 0.4
```

| Rule | Where |
|---|---|
| A stochastic engine needs `replicates`, `replicate_summary` and `seed`; without one, `Calibrator` raises and the message lists the summaries | `rust_loss_config` in `py-commol/commol/context/_model/engine_state.py` |
| A deterministic engine does not use `replicates` and `replicate_summary` | The same |
| The summaries and the CRPS are computed in `LossConfig`, and the observation weights apply to every option | `LossConfig::replicate_loss` in `crates/commol-calibration/src/types.rs` |
| A constraint at positions is checked on every replicate | `CalibrationProblem::evaluate` |

With one replicate the CRPS is the absolute error. The
[predictive](uncertainty/index.md#the-predictive) of a sample runs every
replicate of each draw.

## Results

`Calibrator(simulation, problem)` checks the problem against the model, and
`run()` returns a `CalibrationResult`:

| `CalibrationResult` field | Meaning |
|---|---|
| `best_parameters` | `dict[str, float]`: the best value of every unknown, in the order of `problem.parameters` |
| `final_loss` | The loss at `best_parameters` |
| `iterations` | The optimizer iterations performed |
| `converged` | `True` only when the optimizer met its convergence test or `target_cost`; a run that used all `max_iterations` is `False`. See [Optimizers](calibration-optimizers.md#converged-and-termination_reason). |
| `termination_reason` | Why it stopped |
| `starts`, `failed_runs` | Every start that finished, and why each failed one did. See [Several starts](calibration-optimizers.md#several-starts). |

```python
print(result.converged, result.termination_reason)
print(f"loss = {result.final_loss:.3g} after {result.iterations} iterations")
```

| `Calibrator` member | What it does |
|---|---|
| `run(starts=1)` | Runs the optimizer from `starts` start points and returns the best `CalibrationResult`. See [Several starts](calibration-optimizers.md#several-starts). |
| `run_with_history()` | Returns a `CalibrationHistory`: the result and every evaluation of a feasible point. |
| `uncertainty(method)` | Runs an uncertainty method. See [Uncertainty](uncertainty/index.md#the-entry-point). |
| `num_observations`, `num_parameters`, `parameter_names` | Facts about the problem. |

## Writing the result back

`Model.apply_calibration_parameters(result, problem)` writes every calibrated
value into the model, routed by its `parameter_type`:

| `parameter_type` | Written to |
|---|---|
| `"parameter"` | The model parameter with that id |
| `"initial_condition"` | The initial conditions, so the model rebuilds exactly the state the optimizer scored |
| `"factor"` | Nothing: it describes the observations, not the model |

It accepts a `CalibrationResult` or a plain `{id: value}` mapping, such as one
of `sample.draws()`; a `ParameterSample` has no single set of values and raises
`TypeError`. It
computes the new definition first, so a rejected value leaves the model
unchanged. To set values by hand, use `Model.update_parameters({...})`.

## Checklist

- [ ] Every unknown is `None` in the model and has a `CalibrationParameter`.
- [ ] Every observation names an output of the model (`Simulation.simulation_outputs`).
- [ ] The bounds are realistic, and the result is not stuck at a bound.
- [ ] `result.converged` is `True`, or you know why not.
- [ ] You simulate the calibrated model with a **new** `Simulation`.

## Next steps

| To learn | Read |
|---|---|
| Fitting initial conditions | [Calibrating Initial Conditions](calibration-initial-conditions.md) |
| Fitting several or partial observations | [Observations](calibration-observations.md) |
| Tuning the optimizer | [Optimizers](calibration-optimizers.md) |
| Uncertainty | [Uncertainty](uncertainty/index.md) |
