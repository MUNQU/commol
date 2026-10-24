# Model Calibration

[Back to Home](https://munqu.github.io/commol/latest/index.md)

Calibration finds the parameter values (and, if you want, initial conditions and observation scale factors) that make a model reproduce observed data. You describe the observations and the unknowns in a `CalibrationProblem`, and a `Calibrator` runs an optimizer in the Rust engine.

| Topic                                                              | Page                                                                                                                  |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| The workflow, the problem, losses and results                      | This page                                                                                                             |
| Maximum likelihood with a model of the noise of each series        | [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md)             |
| Calibrating initial conditions                                     | [Calibrating Initial Conditions](https://munqu.github.io/commol/latest/guide/calibration-initial-conditions/index.md) |
| Several outputs, aggregates, weights, scale factors, windowed data | [Observations](https://munqu.github.io/commol/latest/guide/calibration-observations/index.md)                         |
| Nelder-Mead and particle swarm settings                            | [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/index.md)                             |
| Relations the result must respect                                  | [Constraints](https://munqu.github.io/commol/latest/guide/calibration-constraints/index.md)                           |
| Reports of a calibrated model and run                              | [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md)                                               |
| Samples, predictive bands and scores                               | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)                                       |

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

A `Simulation` follows its model: after `apply_calibration_parameters` or `update_parameters`, its next `run` (and its `engine`) uses the new values, also when it was created while a parameter was `None`. See [Simulations](https://munqu.github.io/commol/latest/guide/simulations/#simulation).

## The problem

| `CalibrationProblem` field    | Type                                                    | Default  | Meaning                                                                                                                                                                                                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observed_data`               | `list[ObservedDataPoint]`                               | required | What the model must reproduce. See [Observations](https://munqu.github.io/commol/latest/guide/calibration-observations/index.md).                                                                                                                                                                                               |
| `parameters`                  | `list[CalibrationParameter]`                            | required | The unknowns and their bounds.                                                                                                                                                                                                                                                                                                  |
| `loss_function`               | `str \| None`                                           | `None`   | How the misfit is measured; see [Losses](#losses). A problem has `loss_function` or `observation_models`, exactly one.                                                                                                                                                                                                          |
| `observation_models`          | `dict[str, ObservationModel] \| None`                   | `None`   | One observation model per series label: the fit maximizes their likelihood. See [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md).                                                                                                                                      |
| `optimization_config`         | `NelderMeadConfig` or `ParticleSwarmConfig`             | required | The optimizer and its settings. See [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/index.md).                                                                                                                                                                                                  |
| `constraints`                 | `list[CalibrationConstraint]`                           | `[]`     | Relations the result must respect. See [Constraints](https://munqu.github.io/commol/latest/guide/calibration-constraints/index.md).                                                                                                                                                                                             |
| `normalize_observations`      | `bool`                                                  | `False`  | Divides each residual by the RMS of its observation series, so series of very different size weigh the same. Needs a loss.                                                                                                                                                                                                      |
| `initial_condition_remainder` | `str \| None`                                           | `None`   | The compartment or bin that absorbs the difference when initial conditions are calibrated. See [Calibrating Initial Conditions](https://munqu.github.io/commol/latest/guide/calibration-initial-conditions/index.md).                                                                                                           |
| `replicates`                  | `int \| None`                                           | `None`   | How many replicates of a stochastic engine each evaluation runs. Required with a stochastic engine. See [Stochastic engines](#stochastic-engines) and [Stochastic Simulation](https://munqu.github.io/commol/latest/guide/stochastic/index.md).                                                                                 |
| `replicate_summary`           | `"mean"`, `"median"`, `"crps"`, `Quantile(q)` or `None` | `None`   | What each observation is compared with when an evaluation runs replicates. Required with a stochastic engine. See [Stochastic engines](#stochastic-engines).                                                                                                                                                                    |
| `seed`                        | `int \| None`                                           | `None`   | The seed of every random number of a calibration: the replicates of a stochastic engine, the starts, a particle swarm and the uncertainty methods. Without it, the engine draws one, which `Calibrator.seed` gives. See [Reproducibility](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#reproducibility). |

### Observations

An `ObservedDataPoint(at=..., output=..., value=...)` is one observed value of an output at a position on the axis. Its other fields observe a sum of outputs, weigh it, scale the prediction by a factor or compare a gain over a window: [Observations](https://munqu.github.io/commol/latest/guide/calibration-observations/index.md) lists them all.

### Unknowns

| `CalibrationParameter` field | Type            | Default  | Meaning                                                           |
| ---------------------------- | --------------- | -------- | ----------------------------------------------------------------- |
| `id`                         | `str`           | required | What to calibrate; depends on `parameter_type`.                   |
| `parameter_type`             | `str`           | required | `"parameter"`, `"initial_condition"`, `"factor"` or `"noise"`.    |
| `min_bound`, `max_bound`     | `float`         | required | The search interval.                                              |
| `initial_guess`              | `float \| None` | `None`   | Where the optimizer starts; the midpoint of the bounds if `None`. |

| `parameter_type`      | `id` names                                                                                                                                      | The value is                                                                                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"parameter"`         | A model parameter with a number, or one declared with `value=None`; not a formula or a series parameter, which the engine computes at each step | The parameter value                                                                                                                                                                     |
| `"initial_condition"` | A bin, a category, or an expanded compartment                                                                                                   | An amount in the unit of its bin, or the split of a category; see [Calibrating Initial Conditions](https://munqu.github.io/commol/latest/guide/calibration-initial-conditions/index.md) |
| `"factor"`            | Any new id, used by observations through `factor_id`                                                                                            | A factor on the predictions; not part of the model                                                                                                                                      |
| `"noise"`             | Any new id, named by an [observation model](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/#noise-parameters)       | An argument of the model, such as its standard deviation; not part of the model                                                                                                         |

`"parameter"`, `"factor"` and `"noise"` work with every kind of model. The other types are values of the model kind, and the kind declares which it accepts (`model.structure.calibration_value_types()`): an aggregate model accepts `"initial_condition"`. A type the kind does not accept raises a `ValueError` that names the kind when you create the `Calibrator`. A `"parameter"` that the model does not have, or that a formula or a series sets, raises a `ValueError` with the engine's message at the same time.

## Losses

A problem has a loss or observation models, exactly one, and neither has a default: what to fit is a modelling choice. A loss measures the residuals; [observation models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md) say how each series scatters, and the fit maximizes their likelihood, so its loss is `−log L`. `CalibrationProblem` raises a `ValueError` with both or neither.

| `loss_function` | Formula (residual `r = observed − predicted`, weight `w`) | Use when                              |
| --------------- | --------------------------------------------------------- | ------------------------------------- |
| `"sse"`         | `Σ w·r²`                                                  | The usual choice                      |
| `"rmse"`        | `√(Σ w·r² / Σ w)`                                         | You want the error in the data's unit |
| `"mae"`         | \`Σ w·                                                    | r                                     |

How weights count

Every loss uses a weight the same way: `weight=2.0` makes an observation count twice as much as one with weight 1.0. With every weight at its default of 1.0, the losses are the plain sum of squares, root mean square and mean absolute error. `LossConfig` in `crates/commol-calibration/src/types.rs` computes them; the discrepancy (`crates/commol-calibration/src/discrepancy.rs`) applies them.

With `normalize_observations=True`, each residual is first divided by the RMS of the observations of its series (its `label`).

A run the model does not allow gets a penalty instead of a loss: a run with values that are not finite, or a difference step that leaves an amount below zero (`InfeasibleStateError`, see [Amounts below zero](https://munqu.github.io/commol/latest/guide/transitions/#amounts-below-zero)). The penalty is much larger than any fit, and the run is not recorded in the history, so the optimizer moves away from it and goes on. Any other error of a run, such as a rate that cannot be evaluated or that is not a number, stops the calibration with its message (`OptimizationObjective` in `crates/commol-calibration/src/objective.rs`, the only place that turns an infeasible point into a penalty).

## Stochastic engines

With a stochastic engine, one run per evaluation would make the loss change from one evaluation to the next for the same parameters, and Nelder-Mead and particle swarm would chase the noise. So each evaluation runs `replicates` replicates with **common random numbers**: the same replicate seeds at every evaluation, derived from `seed`. The loss is then a deterministic function of the parameters (`CalibrationProblem::with_replicates` in `crates/commol-calibration/src/calibration_problem.rs`).

You choose what each observation is compared with; there is no default, because what to fit is a modelling choice:

| `replicate_summary` | Compared with each observation               | Loss                                         |
| ------------------- | -------------------------------------------- | -------------------------------------------- |
| `"mean"`            | The mean of the replicates                   | `loss_function`                              |
| `"median"`          | The median                                   | `loss_function`                              |
| `Quantile(q)`       | The quantile `q`, a level between 0 and 1    | `loss_function`                              |
| `"crps"`            | The empirical distribution of the replicates | The continuous ranked probability score, \`E |

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

| Rule                                                                                                                                                                                                                   | Where                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| A stochastic engine needs `replicates`, `replicate_summary` and `seed`; without one, `Calibrator` raises a `ValueError` that names the engine, and the message lists the summaries                                     | `CalibrationProblem::with_replicate_settings` in `crates/commol-calibration/src/calibration_problem.rs`; `SimulationEngine::is_stochastic` |
| A deterministic engine does not use `replicates` and `replicate_summary`                                                                                                                                               | The same                                                                                                                                   |
| A stochastic engine needs a loss: with `observation_models`, `Calibrator` raises a `ValueError`                                                                                                                        | `CalibrationProblem::new`; `Discrepancy::set_summary`                                                                                      |
| The summaries and the CRPS are computed in `LossConfig`, and the observation weights apply to every option                                                                                                             | `LossConfig::replicate_loss` in `crates/commol-calibration/src/types.rs`                                                                   |
| The median and `Quantile(q)` are the one quantile of commol, the one of `weighted_quantiles` and `SimulationReplicates.quantiles` (NumPy's `method="hazen"`): `Quantile(0.25)` of the replicates 1, 2, 3, 4, 5 is 1.75 | `ReplicateSummary::summarize`; `statistics::quantiles` in `crates/commol-calibration/src/statistics.rs`                                    |
| A constraint at positions is checked on every replicate                                                                                                                                                                | `CalibrationProblem::evaluate`                                                                                                             |

With one replicate the CRPS is the absolute error. The [predictive](https://munqu.github.io/commol/latest/guide/uncertainty/#the-predictive) of a sample runs every replicate of each draw.

## Results

`Calibrator(simulation, problem)` checks the problem against the model, and `run()` returns a `CalibrationResult`:

| `CalibrationResult` field | Meaning                                                                                                                                                                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `best_parameters`         | `dict[str, float]`: the best value of every unknown, in the order of `problem.parameters`                                                                                                                                                                            |
| `final_loss`              | The loss at `best_parameters`; with observation models, the negative log-likelihood                                                                                                                                                                                  |
| `iterations`              | The optimizer iterations performed                                                                                                                                                                                                                                   |
| `converged`               | `True` only when the optimizer met its convergence test (a stop rule for a particle swarm); a run that used all `max_iterations` is `False`. See [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#converged-and-termination_reason). |
| `termination_reason`      | Why it stopped                                                                                                                                                                                                                                                       |
| `starts`, `failed`        | Every start that finished, and each start that failed with its reason. See [Several starts](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#several-starts).                                                                                     |
| `seed`                    | The seed of the calibration's random numbers (`Calibrator.seed`). See [Reproducibility](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#reproducibility).                                                                                        |

```python
print(result.converged, result.termination_reason)
print(f"loss = {result.final_loss:.3g} after {result.iterations} iterations")
```

| `Calibrator` member                                     | What it does                                                                                                                                                                                      |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run(starts=1)`                                         | Runs the optimizer from `starts` start points and returns the best `CalibrationResult`. See [Several starts](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#several-starts). |
| `run_with_history()`                                    | Returns a `CalibrationHistory`: the result and every evaluation of a feasible point.                                                                                                              |
| `uncertainty(method)`                                   | Runs an uncertainty method. See [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/#the-entry-point).                                                                          |
| `identifiability(method)`                               | Runs an identifiability method. See [Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/index.md).                                                          |
| `seed`                                                  | The seed of the calibration's random numbers: `problem.seed`, or a seed drawn once and logged.                                                                                                    |
| `num_observations`, `num_parameters`, `parameter_names` | Facts about the problem.                                                                                                                                                                          |

## Writing the result back

`Model.apply_calibration_parameters(result, problem)` writes every calibrated value into the model, routed by its `parameter_type`:

| `parameter_type`      | Written to                                                                           |
| --------------------- | ------------------------------------------------------------------------------------ |
| `"parameter"`         | The model parameter with that id                                                     |
| `"initial_condition"` | The initial conditions, so the model rebuilds exactly the state the optimizer scored |
| `"factor"`, `"noise"` | Nothing: they describe the observations, not the model                               |

It accepts a `CalibrationResult` or a plain `{id: value}` mapping, such as one of `sample.draws()`; a `ParameterSample` has no single set of values and raises `TypeError`. It computes the new definition first, so a rejected value leaves the model unchanged. To set values by hand, use `Model.update_parameters({...})`.

## Checklist

- [ ] Every unknown is `None` in the model and has a `CalibrationParameter`.
- [ ] Every observation names an output of the model (`Simulation.simulation_outputs`).
- [ ] The bounds are realistic, and the result is not stuck at a bound.
- [ ] `result.converged` is `True`, or you know why not.
- [ ] You simulate the calibrated model (`apply_calibration_parameters`, then `run`).

## Next steps

| To learn                                | Read                                                                                                                  |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Fitting initial conditions              | [Calibrating Initial Conditions](https://munqu.github.io/commol/latest/guide/calibration-initial-conditions/index.md) |
| Fitting several or partial observations | [Observations](https://munqu.github.io/commol/latest/guide/calibration-observations/index.md)                         |
| Tuning the optimizer                    | [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/index.md)                             |
| Uncertainty                             | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)                                       |
