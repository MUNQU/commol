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
| Ensembles and confidence intervals | [Probabilistic Calibration](probabilistic-calibration.md) |

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

def basic_model(k1: float | None, k2: float | None):
    return (
        ModelBuilder(name="Basic Model")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="k2", value=k2)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
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

# Observations generated from k1 = 0.4, k2 = 0.1
truth = Simulation(basic_model(0.4, 0.1)).run(num_steps=50)
observed_data = [
    ObservedDataPoint(step=step, compartment="B", value=truth["B"][step])
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

# 4. Write the values back, and simulate with a new Simulation
model.apply_calibration_parameters(result, problem)
calibrated = Simulation(model).run(num_steps=100)
```

!!! warning "A `Simulation` copies the model"
    `Simulation(model)` builds its engine from the model as it is at that
    moment. After you change the model (`apply_calibration_parameters`,
    `update_parameters`, …), create a **new** `Simulation`. An old one keeps the
    old values, and one created while a parameter was `None` fails with
    "Parameter index … not found".

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
| `probabilistic_config` | `ProbabilisticCalibrationConfig \| None` | `None` | Turns the problem into an ensemble calibration. See [Probabilistic Calibration](probabilistic-calibration.md). |
| `seed` | `int \| None` | `None` | Makes every random choice reproducible. |

### Observations

| `ObservedDataPoint` field | Type | Default | Meaning |
|---|---|---|---|
| `step` | `int` | required | The step of the observation. |
| `compartment` | `str` | required | The output it observes: a compartment or accumulator name, such as `B` or `B_g1`. With `compartments`, a label. |
| `value` | `float` | required | The observed value. |
| `weight` | `float` | `1.0` | How much the observation counts in every loss; see [Losses](#losses). |
| `compartments` | `list[str] \| None` | `None` | Outputs to sum before the comparison. |
| `scale_id` | `str \| None` | `None` | A `scale` parameter that multiplies the prediction. |
| `window_steps` | `int \| None` | `None` | Compares with the gain over the last `window_steps` steps instead of the value. |

### Unknowns

| `CalibrationParameter` field | Type | Default | Meaning |
|---|---|---|---|
| `id` | `str` | required | What to calibrate; depends on `parameter_type`. |
| `parameter_type` | `str` | required | `"parameter"`, `"initial_condition"` or `"scale"`. |
| `min_bound`, `max_bound` | `float` | required | The search interval. |
| `initial_guess` | `float \| None` | `None` | Where the optimizer starts; the midpoint of the bounds if `None`. |

| `parameter_type` | `id` names | The value is |
|---|---|---|
| `"parameter"` | A model parameter, usually declared with `value=None` | The parameter value |
| `"initial_condition"` | A bin, a category, or an expanded compartment | A fraction of the population; see [Calibrating Initial Conditions](calibration-initial-conditions.md) |
| `"scale"` | Any new id, used by observations through `scale_id` | A factor on the predictions; not part of the model |

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
    `crates/commol-calibration/src/types.rs` computes them, for the optimizer
    and for the ensemble central loss.

With `normalize_observations=True`, each residual is first divided by the RMS
of the observations of its series (its output, or its aggregate label).

## Results

`Calibrator(simulation, problem)` checks the problem against the model, and
`run()` returns a `CalibrationResult`:

| `CalibrationResult` field | Meaning |
|---|---|
| `best_parameters` | `dict[str, float]`: the best value of every unknown |
| `final_loss` | The loss at `best_parameters` |
| `iterations` | The optimizer iterations performed |
| `converged` | Whether the optimizer met its convergence test |
| `termination_reason` | Why it stopped |

```python
print(result.converged, result.termination_reason)
print(f"loss = {result.final_loss:.3g} after {result.iterations} iterations")
```

| `Calibrator` member | What it does |
|---|---|
| `run()` | Runs the optimizer and returns a `CalibrationResult`. |
| `run_with_history()` | Also returns every evaluation the optimizer made. |
| `run_probabilistic()` | Runs an ensemble calibration; needs `probabilistic_config`. |
| `num_observations`, `num_parameters`, `parameter_names` | Facts about the problem. |

## Writing the result back

`Model.apply_calibration_parameters(result, problem)` writes every calibrated
value into the model, routed by its `parameter_type`:

| `parameter_type` | Written to |
|---|---|
| `"parameter"` | The model parameter with that id |
| `"initial_condition"` | The initial conditions, so the model rebuilds exactly the population the optimizer scored |
| `"scale"` | Nothing: it describes the observations, not the model |

It accepts a `CalibrationResult`, a `ProbabilisticCalibrationResult` (it then
applies the ensemble's `point_parameters`) or a plain `{id: value}` mapping. It
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
| Uncertainty | [Probabilistic Calibration](probabilistic-calibration.md) |
