# Constraints

[Back to Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)

A constraint is an expression that must be **zero or more** for a calibration result to be acceptable, such as `"3.0 - k1 / k2"` for `k1 / k2 <= 3`. A constraint on unknowns is checked before each simulation; a constraint on outputs is checked at the steps you list.

## Quick path

```python
from commol import (
    Aggregate,
    CalibrationConstraint,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)

def model(k1: float | None, k2: float | None):
    return (
        ModelBuilder(name="Constrained")
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

# The data comes from k1 / k2 = 4
truth = Simulation(model(0.4, 0.1)).run(num_steps=50)
observed_data = [
    ObservedDataPoint(at=step, output="B", value=truth["B"][step])
    for step in range(5, 51, 5)
]

def calibrate(constraints: list[CalibrationConstraint]) -> dict[str, float]:
    problem = CalibrationProblem(
        observed_data=observed_data,
        parameters=[
            CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
            CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.01, max_bound=1.0),
        ],
        constraints=constraints,
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(particles=30, max_iterations=300),
        seed=3,
    )
    result = Calibrator(Simulation(model(None, None)), problem).run()
    return result.best_parameters

best = calibrate([CalibrationConstraint(id="ratio", expression="3.0 - k1 / k2")])
print(round(best["k1"] / best["k2"], 2))  # 3.0: the best fit that respects k1 / k2 <= 3
```

## `CalibrationConstraint`

| Field         | Type                  | Default  | Meaning                                                                                                                                                                                                                                                            |
| ------------- | --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `id`          | `str`                 | required | A unique identifier, used in error messages.                                                                                                                                                                                                                       |
| `expression`  | `str`                 | required | Must evaluate to `>= 0` when the constraint holds.                                                                                                                                                                                                                 |
| `description` | `str \| None`         | `None`   | A free-text description.                                                                                                                                                                                                                                           |
| `weight`      | `float`               | `1.0`    | Above zero. Multiplies the amount of a violation in the penalty.                                                                                                                                                                                                   |
| `at`          | `list[float] \| None` | `None`   | The positions on the axis at which to evaluate it. Required when the expression names an output. The engine accepts the positions it accepts for [observations](https://munqu.github.io/commol/latest/guide/calibration-observations/#positions-between-integers). |

## Writing the expression

Move every term to one side, so that the expression is `>= 0` when it holds:

| Condition                     | `expression`                          |
| ----------------------------- | ------------------------------------- |
| `k1 <= 0.5`                   | `"0.5 - k1"`                          |
| `k1 >= k2`                    | `"k1 - k2"`                           |
| `k1 / k2 <= 5`                | `"5.0 - k1 / k2"`                     |
| `2 <= k1 / k2`                | `"k1 / k2 - 2.0"`                     |
| `k1 + k2 + k3 <= 1`           | `"1.0 - (k1 + k2 + k3)"`              |
| `B <= 300` at steps 0 to 50   | `"300 - B"` with `at=list(range(51))` |
| `A + B <= 100` at position 30 | `"100 - (A + B)"` with `at=[30]`      |

The expression uses the [expression syntax](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md) and passes the same security check.

| Kind         | Can name                                                                                                                                           | Checked                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Without `at` | The calibration unknowns only                                                                                                                      | Once per evaluation, before the simulation                                           |
| With `at`    | The unknowns and the simulation outputs (compartments, accumulators and the total of a bin over its categories, such as `B` of a stratified model) | At each listed position of the simulation, in every replicate of a stochastic engine |

A constraint cannot name a model parameter that is not calibrated, and an expression that names an output without `at` is rejected when the calibration starts.

```python
best = calibrate(
    [CalibrationConstraint(id="peak", expression="300 - B", at=list(range(51)))]
)
peak = max(Simulation(model(best["k1"], best["k2"])).run(num_steps=50)["B"])
print(round(peak))  # 300: the best fit whose B never exceeds 300
```

## How a violation counts

A violated constraint does not add a small cost to the fit: it **replaces** the loss with a penalty much larger than any fit, so the optimizer moves away from it:

```text
loss = P × (1 + Σ weight × |violation|)
P    = max(1000 × n × (largest observed value)², 1e10)
```

where the sum runs over the violated constraints (and, for a constraint with `at`, over its positions) and `n` is the number of observations. With [observation models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md), `P` comes from the cost of each observation at the narrowest noise the bounds allow (`Discrepancy::penalty_base` in `crates/commol-calibration/src/discrepancy.rs`). Parameter constraints are checked first, and the simulation is skipped when one fails. So constraints act almost as hard limits; `weight` only ranks violations against each other.

## Checklist

- [ ] Each expression is `>= 0` when the condition holds.
- [ ] Expressions that name outputs have `at`.
- [ ] Expressions name only calibration unknowns (and outputs).
- [ ] The constraints do not contradict each other, such as `k1 - 0.5` and `0.4 - k1`.
- [ ] The search box contains points that satisfy every constraint.

## Next steps

| To learn             | Read                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------- |
| Tuning the optimizer | [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/index.md) |
| Reporting the result | [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md)                   |
| How sure the fit is  | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)           |
