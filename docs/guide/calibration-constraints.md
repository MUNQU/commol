# Constraints

[Back to Model Calibration](calibration.md)

A constraint is an expression that must be **zero or more** for a calibration
result to be acceptable, such as `"3.0 - k1 / k2"` for `k1 / k2 <= 3`. A
constraint on unknowns is checked before each simulation; a constraint on
outputs is checked at the steps you list.

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

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | `str` | required | A unique identifier, used in error messages. |
| `expression` | `str` | required | Must evaluate to `>= 0` when the constraint holds. |
| `description` | `str \| None` | `None` | A free-text description. |
| `weight` | `float` | `1.0` | Above zero. Multiplies the amount of a violation in the penalty. |
| `at` | `list[float] \| None` | `None` | The positions on the axis at which to evaluate it. Required when the expression names an output. The engine accepts the positions it accepts for [observations](calibration-observations.md#positions-between-integers). |

## Writing the expression

Move every term to one side, so that the expression is `>= 0` when it holds:

| Condition | `expression` |
|---|---|
| `k1 <= 0.5` | `"0.5 - k1"` |
| `k1 >= k2` | `"k1 - k2"` |
| `k1 / k2 <= 5` | `"5.0 - k1 / k2"` |
| `2 <= k1 / k2` | `"k1 / k2 - 2.0"` |
| `k1 + k2 + k3 <= 1` | `"1.0 - (k1 + k2 + k3)"` |
| `B <= 300` at steps 0 to 50 | `"300 - B"` with `at=list(range(51))` |
| `A + B <= 100` at position 30 | `"100 - (A + B)"` with `at=[30]` |

The expression uses the [expression syntax](mathematical-expressions.md), and
the parser of the engine checks it when the constraint is built, within the
[limits of the parser](mathematical-expressions.md#limits-of-the-parser).

| Kind | Can name | Checked |
|---|---|---|
| Without `at` | The calibration unknowns and the variables of a [matrix](#a-bound-on-a-matrix) | Once per evaluation, before the simulation |
| With `at` | The same, and the simulation outputs (compartments, accumulators and the total of a bin over its categories, such as `B` of a stratified model) | At each listed position of the simulation, in every replicate of a stochastic engine |

A constraint cannot name a model parameter that is not calibrated, and an
expression that names an output without `at` is rejected when the
calibration starts.

```python
best = calibrate(
    [CalibrationConstraint(id="peak", expression="300 - B", at=list(range(51)))]
)
peak = max(Simulation(model(best["k1"], best["k2"])).run(num_steps=50)["B"])
print(round(peak))  # 300: the best fit whose B never exceeds 300
```

## A bound on a matrix

A constraint reads the entries `M_<i>_<j>` and the spectral radius `M_radius`
of a [matrix between categories](building-models.md#matrices-between-categories)
`M`. When calibrated parameters scale its rows or its columns, the constraint
reads the scaled matrix at the values of each point; a scale that is not
calibrated keeps its value in the model. So a bound on the growth factor of a
coupling, `k1 · ρ(diag(a) · M)`, is one constraint:

```python
import numpy as np

from commol import NelderMeadConfig

def coupled_model(k1: float | None, a1: float | None):
    return (
        ModelBuilder(name="Coupled")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="a1", value=a1)
        .add_matrix(
            id="M",
            stratification="group",
            values={"g1": {"g1": 2.0, "g2": 1.0}, "g2": {"g1": 1.0, "g2": 3.0}},
            row_scales={"g1": "a1"},
        )
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * matrix_sum(M, B / total)")
        .build(
            Aggregate(
                bin_amounts={"A": 900, "B": 100},
                stratification_fractions={"group": {"g1": 0.4, "g2": 0.6}},
            ),
            engine="DifferenceEquations",
        )
    )

# The data come from k1 = 0.05 and a1 = 2: k1 · ρ = 0.05 × 5 = 0.25
coupled_truth = Simulation(coupled_model(0.05, 2.0)).run(num_steps=20)
coupled_problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(at=step, output=output, value=coupled_truth[output][step])
        for output in ("B_g1", "B_g2")
        for step in range(2, 21, 2)
    ],
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=0.2, initial_guess=0.02),
        CalibrationParameter(id="a1", parameter_type="parameter", min_bound=0.25, max_bound=8.0, initial_guess=1.0),
    ],
    constraints=[CalibrationConstraint(id="growth", expression="0.2 - k1 * M_radius")],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=400),
    seed=3,
)
coupled_best = Calibrator(Simulation(coupled_model(None, None)), coupled_problem).run().best_parameters
a1 = coupled_best["a1"]
growth = coupled_best["k1"] * max(abs(np.linalg.eigvals([[2.0 * a1, a1], [1.0, 3.0]])))
print(growth <= 0.2 + 1e-6)  # True: the fit keeps k1 · ρ at or below 0.2
```

| Rule | Why |
|---|---|
| The radius is that of `diag(a) · values · diag(b)` at the point | The rates of the run read the same matrix (`ScaledMatrix` in `crates/commol-core/src/scaled_matrix.rs`), so the bound holds for the run it allows |
| It is computed once per point, before the run | The matrix does not change during a run |
| A scale with no value that the calibration does not set is refused when the calibration starts | The constraint would have no matrix to read |

## How a violation counts

A violated constraint does not add a small cost to the fit: it **replaces** the
loss with a penalty much larger than any fit, so the optimizer moves away from
it:

```text
loss = P × (1 + Σ weight × |violation|)
P    = max(1000 × n × (largest observed value)², 1e10)
```

where the sum runs over the violated constraints (and, for a constraint with
`at`, over its positions) and `n` is the number of observations. With
[observation models](uncertainty/observation-models.md), `P` comes from the
cost of each observation at the narrowest noise the bounds allow
(`Discrepancy::penalty_base` in `crates/commol-calibration/src/discrepancy.rs`).
Parameter constraints are checked first, and the simulation is
skipped when one fails. So constraints act almost as hard limits; `weight`
only ranks violations against each other.

## Checklist

- [ ] Each expression is `>= 0` when the condition holds.
- [ ] Expressions that name outputs have `at`.
- [ ] Expressions name only calibration unknowns, variables of a matrix and, with `at`, outputs.
- [ ] The constraints do not contradict each other, such as `k1 - 0.5` and `0.4 - k1`.
- [ ] The search box contains points that satisfy every constraint.

## Next steps

| To learn | Read |
|---|---|
| Tuning the optimizer | [Optimizers](calibration-optimizers.md) |
| Reporting the result | [Reports](reports.md) |
| How sure the fit is | [Uncertainty](uncertainty/index.md) |
