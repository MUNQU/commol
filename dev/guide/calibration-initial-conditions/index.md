# Calibrating Initial Conditions

[Back to Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)

An `initial_condition` unknown fits how the state starts: the amount of a bin in its unit, the split of a category, or the amount of one expanded compartment. The Rust engine applies the values in a fixed order, and a **remainder** compartment keeps the sum of the amounts the model states.

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

def model_with_b(b_amount: float | None):
    # A holds what B does not, so A and B keep 1000 between them.
    a_amount = 1000 if b_amount is None else 1000 - b_amount
    return (
        ModelBuilder(name="Initial B")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=0.3)
        .add_parameter(id="k2", value=0.1)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .build(
            Aggregate(bin_amounts={"A": a_amount, "B": b_amount, "C": 0}),
            engine="DifferenceEquations",
        )
    )

truth = Simulation(model_with_b(20)).run(num_steps=40)
observed_data = [
    ObservedDataPoint(at=step, output="B", value=truth["B"][step])
    for step in range(5, 41, 5)
]

model = model_with_b(None)
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(
            id="B", parameter_type="initial_condition", min_bound=0.0, max_bound=100.0
        )
    ],
    initial_condition_remainder="A",
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=1000),
)

result = Calibrator(Simulation(model), problem).run()
print(round(result.best_parameters["B"], 2))  # 20.0

model.apply_calibration_parameters(result, problem)
print({c.name: round(c.amount, 1) for c in model.structure.initial_state(model.process)})
# {'A': 980.0, 'B': 20.0, 'C': 0.0}
```

Here `B` is calibrated, and `A`, the remainder, takes what `B` leaves of the 1000 that the model states for `A` and `B`.

## What an id can name

| `id`                                      | Example                | The value is                                                                                                                               |
| ----------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| A bin                                     | `"B"`                  | The amount of the bin, in its unit. Its compartments share it in the proportions of the initial state, also when the `Simulation` has run. |
| A bin prefix (a bin and some categories)  | `"B_g1"`               | The amount in those compartments, shared in the same way. Compartments are matched by label, so `B_g1` never covers a bin `B_g1x`.         |
| An expanded compartment                   | `"B_g1_t2"`            | The amount of that compartment alone, in the unit of its bin.                                                                              |
| A category of a **binary** stratification | `"g1"` or `"group:g1"` | The split: every `(g1, g2)` pair of compartments keeps its total and gives `g1` this fraction.                                             |

A category name that two stratifications declare must be qualified with its stratification, as in `"group:g1"`; a bare name is rejected as ambiguous.

The engine resolves every id when you create the `Calibrator` (`InitialConditionPlan` in `crates/commol-model/src/initial_conditions.rs`), so a wrong id fails before the optimization starts:

| `id`                                                                                                   | Error                                                                 |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Names nothing in the model                                                                             | `Invalid bin, compartment or category 'X'`, with the compartments     |
| A category of a stratification with three or more categories, such as `"g1"` of `group = [g1, g2, g3]` | `Initial condition category 'group:g1' needs a binary stratification` |
| A category name that two stratifications declare                                                       | `… is ambiguous`, with the qualified names                            |

## The remainder

Without help, calibrated values change the sum of the amounts. Name a compartment or bin in `initial_condition_remainder`, and it takes what the values leave of the amounts the model states:

| Rule                                             | Detail                                                                                                                                                                                                         |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| It is not calibrated                             | It takes the sum of the amounts the model states, less every other compartment.                                                                                                                                |
| It keeps the sum of what the values and it cover | A compartment that no value covers keeps its amount, so it does not change the remainder. A bin left `None` for calibration counts as 0, so the remainder states the sum: `A` holds 1000 while `B` is unknown. |
| It has the unit of every amount it gives         | The `Calibrator` rejects a calibrated bin, prefix or compartment whose unit is not the unit of the remainder, and names both. A category split has no unit.                                                    |
| It keeps its own split                           | A bin remainder shares what it takes across its compartments in the proportions of the initial state.                                                                                                          |
| It must not overlap a calibrated id              | Building the problem rejects it otherwise.                                                                                                                                                                     |
| A value that would make it negative is penalized | The optimizer moves away from it.                                                                                                                                                                              |

Without a remainder, values are applied as given, and the other compartments keep their amounts. So a model of a count in `A` and `B` and of a concentration in `C` can fit the amount of `C` alone, or fit `B` with `A` as the remainder.

## Order of application

Whatever their order in `parameters`, values are written in layers, like a model definition:

1. bins and bin prefixes;
1. category splits;
1. expanded compartments;
1. the remainder.

Within a layer, the order of `parameters` applies. So an expanded compartment always wins over the bin that contains it.

## One expanded compartment

Fitting a single compartment of a stratified model, such as a state that is empty in one group only, needs the remainder:

```python
def stratified(b_g1: float | None):
    return (
        ModelBuilder(name="Composite")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_parameter(id="k1", value=0.2)
        .add_transition(id="t_ba", source=["B"], target=["A"], rate="k1")
        .build(
            Aggregate(
                bin_amounts={"A": 900, "B": 100},
                stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
                compartment_amounts={}
                if b_g1 is None
                else {"B_g1": b_g1, "A_g1": 500 - b_g1},
            ),
            engine="DifferenceEquations",
        )
    )

truth = Simulation(stratified(10)).run(num_steps=20)
problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(at=step, output="A_g1", value=truth["A_g1"][step])
        for step in range(2, 21, 2)
    ],
    parameters=[
        CalibrationParameter(
            id="B_g1",
            parameter_type="initial_condition",
            min_bound=0.0,
            max_bound=100.0,
            initial_guess=30.0,  # Nelder-Mead is local: start near the answer
        )
    ],
    initial_condition_remainder="A_g1",
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=1000),
)

composite = stratified(None)
result = Calibrator(Simulation(composite), problem).run()
print(round(result.best_parameters["B_g1"], 2))  # 10.0

composite.apply_calibration_parameters(result, problem)
print({c.name: round(c.amount, 1) for c in composite.structure.initial_state(composite.process)})
# {'A_g1': 490.0, 'A_g2': 450.0, 'B_g1': 10.0, 'B_g2': 50.0}
```

## Writing back

`apply_calibration_parameters` writes the values so that the model rebuilds the state the optimizer scored:

1. Each bin whose every compartment a value or the remainder sets gets the amount the engine gave it, in the unit of the bin, and each category gets the split the engine applied, so the definition stays readable.
1. Every compartment whose amount still differs from the scored one is pinned as a `compartment_amounts` override.

So an expanded-compartment id always becomes an override, and so does a bin or category value that overlaps an existing override. A bin of which a value sets only some compartments keeps its amount. Reloading the saved model gives the scored state exactly. A calibrated category split is written with `update_stratification_fractions`: the named category gets the value and its complement the rest.

## Checklist

- [ ] Each unknown amount is `None` in the model, and the bounds are amounts in the unit of its bin (a category split is in `[0, 1]`).
- [ ] A remainder is named when the calibrated values should keep a sum, and it holds that sum in the model.
- [ ] The remainder has the unit of every amount it gives.
- [ ] The remainder is not also calibrated.
- [ ] Category ids name a binary stratification, qualified when the name is shared.

## Next steps

| To learn                            | Read                                                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| The rules of initial conditions     | [Building Models](https://munqu.github.io/commol/latest/guide/building-models/#the-structure-aggregate)    |
| Reporting fitted initial conditions | [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md)                                    |
| The design behind this page         | [Initial conditions](https://munqu.github.io/commol/latest/codebase/decisions/initial-conditions/index.md) |
