# Calibrating Initial Conditions

[Back to Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)

An `initial_condition` unknown fits how the population starts: the share of a bin, the split of a category, or the head count of one expanded compartment. The Rust engine applies the values in a fixed order, and a **remainder** compartment keeps the total at `population_size`.

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

def model_with_b(b_fraction: float | None):
    a_fraction = None if b_fraction is None else 1.0 - b_fraction
    return (
        ModelBuilder(name="Initial B")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=0.3)
        .add_parameter(id="k2", value=0.1)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": a_fraction},
                {"bin": "B", "fraction": b_fraction},
                {"bin": "C", "fraction": 0.0},
            ],
        )
        .build(typology="DifferenceEquations")
    )

truth = Simulation(model_with_b(0.02)).run(num_steps=40)
observed_data = [
    ObservedDataPoint(step=step, compartment="B", value=truth["B"][step])
    for step in range(5, 41, 5)
]

model = model_with_b(None)
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(
            id="B", parameter_type="initial_condition", min_bound=0.0, max_bound=0.1
        )
    ],
    initial_condition_remainder="A",
    optimization_config=NelderMeadConfig(max_iterations=1000),
)

result = Calibrator(Simulation(model), problem).run()
print(round(result.best_parameters["B"], 4))  # 0.02

model.apply_calibration_parameters(result, problem)
print({c.name: round(c.population, 1) for c in model.initial_state()})
# {'A': 980.0, 'B': 20.0, 'C': 0.0}
```

Here `B` is calibrated, and `A`, the remainder, takes what is left of the population.

## What an id can name

| `id`                                      | Example                | The value is                                                                                                                         |
| ----------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| A bin                                     | `"B"`                  | The fraction of the population in the bin. Its compartments share it in their current proportions.                                   |
| A bin prefix (a bin and some categories)  | `"B_g1"`               | The fraction in those compartments, shared in the same way. Compartments are matched by label, so `B_g1` never covers a bin `B_g1x`. |
| An expanded compartment                   | `"B_g1_t2"`            | The fraction of the population in that compartment alone.                                                                            |
| A category of a **binary** stratification | `"g1"` or `"group:g1"` | The split: every `(g1, g2)` pair of compartments keeps its total and gives `g1` this fraction.                                       |

A category name that two stratifications declare must be qualified with its stratification, as in `"group:g1"`; a bare name is rejected as ambiguous.

The engine resolves every id when you create the `Calibrator` or the `ProbabilisticCalibrator` (`check_initial_conditions` in `crates/commol-calibration/src/initial_conditions.rs`), so a wrong id fails before the optimization starts:

| `id`                                                                                             | Error                                                              |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Names nothing in the model                                                                       | `Invalid bin, compartment or category 'X'`, with the compartments  |
| A category of a stratification with three or more categories, such as `"y"` of `age = [y, m, o]` | `Initial condition category 'age:y' needs a binary stratification` |
| A category name that two stratifications declare                                                 | `… is ambiguous`, with the qualified names                         |

## The remainder

Without help, calibrated values change the total population. Name a compartment or bin in `initial_condition_remainder`, and it absorbs the difference:

| Rule                                             | Detail                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| It is not calibrated                             | It takes `population_size` minus everything else.                                           |
| It keeps its own split                           | A bin remainder shares the difference across its compartments in their current proportions. |
| It must not overlap a calibrated id              | Building the problem rejects it otherwise.                                                  |
| A value that would make it negative is penalized | The optimizer moves away from it.                                                           |

Without a remainder, values are applied as given, and only an initial population above `population_size` is penalized.

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
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "A", "fraction": 0.9}, {"bin": "B", "fraction": 0.1}],
            stratification_fractions=[
                {
                    "stratification": "group",
                    "fractions": [
                        {"category": "g1", "fraction": 0.5},
                        {"category": "g2", "fraction": 0.5},
                    ],
                }
            ],
            # The true state: B_g1 holds b_g1, and A_g1 holds the rest of g1
            compartment_fractions=(
                []
                if b_g1 is None
                else [
                    {"compartment": "B_g1", "fraction": b_g1},
                    {"compartment": "A_g1", "fraction": 0.5 - b_g1},
                ]
            ),
        )
        .build(typology="DifferenceEquations")
    )

truth = Simulation(stratified(0.01)).run(num_steps=20)
problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(step=step, compartment="A_g1", value=truth["A_g1"][step])
        for step in range(2, 21, 2)
    ],
    parameters=[
        CalibrationParameter(
            id="B_g1",
            parameter_type="initial_condition",
            min_bound=0.0,
            max_bound=0.1,
            initial_guess=0.03,  # Nelder-Mead is local: start near the answer
        )
    ],
    initial_condition_remainder="A_g1",
    optimization_config=NelderMeadConfig(max_iterations=1000),
)

composite = stratified(None)
result = Calibrator(Simulation(composite), problem).run()
print(round(result.best_parameters["B_g1"], 4))  # 0.01

composite.apply_calibration_parameters(result, problem)
print({c.name: round(c.population, 1) for c in composite.initial_state()})
# {'A_g1': 490.0, 'A_g2': 450.0, 'B_g1': 10.0, 'B_g2': 50.0}
```

## Writing back

`apply_calibration_parameters` writes the values so that the model rebuilds the population the optimizer scored:

1. Each bin and each category gets the fraction the engine applied, so the definition stays readable.
1. Every compartment whose head count still differs from the scored one is pinned as a `compartment_fractions` override.

So an expanded-compartment id always becomes an override, and so does a bin or category value that overlaps an existing override. Reloading the saved model gives the scored population exactly. A calibrated category split is written with `update_stratification_fractions`: the named category gets the value and its complement the rest.

## Checklist

- [ ] Each unknown fraction is `None` in the model.
- [ ] A remainder is named when the calibrated values should keep the total.
- [ ] The remainder is not also calibrated.
- [ ] Category ids name a binary stratification, qualified when the name is shared.

## Next steps

| To learn                            | Read                                                                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| The rules of initial conditions     | [Building Models](https://munqu.github.io/commol/latest/guide/building-models/#initial-conditions)                                         |
| Reporting fitted initial conditions | [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md)                                                                    |
| The design behind this page         | [Per-compartment initial conditions](https://munqu.github.io/commol/latest/development/design-per-compartment-initial-conditions/index.md) |
