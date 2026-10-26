# Examples

[Back to Home](https://munqu.github.io/commol/latest/index.md)

Complete models with abstract names. Each example runs on its own, shows one technique, and reads or draws its results. They follow the order of the guide: rates and stratifications first, then other engines, then a fit, its uncertainty and a comparison of scenarios.

| Example                                                                                             | Technique                                                                      | Guide                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [1. A chain of states](#example-1-a-chain-of-states)                                                | Absolute and proportional rates; read and plot the results                     | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md)                                                                                    |
| [2. A periodic rate](#example-2-a-periodic-rate)                                                    | `sin` of `step` in a rate                                                      | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md)                                                              |
| [3. A rate per group](#example-3-a-rate-per-group)                                                  | Stratified rates; a figure of the sum of a bin                                 | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md)                                                                                        |
| [4. Rates of an intersection](#example-4-rates-of-an-intersection)                                  | Two stratifications, `total_<category>`, `per_compartment`                     | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md)                                                                                        |
| [5. A conditional stratification](#example-5-a-conditional-stratification)                          | A stratification that splits one group only                                    | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/#conditional-stratifications)                                                                |
| [6. A capacity limit](#example-6-a-capacity-limit)                                                  | `max` in a rate                                                                | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md)                                                              |
| [7. A cycle in continuous time](#example-7-a-cycle-in-continuous-time)                              | A transition back to the first state, run as differential equations            | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/#differential-equations)                                                                 |
| [8. Rates that follow a pattern](#example-8-rates-that-follow-a-pattern)                            | `AxisPattern` as a rate, and a schedule per group                              | [Axis Patterns](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md)                                                                                    |
| [9. Runs at random](#example-9-runs-at-random)                                                      | Replicates of a stochastic engine, their mean and band                         | [Stochastic Simulation](https://munqu.github.io/commol/latest/guide/stochastic/index.md)                                                                               |
| [10. A fit, its uncertainty and two scenarios](#example-10-a-fit-its-uncertainty-and-two-scenarios) | `Calibrator`, a bootstrap, `Predictive`, `Scenarios`, `plot_compared`, threads | [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md), [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md) |

## Example 1: A chain of states

`A` moves to `B` at an absolute rate, which names compartments, and `B` moves to `C` at a proportional rate, which names none. A fourth state `M` sits between `A` and `B`:

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Chain Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="M", name="Intermediate")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.4, description="A to M rate")
    .add_parameter(id="k2", value=0.2, description="M to B rate")
    .add_parameter(id="k3", value=0.1, description="B to C rate")
    .add_transition(id="flow_AM", source=["A"], target=["M"], rate="k1 * A * B / total")
    .add_transition(id="flow_MB", source=["M"], target=["B"], rate="k2")
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k3")
    .build(
        Aggregate(bin_amounts={"A": 999, "M": 0, "B": 1, "C": 0}),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=100)

peak = max(results["B"])
print(f"B peaks at {peak:.0f} at step {results['B'].index(peak)}")
results.plot.series(output_file="chain.png")
```

## Example 2: A periodic rate

The rate of `A` to `B` varies with a period of 365 steps; `step` is the position on the axis and `pi` a constant of the [expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md):

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Periodic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k_mean", value=0.3)
    .add_parameter(id="k_amp", value=0.2)
    .add_parameter(id="k2", value=0.1)
    .add_transition(
        id="periodic_flow",
        source=["A"],
        target=["B"],
        rate="k_mean * (1 + k_amp * sin(2 * pi * step / 365)) * A * B / total",
    )
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .build(
        Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=365 * 3)
results.plot.series(series=["B"], tick_every=365, output_file="periodic.png")
```

## Example 3: A rate per group

A stratification splits every bin into three groups, and the transition from `B` to `C` has one rate per group. In a figure, a bin id such as `B` draws the sum of its compartments:

```python
from commol import (
    Aggregate,
    ModelBuilder,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)

model = (
    ModelBuilder(name="Stratified Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_stratification(id="group", categories=["g1", "g2", "g3"])
    .add_parameter(id="k1", value=0.0003)
    .add_parameter(id="k2_g1", value=0.15)
    .add_parameter(id="k2_g2", value=0.12)
    .add_parameter(id="k2_g3", value=0.08)
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1")
    .add_transition(
        id="flow_BC",
        source=["B"],
        target=["C"],
        stratified_rates=[
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="group", category="g1")
                ],
                rate="k2_g1",
            ),
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="group", category="g2")
                ],
                rate="k2_g2",
            ),
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="group", category="g3")
                ],
                rate="k2_g3",
            ),
        ],
    )
    .build(
        Aggregate(
            bin_amounts={"A": 9900, "B": 100, "C": 0},
            stratification_fractions={"group": {"g1": 0.25, "g2": 0.55, "g3": 0.2}},
        ),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=100)

for name in ["B_g1", "B_g2", "B_g3"]:
    print(f"{name}: {results[name][-1]:.2f}")
results.plot.series(series=["B", "B_g1", "B_g3"], output_file="groups.png")
```

## Example 4: Rates of an intersection

Two stratifications, and a rate for the compartments of type `t2` only. In group `g1`, the rate divides by the total of the subgroup, `total_g1`, instead of the total of the model. `per_compartment=True` makes `A` and `B` in the rate the compartments of each flow, so each compartment of `A` loses an amount computed from its own amount:

```python
from commol import (
    Aggregate,
    ModelBuilder,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)

model = (
    ModelBuilder(name="Intersection Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_stratification(id="type", categories=["t1", "t2"])
    .add_parameter(id="k1_low", value=0.3)
    .add_parameter(id="k1_high", value=0.6)
    .add_parameter(id="k2", value=0.1)
    .add_transition(
        id="flow_AB",
        source=["A"],
        target=["B"],
        rate="k1_low * A * B / total",
        stratified_rates=[
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="group", category="g1")
                ],
                rate="k1_low * A * B / total_g1",
            ),
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="type", category="t2")
                ],
                rate="k1_high * A * B / total",
            ),
        ],
        per_compartment=True,
    )
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .build(
        Aggregate(
            bin_amounts={"A": 990, "B": 10, "C": 0},
            stratification_fractions={
                "group": {"g1": 0.6, "g2": 0.4},
                "type": {"t1": 0.8, "t2": 0.2},
            },
        ),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=100)
print(sorted(name for name in results if name.startswith("B_")))
# ['B_g1_t1', 'B_g1_t2', 'B_g2_t1', 'B_g2_t2']
```

## Example 5: A conditional stratification

The stratification `subtype` splits only the compartments of group `g2`; the compartments of `g1` are not split further:

```python
from commol import (
    Aggregate,
    ModelBuilder,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)

model = (
    ModelBuilder(name="Conditional Stratification Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_stratification(
        id="subtype",
        categories=["s1", "s2"],
        conditions=[StratificationCondition(stratification="group", category="g2")],
    )
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2_default", value=0.1)
    .add_parameter(id="k2_s1", value=0.15)
    .add_parameter(id="k2_s2", value=0.05)
    .add_transition(
        id="flow_AB",
        source=["A"],
        target=["B"],
        rate="k1 * A * B / total",
        per_compartment=True,
    )
    .add_transition(
        id="flow_BC",
        source=["B"],
        target=["C"],
        rate="k2_default",
        stratified_rates=[
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="subtype", category="s1")
                ],
                rate="k2_s1",
            ),
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="subtype", category="s2")
                ],
                rate="k2_s2",
            ),
        ],
    )
    .build(
        Aggregate(
            bin_amounts={"A": 9900, "B": 100, "C": 0},
            stratification_fractions={
                "group": {"g1": 0.6, "g2": 0.4},
                "subtype": {"s1": 0.7, "s2": 0.3},
            },
        ),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=100)
print(sorted(name for name in results if name.startswith("B")))
# ['B_g1', 'B_g2_s1', 'B_g2_s2']
```

## Example 6: A capacity limit

The flow from `B` slows as `C` fills, and stops when `C` reaches `capacity`. The rate names `B`, so it is the flow itself; the factor `max(0, 1 - C / capacity)` never turns it negative. A difference model moves the flow of a whole step at once, so the last step can pass `capacity` by a little:

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Capacity Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.5)
    .add_parameter(id="k2_max", value=0.15)
    .add_parameter(id="capacity", value=100.0)
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(
        id="flow_BC",
        source=["B"],
        target=["C"],
        rate="k2_max * B * max(0, 1 - C / capacity)",
    )
    .build(
        Aggregate(bin_amounts={"A": 950, "B": 50, "C": 0}),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=100)
print(round(results["C"][-1], 1))  # 100.0
```

## Example 7: A cycle in continuous time

Amounts in `C` return to `A`, so the process never ends. The same process runs as differential equations: only the engine changes, and the results can be read at any position, here every half step:

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Cyclic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_parameter(id="k3", value=0.01, description="C to A rate")
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .add_transition(id="return_CA", source=["C"], target=["A"], rate="k3")
    .build(
        Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
        engine="DifferentialEquations",
    )
)

simulation = Simulation(model)
results = simulation.run_at([step / 2 for step in range(2001)])
print(results.positions[:3])  # [0.0, 0.5, 1.0]
print(round(sum(results[name][-1] for name in ["A", "B", "C"])))  # 1000
```

## Example 8: Rates that follow a pattern

An `AxisPattern` is a rate that changes along the axis. The first model combines an opening pulse, an event every 30 steps and a sinusoid. The second gives each group its own schedule: the first `add_group` call is on the class, the next ones on the pattern it returns:

```python
from commol import (
    Aggregate,
    AxisPattern,
    ModelBuilder,
    Simulation,
    StratificationCondition,
)

combined = AxisPattern.combine(
    AxisPattern.pulse(at=0, value=0.3),
    AxisPattern.periodic(period=30, value=0.05),
    AxisPattern.sinusoid(amplitude=0.005, period=365, baseline=0.01),
)
model = (
    ModelBuilder(name="Pattern Model")
    .add_bin(id="A", name="Source")
    .add_bin(id="B", name="Sink")
    .add_transition(id="flow", source=["A"], target=["B"], rate=combined)
    .build(Aggregate(bin_amounts={"A": 10000, "B": 0}), engine="DifferenceEquations")
)
results = Simulation(model).run(num_steps=365)

per_group = AxisPattern.add_group(
    conditions=[StratificationCondition(stratification="group", category="cat1")],
    schedule=AxisPattern.periodic(period=7, value=0.05),
).add_group(
    conditions=[StratificationCondition(stratification="group", category="cat2")],
    schedule=AxisPattern.periodic(period=30, value=0.05),
)
grouped = (
    ModelBuilder(name="Schedule Model")
    .add_bin(id="A", name="Source")
    .add_bin(id="B", name="Sink")
    .add_stratification(id="group", categories=["cat1", "cat2"])
    .add_transition(id="flow", source=["A"], target=["B"], rate=per_group)
    .build(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0},
            stratification_fractions={"group": {"cat1": 0.5, "cat2": 0.5}},
        ),
        engine="DifferenceEquations",
    )
)
grouped_results = Simulation(grouped).run(num_steps=60)
print(grouped_results["B_cat1"][-1] > grouped_results["B_cat2"][-1])  # True
```

## Example 9: Runs at random

The `StochasticSteps` engine moves whole units at random, one step at a time. `replicates` runs the model once per seed; the mean, the band and the figure reduce the runs position by position:

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Stochastic Model")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .build(Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}), engine="StochasticSteps")
)

replicates = Simulation(model).replicates(40, seed=7).run(num_steps=100)
mean = replicates.mean()
spread = replicates.member_statistics("C", confidence_level=0.9)
print(round(mean["C"][-1]), round(spread["ci_lower"][-1]), round(spread["ci_upper"][-1]))
replicates.plot.series(series=["B", "C"], confidence_level=0.9, output_file="replicates.png")
```

## Example 10: A fit, its uncertainty and two scenarios

A fit of `k1` to noisy observations of `B`, a sample of `k1` from a Bayesian bootstrap, its predictive band, and a comparison of two scenarios of `k2`. The work of the bootstrap and of the draws runs on two threads:

```python
import numpy as np

import commol.parallelism
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
from commol.resampling import BayesianBootstrap
from commol.uncertainty import Predictive, Scenarios
from commol.quantities import ValueAt
from commol.plotting import plot_compared


def basic_model(k1: float | None):
    return (
        ModelBuilder(name="Basic Model")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="k2", value=0.1)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .build(
            Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
            engine="DifferenceEquations",
        )
    )


truth = Simulation(basic_model(0.4)).run(num_steps=40)
rng = np.random.default_rng(1)
observed_data = [
    ObservedDataPoint(at=step, output="B", value=truth["B"][step] * float(np.exp(rng.normal(0.0, 0.05))))
    for step in range(2, 41, 2)
]
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=500),
    seed=42,
)
calibrator = Calibrator(Simulation(basic_model(None)), problem)

with commol.parallelism.threads(2):
    fit = calibrator.run()
    bootstrap = calibrator.uncertainty(BayesianBootstrap(replicates=20, block_length=1))
    predictive = Predictive(calibrator, bootstrap.sample, positions=list(range(61)))
    scenarios = Scenarios(
        calibrator,
        bootstrap.sample,
        {"fitted": {}, "slower": {"k2": 0.05}},
        positions=list(range(61)),
    )

print(round(fit.best_parameters["k1"], 2))  # close to 0.4
lower, upper = bootstrap.sample.interval(0.9)["k1"]
print(round(lower, 3), round(upper, 3))  # the 90 % interval of k1
predictive.plot.series(series=["B"], observed_data=observed_data, output_file="band.png")

[difference] = scenarios.compare("slower", "fitted", [ValueAt(series="C", at=60)], 0.9)
print(difference.label, round(difference.median))  # how much less C reaches by step 60
plot_compared(scenarios, series=["B", "C"], output_file="scenarios.png")
```

## Next steps

| To learn                                              | Read                                                                                    |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Every builder option                                  | [Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md) |
| Reading, aggregating and plotting results             | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md) |
| Calibration in depth                                  | [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)   |
| The uncertainty methods, the predictive and scenarios | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)         |
| Every class and argument                              | [API Reference](https://munqu.github.io/commol/latest/api/index.md)                     |
