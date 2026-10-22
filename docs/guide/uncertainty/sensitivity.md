# Sensitivity Analysis

[Back to Uncertainty](index.md)

A sensitivity analysis asks the forward questions of a model: how uncertainty
in its inputs moves its outputs, and which inputs matter most. It needs no
data. You say what varies and how (an `InputSpace`), what to read from each
run (quantities of interest), and a method: Monte Carlo propagation, partial
rank correlation (PRCC), Morris screening, Sobol indices or a polynomial chaos
expansion. Every method runs through `SensitivityAnalysis(simulation,
inputs).run(method)`, in one call into Rust: the design
(`crates/commol-inference/src/design.rs`), the runs (`batch.rs`) and the
estimator (`global_sensitivity/`), joined by `ForwardAnalysis`
(`global_sensitivity/analysis.rs`).

## Quick path

```python
from commol import (
    Aggregate,
    CalibrationParameter,
    InputSpace,
    ModelBuilder,
    Normal,
    Peak,
    PRCC,
    SensitivityAnalysis,
    Simulation,
    Uniform,
)


def abc_model(k1: float, k2: float):
    return (
        ModelBuilder(name="ABC")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_bin(id="C", name="C")
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="k2", value=k2)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .build(
            Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
            engine="DifferenceEquations",
        )
    )


simulation = Simulation(abc_model(0.3, 0.1))
inputs = InputSpace.independent(
    [
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.2, max_bound=0.6),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.05, max_bound=0.2),
    ],
    {"k1": Uniform(), "k2": Normal(mean=0.1, sd=0.03)},
)
analysis = SensitivityAnalysis(simulation, inputs, seed=1)

peak = Peak(outputs="B", positions=list(range(101)))
result = analysis.run(PRCC(draws=200, quantities=[peak]))
print(result.inputs, [round(c, 2) for c in result.coefficients[0]])  # ['k1', 'k2'] [0.97, -0.95]
```

The peak of `B` rises with `k1` and falls with `k2`: the PRCC of `k1` is near
1 and that of `k2` near −1. Over the axis (below), `C` early on rises with
both, since more recovery fills `C` faster, and late falls with `k2`, since
more recovery stops the outbreak sooner.

## Forward and inverse questions

| Question | Needs data | Entry point |
|---|---|---|
| Which inputs fit the data, and how well do the data determine them? | Yes | `Calibrator.run`, `Calibrator.uncertainty`, `Calibrator.identifiability` |
| How does uncertainty in the inputs move the outputs? | No | `SensitivityAnalysis.run(Propagation(...))` |
| Which inputs matter most for an output? | No | `SensitivityAnalysis.run` with `PRCC`, `Morris`, `Sobol` or `PolynomialChaos` |

The two meet: a sample that a calibration returns, such as a posterior, can be
the input space of an analysis (`InputSpace.from_sample`).

## What varies

Each input is a `CalibrationParameter`: a model parameter or an initial
condition, with its bounds, as a calibration sets it. The values reach the
engine as a calibration writes them (`ForwardModel::unobserved`,
`crates/commol-calibration/src/forward.rs`).

| Constructor | Inputs | Methods that take it |
|---|---|---|
| `InputSpace.independent(parameters, distributions)` | One distribution per input (`Uniform()`, `LogUniform()`, `Beta(a, b)`, `Normal(mean, sd)`, `LogNormal(mu, sigma)`, `Gamma(shape, rate)`), truncated to its bounds, independent of the others | Every method |
| `InputSpace.uniform_on_bounds(parameters)` | `Uniform()` on the bounds of every input; they must be finite | Every method |
| `InputSpace.from_sample(sample, parameters)` | The draws of a `ParameterSample`, with their weights; the inputs may depend on each other. The parameters give the type of each value, which a sample does not hold | `Propagation` and `PRCC` |

An initial condition is an amount in the unit of its bin, or a category split
in `[0, 1]`, with `initial_condition_remainder` as in a calibration problem.
A factor or a noise parameter is not an input: both belong to observations,
and an analysis has none.

`SensitivityAnalysis(simulation, inputs, replicates=None, seed=None)` holds
the inputs for every method you run on it. With a stochastic engine
`replicates` (at least 1) and `seed` are required. A point of the design of
PRCC, Morris, Sobol or a polynomial chaos runs `replicates` replicates, on the
same seeds at every point, and holds their mean: the indices are those of the
mean of `replicates` replicates, and they come close to those of the expected
value as `replicates` grows. A propagation, and the `Predictive` of its draws,
keep every replicate, each with a seed of its own. A deterministic engine
does not use `replicates`, so the same analysis runs on either engine; with no
seed, one is drawn and logged.

## Quantities of interest

A quantity reduces each run to a scalar, in Rust, through the observation
operator of the calibration (`crates/commol-inference/src/quantity.rs`), so a
design of thousands of runs never returns to Python. `outputs` is one output
or a list of outputs summed, as in an observation.

| Quantity | Value |
|---|---|
| `ValueAt(outputs, at)` | The value at `at` |
| `Gain(outputs, start, end)` | The value at `end` minus the value at `start`: what entered a cumulative output over `(start, end]`, as an observation with a window |
| `Peak(outputs, positions)` | The largest value at the positions |
| `PeakPosition(outputs, positions)` | The first of the positions where the value is largest |
| `Trajectory(outputs, positions)` | One `ValueAt` per position, for an analysis over the axis |

Each quantity has a `label`, by default what it reads, such as `"B at 10"`.
The default label writes each position in full, as Python reads it back, so
two positions give two labels: `"B at 1000000"` and `"B at 1000001"`.

## Which method for which budget

| Method | Answers | Runs | Inputs |
|---|---|---|---|
| `Propagation(draws, quantities)` | The distribution of each quantity, and the band of every output | `draws` | Independent or a sample |
| `PRCC(draws, quantities)` | How steadily each quantity rises or falls with each input | `draws`, more than the inputs + 1 | Independent or a sample |
| `Morris(trajectories, levels, quantities)` | Which inputs matter, in few runs | `trajectories · (inputs + 1)` | Independent |
| `Sobol(base_draws, bootstrap, confidence_level, quantities)` | The share of the variance each input explains, alone and with interactions, with intervals | `base_draws · (inputs + 2)` | Independent |
| `PolynomialChaos(degree, draws, quantities)` | A polynomial surrogate and its Sobol indices, in far fewer runs for a smooth quantity | `draws`, more than the terms | Independent |

Start with Morris to drop the inputs that do not matter, then Sobol or a
polynomial chaos on the rest. For a slow model, a polynomial chaos needs the
fewest runs; read its leave-one-out error before its indices.

## Propagation

`Propagation` returns a `PropagationResult`: the `sample` of the draws (a
Latin hypercube of the distributions, or the input sample as it is), the
quantities of each run, and their `weights`. `summary(confidence_level)` gives
the weighted mean, median and central interval of each quantity. The sample is
a `ParameterSample`, so `Predictive(analysis, result.sample, positions)` gives
the bands and plots of a calibration.

```python
from commol import Predictive, Propagation, ValueAt

result = analysis.run(
    Propagation(draws=100, quantities=[peak, ValueAt(outputs="C", at=100)])
)
for summary in result.summary(0.9):
    print(summary.label, round(summary.lower), round(summary.median), round(summary.upper))
# peak of B 172 421 655
# C at 100 809 980 999

band = Predictive(analysis, result.sample, [0.0, 25.0, 50.0]).band(0.9)
print([round(value) for value in band.median["B"]])  # [10, 293, 53]
```

`Predictive.at_observations` of an analysis takes `NoNoise()` only: residual
noise needs the residuals of a fit, and `DistributionNoise()` the observation
models of a problem.

## PRCC

`PRCC` (Marino et al. 2008) ranks the inputs and each quantity, and gives the
correlation of the rank of an input with the rank of the quantity once the
other inputs are accounted for. A coefficient near 1 or −1 says the quantity
rises or falls steadily with the input; near 0, that it does not. `p_values`
test each coefficient against 0.

With inputs from a sample, the sample is the design, and the ranks, the
correlations and the test read its weights: a rank is the weight below a
value plus half the weight at it, and the test counts Kish's effective size
in place of the draws. Equal weights give the usual PRCC. A `Trajectory` gives
the coefficients over the axis; a quantity with one value in every run, such
as a state at position 0, has NaN coefficients.

A NaN has no rank, so a quantity that is NaN at a draw raises a `ValueError`
before PRCC ranks any value. The message gives the index of the quantity, in
the order of the labels, and the draws where it is NaN; a `Propagation` with
the same inputs and seed gives the value of each run.

```python
from commol import Trajectory

over_axis = analysis.run(
    PRCC(draws=100, quantities=[Trajectory(outputs="C", positions=[10, 50, 100])])
)
print(over_axis.labels)  # ['C at 10', 'C at 50', 'C at 100']
print([[round(c, 2) for c in row] for row in over_axis.coefficients])
# [[0.99, 0.9], [0.91, 0.31], [0.96, -0.89]]
```

## Morris screening

`Morris` (Morris 1991) runs `trajectories` trajectories: each starts at a
random level of every input and moves the inputs one at a time, by half the
`levels`. The levels are the centres of strata of equal probability of each
distribution, so a level is finite for an unbounded distribution too, and an
effect is on the scale of `F(x)`, so inputs of different units compare.
`mu_star` (Campolongo, Cariboni and Saltelli 2007) ranks the inputs; a `sigma`
near or above `mu_star` says the effect depends on the other inputs or is not
linear.

```python
from commol import Morris

screening = analysis.run(Morris(trajectories=20, levels=4, quantities=[peak]))
print([round(value) for value in screening.mu_star[0]], screening.runs)  # [385, 279] 60
```

## Sobol indices

`Sobol` uses the design of Saltelli (2010) on two matrices of independent
Monte Carlo draws. The first-order index of an input (`first`) is the share of
the variance of the quantity it explains alone; the total index (`total`) adds
every interaction it takes part in, so `first ≤ total` up to the Monte Carlo
error that the intervals show: below, the estimates of `k1` cross, and their
intervals overlap. The intervals are percentiles of a bootstrap
of the rows, at `confidence_level`.

Sobol, Morris and the polynomial chaos need independent inputs. With
dependent inputs, such as a posterior, moving one input breaks its dependence
on the others, and the indices lose their meaning and no longer add up; an
input space from a sample raises a `ValueError` that names PRCC and
propagation, which take it.

```python
from commol import Sobol

indices = analysis.run(
    Sobol(base_draws=1024, bootstrap=200, confidence_level=0.9, quantities=[peak])
)
for name, i in [("k1", 0), ("k2", 1)]:
    first = (indices.first_lower[0][i], indices.first[0][i], indices.first_upper[0][i])
    total = (indices.total_lower[0][i], indices.total[0][i], indices.total_upper[0][i])
    print(name, [round(s, 2) for s in first], [round(s, 2) for s in total])
# k1 [0.47, 0.63, 0.78] [0.55, 0.59, 0.63]
# k2 [0.21, 0.36, 0.48] [0.38, 0.41, 0.45]
```

## Polynomial chaos

`PolynomialChaos` (Sudret 2008) takes every input to `u = F(x)`, uniform on
`[0, 1]`, and fits orthonormal Legendre polynomials of `u` of total degree
`degree` by least squares on a Latin hypercube of `draws` points. One basis
fits every distribution, truncated ones too. The Sobol indices come from the
squares of the coefficients. `loo_error` is the leave-one-out error relative
to the variance: the indices are those of the expansion, and they are those
of the model only as far as the expansion predicts it. `predict` gives the
quantities at any input values, as a surrogate.

```python
from commol import PolynomialChaos

chaos = analysis.run(PolynomialChaos(degree=4, draws=60, quantities=[peak]))
print(round(chaos.loo_error[0], 4), [round(s, 2) for s in chaos.total[0]])  # 0.0036 [0.61, 0.39]
print([round(value) for value in chaos.predict([[0.4, 0.1]])[0]])  # [428]
```

## Plots

Each result draws its own figure with `plot()`, one panel per quantity, and
returns the matplotlib `Figure` ([Plotting](../simulations.md#plotting): the
extra `commol[plot]`, `output_file`, `config` and `axes`). The `plot` of a
`PRCCResult` draws one line per input over the positions of a `Trajectory`
and one bar per input for any other quantity. The `plot` of a `MorrisResult`
draws `sigma` against `mu_star`, one point per input. The `plot` of a
`SobolResult` draws the first-order and total indices with their intervals,
and the `plot` of a `ChaosResult` draws those of the expansion.

```python
figure = over_axis.plot(x_label="Step")
figure = screening.plot()
figure = indices.plot()
```

## Checklist

- [ ] Each input has bounds and a distribution you can defend; a uniform
      needs finite bounds.
- [ ] Morris, Sobol and the polynomial chaos run on independent inputs; a
      posterior goes to PRCC or propagation.
- [ ] A stochastic engine has `replicates` and a `seed`.
- [ ] The intervals of Sobol are narrow enough to rank the inputs, and none is
      NaN (a bootstrap replicate of no variance); if not, raise `base_draws`.
- [ ] The leave-one-out error of a polynomial chaos is small before you read
      its indices, and `predict` reads values inside the bounds of the inputs.

## Next steps

| To learn | Read |
|---|---|
| The parameter sample, the predictive and its scores | [Uncertainty](index.md) |
| A posterior to propagate | [Sequential Monte Carlo](smc.md), [Observation Models](observation-models.md) |
| Which parameters the data determine | [Identifiability](identifiability.md) |
