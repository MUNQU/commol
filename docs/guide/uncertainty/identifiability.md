# Identifiability

[Back to Uncertainty](index.md)

A parameter the data cannot determine shows up as a wide interval, or as
starts that end apart with equal losses. This page shows how to find out why:
which parameters the model and the data determine, which they do not, and
which combination of parameters the data determine in their place. It is for
anyone who fits more than one parameter.

## Quick path

`k1` and `alpha` enter the model only as their product `k1 * alpha`, so the
data determine the product and not each one.

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
from commol.identifiability import (
    Collinearity,
    SampleIdentifiability,
    StartsIdentifiability,
    StructuralRank,
)
from commol.uncertainty import ParameterSample, SampleMethod


def basic_model(k1: float | None, alpha: float | None, k2: float | None):
    return (
        ModelBuilder(name="Basic Model")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="alpha", value=alpha)
        .add_parameter(id="k2", value=k2)
        .add_transition(
            id="t_ab", source=["A"], target=["B"], rate="k1 * alpha * A * B / total"
        )
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .build(
            Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
            engine="DifferenceEquations",
        )
    )


truth = Simulation(basic_model(0.4, 1.0, 0.1)).run(num_steps=60)
problem = CalibrationProblem(
    observed_data=[
        # B with a measurement error of ±5 %.
        ObservedDataPoint(at=step, output="B", value=truth["B"][step] * (1 + 0.05 * (-1) ** step))
        for step in range(5, 61, 5)
    ],
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.1, max_bound=1.0),
        CalibrationParameter(id="alpha", parameter_type="parameter", min_bound=0.5, max_bound=2.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.01, max_bound=0.5),
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=2000),
    seed=1,
)
calibrator = Calibrator(Simulation(basic_model(None, None, None)), problem)

# Before a fit: the structure of the model and the design.
structural = calibrator.identifiability(StructuralRank(points=5, design="observations"))
for name, verdicts in structural.verdicts.items():
    print(name, verdicts.structural.verdict, verdicts.structural.combinations)
# k1 in_combination ['k1 · alpha']
# alpha in_combination ['k1 · alpha']
# k2 identifiable []

# After a fit: the starts and the collinearity at the best point.
result = calibrator.run(starts=6)
starts = calibrator.identifiability(
    StartsIdentifiability(result=result, parameter_tolerance=0.05, loss_tolerance=0.01)
)
print(starts.starts[0].flagged)  # ['k1', 'alpha']
collinearity = calibrator.identifiability(
    Collinearity(at=result, max_subset_size=3, threshold=10)
)
print([subset.parameters for subset in collinearity.collinearity[0].admissible])
# [['k1', 'k2'], ['alpha', 'k2']]

# One report with every piece of evidence.
report = structural.merge(starts, collinearity)
for name, verdicts in report.verdicts.items():
    print(name, verdicts.structural.verdict, verdicts.practical.verdict)
# k1 in_combination poorly_determined
# alpha in_combination poorly_determined
# k2 identifiable determined
```

Fix `alpha` (or `k1`) at a value, or calibrate their product as one
parameter: the data determine `k1 · alpha`, and the fit says nothing about
how it splits.

## Structural against practical

| Kind | Question | Depends on | Methods |
|---|---|---|---|
| **Structural** | Can any data from this design determine the parameter? | The model and the positions and outputs observed, not the observed values | `StructuralRank` |
| **Practical** | Do these data determine it? | The observed values, their noise and the fit | `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability`, `ProfileIdentifiability` |

A parameter that is not structurally identifiable is never practically
identifiable: no amount of data of the same design fixes it. A parameter that
is structurally identifiable can still be poorly determined by the data at
hand.

## The methods

Every method runs through `Calibrator.identifiability(method)` and returns an
`IdentifiabilityReport`. A threshold or a tolerance has no default: it is a
modelling judgment, and the table gives the range the literature uses.
`StructuralRank` and `Collinearity` differentiate the predictions, so they
leave out the [noise parameters](#noise-parameters).

| Method | Needs | Answers | Engine |
|---|---|---|---|
| `StructuralRank(points, design)` | The problem only | The rank of the sensitivities, the non-identifiable parameters and the identifiable combinations | Deterministic |
| `Collinearity(at, max_subset_size, threshold)` | A fit (`CalibrationResult`) or a sample | Which subsets of parameters can be fitted together; `threshold` from 5 to 20 | Deterministic |
| `SampleIdentifiability(sample, reference, correlation_threshold, priors)` | Any `ParameterSample` | How much the sample contracts each parameter against its bounds or its prior, which pairs move together, which marginals have several modes; `correlation_threshold` from 0.8 to 0.95 | Any |
| `StartsIdentifiability(result, parameter_tolerance, loss_tolerance)` | A result of `Calibrator.run(starts=n)` with two or more converged starts | Which parameters differ between starts of equal loss | Any |
| `ProfileIdentifiability(result)` | A `ProfileResult` of the parameters of the problem, in problem order | The verdict of the shape of each profile ([Profile Likelihood](profiles.md#verdicts-and-the-identifiability-report)); it runs no model | Any |

### Structural rank

`StructuralRank` evaluates the sensitivities of the predicted observations at
`points` rows of a Latin hypercube over the bounds, drawn from the seed of
the calibrator (`Calibrator.seed`), so `problem.seed` repeats them. The
sensitivity of observation `i` to parameter `j` is scaled to a relative
change of both, `S_ij = (∂y_i/∂θ_j) · θ_j / s_i`, where `s_i` is the root mean
square of the series of observation `i`, so the answer does not depend on
units.

| `design` | Positions and outputs | Question |
|---|---|---|
| `"observations"` | Those of the problem's observations; their values are ignored | Can this design determine the parameters? |
| `"dense"` | Every observed series, with its outputs, factor and window, at every unit position from 0 to the last observation | Can denser data of the same series determine them? |

At each point the rank counts the singular values of `S` above the norm of the
estimated error of `S`. The **generic rank** is the rank at most points; each
point's singular values are in `report.structural[0].points`. A parameter is:

| Verdict | Meaning |
|---|---|
| `identifiable` | No combination of the other parameters undoes a change of it |
| `in_combination` | Only the combinations in `combinations` are identifiable |
| `non_identifiable` | No observation changes with it |

Only the design of the observations gives the structural verdict of a
parameter; a dense design keeps its verdicts in its section.

```python
dense = calibrator.identifiability(StructuralRank(points=5, design="dense"))
print(dense.structural[0].generic_rank, dense.structural[0].verdicts["k1"])
# 2 in_combination
```

Denser data of `B` do not split the product either: the deficit is in the
model, not in the design.

#### When the solver tolerance sets the rank

The estimated error of `S` grows with the tolerance of an ODE solver, and it
can hide a singular value that a tighter tolerance resolves. With
`DifferentialEquations`, each point whose rank is below the number of
parameters differentiated is evaluated again at `rtol / 100` and
`atol / 100`, no lower than `1e-14`, and the verdicts are computed again with
those ranks. A parameter whose verdict changes there has no structural
verdict: its `verdict` is `None`, and its reason names `ODESolver.rtol` and
`ODESolver.atol`. When a point cannot run at the tighter tolerances, every
parameter that is not `identifiable` has no verdict, and the section lists
why in `unchecked`. A parameter that no observation reads stays
`non_identifiable` when every point runs at the tighter tolerances.
`DifferenceEquations` runs are exact up to rounding: no point is evaluated
again.

| Field of `PointRank` | Meaning |
|---|---|
| `accuracy` | `relative` (`rtol`) and `absolute` (`atol`) accuracy of the runs |
| `rank` | The rank at that accuracy |
| `tighter` | The `accuracy`, the singular values, the `threshold` and the `rank` at the tighter tolerances; `None` when the point was not evaluated again |

The section gives `tighter_generic_rank`, the generic rank with the tighter
rank of each point evaluated again, and the reason of a verdict that stays
names it when it differs from `generic_rank`.

A verdict of `None` says that the result depends on the tolerance. Tighten
`ODESolver.rtol` and `ODESolver.atol` and run `StructuralRank` again, until
the rank stops changing. A fit can reach its loss at looser tolerances; the
finite differences may not.

### How to read a combination

A combination is a product of powers of the parameters that the data
determine: `k1 · alpha` means that every pair `(k1, alpha)` with the same
product fits the data equally well. `k1 / k2^2` means that `k1` and `k2²` can
grow together. A parameter whose bounds allow negative values reads as its
absolute value, `|k1|`. The exponents are rounded to the nearest half.

### Collinearity

`Collinearity` computes, after Brun et al. (2001), the sensitivity measure of
each parameter and the collinearity index `γ` of each subset: 1 when the
sensitivities of the subset are orthogonal, infinite when one is a combination
of the others. A subset with `γ` below `threshold` can be fitted together;
`admissible` lists the largest such subsets. With a sample, the index is
computed at every draw and summarized by its weighted median, and
`share_below` gives the share of the draws below the threshold.

A parameter is `poorly_determined` when a pair that holds it has an index at
or above the threshold, or no observation is sensitive to it.

### Noise parameters

A noise parameter of an [observation model](observation-models.md#noise-parameters)
(`parameter_type="noise"`), such as the `overdispersion` of
`NegativeBinomialObservations`, changes the likelihood and never a
prediction: its column of the sensitivities is zero. `StructuralRank` and
`Collinearity` leave it out.

| Method | A noise parameter |
|---|---|
| `StructuralRank` | Not in the rank, the singular values, `projections` or `verdicts` of the section; its structural `verdict` is `None`, with the reason. The generic rank is at most the number of the other parameters |
| `Collinearity` | Not in `sensitivity`, `pair_gamma` or a subset; no other parameter is collinear with it; no practical verdict. `max_subset_size` is at most the number of the other parameters, and it needs at least 2 of them |

Each section names the parameters it leaves out in `left_out`, each with the
reason. The verdict of a noise parameter comes from the methods that read the
fit: `StartsIdentifiability`, `SampleIdentifiability` and the profiles. A
problem whose calibrated parameters are all noise parameters has nothing to
differentiate: both methods raise a `ValueError`.

```python
from commol.observations import NegativeBinomialObservations

counts = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(at=point.at, output="B", value=float(round(point.value)))
        for point in problem.observed_data
    ],
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.1, max_bound=1.0),
        CalibrationParameter(id="phi", parameter_type="noise", min_bound=0.0, max_bound=1.0),
    ],
    observation_models={"B": NegativeBinomialObservations(overdispersion="phi")},
    optimization_config=NelderMeadConfig(max_iterations=2000),
    seed=1,
)
with_noise = Calibrator(Simulation(basic_model(None, 1.0, 0.1)), counts)
rank = with_noise.identifiability(StructuralRank(points=5, design="observations"))
print(rank.structural[0].generic_rank, rank.structural[0].left_out)
# 1 {'phi': 'a noise parameter of an observation model, which no prediction reads'}
print(rank.verdicts["phi"].structural.verdict)  # None
```

### A sample

`SampleIdentifiability` reads any sample, such as the draws of an uncertainty
method. Here the sample is the ends of the starts.

```python
names = calibrator.parameter_names
sample = ParameterSample.equally_weighted(
    names,
    [(p.min_bound, p.max_bound) for p in problem.parameters],
    [[start.end[name] for name in names] for start in result.starts],
    SampleMethod(name="starts"),
)
from_sample = calibrator.identifiability(
    SampleIdentifiability(sample=sample, reference="bounds", correlation_threshold=0.9)
)
pair = from_sample.sample[0].correlated_pairs[0]
print(pair.first, pair.second, round(pair.correlation, 1))  # k1 alpha -1.0
```

| Field | Meaning |
|---|---|
| `contraction` | `1 − Var(sample) / Var(reference)`: near 0 when the sample spreads over the bounds, near 1 when it sits on one value |
| `correlated_pairs` | The pairs whose absolute correlation is above `correlation_threshold` |
| `modes` | The modes of each marginal: the maxima of a kernel density above a tenth of the highest |

The `reference` of the contraction is `"bounds"`, the uniform distribution on
the bounds, or `"prior"` with `priors`, the prior of each parameter: the
sample of a [Gibbs posterior or ABC](smc.md) contracts its priors by how much
the data inform each parameter, and a sample drawn from the priors themselves
has a contraction near 0.

```python
from commol.distributions import Priors, Uniform

against_priors = calibrator.identifiability(
    SampleIdentifiability(
        sample=sample,
        reference="prior",
        priors=Priors(distributions={name: Uniform() for name in names}),
        correlation_threshold=0.9,
    )
)
section = against_priors.sample[0]
print(section.reference, section.contraction == from_sample.sample[0].contraction)  # prior True
```

A parameter is `poorly_determined` when it is in a correlated pair or its
marginal has two or more modes.

### Starts

`StartsIdentifiability` compares the converged starts whose loss exceeds the
best by at most `loss_tolerance` of the best loss, and flags a parameter whose
ends among them differ by more than `parameter_tolerance` of its bound width.
A loss near zero, as with data made by the model itself, leaves no start
within a relative tolerance but the best; use data with noise.

## The report

An `IdentifiabilityReport` holds, for each parameter, a **structural** and a
**practical** verdict, and the sections of the methods that ran
(`structural`, `collinearity`, `sample`, `starts`, `profile`). The profiles of
a `ProfileLikelihood` give their report with
`calibrator.identifiability(ProfileIdentifiability(result=profiles))`, which
runs no model; a flat
profile gives the structural verdict `in_combination` when resetting the
other parameters to their fitted values breaks it, and `non_identifiable`
when nothing the data read depends on the parameter
([Profile Likelihood](profiles.md#verdicts-and-the-identifiability-report)). `merge` joins reports: a
verdict the reports agree on stays, with every reason; one they disagree on is
`conflicting`, with both reasons. A structural verdict of `None`, as that of
a noise parameter, adds its reasons and leaves the verdicts of the other
reports. `report.report()` gives it as plain data
([Reports](../reports.md#the-report-of-an-identifiability-analysis)).

```python
data = report.report()
print(data["parameters"]["k2"]["practical"]["verdict"])  # determined
```

## Limits

- **Local.** The sensitivities are derivatives at points: a parameter can be
  identifiable near every point and still have two distant values that fit
  equally well. Symbolic methods give a global answer for differential
  equations; they are outside commol.
- **Numerical.** The derivatives are finite differences; a singular value
  counts only when it is above their estimated error. The error grows with
  the tolerance of an ODE solver (`rtol`, `atol`): a rank that a hundredth of
  the tolerances raises gives no verdict, but one evaluation at a tighter
  tolerance cannot show that a yet tighter one changes nothing.
- **Generic points.** A point where the rank is lower or higher than at most
  points is special; the generic rank is the rank at most points.
- **Deterministic engines.** The runs of a stochastic engine change in jumps
  with the parameters, so `StructuralRank` and `Collinearity` refuse one;
  `SampleIdentifiability` and `StartsIdentifiability` accept it.

## Checklist

- [ ] You checked the structural rank of the design before you fitted.
- [ ] Each parameter `in_combination` is fixed, or replaced by its combination.
- [ ] The thresholds you chose are in the report you share.

## Next steps

| To learn | Read |
|---|---|
| Several starts of an optimizer | [Optimizers](../calibration-optimizers.md#several-starts) |
| Scale factors, which often pair with a rate | [Observations](../calibration-observations.md) |
| What a sample is | [Uncertainty](index.md#the-parameter-sample) |
| Intervals and verdicts from profiles | [Profile Likelihood](profiles.md) |
