# Identifiability

[Back to Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)

A parameter the data cannot determine shows up as a wide interval, or as starts that end apart with equal losses. This page shows how to find out why: which parameters the model and the data determine, which they do not, and which combination of parameters the data determine in their place. It is for anyone who fits more than one parameter.

## Quick path

`k1` and `alpha` enter the model only as their product `k1 * alpha`, so the data determine the product and not each one.

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

Fix `alpha` (or `k1`) at a value, or calibrate their product as one parameter: the data determine `k1 · alpha`, and the fit says nothing about how it splits.

## Structural against practical

| Kind           | Question                                               | Depends on                                                                | Methods                                                                                    |
| -------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Structural** | Can any data from this design determine the parameter? | The model and the positions and outputs observed, not the observed values | `StructuralRank`                                                                           |
| **Practical**  | Do these data determine it?                            | The observed values, their noise and the fit                              | `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability`, `ProfileIdentifiability` |

A parameter that is not structurally identifiable is never practically identifiable: no amount of data of the same design fixes it. A parameter that is structurally identifiable can still be poorly determined by the data at hand.

## The methods

Every method runs through `Calibrator.identifiability(method)` and returns an `IdentifiabilityReport`. A threshold or a tolerance has no default: it is a modelling judgment, and the table gives the range the literature uses. `StructuralRank` and `Collinearity` differentiate the predictions, so they leave out the [noise parameters](#noise-parameters).

| Method                                                                                                                        | Needs                                                                    | Answers                                                                                                                                                                                                                                               | Engine        |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `StructuralRank(points, design)`                                                                                              | The problem only                                                         | The rank of the sensitivities, the non-identifiable parameters and the identifiable combinations                                                                                                                                                      | Deterministic |
| `Collinearity(at, max_subset_size, threshold)`                                                                                | A fit (`CalibrationResult`) or a sample                                  | Which subsets of parameters can be fitted together; `threshold` from 5 to 20                                                                                                                                                                          | Deterministic |
| `SampleIdentifiability(sample, reference, correlation_threshold, contraction_threshold, bound_share, log_parameters, priors)` | Any `ParameterSample`                                                    | What the sample learned about each parameter against its bounds or its prior: contraction, correlated pairs, modes, mass at each side, bounds that set a side, overlap and the directions of the parameters; `correlation_threshold` from 0.8 to 0.95 | Any           |
| `StartsIdentifiability(result, parameter_tolerance, loss_tolerance)`                                                          | A result of `Calibrator.run(starts=n)` with two or more converged starts | Which parameters differ between starts of equal loss                                                                                                                                                                                                  | Any           |
| `ProfileIdentifiability(result)`                                                                                              | A `ProfileResult` of the parameters of the problem, in problem order     | The verdict of the shape of each profile ([Profile Likelihood](https://munqu.github.io/commol/latest/guide/uncertainty/profiles/#verdicts-and-the-identifiability-report)); it runs no model                                                          | Any           |

### Structural rank

`StructuralRank` evaluates the sensitivities of the predicted observations at `points` rows of a Latin hypercube over the bounds, drawn from the seed of the calibrator (`Calibrator.seed`), so `problem.seed` repeats them. The sensitivity of observation `i` to parameter `j` is scaled to a relative change of both, `S_ij = (∂y_i/∂θ_j) · θ_j / s_i`, where `s_i` is the root mean square of the series of observation `i`, so the answer does not depend on units.

| `design`         | Positions and outputs                                                                                                    | Question                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| `"observations"` | Those of the problem's observations; their values are ignored                                                            | Can this design determine the parameters?          |
| `"dense"`        | Every observed series, with its outputs, factor, scale and window, at every unit position from 0 to the last observation | Can denser data of the same series determine them? |

At each point the rank counts the singular values of `S` above the norm of the estimated error of `S`. The **generic rank** is the rank at most points; each point's singular values are in `report.structural[0].points`. A parameter is:

| Verdict            | Meaning                                                      |
| ------------------ | ------------------------------------------------------------ |
| `identifiable`     | No combination of the other parameters undoes a change of it |
| `in_combination`   | Only the combinations in `combinations` are identifiable     |
| `non_identifiable` | No observation changes with it                               |

Only the design of the observations gives the structural verdict of a parameter; a dense design keeps its verdicts in its section.

```python
dense = calibrator.identifiability(StructuralRank(points=5, design="dense"))
print(dense.structural[0].generic_rank, dense.structural[0].verdicts["k1"])
# 2 in_combination
```

Denser data of `B` do not split the product either: the deficit is in the model, not in the design.

#### When the solver tolerance sets the rank

The estimated error of `S` grows with the tolerance of an ODE solver, and it can hide a singular value that a tighter tolerance resolves. With `DifferentialEquations`, each point whose rank is below the number of parameters differentiated is evaluated again at `rtol / 100` and `atol / 100`, no lower than `1e-14`, and the verdicts are computed again with those ranks. A parameter whose verdict changes there has no structural verdict: its `verdict` is `None`, and its reason names `ODESolver.rtol` and `ODESolver.atol`. When a point cannot run at the tighter tolerances, every parameter that is not `identifiable` has no verdict, and the section lists why in `unchecked`. A parameter that no observation reads stays `non_identifiable` when every point runs at the tighter tolerances. `DifferenceEquations` runs are exact up to rounding: no point is evaluated again.

| Field of `PointRank` | Meaning                                                                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `accuracy`           | `relative` (`rtol`) and `absolute` (`atol`) accuracy of the runs                                                                             |
| `rank`               | The rank at that accuracy                                                                                                                    |
| `tighter`            | The `accuracy`, the singular values, the `threshold` and the `rank` at the tighter tolerances; `None` when the point was not evaluated again |

The section gives `tighter_generic_rank`, the generic rank with the tighter rank of each point evaluated again, and the reason of a verdict that stays names it when it differs from `generic_rank`.

A verdict of `None` says that the result depends on the tolerance. Tighten `ODESolver.rtol` and `ODESolver.atol` and run `StructuralRank` again, until the rank stops changing. A fit can reach its loss at looser tolerances; the finite differences may not.

### How to read a combination

A combination is a product of powers of the parameters that the data determine: `k1 · alpha` means that every pair `(k1, alpha)` with the same product fits the data equally well. `k1 / k2^2` means that `k1` and `k2²` can grow together. A parameter whose bounds allow negative values reads as its absolute value, `|k1|`. The exponents are rounded to the nearest half.

### Collinearity

`Collinearity` computes, after Brun et al. (2001), the sensitivity measure of each parameter and the collinearity index `γ` of each subset: 1 when the sensitivities of the subset are orthogonal, infinite when one is a combination of the others. A subset with `γ` below `threshold` can be fitted together; `admissible` lists the largest such subsets. With a sample, the index is computed at every draw and summarized by its weighted median, and `share_below` gives the share of the draws below the threshold.

A parameter is `poorly_determined` when a pair that holds it has an index at or above the threshold, or no observation is sensitive to it.

### Noise parameters

A noise parameter of an [observation model](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/#noise-parameters) (`parameter_type="noise"`), such as the `overdispersion` of `NegativeBinomialObservations`, changes the likelihood and never a prediction: its column of the sensitivities is zero. `StructuralRank` and `Collinearity` leave it out.

| Method           | A noise parameter                                                                                                                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `StructuralRank` | Not in the rank, the singular values, `projections` or `verdicts` of the section; its structural `verdict` is `None`, with the reason. The generic rank is at most the number of the other parameters           |
| `Collinearity`   | Not in `sensitivity`, `pair_gamma` or a subset; no other parameter is collinear with it; no practical verdict. `max_subset_size` is at most the number of the other parameters, and it needs at least 2 of them |

Each section names the parameters it leaves out in `left_out`, each with the reason. The verdict of a noise parameter comes from the methods that read the fit: `StartsIdentifiability`, `SampleIdentifiability` and the profiles. A problem whose calibrated parameters are all noise parameters has nothing to differentiate: both methods raise a `ValueError`.

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

`SampleIdentifiability` reads any sample, such as the draws of an uncertainty method, and says what it learned about each parameter against its **reference**: `"bounds"`, the uniform distribution on the bounds, or `"prior"` with `priors`, the prior of each parameter. Here the sample is the ends of the starts.

```python
names = calibrator.parameter_names
sample = ParameterSample.equally_weighted(
    names,
    [(p.min_bound, p.max_bound) for p in problem.parameters],
    [[start.end[name] for name in names] for start in result.starts],
    SampleMethod(name="starts"),
)
from_sample = calibrator.identifiability(
    SampleIdentifiability(
        sample=sample,
        reference="bounds",
        correlation_threshold=0.9,
        contraction_threshold=0.5,
        bound_share=0.05,
    )
)
pair = from_sample.sample[0].correlated_pairs[0]
print(pair.first, pair.second, round(pair.correlation, 1))  # k1 alpha -1.0
```

Three settings have no default, as every threshold of this page: `correlation_threshold` (from 0.8 to 0.95 in the literature), `contraction_threshold`, strictly between 0 and 1, and `bound_share`, the share `ε` of the reference at each side, strictly between 0 and 0.5. Draws that are equal in every parameter, as after a resampling, count once, with the sum of their weights: the section reads the Kish size of the distinct draws (`effective_size`).

| Field                                              | Meaning                                                                                                                                                                   | Read it against the reference                                                                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `contraction`                                      | `1 − Var(sample) / Var(reference)`, with the variance of the weighted draws                                                                                               | Near 0: the sample spreads as the reference, the data add nothing. Near 1: it sits on one value. Below 0: it spreads more than the reference |
| `correlation`, `correlated_pairs`                  | The weighted correlations, and the pairs above `correlation_threshold`                                                                                                    | A pair that moves together is fitted only together                                                                                           |
| `modes`                                            | The modes of each marginal, in parameter units                                                                                                                            | Two modes are two sets of values that fit                                                                                                    |
| `mass_below`, `mass_above`                         | The weight of the draws where the distribution function of the reference is below `ε`, and above `1 − ε`                                                                  | A sample of its reference has `ε` at each side; more says the sample leans to that side                                                      |
| `at_bound`                                         | The sides whose finite bound sets that side of the marginal                                                                                                               | See below                                                                                                                                    |
| `overlap`, `bins`                                  | `Σ_b min(w_b, 1/B)` over `B = ⌈2 n_eff^{1/3}⌉` bins of equal reference mass (Rice's rule)                                                                                 | 1 for a sample of its reference, `1/B` for a sample in one bin                                                                               |
| `directions`                                       | The eigenvectors of the covariance with each parameter divided by the standard deviation of its reference (`Direction`): a `contraction`, the `loadings` and the `scales` | A direction of contraction near 0 is a combination the data do not inform                                                                    |
| `hidden_ridges`, `draws_per_parameter`, `warnings` | How many directions among the `determined` parameters contract less than `contraction_threshold`, `n_eff / p`, and what the section warns of                              | A ridge that no pair shows; an effective size below `p + 1` or `30·p`                                                                        |

A parameter is `poorly_determined` when the effective size is below `p + 1` for `p` parameters, its contraction is below `contraction_threshold`, it is in a correlated pair, or its marginal has two or more modes. The count of hidden ridges corrects the contraction of each direction by a Bayesian bootstrap whose weights are drawn from the seed of the calibrator (`Calibrator.seed`), so `problem.seed` repeats it. `"prior"` without `priors`, or `"bounds"` with them, raises a `ValueError`: the engine never measures against a reference you did not choose.

```python
from commol.distributions import Priors, Uniform

against_priors = calibrator.identifiability(
    SampleIdentifiability(
        sample=sample,
        reference="prior",
        priors=Priors(distributions={name: Uniform() for name in names}),
        correlation_threshold=0.9,
        contraction_threshold=0.5,
        bound_share=0.05,
    )
)
section = against_priors.sample[0]
print(section.reference, section.contraction == from_sample.sample[0].contraction)  # prior True
```

#### What the contraction measures

Against an informative prior, the contraction measures what the data add to the prior: the sample of a [Gibbs posterior or ABC](https://munqu.github.io/commol/latest/guide/uncertainty/smc/index.md) contracts its prior by how much the data inform each parameter, and a sample drawn from the prior has a contraction near 0. Data as informative as the prior give 0.5, so a parameter that collinearity calls `determined` may fall below a `contraction_threshold` of 0.5, and a merge of the two reports reads `conflicting`. A `GibbsPosterior` on the observation models of the problem with `eta=k` is data cloning (Lele, Nadeem and Schmuland 2010): its posterior is that of `k` copies of the data, so what still contracts as `k` grows is what the likelihood determines, and what stays is what the prior does. With a loss in place of observation models, `eta=k` is a tempering of the loss, not cloning.

#### The overlap

An overlap of 0.35 is a guideline of latent-class and capture–recapture models with uniform priors on probabilities (Garrett and Zeger 2000; Gimenez, Morgan and Brooks 2009): below it, the data inform the parameter. For a normal marginal on a uniform reference it is about a contraction of 0.93. The estimate has a floor of `1/B`, and it sits low for a flat sample, whose bins scatter around `1/B`: a sample of its reference has an overlap above 0.85 at `n_eff = 500` and about 0.82 at 30.

#### A bound that sets a side

`at_bound` names a side when the density of the sample at a finite bound is at the top of its mode and above the density of the reference there, smoothed by the same kernel, by more than three standard errors. commol cannot tell a natural bound, such as a rate of 0, from one you chose: either way the bound, not the data, sets that side, and the interval of the parameter ends there. `at_bound` does not set the verdict: its reason, such as "Sample: the lower bound, not the data, sets one side; 79 % of its mass is in the bottom 5 % of its reference.", stays when the parameter is `determined`. A sample of a prior that piles at its bound, such as `Beta(a=5, b=1)`, is not `at_bound`.

#### The scale of each parameter

The contraction, the correlations, the modes, the density at a bound and the directions read each parameter as it is, or `ln θ` for the parameters of `log_parameters`. A product held fixed is a line on the log scale, and a heavy-tailed prior such as `LogNormal(mu=0, sigma=2)` gives a meaningless contraction on the natural scale. A parameter of `log_parameters` has a lower bound of at least 0 and every draw above 0; a bound of 0 is no bound of `ln θ`. The masses and the overlap read the distribution function of the reference, which no scale changes.

```python
import random

rng = random.Random(1)
rows = []
for _ in range(2000):
    # k1 · alpha held near 0.3, and k2 piled at its lower bound 0.
    k1 = 0.3 * 2.5 ** rng.uniform(-1.0, 1.0)
    rows.append([k1, 0.3 / k1 * (1 + 0.01 * rng.gauss(0, 1)), abs(rng.gauss(0, 0.005))])
draws = ParameterSample.equally_weighted(
    names, [(0.1, 3.0), (0.1, 3.0), (0.0, 0.5)], rows, SampleMethod(name="given")
)
logged = calibrator.identifiability(
    SampleIdentifiability(
        sample=draws,
        reference="bounds",
        correlation_threshold=0.99,
        contraction_threshold=0.5,
        bound_share=0.05,
        log_parameters=["k1", "alpha"],
    )
).sample[0]
ridge = logged.directions[-1]
print(round(ridge.contraction, 3), {n: round(v, 2) for n, v in ridge.loadings.items()})
# 1.0 {'k1': 0.71, 'alpha': 0.71, 'k2': 0.02}
print(logged.at_bound["k2"])  # ['lower']
```

#### Limits of a sample

- **Few draws lose weak modes.** A mode counts when its dip is deeper than three standard errors of the density; a weak mode needs many draws to pass. Each dip is tested on its own, so a flat marginal of many draws shows a false second mode in about 2 % of samples of 20 000 draws (0.5 % at 2000).
- **The effective size.** The weights of the distinct draws, which set the standard errors, and their Kish size do not see correlated draws, such as the particles of a short chain: they can overstate the size, and the standard errors are then too small. The Kish size is the number of equally weighted draws the weights are worth, and a covariance of `p` parameters from fewer than `p + 1` equal draws is singular. A sample worth fewer than `p + 1` draws, such as one whose weight sits on one draw, has a contraction near 1 that the weights make, not the data: every verdict is `poorly_determined`, with the reason "Sample: effective size 1, below p + 1 = 3 for 2 parameters.", and `warnings` says it. Its own weighted covariance may still have full rank; the floor reads what the weights are worth. No setting moves it: below `p + 1` no covariance estimate exists to judge. Between `p + 1` and `30·p`, `warnings` names the effective size.
- **Hidden ridges.** Below `n_eff ≈ 30·p` the count of hidden ridges over-counts: the eigenvalues of a sample covariance spread with the dimension, and the bootstrap correction does not remove all of it.
- **A covariance is linear.** The directions and the correlations see a straight ridge; a curved or multimodal posterior, such as a product on the natural scale or two sets of values, hides behind a contraction that looks high. Read the modes and the figures of the sample with them.

### Starts

`StartsIdentifiability` compares the converged starts whose loss exceeds the best by at most `loss_tolerance` of the best loss, and flags a parameter whose ends among them differ by more than `parameter_tolerance` of its bound width. A loss near zero, as with data made by the model itself, leaves no start within a relative tolerance but the best; use data with noise. A particle swarm without a stop rule converges at no start; with a [polish](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#polish-with-nelder-mead), each start ends with Nelder-Mead, and the method reads the polished ends.

## The report

An `IdentifiabilityReport` holds, for each parameter, a **structural** and a **practical** verdict, and the sections of the methods that ran (`structural`, `collinearity`, `sample`, `starts`, `profile`). The profiles of a `ProfileLikelihood` give their report with `calibrator.identifiability(ProfileIdentifiability(result=profiles))`, which runs no model; a flat profile gives the structural verdict `in_combination` when resetting the other parameters to their fitted values breaks it, and `non_identifiable` when nothing the data read depends on the parameter ([Profile Likelihood](https://munqu.github.io/commol/latest/guide/uncertainty/profiles/#verdicts-and-the-identifiability-report)). `merge` joins reports: a verdict the reports agree on stays, with every reason; one they disagree on is `conflicting`, with both reasons. A structural verdict of `None`, as that of a noise parameter, adds its reasons and leaves the verdicts of the other reports. `report.report()` gives it as plain data ([Reports](https://munqu.github.io/commol/latest/guide/reports/#the-report-of-an-identifiability-analysis)).

```python
data = report.report()
print(data["parameters"]["k2"]["practical"]["verdict"])  # determined
```

## Limits

- **Local.** The sensitivities are derivatives at points: a parameter can be identifiable near every point and still have two distant values that fit equally well. Symbolic methods give a global answer for differential equations; they are outside commol.
- **Numerical.** The derivatives are finite differences; a singular value counts only when it is above their estimated error. The error grows with the tolerance of an ODE solver (`rtol`, `atol`): a rank that a hundredth of the tolerances raises gives no verdict, but one evaluation at a tighter tolerance cannot show that a yet tighter one changes nothing.
- **Generic points.** A point where the rank is lower or higher than at most points is special; the generic rank is the rank at most points.
- **Deterministic engines.** The runs of a stochastic engine change in jumps with the parameters, so `StructuralRank` and `Collinearity` refuse one; `SampleIdentifiability` and `StartsIdentifiability` accept it.

## Checklist

- [ ] You checked the structural rank of the design before you fitted.
- [ ] Each parameter `in_combination` is fixed, or replaced by its combination.
- [ ] The thresholds you chose are in the report you share.

## Next steps

| To learn                              | Read                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Several starts of an optimizer        | [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#several-starts) |
| Factors, which often pair with a rate | [Observations](https://munqu.github.io/commol/latest/guide/calibration-observations/index.md)    |
| What a sample is                      | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/#the-parameter-sample)     |
| Intervals and verdicts from profiles  | [Profile Likelihood](https://munqu.github.io/commol/latest/guide/uncertainty/profiles/index.md)  |
