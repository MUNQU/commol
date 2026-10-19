# Conformal Prediction Intervals

[Back to Uncertainty](index.md)

Conformal prediction gives intervals for observations, with a coverage
guarantee that needs no noise model, no prior and no correct model. It answers
"will the next observations fall inside?", not "which parameters fit?". Each
variant scores how far observations the fit did not use fall from their
predictions, and a quantile of those scores sets the width of every interval.
Use it when you must state how often new data fall inside an interval, and
you cannot or will not say how the data scatter around the model. Every
variant runs through `Calibrator.uncertainty`, in Rust
(`crates/commol-inference/src/conformal/`).

## Quick path

```python
import numpy as np

from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
    SplitConformal,
)


def ab_model(k: float | None):
    return (
        ModelBuilder(name="AB")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=k)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k")
        .build(
            Aggregate(bin_amounts={"A": 1000, "B": 0}),
            engine="DifferenceEquations",
        )
    )


# Noisy data of B at steps 1 to 80: fit steps 1 to 40, hold out 41 to 80.
truth = Simulation(ab_model(0.05)).run(num_steps=90)
rng = np.random.default_rng(1)
observations = [
    ObservedDataPoint(at=step, output="B", value=truth["B"][step] + rng.normal(0.0, 5.0))
    for step in range(1, 81)
]
fitted = [point for point in observations if point.at <= 40]
held_out = [point for point in observations if point.at > 40]

problem = CalibrationProblem(
    observed_data=fitted,
    parameters=[
        CalibrationParameter(id="k", parameter_type="parameter", min_bound=0.0, max_bound=0.5)
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=500),
    seed=42,
)
calibrator = Calibrator(Simulation(ab_model(None)), problem)

result = calibrator.uncertainty(
    SplitConformal(
        calibration_data=held_out,
        score="absolute",
        by="series",
        level=0.9,
        positions=[85.0, 90.0],
    )
)
intervals = result.intervals.series["B"]
print([round(value, 1) for value in intervals.lower])  # [981.0, 983.9]
print([round(value, 1) for value in intervals.upper])  # [993.4, 996.3]
print(result.guarantee)
```

`result.intervals` holds the intervals of each series at `positions`,
`result.scores` the scores they came from, and `result.guarantee` what the
coverage claim assumes.

## What the guarantee means

A conformal interval of level `1 − α` covers a **new observation** with
probability at least `1 − α`, averaged over the data and the new observation.

| It says | It does not say |
|---|---|
| Of many new observations, at least a share `1 − α` falls inside its interval | That one interval holds its observation with probability `1 − α` given the data |
| The claim holds whatever the noise, and even when the model is wrong | That the interval contains the model curve or the true parameters |
| A wrong model gives wide intervals, not wrong ones | That the intervals are narrow |

The claim holds when the scores are **exchangeable**: any order of the scores
of the held-out and the new observations is as likely as any other. Noise that
is independent and of one distribution along the series is exchangeable; a
trend in the errors, a change of regime, or errors that depend on each other
are not. Each variant names its assumption in `result.guarantee`.

The quantile of `n` scores at level `1 − α` is the `⌈(n + 1)(1 − α)⌉`-th
smallest (`conformal_quantile` in `crates/commol-inference/src/conformal/quantile.rs`).
With fewer than `1/α − 1` scores that rank is above `n`, so the interval is
unbounded: `-inf` to `inf`. `result.intervals.unbounded` names the positions of
each series where that happens. At level 0.9 a quantile needs 9 scores.

## Scores and pooling

Every variant takes a `score`, a `by` and a `level`, none with a default: they
change the width of the intervals, which is a modelling choice.

| `score` | Score of `y` against `ŷ` | Interval | Use it when |
|---|---|---|---|
| `"absolute"` | `|y − ŷ|` | One width per quantile | The noise has one size along the series |
| `"normalized"` | `|y − ŷ| / s`, with `s` the root mean square of the observed values of the series in the problem | Width proportional to `s` | Series of different size share one quantile (`by="all"`) |
| `"spread"` | `|y − ŷ| / σ̂`, with `σ̂` the spread of the predictions at that point | Width proportional to `σ̂` | The uncertainty of the prediction changes along the series, as here, where the sample's runs come together as B saturates |

`by="series"` gives each series (observation label) its own quantile;
`by="all"` gives every series one quantile. Per series is the safer first
choice: a small series does not borrow the errors of a large one, but it needs
`1/α − 1` scores of its own.

- `score="normalized"` with `by="series"` raises a `ValueError`: its factor is
  one constant per series, so its intervals equal those of `"absolute"`.
- `score="spread"` needs predictions with a spread. `SplitConformal` and
  `AdaptiveConformal` take it from a `sample`, whose predictive gives the
  prediction (its weighted median) and the spread (its weighted standard
  deviation); without one they raise a `ValueError`. `JackknifePlus` and `EnbPI`
  take it from their own refits. `RollingConformal` fits once per origin, so it
  rejects `"spread"`.

```python
from commol import ParameterSample, SampleMethod

sample = ParameterSample.equally_weighted(
    names=["k"],
    bounds=[(0.0, 0.5)],
    values=[[0.048], [0.05], [0.052]],
    method=SampleMethod(name="by hand"),
)
spread = calibrator.uncertainty(
    SplitConformal(
        calibration_data=held_out,
        score="spread",
        by="series",
        level=0.9,
        positions=[85.0, 90.0],
        sample=sample,
    )
)
widths = [
    round(upper - lower, 1)
    for lower, upper in zip(spread.intervals.series["B"].lower, spread.intervals.series["B"].upper)
]
print(widths)  # [6.3, 5.2]
```

## Which variant to use

| Variant | Held-out data | Fits | Guarantee | Use it for |
|---|---|---|---|---|
| `SplitConformal` | `calibration_data` | One | At least `1 − α`, exchangeable scores | Many observations, some of which you can leave out of the fit |
| `RollingConformal` | None: each origin holds out what follows it | One per origin | At least `1 − α` per horizon, scores of a horizon exchangeable across origins | Forecasts: intervals ahead of the data that widen with the horizon |
| `JackknifePlus` | None | One per block | At least `1 − 2α` with one observation per block; the CV+ bound with longer blocks; exchangeable blocks | Short series, where no data can be spared |
| `EnbPI` | None | `replicates` | Near `1 − α`, approximately, under stationary and strongly mixing errors | Time series with dependent errors |
| `AdaptiveConformal` | The `stream` of new observations | One | The long-run share of misses tends to `α`, with no assumption | Data that arrive one by one and can change regime |

## How to hold data out

| Variant | What to hold out |
|---|---|
| `SplitConformal` | Pass the problem the observations to fit, and `calibration_data` the rest. For a forecast, hold out the latest positions, as in the quick path, so the scores measure errors ahead of the fit. The held-out points must not be in the problem. |
| `RollingConformal` | Nothing: give `origins`. Each fits the observations at or before it and scores the later ones, so the problem holds every observation. |
| `JackknifePlus`, `EnbPI` | Nothing: they leave blocks out of their own refits. A block holds every observation of every series at `block_length` consecutive positions, so dependence stays inside a block. |
| `AdaptiveConformal` | The observations that arrive after the fit, in `stream`. |

## Split conformal

`SplitConformal(calibration_data, score, by, level, positions, sample=None,
starts=1)` fits the problem's observations (`Calibrator.run(starts)`), or
reads a `sample`, scores `calibration_data` and gives intervals at `positions`
for each series of `calibration_data`. A series of `calibration_data` is read
as its observations define it: its outputs, factor and window.

## Rolling origins

`RollingConformal(origins, score, by, level, positions, starts=1)` gives
intervals ahead of the data, with one quantile per horizon. Horizon `k` is the
`k`-th observation of a series after an origin: a count, not a distance on the
axis, because positions can be irregular. Each origin fits the observations at
or before it (one refit per origin, from the fit of every observation) and
scores the later ones. The intervals are around the fit of every observation,
at `positions`, which must all be after the data; the `k`-th position is
horizon `k`, so give positions spaced as the observations. A horizon with fewer
than `1/α − 1` origins that reach it has an unbounded interval.

```python
from commol import RollingConformal

every_point = CalibrationProblem(
    observed_data=observations,
    parameters=problem.parameters,
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=500),
    seed=42,
)
calibrator_all = Calibrator(Simulation(ab_model(None)), every_point)
rolling = calibrator_all.uncertainty(
    RollingConformal(
        origins=[float(step) for step in range(30, 71)],
        score="absolute",
        by="series",
        level=0.8,
        positions=[81.0, 82.0, 83.0],
    )
)
print(rolling.intervals.unbounded)  # {}
```

## Jackknife+

`JackknifePlus(block_length, score, by, level, positions, starts=1)` refits
once per block left out (Barber et al. 2021). Observation `i` of block `b` has
the residual `Rᵢ` of the fit `μ̂₋ᵦ` that left its block out, and the interval at
a point `x` is

```text
[ q⁻_α { μ̂₋ᵦ₍ᵢ₎(x) − Rᵢ } ,  q⁺_{1−α} { μ̂₋ᵦ₍ᵢ₎(x) + Rᵢ } ]
```

where `q⁺` is the `⌈(1 − α)(n + 1)⌉`-th smallest value and `q⁻` the
`⌊α(n + 1)⌋`-th smallest. The guarantee holds with no held-out set, under
exchangeable blocks. With `n` scored observations in `K` blocks it is the bound
of K-fold CV+ (Theorem 4):

```text
1 − 2α − min{ 2(1 − 1/K) / (n/K + 1),  (1 − K/n) / (K + 1) }
```

With one observation per block, `K = n` and the bound is `1 − 2α`. With longer
blocks, jackknife+ is CV+ with the blocks as folds, so CV+ has no class of its
own. The theorem assumes blocks of `n/K` observations each.
`cv_plus_bound` in `crates/commol-inference/src/conformal/jackknife_plus.rs`
computes it, and `result.guarantee` states it.

```python
from commol import JackknifePlus

jackknife = calibrator.uncertainty(
    JackknifePlus(block_length=4, score="absolute", by="series", level=0.9, positions=[45.0])
)
print(jackknife.guarantee)  # ... at least 0.7318 ... with K = 10 blocks of 4 positions ...
```

## EnbPI

`EnbPI(replicates, block_length, score, by, level, positions, starts=1)` draws
its own replicates (Xu and Xie 2021): each draws as many blocks as there are,
with replacement, and fits with each observation's weight multiplied by the
count of its block, so the blocks it did not draw get weight 0. The
leave-one-out prediction of an observation is the mean of the replicates that
left its block out; the point prediction at a position is the mean of the
leave-one-out predictions there. No fit beyond the replicates is needed. A
block left out by no replicate raises a `ValueError` that asks for more
replicates: a block is left out by a replicate with probability about
`1/e ≈ 0.37`.

## Adaptive conformal inference

`AdaptiveConformal(level, step, score, by, stream, sample=None, starts=1)`
walks the observations of `stream` in order of position (Gibbs and Candès
2021). The interval of each point is the quantile, at the working level
`1 − α_t`, of the scores of the earlier points of the stream; then the point
is scored and the working level moves:

```text
α_{t+1} = α_t + γ (α − err_t),   err_t = 1 for a miss, 0 for a cover
```

So a miss widens the next interval and a cover narrows it. The long-run share
of misses tends to `α` whatever the data. `by="series"` keeps one working level
per series, `by="all"` one for all. The first points have few earlier scores,
so their intervals are unbounded. `result.intervals` holds the interval given
at each point before it was seen.

```python
from commol import AdaptiveConformal

adaptive = calibrator.uncertainty(
    AdaptiveConformal(level=0.9, step=0.05, score="absolute", by="series", stream=held_out)
)
series = adaptive.intervals.series["B"]
misses = [
    not lower <= point.value <= upper
    for lower, upper, point in zip(series.lower, series.upper, held_out)
]
print(len(adaptive.intervals.unbounded["B"]))  # 7
print(round(sum(misses) / len(misses), 3))  # 0.125
```

## Results, plots and reports

Every variant returns a `ConformalResult(intervals, scores, guarantee)`.
`intervals` is a `PredictionIntervals`: its `level`, a `SeriesIntervals` per
series (`positions`, `prediction`, `lower`, `upper`), and `unbounded`.
`SimulationPlotter.plot_series(prediction_intervals=...)` draws the band of
each series on its subplot, and
`simulation_report(..., prediction_intervals=...)` adds it under the key
`prediction_intervals`.

```python
from commol.api.plotter import SimulationPlotter

run = Simulation(ab_model(0.05)).run(num_steps=90)
figure = SimulationPlotter(Simulation(ab_model(0.05)), run).plot_series(
    observed_data=observations, prediction_intervals=result.intervals, series=["B"]
)
```

## Checklist

- [ ] The held-out observations are not in the problem, and come from where the
      new observations will.
- [ ] Each quantile has at least `1/α − 1` scores; `result.intervals.unbounded` is empty.
- [ ] The assumption in `result.guarantee` fits your data: exchangeable scores,
      mixing errors, or none.
- [ ] For a forecast, you use `RollingConformal`, `EnbPI` or `AdaptiveConformal`, not
      one split of errors one step ahead.

## Next steps

| To learn | Read |
|---|---|
| Prediction intervals from the residuals of a fit, and the scores of a predictive | [Resampling](resampling.md#prediction-intervals), [Uncertainty](index.md#scores) |
| Bootstraps and the jackknife of the parameters | [Resampling](resampling.md) |
| Plots | [Running Simulations](../simulations.md#plotting) |
