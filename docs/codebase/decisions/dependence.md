# Dependence

[Back to Design decisions](../design-decisions.md)

How a series of values is read for dependence, and which law its tests
take.

## A test of dependence takes its reference

`residual_dependence` and `series_dependence` (`crates/commol-inference/src/dependence/`)
read the values of each series on the grid of its smallest spacing, each lag
over its pairs, with the robust portmanteau `Q̃` and the `τ` of Geyer's
initial monotone sequence (`crates/commol-statistics/src/dependence.rs`).
The law of their statistics depends on where the residuals come from, so
`residual_dependence` takes a required `reference`, one class per member:
`AtParameters(replicates)` draws data at the parameters of the draw with no
refit, and holds for held-out data; `Refitted(replicates, starts=1)` fits
data drawn at the fit again, and holds for the data of the fit whatever its
number of parameters. Discrete quantile residuals are averaged over a
required number of randomizations, common to every draw, and compared with
the law of the averaged statistics. A sample of many draws gives the
weighted statistics and the spread of the p-values of the draws, with no
test of its own. `series_dependence` reads one set of values, with a band of
`τ` from signs ±1 on the values; it has no source, so it takes its `seed`.
A statistic that neither the set nor any replicate defines, as `τ` of a
constant series, has no test, and an integrated `cut_lag` is the largest of
its sets.

| Alternative | Why not |
|---|---|
| `χ²` as the reference of every test | It is not the law of averaged statistics (0.7 to 3.7 % at 5 %), and at a fit of 10 parameters to 35 positions `AtParameters` rejects in 24 to 26 % |
| `Literal["asymptotic"] \| Simulated(...)` | The first member has settings too: a family of settings is one class per member. |
| One randomization of discrete residuals | A p-value that moves with the seed for short series of small counts. |
| A p-value of the mean of the p-values of the draws | It holds its level only under an exact posterior (7 to 58 % otherwise). |
| The flat-top estimator of `τ` | `τ̂ ≤ 0` in 89 to 94 % of the residuals of fits with strong dependence, where commol reads `τ` most. |
| Lags by order of the values | A gap would pair different distances. |
| A band of `τ` from independent normal series | It does not keep the size of each value: see [Any values](../../guide/uncertainty/dependence.md#any-values). |

**Guards:** `crates/commol-inference/tests/test_dependence.rs`,
`tests/uncertainty/test_dependence.py`, the studies of
`tests/uncertainty/test_dependence_coverage.py`.
