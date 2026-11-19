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

## A lag of few pairs leaves the portmanteau

`robust_portmanteau` (`crates/commol-statistics/src/dependence.rs`) sums
only the lags of `PORTMANTEAU_PAIRS = 4` pairs or more, and counts only
those as degrees of freedom. A lag of `N` pairs adds a term at most `N`,
of mean near 1 but with less spread than `χ²(1)`: one pair adds 1 whatever
the values. A lag left out keeps its correlation, and `few_pairs` of the
series gives it with its pairs, so the result says what `Q̃` did not read.
When no lag is left, as for a series of 4 values, `Q̃`, its p-value and
the draw p-values are `None`, `degrees_of_freedom` is 0 and the
`portmanteau_reason` says why; `τ` and the correlations still read the
series.

The minimum comes from a study of the rejection rate at 5 % of `Q̃` against
`χ²` for independent values, with `max_lag` at the last lag of the grid:
series of 8 to 50 values, normal and `t₃`, one whose spread grows 30 times,
and 12 patterns of 30 % or 50 % gaps on grids of 20 to 60 (27 designs,
200 000 sets each). The table gives the mean of `|rate − 5 %|`
over all designs and over those of 12 values or more, and the lowest rate
of these; minimum 1 is no minimum.

| Minimum pairs | Mean distance, all | Mean distance, 12 values or more | Lowest rate, 12 values or more |
|---|---|---|---|
| 1 | 1.48 | 1.21 | 0.9 % |
| 2 | 1.28 | 1.03 | 1.8 % |
| 3 | 1.22 | 1.00 | 2.0 % |
| **4** | **1.22** | **1.01** | **2.2 %** |
| 5 | 1.29 | 1.10 | 1.2 % |
| 6 | 1.34 | 1.08 | 1.6 % |
| 8 | 1.38 | 1.24 | 2.6 % |
| 10 | 1.50 | 1.50 | 2.0 % |

Minima 3 and 4 tie on the mean distance, and 4 keeps the lowest rate
higher (2.2 % against 2.0 %); 8 keeps it higher still, far from 5 % on
the mean. Above 4, each lag left out is one that would carry the test. A
lag of 4 to about 20 pairs still pulls the rate down (one lag of 4 pairs
alone rejects in 0.2 %, of 10 in 3.1 %), so a short series stays
conservative, and many lags make `Q̃` liberal (7.5 % at `max_lag=17` of
35 values); the guide gives both. The designs and every rate are in
[M1](https://github.com/MUNQU/commol/issues/511#issuecomment-7009198279).

| Alternative | Why not |
|---|---|
| Every lag with a pair in `Q̃` | Each lag of few pairs pulls the rate down: 3.1 % in the study of the guide, where the minimum gives 4.3 % |
| A minimum that grows with the values, as `n / 4` | It leaves out more lags the longer the series, where a fixed minimum already reaches the level; from 5 pairs on the rates move away from 5 % again |
| Every lag, read against replicates in place of `χ²` | The replicates would hold the level, but a lag of one pair would still add a constant to `Q̃` and take its power; one set reads `Q̃` against `χ²` with no replicate |
| No `Q̃` for a series with any lag of few pairs | It refuses the test of a series whose short lags hold all its information |

**Guards:** `crates/commol-inference/tests/test_dependence.rs`
(`a_lag_with_few_pairs_leaves_the_portmanteau`), `tests/uncertainty/test_dependence.py`,
`tests/uncertainty/test_dependence_coverage.py::TestValues::test_lags_with_few_pairs_keep_the_level`.
