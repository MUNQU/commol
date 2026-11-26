# Dependence

[Back to Design decisions](../design-decisions.md)

How a series of values is read for dependence, which law its tests take, and
who decides the largest lag and the length of a block.

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

## The user gives the largest lag

Both analyses take `max_lag`, required, and read every series at the lags
`1 … max_lag`. The engine reads it as a `usize` with the rule `MAX_LAG` (at
least 1, `crates/commol-inference/src/ranges.rs`), and a binding as any count
(`py-commol/src/counts.rs`). The number says which dependence the test can
see, and the p-value of `Q̃` depends on it, so the user writes it. A rule from
the number of values, as the smaller of 10 and a fifth of them, reads at most
10 days of a daily series whatever the period of its process, and a cycle of 7
days only from 35 values on.

One number serves series of different lengths, because a lag of fewer than 4
pairs leaves `Q̃`. At 5 %, that rule rejects 2.6 % of independent normal series
of 12 values, 4.4 % of 20 and 5.4 % of 35, and 10 lags reject 4.9, 6.2 and
6.2 %. So one number is nearer 5 % with 12 values, and up to 1.8 points above
the rule from 20 values on. The law of `Q̃` is
[#530](https://github.com/MUNQU/commol/issues/530).

These rates, and those of 4, 7 and 10 lags in
[Choose the largest lag](../../guide/uncertainty/dependence.md#choose-the-largest-lag),
are from [#561](https://github.com/MUNQU/commol/issues/561). Those of 17 and 33
lags are from
[M1 of #511](https://github.com/MUNQU/commol/issues/511#issuecomment-7009198279).
The smallest p-value of 4, 7 and 10 lags rejects 8.4 % of independent normal
series of 20 values, 9.1 % of those of 35 and 9.2 % of those of 100 (40 000 sets
each of `series_dependence`).

`τ`, its band, `cut_lag` and the statistics of the problem do not read
`max_lag`, so two analyses of one seed share them.

| Alternative | Why not |
|---|---|
| An automatic number (`"auto"`), since it is a fixed rule and changes no interval | It sets a p-value with a number nobody wrote. |
| A number per series | No caller, and two analyses give it. |
| A function that suggests a number | A second way to what the guide says. |
| A message of its own for the text `"auto"` | No release has that text, and any text has the error of a whole-number argument, which names `max_lag`. |
| An enum of one variant in the engine | `usize` with its rule says the same. |

**Guards:** `tests/uncertainty/test_dependence.py::test_the_largest_lag_is_a_whole_number`,
`tests/uncertainty/test_dependence.py::test_one_largest_lag_serves_series_of_different_lengths`,
`tests/uncertainty/test_counts.py`, `crates/commol-inference/tests/test_errors.rs`
(`a_setting_out_of_range_is_invalid_in_every_method`), and the example of the
[Quick path](../../guide/uncertainty/dependence.md#quick-path).

## The user gives the length of a block

Every method that reads blocks takes `block_length`, a whole number of at
least 1, required, and groups the observations with `BlockScheme::new`
(`crates/commol-inference/src/resampling/blocks.rs`). commol measures the
dependence (`residual_dependence`) and does not choose a length from it. The
length that serves best depends on the method and on the question. A rule such
as that of Politis and White (2004) minimizes the mean squared error of a
variance, not the error of the coverage: in a simulation with constant means
and 90 % intervals of the sandwich, for 12 series of 35 values of an AR(1) of
0.6 it covers 83 % and blocks of 12 positions 88 %; for one such series, 71 %
and 88 %. Nor does one number reduce several series: the largest of their
estimates grows with their count, a pooled one covers less when one of them is
dependent, and structural zeros and series of other spacings each need another
rule. The choice changes the result (in one calibration 12 positions gave an
`η` of 0.093 and 4 gave 0.227), so it is in the settings of the method. The
figures are from [#548](https://github.com/MUNQU/commol/issues/548).

A length counts the distinct positions of all the series, whether or not an
observation there carries information: to skip a position, `BlockScheme` would
read the scores of a fit, and the blocks would change with it.

The error of a text that is no whole number is the one of any such setting.

| Alternative | Why not |
|---|---|
| A function that estimates the length of each series | It repeats `residual_dependence`, reads each series on its own grid when a block counts the positions of all, and has no caller. |
| An automatic length pooled over the series | One rule for uses commol does not know: the bootstraps, the meat of the sandwich and `η`. |
| An automatic length with a warning | The number stays the one nobody chose. |
| A message of its own for the text `"auto"` | The rule of `block_length` names the setting, as for any other text; the changelog gives the migration. |

**Guards:** `tests/uncertainty/test_ranges.py::test_a_block_length_is_a_whole_number`,
`tests/uncertainty/test_results_equality.py::test_no_result_has_the_estimates_of_a_block_length`,
`crates/commol-inference/tests/test_resampling.rs::a_given_block_length_gives_the_blocks_of_its_windows`,
`tests/uncertainty/test_smc.py::test_a_position_of_observations_with_no_information_counts_in_the_windows`,
`tests/uncertainty/test_block_length.py`
and the example of [Choose the length](../../guide/uncertainty/resampling.md#choose-the-length).
