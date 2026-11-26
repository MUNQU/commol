# Windows

[Back to Design decisions](../design-decisions.md)

How an observation over a window reads a run. Positions on the axis have
[a page of their own](time.md).

## Windowed observations

An observation at `at` with `window = W` is compared with
`output(at) - output(at - W)`, anchored at its own position. The Rust loss and
`windowed_totals` (`_simulation/windows.py`) give the same number from one rule
(`window_increments`), and the predictive reads windows through the observation operator
([One observation operator](observations.md#one-observation-operator)). With
`parts=K` the window splits at `start + k · W / K`, each boundary a position
of the engine ([parts](observations.md#the-parts-of-a-window-are-observations-of-every-reader)).
Window `w` covers `(w · W, (w + 1) · W]`; `window_index` places a position with
`max(ceil(at / W) − 1, 0)`, and the `observed` block of a report (`_observed_report` in `_simulation/reports.py`) calls it.
`plot.series` of results reads a windowed observation through its own window at its own position, never on a
grid of windows from 0: an observation at 10.5 with a window of 7 falls on no window end of that grid.

**Guards:** `tests/simulation/test_windows.py::TestAgreementWithTheCalibrationLoss` (the loss at
the generating parameters is zero),
`tests/reporting/test_simulation_report.py::test_observed_report_places_each_observation_by_window_index`,
`tests/plotting/test_band_figures.py::TestPredictiveSeries::test_a_windowed_band_of_one_draw_is_its_windowed_run`,
`tests/plotting/test_figures.py::test_each_observed_point_is_read_through_its_own_window_and_position`,
`crates/commol-calibration/tests/test_parts.rs::boundaries_come_from_the_start_and_the_end`.
