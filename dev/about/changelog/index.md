# Changelog

[Back to Home](https://munqu.github.io/commol/latest/index.md)

All notable changes to Commol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Calibrator.run(starts=n)` runs the optimizer from `n` start points in parallel and returns the best: the initial guess, then rows of a Latin hypercube over the bounds drawn from the problem's `seed`. `CalibrationResult.starts` lists every start (`StartSummary`: start, end, loss, converged); distinct ends with equal losses show that the data do not identify the parameters. See [Several starts](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#several-starts).
- `CalibrationResult.failed_runs`: why each failed start failed, also logged as a warning.
- `ParameterSample`: weighted draws of the parameters of a problem, with their invariants checked at construction and weighted statistics computed in Rust (`mean`, `covariance`, `correlation`, `quantiles`, `interval`, `effective_size`, `draws`, `resample`, `summary`). It saves to JSON and loads back equal.
- `Predictive(calibrator, sample, positions)`: the run of every draw, in parallel, its band (`band`, `series_band`) and every draw's value of a set of observations through the observation operator of the loss (`at_observations`). `SimulationPlotter` and `simulation_report` take a predictive.
- `commol.scores`: empirical coverage, interval score, weighted interval score, the CRPS of a weighted sample and PIT values of a predictive, per observation and per series.
- `Calibrator.uncertainty(method)`: one entry point for every uncertainty method. A method's configuration subclasses `UncertaintyMethod[R]` and registers its runner; the result type follows the configuration's type.
- `weighted_quantiles`, and a `weights` argument for `member_statistics`.
- `CalibrationHistory` and `CalibrationEvaluation`, returned by `Calibrator.run_with_history()`.
- The crate `commol-inference`: the parameter sample, the one quantile, the predictive and the scores. See [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md).

### Changed

- The `latest` version of the documentation site is the documentation of the last release, published from its tag; the `dev` version follows `main`.
- **Breaking:** every interval uses one quantile: the weighted inverse of the empirical distribution between the midpoints of the cumulative weights, NumPy's `method="hazen"` with equal weights. `member_statistics` and `SimulationReplicates.quantiles` used NumPy's `linear`; their intervals move slightly.
- **Breaking:** `CalibrationResult.best_parameters` keeps the order of `CalibrationProblem.parameters`.
- **Breaking:** `Calibrator.run_with_history()` returns a `CalibrationHistory` instead of the extension's object.
- **Breaking:** `applied_parameters_report(..., sample=, confidence_level=)` reads the statistics of a `ParameterSample`; `parameter_statistics` is gone.
- **Breaking (Rust):** `CalibrationProblem` is split into layers (`ForwardModel`, `ObservationOperator`, `Feasibility`, `Discrepancy`, `OptimizationObjective`); `optimize` takes the problem by reference, and the optimizer configurations have no `verbose` field. `CalibrationEvaluation` has no `predictions`.
- A verbose optimizer writes its table to Python's `sys.stdout` only, and needs one start; the library never writes to the standard error.

### Fixed

- Several starts of Nelder-Mead began at the same point: each start now comes from the run seed.
- A parameter value could land on another parameter when results were read by name from an unordered mapping.
- A failed run of a multi-start optimization was written to the standard error and dropped; it is now part of the result.
- An error of the model at a vertex of the initial Nelder-Mead simplex made the optimizer panic; it now stops the calibration with its message (the argmin fork returns the error).

### Removed

- **Breaking:** the ensemble calibration: `Calibrator.run_probabilistic`, `ProbabilisticCalibrator`, `CalibrationProblem.probabilistic_config`, `ProbabilisticCalibrationConfig`, `ProbabilisticCalibrationResult`, `EnsembleSolution`, `EnsembleSelectionSummary`, `ParameterSetStatistics` and their settings classes. Its intervals had no statistical meaning: its members were points the optimizer visited, and the selection maximized the coverage of the data it was fitted to. The closest replacement with no noise model is `Calibrator.uncertainty(BayesianBootstrap(...))`, in the next release.
- The dependency on `scikit-learn`.

## [0.4.0] - 2026-10-16

### Added

- `Model.describe(layout="default")` returns a `ModelDescription`: the model as data, sections of tables, equations, state graphs and text, for every kind of model. `render(format)` gives text, Markdown (with the state graph as a Mermaid diagram), a LaTeX document, JSON or Graphviz DOT, and `save(path)` writes it in the format of the suffix (`.txt`, `.md`, `.tex`, `.json`, `.dot`). In a notebook it shows as Markdown. See [Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md).
- `layout="odd"`: the description follows the ODD protocol for describing individual- and agent-based models (Grimm et al. 2006, 2010, 2020), and marks what commol cannot fill "To be written by the modeller.".
- `EngineType.dynamics_form` and `EngineType.schedule`: how an engine advances (`difference_equation`, `differential_equation`, `stochastic_events`, `rules`), as a description shows it.
- The process of a model apart from its structure: `Model.process` (`Process`: bins, accumulators, stratifications, transitions) is written once, and `Model.structure` lays it out. `Aggregate` is the structure of amounts per compartment, the kind `aggregate`.
- `neighbors_<bin>` and `neighbors` in a transition rate: the amount of a bin among the neighbors of a unit and their total. A rate that reads them and no compartment is proportional. `Aggregate(neighbors=Complete())` makes every unit a neighbor of every other; `Groups(stratification, weights)` makes units neighbors by the categories of a stratification. A rate that reads them needs `neighbors`, with no default.
- Variants of a structure: a subclass with the same kind, fields and engines, saved with its `variant` and loaded back as its own class.
- Seeds and replicates for stochastic engines: `Simulation.run(…, seed=)` and `run_at(…, seed=)` (`SimulationResults.seed` keeps the seed, drawn when none is given), `Simulation.replicates(n, seed=)` with `run` and `run_at` that return `SimulationReplicates` (runs in parallel, one seed each; `mean`, `quantiles`, `member_statistics`), `Simulation.is_stochastic`, and `EngineType.is_stochastic`. `SimulationPlotter` draws the mean and a band of replicates, and `simulation_report` takes them as `ensemble_runs`.
- `CalibrationProblem(replicates=…, replicate_summary=…)` calibrates a stochastic engine with common random numbers: `"mean"`, `"median"`, `Quantile(q)` or `"crps"` (the continuous ranked probability score). A stochastic engine needs them and a `seed`; there is no default summary.
- The crate `commol-random`: a portable generator (`Xoshiro256PlusPlus`), the derivation of replicate seeds, samplers and a sum tree of rates, for every stochastic engine.
- Two stochastic engines of an aggregate model, in the crate `commol-stochastic`, that run the process of any model as whole units moved at random, with no change to it: `"StochasticEvents"` (continuous time: Gillespie's direct method, exact also when a rate reads `t`, or `method="tau_leaping"`) and `"StochasticSteps"` (the chain binomial, with per-step probabilities). Their settings are `StochasticEventsSettings` and `StochasticStepsSettings`. A total must be whole, and initial amounts that are not whole need `rounding="largest_remainder"` or `"random"`, with no default. A per-step probability outside `[0, 1]`, or probabilities out of one compartment that sum to more than 1, raise `InfeasibleStateError`. See [Stochastic Simulation](https://munqu.github.io/commol/latest/guide/stochastic/index.md).
- The `dynamics` section of a stochastic model is an event table: each flow, the change of one event and its propensity or its draw; the `schedule` section lists the settings of the engine.

### Changed

- **Breaking:** a model is built from a process and a structure. `ModelBuilder` writes the process, and `build(structure, engine, settings=None)` joins it with a structure; the model is a `Model` of the class of the structure.

| Before                                                                                                                                                                              | Now                                                                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `builder.set_initial_conditions(total=…, bin_fractions=[{"bin": "A", "fraction": 0.9}, …], stratification_fractions=[…], compartment_fractions=[…])` then `builder.build(engine=…)` | `builder.build(Aggregate(total=…, bin_fractions={"A": 0.9, …}, stratification_fractions={"group": {"g1": 0.5, …}}, compartment_fractions={"A_g1": 0.0}), engine=…)` |
| `CompartmentalStructure`, `Model[CompartmentalStructure]`                                                                                                                           | `Aggregate`, `Model[Aggregate]`                                                                                                                                     |
| `model.structure.bins`, `stratifications`, `transitions`, `accumulators`                                                                                                            | `model.process.bins`, …                                                                                                                                             |
| `model.structure.initial_conditions.total`, `bin_fractions`, …                                                                                                                      | `model.structure.total`, `bin_fractions`, … (mappings by id)                                                                                                        |
| `model.structure.initial_state()`, `subgroup_total(…)`, `get_conditioning_categories(…)`, `update_compartment_fractions(…)`                                                         | The same methods, with the process first: `model.structure.initial_state(model.process)`, …                                                                         |
| `model.structure.update_initial_conditions(…)`                                                                                                                                      | `model.structure.update_bin_fractions(…)`                                                                                                                           |
| `model.structure.outputs_by_source()`, `series_label(id)`                                                                                                                           | `outputs_by_source(model)`, `series_label(model, id)`                                                                                                               |
| `CompartmentFraction`, `BinFraction`, `StratificationFraction(s)`                                                                                                                   | Mappings in `Aggregate`                                                                                                                                             |
| `CompartmentalEngineType`                                                                                                                                                           | `EngineName`                                                                                                                                                        |

- **Breaking:** the model file has `process` and `structure`, the structure has `kind: "aggregate"` and `variant`, and the fractions are objects keyed by id. A file of 0.3 does not load; rebuild the model and save it again.
- **Breaking:** an engine names a method, not a kind: `EngineType.kinds` replaces `EngineType.kind`, and the error of an engine that does not run a kind lists the engines of that kind.
- **Breaking:** `neighbors` and the names that start with `neighbors_` are reserved: no parameter, bin or accumulator may take them.
- `EngineType.equation_lhs` comes from the form of the dynamics of the engine (`DynamicsForm` in Rust), and returns `None` for an engine without equations.
- Python names follow PEP 8: ruff checks them (`N` rules).

### Fixed

- The unit check reads the flows the engine plans, so a proportional rate is checked as its flow, the rate times its source: `k2` in `1/day` from a bin in `item` passes as `item/day` instead of failing as `1/day`.

### Removed

- **Breaking:** `Model.print_equations`, `ModelStructure.print_equations` and `PrintEquationsOutputFormat`. `model.describe().render("text")` shows the same compact and expanded forms, and `render("latex")` writes them in LaTeX; a description with some units missing shows none instead of raising.

## [0.3.0] - 2026-10-15

### Added

- An ODE engine: a model built with `engine="DifferentialEquations"` runs in continuous time, `dX/dt = Σ flows`, integrated by `diffsol` (`commol-differential`). It has the same flows, outputs, accumulators, calibration, ensembles, reports and plots as a difference model. On `[n, n + 1)` a rate sees `step = n` and the continuous `t`. The engine restarts its integration at each integer when a rate reads `step`, and where `t` reaches a value a rate compares it with (`t >= t_on`), so the solver cannot step over a jump.
- `OdeSolver` (`method` `"tsit45"` or `"bdf"`, `rtol`, `atol`), the settings of the `DifferentialEquations` engine, stored in the model as `Model.engine.settings` and set with `build(engine="DifferentialEquations", settings=...)`.
- One list of engines in the Rust engine (`EngineType`): each entry names the model kind it runs, validates its settings, and says the form of its equations and what a rate means (`per_step` or `continuous`). Python asks it through `commol_rs.engine.EngineType` instead of branching on engine names.
- `Simulation.run_at(at)`: the outputs at the given sorted positions. A differential model takes any position, such as `0.5`.
- `SimulationResults`: what `Simulation.run` and `run_at` return. A `dict` from output name to values, as before, with the positions in `positions` and the rows in `rows()`.
- The window grid as module functions: `window_start`, `window_end`, `window_index` and `window_ends`, beside `windowed_totals`.
- A public Pydantic class rejects an unknown field and names it (`extra="forbid"` on every class of `commol.__all__`), so a misspelt or an old name is an error instead of being dropped.
- Impulses: `add_transition(..., impulse_at=[...])` or `impulse_every=(period, offset)` moves the rate (a fraction or an amount) at once at those steps, in both engines, before the flows of the step. `print_equations` lists impulses apart.
- `ModelBuilder(..., step_unit=...)` (`Model.step_unit`): the unit of one step for the unit check, of any single dimension: the axis is the independent variable of the model, which need not be time (for example `"meter"`). Without it, the step unit is `step`, the default increment of any axis.
- `AxisUnits(step_unit, engine)` and `Model.axis_units()`: `steps(value, unit)` converts a value in any unit of the step dimension to steps on `pint` (`3 day` with a step of `hour` is `72.0`; `2 km` with a step of `m` is `2000.0`), and raises, naming both units, for a unit of another dimension. `rate_from_mean_span(span, unit)` and `rate_from_probability(probability, period, unit)` ask the engine what a rate means: the per-step rate for `DifferenceEquations` (`1 - exp(-1/n)`, `1 - (1 - p)^(1/n)`) and the continuous rate for `DifferentialEquations` (`1/n`, `-ln(1 - p)/n`).
- `EngineType.rate_from_mean_span` and `rate_from_probability` in the engine's list (`RateMeaning` in Rust), which hold those formulas.
- The unit check reads `if(condition, a, b)`: both branches share a unit.
- One interface for every kind of model: `ModelStructure`, an abstract base class that the structure of each kind inherits. `Simulation`, `Calibrator`, `ProbabilisticCalibrator`, the reports and `SimulationPlotter` use only it, so another kind (for example an agent network) is a new subclass with its engines in the engine's list. `CompartmentalStructure` is the structure of a compartmental model. A subclass that misses a method cannot be made, and the error names the method.
- An engine runs one model kind, and a model whose engine runs another kind is rejected when it is made, with an error that names the engine and both kinds.
- Each kind declares the values calibration may set besides `parameter` and `factor` (`ModelStructure.calibration_value_types`); a compartmental model accepts `initial_condition`. A type the kind does not accept raises an error that names the kind.

### Changed

- **Breaking:** a `DifferenceEquations` step that leaves a compartment below zero (by more than `10⁻¹²` of the total) raises `InfeasibleStateError`, a `ValueError`. The message names the compartment, the step, its amount before the step and the flows out of it. Before, the run went on with negative amounts. A calibration gives such a run a penalty and goes on; any other error of a run still stops it. See [Amounts below zero](https://munqu.github.io/commol/latest/guide/transitions/#amounts-below-zero).
- **Breaking (Rust):** `SimulationEngine::run_at_positions_into_buffer`, `run_at`, `run` and `step` return `SimulationError` (`Infeasible` or `Failed`) instead of `String`.
- A position on the axis is a `float` everywhere, and the engine alone decides which positions its model type accepts: a difference model integer steps, a differential model any non-negative position. Its error names the observation, constraint, impulse transition or series parameter that holds a rejected position. The engine trait runs only at positions (`run_at_positions_into_buffer`); `run(n)` returns the grid it ran on.
- The engine of a model and its settings are one field, `Model.engine = {type, settings}` (`ModelEngine`), which replaces `Model.dynamics.typology`. The builder takes the engine where it builds, so one builder builds the model for each engine with its own settings. Settings of one engine given to another raise an error from the engine that names both. A model file saved before has `dynamics.typology` and does not load: move it to `"engine": {"type": …, "settings": null}`.

| Before                                                                                                   | Now                                                          |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `builder.build(typology="DifferenceEquations")`                                                          | `builder.build(engine="DifferenceEquations")`                |
| `model.dynamics.typology`                                                                                | `model.engine.type`                                          |
| `commol_rs.difference.DifferenceEquations(model)`, `commol_rs.differential.DifferentialEquations(model)` | `commol_rs.engine.Engine(model)`, the engine the model names |
| - A model is a header, an engine and a `structure`: `Model.structure`, tagged                            |                                                              |
| in JSON by `kind`, replaces `Model.population` and `Model.dynamics`, and the                             |                                                              |
| transitions are held once. `Model` is generic in its structure class:                                    |                                                              |
| `ModelBuilder.build` returns `Model[CompartmentalStructure]`, and                                        |                                                              |
| `Model.from_json` reads the class from the `kind`. The methods that only a                               |                                                              |
| compartmental model has moved to its structure. A model file saved before                                |                                                              |
| has `population` and `dynamics` and does not load: move the fields of                                    |                                                              |
| `population` and `dynamics.transitions` into                                                             |                                                              |
| `"structure": {"kind": "compartmental", …}`.                                                             |                                                              |

| Before                                                                                                        | Now                                                                                    |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `model.population.bins`, `.accumulators`, `.stratifications`, `.initial_conditions`                           | `model.structure.bins`, …                                                              |
| `model.dynamics.transitions` (and `model.population.transitions`)                                             | `model.structure.transitions`                                                          |
| `model.initial_state()`                                                                                       | `model.structure.initial_state()`                                                      |
| `model.update_initial_conditions(…)`, `update_stratification_fractions(…)`, `update_compartment_fractions(…)` | `model.structure.update_initial_conditions(…)`, …                                      |
| `model.subgroup_population(…)`, `model.get_conditioning_categories(…)`                                        | `model.structure.subgroup_total(…)`, `model.structure.get_conditioning_categories(…)`  |
| `model.get_outputs_by_source()`                                                                               | `model.structure.outputs_by_source()`                                                  |
| `model.get_uncalibrated_initial_conditions()`                                                                 | `model.structure.uncalibrated_values()`                                                |
| `commol.context.dynamics` (`Transition`, `StratifiedRate`)                                                    | `commol.context.transition`                                                            |
| `commol.context.engine.EngineType`                                                                            | `CompartmentalEngineType`; `ModelEngine.type` is a `str` that the engine's list checks |
| `commol_rs.core.Model(population=…, dynamics=…)`                                                              | `commol_rs.core.Model(structure=commol_rs.core.CompartmentalStructure(…))`             |
| `commol_rs.model.initial_state(model)`, `expanded_outputs`, `transition_flows`                                | they take a `commol_rs.core.CompartmentalStructure`                                    |
| `commol_rs.calibration.check_initial_conditions`                                                              | `check_model_values`                                                                   |
| `commol_rs.calibration.calibrate(…, initial_population_size, …)` and the other calibration functions          | no `initial_population_size`: the engine knows it                                      |

- `applied_parameters_report` returns `parameters` and the sections of the kind of the model; a compartmental model adds `initial_conditions`, with the same keys as before.
- The Rust engine trait (`SimulationEngine`) is the same for every model kind: `compartments`, `state` (was `population`), `set_initial_condition` and `output_labels` moved to `CompartmentalEngine` (`commol-model`), and the engine resolves and writes the values of its kind (`plan_values`, `apply_values`). The initial-condition plan moved from `commol-calibration` to `commol-model`. The crates are not published.
- Names of positions, lengths and counts, and names with more than one meaning. No old name is kept; an old keyword raises an error that names it.

| Before                                                                                                    | Now                                                    |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `ObservedDataPoint(step=…)`                                                                               | `ObservedDataPoint(at=…)`                              |
| `ObservedDataPoint(window_steps=…)`                                                                       | `ObservedDataPoint(window=…)`                          |
| `ObservedDataPoint(compartment="B")`                                                                      | `ObservedDataPoint(output="B")`                        |
| `ObservedDataPoint(compartment="label", compartments=[…])`                                                | `ObservedDataPoint(label="label", outputs=[…])`        |
| `ObservedDataPoint(scale_id=…)`                                                                           | `ObservedDataPoint(factor_id=…)`                       |
| `CalibrationParameter(parameter_type="scale")`, `CalibrationParameterType.SCALE`                          | `"factor"`, `CalibrationParameterType.FACTOR`          |
| `CalibrationConstraint(time_steps=…)`                                                                     | `CalibrationConstraint(at=…)`                          |
| `NelderMeadConfig(initial_step=…)`                                                                        | `NelderMeadConfig(initial_simplex_size=…)`             |
| `Simulation.run_at(steps)`                                                                                | `Simulation.run_at(at)`                                |
| `Simulation.windowed_totals(results, source_ids, window_steps, at_steps)`                                 | `Simulation.windowed_totals(results, ids, window, at)` |
| `Simulation.total_series(results, source_ids)`, `group_outputs(source_ids)`, `outputs_for(source_id)`     | `ids`, `id`                                            |
| `windowed_totals(series, window_steps, at_steps)`                                                         | `windowed_totals(series, positions, window, at)`       |
| `window_end_steps(window_steps, num_steps)`                                                               | `window_ends(window, end)`                             |
| `TimeScale.window_start`, `window_end`, `window_index`                                                    | `commol.window_start`, `window_end`, `window_index`    |
| `simulation_report(…, steps, window_steps, …)`, `series_report`, `accumulators_report`, `observed_report` | `at`, `window`                                         |
| `SimulationPlotter.plot_series(step_to_label=…)`                                                          | `position_label=…`, which takes a `float`              |
| `TimePattern` (`commol.api.time_patterns`)                                                                | `AxisPattern` (`commol.api.patterns`)                  |
| `TimePattern.window(start, end, amount)`                                                                  | `AxisPattern.interval(start, end, value)`              |
| `TimeSeriesValue`, "time-series parameter"                                                                | `SeriesValue`, "series parameter"                      |
| `EnsembleSolution.windowed_prediction_steps`                                                              | `windowed_prediction_positions`                        |
| `engine.current_step()`                                                                                   | `engine.current_position()`                            |
| `simulation.run(n, output_format="list_of_lists")`                                                        | `simulation.run(n).rows()`                             |

- Reports, windows, plots and ensemble statistics read a value at a position of the results, not at a list index, so a differential model windows and reports at non-integer positions once its results hold them. The report key `scale_id` is `factor_id`, and `compartments` is `outputs`.
- `EnsembleSolution` has `prediction_positions`; its prediction series hold the unit grid and every observation position and window start.
- Series parameters take `(position, value)` points with `float` positions. `"linear"` interpolates in `t` and `"step_function"` holds from any position; a differential engine restarts its integration at their points, so a `"linear"` series is a ramp, not a staircase. `"pulse"` keeps integer positions. A difference model gives the same results and rejects a point between integers.
- `AxisPattern.interval` and `linear_ramp` read `t` and take `float` bounds. A difference model gives the same results; in a differential model an interval opens at any position and a ramp is continuous.
- A pattern parameter that the builder generates is named `_series_<n>` (was `_ts_<n>`).
- The model compiler moved from `commol-difference` to the new crate `commol-model`, which both engines use; `commol_rs.difference` no longer has `initial_state`, `expanded_outputs`, `transition_flows` and `check_names`, which are in `commol_rs.model`. No public Python name changes.
- `SimulationPlotter.plot_series` and `plot_cumulative` take `series` instead of `bins`. An entry is an output name or a bin or accumulator id; an id plots the sum of its outputs, so a bin of a stratified model can be plotted. There is no `bins` alias: rename the argument.
- Every loss uses the observation weights the same way: `"sse"` is `Σ w·r²`, `"rmse"` is `√(Σ w·r² / Σ w)` and `"mae"` is `Σ w·|r| / Σ w`. Before, `"sse"` squared the weight and `"rmse"` and `"mae"` ignored it. With every weight at 1 nothing changes. The ensemble central loss follows the same rule.
- `Simulation.run` starts from the initial state at every call, so two calls give the same result. Before, a second `run` continued the first one. To continue a run, call `simulation.engine.step()`.
- `Model.check_unit_consistency` infers one step unit from the parameter units (`day`, `week`, `step`, …; `step` when there is none) and requires every flow to be exactly `<bin unit>/<step unit>`. A model in `"1/step"` now passes. Rates in two units of time, such as `"1/day"` and `"1/week"`, are an error that names each parameter, because the engine converts nothing. Before, the check accepted any convertible unit and defaulted to `day`.
- Nelder-Mead searches inside the bounds: it optimizes an unbounded coordinate and scores `x = lo + (hi − lo)·(sin u + 1)/2`. Its first simplex is set by `NelderMeadConfig.initial_simplex_size` (a fraction of each bound width, default 0.1), which replaces `simplex_perturbation`; there is no alias. Every Nelder-Mead result can change.
- The library names what the compartments hold without assuming individuals: the **state** (the amount of every compartment), the **amount** (the value of one compartment), the **total** (their sum, which the initial fractions are taken of) and a **subgroup total** (the sum of the compartments with some categories). The total is a `float`, finite and greater than 0, so a model of a mass or a concentration can be stated; before, it was an `int`. The rate variable `N` is `total` and `N_<cat>…` is `total_<cat>…`, so `N` is a free id (a bin may be named `N`). `AxisPattern` takes a `value`. No old name is kept: a model file with `population_size` does not load, and the error names the field; rename it to `total`, and `N` to `total` in every rate, formula and constraint.

| Before                                                                                                                             | Now                                                                                         |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `set_initial_conditions(population_size=…)`, `InitialConditions.population_size: int`, JSON key `population_size`                  | `set_initial_conditions(total=…)`, `InitialConditions.total: float`, JSON key `total`       |
| Rate, formula and constraint variables `N`, `N_<cat>…` (`N_g1`, `N_g1_t1`)                                                         | `total`, `total_<cat>…` (`total_g1`, `total_g1_t1`)                                         |
| `RESERVED_VARIABLE_NAMES = ("N", …)`                                                                                               | `("total", "step", "t", "pi", "e")`                                                         |
| `InitialConditions.subgroup_population(…)`, `CompartmentalStructure.subgroup_population(…)`                                        | `subgroup_total(…)`                                                                         |
| `InitialCompartment.population`                                                                                                    | `InitialCompartment.amount`                                                                 |
| `applied_parameters_report`: row `"N"`, group `"population"`                                                                       | row `"total"`, group `"total"`                                                              |
| `AxisPattern.pulse(at, amount)`, `pulses(at, amount)`, `periodic(period, amount, offset)`, `interval(start, end, amount)`          | `value` instead of `amount`                                                                 |
| `get_predefined_variable_units(stratifications, bin_unit)`                                                                         | `get_predefined_variable_units(subgroup_totals, bin_unit)`, with the names the engine gives |
| `commol_rs.engine.Engine.population`                                                                                               | `Engine.state`                                                                              |
| `commol_rs.core.InitialConditions(population_size=…)`                                                                              | `InitialConditions(total=…)`                                                                |
| `commol_rs.calibration.calibrated_initial_population`                                                                              | `calibrated_initial_state`                                                                  |
| —                                                                                                                                  | `commol_rs.model.subgroup_total_names(structure)`                                           |
| Rust: `CompartmentalEngine::population`, `population_size`; `DifferenceEquations::population`, `DifferentialEquations::population` | `state`, `total`                                                                            |

- The library and its documentation assume no application and no axis: text says "along the axis" or "position", not "over time", and examples use abstract names. A rate that names no compartment is **proportional** (was "per-capita"), because a source can hold a mass or a concentration. The errors of a constraint at positions say "Constraint '…' evaluation failed at position …" and "Position … not found …" (were "Time-dependent constraint" and "Time step").

| Before                                                                | Now                                                        |
| --------------------------------------------------------------------- | ---------------------------------------------------------- |
| `AxisPattern.seasonal(amplitude, period, phase, baseline)`            | `AxisPattern.sinusoid(amplitude, period, phase, baseline)` |
| `TransitionFlow.per_capita_of` (`commol.context._model.engine_state`) | `TransitionFlow.proportional_to`                           |
| Guide anchor `#absolute-and-per-capita-rates`                         | `#absolute-and-proportional-rates`                         |
| Guide anchor `#time-dependent-rates`                                  | `#rates-that-change-along-the-axis`                        |

- The unit check no longer infers the step unit from the parameter units: it is the declared `step_unit` or else `step`. A model with rates in `1/day` and no `step_unit` fails the check, and the message says to declare it: add `step_unit="day"` to `ModelBuilder`.
- A unit means what `pint` defines. The list of time aliases (`decade`, `quarter`, `semester`, `mo`, `wk`, …) is gone: `pint` reads `quarter` as a mass and `decade` as a number, so `1/quarter` was a mass without error. A name that `pint` does not define is a unit of its own dimension.
- In the unit check, `t` has the step unit (it is a position on the axis), and a number written beside it in a sum, a difference or a comparison is read in that unit. `step` stays dimensionless. A rate such as `sin(2 * pi * t / 365)` now fails the check; write `sin(2 * pi * t / period)` with `period` in the step unit.
- Guides and docstrings say what a rate means for each engine ([What a rate means](https://munqu.github.io/commol/latest/guide/core-concepts/#what-a-rate-means)) instead of "every flow applies once per step".

### Removed

- `TimeScale` (`commol.api.time_scale`). Its rate conversions gave per-step rates only, which a differential model reads as too small.

| Before                                                             | Now                                                             |
| ------------------------------------------------------------------ | --------------------------------------------------------------- |
| `TimeScale(step_seconds=3600)`, `TimeScale.from_step_unit("hour")` | `AxisUnits(step_unit="hour")`, or `model.axis_units()`          |
| `steps_from_seconds/minutes/hours/days/weeks(x)`                   | `steps(x, "second")`, `steps(x, "minute")`, …                   |
| `steps_per_hour`, `steps_per_day`, `steps_per_week`                | `steps(1, "hour")`, `steps(1, "day")`, `steps(1, "week")`       |
| `rate_from_mean_duration(seconds)`                                 | `rate_from_mean_span(span, unit)`, which asks the engine        |
| `rate_from_probability(p, period_steps)`                           | `rate_from_probability(p, period, unit)`, which asks the engine |
| `window_start`, `window_end`, `window_index`                       | `commol.window_start`, `window_end`, `window_index`             |

`steps` returns a `float` and does not raise for a fraction of a step; the engine rejects a position it does not accept where the position is used.

- Transition conditions: `Transition.condition`, the `condition` argument of `ModelBuilder.add_transition`, `ModelBuilder.create_condition`, the `Condition` and `Rule` types, and the `LogicOperators` and `VariablePrefixes` enums, in Python and in Rust. The engine never evaluated them, so a condition that was never true still moved population. Write the condition in the rate (`"if(step > 30, k, 0)"`), or use an `AxisPattern` or a stratified rate. A model file with a `condition` key raises an error that names it.
- `Simulation.run(..., output_format=...)` and `run_at(..., output_format=...)`: both return `SimulationResults`; use `rows()` for one row per position.
- `commol_core::integer_steps`, `SimulationEngine::run_into_buffer` and `run_at_steps_into_buffer`; the Python checks that repeated the engine's rules on positions (`check_steps_for_typology`, the impulse check of `Dynamics`, the order check of `Simulation.run_at`).
- The `"weighted_sse"` loss and `LossConfig.weighted_sse()`. Use `"sse"`, which uses the weights.
- `commol.constants.ModelTypes`. The names of the compartmental engines are the `Literal` `commol.context.engine.CompartmentalEngineType`.
- `Population` and `Dynamics`, in Python (`commol.context.population`, `commol.context.dynamics`) and in Rust, and their PyO3 classes; the structure of a model is `CompartmentalStructure`.
- `Model.get_uncalibrated_initial_conditions`, `Model.initial_state`, and the other methods of a compartmental model on `Model`: they are on `model.structure` (see Changed).

### Fixed

- Examples 6 to 9 of the guide (`examples.md`) no longer go below zero. Example 6 writes the rate it describes, `k2_max * B * max(0, 1 - C / capacity)`; Examples 7, 8 and 9 set `per_compartment=True` on `flow_AB`, so each compartment of `A` loses an amount computed from its own amount.
- A differential model: on a piece of integration that ends where a rate jumps (a value `t` is compared with, a series point), the right-hand side sees `t` below the end. Before, the last solver step of the piece already saw the rate after the jump, so a rate of 0.5 that starts at 2.5 moved `3·10⁻⁸` of the state before 2.5.
- `CalibrationResult.converged` is `True` only when the optimizer met its convergence test or `target_cost`. Before, every finished run reported `True`, also one that stopped at `max_iterations`.
- Nelder-Mead from a start value of 0 builds a real simplex, and no longer stops at a bound where the clamped loss is flat. Before, the simplex was degenerate at 0, and could collapse on a bound and report `converged=True`.
- The verbose Nelder-Mead run uses the configured simplex. Before, it always used a 10% perturbation.
- `Model.check_unit_consistency` names every bin without a unit. Before, it failed with "'None' is not defined in the unit registry".
- A `Simulation` follows its model: `run` and `engine` rebuild the engine when the model changed after the `Simulation` was created, for example with `update_parameters` or `apply_calibration_parameters`. Before, it used the old values, or failed with "Parameter index … not found" for a parameter that was `None`.
- `pi` and `e` work in every transition rate, such as `"pi * k * A"`. Before, outside a function argument, the simulation failed with "Variable 'pi' not found".
- `%` (remainder) works as the only operator of a rate, as in `"step % 7"`. Before, the rate was read as a parameter name.
- A `total_…` variable that names no subgroup total, such as `total_t1_g1` when `group` is declared before `type`, is a build error that lists the valid names. Before, a formula parameter or the engine gave it the value 0.
- A bin or an accumulator id cannot be `total`, `step`, `t`, `pi` or `e`, and a bin cannot expand into a name that is also a `total_<categories>` subgroup total. Before, the engine variable hid such a bin, and the two sums shared one value.
- A parameter id cannot be `total`, `step`, `t`, `pi`, `e`, an expanded compartment or a subgroup total. Before, only bin ids were rejected, and the parameter and the engine variable shared one value.
- `ModelBuilder.reset` removes the accumulators, and `ModelBuilder.clone` copies them and the `bin_unit`. Before, `reset` kept the accumulators and a clone lost them.
- An `initial_condition` id is resolved by the engine when the `Calibrator` or the `ProbabilisticCalibrator` is created. A category of a non-binary stratification now gives "needs a binary stratification" instead of "not found", and bin prefixes (`"B_g1"`) and qualified categories (`"group:g1"`) are accepted, as documented. Before, both were rejected as "not found".

### Documentation

- The guide is rewritten and split into pages with one job each (building models, transitions, expressions, time patterns, units, simulations, and six calibration pages). Every example runs in CI, and every claim was checked against the engine. Behaviors that were described wrongly are now documented as they are, among them: a violated constraint replaces the loss with a penalty.
- The API reference documents every public name, and `Model`, plotting and reports have their own pages.
- The site publishes `llms.txt` and `llms-full.txt` for AI agents, and marks the pages of the `dev` version.
- `README.md` presents Commol as a library for declarative dynamical modelling: the idea behind it, what it can model and its limits, an example that derives, calibrates and simulates a model, and links to the guides. The citation title is "Commol: High-Performance Declarative Dynamical Modelling in Python".
- The package metadata on PyPI has the same description, and keywords and classifiers (Python versions, audience, topic, development status).

## [0.2.0] - 2026-10-12

### Added

- Per-compartment initial conditions: `InitialConditions.compartment_fractions` (Python and Rust) accepts overrides keyed by expanded compartment name, expressed as fractions of the total population and applied after the uniform expansion. Includes `ModelBuilder.set_initial_conditions(compartment_fractions=...)`, `Model.update_compartment_fractions`, and `Model.initial_state()`, which returns each expanded compartment with its bin, categories and initial population as the engine builds it. The Rust engine is the single source of truth for the expansion and validates the overrides (unknown or repeated name, fraction outside `0.0..=1.0`, total above `population_size`).
- Rust `commol_difference::initial_state` and `commol_calibration::calibrated_initial_population`, exposed to Python. The calibration objective, the calibrated predictions and the model write-back share one implementation of how calibration values reach the engine.

### Changed

- **Breaking:** calibrated initial conditions no longer auto-calculate the last population-fraction value as the remainder of the population. The new `CalibrationProblem.initial_condition_remainder` names the compartment or bin that absorbs the difference; it is not calibrated and keeps its own split. Without it, values are applied as given and only an initial population above `population_size` is penalized. A single exact compartment, such as a composite state, can now be calibrated on its own.
- **Breaking:** calibrated initial conditions are applied in fixed layers — bins, category splits, exact compartments, remainder — instead of in parameter order.
- **Breaking:** a stratification category may be named with a qualified id (`sex:female`). A bare name that more than one stratification declares is rejected as ambiguous, in calibration and in `update_stratification_fractions`, and the report qualifies its rows. Category pairs come from the engine's compartment labels instead of parsing compartment names.
- **Breaking:** `get_category_fraction`, `subgroup_population` and `get_conditioning_categories` resolve category ids like calibration: a qualified id (`sex:female`) selects its stratification and a bare name that more than one stratification declares raises instead of silently using the first one. `get_conditioning_categories` returns qualified ids for shared names.
- Bin and bin-prefix initial-condition ids, and the remainder, resolve from the engine's compartment labels instead of name prefixes, so a bin `A` no longer covers the compartments of a bin `A_x`.
- The Rust engine expands bins and accumulators in one function; the equation printer and unit checker use it (`expanded_outputs`) instead of their own Python expansion. The Rust `ModelTypes` gains `DifferentialEquations`, so any model definition can be expanded; the difference engine rejects it.
- **Breaking:** `update_bin_fractions` / `update_initial_conditions` apply the same rule as construction: the bin fractions must sum to 1.0 after the update.
- **Breaking (Rust):** `SimulationEngine` requires `compartment_labels()`. `CalibrationProblem::fix_auto_calculated_parameters` and `get_parameter_fix_info` are removed; `applied_parameter_values` returns the clamped values the objective scored, which is what results now report.
- **Breaking:** transitions are planned by the Rust engine from compartment labels (bin and categories) instead of compartment names, in one planner shared by the simulation and `print_equations`. The printed equations are therefore the simulated ones: a per-capita rate is shown multiplied by its source (`gamma * I`), a transition without flows shows `(no flows)`, and the compact form lists the flows of each transition.
- **Breaking:** building a model rejects a stratified rate whose `category` conditions match no source compartment (use `to` to route into a stratification the source does not carry), a transition with several sources and a per-capita rate, a category transition without exactly one source and one target, and a flow that ends in categories no compartment has. These used to be skipped without notice.
- Constraint aggregates (a bin or bin prefix in a constraint expression) are built from output labels, so a bin `A` no longer sums the outputs of a bin `A_x`.

### Fixed

- A transition between bins no longer moves bins whose id starts with the source bin id: `A -> B` moved `AB -> BB` as well.
- Transitions between the categories of a stratification (`young -> old`) are now simulated; the engine used to ignore them while `print_equations` showed them.
- A transition with several sources or targets (`A + B -> 2 C`) now applies its flow to every source and target, as documented; the engine used to move only the first source to the first target.
- The applied-parameters report now shows the effective head count for bin, stratification-category, and subgroup rows — the uniform expansion plus the declared `compartment_fractions` overrides — and reports each `fraction` as that head count over the declared `population_size`, so every row shares one denominator with `N`. An aggregate row written back as overrides no longer reports the stale pre-calibration value as `calibrated`, and a bin left uncalibrated keeps its `null` marker even when overrides touch some of its compartments.
- A conditional-stratification category is calibrated as a fraction of its conditioning subgroup but reported as a fraction of the whole population, so the report rescales its `interval` by the subgroup's share of the population; the interval now brackets the reported `fraction`.
- `apply_calibration_parameters` writes initial-condition values back so the model rebuilds the initial population the engine scored for them. Bin and category values update the fractions they declare, as the engine applied them; every compartment those fractions do not reproduce is pinned as a per-compartment override. The call computes the result before it changes the model, so a rejected value leaves the model unchanged.
- `population_size` must be greater than zero, reported as a domain `ValueError` rather than a later `ZeroDivisionError`. `update_bin_fractions` re-applies the sum-to-1/conservation rule.
- Calibrating a bare category name that two stratifications declare no longer builds an initial population larger than `population_size`: the engine used to split the overlapping pairs of both stratifications from one snapshot.
- The PyO3 `InitialConditions` binding exposes `population_size`, `bin_fractions`, `stratification_fractions` and `compartment_fractions` with getters, and the `_commol_rs` stub declares only those real attributes (correcting the `bin_fractions` field name).
- An initial-condition override that names an unknown compartment raises a `ValueError` when the model is built or updated, and a `RuntimeError` from the `DifferenceEquations` engine constructor.
- The greedy ensemble search of probabilistic calibration takes its random starts from the shuffled part of the sample (argmin fork). The starts were biased towards low indices, so the same seed can now select a different ensemble.

### Dependencies

- cranelift 0.126, rand 0.10 (with the argmin fork migrated to it), thiserror 2, evalexpr 13 and pyo3 0.29.

## [0.1.0] to [0.1.3] - 2026

See the [GitHub Releases](https://github.com/MUNQU/commol/releases) page for the details of these versions.

### Added

- GitHub Actions CI/CD pipelines for code quality, building, and PyPI releases
- MkDocs Material documentation site
- Comprehensive development workflow documentation
- Contributing guidelines and release process documentation

## [0.0.1] - 2025

### Added

- Initial release of Commol
- ModelBuilder fluent API for constructing compartment models
- Support for difference equations models
- Compartment states (bins) and stratifications
- Mathematical expression support in transition rates
- Rust-powered simulation engine
- Python API with Pydantic validation
- Basic compartment model support (difference equations)
- Initial conditions and parameter management
- Simulation runner with multiple output formats

### Security

- Expression validation to prevent code injection
- Input sanitization for mathematical formulas

______________________________________________________________________

## Version History

- **Unreleased** - Current development version
- **0.4.0** - One process under several structures and engines, stochastic engines and model descriptions
- **0.3.0** - ODE engine, positions on the axis and one interface for every model kind
- **0.2.0** - Per-compartment initial conditions and engine-planned transitions
- **0.1.0 to 0.1.3** - First 0.1 versions
- **0.0.1** - Initial release with core functionality

For detailed information about each release, see the [GitHub Releases](https://github.com/MUNQU/commol/releases) page.
