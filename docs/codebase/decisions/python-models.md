# Python Models

[Back to Design decisions](../design-decisions.md)

How the Pydantic models of the Python package read their fields, check their
values and compare.

## Public models forbid unknown fields

Every public Pydantic class sets `model_config = ConfigDict(extra="forbid")`.
An unknown field, such as a misspelt or an old name (`window_steps`), raises a
`ValidationError` that names it, instead of being dropped with no error.

**Guard:** `tests/building/test_unknown_fields.py`.

## A value of a model has its rule in the engine

An initial amount, the fractions of a stratification, an impulse and the
points of a series have their rule once, in `crates/commol-model/src/rules.rs`
(`check_bin_amount`, `check_compartment_amount`, `check_fractions`,
`check_impulse`, `check_series`). The model
compiler calls them when it compiles a model, `set_initial_condition` when a
calibration writes an amount, and the Python object of the value when it is
built: the validators of `InitialConditions`, `Transition` and `SeriesValue`
call the bindings of `commol_rs.model`. A value outside its rule fails at the
line that sets it, with the message of the engine. A range is a `Real` of
`commol-core` after the owner of the value (`Bin 'B': initial amount is a
finite number of at least 0, got -1`); a rule that is not a range, such as one
point per position or fractions that sum to 1, has one message in the same
module. `update_stratification_fractions` gives the one category left unnamed
the rest of 1 and checks the result with `check_fractions`; the rule that at
most one category is left unnamed stays in Python, because it is a rule of the
update, not of a value, and the engine never gets an update.

| Alternative | Why not |
|---|---|
| A check in Python and one in the compiler | Two messages for one rule, and rules that only one side has, such as the series without a point. |
| A check only when the model compiles | The error comes far from the line that set the value. |
| A row of `RANGES` for the object (`RangeChecked`) | A rule of `RANGES` has a fixed name, so the message cannot name the bin or the transition, and `(period, offset)` has two rules. |

**Rule:** a new value of a model gets its rule in `rules.rs`, which the
compiler and the validator of its Python object call.

**Guards:** `tests/building/test_value_rules.py`,
`crates/commol-model/tests/value_rules.rs`.

## A data object crosses as the JSON of its dump

Every data object of the Python API that reaches the engine, the model, the
calibration problem and each inference method included, crosses as the JSON
of its `model_dump()` (`engine_json` of
`py-commol/commol/context/engine_json.py`). The bridge reads it with one
helper, `read` of `py-commol/src/json.rs`, into the type the engine owns,
which takes the spelling of the Python API (`rename_all = "snake_case"`) and
denies unknown fields. An error names the path of the field and gives the
message of the engine (`optimization_config.inertia_factor: inertia_factor is
a number greater than 0, got -1`). A number that is not finite crosses as
`Infinity`, `-Infinity` or `NaN`, which the reader takes.

Each union of classes carries one tag, `kind`, with a default, so a field is
never read by the first variant that matches; `tagged!` of
`commol_core::json` reads the tag, keeps the path and checks the range rules
of the variant. A method crosses whole, into one settings type of
`crates/commol-inference/src/settings/`, and the engine decides what the
method needs: the fit with its starts, the calibration of `η`, the threshold
of a profile and the checks of the names. A field that holds a result with
behavior in the engine, such as a sample, crosses as its handle: the class
names it in `_engine_handles`, and `engine_handles` gives it as a keyword
argument. A `#[pyclass]` is a handle or a result, never data.

| Alternative | Why not |
|---|---|
| A PyO3 class for each data class, filled field by field in Python | Three shapes of each value (the pydantic class, the conversion, the PyO3 class), four edits per field, and nothing checks that they agree. |
| The fields of a method as positional arguments | A long signature per method, and a copy of its fields in the runner. |
| An untagged union | No path in the error, and the first variant that matches wins. |
| `pythonize` (Python objects to serde with no string) | A second mechanism next to the JSON of the model. |

**Rule:** a new field of a data object gets the same name in its Rust type,
and nothing else; a new data class gets its Rust type, an entry of
`objects!` in `py-commol/src/json.rs` and a valid object in
`tests/support/data_objects.py`.

**Guards:** `tests/suite/test_bridge.py` (the engine reads back each dump),
`tests/suite/test_architecture.py` (no data class in the bridge, no
conversion in Python, each runner sends its method whole),
`tests/calibration/test_settings.py` (the path of an error),
`crates/commol-core/tests/json.rs`, `crates/commol-calibration/tests/test_json.rs`.

## Results compare by their values

Every result of commol compares with one rule, `ComparedByValue`
(`py-commol/commol/context/result_model.py`). Its subclass `ResultModel` is
the base of every result model: the result of a calibration
(`CalibrationResult`, `CalibrationHistory`), of each uncertainty,
identifiability and sensitivity method, the parameter sample, the bands of a
predictive, the summary of a scenario, a score, and their parts
(`StartSummary`, `SeriesIntervals`, …); it compares its public fields. The
results of a run and of replicates (`SimulationResults`, a `dict`, and
`SimulationReplicates`, a `list`, in `py-commol/commol/api/results.py`)
compare their values or runs, positions, seed and model. Two results are equal
when they are of one class and those values are the same. The comparison reads
`canonical`, the values as nested tuples, also of a setting or a `Model`
inside a result, such as the `Threshold` of a `ProfileResult`:

| Value | Rule |
|---|---|
| NaN | Equal to every NaN, whatever the object |
| `0.0` and `-0.0` | Equal, with one hash, as `==` of floats |
| A list or a tuple | Item by item, in order; a list is not equal to a tuple |
| A dict | Its items, in any key order |
| A private attribute, such as the Rust sample of `ParameterSample` | Not compared |

An object that is not a result, such as a plain `dict` or `list` with the
same items, is not equal to one: `__eq__` gives `False`, not
`NotImplemented`, so `dict.__eq__` does not answer by the values, and
`__ne__` is its negation. A frozen result (`frozen=True`) has the hash of
`canonical`, so equal results have equal hashes. A result that can change
(`SimulationResults`, `SimulationReplicates`, `CalibrationResult`,
`CalibrationHistory`) has no hash. A setting keeps the `==` of Pydantic.

| Alternative | Why not |
|---|---|
| The `==` of Pydantic | A list compares the identity of its items first, and a NaN is not equal to itself, so two runs with one seed differ when they hold a NaN: the engine gives new objects. It also compares the private attributes. |
| No `==` on results | A caller could not check that a seed repeats a result. |
| Comparing `model_dump_json()` | Each caller must know it, and a string is built for each comparison. |
| An `__eq__` per result | Rules that drift apart from one result to the next. |
| The `==` of `dict` and `list` for runs and replicates, with NaN equal to NaN | Two rules, and runs at other positions or with other seeds and the same values stay equal. |

**Rule:** a new result model has `ResultModel` among its bases, and a new
method returns one.

**Guards:** `tests/uncertainty/test_results_equality.py`,
`tests/uncertainty/test_results_equality.py::test_every_method_of_commol_returns_a_result_model`
and `tests/simulation/test_results_compare.py`.
