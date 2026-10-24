# Python Models

[Back to Design decisions](../design-decisions.md)

How the Pydantic models of the Python package read their fields and compare.

## Public models forbid unknown fields

Every public Pydantic class sets `model_config = ConfigDict(extra="forbid")`.
An unknown field, such as a misspelt or an old name (`window_steps`), raises a
`ValidationError` that names it, instead of being dropped with no error.

**Guard:** `tests/building/test_unknown_fields.py`.

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
