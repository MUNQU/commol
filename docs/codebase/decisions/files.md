# Files

[Back to Design decisions](../design-decisions.md)

How a model and a result are written to a JSON file and read back.

## Model files are written in full

`Model.to_json` writes every field, nulls included. Dropping nulls is not
round-trip safe: an uncalibrated `Parameter.value` or a `None` in `bin_amounts`
would lose a required key and fail to reload. `render_json`
(`_data/serialization.py`) keeps files readable by writing long numeric
arrays on one line. It is internal on purpose: an output format in the public
API becomes a compatibility promise.

**Guards:** `tests/building/test_save_and_load.py::test_an_uncalibrated_parameter_round_trips`,
`tests/building/test_save_and_load.py::test_an_uncalibrated_bin_amount_round_trips`.

## A result file is read by its class

Every result model writes itself with `to_json(file_path, indent=2)` and its
class reads it back with the class method `from_json(file_path)`, which
returns that class (`Self`). Both live on `ResultModel`
(`_data/result_model.py`), so a new result has them, and they write as
`Model.to_json` does: UTF-8, every field, long numeric arrays on one line
(`render_json`). `to_json` renders the JSON of `model_dump_json`, which
writes a number that is not finite as the text `"Infinity"`, `"-Infinity"`
or `"NaN"` (`ser_json_inf_nan`); `model_dump(mode="json")` keeps the float,
and `json.dumps` would write a bare `NaN`, which is not JSON. The file holds the fields of the result and no tag of its
class: the class that reads it validates it, so the file of another result
raises the `ValidationError` of pydantic, which names each field. A result
read back holds what the methods read: a private attribute that a validator
builds, such as the Rust sample of `ParameterSample`, is built again. A file
is read by the release that wrote it, as a model file is. JSON is the format
of a file only: in one process, a method reads the object the calibration
returned.

| Alternative | Why not |
|---|---|
| `Calibrator.read_result(path)` | Reading values would need a model and a problem, and the return would name no class. |
| A function `commol.read_result(path)` | Its return is a union of every result, which a typed caller narrows by hand. |
| `save` and `load` | A second pair of names for the idea of `Model.to_json` and `Model.from_json`. |
| A tag of the class in the file | `from_json` of one class already says what it reads, and a model file has no tag. |

**Guards:** `tests/reporting/test_result_json.py` (`test_every_result_reads_back_equal`
takes every subclass of `ResultModel` on the public surface, so a new result
fails until `EXAMPLES_OF_RESULTS` of `tests/support/results.py` has one;
`test_a_file_of_another_result_is_refused`),
`tests/uncertainty/test_results_equality.py::test_a_result_saved_to_json_loads_back_equal_with_a_nan`.
