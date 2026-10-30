# Settings

[Back to Design decisions](../design-decisions.md)

Where the rules of a calibration problem and of an uncertainty method live,
and who refuses a value that breaks one.

## The settings hold every rule of a problem and of a method

The settings the engine reads from JSON hold every rule that joins the
fields of a calibration problem or of an uncertainty method, and refuse a
value that breaks one: `ProblemSettings::check` and
`CalibrationParameter::check` in `crates/commol-calibration/src/`, and the
`check` of each type in `crates/commol-inference/src/settings/`, through
the `check` arm of `checked!`. A rule that reads an engine handle, such as
the sample of an input space, is checked by the method of the settings that
receives it (`InputSpaceSettings::inputs`,
`SampleIdentifiabilitySettings::reference_priors`,
`SplitConformalSettings::check_predictor`). A binding converts types and
calls them; it holds no default and no fallback. A missing required field,
such as `priors` or `eta`, gets a message that names the field and the fix,
not the "missing field" of serde.

Python holds no copy. A data object whose fields join in a rule lists
`EngineChecked` (`_data/engine_json.py`) among its bases, so the user gets
the message of the engine at construction, and a problem or a method written
as JSON meets the same rules. The check reads the JSON and writes nothing
back (`commol_rs.json.check`), with the reader of the class that lists
`EngineChecked`, so a subclass keeps it. An observation keeps one default in
Python, its label, the name of its one output. The labels of the quantities of interest, the
normalized weights of a propagation, the standard deviations of a sample and
the parameters a scenario may set come from the engine too.

| Alternative | Why not |
|---|---|
| The binding refuses | A binding converts types ([Repository map](../repository-map.md)); a caller of the settings in Rust would skip the rule. |
| Python keeps its copy, with its own message | Two copies drift: Python refused a parameter of zero width, Rust accepted it; a JSON problem met only the Rust rules. |
| A fallback when a choice is missing (the bounds for a missing prior, a uniform input space) | A result the user did not ask for, with no error: the choice is a modelling decision. |
| Clamp an `initial_guess` outside the bounds | It silently changes a value the user gave. |

**Guards:** `crates/commol-calibration/tests/test_json.rs`,
`crates/commol-calibration/tests/test_constraints.rs::an_empty_at_is_refused`,
`crates/commol-inference/tests/test_quantities.rs::a_repeated_label_is_refused`,
`tests/uncertainty/test_identifiability.py::test_the_reference_matches_the_priors`,
`tests/uncertainty/test_sensitivity.py::test_the_input_space_is_given_once`,
`tests/uncertainty/test_smc.py::test_the_priors_are_required`.
