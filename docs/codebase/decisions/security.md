# Security

[Back to Design decisions](../design-decisions.md)

How user expressions are checked before they reach the engine.

## Every user expression is validated

Rate expressions (`context/transition.py`), formula parameters
(`context/parameter.py`), axis patterns (`api/patterns.py`) and calibration
constraints (`context/calibration.py`) pass through
`validate_expression_security` (`utils/security.py`) before they reach the
engine. The limits are in
`SecurityConfig`.

**Rule:** any new field that accepts an expression must call it.

**Guard:** `tests/rates/test_security.py` (the limits and the rejected
expressions); `tests/axis_patterns/test_patterns.py` (a pattern calls it).

## The engine defines the language of formulas

`validate_expression_security` runs two kinds of check. The text guards (the
length, the encoding, the dangerous patterns, the nesting depth, the function
calls, the numeric literals and the name length) read the text in Python.
Then one Rust check decides whether the expression is in the language of
formulas: `MathExpression::check` (`crates/commol-core/src/math_expression/mod.rs`,
`language.rs`) reads the tree that the JIT compiles, so a formula that passes
it compiles. It rejects a function that is not in `FUNCTIONS`, a call with the
wrong number of arguments, `&&` and `||`, and a comparison or `!` outside the
condition of an `if`, and it counts the nodes of the tree, which
`SecurityConfig.max_complexity_score` limits. The JIT compiles the functions
of `FUNCTIONS` and no other, and the unit check (`_FUNCTION_UNITS` in
`utils/equations.py`) has one rule for each of them.

The engine runs the same check again when it builds a model (`check_model` in
`crates/commol-model/src/builder.rs`) and a calibration problem
(`Feasibility::new`), and `check_model` also rejects a variable that the model
does not define. Python keeps no list of functions, no parser of the
variables and no copy of the engine variables (`commol/constants.py` reads
them from `commol_rs.model`).

| Alternative | Why not |
|---|---|
| The language checked in Python | A second parser and a second list of functions, which drift from the JIT: a function that one accepts and the other does not compile. |
| `evalexpr` evaluation with dummy values | It evaluates another language than the JIT compiles: it rejects functions the JIT compiles and accepts a comparison the guide rejects. |
| The complexity score counted on the text | One more parser of the language in Python. |

**Guard:** `crates/commol-core/tests/expressions.rs` (the language and the
value of each function), `py-commol/tests/units/test_expression_units.py::test_the_unit_check_knows_every_function_of_the_engine`,
`py-commol/tests/rates/test_security.py::test_a_formula_above_the_complexity_score_is_rejected`,
`py-commol/tests/rates/test_variables.py::test_an_undefined_variable_is_rejected_by_the_engine`.
