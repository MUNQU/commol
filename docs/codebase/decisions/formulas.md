# Formulas

[Back to Design decisions](../design-decisions.md)

How a formula is checked before it reaches the engine, which limits it
has, and how Python reads it.

## The parser of the engine is the only check of a formula

Rate expressions (`context/transition.py`), formula parameters
(`context/parameter.py`), axis patterns (`AxisPattern.combine` and
`AxisPattern.from_formula`, `api/patterns.py`) and calibration constraints
(`context/calibration.py`) call the check of the engine
(`commol_rs.core.MathExpression(formula).check()`) when their object is
built. A formula that the parser rejects raises `ValueError` with the message
of the parser. Python has no grammar, no limit and no tokenizer of a formula
of its own, and no setting changes the limits of the parser.

**Rule:** a new field that holds a formula calls the check of the engine in
its validator. No module of commol reads the text of a formula: a module that
needs its structure reads the tree of the engine
([Python reads a formula only as the tree of the engine](#python-reads-a-formula-only-as-the-tree-of-the-engine)).

| Alternative | Why not |
|---|---|
| Text guards in Python in front of the parser | A second grammar, which drifts from the parser. Python never evaluates a formula, so it guards nothing that the parser does not, and a limit that a user can raise is no guard. |
| Limits that a setting changes | A feature for limits that guard nothing. |

**Guard:** `py-commol/tests/rates/test_formula_rules.py::test_a_formula_gets_the_message_of_the_engine_parser`,
`py-commol/tests/suite/test_engine_rules.py::test_only_the_engine_reads_a_formula`.

## Python reads a formula only as the tree of the engine

The check of units and the description need the structure of a formula. They
read the tree that the parser of the engine reads, never the text:
`MathExpression::tree_json` (`crates/commol-core/src/math_expression/tree.rs`)
checks the formula and gives its tree (`Expr`, which derives `Serialize`) as
JSON, the binding `commol_rs.core.MathExpression(formula).tree()` passes it to
Python, and `engine_state.formula_tree` reads it into plain data
(`context/_model/formula_tree.py`: one frozen dataclass per kind of node).
`tree.rs` documents the shape: one object per node, keyed by its kind
(`Constant`, `Variable`, `Chain` with its operators, `Power`, `UnaryOp`,
`FunctionCall`, `Conditional`); a number keeps the value the parser read, and
a value that is not finite is its text (`"Infinity"`), as in the JSON of the
port ([A data object crosses as the JSON of its dump](python-models.md#a-data-object-crosses-as-the-json-of-its-dump)).

- The check of units (`formula_unit` in `utils/equations.py`) walks the tree
  with a `pint` quantity for each variable: `pint` does the algebra of units,
  in the grouping of the parser, and a written number takes its unit as
  [One step unit in the unit check](time.md#one-step-unit-in-the-unit-check)
  says.
- The description (`context/_model/equations.py`) writes each rate and each
  flow from the tree: `FormulaWriter` writes the text, with one space around
  each operator and only the parentheses the tree needs, so the text reads
  back as the same tree; `_LatexWriter` writes LaTeX. The flow of a
  proportional rate is the tree of the rate times its source
  (`TransitionFlow.term`, `product`), which groups a sum as a whole, and a
  rate gives the same description with or without spaces.

**Rule:** a module that needs the structure of a formula reads
`engine_state.formula_tree`; a new node kind or operator of the parser is a
case of the JSON of `tree.rs`, of `read_tree` and of the walks of
`formula_tree.py`.

| Alternative | Why not |
|---|---|
| A PyO3 class per node | A third shape of the tree (Rust, PyO3, Python) to keep in step, against one JSON port, as the model is. |
| A LaTeX writer in Rust | Presentation in the core, with no Rust caller: the engine gives the structure, and the edge draws it, as `api/plotting/panels.py` and `figures.py` do. |
| The text, read by the grammar of Python (`ast.parse` and `eval`) or split at `" * "` and `" / "` | A second grammar: Python reads `^`, `!` and names such as `lambda` in another way or not at all, and a split of the text depends on its spaces. |

**Guard:** `crates/commol-core/src/math_expression/tree.rs::the_tree_keeps_the_grouping_of_the_parser`,
`py-commol/tests/rates/test_formula_rules.py::test_python_reads_the_tree_of_a_formula_from_the_engine`,
`py-commol/tests/units/test_expression_units.py::test_the_unit_follows_the_grouping_of_the_parser`,
`py-commol/tests/description/test_equation_text.py::test_the_spaces_of_a_rate_do_not_change_its_description`,
`py-commol/tests/description/test_equation_text.py::test_a_rate_is_written_with_the_parentheses_its_tree_needs`,
`py-commol/tests/suite/test_engine_rules.py::test_only_the_engine_reads_a_formula`.

## The engine defines the language of formulas

One Rust check decides whether an expression is in the language of formulas:
`MathExpression::check` (`crates/commol-core/src/math_expression/mod.rs`,
`language.rs`) reads the tree that the JIT compiles, so a formula that passes
it compiles. It rejects a function that is not in `FUNCTIONS`, a call with the
wrong number of arguments, `&&` and `||`, and a comparison or `!` outside the
condition of an `if`, and it counts the nodes of the tree, which `MAX_NODES`
limits. The JIT compiles the functions of `FUNCTIONS` and no other, and the
unit check (`_FUNCTION_UNITS` in `utils/equations.py`) has one rule for each
of them.

The engine runs the same check again when it builds a model (`check_model` in
`crates/commol-model/src/builder.rs`) and a calibration problem
(`Feasibility::new`), and `check_model` also rejects a variable that the model
does not define. Python keeps no list of functions, no parser of the
variables and no copy of the engine variables (`commol/constants.py` reads
them from `commol_rs.model`).

| Alternative | Why not |
|---|---|
| The language checked in Python | A second parser and a second list of functions, which drift from the JIT: a function that one accepts and the other does not compile. |
| An evaluation by an expression library with dummy values | It evaluates another language than the JIT compiles: it rejects functions the JIT compiles and accepts a comparison the guide rejects. |

**Guard:** `crates/commol-core/tests/expressions.rs` (the language and the
value of each function), `py-commol/tests/units/test_expression_units.py::test_the_unit_check_knows_every_function_of_the_engine`,
`py-commol/tests/rates/test_variables.py::test_an_undefined_variable_is_rejected_by_the_engine`.

## The engine parses formulas with a parser of its own

`parse_expression` (`crates/commol-core/src/math_expression/jit/parser.rs`)
reads the text of a formula into its tree (`Expr`, `jit/ast.rs`) by
precedence climbing; no other code reads the text of a formula. The grammar sets the
precedence and the grouping: the operators of one level apply from the
left, and a chain of powers groups from the right, as in Python.

A chain of operators of one level is one node (`Expr::Chain`), so a sum of
any number of terms is one level of the tree, and the parser reads the
levels of one nesting with a stack, not with recursion. The parser counts how
deep the formula nests (parentheses, calls, signs and powers) and rejects a
formula deeper than `MAX_NESTING` (100) before it reads the next level. A
tree that the parser returns is at most a few nodes deep for each level of
nesting, so every recursive walk of it (the check, the variables, the JIT,
`Display`, clone and drop) fits the 2 MiB stack of a worker thread of the
parallel work in a debug build, with a margin of more than two.

**Rule:** a walk of a tree recurses only into the operands of a node, and a
change to the grammar keeps `MAX_NESTING` safe at the stack of a worker
thread.

| Alternative | Why not |
|---|---|
| A depth check after the parse of an expression library | The language stays defined by the grammar of the library, which has its own recursion: a chain of signs exhausts the stack inside its parser, before any check. |
| The parse on a thread with a large stack | It hides the limit instead of stating it, and a large enough formula still exhausts the stack. |

**Guard:** `crates/commol-core/tests/expressions.rs::a_formula_deeper_than_the_limit_is_rejected`,
`crates/commol-core/tests/expressions.rs::every_level_of_operators_at_the_limit_reads_on_a_worker_thread`,
`crates/commol-core/tests/expressions.rs::a_chain_of_terms_is_one_level_and_adds_from_the_left`,
`crates/commol-core/tests/expressions.rs::a_chain_of_powers_groups_from_the_right`,
`crates/commol-model/tests/rates.rs::a_rate_at_the_nesting_limit_builds_and_runs_on_a_worker_thread`,
`py-commol/tests/rates/test_formula_rules.py::test_a_formula_gets_the_message_of_the_engine_parser`.

## A formula has at most MAX_NODES nodes, and a message cuts it

The check of the language rejects a tree of more than `MAX_NODES` (100 000)
nodes (`jit/parser.rs`, `language.rs`) before the JIT compiles the formula.
A sum of n terms has 2n - 1 nodes, so the limit allows a sum of 50 000
terms, and stays far below the size that the JIT (Cranelift) rejects: about
600 000 terms, after about 1 s and 450 MiB. A message of the parser or of the
JIT quotes the formula cut after its first 80 characters (`quoted`,
`math_expression/error.rs`), so it never repeats a long text.

| Alternative | Why not |
|---|---|
| No limit of nodes | A huge formula costs about 1 s and 450 MiB before the JIT rejects it, with a message of several MB. |
| The limit of the JIT | A limit of Cranelift, which commol does not own, and too slow to test. |

**Guard:** `crates/commol-core/src/math_expression/language.rs::a_tree_of_the_most_nodes_is_in_the_language`,
`crates/commol-core/src/math_expression/language.rs::a_tree_of_more_nodes_than_the_limit_is_rejected`,
`crates/commol-core/tests/expressions.rs::a_formula_of_more_nodes_than_the_limit_is_rejected_before_it_compiles`,
`crates/commol-core/src/math_expression/error.rs::a_long_formula_is_cut_after_its_first_characters`,
`crates/commol-core/src/math_expression/mod.rs::a_formula_that_does_not_compile_is_quoted_cut`,
`py-commol/tests/rates/test_formula_rules.py::test_a_formula_gets_the_message_of_the_engine_parser`.
