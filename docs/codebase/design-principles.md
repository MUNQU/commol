# Design Principles

[Back to Codebase Guide](index.md)

Every part of the code follows the best architecture pattern for its job:
each crate, module, type, function, test, script and figure. "It works" is
not enough. The bar is: it is in the right place, it has one reason to
change, and the next change is easy. When two designs work, take the one
with the better structure, also when it costs more work.

## Structure

1. **The structure says what the code does.** Name crates, folders and
   modules after their responsibility, not after a technical type
   (`crates/commol-model/src/flows.rs`, not a module `utils`). A new reader
   finds a rule by its name.
2. **Dependencies point inward.** The core holds the decisions: the crates
   of `crates/`, with no Python and no input or output
   ([the golden rule](mental-model.md#the-golden-rule)). The edges hold the
   adapters: the PyO3 bridge (`py-commol/src/`), the Python API
   (`py-commol/commol/`), the figures and the files a user saves. The core
   never imports an edge.
3. **Each boundary is a port.** Two layers meet through one small typed
   interface: each data object as the JSON of its dump, the stub of the extension
   (`commol_rs/_commol_rs.pyi`), a trait (`SimulationEngine`), a protocol
   (`PredictiveSource`). An adapter implements it. Change an adapter without
   a change to the core.
4. **Decide in the core, act at the edge.** A function that decides does no
   input or output: a crate returns the information to its caller and never
   prints. A function that does input or output decides nothing:
   `plotting/panels.py` gives the data of each panel, and
   `plotting/figures.py` draws it. A test of a decision needs no fake.

## Each unit

5. **SOLID.** One reason to change per unit (S). Add a behavior with a new
   entry in data or a new implementation, not with one more branch in an
   `if` or `match` chain (O). An implementation keeps the contract of its
   interface (L). An interface has only what its callers use (I). Depend on
   the interface, not on the implementation (D).
6. **DRY.** One source of truth for each rule, text, value and style. A
   second copy is a defect. Do not merge two things that change for
   different reasons: that is a wrong abstraction, not DRY.
7. **KISS.** Use the simplest design that keeps the structure above.
8. **YAGNI.** No key, option, layer or parameter without a caller today.

## Interface

The interface of commol is what a user touches: the Python API and the
figures.

9. **Shared parts, then the whole.** A style (`PlotConfig`) and the drawing
   of one panel are shared parts. A figure puts the shared parts together.
   It draws no part of its own.
10. **Data, then drawing.** The code that draws takes data and draws it. It
    never runs a model or reduces a result. The code that computes gives
    the data and draws nothing.
11. **One idea, one name.** One idea has one class or function in the
    Python API, and one spelling. Never a second path to the same thing.

## Proof

12. **A rule that a machine can check is a test.** Examples: no crate of
    `crates/` depends on PyO3 or on the bridge
    (`tests/suite/test_architecture.py`), only the engine computes a window
    (`tests/suite/test_engine_rules.py`), and `import commol` loads no
    plotting library (`tests/plotting/test_figures.py`).
13. **Test first, through the port.** A test drives the public interface of
    a unit, not its private parts.
14. **Every decision names its principles.** The column «Why, against each
    alternative» of the decisions of a proposal names each principle that
    it checks: "S: …", "DRY: …".
15. **Leave the code better.** When a change touches code that breaks a
    rule above, fix it in the same pull request if the fix is small. If it
    is not small, open an issue for it. The behavior of `main` stays the
    same in that pull request.

## Navigation

Previous: [Conventions](conventions.md) | Next: [Tests](tests.md)
