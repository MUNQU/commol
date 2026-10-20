#!/usr/bin/env bash
# The Rust coverage, as CI measures it. The Rust tests and the Python tests
# run on code built with the instrumentation of cargo llvm-cov, and one report
# joins them: the bindings (py-commol/src/) have no Rust tests, and pytest is
# what runs them.
#
#   1. builds and runs the Rust tests, instrumented;
#   2. builds the extension, instrumented, and runs pytest on it;
#   3. writes the LCOV report, lcov.info at the root, which joins every
#      instantiation of a line, and checks it with rust_uncovered.py: a line
#      that no test runs carries a marker `// coverage: <reason>`;
#   4. writes the report of cargo llvm-cov with the options given to the
#      script, if any;
#   5. outside CI, builds the extension again without instrumentation when it
#      ends, so the virtual environment is left as it was.
#
# Everything builds in the debug profile, so `debug_assert!` runs and every
# line keeps its own count, in its own target directory
# (target/llvm-cov-target), so the usual builds stay cached.
#
# The script fails when a line that no test runs has no marker, or when a
# marker excuses no such line (docs/codebase/coverage.md).
#
# Usage, from the root of the repository, with the virtual environment of
# py-commol active:
#   .github/scripts/rust_coverage.sh [options of cargo llvm-cov report]
# Example:
#   .github/scripts/rust_coverage.sh --summary-only
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

# Outside CI, the release extension comes back however the script ends.
if [ -z "${CI:-}" ]; then
    trap 'unset CARGO_TARGET_DIR RUSTC_WRAPPER LLVM_PROFILE_FILE
        (cd py-commol && maturin develop --release --features test-kind)' EXIT
fi

export CARGO_TARGET_DIR="$root/target/llvm-cov-target"
eval "$(cargo llvm-cov show-env --sh)"
cargo llvm-cov clean --workspace

cargo test --all-features --workspace
(
    cd py-commol
    maturin develop --features test-kind
    # A debug extension is slow, so the time budget is left to the job Test
    # of Build, which times the release extension.
    pytest -q --deselect tests/suite/test_rules.py::test_slow_tests_are_listed
)

# The test model kind (py-commol/src/test_kind.rs) is code of the tests, built
# only with the feature test-kind: docs/codebase/coverage.md.
ignored='py-commol/src/test_kind\.rs'
if [ $# -gt 0 ]; then
    cargo llvm-cov report --ignore-filename-regex "$ignored" "$@"
fi
cargo llvm-cov report --ignore-filename-regex "$ignored" --lcov --output-path lcov.info
python .github/scripts/rust_uncovered.py lcov.info
