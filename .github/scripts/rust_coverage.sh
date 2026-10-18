#!/usr/bin/env bash
# The Rust coverage, as CI measures it. The Rust tests and the Python tests
# run on code built with the instrumentation of cargo llvm-cov, and one report
# joins them: the bindings (py-commol/src/) have no Rust tests, and pytest is
# what runs them.
#
#   1. builds and runs the Rust tests, instrumented;
#   2. builds the extension, instrumented, and runs pytest on it;
#   3. writes the report, with the options given to the script;
#   4. outside CI, builds the extension again without instrumentation, so the
#      virtual environment is left as it was.
#
# Everything builds in the debug profile, so `debug_assert!` runs and every
# line keeps its own count, in its own target directory
# (target/llvm-cov-target), so the usual builds stay cached.
#
# Usage, from the root of the repository, with the virtual environment of
# py-commol active:
#   .github/scripts/rust_coverage.sh [options of cargo llvm-cov report]
# Example:
#   .github/scripts/rust_coverage.sh --lcov --output-path lcov.info
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

export CARGO_TARGET_DIR="$root/target/llvm-cov-target"
eval "$(cargo llvm-cov show-env --sh)"
cargo llvm-cov clean --workspace

cargo test --all-features --workspace
(
    cd py-commol
    maturin develop --features test-kind
    pytest -q
)

# The test model kind (py-commol/src/test_kind.rs) is code of the tests, built
# only with the feature test-kind: docs/codebase/tests.md, "Test coverage".
cargo llvm-cov report --ignore-filename-regex 'py-commol/src/test_kind\.rs' "$@"

if [ -z "${CI:-}" ]; then
    unset CARGO_TARGET_DIR RUSTC_WRAPPER LLVM_PROFILE_FILE
    (cd py-commol && maturin develop --release --features test-kind)
fi
