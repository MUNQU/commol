#!/usr/bin/env bash
# The Rust coverage, as CI measures it. The Rust tests and the Python tests
# run on code built with the instrumentation of cargo llvm-cov, and one report
# joins them: the bindings (py-commol/src/) have no Rust tests, and pytest is
# what runs them.
#
#   1. builds and runs the Rust tests, instrumented;
#   2. builds the extension, instrumented, and runs pytest on it;
#   3. writes the report, with the options given to the script;
#   4. outside CI, builds the extension again without instrumentation when it
#      ends, so the virtual environment is left as it was.
#
# Everything builds in the debug profile, so `debug_assert!` runs and every
# line keeps its own count, in its own target directory
# (target/llvm-cov-target), so the usual builds stay cached.
#
# The script fails when the line coverage falls under RUST_LINES, the coverage
# reached; raise it when coverage rises, and never lower it without a reason
# in the pull request (docs/codebase/coverage.md).
#
# Usage, from the root of the repository, with the virtual environment of
# py-commol active:
#   .github/scripts/rust_coverage.sh [options of cargo llvm-cov report]
# Example:
#   .github/scripts/rust_coverage.sh --lcov --output-path lcov.info
set -euo pipefail

RUST_LINES=0

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
    pytest -q
)

# The test model kind (py-commol/src/test_kind.rs) is code of the tests, built
# only with the feature test-kind: docs/codebase/coverage.md.
cargo llvm-cov report --ignore-filename-regex 'py-commol/src/test_kind\.rs' \
    --fail-under-lines "$RUST_LINES" "$@"
