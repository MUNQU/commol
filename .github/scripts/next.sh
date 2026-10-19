#!/usr/bin/env bash
# The next issues to implement: the proposals and changes that are open,
# blocked by no open issue and assigned to nobody, that can run at the same
# time as every issue in progress and as each other. They come by effective
# priority (the field Priority, Now, Next or Later, raised by the open issues
# each one blocks in any repository; no value is Next), then milestone, then
# number. It prints the priority and where an inherited one comes from, and
# names each issue it skips and the issue it would touch
# (.github/scripts/parallel.py). First it writes the precedence level of each
# open issue into the field Level of the project (--levels).
#
# Usage, from anywhere in the repository: .github/scripts/next.sh [count]   (default 1)
set -euo pipefail
root=$(git rev-parse --show-toplevel)
exec uv run --project "$root/py-commol" --locked python \
    "$root/.github/scripts/parallel.py" --levels --next "${1:-1}"
