#!/usr/bin/env bash
# Start a proposal or a change, in one step, so that taking an issue always
# marks it:
#
#   1. links each open issue to the release before its milestone, and each
#      open release to the issues of its milestone (parallel.py --releases),
#      so that the check below sees those links; a failure there stops the
#      start;
#      then checks that the issue is open, blocked by no open issue and taken
#      by nobody else;
#   2. assigns it to you, which takes it;
#   3. sets it to `In Progress` in the project (and adds it to the project if
#      it is not there);
#   4. creates its branch from `main`, linked to the issue, in a worktree that
#      is a sibling of the repository (or in this checkout with --here);
#   5. writes the precedence level of each open issue into the field Level of
#      the project (parallel.py --levels); a failure there leaves the start
#      done.
#
# Usage, from the root of the repository:
#   .github/scripts/start.sh <number> <type>/<name> [--here] [--dry-run]
# Example:
#   .github/scripts/start.sh 125 refactor/inference-foundation
set -euo pipefail

OWNER="MUNQU"
REPO="MUNQU/commol"
PROJECT=1

usage() {
    echo "usage: $0 <number> <type>/<name> [--here] [--dry-run]" >&2
    exit 2
}

[ $# -ge 2 ] || usage
number="$1"
branch="$2"
shift 2
here=false
dry=false
for option in "$@"; do
    case "$option" in
        --here) here=true ;;
        --dry-run) dry=true ;;
        *) usage ;;
    esac
done
[[ "$number" =~ ^[0-9]+$ ]] || usage
[[ "$branch" == */* ]] || usage

run() {
    if $dry; then
        echo "would run: $*"
    else
        "$@"
    fi
}

me=$(gh api user --jq .login)
root=$(git rev-parse --show-toplevel)

# 1. The links of the releases (to the release before each milestone, and to
# the issues of their own), then the issue: its state, blockers, assignees and
# project items.
run uv run --project "$root/py-commol" --locked python \
    "$root/.github/scripts/parallel.py" --releases
read -r state blocked assignees item <<<"$(gh api graphql \
    -F number="$number" -f query='
query($number: Int!) {
  repository(owner: "MUNQU", name: "commol") {
    issue(number: $number) {
      state
      issueDependenciesSummary { blockedBy }
      assignees(first: 10) { nodes { login } }
      projectItems(first: 10) { nodes { id project { number } } }
    }
  }
}' --jq ".data.repository.issue | [
    .state,
    .issueDependenciesSummary.blockedBy,
    ([.assignees.nodes[].login] | join(\",\") | if . == \"\" then \"-\" else . end),
    ([.projectItems.nodes[] | select(.project.number == $PROJECT) | .id][0] // \"-\")
  ] | @tsv")"

[ "$state" = "OPEN" ] || { echo "#$number is $state" >&2; exit 1; }
[ "$blocked" = "0" ] || {
    echo "#$number is blocked by $blocked open issue(s); see .github/scripts/next.sh" >&2
    exit 1
}
if [ "$assignees" != "-" ] && [[ ",$assignees," != *",$me,"* ]]; then
    echo "#$number is taken by $assignees" >&2
    exit 1
fi

# 2. Take it.
run gh issue edit "$number" --repo "$REPO" --add-assignee "@me"

# 3. In Progress, with the ids of the project found by name.
read -r project_id field_id option_id <<<"$(gh api graphql -f query='
query {
  organization(login: "MUNQU") {
    projectV2(number: 1) {
      id
      field(name: "Status") {
        ... on ProjectV2SingleSelectField { id options { id name } }
      }
    }
  }
}' --jq '.data.organization.projectV2 | [
    .id, .field.id, (.field.options[] | select(.name == "In Progress") | .id)
  ] | @tsv')"
if [ "$item" = "-" ]; then
    if $dry; then
        echo "would run: gh project item-add $PROJECT --owner $OWNER --url https://github.com/$REPO/issues/$number"
        item="<new item>"
    else
        item=$(gh project item-add "$PROJECT" --owner "$OWNER" \
            --url "https://github.com/$REPO/issues/$number" --format json --jq .id)
    fi
fi
run gh project item-edit --project-id "$project_id" --id "$item" \
    --field-id "$field_id" --single-select-option-id "$option_id"

# 4. The branch, linked to the issue.
if $here; then
    run gh issue develop "$number" --repo "$REPO" --base main --name "$branch" --checkout
    where="this checkout"
else
    worktree="$(dirname "$root")/$(basename "$root")-worktrees/${branch#*/}"
    run gh issue develop "$number" --repo "$REPO" --base main --name "$branch" \
        --checkout --worktree "$worktree"
    where="$worktree"
fi

# 5. The precedence levels.
run uv run --project "$root/py-commol" --locked python \
    "$root/.github/scripts/parallel.py" --levels ||
    echo "The levels were not written: run .github/scripts/parallel.py --levels." >&2

if $dry; then
    echo "Dry run: nothing changed."
else
    echo "#$number: assigned to $me, In Progress, branch $branch in $where."
fi
if command -v codegraph >/dev/null 2>&1 && ! $here; then
    echo "The worktree needs its own CodeGraph index: codegraph init -y $worktree"
fi
