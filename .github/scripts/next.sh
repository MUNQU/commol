#!/usr/bin/env bash
# The next proposals to implement, in one GraphQL call and a few lines of
# output: open, blocked by no open issue, assigned to nobody, sorted by the
# field Order of the project (an issue with no Order comes last).
#
# Usage, from anywhere: .github/scripts/next.sh [count]   (default 1)
set -euo pipefail
count="${1:-1}"
gh api graphql -f query='
query {
  organization(login: "MUNQU") {
    projectV2(number: 1) {
      items(first: 100) {
        nodes {
          order: fieldValueByName(name: "Order") {
            ... on ProjectV2ItemFieldNumberValue { number }
          }
          content {
            ... on Issue {
              number title state
              labels(first: 20) { nodes { name } }
              assignees(first: 1) { totalCount }
              issueDependenciesSummary { blockedBy }
            }
          }
        }
      }
    }
  }
}' --jq "[.data.organization.projectV2.items.nodes[]
  | select(.content.state == \"OPEN\"
      and .content.issueDependenciesSummary.blockedBy == 0
      and .content.assignees.totalCount == 0
      and ([.content.labels.nodes[].name] | index(\"proposal\")))
  | {order: (.order.number // 1e9), number: .content.number, title: .content.title}]
  | sort_by(.order) | .[:${count}][] | \"#\(.number) \(.title)\""
