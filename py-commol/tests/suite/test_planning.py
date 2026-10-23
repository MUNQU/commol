"""Check ``.github/scripts/parallel.py``: what a change touches, the open
issues and blockers it loads, the next issues to take, and what each option
runs. No test calls GitHub: each one patches ``rest``. The rules are in
``docs/development/workflow.md``, "Planning"."""

import re
import sys
from unittest.mock import patch

import pytest

from tests.support.planning import an_issue, parallel

CHANGE = """## Behavior

- **Now:** it fails.

## Where

- `crates/commol-model/src/system.rs` (`derivatives`), `docs/guide/stochastic.md`
- `lib.rs`, the command `cargo test`

## Done when

- [ ] It works
"""


def numbers(issues) -> list[int]:
    return [issue.number for issue in issues]


# What a change touches


def test_the_paths_of_where_are_what_a_change_touches() -> None:
    assert parallel.where(CHANGE) == [
        "crates/commol-model/src/system.rs",
        "docs/guide/stochastic.md",
        "lib.rs",
    ]


def test_a_body_without_where_touches_nothing() -> None:
    assert parallel.where("## Goal\n\nSomething.\n") == []


# The next issues


def test_the_next_issues_come_by_the_version_of_their_milestone() -> None:
    # Milestone 3 is 0.15.0 and milestone 16 is 0.10.1: the numbers of the
    # milestones do not follow their versions.
    issues = {
        198: an_issue(198, changes=("a.py",), milestone=("", 3), version="0.15.0"),
        201: an_issue(201, changes=("b.rs",), milestone=("", 16), version="0.10.1"),
        197: an_issue(
            197,
            changes=("c.py",),
            milestone=("", 16),
            version="0.10.1",
            kind="proposal",
        ),
    }

    chosen, skipped = parallel.next_issues(issues, {}, 3)

    assert numbers(chosen) == [197, 201, 198]
    assert skipped == []


def test_the_free_issues_come_by_priority_then_version_then_due_date() -> None:
    issues = {
        1: an_issue(1, changes=("src/1.py",)),
        2: an_issue(2, changes=("src/2.py",)),
        3: an_issue(3, changes=("src/3.py",)),
        4: an_issue(
            4,
            changes=("src/4.py",),
            milestone=("2026-11-01T00:00:00Z", 1),
            version="0.11.0",
        ),
        5: an_issue(5, changes=("src/5.py",), milestone=("", 1), version="0.11.0"),
        6: an_issue(6, changes=("src/6.py",), milestone=("", 2), version="0.10.1"),
        7: an_issue(7, changes=("src/7.py",), milestone=("", 3), version="Backlog"),
    }
    ranks = {1: ("Later", ""), 2: ("Now", "")}

    chosen, skipped = parallel.next_issues(issues, ranks, 7)

    # A milestone that names no version comes after every version, and before
    # no milestone.
    assert numbers(chosen) == [2, 6, 4, 5, 7, 3, 1]
    assert skipped == []


def test_an_issue_without_priority_counts_as_next() -> None:
    issues = {n: an_issue(n, changes=(f"src/{n}.py",)) for n in (1, 2, 3)}
    ranks = {2: ("Later", ""), 3: ("Now", "")}

    chosen, _ = parallel.next_issues(issues, ranks, 3)

    assert numbers(chosen) == [3, 1, 2]


def test_a_release_is_never_next() -> None:
    issues = {1: an_issue(1, kind="release"), 2: an_issue(2, kind="proposal")}

    chosen, _ = parallel.next_issues(issues, {}, 2)

    assert numbers(chosen) == [2]


def test_a_held_or_blocked_issue_is_never_next() -> None:
    issues = {
        1: an_issue(1, held=True),
        2: an_issue(2, blocked_by=frozenset({1})),
        3: an_issue(3),
    }

    chosen, _ = parallel.next_issues(issues, {}, 3)

    assert numbers(chosen) == [3]


def test_an_issue_that_touches_one_in_progress_is_skipped() -> None:
    issues = {
        1: an_issue(1, changes=("src/models.py",), held=True),
        2: an_issue(2, changes=("src/models.py::Model",)),
        3: an_issue(3, changes=("src/views.py",)),
    }

    chosen, skipped = parallel.next_issues(issues, {}, 1)

    assert numbers(chosen) == [3]
    assert [(issue.number, other.number) for issue, other, _ in skipped] == [(2, 1)]


def test_the_issues_it_returns_can_run_together() -> None:
    issues = {
        1: an_issue(1, changes=("src/models.py",)),
        2: an_issue(2, adds=("src/models.py::Other",)),
        3: an_issue(3, changes=("src/views.py",)),
    }

    chosen, skipped = parallel.next_issues(issues, {}, 2)

    assert numbers(chosen) == [1, 3]
    assert [(issue.number, other.number) for issue, other, _ in skipped] == [(2, 1)]


# The effective priority


def priority_of(key: str, own: dict[str, str], links: dict[str, list[str]]):
    return parallel.effective_priority(key, own.get, lambda k: links.get(k, []))


def test_an_issue_inherits_the_priority_of_the_issue_it_blocks() -> None:
    links = {"o/engine#1": ["o/product#9"]}

    assert priority_of("o/engine#1", {"o/product#9": "Now"}, links) == (
        "Now",
        "o/product#9",
    )


def test_the_priority_comes_through_a_chain_of_links() -> None:
    own = {"o/engine#1": "Later", "o/product#9": "Now"}
    links = {"o/engine#1": ["o/engine#2"], "o/engine#2": ["o/product#9"]}

    assert priority_of("o/engine#1", own, links) == ("Now", "o/product#9")


def test_a_higher_priority_of_its_own_stays() -> None:
    own = {"o/engine#1": "Now", "o/product#9": "Later"}

    assert priority_of("o/engine#1", own, {"o/engine#1": ["o/product#9"]}) == (
        "Now",
        "",
    )


def test_an_issue_that_is_now_follows_no_link() -> None:
    def blocks(key: str) -> list[str]:
        raise AssertionError(f"no link of {key} is read")

    assert parallel.effective_priority("o/a#1", {"o/a#1": "Now"}.get, blocks) == (
        "Now",
        "",
    )


def test_a_cycle_of_links_ends() -> None:
    links = {"o/a#1": ["o/a#2"], "o/a#2": ["o/a#1"]}

    assert priority_of("o/a#1", {}, links) == ("Next", "")


def test_the_priorities_of_a_repository_are_read_from_the_items() -> None:
    def item(number: int, repository: str, priority: str | None) -> dict:
        value = {"name": {"raw": priority}} if priority is not None else None
        return {
            "content_type": "Issue",
            "content": {
                "number": number,
                "repository_url": f"https://api.github.com/repos/{repository}",
            },
            "fields": [{"name": "Priority", "value": value}],
        }

    items = [
        item(1, "o/a", "Now"),
        item(2, "o/a", None),
        item(3, "o/b", "Later"),
        {"content_type": "DraftIssue", "content": {}, "fields": []},
    ]

    assert parallel.priorities_of(items, "o/a") == {1: "Now"}


def run_main(*arguments: str) -> dict[str, list[tuple]]:
    """Run ``parallel.main`` with ``arguments``; the writes of the levels fail.

    Returns the calls of ``write_release_links``, ``show_next`` and
    ``show_parallel``; the first is left out when it has none.
    """
    calls: dict[str, list[tuple]] = {"next": [], "parallel": []}
    failure = SystemExit("GitHub: PATCH orgs/o/projectsV2/1/items/10: HTTP 403")

    def links(*a) -> None:
        calls.setdefault("releases", []).append(a)

    with (
        patch.object(sys, "argv", ["parallel.py", *arguments]),
        patch.object(parallel, "load", return_value={}),
        patch.object(parallel, "write_release_links", links),
        patch.object(parallel, "write_levels", side_effect=failure),
        patch.object(parallel, "show_next", lambda *a: calls["next"].append(a)),
        patch.object(parallel, "show_parallel", lambda *a: calls["parallel"].append(a)),
    ):
        parallel.main()
    return calls


def test_a_failed_level_write_still_lists_the_next_issues(capsys) -> None:
    calls = run_main("--levels", "--next", "2")

    assert calls["next"] == [({}, 2)]
    assert "Level: not written: GitHub: PATCH" in capsys.readouterr().err


def test_no_option_prints_what_can_run_in_parallel() -> None:
    assert run_main() == {"next": [], "parallel": [({},)]}


def test_levels_alone_print_nothing_else() -> None:
    assert run_main("--levels") == {"next": [], "parallel": []}


def test_releases_alone_write_only_the_links() -> None:
    assert run_main("--releases") == {"releases": [({},)], "next": [], "parallel": []}


def test_the_links_come_before_the_next_issues() -> None:
    assert run_main("--releases", "--next", "1") == {
        "releases": [({},)],
        "next": [({}, 1)],
        "parallel": [],
    }


def test_the_milestone_of_an_issue_is_its_due_date_and_number() -> None:
    due = {"milestone": {"number": 2, "due_on": "2026-11-01T00:00:00Z"}}

    assert parallel.milestone_of(due) == ("2026-11-01T00:00:00Z", 2)
    assert parallel.milestone_of({"milestone": {"number": 4}}) == ("", 4)
    assert parallel.milestone_of({"milestone": None}) is None


def test_the_version_of_an_issue_is_the_title_of_its_milestone() -> None:
    assert parallel.milestone_version({"milestone": {"title": "0.9.1"}}) == (0, 9, 1)
    assert parallel.milestone_version({"milestone": {"number": 4}}) is None
    assert parallel.milestone_version({"milestone": None}) is None


# The issues as GitHub gives them


def an_item(number: int, *labels: str, body: str = "", version: str = "") -> dict:
    """An open issue of the repository ``o/a`` as the REST API gives it."""
    return {
        "number": number,
        "id": 10 * number,
        "title": f"issue {number}",
        "body": body,
        "state": "open",
        "labels": [{"name": label} for label in labels],
        "assignees": [],
        "milestone": {"number": number, "title": version} if version else None,
        "repository_url": "https://api.github.com/repos/o/a",
    }


def github_of(items: list[dict], blockers: dict[int, list[dict]]):
    """A fake ``rest`` of the repository ``o/a``, and the list of its calls.

    It gives the open issues ``items`` and the open blockers of each issue;
    an issue blocks none, and the organization has no project. A write
    returns an object with an id.
    """
    calls: list[tuple[str, str, object]] = []

    def rest(path: str, *, method: str = "GET", data=None, **_: object):
        calls.append((path, method, data))
        if method != "GET":
            return {"id": 1}
        route = path.split("?")[0]
        if route == "repos/o/a/issues":
            return items
        found = re.fullmatch(r"repos/o/a/issues/(\d+)/dependencies/blocked_by", route)
        return blockers.get(int(found.group(1)), []) if found else []

    return rest, calls


def test_an_open_blocker_without_a_kind_label_blocks_an_issue() -> None:
    # 2 is open and has no label proposal, change or release.
    rest, _ = github_of([an_item(1, "proposal"), an_item(2)], {1: [an_item(2)]})

    with (
        patch.object(parallel, "repository", return_value="o/a"),
        patch.object(parallel, "rest", side_effect=rest),
    ):
        issues = parallel.load()

    chosen, _ = parallel.next_issues(issues, {}, 1)

    assert numbers(chosen) == []
    assert parallel.levels(issues) == {1: 1}


def options_of_parallel(capsys) -> list[list[str]]:
    """The options of ``parallel.py`` that its help names, ``--help`` left out,
    each with a value when it takes one."""
    with patch.object(sys, "argv", ["parallel.py", "--help"]):
        with pytest.raises(SystemExit):
            parallel.main()
    found = re.findall(r"^\s+(--[\w-]+)( [A-Z]+)?", capsys.readouterr().out, re.M)
    return [
        [option, "2"] if value else [option]
        for option, value in found
        if option != "--help"
    ]


def test_no_option_edits_an_issue_body(capsys) -> None:
    touches = "## Touches\n\n- Changes: `{}`\n\n## Delivery\n\n- Parallel with: none\n"
    items = [
        an_item(1, "proposal", body=touches.format("a.py"), version="0.10.1"),
        an_item(2, "proposal", body=touches.format("b.py"), version="0.10.1"),
        an_item(3, "release", version="0.10.0"),
    ]
    options = options_of_parallel(capsys)
    assert options
    runs = [[], *options, [part for option in options for part in option]]

    for arguments in runs:
        rest, calls = github_of(items, {})
        parallel.blocking.cache_clear()
        parallel.project_priorities.cache_clear()
        with (
            patch.object(sys, "argv", ["parallel.py", *arguments]),
            patch.object(parallel, "repository", return_value="o/a"),
            patch.object(parallel, "rest", side_effect=rest),
        ):
            parallel.main()

        edits = [
            (path, method)
            for path, method, data in calls
            if (
                re.fullmatch(r"repos/o/a/issues/\d+", path.split("?")[0])
                and method != "GET"
            )
            or (isinstance(data, dict) and "body" in data)
        ]
        assert edits == [], arguments
