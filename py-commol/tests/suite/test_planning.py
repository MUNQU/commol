"""Check the planning scripts of ``.github/scripts/``: what a change touches,
the next issues to take, and the REST calls they make to GitHub. The rules are
in ``docs/development/workflow.md``, "Planning"."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str) -> ModuleType:
    """The script ``name`` of ``.github/scripts/``, loaded as a module."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parallel = load("parallel")
github = load("github")

CHANGE = """## Behavior

- **Now:** it fails.

## Where

- `crates/commol-model/src/system.rs` (`derivatives`), `docs/guide/stochastic.md`
- `lib.rs`, the command `cargo test`

## Done when

- [ ] It works
"""


def an_issue(
    number: int,
    *,
    changes: tuple[str, ...] = (),
    adds: tuple[str, ...] = (),
    held: bool = False,
    blocked_by: frozenset[int] = frozenset(),
    milestone: tuple[str, int] | None = None,
    kind: str = "change",
    version: str = "",
):
    """An open issue of the planning script, with what it touches.

    Its id is ten times its number; ``version`` is the title of its milestone.
    """
    issue = parallel.Proposal(number, f"issue {number}", "", kind=kind)
    issue.changes = list(changes)
    issue.adds = list(adds)
    issue.held = held
    issue.blocked_by = set(blocked_by)
    issue.milestone = milestone
    issue.issue_id = 10 * number
    issue.version = parallel.version_of(version)
    return issue


def numbers(issues) -> list[int]:
    return [issue.number for issue in issues]


def response(status: str, body: str, *headers: str) -> subprocess.CompletedProcess:
    """What ``gh api -i`` prints for one answer of GitHub."""
    text = "\n".join([f"HTTP/2.0 {status}", *headers, "", body])
    code = 0 if status.startswith("2") else 1
    return subprocess.CompletedProcess([], code, stdout=text, stderr="")


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


def test_a_change_of_an_older_milestone_comes_first() -> None:
    issues = {
        198: an_issue(198, changes=("a.py",), milestone=("", 2), kind="proposal"),
        201: an_issue(201, changes=("b.rs",), milestone=("", 1)),
        197: an_issue(197, changes=("c.py",), milestone=("", 1)),
    }

    chosen, skipped = parallel.next_issues(issues, {}, 3)

    assert numbers(chosen) == [197, 201, 198]
    assert skipped == []


def test_the_free_issues_come_by_priority_then_milestone_then_number() -> None:
    issues = {
        1: an_issue(1, changes=("src/1.py",)),
        2: an_issue(2, changes=("src/2.py",)),
        3: an_issue(3, changes=("src/3.py",)),
        4: an_issue(4, changes=("src/4.py",), milestone=("2026-11-01T00:00:00Z", 3)),
        5: an_issue(5, changes=("src/5.py",), milestone=("", 2)),
        6: an_issue(6, changes=("src/6.py",), milestone=("", 1)),
    }
    ranks = {1: ("Later", ""), 2: ("Now", "")}

    chosen, skipped = parallel.next_issues(issues, ranks, 6)

    assert numbers(chosen) == [2, 4, 6, 5, 3, 1]
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


# The precedence level


def test_an_issue_is_one_level_above_its_highest_blocker() -> None:
    issues = {
        1: an_issue(1),
        2: an_issue(2, blocked_by=frozenset({1})),
        3: an_issue(3, blocked_by=frozenset({1, 2})),
    }

    assert parallel.levels(issues) == {1: 0, 2: 1, 3: 2}


def test_a_blocker_of_another_repository_counts_as_level_zero() -> None:
    elsewhere = an_issue(1)
    elsewhere.blocked_elsewhere = 1

    assert parallel.levels({1: elsewhere, 2: an_issue(2)}) == {1: 1, 2: 0}


def test_a_cycle_of_blockers_ends() -> None:
    issues = {
        1: an_issue(1, blocked_by=frozenset({2})),
        2: an_issue(2, blocked_by=frozenset({1})),
    }

    assert set(parallel.levels(issues)) == {1, 2}


def test_the_levels_of_a_repository_are_read_from_the_items() -> None:
    def item(item_id: int, number: int, repository: str, level: int | None) -> dict:
        return {
            "id": item_id,
            "content_type": "Issue",
            "content": {
                "number": number,
                "repository_url": f"https://api.github.com/repos/{repository}",
            },
            "fields": [{"name": "Level", "value": level}],
        }

    items = [
        item(10, 1, "o/a", 2),
        item(20, 2, "o/a", None),
        item(30, 3, "o/b", 1),
        {"id": 40, "content_type": "DraftIssue", "content": {}, "fields": []},
    ]

    assert parallel.levels_of(items, "o/a") == {1: (10, 2), 2: (20, None)}


def test_only_the_levels_that_change_are_written() -> None:
    current = {1: (10, 0), 2: (20, None), 3: (30, 1.0), 4: (40, 2), 5: (50, None)}

    # 1 keeps its level, 2 gets one, 3 moves, 4 has closed, 5 is closed already.
    written = parallel.level_changes({1: 0, 2: 1, 3: 2}, current)

    assert written == [(20, 1), (30, 2), (40, None)]


def test_an_open_issue_outside_the_project_gets_no_level() -> None:
    assert parallel.level_changes({1: 0}, {}) == []


def project_answers(items: list, added: dict):
    """The answers of ``rest`` for the levels: the items, then each new item."""

    def answer(_path: str, **options) -> object:
        if options.get("method") == "POST":
            return added
        if options.get("method") == "PATCH":
            return None
        return items

    return answer


def test_an_open_issue_outside_the_project_is_added_before_its_level(
    capsys,
) -> None:
    issues = {1: an_issue(1), 2: an_issue(2, blocked_by=frozenset({1}))}
    items = [
        {
            "id": 100,
            "content_type": "Issue",
            "content": {
                "number": 1,
                "repository_url": "https://api.github.com/repos/o/a",
            },
            "fields": [{"name": "Level", "value": 0}],
        }
    ]

    with (
        patch.object(parallel, "repository", return_value="o/a"),
        patch.object(
            parallel, "project_field", return_value=("orgs/o/projectsV2/1", 7)
        ),
        patch.object(
            parallel, "rest", side_effect=project_answers(items, {"id": 200})
        ) as rest,
    ):
        parallel.write_levels(issues)

    # 1 is an item with its level already; 2 is not an item, so it is added
    # (its id is 20) and then gets its level.
    assert [(call.args, call.kwargs) for call in rest.call_args_list] == [
        (("orgs/o/projectsV2/1/items?per_page=100&fields=7",), {"every_page": True}),
        (
            ("orgs/o/projectsV2/1/items",),
            {"method": "POST", "data": {"type": "Issue", "id": 20}},
        ),
        (
            ("orgs/o/projectsV2/1/items/200",),
            {"method": "PATCH", "data": {"fields": [{"id": 7, "value": 1}]}},
        ),
    ]
    out = capsys.readouterr().out
    assert "Project: #2 added" in out
    assert "Level: 1 written" in out


def test_an_added_item_without_an_id_stops_the_levels() -> None:
    with (
        patch.object(parallel, "repository", return_value="o/a"),
        patch.object(
            parallel, "project_field", return_value=("orgs/o/projectsV2/1", 7)
        ),
        patch.object(parallel, "rest", side_effect=project_answers([], {})),
        pytest.raises(SystemExit, match="#1 .*no item"),
    ):
        parallel.write_levels({1: an_issue(1)})


# The release before the milestone


def test_the_versions_of_milestones_compare_as_numbers() -> None:
    assert parallel.version_of("0.9.1") < parallel.version_of("0.10.0")
    assert parallel.version_of("v1.2") == parallel.version_of("1.2.0")
    assert parallel.version_of("Backlog") is None
    assert parallel.version_of("") is None


def test_an_issue_is_blocked_by_the_release_of_the_milestone_before() -> None:
    issues = {
        140: an_issue(140, kind="release", version="0.6.0"),
        137: an_issue(137, version="0.6.0"),
        186: an_issue(186, kind="release", version="0.8.0"),
        187: an_issue(187, kind="release", version="0.9.0"),
        190: an_issue(190, kind="proposal", version="0.10.0"),
        191: an_issue(191, kind="release", version="0.10.0"),
    }

    added, removed, refused = parallel.release_links(issues)

    assert added == [
        (140, 137),
        (186, 140),
        (187, 186),
        (190, 187),
        (191, 187),
        (191, 190),
    ]
    assert removed == []
    assert refused == []
    assert issues[190].blocked_by == {187}
    assert issues[137].blocked_by == set()


def test_a_new_release_in_between_takes_the_place_of_the_old_one() -> None:
    issues = {
        187: an_issue(187, kind="release", version="0.9.0"),
        190: an_issue(190, version="0.10.0", blocked_by=frozenset({187})),
        191: an_issue(191, kind="release", version="0.10.0"),
        200: an_issue(200, kind="release", version="0.9.1"),
    }
    issues[191].blocked_by = {187, 190}

    added, removed, refused = parallel.release_links(issues)

    assert added == [(190, 200), (191, 200), (200, 187)]
    assert removed == [(190, 187), (191, 187)]
    assert refused == []
    assert issues[191].blocked_by == {190, 200}


def test_a_link_that_is_there_already_is_kept() -> None:
    issues = {
        187: an_issue(187, kind="release", version="0.9.0"),
        190: an_issue(190, version="0.10.0", blocked_by=frozenset({187})),
    }

    assert parallel.release_links(issues) == ([], [], [])


def test_an_issue_without_a_milestone_keeps_its_links_to_releases() -> None:
    issues = {
        187: an_issue(187, kind="release", version="0.9.0"),
        5: an_issue(5, blocked_by=frozenset({187})),
        6: an_issue(6, version="Backlog"),
    }

    assert parallel.release_links(issues) == ([], [], [])
    assert issues[5].blocked_by == {187}


def test_a_link_that_closes_a_cycle_is_refused() -> None:
    # 145 is in 0.10.0, but release 0.9.0 waits for 150, which waits for 145.
    issues = {
        145: an_issue(145, version="0.10.0"),
        150: an_issue(150, version="0.9.0", blocked_by=frozenset({145})),
        187: an_issue(187, kind="release", version="0.9.0"),
    }
    issues[187].blocked_by = {150}

    added, removed, refused = parallel.release_links(issues)

    assert (added, removed, refused) == ([], [], [(145, 187)])
    assert issues[145].blocked_by == set()


def test_the_links_are_written_through_rest(capsys) -> None:
    issues = {
        187: an_issue(187, kind="release", version="0.9.0"),
        190: an_issue(190, version="0.10.0", blocked_by=frozenset({187})),
        200: an_issue(200, kind="release", version="0.9.1"),
    }

    with (
        patch.object(parallel, "repository", return_value="o/a"),
        patch.object(parallel, "rest") as rest,
    ):
        parallel.write_release_links(issues)

    assert [call.args for call in rest.call_args_list] == [
        ("repos/o/a/issues/190/dependencies/blocked_by/1870",),
        ("repos/o/a/issues/190/dependencies/blocked_by",),
        ("repos/o/a/issues/200/dependencies/blocked_by",),
    ]
    assert [call.kwargs for call in rest.call_args_list] == [
        {"method": "DELETE"},
        {"method": "POST", "data": {"issue_id": 2000}},
        {"method": "POST", "data": {"issue_id": 1870}},
    ]
    assert "Releases: 2 links added, 1 removed" in capsys.readouterr().out


def test_a_refused_link_names_the_issue_to_move(capsys) -> None:
    issues = {
        145: an_issue(145, version="0.10.0"),
        150: an_issue(150, version="0.9.0", blocked_by=frozenset({145})),
        187: an_issue(187, kind="release", version="0.9.0"),
    }
    issues[187].blocked_by = {150}

    with (
        patch.object(parallel, "repository", return_value="o/a"),
        patch.object(parallel, "rest") as rest,
    ):
        parallel.write_release_links(issues)

    assert rest.call_args_list == []
    assert "#145 is in a later milestone than #187" in capsys.readouterr().out


# The issues of the milestone of a release


def test_a_release_is_blocked_by_the_issues_of_its_milestone() -> None:
    issues = {
        140: an_issue(140, kind="release", version="0.6.0"),
        220: an_issue(220, version="0.6.0"),
        225: an_issue(225, kind="proposal", version="0.6.0"),
        239: an_issue(239, kind="proposal", version="0.6.2"),
    }
    issues[140].blocked_by = {220}

    added, removed, refused = parallel.release_links(issues)

    assert added == [(140, 225), (239, 140)]
    assert (removed, refused) == ([], [])
    assert issues[140].blocked_by == {220, 225}


def test_a_release_loses_the_link_to_an_issue_moved_to_another_milestone() -> None:
    issues = {
        240: an_issue(240, kind="release", version="0.6.2"),
        244: an_issue(244, version="0.6.1"),
        248: an_issue(248, kind="release", version="0.6.1"),
    }
    issues[240].blocked_by = {244, 248}

    added, removed, refused = parallel.release_links(issues)

    assert added == [(248, 244)]
    assert removed == [(240, 244)]
    assert refused == []
    assert issues[240].blocked_by == {248}


def test_a_release_keeps_its_links_to_issues_without_a_version_and_to_releases() -> (
    None
):
    issues = {
        5: an_issue(5),
        6: an_issue(6, version="Backlog"),
        187: an_issue(187, kind="release", version="0.9.0"),
        191: an_issue(191, kind="release", version="0.10.0"),
    }
    # 999 is not an open issue of the repository: the link stays.
    issues[191].blocked_by = {5, 6, 187, 999}

    assert parallel.release_links(issues) == ([], [], [])
    assert issues[191].blocked_by == {5, 6, 187, 999}


@pytest.mark.parametrize(
    ("release", "inner", "later"), [(140, 241, 245), (187, 146, 150)]
)
def test_a_release_keeps_the_issue_it_ships_over_the_release_before(
    release: int, inner: int, later: int
) -> None:
    # inner is in 0.9.0 and waits for later, of 0.10.0: the release waits for
    # inner whatever the numbers, and later does not wait for the release.
    issues = {
        release: an_issue(release, kind="release", version="0.9.0"),
        inner: an_issue(inner, version="0.9.0", blocked_by=frozenset({later})),
        later: an_issue(later, version="0.10.0"),
    }

    added, removed, refused = parallel.release_links(issues)

    assert (added, removed, refused) == ([(release, inner)], [], [(later, release)])


def test_a_release_link_that_closes_a_cycle_names_the_issue_to_move(capsys) -> None:
    # 150 is in 0.9.0, but it waits for 145, of 0.10.0, which waits for 187.
    issues = {
        145: an_issue(145, version="0.10.0", blocked_by=frozenset({187})),
        150: an_issue(150, version="0.9.0", blocked_by=frozenset({145})),
        187: an_issue(187, kind="release", version="0.9.0"),
    }

    with (
        patch.object(parallel, "repository", return_value="o/a"),
        patch.object(parallel, "rest") as rest,
    ):
        parallel.write_release_links(issues)

    assert rest.call_args_list == []
    assert issues[187].blocked_by == set()
    assert (
        "#150 waits for #187 through its blockers: move #150 to a later milestone"
        in capsys.readouterr().out
    )


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


def test_levels_and_write_also_write_what_can_run_in_parallel() -> None:
    assert run_main("--levels", "--write")["parallel"] == [({}, True)]


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


# The REST calls


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:MUNQU/commol.git",
        "https://github.com/MUNQU/commol",
        "https://github.com/MUNQU/commol.git",
    ],
)
def test_the_repository_comes_from_the_remote_url(url: str) -> None:
    assert github.repository_of(url) == "MUNQU/commol"


def test_the_repository_ignores_an_insteadof_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=tmp_path, check=True)

    git("init", "-q")
    git("remote", "add", "origin", "git@github.com:MUNQU/commol.git")
    git("config", "url.https://mirror.example/.insteadOf", "git@github.com:")
    monkeypatch.chdir(tmp_path)

    assert github.repository.__wrapped__() == "MUNQU/commol"


def test_every_page_is_read() -> None:
    pages = [
        response(
            "200 OK",
            "[1, 2]",
            'Link: <https://api.github.com/repositories/9/issues?page=2>; rel="next"',
        ),
        response("200 OK", "[3]"),
    ]

    with patch.object(github.subprocess, "run", side_effect=pages) as run:
        assert github.rest("repos/o/a/issues", every_page=True) == [1, 2, 3]

    assert run.call_args_list[1].args[0][-1] == (
        "https://api.github.com/repositories/9/issues?page=2"
    )


def test_a_failed_request_stops_with_one_line() -> None:
    failed = response("404 Not Found", '{"message": "Not Found"}')

    with (
        patch.object(github.subprocess, "run", return_value=failed),
        pytest.raises(SystemExit) as stop,
    ):
        github.rest("repos/o/a/issues/9")

    assert str(stop.value) == "GitHub: GET repos/o/a/issues/9: Not Found (HTTP 404)"


def test_a_rate_limit_gives_the_reset_time() -> None:
    limited = response(
        "403 Forbidden",
        '{"message": "API rate limit exceeded for user ID 1."}',
        "X-Ratelimit-Remaining: 0",
        "X-Ratelimit-Reset: 1792325382",
    )

    with (
        patch.object(github.subprocess, "run", return_value=limited),
        pytest.raises(SystemExit) as stop,
    ):
        github.rest("repos/o/a/issues")

    assert str(stop.value) == (
        "GitHub: GET repos/o/a/issues: API rate limit exceeded for user ID 1. "
        "(HTTP 403). GitHub limits this account until 12:09 UTC: stop, and try "
        "again then."
    )
