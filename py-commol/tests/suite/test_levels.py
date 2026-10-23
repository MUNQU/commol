"""Check the precedence levels of ``.github/scripts/parallel.py`` (``--levels``):
the level of each open issue and the field ``Level`` of the project. The rules
are in ``docs/development/workflow.md``, "Planning"."""

from unittest.mock import patch

import pytest

from tests.support.planning import an_issue, parallel


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
