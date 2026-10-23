"""Check the links of the releases of ``.github/scripts/parallel.py``
(``--releases``): each issue waits for the release before its milestone, and
each release for the issues of its milestone. The rules are in
``docs/development/workflow.md``, "Milestones"."""

from unittest.mock import patch

import pytest

from tests.support.planning import an_issue, parallel


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
