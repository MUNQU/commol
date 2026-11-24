"""Check the links of the releases of ``.github/scripts/parallel.py``
(``--releases``): each issue waits for the release before its milestone, and
each release and each audit for the issues of its milestone; and the templates
of a release and of an audit. The rules are in
``docs/development/workflow.md``, "Milestones"."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.support.planning import an_issue, parallel

ROOT = Path(__file__).resolve().parents[3]


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


# The audit of a milestone


def test_an_audit_is_blocked_by_the_issues_of_its_milestone() -> None:
    issues = {
        264: an_issue(264, kind="release", version="0.11.0"),
        516: an_issue(516, version="0.11.0"),
        548: an_issue(548, kind="proposal", version="0.11.0"),
        549: an_issue(549, kind="proposal", version="0.11.0", audit=True),
        134: an_issue(134, kind="proposal", version="0.13.0"),
        141: an_issue(141, kind="proposal"),
    }

    added, removed, refused = parallel.release_links(issues)

    assert added == [
        (134, 264),
        (264, 516),
        (264, 548),
        (264, 549),
        (549, 516),
        (549, 548),
    ]
    assert (removed, refused) == ([], [])
    assert issues[549].blocked_by == {516, 548}


def test_an_audit_loses_the_link_to_an_issue_moved_to_another_milestone() -> None:
    issues = {
        5: an_issue(5),
        516: an_issue(516, version="0.11.1"),
        549: an_issue(549, kind="proposal", version="0.11.0", audit=True),
    }
    # 5 has no version in its milestone and 999 is not loaded: their links stay.
    issues[549].blocked_by = {5, 516, 999}

    added, removed, refused = parallel.release_links(issues)

    assert (added, removed, refused) == ([], [(549, 516)], [])
    assert issues[549].blocked_by == {5, 999}


def test_an_audit_is_not_blocked_by_another_audit() -> None:
    issues = {
        328: an_issue(328, kind="proposal", version="0.12.0", audit=True),
        549: an_issue(549, kind="proposal", version="0.12.0", audit=True),
        550: an_issue(550, kind="proposal", version="0.13.0", audit=True),
    }
    # Links written by hand, to an audit of the milestone and to one of another.
    issues[328].blocked_by = {549, 550}

    assert parallel.release_links(issues) == ([], [(328, 549), (328, 550)], [])
    assert issues[328].blocked_by == set()


def test_an_audit_without_a_milestone_waits_for_nothing() -> None:
    issues = {
        5: an_issue(5),
        549: an_issue(549, kind="proposal", audit=True),
    }

    assert parallel.release_links(issues) == ([], [], [])


def test_an_audit_link_that_closes_a_cycle_is_refused(capsys) -> None:
    # 134 is in the milestone of the audit 328 and waits for it.
    issues = {
        134: an_issue(134, version="0.12.0", blocked_by=frozenset({328})),
        328: an_issue(328, kind="proposal", version="0.12.0", audit=True),
    }

    with (
        patch.object(parallel, "repository", return_value="o/a"),
        patch.object(parallel, "rest") as rest,
    ):
        parallel.write_release_links(issues)

    assert rest.call_args_list == []
    assert issues[328].blocked_by == set()
    assert (
        "#134 waits for the audit #328 of its milestone: remove the link that"
        " makes it wait" in capsys.readouterr().out
    )


def test_a_release_keeps_its_audit_over_the_issues_of_the_audit() -> None:
    # 600 waits for 141, which has no milestone and waits for the release: the
    # release still waits for its audit, and the audit does not wait for 600.
    issues = {
        141: an_issue(141, blocked_by=frozenset({264})),
        264: an_issue(264, kind="release", version="0.11.0"),
        549: an_issue(549, kind="proposal", version="0.11.0", audit=True),
        600: an_issue(600, version="0.11.0", blocked_by=frozenset({141})),
    }

    added, _, refused = parallel.release_links(issues)

    assert added == [(264, 549)]
    assert refused == [(264, 600), (549, 600)]


def a_closed_issue(
    number: int,
    version: str,
    *labels: str,
    closed_at: str = "2026-12-01T00:00:00Z",
    reason: str = "completed",
) -> dict:
    """A closed issue, as the REST API gives it."""
    return {
        "number": number,
        "state": "closed",
        "state_reason": reason,
        "closed_at": closed_at,
        "labels": [{"name": label} for label in labels],
        "milestone": {"number": 1, "title": version},
    }


def an_audit(number: int, version: str, **fields: str) -> dict:
    """A closed audit, as the REST API gives it."""
    return a_closed_issue(number, version, "proposal", "audit", **fields)


def warnings_of(
    issues: dict,
    audits: list[dict],
    closed: list[dict],
    capsys,
    since: str = "2026-12-01T00:00:00Z",
) -> list[str]:
    """The lines that ``warn_of_audits`` prints when GitHub has the closed audits
    ``audits``, and the issues ``closed``, some of them after ``since``."""
    paths: list[str] = []

    def rest(path: str, **options: object) -> list[dict]:
        paths.append(path)
        assert options == {"every_page": True}
        return audits if "labels=" in path else closed

    with (
        patch.object(parallel, "repository", return_value="o/a"),
        patch.object(parallel, "rest", side_effect=rest),
    ):
        parallel.warn_of_audits(issues)

    path = "repos/o/a/issues?state=closed"
    assert paths[0] == f"{path}&labels=proposal,audit&per_page=100"
    assert paths[1:] in ([], [f"{path}&since={since}&per_page=100"])
    written = capsys.readouterr()
    assert written.out == ""
    return written.err.splitlines()


OPEN_ONE = "open one from the template audit.md"


def test_a_release_with_no_audit_is_named(capsys) -> None:
    issues = {
        264: an_issue(264, kind="release", version="0.11.0"),
        265: an_issue(265, kind="release", version="0.12.0"),
        328: an_issue(328, kind="proposal", version="0.12.0", audit=True),
        300: an_issue(300, kind="release", version="Backlog"),
    }
    # The closed audits are of another milestone and of none.
    audits = [an_audit(9, "0.10.0"), an_audit(8, "Backlog")]

    assert warnings_of(issues, audits, [], capsys) == [
        f"Audits: 0.11.0 has no audit: {OPEN_ONE}"
    ]


def test_a_milestone_with_no_release_and_no_audit_is_named(capsys) -> None:
    issues = {516: an_issue(516, version="0.11.1"), 141: an_issue(141)}

    assert warnings_of(issues, [], [], capsys) == [
        f"Audits: 0.11.1 has no audit: {OPEN_ONE}"
    ]


def test_a_closed_audit_with_an_open_issue_is_named(capsys) -> None:
    issues = {
        264: an_issue(264, kind="release", version="0.11.0"),
        516: an_issue(516, version="0.11.0"),
        518: an_issue(518, version="0.11.0"),
        600: an_issue(600, version="0.12.0"),
        601: an_issue(601, kind="proposal", version="0.12.0", audit=True),
    }

    assert warnings_of(issues, [an_audit(549, "0.11.0")], [], capsys) == [
        "Audits: 0.11.0 has issues that its closed audit #549 did not read"
        " (#516, #518): reopen it"
    ]


def test_a_closed_audit_with_an_issue_closed_after_it_is_named(capsys) -> None:
    issues = {264: an_issue(264, kind="release", version="0.11.0")}
    later, earlier = "2026-12-02T00:00:00Z", "2026-11-30T00:00:00Z"
    closed = [
        a_closed_issue(600, "0.11.0", "change", closed_at=later),
        # Not work of the milestone that the audit missed: closed before it, of
        # another milestone, not done, not planned work, a pull request.
        a_closed_issue(601, "0.11.0", "change", closed_at=earlier),
        a_closed_issue(602, "0.12.0", "change", closed_at=later),
        a_closed_issue(603, "0.11.0", "change", closed_at=later, reason="duplicate"),
        a_closed_issue(604, "0.11.0", "question", closed_at=later),
        a_closed_issue(605, "0.11.0", "change", closed_at=later) | {"pull_request": {}},
    ]

    assert warnings_of(issues, [an_audit(549, "0.11.0")], closed, capsys) == [
        "Audits: 0.11.0 has issues that its closed audit #549 did not read"
        " (#600): reopen it"
    ]


def test_an_audit_closed_again_reads_what_closed_before(capsys) -> None:
    issues = {264: an_issue(264, kind="release", version="0.11.0")}
    again = "2026-12-03T00:00:00Z"
    audits = [an_audit(560, "0.11.0", closed_at=again), an_audit(570, "0.11.0")]
    closed = [a_closed_issue(600, "0.11.0", "change", closed_at="2026-12-02T00:00:00Z")]

    assert warnings_of(issues, audits, closed, capsys, since=again) == []


def test_a_closed_audit_of_a_finished_milestone_is_silent(capsys) -> None:
    issues = {264: an_issue(264, kind="release", version="0.11.0")}

    assert warnings_of(issues, [an_audit(549, "0.11.0")], [], capsys) == []


@pytest.mark.parametrize("reason", ["not_planned", "duplicate"])
def test_an_audit_that_was_not_done_covers_nothing(reason: str, capsys) -> None:
    issues = {264: an_issue(264, kind="release", version="0.11.0")}

    audits = [an_audit(549, "0.11.0", reason=reason)]

    assert warnings_of(issues, audits, [], capsys) == [
        f"Audits: 0.11.0 has no audit: {OPEN_ONE}"
    ]


def test_a_pull_request_with_the_label_covers_nothing(capsys) -> None:
    issues = {264: an_issue(264, kind="release", version="0.11.0")}

    audits = [an_audit(400, "0.11.0") | {"pull_request": {}}]

    assert warnings_of(issues, audits, [], capsys) == [
        f"Audits: 0.11.0 has no audit: {OPEN_ONE}"
    ]


def test_an_open_audit_covers_a_milestone_with_a_closed_one(capsys) -> None:
    issues = {
        264: an_issue(264, kind="release", version="0.11.0"),
        516: an_issue(516, version="0.11.0"),
        560: an_issue(560, kind="proposal", version="0.11.0", audit=True),
    }

    assert warnings_of(issues, [an_audit(549, "0.11.0")], [], capsys) == []


# The templates


def checklist(path: str) -> list[str]:
    """The items of the checklist of the release in ``path``, in order; the
    version of the template and of the page is the same placeholder."""
    text = (ROOT / path).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.startswith("- [ ] ")]


def test_the_checklist_of_the_template_is_the_one_of_the_release_page() -> None:
    template = checklist(".github/ISSUE_TEMPLATE/release.md")

    assert template == checklist("docs/development/release.md")
    assert [item for item in template if "audit" in item] == [
        "- [ ] The audit of X.Y.Z closed (its number), and `python"
        " .github/scripts/parallel.py --releases` prints no line `Audits:` about"
        " X.Y.Z: the audit never moves to another milestone"
    ]


def test_the_template_of_an_audit_has_the_labels_that_the_script_reads() -> None:
    template = ROOT / ".github" / "ISSUE_TEMPLATE" / "audit.md"
    found = [
        line.removeprefix("labels: ").split(", ")
        for line in template.read_text(encoding="utf-8").splitlines()
        if line.startswith("labels: ")
    ]

    assert found == [["proposal", parallel.AUDIT]]


def test_the_template_of_an_audit_asks_for_what_the_rules_say() -> None:
    text = (ROOT / ".github" / "ISSUE_TEMPLATE" / "audit.md").read_text(
        encoding="utf-8"
    )

    assert re.findall(r"^### (R\d) - (.*)$", text, re.MULTILINE) == [
        ("R1", "Each method agrees with its articles"),
        ("R2", "Compositions, and degenerate and extreme inputs"),
        ("R3", "The audit closes last and blocks the release"),
    ]
    assert 'its verdict is "not verified"' in text
    assert "the audit does not close (D4)" in text
    assert "git log --first-parent" in text


def test_a_proposal_of_a_method_starts_with_its_verdict_of_use() -> None:
    tasks = checklist(".github/ISSUE_TEMPLATE/proposal.md")

    assert tasks[0].startswith("- [ ] T0: the verdict of use")
