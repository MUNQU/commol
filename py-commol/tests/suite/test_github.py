"""Check ``.github/scripts/github.py``, the REST calls of the planning scripts:
the repository of the checkout, every page of an answer, and the line that
says why GitHub refused a request. No test calls GitHub: each one patches
``subprocess.run``."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.support.planning import load

github = load("github")


def response(status: str, body: str, *headers: str) -> subprocess.CompletedProcess:
    """What ``gh api -i`` prints for one answer of GitHub."""
    text = "\n".join([f"HTTP/2.0 {status}", *headers, "", body])
    code = 0 if status.startswith("2") else 1
    return subprocess.CompletedProcess([], code, stdout=text, stderr="")


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
