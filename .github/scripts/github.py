"""Talk to GitHub through its REST API only, with ``gh api``.

``parallel.py``, ``affected.py`` and ``next.sh`` call GitHub only through REST,
never through GraphQL (``gh api graphql``, ``gh issue``, ``gh project``):
REST has a limit of its own, and one request costs one point of it. ``rest``
makes every request. It reads every page when asked, and stops with one line
when GitHub refuses a request, with the time when the limit resets if GitHub
limits the account. It never retries.
"""

import json
import re
import subprocess
from datetime import UTC, datetime
from functools import cache

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)

NEXT_PAGE = re.compile(r'<([^>]+)>;\s*rel="next"')
REMOTE = re.compile(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?/?$")


def repository_of(url: str) -> str:
    """Return the ``owner/name`` of a GitHub remote URL (SSH or HTTPS)."""
    found = REMOTE.search(url.strip())
    if found is None:
        raise SystemExit(f"the remote {url.strip()!r} is not a GitHub repository")
    return str(found.group(1))


@cache
def repository() -> str:
    """Return the ``owner/name`` of this checkout, from its remote ``origin``."""
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("this checkout has no remote origin")
    return repository_of(result.stdout)


def typed(value: JsonValue) -> JsonValue:
    """Return a value of ``json.loads`` as what it is: a JSON value."""
    return value


def split_response(text: str) -> tuple[int, dict[str, str], str]:
    """Return the status, the headers (lower case) and the body of ``gh api -i``."""
    head, _, body = text.partition("\n\n")
    lines = head.splitlines()
    status = 0
    if lines and lines[0].startswith("HTTP/"):
        parts = lines[0].split()
        status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return status, headers, body


def failure(request: str, status: int, headers: dict[str, str], body: str) -> str:
    """Return the line that says why GitHub refused a request."""
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    message = data.get("message") if isinstance(data, dict) else None
    text = f"GitHub: {request}: {message or body.strip() or 'no answer'}"
    if status:
        text += f" (HTTP {status})"
    limited = (
        "rate limit" in str(message).lower()
        or headers.get("x-ratelimit-remaining") == "0"
    )
    if limited:
        wait = headers.get("retry-after", "")
        reset = headers.get("x-ratelimit-reset", "")
        if wait.isdigit():
            until = f"for {wait} seconds"
        elif reset.isdigit():
            until = f"until {datetime.fromtimestamp(int(reset), UTC):%H:%M} UTC"
        else:
            until = "for a while"
        text += f". GitHub limits this account {until}: stop, and try again then."
    return text


def rest(
    path: str,
    *,
    method: str = "GET",
    data: JsonValue = None,
    every_page: bool = False,
) -> JsonValue:
    """Make one request of the REST API of GitHub and return its JSON answer.

    ``data`` is the JSON body of the request. With ``every_page``, the answer
    is a list and the pages that the header ``Link`` names are read and joined.
    """
    found: list[JsonValue] = []
    url = path
    while True:
        command = ["gh", "api", "-i", "-X", method]
        if data is not None:
            command += ["--input", "-"]
        result = subprocess.run(
            [*command, url],
            input=None if data is None else json.dumps(data),
            capture_output=True,
            text=True,
            check=False,
        )
        status, headers, body = split_response(result.stdout)
        if result.returncode != 0 or status >= 400:
            text = body or result.stderr
            raise SystemExit(failure(f"{method} {path}", status, headers, text))
        answer = typed(json.loads(body)) if body.strip() else None
        if not every_page:
            return answer
        if not isinstance(answer, list):
            raise SystemExit(f"GitHub: {method} {path}: the answer is not a list")
        found.extend(answer)
        link = NEXT_PAGE.search(headers.get("link", ""))
        if link is None:
            return found
        url = link.group(1)
