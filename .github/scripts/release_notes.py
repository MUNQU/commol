"""Write the notes of a GitHub release from the changelog.

The notes are the section ``## [X.Y.Z]`` of ``docs/about/changelog.md``, with
its links to other pages turned into links to the published site, then the
install command and the comparison with the previous release. The changelog is
the one source; the release notes are never written by hand.

Usage, from the root of the repository::

    python .github/scripts/release_notes.py 0.4.0 > notes.md
    gh release create v0.4.0 --verify-tag --title "Version 0.4.0" \\
        --notes-file notes.md --latest
"""

import argparse
import posixpath
import re
import sys
from pathlib import Path

CHANGELOG = Path("docs/about/changelog.md")
SITE = "https://munqu.github.io/commol/latest"
REPOSITORY = "https://github.com/MUNQU/commol"
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)
LINK = re.compile(r"\]\((?P<target>(?!https?://|#)[^)\s]+\.md)(?P<anchor>#[^)\s]*)?\)")


def section(text: str, version: str) -> tuple[str, str | None]:
    """
    The body of the section of `version`, and the version of the next section.

    Parameters
    ----------
    text : str
        The changelog.
    version : str
        The version, such as ``0.4.0``.

    Returns
    -------
    tuple[str, str | None]
        The body, without its heading, and the previous version, or None when
        the section is the last one.

    Raises
    ------
    SystemExit
        If the changelog has no section for `version`.
    """
    headings = list(HEADING.finditer(text))
    for index, heading in enumerate(headings):
        if heading.group("version") == version:
            start = text.index("\n", heading.start()) + 1
            after = headings[index + 1] if index + 1 < len(headings) else None
            end = after.start() if after else len(text)
            body = text[start:end]
            body = body.split("\n## Version History", 1)[0]
            previous = after.group("version") if after else None
            return body.strip(), previous
    sys.exit(f"{CHANGELOG} has no section [{version}]")


BLOCK_START = re.compile(r"^(#|[-*] |\d+\. |\||```|<!--)")


def unwrap(text: str) -> str:
    """
    Join the wrapped lines of each paragraph and list item.

    GitHub shows every line break of release notes, so a paragraph wrapped at 80
    columns in the changelog would show as short lines. Headings, list items,
    table rows and fenced code keep their lines.

    Parameters
    ----------
    text : str
        Markdown with wrapped lines.

    Returns
    -------
    str
        The same Markdown, one line per paragraph or list item.
    """
    lines: list[str] = []
    fenced = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            lines.append(line)
            continue
        previous = lines[-1].strip() if lines else ""
        joins = (
            not fenced
            and stripped
            and previous
            and not BLOCK_START.match(stripped)
            and not previous.startswith(("#", "|", "```"))
        )
        if joins:
            lines[-1] = f"{lines[-1]} {stripped}"
        else:
            lines.append(line)
    return "\n".join(lines)


def site_link(match: re.Match[str]) -> str:
    """A link of the changelog to another page, as a link to the site."""
    page = posixpath.normpath(posixpath.join("about", match.group("target")))
    page = page.removesuffix(".md").removesuffix("index")
    url = f"{SITE}/{page}/".replace("//", "/").replace("https:/", "https://")
    return f"]({url}{match.group('anchor') or ''})"


def notes(version: str) -> str:
    """
    The notes of the release of `version`.

    Parameters
    ----------
    version : str
        The version, such as ``0.4.0``.

    Returns
    -------
    str
        The notes, in Markdown.
    """
    body, previous = section(CHANGELOG.read_text(encoding="utf-8"), version)
    parts = [f"## What's New in {version}", unwrap(LINK.sub(site_link, body))]
    install = f"pip install --upgrade commol=={version}"
    parts.append(f"## Installation\n\n```bash\n{install}\n```")
    if previous is not None and re.fullmatch(r"\d+\.\d+\.\d+\S*", previous):
        parts.append(
            f"## Full Changelog\n\n{REPOSITORY}/compare/v{previous}...v{version}"
        )
    return "\n\n".join(parts) + "\n"


def main() -> None:
    """Print the notes of the version given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the version, such as 0.4.0")
    print(notes(parser.parse_args().version), end="")


if __name__ == "__main__":
    main()
