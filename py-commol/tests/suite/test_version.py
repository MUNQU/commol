"""Check that the package reports its version and the commit it was built from.

The tests run on a build from the git source, which is never a release: its
version is the placeholder of ``[workspace.package]`` in the root
``Cargo.toml``.
"""

import importlib.metadata
import re
import subprocess
from pathlib import Path

import commol

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_the_package_reports_its_version_and_commit() -> None:
    info = commol.build_info()

    assert info.commit is not None
    assert re.fullmatch(r"[0-9a-f]{40}", info.commit)
    kind = subprocess.run(
        ["git", "cat-file", "-t", info.commit],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert kind == "commit"

    installed = importlib.metadata.version("commol")
    assert info.version == f"{installed}+g{info.commit[:7]}"
    assert commol.__version__ == info.version
