"""Check that CI runs with read rights and pinned tools, and that the commands
the documentation shows are the commands CI runs. The workflows are in
``.github/workflows/``; the rules are in ``docs/development/workflow.md``
(Continuous Integration Pipelines)."""

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PYPROJECT = REPO_ROOT / "py-commol" / "pyproject.toml"
HOOKS = REPO_ROOT / ".pre-commit-config.yaml"
SET_VERSION = REPO_ROOT / ".github" / "actions" / "set-version" / "action.yml"
DOCUMENTED = [
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
    REPO_ROOT / "docs" / "getting-started" / "installation.md",
    REPO_ROOT / "docs" / "development" / "workflow.md",
]
WRITERS = {("docs.yaml", "deploy-docs")}
"""The jobs that write to the repository: the deploy of the site pushes the
branch ``gh-pages``."""


def load(path: Path) -> dict[str, Any]:
    """The YAML document of ``path``."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def workflows() -> dict[str, dict[str, Any]]:
    """Every workflow of CI, by its file name."""
    return {path.name: load(path) for path in sorted(WORKFLOWS.glob("*.yaml"))}


def steps() -> list[tuple[str, str, dict[str, Any]]]:
    """Every step of every workflow, with its workflow and its job."""
    return [
        (name, job_name, step)
        for name, workflow in workflows().items()
        for job_name, job in workflow["jobs"].items()
        for step in job.get("steps", [])
    ]


def test_every_workflow_reads_the_repository_by_default() -> None:
    """Every workflow reads with its token, and only the deploy of the site
    asks to write."""
    found = workflows()
    assert found
    no_read = sorted(
        name
        for name, workflow in found.items()
        if workflow.get("permissions") != {"contents": "read"}
    )
    assert not no_read, (
        f"Set `permissions: contents: read` at the top of: {', '.join(no_read)}"
    )
    writers = {
        (name, job_name)
        for name, workflow in found.items()
        for job_name, job in workflow["jobs"].items()
        if job.get("permissions", {}).get("contents") == "write"
    }
    assert writers == WRITERS


def test_the_release_builds_with_the_locked_maturin() -> None:
    """The sdist and every wheel of a release build with the maturin of
    ``uv.lock``."""
    release = load(WORKFLOWS / "release.yaml")
    commands = [
        step["run"]
        for job in release["jobs"].values()
        for step in job.get("steps", [])
        if "run" in step
    ]
    assert not [command for command in commands if "pip install" in command]
    # maturin is a tool of the extra `dev`.
    assert any(
        re.match(r"uv run --locked .*--extra dev maturin sdist ", command)
        for command in commands
    )

    # The job `maturin` reads the version of the lock; every wheel job takes it.
    lock_job = release["jobs"]["maturin"]
    assert "py-commol/uv.lock" in lock_job["steps"][-1]["run"]
    assert lock_job["outputs"]["version"] == "${{ steps.lock.outputs.version }}"
    wheels = {
        job_name: job
        for job_name, job in release["jobs"].items()
        if any(
            step.get("uses", "").startswith("PyO3/maturin-action@")
            for step in job.get("steps", [])
        )
    }
    assert wheels
    for job_name, job in wheels.items():
        assert job["needs"] == "maturin", job_name
        for step in job["steps"]:
            if step.get("uses", "").startswith("PyO3/maturin-action@"):
                assert step["with"]["maturin-version"] == (
                    "${{ needs.maturin.outputs.version }}"
                ), job_name


def test_ci_and_the_hooks_use_the_uv_of_the_project() -> None:
    """``pyproject.toml`` pins one uv; CI installs it, and the hook ``uv-lock``
    runs the uv of the machine, which refuses to run at another version."""
    pin = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["uv"][
        "required-version"
    ]
    assert re.fullmatch(r"==\d+\.\d+\.\d+", pin), pin

    installs = [
        (name, job_name, step)
        for name, job_name, step in steps()
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    ]
    assert installs
    for name, job_name, step in installs:
        settings = step.get("with", {})
        assert "version" not in settings, (name, job_name)
        assert settings.get("version-file") == "py-commol/pyproject.toml", (
            name,
            job_name,
        )

    hooks = load(HOOKS)["repos"]
    assert not [repo for repo in hooks if "uv-pre-commit" in repo["repo"]]
    [uv_lock] = [
        hook
        for repo in hooks
        if repo["repo"] == "local"
        for hook in repo["hooks"]
        if hook["id"] == "uv-lock"
    ]
    assert uv_lock["entry"] == "uv lock --check --project py-commol"
    assert uv_lock["language"] == "system"


def test_the_documented_clippy_is_the_one_ci_runs() -> None:
    """The documentation, the pull request template and the hook run the
    clippy of CI, and build the extension with the test model kind that the
    tests need."""
    [clippy] = [
        step["run"]
        for _, _, step in steps()
        if step.get("run", "").startswith("cargo clippy")
    ]
    assert "--workspace --all-targets --all-features" in clippy
    hooks = load(HOOKS)["repos"]
    [hook] = [
        hook for repo in hooks for hook in repo["hooks"] if hook["id"] == "cargo-clippy"
    ]
    assert hook["entry"] == clippy

    for path in DOCUMENTED:
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(REPO_ROOT).as_posix()
        if path.name in {"AGENTS.md", "PULL_REQUEST_TEMPLATE.md"}:
            assert clippy in text, name
            assert "maturin develop --release --features test-kind" in text, name
        for command in re.findall(r"cargo clippy[^`\n]*", text):
            assert "--workspace --all-targets --all-features" in command, (
                name,
                command,
            )
        # The builds that a reader runs: the lines of the shell blocks.
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            for command in re.findall(r"maturin develop[^\n]*", block):
                assert "--features test-kind" in command, (name, command)


def test_the_release_sets_the_version_of_every_crate() -> None:
    """The release writes its version in the root manifest, in
    ``[workspace.package]``, the one version every crate takes, and no other
    line of that manifest matches the edit of ``set-version``."""
    action = SET_VERSION.read_text(encoding="utf-8")
    edits = re.findall(r"sed -i .*' (\S+)$", action, re.M)
    assert edits
    assert set(edits) == {"Cargo.toml"}
    manifest = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert len(re.findall(r'^version = ".*"', manifest, re.M)) == 1
    assert tomllib.loads(manifest)["workspace"]["package"]["version"]


def test_every_crate_takes_its_package_fields_from_the_workspace() -> None:
    """Every crate takes its version, edition and the other shared fields
    from ``[workspace.package]``, and its shared dependencies from
    ``[workspace.dependencies]``."""
    root = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    shared = set(root["workspace"]["dependencies"])
    assert {"argmin", "finitediff", "libm", "rayon", "serde", "serde_json"} <= shared
    fields = ["version", "authors", "edition", "license", "homepage", "repository"]
    manifests = [
        *sorted((REPO_ROOT / "crates").glob("*/Cargo.toml")),
        REPO_ROOT / "py-commol" / "Cargo.toml",
    ]
    for path in manifests:
        manifest = tomllib.loads(path.read_text(encoding="utf-8"))
        name = path.relative_to(REPO_ROOT).as_posix()
        package = manifest["package"]
        for field in fields:
            assert package.get(field) == {"workspace": True}, (name, field)
        for table in ("dependencies", "dev-dependencies"):
            for dependency, spec in manifest.get(table, {}).items():
                if dependency in shared:
                    assert isinstance(spec, dict) and spec.get("workspace"), (
                        name,
                        dependency,
                    )
                else:
                    assert not (isinstance(spec, dict) and "git" in spec), (
                        name,
                        dependency,
                    )
