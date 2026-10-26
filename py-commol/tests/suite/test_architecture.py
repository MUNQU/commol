"""Check the layers of commol (``docs/codebase/design-principles.md``): the
Rust crates of ``crates/`` decide, and the PyO3 bridge of ``py-commol/`` is
their Python edge. No crate of ``crates/`` depends on PyO3 or on the bridge."""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BRIDGE = REPO_ROOT / "py-commol" / "Cargo.toml"
EDGE_CRATES = ("pyo3", "numpy", "pythonize")


def edge_dependencies(crates: Path, bridge: str) -> list[str]:
    """Each dependency of a crate under ``crates`` on an edge: ``crate dependency``."""
    found = []
    for manifest in sorted(crates.glob("*/Cargo.toml")):
        package = tomllib.loads(manifest.read_text(encoding="utf-8"))
        name = package["package"]["name"]
        for table in ("dependencies", "build-dependencies"):
            for dependency in package.get(table, {}):
                if dependency in (*EDGE_CRATES, bridge):
                    found.append(f"{name} {dependency}")
    return found


def test_no_crate_of_the_core_depends_on_python() -> None:
    bridge = tomllib.loads(BRIDGE.read_text(encoding="utf-8"))["package"]["name"]

    assert edge_dependencies(REPO_ROOT / "crates", bridge) == []


def test_the_check_names_each_dependency_on_an_edge(tmp_path: Path) -> None:
    manifests = {
        "plain": '[package]\nname = "plain"\n[dependencies]\nserde = "1"\n',
        "bound": '[package]\nname = "bound"\n[dependencies]\npyo3 = "0.27"\n',
        "tool": '[package]\nname = "tool"\n[build-dependencies]\nbridge = "1"\n',
    }
    for folder, manifest in manifests.items():
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "Cargo.toml").write_text(manifest, encoding="utf-8")

    assert edge_dependencies(tmp_path, "bridge") == ["bound pyo3", "tool bridge"]
