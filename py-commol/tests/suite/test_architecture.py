"""Check the layers of commol (``docs/codebase/design-principles.md``): the
Rust crates of ``crates/`` decide, and the PyO3 bridge of ``py-commol/`` is
their Python edge. No crate of ``crates/`` depends on PyO3 or on the bridge."""

import ast
import enum
import importlib
import re
import tomllib
import typing
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType


from tests.suite import public_surface
from tests.support.surface import (
    annotations,
    choice_faults,
    exported,
    inputs,
    is_commol,
    named_types,
    own_classes,
    results,
    surface,
    validators,
)

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


# The port of the bridge (#401): data crosses as JSON

BRIDGE_SOURCES = REPO_ROOT / "py-commol" / "src"
PACKAGE = REPO_ROOT / "py-commol" / "commol"

HANDLES = {
    "Engine": "a model compiled by the engine, which simulates it",
    "EngineType": "an entry of the list of engines, which checks settings",
    "ModelKind": "a kind of model, which names its values",
    "MathExpression": "a formula read by the engine, which evaluates it",
    "CalibrationProblem": "a calibration problem, which methods run on",
    "ParameterSample": "a sample of parameter values, which methods read",
    "Predictive": "the predictive of a sample, which draws",
    "ObservationDraws": "draws of observations, which scores read",
    "Distribution": "a distribution, which gives densities and draws",
    "ForwardAnalysis": "the inputs of a sensitivity analysis, which run",
    "Parallelism": "the threads of this context, which every parallel method reads",
}
"""Each class of the bridge with behavior in Rust, and what it does."""

RESULTS = {
    "CalibrationEvaluation",
    "CalibrationResult",
    "CalibrationResultWithHistory",
    "StartOutcome",
    "Starts",
    "BootstrapRun",
    "JackknifeRun",
    "WeightedSummary",
    "ConformalRun",
    "PointRank",
    "StructuralAnalysis",
    "CollinearityVerdict",
    "CollinearityAnalysis",
    "SampleVerdict",
    "SampleAnalysis",
    "StartsAnalysis",
    "Profile",
    "ProfileStructure",
    "ProfileRun",
    "LocalRun",
    "MixtureRun",
    "SobolRun",
    "ChaosRun",
    "SMCRun",
    "EtaCalibration",
}
"""Each class of the bridge that carries a result of the engine to Python."""


def bridge_classes(sources: Path) -> list[str]:
    """The Python name of each ``#[pyclass]`` of the Rust files in ``sources``."""
    found = []
    for source in sorted(sources.glob("*.rs")):
        text = source.read_text(encoding="utf-8")
        for match in re.finditer(r"#\[pyclass\(([^\]]*)\)\]", text):
            name = re.search(r'name\s*=\s*"(\w+)"', match.group(1))
            found.append(name.group(1) if name else "?")
    return found


def test_no_bridge_class_only_carries_data() -> None:
    """Each class of the bridge is a handle or a result: data crosses as the
    JSON of its pydantic model."""
    classes = bridge_classes(BRIDGE_SOURCES)

    unknown = sorted(set(classes) - set(HANDLES) - RESULTS)
    assert not unknown, (
        "A class of the bridge that only carries data crosses as JSON; list a "
        "handle in HANDLES and a result in RESULTS:\n" + "\n".join(unknown)
    )


def test_the_check_of_bridge_classes_reads_each_name(tmp_path: Path) -> None:
    (tmp_path / "a.rs").write_text(
        '#[pyclass(name = "Run", frozen)]\nstruct A;\n'
        '#[pyclass(frozen, name = "Settings")]\nstruct B;\n',
        encoding="utf-8",
    )

    assert bridge_classes(tmp_path) == ["Run", "Settings"]


def modules(package: Path) -> Iterator[tuple[str, ast.Module]]:
    """Each module of ``package``, with its path from the package and its tree."""
    for path in sorted(package.rglob("*.py")):
        if "commol_rs" not in path.parts:
            text = path.read_text(encoding="utf-8")
            yield path.relative_to(package).as_posix(), ast.parse(text)


def bridge_constructions(tree: ast.Module) -> Iterator[str]:
    """Each class of ``commol_rs`` that ``tree`` builds, as ``module.Class``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = ast.unparse(node.func).split(".")
        if chain[0] == "commol_rs" and chain[-1][:1].isupper():
            yield ".".join(chain[1:])


def public_rust_names(tree: ast.Module) -> Iterator[str]:
    """Each public name that ``tree`` defines and that contains ``rust``."""
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [ast.unparse(target) for target in node.targets]
        elif isinstance(node, ast.AnnAssign):
            names = [ast.unparse(node.target)]
        for name in names:
            last = name.split(".")[-1]
            if "rust" in last.lower() and not last.startswith("_"):
                yield name


def test_no_python_code_converts_to_rust() -> None:
    """No module of commol builds a data object of the bridge, and no public
    name of commol contains ``rust``: data crosses as the JSON of its
    ``model_dump()``."""
    wrong = []
    for path, tree in modules(PACKAGE):
        for built in bridge_constructions(tree):
            if built.split(".")[-1] not in HANDLES:
                wrong.append(f"{path}: builds {built}")
        wrong.extend(f"{path}: {name}" for name in public_rust_names(tree))

    assert not wrong, (
        "Send the JSON of a pydantic model to the engine, and keep a member "
        "that holds a handle private:\n" + "\n".join(sorted(wrong))
    )


def test_the_check_of_conversions_names_each_one() -> None:
    tree = ast.parse(
        "a = commol_rs.calibration.LossConfig.sse()\n"
        "b = commol_rs.engine.Engine(text)\n"
        "c = commol_rs.inference.Settings(1)\n"
        "def rust_problem(): ...\n"
        "def _rust_problem(): ...\n"
        "self.rust_value = 1\n"
    )

    assert list(bridge_constructions(tree)) == ["engine.Engine", "inference.Settings"]
    assert sorted(public_rust_names(tree)) == ["rust_problem", "self.rust_value"]


def public_modules() -> list[ModuleType]:
    """Each public module of commol, imported."""
    return [importlib.import_module(module) for module in public_surface.EXPORTS]


def enums_read_by_name(sources: Path) -> set[str]:
    """Each type whose ``from_name`` a binding under ``sources`` calls."""
    return {
        match.group(1)
        for source in sorted(sources.glob("*.rs"))
        for match in re.finditer(r"(\w+)::from_name\(", source.read_text("utf-8"))
    }


def test_no_binding_reads_a_value_of_an_enum_from_its_name() -> None:
    """A value of an enum crosses as the JSON of its name, read with serde
    and the spelling of Python (D2 of #401)."""
    assert enums_read_by_name(BRIDGE_SOURCES) == set()


def test_the_check_names_each_enum_read_by_name(tmp_path: Path) -> None:
    (tmp_path / "keep.rs").write_text(
        "let keep = RateKeeps::from_name(keep)?;\nlet x = Levels::from_name(y);\n",
        encoding="utf-8",
    )

    assert enums_read_by_name(tmp_path) == {"RateKeeps", "Levels"}


# The public surface (#404): the names a user writes


def private_inputs() -> list[str]:
    """Each argument of the surface that names a class of commol a user
    cannot pass: one that is not exported, not a result of the table, not a
    base of either and not a protocol that they implement."""
    reachable = {
        item
        for item in [*exported().values(), *results()]
        if isinstance(item, type)
        and (item.__name__ in public_surface.RESULTS or item in exported().values())
    }
    allowed = {base for cls in reachable for base in own_classes(cls)}
    return sorted(
        f"{owner}: {item.__name__}"
        for owner, annotation in inputs()
        for item in named_types(annotation)
        if isinstance(item, type)
        and is_commol(item)
        and item not in allowed
        and not getattr(item, "_is_protocol", False)
    )


def test_the_public_surface_is_the_table() -> None:
    """The exports are the table of ``tests/suite/public_surface.py``, and a
    user can reach every class that a public signature takes."""
    found = {
        module: set(importlib.import_module(module).__all__)
        for module in public_surface.EXPORTS
    }

    assert found == {
        module: set(names) for module, names in public_surface.EXPORTS.items()
    }
    assert private_inputs() == []


def homes(exports: dict[str, list[str]]) -> dict[str, list[str]]:
    """Each name that more than one module of ``exports`` exports, with those
    modules."""
    found: dict[str, list[str]] = {}
    for module, names in exports.items():
        for name in names:
            found.setdefault(name, []).append(module)
    return {name: where for name, where in found.items() if len(where) > 1}


def test_each_public_name_has_one_home() -> None:
    """Each public name is in the ``__all__`` of one module, and the root
    holds no object of another module."""
    modules = public_modules()
    root = importlib.import_module("commol")
    shared = homes({module.__name__: list(module.__all__) for module in modules})
    shared.update(
        {
            f"commol.{name}": ["commol", module.__name__]
            for module in modules
            if module is not root
            for name in module.__all__
            if getattr(root, name, None) is getattr(module, name)
        }
    )

    assert not shared, "Export each name from one module:\n" + "\n".join(
        f"{name}: {where}" for name, where in sorted(shared.items())
    )


def test_the_check_of_homes_names_each_shared_name() -> None:
    exports = {"commol": ["Model"], "commol.a": ["X", "Y"], "commol.b": ["Y"]}

    assert homes(exports) == {"Y": ["commol.a", "commol.b"]}


def package_faults(package: Path, public: set[str]) -> list[str]:
    """What in ``package`` is neither a public module, the extension nor a
    private package: a module outside a package, a public package that is not
    a public module, and a public module that is not a package."""
    faults = []
    packages = set()
    for entry in sorted(package.iterdir()):
        if entry.suffix == ".py" and entry.name != "__init__.py":
            faults.append(f"the module {entry.name} is outside a package")
        elif (entry / "__init__.py").is_file():
            packages.add(entry.name)
            if not entry.name.startswith("_") and entry.name not in public:
                faults.append(f"the package {entry.name} is not a public module")
    faults += [
        f"the module {name} is not a package" for name in sorted(public - packages)
    ]
    return faults


def test_the_packages_are_the_modules() -> None:
    """The packages of commol are its public modules, the extension
    ``commol_rs`` and private packages (``_…``)."""
    public = {
        module.removeprefix("commol.")
        for module in public_surface.EXPORTS
        if module != "commol"
    }

    assert package_faults(PACKAGE, public | {"commol_rs"}) == []


def test_the_check_of_packages_names_each_fault(tmp_path: Path) -> None:
    for name in ("models", "_inner", "api"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "helpers.py").write_text("", encoding="utf-8")
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")

    assert package_faults(tmp_path, {"models", "plots"}) == [
        "the package api is not a public module",
        "the module helpers.py is outside a package",
        "the module plots is not a package",
    ]


LAYERS = (
    "commol_rs",
    "_build",
    "_data",
    "_statistics",
    "parallelism",
    "engines",
    "distributions",
    "quantities",
    "_fit",
    "observations",
    "_problem",
    "_model",
    "plotting",
    "_simulation",
    "uncertainty",
    "_calibration",
    "scores",
    "resampling",
    "profiles",
    "local",
    "smc",
    "conformal",
    "identifiability",
    "sensitivity",
    "commol",
)
"""The packages of commol from the bottom up: a package imports only the
packages below it. The root ``commol`` is at the top."""


def package_of(module: str) -> str:
    """The package of a dotted module of commol: ``commol`` for the root."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else "commol"


def imported_modules(
    tree: ast.Module, module: str, packages: set[str]
) -> Iterator[str]:
    """Each module of commol that ``tree`` imports, wherever the import is: at
    the top, in a function or under ``TYPE_CHECKING``; a name of the root that
    is not one of ``packages`` is the root ``commol``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names if a.name.split(".")[0] == "commol")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = module.split(".")[: -node.level]
                base = ".".join([*parts, base] if base else parts)
            if base.split(".")[0] != "commol":
                continue
            if base == "commol":
                yield from (
                    f"commol.{a.name}" if a.name in packages else "commol"
                    for a in node.names
                )
            else:
                yield base


def upward_imports(package: Path, layers: tuple[str, ...]) -> list[str]:
    """Each import of a module of ``package`` that points up ``layers``, or at
    a package that ``layers`` does not order."""
    names = {path.name for path in package.iterdir() if path.is_dir()}
    found = []
    for path, tree in modules(package):
        module = "commol." + path.removesuffix(".py").replace("/", ".")
        module = module.removesuffix(".__init__")
        source = package_of(module) if module != "commol" else "commol"
        for target in imported_modules(tree, module, names):
            other = package_of(target)
            if other == source:
                continue
            if source not in layers or other not in layers:
                found.append(f"{path}: {source} or {other} is not in the layers")
            elif layers.index(other) > layers.index(source):
                found.append(f"{path}: {source} imports {target}, above it")
    return sorted(set(found))


def test_the_dependencies_point_one_way() -> None:
    """Each package imports only the packages below it in ``LAYERS``, with no
    exception: no import upward at the top of a module, in a function or under
    ``TYPE_CHECKING``."""
    wrong = upward_imports(PACKAGE, LAYERS)

    assert not wrong, (
        "Read what a package above gives through a protocol of the package "
        "below, or move the code to the package it serves:\n" + "\n".join(wrong)
    )


def test_the_check_names_an_import_upward(tmp_path: Path) -> None:
    for name in ("low", "high"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "low" / "a.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "import commol.low.b\n"
        "if TYPE_CHECKING:\n"
        "    from commol.high.c import C\n"
        "def f():\n"
        "    from commol import high\n",
        encoding="utf-8",
    )
    (tmp_path / "high" / "c.py").write_text(
        "from commol.low import a\nfrom commol.odd import d\n", encoding="utf-8"
    )

    assert upward_imports(tmp_path, ("low", "high", "commol")) == [
        "high/c.py: high or odd is not in the layers",
        "low/a.py: low imports commol.high, above it",
        "low/a.py: low imports commol.high.c, above it",
    ]


def test_a_choice_is_a_literal() -> None:
    """No signature of the surface names an enum or a public type alias."""
    wrong = sorted(
        f"{owner}: {fault}"
        for owner, annotation in annotations()
        for fault in choice_faults(annotation)
    )

    assert not wrong, (
        "Write a choice among names as a Literal, and keep a union private "
        "(``type _Name = ...``):\n" + "\n".join(wrong)
    )


def test_the_check_of_choices_names_each_enum_and_alias() -> None:
    class Mode(enum.StrEnum):
        SUM = "sum"

    type Pooling = typing.Literal["series", "all"]
    type _Union = int | str

    assert choice_faults(list[Mode] | None, "tests.") == ["the enum Mode"]
    assert choice_faults(dict[str, Pooling], "tests.") == ["the alias Pooling"]
    assert choice_faults(typing.Literal["a", "b"] | _Union, "tests.") == []


def test_no_public_member_is_internal() -> None:
    """No class of the surface has a public validator, and no exported class
    has an exported base."""
    public = {item for item in exported().values() if isinstance(item, type)}
    wrong = sorted(
        f"{cls.__name__}.{name}"
        for cls in (item for item in surface() if isinstance(item, type))
        for name in validators(cls)
        if not name.startswith("_")
    )
    wrong += sorted(
        f"{cls.__name__} has the exported base {base.__name__}"
        for cls in public
        for base in own_classes(cls)[1:]
        if base in public
    )

    assert not wrong, (
        "Start the name of a validator with _, and keep a base private:\n"
        + "\n".join(wrong)
    )
