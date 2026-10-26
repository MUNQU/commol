"""Check the layers of commol (``docs/codebase/design-principles.md``): the
Rust crates of ``crates/`` decide, and the PyO3 bridge of ``py-commol/`` is
their Python edge. No crate of ``crates/`` depends on PyO3 or on the bridge."""

import ast
import enum
import importlib
import inspect
import re
import tomllib
import typing
from collections.abc import Callable, Iterator
from pathlib import Path

from pydantic import BaseModel

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


def runner_functions() -> list[Callable[..., object]]:
    """The runner of each inference method of commol, without the methods
    that tests register."""
    import commol.api.inference  # noqa: F401  (registers the runners)
    from commol.api.inference.registry import _RUNNERS

    runners = dict.fromkeys(_RUNNERS.values())
    return [runner for runner in runners if runner.__module__.startswith("commol.")]


def reached(function: Callable[..., object]) -> list[ast.FunctionDef]:
    """``function`` and each function of its module that it calls, nested."""
    module = ast.parse(inspect.getsource(inspect.getmodule(function)))
    defined = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    found: dict[str, ast.FunctionDef] = {}
    pending = [function.__name__]
    while pending:
        name = pending.pop()
        if name in found or name not in defined:
            continue
        found[name] = defined[name]
        pending.extend(
            node.func.id
            for node in ast.walk(defined[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )
    return list(found.values())


def runner_faults(functions: list[ast.FunctionDef], method: str) -> list[str]:
    """What keeps the runner of ``functions`` from sending its method whole:
    a check of a type, a number of calls to the bridge other than one, and a
    call that does not get ``engine_json(method)`` or gets a field of
    ``method``."""
    faults = []
    calls = []
    for function in functions:
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                text = ast.unparse(node.func)
                if text == "isinstance":
                    faults.append(f"isinstance in {function.name}")
                if text.startswith("commol_rs."):
                    calls.append(node)
    if len(calls) != 1:
        faults.append(f"{len(calls)} calls to the bridge")
    for call in calls:
        arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
        if not any(f"engine_json({method})" == ast.unparse(a) for a in arguments):
            faults.append("the bridge does not get engine_json(method)")
        faults.extend(
            f"the bridge gets {ast.unparse(node)}"
            for argument in arguments
            for node in ast.walk(argument)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == method
        )
    return list(dict.fromkeys(faults))


def test_a_runner_sends_its_method_whole() -> None:
    """Each runner calls one function of the bridge with the JSON of its
    method and no field of it, and checks no type: the method decides in the
    engine."""
    wrong = []
    for runner in runner_functions():
        method = list(inspect.signature(runner).parameters)[1]
        wrong.extend(
            f"{runner.__module__}.{runner.__name__}: {fault}"
            for fault in runner_faults(reached(runner), method)
        )

    assert not wrong, (
        "Send the method whole, as engine_json(method), to one function of "
        "the bridge:\n" + "\n".join(sorted(wrong))
    )


def test_the_check_of_runners_names_each_fault() -> None:
    tree = ast.parse(
        "def run(calibrator, method):\n"
        "    if isinstance(method.eta, float):\n"
        "        pass\n"
        "    commol_rs.inference.laplace(problem, method.draws)\n"
        "    commol_rs.inference.fit(problem)\n"
        "def good(calibrator, method):\n"
        "    return commol_rs.inference.laplace(problem, engine_json(method))\n"
    )
    run, good = (node for node in tree.body if isinstance(node, ast.FunctionDef))

    assert runner_faults([run], "method") == [
        "isinstance in run",
        "2 calls to the bridge",
        "the bridge does not get engine_json(method)",
        "the bridge gets method.draws",
    ]
    assert runner_faults([good], "method") == []


def method_classes() -> list[type[BaseModel]]:
    """Each class of an inference method of commol that has a runner, without
    the methods that tests register."""
    import commol.api.inference  # noqa: F401  (registers the runners)
    from commol.api.inference.registry import _RUNNERS

    methods = [method for method in _RUNNERS if method.__module__.startswith("commol.")]
    return sorted(methods, key=lambda method: method.__name__)


def test_no_method_excludes_a_field() -> None:
    """No method class leaves a field out of its dump: the dump is the one
    shape of a method, which the engine reads and a sample records."""
    wrong = sorted(
        f"{method.__name__}.{name}"
        for method in method_classes()
        for name, field in method.model_fields.items()
        if field.exclude
    )

    assert not wrong, "Keep every field of a method in its dump:\n" + "\n".join(wrong)


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
