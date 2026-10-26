"""Check the inference methods of commol: each declares its runner, which
sends the method whole to the engine, and keeps every field in its dump."""

import ast
import inspect
from collections.abc import Callable

from pydantic import BaseModel

from commol._calibration.method import InferenceMethod
from tests.suite.test_architecture import public_modules


def runner_functions() -> list[Callable[..., object]]:
    """The runner of each inference method of commol, without the methods
    of the tests."""
    runners = dict.fromkeys(method._runner for method in method_classes())
    return [runner for runner in runners if runner is not None]


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
    if len(calls) > 1:
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


def calls_the_engine(runner: Callable[..., object]) -> bool:
    """Whether ``runner``, or a function of its module that it calls, calls a
    function of the bridge. A runner that calls none reads a result the
    engine already gave, as `ProfileIdentifiability` reads its profiles."""
    return any(
        isinstance(node, ast.Call) and ast.unparse(node.func).startswith("commol_rs.")
        for function in reached(runner)
        for node in ast.walk(function)
    )


def test_a_runner_sends_its_method_whole() -> None:
    """Each runner calls at most one function of the bridge, with the JSON of
    its method and no field of it, and checks no type: the method decides in
    the engine."""
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


def method_classes() -> list[type[InferenceMethod[BaseModel]]]:
    """Each class of an inference method of commol that declares its runner,
    without the methods of the tests."""
    public_modules()  # each module defines its methods
    found: list[type[InferenceMethod[BaseModel]]] = []
    pending: list[type[InferenceMethod[BaseModel]]] = [InferenceMethod]
    while pending:
        method = pending.pop()
        pending.extend(method.__subclasses__())
        if method._runner is not None and method.__module__.startswith("commol."):
            found.append(method)
    return sorted(set(found), key=lambda method: method.__name__)


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
