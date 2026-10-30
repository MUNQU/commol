"""Walk the public surface of commol: the exported names, the members of each,
the types that their signatures name, quoted or not, and the results that they
return; and check that a user can pass every class a signature takes and name
every class a member returns."""

import builtins
import collections.abc
import enum
import importlib
import inspect
import pkgutil
import sys
import typing
from collections.abc import Callable, Iterable, Iterator
from types import ModuleType
from typing import NamedTuple

import commol
from tests.suite.public_surface import EXPORTS

type Annotations = dict[str, object]
"""The annotation of each argument of a signature, without its return."""


def commol_modules() -> list[ModuleType]:
    """Every module of commol, without the extension."""
    found = [commol]
    for info in pkgutil.walk_packages(commol.__path__, "commol."):
        if not info.name.startswith("commol.commol_rs"):
            found.append(importlib.import_module(info.name))
    return found


def namespace() -> dict[str, object]:
    """Each class and type alias that a module of commol defines, by name: the
    names a string annotation of commol can read."""
    names: dict[str, object] = {}
    for module in commol_modules():
        for name, item in vars(module).items():
            if (
                isinstance(item, type | typing.TypeAliasType)
                and getattr(item, "__module__", None) == module.__name__
            ):
                names.setdefault(name, item)
    return names


NAMESPACE = namespace()


class Outside(type):
    """A type that a module of commol names only to check its types, such as
    a figure of matplotlib: it is not a class of commol."""


class Names(dict[str, object]):
    """The names a string annotation reads: the classes and aliases of
    commol, and an `Outside` for a name that its module imports only to check
    its types."""

    def __init__(self, module: ModuleType) -> None:
        super().__init__(NAMESPACE)
        self.module = module

    def __missing__(self, name: str) -> object:
        if name in vars(self.module) or hasattr(builtins, name):
            raise KeyError(name)
        return Outside(name, (), {})


def annotations_of(item: object) -> Annotations:
    """The annotations of a function or a class, each string read as code."""
    module = sys.modules[getattr(item, "__module__", "commol")]
    return inspect.get_annotations(
        item,  # ty: ignore[invalid-argument-type]
        eval_str=True,
        globals=vars(module),
        locals=Names(module),
    )


def exported() -> dict[str, object]:
    """Each exported object, by its public path (``commol.Simulation``)."""
    return {
        f"{module}.{name}": getattr(importlib.import_module(module), name)
        for module, names in EXPORTS.items()
        for name in names
    }


def own_classes(cls: type) -> list[type]:
    """``cls`` and each base of it that a module of commol defines."""
    return [
        klass
        for klass in cls.__mro__
        if klass.__module__.startswith("commol.") and klass.__module__ != "commol"
    ]


SYNTAX = frozenset(
    {"__getitem__", "__iter__", "__next__", "__call__", "__enter__", "__contains__"}
)
"""The methods that a user calls through syntax, such as ``scenarios["a"]``."""


def functions(cls: type) -> Iterator[tuple[str, Callable[..., object]]]:
    """Each public method and property of ``cls``, its own and inherited from
    commol, with each method of ``SYNTAX`` and ``__init__`` for a class that
    is not a pydantic model."""
    seen: set[str] = set()
    for klass in own_classes(cls):
        for name, member in vars(klass).items():
            public = (
                not name.startswith("_")
                or name in SYNTAX
                or (name == "__init__" and not hasattr(cls, "model_fields"))
            )
            if name in seen or not public:
                continue
            seen.add(name)
            if isinstance(member, property) and member.fget is not None:
                yield name, member.fget
            elif isinstance(member, classmethod | staticmethod):
                yield name, member.__func__
            elif inspect.isfunction(member):
                yield name, member


def fields(cls: type) -> Annotations:
    """The annotation of each public field of ``cls``."""
    found: Annotations = {}
    for klass in reversed(own_classes(cls)):
        for name, annotation in annotations_of(klass).items():
            if not name.startswith("_") and typing.get_origin(annotation) is not (
                typing.ClassVar
            ):
                found[name] = annotation
    return found


class Signature(NamedTuple):
    """A signature that the surface offers a user."""

    owner: str
    """``Class.member``, a class, or a function."""
    arguments: Annotations
    """The annotation of each argument."""
    returned: object
    """The annotation of what it returns, or None."""


def signatures(item: object, given: bool = False) -> Iterator[Signature]:
    """Each signature that ``item`` offers a user. The fields of a class are
    the arguments that build it or, when a member of the surface ``given``
    the class to the user, what it returns."""
    if not isinstance(item, type):
        hints = annotations_of(item)
        returned = hints.pop("return", None)
        yield Signature(getattr(item, "__qualname__", ""), hints, returned)
        return
    hints = fields(item)
    if given:
        for name, annotation in hints.items():
            yield Signature(f"{item.__name__}.{name}", {}, annotation)
    elif hasattr(item, "model_fields"):
        yield Signature(item.__name__, hints, None)
    for name, function in functions(item):
        hints = annotations_of(function)
        returned = hints.pop("return", None)
        hints.pop("self", None)
        hints.pop("cls", None)
        yield Signature(f"{item.__name__}.{name}", hints, returned)


class Written(dict[str, object]):
    """The names a quoted annotation of commol reads: the classes and aliases
    of commol, ``typing`` and ``collections.abc``, and an `Outside` for any
    other name that is not a builtin."""

    def __init__(self) -> None:
        super().__init__({**vars(collections.abc), **vars(typing), **NAMESPACE})

    def __missing__(self, name: str) -> object:
        if hasattr(builtins, name):
            raise KeyError(name)
        return Outside(name, (), {})


def resolved(annotation: object) -> object:
    """``annotation``, with a string or a ``ForwardRef``, such as
    ``"dict[str, Entry]"``, read as the type it writes."""
    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        return eval(annotation, {"__builtins__": builtins}, Written())
    return annotation


def named_types(annotation: object) -> Iterator[object]:
    """Each class and type alias that ``annotation`` names, nested or not: a
    string or a ``ForwardRef`` at any depth reads the class it names."""
    annotation = resolved(annotation)
    if isinstance(annotation, list | tuple):
        for item in annotation:
            yield from named_types(item)
        return
    if isinstance(annotation, typing.TypeAliasType):
        yield annotation
        yield from named_types(annotation.__value__)
        return
    origin = typing.get_origin(annotation)
    if origin is typing.Literal or isinstance(annotation, typing.TypeVar):
        return
    if origin is typing.Annotated:
        yield from named_types(typing.get_args(annotation)[0])
        return
    if isinstance(annotation, type):
        metadata = getattr(annotation, "__pydantic_generic_metadata__", None) or {}
        yield metadata.get("origin") or annotation
    elif isinstance(origin, type):
        yield origin
    yield from named_types(typing.get_args(annotation))


def home_of(module: str) -> str:
    """The public module that exports the classes of a module of commol: the
    root for a private package, and ``commol.<page>`` for the package of a
    page."""
    package = module.split(".")[1]
    return "commol" if package.startswith("_") else f"commol.{package}"


def is_commol(item: object) -> bool:
    """Whether ``item`` is a class that a module of commol defines."""
    return isinstance(item, type) and item.__module__.startswith("commol.")


def generic_results(cls: type) -> Iterator[type]:
    """The result type each generic base of ``cls`` names, such as
    ``BootstrapResult`` of ``UncertaintyMethod[BootstrapResult]``."""
    for klass in cls.__mro__:
        metadata = getattr(klass, "__pydantic_generic_metadata__", None) or {}
        bases = [*metadata.get("args", ()), *getattr(klass, "__orig_bases__", ())]
        for argument in named_types(bases):
            if is_commol(argument) and argument not in cls.__mro__:
                yield argument


def walk() -> tuple[dict[type, set[object]], set[type]]:
    """Each class of commol that the surface returns, with the objects that
    return it, and the classes a user gets back: from a member, or as the
    type of a field of one."""
    returned_by: dict[type, set[object]] = {}
    given: set[type] = set()
    pending: list[object] = list(exported().values())
    while pending:
        item = pending.pop()
        returned = [
            found
            for signature in signatures(item, item in given)
            for found in named_types(signature.returned)
        ]
        if isinstance(item, type):
            returned.extend(generic_results(item))
        for result in returned:
            if not isinstance(result, type) or not is_commol(result):
                continue
            returned_by.setdefault(result, set()).add(item)
            if result not in given:
                given.add(result)
                pending.append(result)
    return returned_by, given


def results() -> dict[type, set[object]]:
    """Each class of commol that the surface returns, with the objects that
    return it."""
    returned_by, _ = walk()
    return returned_by


def surface() -> list[object]:
    """Each exported object and each result it reaches that no module
    exports."""
    public = exported().values()
    return [*public, *(cls for cls in results() if cls not in public)]


def inputs() -> Iterator[tuple[str, object]]:
    """Each argument annotation of the surface, as ``(owner(argument),
    annotation)``."""
    _, given = walk()
    for item in surface():
        for signature in signatures(item, item in given):
            for name, annotation in signature.arguments.items():
                yield f"{signature.owner}({name})", annotation


class Output(NamedTuple):
    """A class or a type that the surface gives a user back."""

    item: object
    """The object of the surface that gives it: a class or a function."""
    owner: str
    """``Class.member``, a class, or a function."""
    annotation: object
    """The annotation of what it gives."""


def outputs() -> Iterator[Output]:
    """Each annotation that the surface returns: the return of each member,
    the type of each field of a class a user gets back, and the result that
    a generic base of a method names."""
    _, given = walk()
    for item in surface():
        for signature in signatures(item, item in given):
            if signature.returned is not None:
                yield Output(item, signature.owner, signature.returned)
        if isinstance(item, type):
            for result in generic_results(item):
                yield Output(item, item.__name__, result)


def annotations() -> Iterator[tuple[str, object]]:
    """Each annotation of the surface, of an argument or of a return."""
    yield from inputs()
    for output in outputs():
        yield output.owner, output.annotation


def choice_faults(annotation: object, package: str = "commol.") -> list[str]:
    """Each enum and each public type alias of ``package`` that ``annotation``
    names: a choice among names is a ``Literal``, and a union stays
    private."""
    faults = []
    for item in named_types(annotation):
        if isinstance(item, type) and issubclass(item, enum.Enum):
            faults.append(f"the enum {item.__name__}")
        if (
            isinstance(item, typing.TypeAliasType)
            and item.__module__.startswith(package)
            and not item.__name__.startswith("_")
        ):
            faults.append(f"the alias {item.__name__}")
    return faults


def validators(cls: type) -> list[str]:
    """The name of each pydantic validator and serializer of ``cls``."""
    decorators = getattr(cls, "__pydantic_decorators__", None)
    if decorators is None:
        return []
    kinds = (
        "validators",
        "field_validators",
        "root_validators",
        "model_validators",
        "field_serializers",
        "model_serializers",
    )
    return [name for kind in kinds for name in getattr(decorators, kind)]


def private_inputs() -> list[str]:
    """Each argument of the surface that names a class of commol a user
    cannot pass: one that is not exported, not a base of an exported class
    and not a protocol that they implement."""
    reachable = {item for item in exported().values() if isinstance(item, type)}
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


def private_outputs(
    found: Iterable[tuple[str, object]], public: set[object]
) -> list[str]:
    """Each class of commol that a member of ``found`` returns and that
    ``public`` does not hold, as ``owner: Class``; ``found`` holds ``(owner,
    annotation)``."""
    return sorted(
        {
            f"{owner}: {item.__name__}"
            for owner, annotation in found
            for item in named_types(annotation)
            if is_commol(item) and item not in public
        }
    )
