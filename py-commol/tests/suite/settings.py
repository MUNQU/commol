"""The settings of commol that the rule tests of ``test_rules.py`` read: the
public settings with their names, and the range constraints that a pydantic
model of commol declares."""

import importlib
import inspect
import pkgutil
import typing
from collections.abc import Iterable, Iterator

import annotated_types
from pydantic import BaseModel
from pydantic.fields import FieldInfo

import commol

RANGE_BOUNDS = ("gt", "ge", "lt", "le")
"""The bounds of a range constraint of pydantic and ``annotated_types``."""


def model_classes(annotation: object) -> Iterator[type[BaseModel]]:
    """The pydantic models of commol that ``annotation`` names, nested or not."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation.__module__.startswith("commol."):
            yield annotation
        return
    for argument in typing.get_args(getattr(annotation, "__value__", annotation)):
        yield from model_classes(argument)


def arguments(owner: str, function: object) -> Iterator[tuple[str, str]]:
    """Each argument of ``function``, a method of a class or a function, as
    ``(owner, name)``."""
    if isinstance(function, classmethod | staticmethod):
        function = function.__func__
    if inspect.isfunction(function):
        for argument in inspect.signature(function).parameters:
            yield owner, argument


def public_settings() -> Iterator[tuple[str, str]]:
    """Each field of a public pydantic model, and of the models its fields
    hold, and each argument of a public function or method, as ``(owner,
    name)``."""
    seen: set[type] = set()
    pending: list[object] = [getattr(commol, name) for name in commol.__all__]
    while pending:
        item = pending.pop()
        if not isinstance(item, type):
            yield from arguments(getattr(item, "__qualname__", ""), item)
            continue
        if item in seen:
            continue
        seen.add(item)
        if issubclass(item, BaseModel):
            for name, field in item.model_fields.items():
                yield item.__name__, name
                pending.extend(model_classes(field.annotation))
        for attribute, member in vars(item).items():
            if not attribute.startswith("_") or attribute == "__init__":
                yield from arguments(f"{item.__name__}.{attribute}", member)


def commol_models() -> Iterator[type[BaseModel]]:
    """Every pydantic model that a module of commol defines, private ones
    included."""
    for module in pkgutil.walk_packages(commol.__path__, "commol."):
        if module.name.startswith("commol.commol_rs"):
            continue
        for _, item in inspect.getmembers(importlib.import_module(module.name)):
            if (
                isinstance(item, type)
                and issubclass(item, BaseModel)
                and item.__module__ == module.name
            ):
                yield item


def bounds(metadata: Iterable[object]) -> Iterator[str]:
    """The range constraints of ``metadata``, such as ``ge=1`` or
    ``allow_inf_nan=False``, the metadata of a field or of an ``Annotated``."""
    for item in metadata:
        if isinstance(item, FieldInfo):
            yield from bounds(item.metadata)
            continue
        if isinstance(item, annotated_types.BaseMetadata | annotated_types.Interval):
            for bound in RANGE_BOUNDS:
                if getattr(item, bound, None) is not None:
                    yield f"{bound}={getattr(item, bound)}"
        if getattr(item, "allow_inf_nan", None) is False:
            yield "allow_inf_nan=False"


def annotation_bounds(annotation: object) -> Iterator[str]:
    """The range constraints inside ``annotation``: of an ``Annotated``, of a
    type alias, of a list or a union, nested or not."""
    annotation = getattr(annotation, "__value__", annotation)
    if typing.get_origin(annotation) is typing.Annotated:
        base, *metadata = typing.get_args(annotation)
        yield from bounds(metadata)
        yield from annotation_bounds(base)
        return
    for argument in typing.get_args(annotation):
        yield from annotation_bounds(argument)


def declared_ranges() -> Iterator[str]:
    """Each field of a model of commol that declares a range constraint, as
    ``module.Model.field: constraints``."""
    for model in commol_models():
        for name, field in model.model_fields.items():
            found = [*bounds(field.metadata), *annotation_bounds(field.annotation)]
            if found:
                yield f"{model.__module__}.{model.__name__}.{name}: {', '.join(found)}"
