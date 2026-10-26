"""The port of the bridge: how a data object of the Python API crosses to the
engine.

Every data object crosses as the JSON of its ``model_dump()``, which the
engine reads into the type it owns and checks, with the path of the field an
error is about. JSON has no number that is not finite, so such a number
crosses as the text ``"Infinity"``, ``"-Infinity"`` or ``"NaN"``: the serializer
of pydantic writes it with ``inf_nan_mode="strings"``, the mode that the
setting ``ser_json_inf_nan="strings"`` of a model selects, here once for every
class that crosses. The engine reads such a text
as a number only where its type asks for a number, and writes the same
texts.

A field that holds a result with behavior in the engine, such as the sample
of a method, is not part of the JSON: the class names it in its class
variable ``_engine_handles``, and it crosses as its handle
(`engine_handles`).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic_core import to_json

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import EngineHandle


def _dump(value: BaseModel) -> object:
    handles = getattr(value, "_engine_handles", ())
    return value.model_dump(exclude=set(handles))


def engine_json(value: BaseModel | Sequence[BaseModel]) -> str:
    """
    The JSON of `value` that the engine reads.

    Parameters
    ----------
    value : BaseModel | Sequence[BaseModel]
        A data object, or a list of them.

    Returns
    -------
    str
        The JSON of its ``model_dump()``, without the fields that cross as
        handles, with the texts ``"Infinity"``, ``"-Infinity"`` and ``"NaN"``
        for the numbers that are not finite.
    """
    dumped = (
        _dump(value)
        if isinstance(value, BaseModel)
        else [_dump(item) for item in value]
    )
    return to_json(dumped, inf_nan_mode="strings").decode()


def _handle(value: object) -> "EngineHandle":
    """The handle of `value`: the engine object of a sample, or the JSON of a
    result that has none."""
    from commol.context.inference.sample import ParameterSample

    if isinstance(value, ParameterSample):
        return value._rust
    if isinstance(value, BaseModel):
        return engine_json(value)
    assert value is None, "a handle is a sample, a result or None"
    return None


def engine_handles(value: BaseModel) -> dict[str, "EngineHandle"]:
    """
    The handle of each field of `value` that crosses as a handle.

    Parameters
    ----------
    value : BaseModel
        A data object.

    Returns
    -------
    dict[str, EngineHandle]
        The keyword arguments of the function of the bridge: each field of
        ``_engine_handles`` by name, with the engine object of a sample, the
        JSON of a result, or None.
    """
    handles: tuple[str, ...] = getattr(value, "_engine_handles", ())
    return {name: _handle(getattr(value, name)) for name in handles}
