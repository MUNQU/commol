"""The base of every result: equality over its values, the hash of a frozen
result model, and the JSON file of a result model."""

import json
import math
from abc import abstractmethod
from pathlib import Path
from typing import Self, override

from pydantic import BaseModel, ConfigDict

from commol._data.serialization import render_json

_NAN = object()
"""The canonical form of every NaN: one object, so a NaN equals a NaN."""


def canonical(value: object) -> object:
    """
    The values of `value` as nested tuples, with every NaN as one object.

    A result is its class and the values it compares (`ComparedByValue`); any
    other model is its class and its public fields, never its private
    attributes; a list or a tuple keeps its order and its kind; a dict is a set of its
    items, so the order of its keys does not count. ``0.0`` and ``-0.0`` stay
    floats, so they are equal and have one hash, as with ``==``.

    Parameters
    ----------
    value : object
        A result, a model, a container of values or a value.

    Returns
    -------
    object
        Equal for two values that hold the same values, with equal hashes.
    """
    if isinstance(value, ComparedByValue):
        return type(value), canonical(value.compared_values())
    if isinstance(value, BaseModel):
        return type(value), canonical(_public_fields(value))
    if isinstance(value, float):
        return _NAN if math.isnan(value) else value
    if isinstance(value, (list, tuple)):
        return (list if isinstance(value, list) else tuple), tuple(
            canonical(item) for item in value
        )
    if isinstance(value, dict):
        return dict, frozenset(
            (canonical(key), canonical(item)) for key, item in value.items()
        )
    return value


def _public_fields(model: BaseModel) -> tuple[tuple[str, object], ...]:
    """
    The public fields of `model` with their values, in the order of its class.

    Parameters
    ----------
    model : BaseModel
        A model.

    Returns
    -------
    tuple[tuple[str, object], ...]
        One ``(name, value)`` pair per field.
    """
    return tuple((name, getattr(model, name)) for name in type(model).model_fields)


class ComparedByValue:
    """
    The equality of every result of commol: a result model, the results of a
    run (`SimulationResults`) and of replicates (`SimulationReplicates`).

    Two results are equal when they are of one class and the values they
    compare (`compared_values`) are the same, a NaN equal to a NaN, whatever
    the objects that hold them (`canonical`). Anything that is not a result,
    such as a plain ``dict`` or ``list`` with the same items, is not equal to
    one.
    """

    @abstractmethod
    def compared_values(self) -> object:
        """
        The values two results of this class compare.

        Returns
        -------
        object
            The values, in a form `canonical` reads.
        """

    @override
    def __eq__(self, other: object) -> bool:
        """Equal when of one class and with the same values (`canonical`)."""
        if not isinstance(other, ComparedByValue):
            return False
        return canonical(self) == canonical(other)

    @override
    def __ne__(self, other: object) -> bool:
        """Not equal when `__eq__` is false, also for a subclass of ``dict``."""
        return not self == other


def _hash_of_values(self: BaseModel) -> int:
    return hash(canonical(self))


class ResultModel(ComparedByValue, BaseModel):
    """
    The base of every result model of commol.

    Two results are equal when they are of one class and their public fields
    hold the same values, a NaN equal to a NaN (`ComparedByValue`); private
    attributes, such as a Rust object, do not count. A frozen result
    (``frozen=True`` in its ``model_config``) has the hash of those values, so
    equal results have equal hashes; a result that can change has no hash.

    A result writes itself to a JSON file (`to_json`), and its class reads the
    file back to an equal result (`from_json`), so another process can analyze
    it. The JSON of a result is standard JSON: a number that is not finite is
    the text ``"Infinity"``, ``"-Infinity"`` or ``"NaN"``, as at the port of
    the engine (``_data/engine_json.py``), and the result reads it back.
    """

    model_config = ConfigDict(ser_json_inf_nan="strings")

    @classmethod
    @override
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Give a frozen result the hash of its values."""
        super().__pydantic_init_subclass__(**kwargs)
        if cls.model_config.get("frozen"):
            cls.__hash__ = _hash_of_values

    @override
    def compared_values(self) -> object:
        """The public fields with their values (`_public_fields`)."""
        return _public_fields(self)

    @classmethod
    def from_json(cls, file_path: str | Path) -> Self:
        """
        Read a result of this class from a JSON file that `to_json` wrote.

        The method reads the file as UTF-8, the encoding of JSON. The file
        holds the fields of the result and not the name of its class: this
        class validates the fields, so the file of a result of another class
        raises the ``pydantic.ValidationError`` that names each field that
        does not fit.

        Parameters
        ----------
        file_path : str | Path
            The path to the JSON file.

        Returns
        -------
        Self
            The result, equal to the result that wrote the file.

        Raises
        ------
        FileNotFoundError
            If the file at `file_path` does not exist.
        pydantic.ValidationError
            If the file does not hold a result of this class.
        """
        return cls.model_validate_json(Path(file_path).read_text(encoding="utf-8"))

    def to_json(self, file_path: str | Path, indent: int = 2) -> None:
        """
        Write the result to a JSON file, which `from_json` of its class reads.

        The method writes UTF-8 and every field, nulls included. A number
        that is not finite is the text ``"Infinity"``, ``"-Infinity"`` or
        ``"NaN"``, so the file is standard JSON. Long numeric arrays, such as
        the values of a sample, are on one line. The release of commol that
        wrote a file reads it.

        Parameters
        ----------
        file_path : str | Path
            Path of the file to write.
        indent : int, optional
            Indentation width for nested structures.
        """
        # The JSON of pydantic writes a number that is not finite as its text
        # (``ser_json_inf_nan``); ``model_dump(mode="json")`` keeps the float.
        payload = json.loads(self.model_dump_json())
        Path(file_path).write_text(
            render_json(payload, indent=indent), encoding="utf-8"
        )
