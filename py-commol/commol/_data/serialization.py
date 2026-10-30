"""The JSON file of a model or a result: one writer for both."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, JsonValue
from pydantic_core import to_json

INLINE_ARRAY_MIN_LENGTH = 8
_INLINE_TOKEN = re.compile(r'"@@inline:(\d+)@@"')


def _is_number(value: JsonValue) -> bool:
    """True for a JSON number or null, excluding booleans."""
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _is_numeric_array(payload: list[JsonValue]) -> bool:
    """True for a flat array of numbers, or an array of such arrays."""
    if all(_is_number(item) for item in payload):
        return True
    return all(
        isinstance(item, list) and all(_is_number(entry) for entry in item)
        for item in payload
    )


def _inline_numeric_arrays(
    payload: JsonValue, inlined: list[str], min_length: int
) -> JsonValue:
    """Replace long numeric arrays with placeholders holding their compact form."""
    if isinstance(payload, list):
        if len(payload) >= min_length and _is_numeric_array(payload):
            inlined.append(json.dumps(payload))
            return f"@@inline:{len(inlined) - 1}@@"
        return [_inline_numeric_arrays(item, inlined, min_length) for item in payload]
    if isinstance(payload, dict):
        return {
            key: _inline_numeric_arrays(value, inlined, min_length)
            for key, value in payload.items()
        }
    return payload


def render_json(
    payload: JsonValue,
    indent: int = 2,
    inline_min_length: int = INLINE_ARRAY_MIN_LENGTH,
) -> str:
    """
    Render JSON in declaration order, keeping long numeric arrays on one line.

    Numeric arrays of at least `inline_min_length` entries are written on a
    single line. Substitution goes through placeholders, so no pattern is ever
    applied to the rendered data itself.

    Parameters
    ----------
    payload : JsonValue
        A JSON-serializable structure.
    indent : int, optional
        Indentation width for nested structures.
    inline_min_length : int, optional
        Shortest numeric array written on one line.

    Returns
    -------
    str
        The rendered JSON document.
    """
    inlined: list[str] = []
    prepared = _inline_numeric_arrays(payload, inlined, inline_min_length)
    rendered = json.dumps(prepared, indent=indent)
    return _INLINE_TOKEN.sub(lambda match: inlined[int(match.group(1))], rendered)


def write_json_file(model: BaseModel, file_path: str | Path, indent: int = 2) -> None:
    """
    Write `model` to a JSON file that ``model_validate_json`` of its class reads.

    The file is UTF-8 and holds every field, nulls included. A number that is
    not finite is the text ``"Infinity"``, ``"-Infinity"`` or ``"NaN"`` in every
    nested model, whatever its own setting ``ser_json_inf_nan``, so the file is
    standard JSON and reads back as the number. Long numeric arrays are on one
    line (`render_json`).

    Parameters
    ----------
    model : BaseModel
        The model or the result to write.
    file_path : str | Path
        Path of the file to write.
    indent : int, optional
        Indentation width for nested structures.
    """
    payload = json.loads(to_json(model.model_dump(), inf_nan_mode="strings"))
    Path(file_path).write_text(render_json(payload, indent=indent), encoding="utf-8")
