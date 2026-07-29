"""JSON rendering for model files."""

import json
import re

INLINE_ARRAY_MIN_LENGTH = 8
_INLINE_TOKEN = re.compile(r'"@@inline:(\d+)@@"')


def _is_number(value: object) -> bool:
    """True for a JSON number or null, excluding booleans."""
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _is_numeric_array(payload: object) -> bool:
    """True for a flat array of numbers, or an array of such arrays."""
    if not isinstance(payload, list) or not payload:
        return False
    if all(_is_number(item) for item in payload):
        return True
    return all(
        isinstance(item, list) and all(_is_number(entry) for entry in item)
        for item in payload
    )


def _inline_numeric_arrays(
    payload: object, inlined: list[str], min_length: int
) -> object:
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
    payload: object,
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
    payload : object
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
