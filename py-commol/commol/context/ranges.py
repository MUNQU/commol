"""The range rules of the settings, which the engine holds.

A model of a setting declares only its type. `RangeChecked` gives a model the
check of the engine for its object (``commol_rs.core.check_ranges``), which
runs when the model is built: a value outside its range raises ``ValueError``
with the message of the engine, which names the setting and its range.
"""

from collections import defaultdict
from typing import ClassVar, Self

from pydantic import BaseModel, model_validator

from commol.commol_rs import _commol_rs as commol_rs


def _ruled_settings() -> dict[str, tuple[str, ...]]:
    """The settings that have a range rule, per object of the engine."""
    settings: defaultdict[str, list[str]] = defaultdict(list)
    for owner, name, _outside, _message in commol_rs.core.range_rules():
        settings[owner].append(name)
    return {owner: tuple(names) for owner, names in settings.items()}


RULED_SETTINGS = _ruled_settings()
"""The settings with a range rule of each object, as the engine names it."""


class RangeChecked(BaseModel):
    """
    A model whose settings the engine checks with the range rule of each.

    A model that lists `RangeChecked` among its bases is an object of the
    engine, of its own name; a subclass of it keeps the rules of that object.
    """

    _range_object: ClassVar[str]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Name the object of the engine of a model that lists `RangeChecked`."""
        super().__pydantic_init_subclass__(**kwargs)
        if RangeChecked in cls.__bases__:
            cls._range_object = cls.__name__

    @model_validator(mode="after")
    def check_ranges(self) -> Self:
        """
        Check each setting that has a range rule with the rule of the engine.

        Returns
        -------
        Self
            The model.

        Raises
        ------
        ValueError
            With the message of the engine, if a setting is outside its range.
        """
        names = RULED_SETTINGS[self._range_object]
        commol_rs.core.check_ranges(
            self._range_object, {name: getattr(self, name) for name in names}
        )
        return self
