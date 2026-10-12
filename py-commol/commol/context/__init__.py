"""The Pydantic model definition: validation, serialization and updates."""

from .initial_conditions import CompartmentFraction
from .model import Model


__all__ = [
    "CompartmentFraction",
    "Model",
]
