"""Run a model in one call."""

from commol import CompartmentalStructure, ModelBuilder, Simulation
from commol.context.model import Model

from .models import build


def run(model: Model | ModelBuilder, steps: int) -> dict[str, list[float]]:
    """Simulate ``steps`` steps; a builder is built first."""
    if isinstance(model, ModelBuilder):
        model = build(model)
    return Simulation(model).run(steps)


def initial_state(model: Model[CompartmentalStructure]) -> dict[str, float]:
    """The amount of each compartment the engine starts from."""
    engine = Simulation(model).engine
    return dict(zip(engine.compartments, engine.state, strict=True))
