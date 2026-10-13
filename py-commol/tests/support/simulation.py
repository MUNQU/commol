"""Run a model in one call."""

from commol import ModelBuilder, Simulation
from commol.context.model import Model

from .models import build


def run(model: Model | ModelBuilder, steps: int) -> dict[str, list[float]]:
    """Simulate ``steps`` steps; a builder is built first."""
    if isinstance(model, ModelBuilder):
        model = build(model)
    return Simulation(model).run(steps)


def initial_population(model: Model) -> dict[str, float]:
    """The population of each compartment the engine starts from."""
    engine = Simulation(model).engine
    return dict(zip(engine.compartments, engine.population, strict=True))
