"""Run a model in one call."""

from commol import Aggregate, Simulation
from commol._model.model import Model

from .models import FactoryBuilder, build


def run(model: Model | FactoryBuilder, steps: int) -> dict[str, list[float]]:
    """Simulate ``steps`` steps; a builder is built first."""
    if isinstance(model, FactoryBuilder):
        model = build(model)
    return Simulation(model).run(steps)


def initial_state(model: Model[Aggregate]) -> dict[str, float]:
    """The amount of each compartment the engine starts from."""
    engine = Simulation(model).engine
    return dict(zip(engine.compartments, engine.state, strict=True))
