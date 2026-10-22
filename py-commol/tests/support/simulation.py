"""Run a model in one call."""

from commol import Aggregate, Simulation
from commol.context.model import Model

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


_WORD = 2**64 - 1


def _splitmix64(state: int) -> int:
    """The first number of SplitMix64 from ``state``, as `commol-random` has it."""
    z = (state + 0x9E3779B97F4A7C15) & _WORD
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _WORD
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _WORD
    return z ^ (z >> 31)


def draws_seed(seed: int) -> int:
    """
    The seed the replicates of the draws of a propagation derive from:
    ``stream_seed(seed, DRAWS_STREAM)`` of `crates/commol-random`, so
    replicate ``i`` of draw ``j`` of ``n`` replicates is replicate
    ``j * n + i`` of ``Simulation.replicates`` with this seed.
    """
    return _splitmix64(seed ^ ((4 * 0xD1B54A32D192ED03) & _WORD))
