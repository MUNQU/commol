"""
A hand computation of a difference-equation run, to check the engine against.

The engine computes every flow from the state at the start of a step, then
adds them all: ``x(t+1) = x(t) + Σ inflows(t) - Σ outflows(t)``. This module
does the same arithmetic in plain Python for the few compartments of a test
model, so a test can compare whole trajectories instead of a single number.
"""

from collections.abc import Callable, Mapping, Sequence

State = Mapping[str, float]
FlowRate = Callable[[State, int], float]
Flow = tuple[str | None, str | None, FlowRate]
"""Source, target and the flow per step; ``None`` is outside the model."""


def total(state: State) -> float:
    """``total``: the sum of every compartment."""
    return sum(state.values())


def reference_run(
    initial: Mapping[str, float], flows: Sequence[Flow], steps: int
) -> dict[str, list[float]]:
    """
    Run the flows for ``steps`` steps from ``initial``.

    Parameters
    ----------
    initial : Mapping[str, float]
        The amount of each compartment at step 0.
    flows : Sequence[Flow]
        Each flow as ``(source, target, rate)``. ``rate(state, step)`` returns
        what moves in that step, so a proportional rate must
        multiply by the source itself.
    steps : int
        The number of steps.

    Returns
    -------
    dict[str, list[float]]
        The trajectory of each compartment, ``steps + 1`` values long, in the
        ``dict_of_lists`` format of ``Simulation.run``.
    """
    state = dict(initial)
    series = {name: [value] for name, value in state.items()}
    for step in range(steps):
        change = dict.fromkeys(state, 0.0)
        for source, target, rate in flows:
            moved = rate(state, step)
            if source is not None:
                change[source] -= moved
            if target is not None:
                change[target] += moved
        for name in state:
            state[name] += change[name]
            series[name].append(state[name])
    return series


def abc_flows(
    ab: FlowRate,
    bc: FlowRate | None = None,
) -> list[Flow]:
    """The flows of ``abc_builder``: A → B and, if given, B → C."""
    flows: list[Flow] = [("A", "B", ab)]
    if bc is not None:
        flows.append(("B", "C", bc))
    return flows


ABC_START = {"A": 990.0, "B": 10.0, "C": 0.0}
"""The initial state of ``abc_builder`` with its default total and fractions."""
