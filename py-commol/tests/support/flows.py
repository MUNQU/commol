"""
Models written as a list of flows between bins, from absolute initial values.

A reference test states a model as its source does: the initial value of every
bin and the rate of every flow. ``flow_model`` builds it, so a test does not
repeat the builder calls. ``increments`` reads the right-hand side of a model at
one state, so a test can compare it with the equations of its source.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from commol import ModelBuilder, OdeSolver, Simulation
from commol.context.compartmental import CompartmentalStructure
from commol.context.model import Model

from .models import TIGHT, build, build_ode

FINE = OdeSolver(rtol=TIGHT.rtol, atol=1e-12)
"""
``TIGHT`` with a smaller absolute tolerance, for models whose values are of
order one or below: the error of a value near zero stays below its own size.
"""


@dataclass(frozen=True)
class Flow:
    """
    One transition from ``source`` to ``target``; ``None`` is outside the model.

    Write ``rate`` as the flow itself: name the source compartment in it, as in
    ``"gamma * I"``, so the engine reads it as an absolute rate.
    """

    id: str
    source: str | None
    target: str | None
    rate: str
    impulse_every: tuple[float, float] | None = None
    accumulators: tuple[str, ...] = ()


def flow_model(
    values: Mapping[str, float],
    flows: Sequence[Flow],
    parameters: Mapping[str, float | str] | None = None,
    *,
    ode: bool = True,
    settings: OdeSolver = FINE,
    accumulators: Sequence[str] = (),
) -> Model[CompartmentalStructure]:
    """
    Build a model from the initial value of each bin and its flows.

    Parameters
    ----------
    values : Mapping[str, float]
        Bin id to its initial value, in declaration order. Their sum is the
        population size, so it must be a positive integer.
    flows : Sequence[Flow]
        The transitions.
    parameters : Mapping[str, float | str] | None
        Parameter id to its value or formula.
    ode : bool
        Build for differential equations with ``settings``; otherwise for
        difference equations.
    settings : OdeSolver
        The settings of the differential engine.
    accumulators : Sequence[str]
        Accumulator ids that the flows may name.

    Returns
    -------
    Model[CompartmentalStructure]
        The built model.
    """
    builder = ModelBuilder(name="reference", version="1.0")
    for bin_id in values:
        builder.add_bin(id=bin_id, name=bin_id)
    for accumulator in accumulators:
        builder.add_accumulator(id=accumulator, name=accumulator)
    for parameter_id, value in (parameters or {}).items():
        builder.add_parameter(id=parameter_id, value=value)
    for flow in flows:
        builder.add_transition(
            id=flow.id,
            source=[flow.source] if flow.source else [],
            target=[flow.target] if flow.target else [],
            rate=flow.rate,
            impulse_every=flow.impulse_every,
            accumulators=list(flow.accumulators) or None,
        )
    total = sum(values.values())
    # A total that is not an integer goes through as it is, so the builder
    # rejects it instead of this helper rounding it.
    population = int(total) if float(total).is_integer() else total
    builder.set_initial_conditions(
        population_size=population,
        bin_fractions=[
            {"bin": bin_id, "fraction": value / total if total else 0.0}
            for bin_id, value in values.items()
        ],
    )
    return build_ode(builder, settings) if ode else build(builder)


def increments(
    values: Mapping[str, float],
    flows: Sequence[Flow],
    parameters: Mapping[str, float | str] | None = None,
) -> dict[str, float]:
    """
    The right-hand side of the flows at the state ``values``.

    A difference-equation step adds the flows of the state it starts from, and
    both engines compile the same flows, so one step gives the derivatives of
    the differential model at that state.
    """
    results = Simulation(flow_model(values, flows, parameters, ode=False)).run(1)
    return {name: series[1] - series[0] for name, series in results.items()}
