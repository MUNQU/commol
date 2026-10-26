"""Models of every branch of a description, for the snapshots of its
sections and renderings."""

from commol import Aggregate, Complete, Groups, StratificationCondition, StratifiedRate
from commol._model.model import Model
from commol._model.parameter import SeriesValue
from commol.engines.engine import StochasticEventsSettings
from tests.support.kinds import UnstratifiedVariant
from tests.support.models import FactoryBuilder, abc_builder, build


def every_flow_builder() -> FactoryBuilder:
    """
    A pair of sources into one target, a source that is also a target, a
    split into two targets, a flow into its own source, an inflow, an
    impulse at positions before the other flows, an outflow, a category
    transition of stratified rates, a transition with no rate, and an impulse
    at a period.
    """
    builder = (
        FactoryBuilder(name="Every flow", version="1.0")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_parameter(id="k1", value=0.001)
        .add_parameter(id="b", value=1.0)
        .add_transition(
            id="pair", source=["A", "B"], target=["C", "C"], rate="k1 * A * B"
        )
        .add_transition(
            id="spread",
            source=["A", "B"],
            target=["B", "B"],
            rate="k1 * A * B / total",
        )
        .add_transition(id="split", source=["A"], target=["B", "C"], rate="0.05")
        .add_transition(id="loop", source=["B"], target=["B"], rate="0.1")
        .add_transition(id="inflow", source=[], target=["A"], rate="b")
        .add_transition(
            id="dose", source=["A"], target=["B"], rate="0.1", impulse_at=[1, 3]
        )
        .add_transition(id="outflow", source=["C"], target=[], rate="0.01")
        .add_transition(
            id="move",
            source=["g1"],
            target=["g2"],
            rate=None,
            stratified_rates=[
                StratifiedRate(
                    conditions=[
                        StratificationCondition(stratification="group", category="g1")
                    ],
                    rate="0.02",
                )
            ],
        )
        .add_transition(id="pending", source=["A"], target=["C"], rate=None)
        .add_transition(
            id="boost", source=["B"], target=["C"], rate="0.5", impulse_every=(7, 0)
        )
    )
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": 500, "B": 500, "C": 0},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
        )
    )
    return builder


def every_flow() -> Model:
    """The model of every kind of flow, of difference equations."""
    return build(every_flow_builder())


def variant_with_a_series() -> Model:
    """A variant of its structure, a series parameter, days and no description."""
    builder = abc_builder(step_unit="day")
    builder.add_parameter(
        id="s", value=SeriesValue(data=[(0, 1.0), (5, 2.5)], mode="linear")
    )
    return builder.build(
        UnstratifiedVariant(bin_amounts={"A": 990, "B": 10, "C": 0}),
        engine="DifferenceEquations",
    )


def stochastic_with_settings() -> Model:
    """
    A description, settings of its engine, neighbors, no parameters, and
    transitions of two sources, of none, to none and at an impulse.
    """
    builder = (
        FactoryBuilder(name="At rest", description="Two bins at rest.")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_transition(id="ab", source=["A", "B"], target=["B", "B"], rate="0 * A * B")
        .add_transition(id="in", source=[], target=["A", "B"], rate="0")
        .add_transition(id="out", source=["B"], target=[], rate="0")
        .add_transition(id="dose", source=["A"], target=["B"], rate="0", impulse_at=[1])
    )
    builder.set_structure(Aggregate(bin_amounts={"A": 5, "B": 5}, neighbors=Complete()))
    return builder.build(
        builder.structure,
        engine="StochasticEvents",
        settings=StochasticEventsSettings(method="tau_leaping"),
    )


def weighted_groups() -> Model:
    """
    Bins in units, one named with quotes and a backslash, an accumulator, a
    stratification split on a category, weighted groups of neighbors, a
    compartment override, a bin and a parameter left to calibration, and a
    description with a ``|``.
    """
    builder = (
        FactoryBuilder(name="Weighted", bin_unit="item")
        .add_bin(id="A", name='State "A" \\ start')
        .add_bin(id="B", name="State B")
        .add_accumulator(id="moved", name="Moved")
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_stratification(
            id="tier",
            categories=["t1", "t2"],
            conditions=[StratificationCondition(stratification="group", category="g1")],
        )
        .add_parameter(id="k", value=0.1, description="rate | per step")
        .add_parameter(id="s", value=None)
        .add_transition(
            id="ab",
            source=["A"],
            target=["B"],
            rate="s * k * neighbors_B / neighbors",
            accumulators=["moved"],
        )
    )
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": None, "B": 400},
            stratification_fractions={
                "group": {"g1": 0.25, "g2": 0.75},
                "tier": {"t1": 0.5, "t2": 0.5},
            },
            compartment_amounts={"B_g2": 200},
            neighbors=Groups(
                "group",
                weights={"g1": {"g1": 1.0, "g2": 0.5}, "g2": {"g1": 0.5, "g2": 1.0}},
            ),
        )
    )
    return build(builder)


def identity_groups() -> Model:
    """Groups of neighbors without weights, and a split on two categories."""
    builder = (
        FactoryBuilder(name="Identity")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_stratification(
            id="tier",
            categories=["t1", "t2"],
            conditions=[StratificationCondition(stratification="group", category="g1")],
        )
        .add_stratification(
            id="deep",
            categories=["d1", "d2"],
            conditions=[
                StratificationCondition(stratification="group", category="g1"),
                StratificationCondition(stratification="tier", category="t1"),
            ],
        )
        .add_parameter(id="k", value=0.1)
        .add_transition(id="ab", source=["A"], target=["B"], rate="k * neighbors_B")
    )
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": 50, "B": 50},
            stratification_fractions={
                "group": {"g1": 0.5, "g2": 0.5},
                "tier": {"t1": 0.5, "t2": 0.5},
                "deep": {"d1": 0.5, "d2": 0.5},
            },
            neighbors=Groups("group"),
        )
    )
    return build(builder)
