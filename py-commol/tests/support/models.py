"""
Small models that many tests share.

Each factory returns a `FactoryBuilder`: a ``ModelBuilder`` that writes the
process, and the ``Aggregate`` that lays it out (``builder.structure``). A
test can add transitions and parameters before it calls ``build``. It cannot
add bins or stratifications afterwards: the structure names them all.

Every keyword overrides one part of the default model, so a test names only
what it changes. The factories share these keywords:

- ``name``: the model name; ``print_equations`` prints it.
- ``parameters``: id to value, in declaration order. ``None`` leaves a value
  for calibration; a string is a formula. It replaces every default parameter.
- ``amounts``: bin id to its initial amount; ``None`` leaves an amount for
  calibration.
- ``stratifications``: see ``Stratifications``; ``conditions``: see
  ``Conditions``.
- ``compartment_amounts``: per-compartment overrides, as ``Aggregate``
  takes them.
- ``bin_unit``, ``units`` and ``step_unit``: the unit of every bin, of each
  parameter by id, and of one step.
- A rate keyword (``ab``, ``rate``, …): ``None`` leaves that
  transition out.
- ``accumulator``: the id of an accumulator that counts the first transition.
"""

from collections.abc import Mapping
from typing import Self

from commol import (
    Aggregate,
    ModelBuilder,
    ODESolver,
    StochasticEventsSettings,
    StochasticStepsSettings,
)
from commol.context.engine import EngineName
from commol.api.patterns import AxisPattern
from commol.context.model import Model
from commol.context.neighbors import Neighbors

TIGHT = ODESolver(rtol=1e-10, atol=1e-8)
"""Tolerances tight enough for closed-form checks."""

Parameters = Mapping[str, float | str | None]
Rate = str | float | AxisPattern | None
Stratifications = Mapping[str, Mapping[str, float]]
"""Stratification id to its categories and their fractions, in declaration order."""
Conditions = Mapping[str, Mapping[str, str]]
"""Stratification id to the stratification and category it is conditional on."""


class FactoryBuilder(ModelBuilder):
    """A builder of a process, and the aggregate structure its factory chose."""

    structure: Aggregate

    def set_structure(self, structure: Aggregate) -> Self:
        """Choose the structure `build` and `build_ode` lay the process out with."""
        self.structure = structure
        return self


def build(builder: FactoryBuilder) -> Model[Aggregate]:
    """Build a difference-equation model."""
    return builder.build(builder.structure, engine="DifferenceEquations")


def build_ode(
    builder: FactoryBuilder, settings: ODESolver | None = TIGHT
) -> Model[Aggregate]:
    """Build a differential-equation model with `settings`."""
    return builder.build(
        builder.structure, engine="DifferentialEquations", settings=settings
    )


STOCHASTIC_ENGINES: tuple[EngineName, ...] = ("StochasticEvents", "StochasticSteps")
"""The stochastic engines of the aggregate kind."""

MEAN_FIELD: dict[EngineName, EngineName] = {
    "StochasticEvents": "DifferentialEquations",
    "StochasticSteps": "DifferenceEquations",
}
"""The deterministic engine each stochastic engine has as its mean-field limit."""


def build_stochastic(
    builder: FactoryBuilder,
    engine: EngineName = "StochasticEvents",
    settings: StochasticEventsSettings | StochasticStepsSettings | None = None,
) -> Model[Aggregate]:
    """Build a model of a stochastic engine with `settings`."""
    return builder.build(builder.structure, engine=engine, settings=settings)


def stratify(
    builder: ModelBuilder,
    stratifications: Stratifications,
    conditions: Conditions | None = None,
) -> dict[str, dict[str, float]]:
    """
    Add the stratifications to the builder.

    Returns
    -------
    dict[str, dict[str, float]]
        The fractions of the stratifications, as ``Aggregate`` takes them.
    """
    conditions = conditions or {}
    fractions: dict[str, dict[str, float]] = {}
    for stratification_id, categories in stratifications.items():
        builder.add_stratification(
            id=stratification_id,
            categories=list(categories),
            conditions=[
                {"stratification": parent, "category": category}
                for parent, category in conditions.get(stratification_id, {}).items()
            ]
            or None,
        )
        fractions[stratification_id] = dict(categories)
    return fractions


def _finish(
    builder: FactoryBuilder,
    *,
    parameters: Parameters,
    units: Mapping[str, str],
    amounts: Mapping[str, float | None],
    stratifications: Stratifications | None,
    conditions: Conditions | None,
    compartment_amounts: Mapping[str, float] | None,
    neighbors: Neighbors | None = None,
) -> FactoryBuilder:
    for parameter_id, value in parameters.items():
        builder.add_parameter(
            id=parameter_id, value=value, unit=units.get(parameter_id)
        )
    builder.structure = Aggregate(
        bin_amounts=dict(amounts),
        stratification_fractions=stratify(builder, stratifications or {}, conditions),
        compartment_amounts=dict(compartment_amounts or {}),
        neighbors=neighbors,
    )
    return builder


def abc_builder(
    *,
    name: str = "ABC",
    parameters: Parameters | None = None,
    ab: Rate = "k1 * A * B / total",
    bc: Rate = "k2 * B",
    amounts: Mapping[str, float | None] | None = None,
    stratifications: Stratifications | None = None,
    conditions: Conditions | None = None,
    bin_unit: str | None = None,
    step_unit: str | None = None,
    units: Mapping[str, str] | None = None,
    accumulator: str | None = None,
) -> FactoryBuilder:
    """
    A → B → C: ``ab`` at ``k1 * A * B / total``, ``bc`` at ``k2 * B``.

    The defaults are ``k1 = 0.3``, ``k2 = 0.1``, 990 in A and 10 in B. See the
    module docstring for the keywords.
    """
    builder = FactoryBuilder(
        name=name, version="1.0", bin_unit=bin_unit, step_unit=step_unit
    )
    builder.add_bin(id="A", name="State A")
    builder.add_bin(id="B", name="State B")
    builder.add_bin(id="C", name="State C")
    if accumulator is not None:
        builder.add_accumulator(id=accumulator, name=accumulator)
    if ab is not None:
        builder.add_transition(
            id="ab",
            source=["A"],
            target=["B"],
            rate=ab,
            accumulators=[accumulator] if accumulator else None,
        )
    if bc is not None:
        builder.add_transition(id="bc", source=["B"], target=["C"], rate=bc)
    return _finish(
        builder,
        parameters={"k1": 0.3, "k2": 0.1} if parameters is None else parameters,
        units=units or {},
        amounts=amounts if amounts is not None else {"A": 990, "B": 10, "C": 0},
        stratifications=stratifications,
        conditions=conditions,
        compartment_amounts=None,
    )


def abcd_builder(
    *,
    name: str = "ABCD",
    parameters: Parameters | None = None,
    ab: Rate = "k1 * A * C / total",
    bc: Rate = "k3 * B",
    cd: Rate = "k2 * C",
    amounts: Mapping[str, float | None] | None = None,
    bin_unit: str | None = None,
    step_unit: str | None = None,
    units: Mapping[str, str] | None = None,
) -> FactoryBuilder:
    """
    A → B → C → D: ``ab`` at ``k1 * A * C / total``, ``bc`` at ``k3 * B``,
    ``cd`` at ``k2 * C``.

    The defaults are ``k1 = 0.5``, ``k3 = 0.2``, ``k2 = 0.1``, 980 in A, 10
    in B and 10 in C. See the module docstring for the
    keywords.
    """
    builder = FactoryBuilder(
        name=name, version="1.0", bin_unit=bin_unit, step_unit=step_unit
    )
    for bin_id in ("A", "B", "C", "D"):
        builder.add_bin(id=bin_id, name=bin_id)
    for transition_id, source, target, rate in (
        ("ab", "A", "B", ab),
        ("bc", "B", "C", bc),
        ("cd", "C", "D", cd),
    ):
        if rate is not None:
            builder.add_transition(
                id=transition_id, source=[source], target=[target], rate=rate
            )
    return _finish(
        builder,
        parameters=(
            {"k1": 0.5, "k3": 0.2, "k2": 0.1} if parameters is None else parameters
        ),
        units=units or {},
        amounts=(
            amounts if amounts is not None else {"A": 980, "B": 10, "C": 10, "D": 0}
        ),
        stratifications=None,
        conditions=None,
        compartment_amounts=None,
    )


def a_to_b_builder(
    *,
    name: str = "AB",
    parameters: Parameters | None = None,
    rate: Rate = "k",
    transition_id: str = "t_ab",
    per_compartment: bool = False,
    amounts: Mapping[str, float | None] | None = None,
    stratifications: Stratifications | None = None,
    conditions: Conditions | None = None,
    compartment_amounts: Mapping[str, float] | None = None,
    accumulator: str | None = None,
    bin_unit: str | None = None,
    step_unit: str | None = None,
    units: Mapping[str, str] | None = None,
) -> FactoryBuilder:
    """
    One transition, ``transition_id``, from bin A to bin B.

    The default rate ``k = 0.1`` is proportional, so ``0.1 * A`` moves each
    step, and 1000 starts in A. ``per_compartment`` is passed to
    ``add_transition``. See the module docstring for the other keywords.
    """
    builder = FactoryBuilder(
        name=name, version="1.0", bin_unit=bin_unit, step_unit=step_unit
    )
    builder.add_bin(id="A", name="A")
    builder.add_bin(id="B", name="B")
    if accumulator is not None:
        builder.add_accumulator(id=accumulator, name=accumulator)
    builder.add_transition(
        id=transition_id,
        source=["A"],
        target=["B"],
        rate=rate,
        per_compartment=per_compartment or None,
        accumulators=[accumulator] if accumulator else None,
    )
    return _finish(
        builder,
        parameters={"k": 0.1} if parameters is None else parameters,
        units=units or {},
        amounts=amounts if amounts is not None else {"A": 1000, "B": 0},
        stratifications=stratifications,
        conditions=conditions,
        compartment_amounts=compartment_amounts,
    )


def override_builder(amount: float = 300) -> FactoryBuilder:
    """
    ``a_to_b_builder`` split into halves ``g1`` and ``g2``, with ``A_g1`` set.

    Without the override every compartment of A holds 500. The override sets
    ``A_g1`` to ``amount``; ``A_g2`` keeps its 500.
    """
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        stratifications={"group": {"g1": 0.5, "g2": 0.5}},
        compartment_amounts={"A_g1": amount},
    )


def shared_category_builder(
    *,
    group: tuple[float, float] = (0.5, 0.5),
    level: tuple[float, float] = (0.5, 0.5),
    compartment_amounts: Mapping[str, float] | None = None,
) -> FactoryBuilder:
    """
    One bin A, stratified by ``group`` and ``level``, which share the
    categories ``c1`` and ``c2``; so a bare ``c1`` is ambiguous and needs
    ``group:c1`` or ``level:c1``. Each tuple gives the fractions of ``c1`` and
    ``c2``.
    """
    builder = FactoryBuilder(name="SharedCategory", version="1.0")
    builder.add_bin(id="A", name="A")
    builder.add_transition(
        id="flow", source=["A"], target=[], rate="k * A", per_compartment=True
    )
    return _finish(
        builder,
        parameters={"k": 0.1},
        units={},
        amounts={"A": 1000},
        stratifications={
            "group": {"c1": group[0], "c2": group[1]},
            "level": {"c1": level[0], "c2": level[1]},
        },
        conditions=None,
        compartment_amounts=compartment_amounts,
    )


def conditional_pathway_builder() -> FactoryBuilder:
    """
    Bins A and B, where only group ``g2`` moves from A to B.

    ``type`` splits only ``g2``, and the override pins ``B_g1`` to zero, so
    the compartments and their initial amounts are::

        A_g1 450, A_g2_t1 225, A_g2_t2 225,
        B_g1 0,   B_g2_t1 25,  B_g2_t2 25
    """
    builder = FactoryBuilder(name="ConditionalPathway", version="1.0")
    builder.add_bin(id="A", name="A")
    builder.add_bin(id="B", name="B")
    builder.add_transition(
        id="ab",
        source=["A"],
        target=["B"],
        rate="0.0",
        stratified_rates=[
            {
                "conditions": [{"stratification": "group", "category": "g2"}],
                "rate": "k1 * A",
            }
        ],
    )
    return _finish(
        builder,
        parameters={"k1": 0.01},
        units={},
        amounts={"A": 900, "B": 100},
        stratifications={
            "group": {"g1": 0.5, "g2": 0.5},
            "type": {"t1": 0.5, "t2": 0.5},
        },
        conditions={"type": {"group": "g2"}},
        compartment_amounts={"B_g1": 0.0},
    )
