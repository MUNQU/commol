"""
Small models that many tests share.

Each factory returns a ``ModelBuilder`` with its initial conditions set, so a
test can add transitions and parameters before it calls ``build``. It cannot
add bins or stratifications afterwards: the initial conditions name them all.

Every keyword overrides one part of the default model, so a test names only
what it changes. The factories share these keywords:

- ``name``: the model name; ``print_equations`` prints it.
- ``parameters``: id to value, in declaration order. ``None`` leaves a value
  for calibration; a string is a formula. It replaces every default parameter.
- ``fractions``: bin id to its share of the total; ``None`` leaves a
  share for calibration.
- ``total``: the total of the initial conditions.
- ``stratifications``: see ``Stratifications``; ``conditions``: see
  ``Conditions``.
- ``compartment_fractions``: per-compartment overrides, as in
  ``set_initial_conditions``.
- ``bin_unit``, ``units`` and ``step_unit``: the unit of every bin, of each
  parameter by id, and of one step.
- A rate keyword (``ab``, ``rate``, …): ``None`` leaves that
  transition out.
- ``accumulator``: the id of an accumulator that counts the first transition.
"""

from collections.abc import Mapping

from commol import CompartmentalStructure, ModelBuilder, OdeSolver
from commol.api.model_builder import (
    BinFractionDict,
    CompartmentFractionDict,
    StratificationFractionsDict,
)
from commol.api.patterns import AxisPattern
from commol.context.model import Model

TIGHT = OdeSolver(rtol=1e-10, atol=1e-8)
"""Tolerances tight enough for closed-form checks."""

Parameters = Mapping[str, float | str | None]
Rate = str | float | AxisPattern | None
Stratifications = Mapping[str, Mapping[str, float]]
"""Stratification id to its categories and their fractions, in declaration order."""
Conditions = Mapping[str, Mapping[str, str]]
"""Stratification id to the stratification and category it is conditional on."""


def build(builder: ModelBuilder) -> Model[CompartmentalStructure]:
    """Build a difference-equation model."""
    return builder.build(engine="DifferenceEquations")


def build_ode(
    builder: ModelBuilder, settings: OdeSolver | None = TIGHT
) -> Model[CompartmentalStructure]:
    """Build a differential-equation model with `settings`."""
    return builder.build(engine="DifferentialEquations", settings=settings)


def stratify(
    builder: ModelBuilder,
    stratifications: Stratifications,
    conditions: Conditions | None = None,
) -> list[StratificationFractionsDict]:
    """
    Add the stratifications to the builder.

    Returns
    -------
    list[StratificationFractionsDict]
        The fractions of the stratifications, for ``set_initial_conditions``.
    """
    conditions = conditions or {}
    fractions: list[StratificationFractionsDict] = []
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
        fractions.append(
            {
                "stratification": stratification_id,
                "fractions": [
                    {"category": category, "fraction": fraction}
                    for category, fraction in categories.items()
                ],
            }
        )
    return fractions


def _finish(
    builder: ModelBuilder,
    *,
    parameters: Parameters,
    units: Mapping[str, str],
    total: float,
    fractions: Mapping[str, float | None],
    stratifications: Stratifications | None,
    conditions: Conditions | None,
    compartment_fractions: list[CompartmentFractionDict] | None,
) -> ModelBuilder:
    for parameter_id, value in parameters.items():
        builder.add_parameter(
            id=parameter_id, value=value, unit=units.get(parameter_id)
        )
    bin_fractions: list[BinFractionDict] = [
        {"bin": bin_id, "fraction": fraction} for bin_id, fraction in fractions.items()
    ]
    return builder.set_initial_conditions(
        total=total,
        bin_fractions=bin_fractions,
        stratification_fractions=stratify(builder, stratifications or {}, conditions),
        compartment_fractions=compartment_fractions,
    )


def abc_builder(
    *,
    name: str = "ABC",
    parameters: Parameters | None = None,
    ab: Rate = "k1 * A * B / total",
    bc: Rate = "k2 * B",
    total: float = 1000,
    fractions: Mapping[str, float | None] | None = None,
    stratifications: Stratifications | None = None,
    conditions: Conditions | None = None,
    bin_unit: str | None = None,
    step_unit: str | None = None,
    units: Mapping[str, str] | None = None,
    accumulator: str | None = None,
) -> ModelBuilder:
    """
    A → B → C: ``ab`` at ``k1 * A * B / total``, ``bc`` at ``k2 * B``.

    The defaults are ``k1 = 0.3``, ``k2 = 0.1``, a total of 1000, 99 % in A
    and 1 % in B. See the module docstring for the keywords.
    """
    builder = ModelBuilder(
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
        total=total,
        fractions=fractions
        if fractions is not None
        else {"A": 0.99, "B": 0.01, "C": 0.0},
        stratifications=stratifications,
        conditions=conditions,
        compartment_fractions=None,
    )


def abcd_builder(
    *,
    name: str = "ABCD",
    parameters: Parameters | None = None,
    ab: Rate = "k1 * A * C / total",
    bc: Rate = "k3 * B",
    cd: Rate = "k2 * C",
    total: float = 1000,
    fractions: Mapping[str, float | None] | None = None,
    bin_unit: str | None = None,
    step_unit: str | None = None,
    units: Mapping[str, str] | None = None,
) -> ModelBuilder:
    """
    A → B → C → D: ``ab`` at ``k1 * A * C / total``, ``bc`` at ``k3 * B``,
    ``cd`` at ``k2 * C``.

    The defaults are ``k1 = 0.5``, ``k3 = 0.2``, ``k2 = 0.1``, a total of
    1000, 98 % in A, 1 % in B and 1 % in C. See the module docstring for the
    keywords.
    """
    builder = ModelBuilder(
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
        total=total,
        fractions=fractions
        if fractions is not None
        else {"A": 0.98, "B": 0.01, "C": 0.01, "D": 0.0},
        stratifications=None,
        conditions=None,
        compartment_fractions=None,
    )


def a_to_b_builder(
    *,
    name: str = "AB",
    parameters: Parameters | None = None,
    rate: Rate = "k",
    transition_id: str = "t_ab",
    per_compartment: bool = False,
    total: float = 1000,
    fractions: Mapping[str, float | None] | None = None,
    stratifications: Stratifications | None = None,
    conditions: Conditions | None = None,
    compartment_fractions: list[CompartmentFractionDict] | None = None,
    accumulator: str | None = None,
    bin_unit: str | None = None,
    step_unit: str | None = None,
    units: Mapping[str, str] | None = None,
) -> ModelBuilder:
    """
    One transition, ``transition_id``, from bin A to bin B.

    The default rate ``k = 0.1`` is proportional, so ``0.1 * A`` moves each
    step, and the total of 1000 starts in A. ``per_compartment`` is passed to
    ``add_transition``. See the module docstring for the other keywords.
    """
    builder = ModelBuilder(
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
        total=total,
        fractions=fractions if fractions is not None else {"A": 1.0, "B": 0.0},
        stratifications=stratifications,
        conditions=conditions,
        compartment_fractions=compartment_fractions,
    )


def override_builder(fraction: float = 0.3) -> ModelBuilder:
    """
    ``a_to_b_builder`` split into halves ``g1`` and ``g2``, with ``A_g1`` set.

    Without the override every compartment of A holds 500. The override sets
    ``A_g1`` to ``fraction`` of the total; ``A_g2`` keeps its 500.
    """
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        stratifications={"group": {"g1": 0.5, "g2": 0.5}},
        compartment_fractions=[{"compartment": "A_g1", "fraction": fraction}],
    )


def shared_category_builder(
    *,
    group: tuple[float, float] = (0.5, 0.5),
    level: tuple[float, float] = (0.5, 0.5),
    compartment_fractions: list[CompartmentFractionDict] | None = None,
) -> ModelBuilder:
    """
    One bin A, stratified by ``group`` and ``level``, which share the
    categories ``c1`` and ``c2``; so a bare ``c1`` is ambiguous and needs
    ``group:c1`` or ``level:c1``. Each tuple gives the fractions of ``c1`` and
    ``c2``.
    """
    builder = ModelBuilder(name="SharedCategory", version="1.0")
    builder.add_bin(id="A", name="A")
    builder.add_transition(
        id="flow", source=["A"], target=[], rate="k * A", per_compartment=True
    )
    return _finish(
        builder,
        parameters={"k": 0.1},
        units={},
        total=1000,
        fractions={"A": 1.0},
        stratifications={
            "group": {"c1": group[0], "c2": group[1]},
            "level": {"c1": level[0], "c2": level[1]},
        },
        conditions=None,
        compartment_fractions=compartment_fractions,
    )


def conditional_pathway_builder() -> ModelBuilder:
    """
    Bins A and B, where only group ``g2`` moves from A to B.

    ``type`` splits only ``g2``, and the override pins ``B_g1`` to zero, so
    the compartments and their initial amounts are::

        A_g1 450, A_g2_t1 225, A_g2_t2 225,
        B_g1 0,   B_g2_t1 25,  B_g2_t2 25
    """
    builder = ModelBuilder(name="ConditionalPathway", version="1.0")
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
        total=1000,
        fractions={"A": 0.9, "B": 0.1},
        stratifications={
            "group": {"g1": 0.5, "g2": 0.5},
            "type": {"t1": 0.5, "t2": 0.5},
        },
        conditions={"type": {"group": "g2"}},
        compartment_fractions=[{"compartment": "B_g1", "fraction": 0.0}],
    )
