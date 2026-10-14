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
- ``fractions``: bin id to its share of the population; ``None`` leaves a
  share for calibration.
- ``population``: the population size.
- ``stratifications``: see ``Stratifications``; ``conditions``: see
  ``Conditions``.
- ``compartment_fractions``: per-compartment overrides, as in
  ``set_initial_conditions``.
- ``bin_unit`` and ``units``: the unit of every bin, and of each parameter by
  id.
- A rate keyword (``infection``, ``rate``, …): ``None`` leaves that
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
    population: int,
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
        population_size=population,
        bin_fractions=bin_fractions,
        stratification_fractions=stratify(builder, stratifications or {}, conditions),
        compartment_fractions=compartment_fractions,
    )


def sir_builder(
    *,
    name: str = "SIR",
    parameters: Parameters | None = None,
    infection: Rate = "beta * S * I / N",
    recovery: Rate = "gamma * I",
    population: int = 1000,
    fractions: Mapping[str, float | None] | None = None,
    stratifications: Stratifications | None = None,
    conditions: Conditions | None = None,
    bin_unit: str | None = None,
    units: Mapping[str, str] | None = None,
    accumulator: str | None = None,
) -> ModelBuilder:
    """
    S → I → R: ``infection`` at ``beta * S * I / N``, ``recovery`` at ``gamma * I``.

    The defaults are ``beta = 0.3``, ``gamma = 0.1``, 1000 people, 99 % S and
    1 % I. See the module docstring for the keywords.
    """
    builder = ModelBuilder(name=name, version="1.0", bin_unit=bin_unit)
    builder.add_bin(id="S", name="Susceptible")
    builder.add_bin(id="I", name="Infected")
    builder.add_bin(id="R", name="Recovered")
    if accumulator is not None:
        builder.add_accumulator(id=accumulator, name=accumulator)
    if infection is not None:
        builder.add_transition(
            id="infection",
            source=["S"],
            target=["I"],
            rate=infection,
            accumulators=[accumulator] if accumulator else None,
        )
    if recovery is not None:
        builder.add_transition(id="recovery", source=["I"], target=["R"], rate=recovery)
    return _finish(
        builder,
        parameters={"beta": 0.3, "gamma": 0.1} if parameters is None else parameters,
        units=units or {},
        population=population,
        fractions=fractions
        if fractions is not None
        else {"S": 0.99, "I": 0.01, "R": 0.0},
        stratifications=stratifications,
        conditions=conditions,
        compartment_fractions=None,
    )


def seir_builder(
    *,
    name: str = "SEIR",
    parameters: Parameters | None = None,
    exposure: Rate = "beta * S * I / N",
    progression: Rate = "sigma * E",
    recovery: Rate = "gamma * I",
    population: int = 1000,
    fractions: Mapping[str, float | None] | None = None,
    bin_unit: str | None = None,
    units: Mapping[str, str] | None = None,
) -> ModelBuilder:
    """
    S → E → I → R: ``exposure``, ``progression`` and ``recovery``.

    The defaults are ``beta = 0.5``, ``sigma = 0.2``, ``gamma = 0.1``, 1000
    people, 98 % S, 1 % E and 1 % I. See the module docstring for the keywords.
    """
    builder = ModelBuilder(name=name, version="1.0", bin_unit=bin_unit)
    for bin_id, bin_name in (
        ("S", "Susceptible"),
        ("E", "Exposed"),
        ("I", "Infected"),
        ("R", "Recovered"),
    ):
        builder.add_bin(id=bin_id, name=bin_name)
    for transition_id, source, target, rate in (
        ("exposure", "S", "E", exposure),
        ("progression", "E", "I", progression),
        ("recovery", "I", "R", recovery),
    ):
        if rate is not None:
            builder.add_transition(
                id=transition_id, source=[source], target=[target], rate=rate
            )
    return _finish(
        builder,
        parameters=(
            {"beta": 0.5, "sigma": 0.2, "gamma": 0.1}
            if parameters is None
            else parameters
        ),
        units=units or {},
        population=population,
        fractions=fractions
        if fractions is not None
        else {"S": 0.98, "E": 0.01, "I": 0.01, "R": 0.0},
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
    population: int = 1000,
    fractions: Mapping[str, float | None] | None = None,
    stratifications: Stratifications | None = None,
    conditions: Conditions | None = None,
    compartment_fractions: list[CompartmentFractionDict] | None = None,
    accumulator: str | None = None,
    bin_unit: str | None = None,
    units: Mapping[str, str] | None = None,
) -> ModelBuilder:
    """
    One transition, ``transition_id``, from bin A to bin B.

    The default rate ``k = 0.1`` is per capita, so ``0.1 * A`` moves each step,
    and the 1000 people start in A. ``per_compartment`` is passed to
    ``add_transition``. See the module docstring for the other keywords.
    """
    builder = ModelBuilder(name=name, version="1.0", bin_unit=bin_unit)
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
        population=population,
        fractions=fractions if fractions is not None else {"A": 1.0, "B": 0.0},
        stratifications=stratifications,
        conditions=conditions,
        compartment_fractions=compartment_fractions,
    )


def override_builder(fraction: float = 0.3) -> ModelBuilder:
    """
    ``a_to_b_builder`` split into halves ``g1`` and ``g2``, with ``A_g1`` set.

    Without the override every compartment of A holds 500. The override sets
    ``A_g1`` to ``fraction`` of the population; ``A_g2`` keeps its 500.
    """
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        stratifications={"group": {"g1": 0.5, "g2": 0.5}},
        compartment_fractions=[{"compartment": "A_g1", "fraction": fraction}],
    )


def shared_category_builder(
    *,
    age: tuple[float, float] = (0.5, 0.5),
    sex: tuple[float, float] = (0.5, 0.5),
    compartment_fractions: list[CompartmentFractionDict] | None = None,
) -> ModelBuilder:
    """
    One bin A, stratified by ``age`` and ``sex``, which share the categories
    ``y`` and ``o``; so a bare ``y`` is ambiguous and needs ``age:y`` or
    ``sex:y``. Each tuple gives the fractions of ``y`` and ``o``.
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
        population=1000,
        fractions={"A": 1.0},
        stratifications={
            "age": {"y": age[0], "o": age[1]},
            "sex": {"y": sex[0], "o": sex[1]},
        },
        conditions=None,
        compartment_fractions=compartment_fractions,
    )


def conditional_pathway_builder() -> ModelBuilder:
    """
    Bins S and IH, where only the ``oe60`` stratum moves from S to IH.

    ``care`` splits only ``oe60``, and the override pins ``IH_y60`` to zero, so
    the compartments and their initial populations are::

        S_y60 450, S_oe60_home 225, S_oe60_hosp 225,
        IH_y60 0,  IH_oe60_home 25, IH_oe60_hosp 25
    """
    builder = ModelBuilder(name="ConditionalPathway", version="1.0")
    builder.add_bin(id="S", name="Susceptible")
    builder.add_bin(id="IH", name="Hospitalized")
    builder.add_transition(
        id="hosp",
        source=["S"],
        target=["IH"],
        rate="0.0",
        stratified_rates=[
            {
                "conditions": [{"stratification": "age", "category": "oe60"}],
                "rate": "beta * S",
            }
        ],
    )
    return _finish(
        builder,
        parameters={"beta": 0.01},
        units={},
        population=1000,
        fractions={"S": 0.9, "IH": 0.1},
        stratifications={
            "age": {"y60": 0.5, "oe60": 0.5},
            "care": {"home": 0.5, "hosp": 0.5},
        },
        conditions={"care": {"age": "oe60"}},
        compartment_fractions=[{"compartment": "IH_y60", "fraction": 0.0}],
    )
