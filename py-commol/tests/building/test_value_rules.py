"""The rules of the values of a model live in the engine: a value outside
its rule fails when its Python object is built, with the message of the
engine, and the model compiler gives the same message for the same value."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ValidationError

from commol import Aggregate, SeriesValue
from commol._model.engine_state import build_engine
from commol._model.transition import Transition
from tests.support import a_to_b_builder, build

type ModelJson = dict[str, dict]
"""A model as its JSON reads, by field."""


@dataclass(frozen=True)
class ValueCase:
    """A value outside its rule: the object that holds it, the message of the
    engine, and the change of a model JSON that sets it, when JSON can hold
    the value."""

    id: str
    build: Callable[[], BaseModel]
    message: str
    compiled: str | None = None
    write: Callable[[ModelJson], None] | None = None


def _transition(**impulse: object) -> Transition:
    return Transition.model_validate(
        {"id": "t_ab", "source": ["A"], "target": ["B"], "rate": "k", **impulse}
    )


def _set_impulse(**impulse: object) -> Callable[[ModelJson], None]:
    def write(model: ModelJson) -> None:
        model["process"]["transitions"][0].update(impulse)

    return write


def _set_amount(field: str, name: str, amount: float) -> Callable[[ModelJson], None]:
    def write(model: ModelJson) -> None:
        model["structure"][field] = {**model["structure"][field], name: amount}

    return write


def _set_fractions(g1: float, g2: float) -> Callable[[ModelJson], None]:
    def write(model: ModelJson) -> None:
        model["process"]["stratifications"].append(
            {"id": "group", "categories": ["g1", "g2"]}
        )
        model["structure"]["stratification_fractions"] = {"group": {"g1": g1, "g2": g2}}

    return write


def _fractions(g1: float, g2: float) -> Aggregate:
    return Aggregate(
        bin_amounts={"A": 1.0},
        stratification_fractions={"group": {"g1": g1, "g2": g2}},
    )


def _set_series(data: list[tuple[float, float]]) -> Callable[[ModelJson], None]:
    def write(model: ModelJson) -> None:
        model["parameters"].append(
            {"id": "s", "value": {"data": data, "mode": "linear"}}
        )

    return write


AMOUNT = "initial amount is a finite number of at least 0, got"
FRACTION = "fraction is a number from 0 to 1, got"

CASES = [
    ValueCase(
        "bin amount below 0",
        lambda: Aggregate(bin_amounts={"A": 1.0, "B": -1.0}),
        f"Bin 'B': {AMOUNT} -1",
        write=_set_amount("bin_amounts", "B", -1.0),
    ),
    ValueCase(
        "bin amount NaN",
        lambda: Aggregate(bin_amounts={"A": math.nan}),
        f"Bin 'A': {AMOUNT} NaN",
    ),
    ValueCase(
        "bin amount infinite",
        lambda: Aggregate(bin_amounts={"A": math.inf}),
        f"Bin 'A': {AMOUNT} inf",
    ),
    ValueCase(
        "compartment amount below 0",
        lambda: Aggregate(bin_amounts={"A": 1.0}, compartment_amounts={"A": -0.5}),
        f"Compartment 'A': {AMOUNT} -0.5",
        write=_set_amount("compartment_amounts", "A", -0.5),
    ),
    ValueCase(
        "compartment amount NaN",
        lambda: Aggregate(bin_amounts={"A": 1.0}, compartment_amounts={"A": math.nan}),
        f"Compartment 'A': {AMOUNT} NaN",
    ),
    ValueCase(
        "fraction outside 0 to 1",
        lambda: _fractions(1.5, -0.5),
        f"Stratification 'group', category 'g1': {FRACTION} 1.5",
        write=_set_fractions(1.5, -0.5),
    ),
    ValueCase(
        "fraction NaN",
        lambda: _fractions(0.5, math.nan),
        f"Stratification 'group', category 'g2': {FRACTION} NaN",
    ),
    ValueCase(
        "fractions that do not sum to 1",
        lambda: _fractions(0.5, 0.4),
        "Stratification 'group': the fractions of its categories sum to 1 within "
        "1e-6, got 0.9",
        write=_set_fractions(0.5, 0.4),
    ),
    ValueCase(
        "impulse at and every",
        lambda: _transition(impulse_at=[1.0], impulse_every=(2.0, 0.0)),
        "Transition 't_ab': set impulse_at or impulse_every, not both",
        write=_set_impulse(impulse_at=[1.0], impulse_every=[2.0, 0.0]),
    ),
    ValueCase(
        "impulse at no position",
        lambda: _transition(impulse_at=[]),
        "Transition 't_ab': impulse_at has no position; give at least one",
        write=_set_impulse(impulse_at=[]),
    ),
    ValueCase(
        "impulse position below 0",
        lambda: _transition(impulse_at=[1.0, -1.0]),
        "Transition 't_ab': impulse position is a finite number of at least 0, got -1",
        write=_set_impulse(impulse_at=[1.0, -1.0]),
    ),
    ValueCase(
        "impulse position infinite",
        lambda: _transition(impulse_at=[math.inf]),
        "Transition 't_ab': impulse position is a finite number of at least 0, got inf",
    ),
    ValueCase(
        "impulse period 0",
        lambda: _transition(impulse_every=(0.0, 1.0)),
        "Transition 't_ab': impulse period is a finite number greater than 0, got 0",
        write=_set_impulse(impulse_every=[0.0, 1.0]),
    ),
    ValueCase(
        "impulse offset below 0",
        lambda: _transition(impulse_every=(1.0, -1.0)),
        "Transition 't_ab': impulse offset is a finite number of at least 0, got -1",
        write=_set_impulse(impulse_every=[1.0, -1.0]),
    ),
    ValueCase(
        "series without a point",
        lambda: SeriesValue(data=[]),
        "a series has at least one point, got none",
        compiled="Series parameter 's': a series has at least one point, got none",
        write=_set_series([]),
    ),
    ValueCase(
        "series position below 0",
        lambda: SeriesValue(data=[(1.0, 0.0), (-1.0, 0.0)]),
        "series position is a finite number of at least 0, got -1",
        compiled="Series parameter 's': series position is a finite number of at "
        "least 0, got -1",
        write=_set_series([(1.0, 0.0), (-1.0, 0.0)]),
    ),
    ValueCase(
        "series value infinite",
        lambda: SeriesValue(data=[(0.0, math.inf)]),
        "series value is a finite number, got inf",
    ),
    ValueCase(
        "series position twice",
        lambda: SeriesValue(data=[(2.0, 0.0), (1.0, 0.0), (2.0, 1.0)]),
        "a series has one point per position, got two at 2",
        compiled="Series parameter 's': a series has one point per position, got "
        "two at 2",
        write=_set_series([(2.0, 0.0), (1.0, 0.0), (2.0, 1.0)]),
    ),
]


def _compile_error(write: Callable[[ModelJson], None]) -> str:
    """The message of the model compiler for a model JSON that ``write``
    changes, with no Python object in between."""
    model = json.loads(build(a_to_b_builder()).model_dump_json())
    write(model)
    with pytest.raises(RuntimeError) as error:
        build_engine(json.dumps(model))
    return str(error.value)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_a_value_outside_its_rule_gets_the_message_of_the_engine(
    case: ValueCase,
) -> None:
    with pytest.raises(ValidationError) as error:
        case.build()

    assert str(error.value.errors()[0]["ctx"]["error"]) == case.message
    if case.write is not None:
        assert _compile_error(case.write) == (case.compiled or case.message)


def test_fractions_inside_their_rule_build() -> None:
    model = build(
        a_to_b_builder(stratifications={"group": {"g1": 0.6000005, "g2": 0.4}})
    )

    assert model.structure.stratification_fractions["group"]["g1"] == 0.6000005
